"""Tests for the agent loop, using a fake LLM client instead of the real API.

FakeClient returns pre-scripted responses, so these tests check OUR code (the loop,
tool dispatch, error handling, history) without network calls or API costs.
Whether the real model picks good tools is measured separately by evals/run_eval.py.
"""

import inspect
import json
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from src.agent import MAX_ITERATIONS, TOOL_FUNCTIONS, AgentError, FinanceAgent, build_system_prompt
from src.tool_schemas import TOOL_SCHEMAS
from src.tools import load_transactions


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_use_block(name, arguments, block_id="toolu_1"):
    return SimpleNamespace(type="tool_use", name=name, input=arguments, id=block_id)


def response(stop_reason, *blocks):
    return SimpleNamespace(stop_reason=stop_reason, content=list(blocks))


class FakeClient:
    """Stands in for anthropic.Anthropic. Returns scripted responses and records each request."""

    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.requests = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **request):
        # Copy the messages list: the agent keeps appending to the same list object.
        self.requests.append({**request, "messages": list(request["messages"])})
        next_item = self.scripted.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        return next_item


@pytest.fixture(scope="module")
def df():
    return load_transactions()


def make_agent(df, scripted):
    client = FakeClient(scripted)
    return FinanceAgent(client, "test-model", df), client


def api_error(error_class, status):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return error_class("error", response=httpx2.Response(status, request=request), body=None)


# ---------- the loop ----------

def test_tool_call_then_final_answer(df):
    agent, client = make_agent(df, [
        response("tool_use", tool_use_block("monthly_summary", {"month": "2026-07"})),
        response("end_turn", text_block("You spent $X in July.")),
    ])

    reply = agent.ask("How much did I spend in July?")

    assert reply.text == "You spent $X in July."
    assert [call.name for call in reply.tool_calls] == ["monthly_summary"]
    # The second request must carry the tool result, linked to the tool_use id.
    tool_result = client.requests[1]["messages"][-1]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "toolu_1"
    assert tool_result["is_error"] is False
    assert json.loads(tool_result["content"])["month"] == "2026-07"


def test_answer_without_tools(df):
    agent, client = make_agent(df, [response("end_turn", text_block("Hi! Ask me about your spending."))])
    assert agent.ask("hello").text == "Hi! Ask me about your spending."
    assert len(client.requests) == 1


def test_multiple_tool_calls_return_results_in_one_message(df):
    agent, client = make_agent(df, [
        response(
            "tool_use",
            tool_use_block("monthly_summary", {"month": "2026-07"}, "toolu_a"),
            tool_use_block("monthly_summary", {"month": "2026-08"}, "toolu_b"),
        ),
        response("end_turn", text_block("done")),
    ])
    agent.ask("Compare July and August income")

    results = client.requests[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["toolu_a", "toolu_b"]


def test_invalid_tool_input_is_sent_back_as_error(df):
    agent, client = make_agent(df, [
        response("tool_use", tool_use_block("monthly_summary", {"month": "2025-01"})),
        response("end_turn", text_block("I only have data from April to September 2026.")),
    ])
    reply = agent.ask("What about January 2025?")

    tool_result = client.requests[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True
    assert "There is no data for 2025-01" in tool_result["content"]
    assert reply.tool_calls[0].is_error is True


def test_unknown_tool_and_bad_arguments_do_not_crash(df):
    agent, client = make_agent(df, [
        response(
            "tool_use",
            tool_use_block("delete_everything", {}, "toolu_a"),
            tool_use_block("monthly_summary", {"wrong_arg": 1}, "toolu_b"),
        ),
        response("end_turn", text_block("ok")),
    ])
    agent.ask("test")

    unknown, bad_args = client.requests[1]["messages"][-1]["content"]
    assert unknown["is_error"] and "Unknown tool" in unknown["content"]
    assert bad_args["is_error"] and "Invalid arguments" in bad_args["content"]


def test_loop_stops_at_max_iterations(df):
    endless = [response("tool_use", tool_use_block("monthly_summary", {"month": "2026-07"}))] * 20
    agent, client = make_agent(df, endless)

    reply = agent.ask("loop forever")

    assert len(client.requests) == MAX_ITERATIONS
    assert f"within {MAX_ITERATIONS} steps" in reply.text
    assert agent.messages == []  # unfinished exchange was rolled back


def test_cut_off_tool_call_is_not_run(df):
    # stop_reason "max_tokens" means the tool input may be incomplete.
    agent, client = make_agent(df, [
        response("max_tokens", tool_use_block("monthly_summary", {"month": "2026-0"})),
    ])
    reply = agent.ask("question")

    assert reply.tool_calls == []
    assert "cut off" in reply.text
    assert agent.messages == []


# ---------- conversation history ----------

def test_history_is_kept_for_follow_up_questions(df):
    agent, client = make_agent(df, [
        response("end_turn", text_block("July answer")),
        response("end_turn", text_block("June answer")),
    ])
    agent.ask("How much did I spend in July?")
    agent.ask("What about the month before?")

    second_request = client.requests[1]["messages"]
    assert [m["role"] for m in second_request] == ["user", "assistant", "user"]
    assert second_request[0]["content"] == "How much did I spend in July?"


def test_reset_clears_history(df):
    agent, _ = make_agent(df, [response("end_turn", text_block("answer"))])
    agent.ask("question")
    agent.reset()
    assert agent.messages == []


# ---------- API errors ----------

@pytest.mark.parametrize("error, expected_text", [
    (api_error(anthropic.RateLimitError, 429), "Rate limit"),
    (api_error(anthropic.AuthenticationError, 401), "API key was rejected"),
    (api_error(anthropic.NotFoundError, 404), "MODEL_NAME"),
    (api_error(anthropic.InternalServerError, 500), "having problems"),
    (anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com")), "internet"),
])
def test_api_errors_become_friendly_messages(df, error, expected_text):
    agent, _ = make_agent(df, [error])
    with pytest.raises(AgentError, match=expected_text):
        agent.ask("question")
    assert agent.messages == []  # a failed question leaves no trace in the history


def test_history_still_works_after_an_api_error(df):
    agent, client = make_agent(df, [
        response("end_turn", text_block("first answer")),
        api_error(anthropic.RateLimitError, 429),
        response("end_turn", text_block("third answer")),
    ])
    agent.ask("first")
    with pytest.raises(AgentError):
        agent.ask("second")
    agent.ask("third")

    roles = [m["role"] for m in client.requests[-1]["messages"]]
    assert roles == ["user", "assistant", "user"]  # "second" was rolled back


# ---------- schemas and guardrails ----------

def test_every_schema_has_a_matching_function_and_parameters():
    assert {schema["name"] for schema in TOOL_SCHEMAS} == set(TOOL_FUNCTIONS)
    for schema in TOOL_SCHEMAS:
        function_params = list(inspect.signature(TOOL_FUNCTIONS[schema["name"]]).parameters)[1:]  # skip df
        assert set(schema["input_schema"]["properties"]) == set(function_params), schema["name"]


def test_request_sends_tools_and_system_prompt_but_no_raw_data(df):
    agent, client = make_agent(df, [response("end_turn", text_block("ok"))])
    agent.ask("question")

    request = client.requests[0]
    assert request["tools"] == TOOL_SCHEMAS
    assert "2026-04-01" in request["system"] and "2026-09-30" in request["system"]
    # No transaction descriptions may leak into the prompt.
    for description in df["description"].unique():
        assert description not in request["system"]


def test_system_prompt_defines_today_and_last_month(df):
    prompt = build_system_prompt(df)
    assert 'this month" is 2026-09' in prompt
    assert 'last month" is 2026-08' in prompt
