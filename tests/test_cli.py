"""Tests for the CLI loop, driven with scripted input and a fake agent (no API calls)."""

from src.agent import AgentError, AgentReply, ToolCall
from src.cli import run


class FakeAgent:
    def __init__(self, replies):
        self.replies = list(replies)
        self.questions = []
        self.reset_count = 0

    def ask(self, question):
        self.questions.append(question)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def reset(self):
        self.reset_count += 1


def run_with_inputs(agent, inputs):
    """Run the CLI with the given lines of input; return everything it printed."""
    pending = list(inputs)
    printed = []

    def read_input(prompt):
        if not pending:
            raise EOFError
        return pending.pop(0)

    run(agent, read_input=read_input, write=printed.append)
    return "\n".join(printed)


def test_asks_question_and_prints_answer_with_tool_calls():
    agent = FakeAgent([
        AgentReply("You spent $100.00.", [ToolCall("monthly_summary", {"month": "2026-07"}, False)]),
    ])
    output = run_with_inputs(agent, ["How much in July?", "/quit"])

    assert agent.questions == ["How much in July?"]
    assert "[tool] monthly_summary(month='2026-07')" in output
    assert "Assistant: You spent $100.00." in output


def test_failed_tool_call_is_marked():
    agent = FakeAgent([AgentReply("No data.", [ToolCall("monthly_summary", {"month": "2025-01"}, True)])])
    assert "-> error" in run_with_inputs(agent, ["January 2025?"])


def test_commands_are_not_sent_to_the_agent():
    agent = FakeAgent([])
    output = run_with_inputs(agent, ["/help", "/reset", "", "/quit"])

    assert agent.questions == []
    assert agent.reset_count == 1
    assert "Flag anything unusual" in output
    assert "Conversation cleared" in output


def test_agent_error_is_shown_and_the_loop_continues():
    agent = FakeAgent([AgentError("Rate limit reached."), AgentReply("Second answer.")])
    output = run_with_inputs(agent, ["first", "second"])

    assert "Error: Rate limit reached." in output
    assert "Assistant: Second answer." in output


def test_end_of_input_exits_cleanly():
    assert "Goodbye" in run_with_inputs(FakeAgent([]), [])
