"""Measure how accurately the agent answers questions about the sample data.

Each question in tests/eval_questions.json has expected values computed directly
with pandas (see tests/test_eval_questions.py), never by the LLM. The agent's
answer passes if it contains every expected number (within a tolerance) and
every expected keyword, or, for trick questions, if it clearly declines.

This calls the real API, so it needs a .env file and costs a small amount per run.

Run from the project root:
    python -m evals.run_eval
    python -m evals.run_eval --only q01 q13     # run selected questions
    python -m evals.run_eval --verbose          # also print every answer
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from src.agent import AgentError, create_agent

QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "tests" / "eval_questions.json"
DEFAULT_TOLERANCE = 1.0  # dollars
NUMBER_PATTERN = re.compile(r"\d[\d,]*(?:\.\d+)?")
DECLINE_PHRASES = [
    "don't have", "do not have", "no data", "not available", "only covers", "only have",
    "outside", "can't", "cannot", "unable", "not able", "doesn't include", "does not include",
]


@dataclass
class EvalResult:
    question_id: str
    question: str
    passed: bool
    reason: str
    answer: str = ""
    tools_used: list[str] = field(default_factory=list)


def extract_numbers(text: str) -> list[float]:
    """All numbers in the text, with $ signs and thousands separators removed: '$1,750.61' -> 1750.61."""
    return [float(match.replace(",", "")) for match in NUMBER_PATTERN.findall(text)]


def grade(answer: str, expected: dict) -> tuple[bool, str]:
    """Return (passed, reason) for one answer against one eval question's expectations."""
    answer_lower = answer.lower()

    if expected.get("decline"):
        if any(phrase in answer_lower for phrase in DECLINE_PHRASES):
            return True, "declined as expected"
        return False, "should have declined, but did not say the data can't answer it"

    tolerance = expected.get("tolerance", DEFAULT_TOLERANCE)
    found = extract_numbers(answer)
    for number in expected.get("numbers", []):
        if not any(abs(value - abs(number)) <= tolerance for value in found):
            return False, f"missing expected number {number}"

    for alternatives in expected.get("keywords", []):
        if not any(word.lower() in answer_lower for word in alternatives):
            return False, f"missing keyword (any of {alternatives})"

    return True, "all expected values found"


def run_question(agent, question: dict) -> EvalResult:
    agent.reset()
    tools_used = []
    reply = None
    for turn in question["turns"]:
        try:
            reply = agent.ask(turn)
        except AgentError as error:
            return EvalResult(question["id"], " / ".join(question["turns"]), False, f"API error: {error}")
        tools_used += [call.name for call in reply.tool_calls]

    passed, reason = grade(reply.text, question)
    return EvalResult(question["id"], " / ".join(question["turns"]), passed, reason, reply.text, tools_used)


def print_report(results: list[EvalResult], verbose: bool) -> None:
    print()
    print(f"{'ID':<5} {'RESULT':<7} {'TOOLS USED':<44} QUESTION")
    print("-" * 110)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        tools_text = ", ".join(result.tools_used) or "(none)"
        print(f"{result.question_id:<5} {status:<7} {tools_text[:43]:<44} {result.question[:55]}")

    failures = [r for r in results if not r.passed]
    if failures:
        print("\nFailures:")
        for result in failures:
            print(f"  {result.question_id}: {result.reason}")
            print(f"       answer: {result.answer[:300]!r}")

    if verbose:
        print("\nAll answers:")
        for result in results:
            print(f"\n[{result.question_id}] {result.question}\n{result.answer}")

    passed = sum(r.passed for r in results)
    print(f"\nAccuracy: {passed}/{len(results)} = {passed / len(results):.0%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agent against the eval questions.")
    parser.add_argument("--only", nargs="+", metavar="ID", help="question ids to run, e.g. q01 q13")
    parser.add_argument("--verbose", action="store_true", help="print every answer in full")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if args.only:
        questions = [q for q in questions if q["id"] in args.only]
        if not questions:
            sys.exit(f"No questions match {args.only}.")

    try:
        agent = create_agent()
    except AgentError as error:
        sys.exit(f"Setup error: {error}")

    results = []
    for number, question in enumerate(questions, start=1):
        print(f"[{number}/{len(questions)}] {question['id']}: {question['turns'][-1]}", flush=True)
        results.append(run_question(agent, question))
    print_report(results, args.verbose)


if __name__ == "__main__":
    main()
