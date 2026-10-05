"""Command line interface for the finance assistant.

Run from the project root:
    python -m src.cli
"""

import sys
from collections.abc import Callable

from src.agent import AgentError, AgentReply, create_agent

HELP_TEXT = """\
Ask questions about your transactions in plain English, for example:
  How much did I spend on food last month?
  What were my top 5 merchants in July?
  Did my spending go up from July to August?
  Flag anything unusual.

Commands:
  /help   show this message
  /reset  forget the conversation so far
  /quit   exit (Ctrl+C also works)"""


def format_tool_calls(reply: AgentReply) -> list[str]:
    """One line per tool the agent used, so the user can see how the answer was built."""
    lines = []
    for call in reply.tool_calls:
        arguments = ", ".join(f"{key}={value!r}" for key, value in call.arguments.items())
        status = " -> error" if call.is_error else ""
        lines.append(f"  [tool] {call.name}({arguments}){status}")
    return lines


def run(agent, read_input: Callable[[str], str] = input, write: Callable[[str], None] = print) -> None:
    """The read-ask-print loop. Input and output are parameters so tests can drive it."""
    write("Finance assistant. Type /help for examples, /quit to exit.\n")
    while True:
        try:
            question = read_input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            write("\nGoodbye.")
            return

        if not question:
            continue
        if question.lower() in ("/quit", "/exit"):
            write("Goodbye.")
            return
        if question.lower() == "/help":
            write(HELP_TEXT + "\n")
            continue
        if question.lower() == "/reset":
            agent.reset()
            write("Conversation cleared.\n")
            continue

        try:
            reply = agent.ask(question)
        except AgentError as error:
            write(f"Error: {error}\n")
            continue
        except KeyboardInterrupt:
            write("\n(Cancelled.)\n")
            continue

        for line in format_tool_calls(reply):
            write(line)
        write(f"Assistant: {reply.text}\n")


def main() -> None:
    try:
        agent = create_agent()
    except AgentError as error:
        print(f"Setup error: {error}")
        sys.exit(1)
    run(agent)


if __name__ == "__main__":
    main()
