"""
chat.py - A terminal chatbot that streams responses token-by-token.

Run with: python chat.py
"""

from llm import LLMClient

MODEL = "claude-haiku-4-5"  # change this to switch Claude models


def print_banner(model: str) -> None:
    """Show a friendly startup message so the user knows what they're talking to."""
    print("=" * 50)
    print(" Day 1: Streaming CLI Chatbot")
    print(f" Model: {model}")
    print(" Commands: /reset to clear history, /exit or /quit to leave")
    print("=" * 50)


def main() -> None:
    llm = LLMClient(model=MODEL, system_prompt="You are a helpful, concise assistant.")
    history: list[dict] = []  # full conversation, so the bot has memory

    print_banner(MODEL)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            # Ctrl+C or Ctrl+D should exit quietly, not dump a traceback.
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/exit", "/quit"):
            print("Goodbye!")
            break

        if user_input.lower() == "/reset":
            history = []
            print("(conversation history cleared)")
            continue

        history.append({"role": "user", "content": user_input})

        print("Assistant: ", end="", flush=True)
        reply_so_far = ""
        try:
            for chunk in llm.stream(history):
                print(chunk, end="", flush=True)
                reply_so_far += chunk
            print()  # newline after the streamed reply finishes
        except KeyboardInterrupt:
            print("\n(interrupted)")
            continue
        except RuntimeError as error:
            print(f"\n[Error: {error}]")
            continue

        # Only remember the reply if we actually got one, so a failed
        # request doesn't pollute history with an empty assistant turn.
        if reply_so_far:
            history.append({"role": "assistant", "content": reply_so_far})


if __name__ == "__main__":
    main()
