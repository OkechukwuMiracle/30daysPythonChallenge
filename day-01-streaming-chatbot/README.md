# Day 1: Streaming CLI Chatbot

A terminal chatbot that streams the assistant's reply token-by-token, using
Anthropic's Claude API. A reusable `LLMClient` wrapper (`llm.py`) handles the
SDK details so the rest of the structure can be reused across the rest of
this 30-day challenge.

## Setup

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and add your API key:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and set `ANTHROPIC_API_KEY` to a key from
   https://console.anthropic.com/.

## Run

```bash
python chat.py
```

## Commands

- Type normally to chat — replies stream in as they're generated.
- `/reset` — clear conversation history and start fresh.
- `/exit` or `/quit` — end the program.
- `Ctrl+C` — exits cleanly at any time.

## Files

- `llm.py` — `LLMClient` class wrapping the Anthropic SDK (`chat()` and `stream()` methods).
- `chat.py` — the CLI chat loop that uses `LLMClient`.
