# Day 2: Structured Output Extractor

Turns messy, unstructured text into clean, schema-validated JSON using Claude.
Extends Day 1's `llm.py` with a new `LLMClient.extract()` method, and adds
`extract.py` as a demo CLI that parses a free-text resume into a typed
`Resume` object.

## The concept: schema-constrained output

Asking an LLM to "respond in JSON" and hoping for the best is fragile - the
model can wrap output in prose, rename fields, or drop required ones. Instead,
`extract()` defines a single Anthropic **tool** whose `input_schema` is a
Pydantic model's JSON schema, then *forces* the model to call that tool via
`tool_choice`. This makes the API itself constrain the shape of the response.
The tool's `input` is then validated by constructing the Pydantic model from
it, so what comes back is a real typed object - safe to pipe into a database
or another function - not a string you have to hope is valid JSON.

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
python extract.py
```

This uses a hardcoded messy sample resume. To parse your own text instead,
pass a file path:

```bash
python extract.py path/to/resume.txt
```

The script prints the validated `Resume` object as pretty JSON, then proves
it's a real typed object (not just a JSON-looking string) by reporting the
parsed skill count and email.

## Files

- `llm.py` - `LLMClient` class wrapping the Anthropic SDK (`chat()`, `stream()`,
  and now `extract()`).
- `extract.py` - the CLI demo: defines a `Resume` Pydantic schema and runs it
  through `LLMClient.extract()`.
