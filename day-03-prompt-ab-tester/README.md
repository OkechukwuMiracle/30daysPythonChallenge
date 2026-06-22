# Day 3: Prompt A/B Comparison Tool

Runs the same input through multiple competing prompt strategies and
compares the outputs side by side on measurable signals - latency, output
length, estimated token count, and (across repeated runs) how consistent
each variant's output actually is.

## The concept: prompt engineering as data, not guesswork

It's easy to write one prompt, glance at the output, and decide it "feels
good". But that's a single, subjective sample - it tells you nothing about
how the prompt performs on cost (tokens), speed (latency), or reliability
(does it give roughly the same answer every time, or does it wander?).
`compare.py` defines several `PromptVariant`s for the same task, runs each
one against identical input through Day 1-2's `LLMClient`, and prints the
metrics next to the output so you can pick a prompt based on evidence
instead of intuition.

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
python compare.py
```

This compares the 3 built-in summarization variants ("plain", "role-primed",
"structured") against a hardcoded sample paragraph, 3 runs per variant. To
compare against your own text instead, pass a file path:

```bash
python compare.py path/to/article.txt
```

## Adding your own variants

`VARIANTS` near the top of `compare.py` is a plain list of `PromptVariant(label,
system_prompt)` entries - add, remove, or edit entries there. `run_comparison()`
is task-agnostic: it just sends `user_input` through each variant's system
prompt and measures the reply, so the same script works for classification,
rewriting, or extraction prompts, not just summarization.

## Files

- `llm.py` - unchanged from Day 2's `LLMClient` (`chat()`, `stream()`,
  `extract()`). No changes were needed: `compare.py` reuses `chat()` as-is
  and swaps `client.system_prompt` between variants.
- `compare.py` - defines `PromptVariant`, the starter variants, and
  `run_comparison()` / `print_comparison()`.
