# Day 4: Token Counter + Cost Estimator

Every AI engineer needs a gut feeling for what a piece of text actually
costs to run - not because the math is hard, but because without it you
won't notice a runaway prompt, an unbounded `max_tokens`, or a loop calling
the API 10,000 times until the invoice arrives. `count.py` builds that
instinct by counting tokens two ways and turning the result into real
dollar amounts across model tiers.

## Exact vs. estimate

- **Exact** - `LLMClient.count_tokens()` calls Anthropic's native
  `messages.count_tokens` endpoint and returns the real `input_tokens` Claude
  would bill for that text. One API round-trip, no generation cost, but it
  needs network access and an API key.
- **Estimate** - a local heuristic (`len(text) // 4`) with no network call at
  all. It won't match the exact count, but it's close enough for rough
  sizing (e.g. "will this fit my context window?") when a round-trip isn't
  worth the latency.

Use the estimate for quick, offline sizing during development; use the exact
count when you need a real number for billing or context-limit decisions.

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
python count.py
```

This runs against a hardcoded sample paragraph. To count and cost your own
text instead, pass a file path:

```bash
python count.py path/to/article.txt
```

Output shows the exact vs. estimated token count (and how far apart they
are), then a per-model cost table assuming a 500-token reply, and finishes
with a one-line takeaway scaling the cost to 10,000 runs.

## Files

- `llm.py` - Day 1-3's `LLMClient`, extended with `count_tokens()` (exact,
  via the API) and `estimate_cost()` (turns token counts into USD using the
  `PRICING` table at the top of the file).
- `count.py` - the CLI that ties it together: token comparison, cost table,
  bulk-cost takeaway.
- Pricing in `PRICING` is current as of writing - verify against
  https://claude.com/pricing before relying on it for a real budget.
