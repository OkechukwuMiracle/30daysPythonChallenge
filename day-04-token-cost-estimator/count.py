"""
count.py - Day 4: a token counter + cost estimator.

Why this exists: most AI bugs are correctness bugs, but the most common AI
*surprise* is a bill. This script builds the instinct for what a piece of
text actually costs to run by showing two ways to count tokens (EXACT via
Anthropic's count_tokens API, ESTIMATE via a free local heuristic) and what
each model tier would charge for it.
"""

import sys
from pathlib import Path

from llm import PRICING, LLMClient

# The model used for the one EXACT token-count API call. Any model in
# PRICING works here - tokenization is shared across the Claude 4 family, so
# the exact count doesn't meaningfully change between Haiku/Sonnet/Opus.
COUNT_MODEL = "claude-haiku-4-5-20251001"

# Used when no input file is given on the command line.
SAMPLE_TEXT = """
The James Webb Space Telescope, launched in December 2021, has spent its
first years of operation rewriting parts of astronomy textbooks. Designed
primarily to observe in infrared light, it can peer through dust clouds that
block visible-light telescopes like Hubble, revealing stars and galaxies
forming in regions once thought to be empty. Among its most surprising
findings are galaxies that appear fully formed and unexpectedly massive only
a few hundred million years after the Big Bang, a result that has forced
astronomers to reconsider how quickly galaxies can grow in the early
universe. Closer to home, the telescope has also analyzed the atmospheres of
several exoplanets, detecting carbon dioxide and, in at least one case,
sulfur dioxide - a possible signature of photochemical activity. Because the
telescope orbits the Sun at a gravitationally stable point nearly 1.5 million
kilometers from Earth, it cannot be serviced by astronauts the way Hubble
was, so its instruments were engineered with little margin for in-flight
repair.
""".strip()

# A representative reply length to cost out, since the input text alone
# isn't a full request - every real call also pays for the generated output.
SAMPLE_OUTPUT_TOKENS = 500


def estimate_tokens(text: str) -> int:
    """Free, offline token estimate (~4 chars/token average for English).

    No network call, no API key needed - good for rough sizing (e.g. "will
    this fit in my context window?") when an exact count isn't worth the
    round-trip.
    """
    return max(1, len(text) // 4)


def load_text(argv: list[str]) -> str:
    """Read input text from a file path argument, or fall back to the sample."""
    if len(argv) > 1:
        return Path(argv[1]).read_text(encoding="utf-8")
    print("(no file given - using the hardcoded sample paragraph)\n")
    return SAMPLE_TEXT


def print_token_comparison(exact: int, estimate: int) -> None:
    """Show EXACT vs ESTIMATE side by side and how far apart they are.

    The point isn't that the heuristic is perfectly accurate - it's that
    it's *close enough* for sizing decisions, at zero cost and zero latency.
    """
    diff_pct = abs(exact - estimate) / exact * 100 if exact else 0.0
    print("=" * 60)
    print("TOKEN COUNT: exact (API) vs. estimate (heuristic)")
    print("=" * 60)
    print(f"{'Exact (count_tokens API)':<32}: {exact:>6} tokens")
    print(f"{'Estimate (len(text)/4)':<32}: {estimate:>6} tokens")
    print(f"{'Difference':<32}: {diff_pct:>5.1f}%")
    print()


def print_cost_table(input_tokens: int, output_tokens: int) -> None:
    """Print input/output/total cost for this text across every priced model.

    One LLMClient per model: estimate_cost() reads pricing from the client's
    own `self.model`, so the cleanest way to compare models is to ask each
    model's own client for its number rather than re-deriving the math here.
    """
    print("=" * 60)
    print(f"COST TABLE  (input={input_tokens} tokens, assumed output={output_tokens} tokens)")
    print("=" * 60)
    print(f"{'Model':<28}{'Input':>10}{'Output':>10}{'Total':>10}")
    print("-" * 60)

    breakdowns = []
    for model in PRICING:
        client = LLMClient(model=model)
        breakdown = client.estimate_cost(input_tokens, output_tokens)
        breakdowns.append(breakdown)
        print(
            f"{model:<28}"
            f"{'$' + format(breakdown['input_cost'], '.6f'):>10}"
            f"{'$' + format(breakdown['output_cost'], '.6f'):>10}"
            f"{'$' + format(breakdown['total_cost'], '.6f'):>10}"
        )
    print()
    return breakdowns


def main() -> None:
    text = load_text(sys.argv)

    client = LLMClient(model=COUNT_MODEL)
    try:
        exact_tokens = client.count_tokens(text)
    except RuntimeError as error:
        print(f"Could not reach the count_tokens API ({error}); "
              f"falling back to the heuristic only.\n")
        exact_tokens = estimate_tokens(text)

    estimated_tokens = estimate_tokens(text)
    print_token_comparison(exact_tokens, estimated_tokens)

    breakdowns = print_cost_table(exact_tokens, SAMPLE_OUTPUT_TOKENS)

    # Self-computed takeaway: ground the abstract per-call cost in a number
    # that's actually intuitive - what running this at scale would cost.
    haiku = next(b for b in breakdowns if b["model"] == "claude-haiku-4-5-20251001")
    runs = 10_000
    bulk_cost = haiku["total_cost"] * runs
    print(
        f"Takeaway: running this exact request {runs:,} times on "
        f"{haiku['model']} would cost roughly ${bulk_cost:,.2f}."
    )


if __name__ == "__main__":
    main()
