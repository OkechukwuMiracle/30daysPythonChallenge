"""
compare.py - Day 3: a prompt A/B comparison tool.

Why this exists: prompt engineering usually means writing one prompt, glancing
at the output, and trusting a gut feeling that it's "good enough". This script
turns that into measurement instead: the same input is run through several
competing prompt variants for the same task, and the results are compared
side by side on length, latency, and (when run multiple times per variant)
how consistent each variant's output is.

Reuses Day 2's llm.py untouched - LLMClient.chat() already accepts a plain
message list and reads its system prompt from an instance attribute, so
swapping prompts between variants is just reassigning that attribute before
each call. No changes to llm.py were needed.
"""

import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from llm import LLMClient

MODEL = "claude-haiku-4-5"  # change this to switch Claude models


@dataclass(frozen=True)
class PromptVariant:
    """One competing strategy for a task: a label plus its system prompt."""

    label: str
    system_prompt: str


# Three competing strategies for the SAME task (text summarization).
# Add your own variants here - run_comparison() and main() don't change
# regardless of how many variants there are or what task they target.
VARIANTS: list[PromptVariant] = [
    PromptVariant(
        label="plain",
        system_prompt="Summarize the text.",
    ),
    PromptVariant(
        label="role-primed",
        system_prompt=(
            "You are a senior editor at a major newspaper who specializes in "
            "distilling complex articles into sharp, accurate summaries for "
            "busy readers. Summarize the text the user gives you so that a "
            "reader with only 10 seconds still gets the essential point."
        ),
    ),
    PromptVariant(
        label="structured",
        system_prompt=(
            "Summarize the text the user gives you as exactly 3 bullet points. "
            "Each bullet must be no more than 15 words. Start each bullet with "
            "a dash ('-'). Output nothing except the 3 bullets."
        ),
    ),
]

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


@dataclass
class RunResult:
    """The measurements for a single call to a single variant."""

    output: str
    latency_s: float
    char_len: int
    est_tokens: int


@dataclass
class VariantResult:
    """All runs collected for one variant, plus the aggregate metrics."""

    variant: PromptVariant
    runs: list[RunResult]

    @property
    def avg_latency_s(self) -> float:
        return statistics.mean(r.latency_s for r in self.runs)

    @property
    def avg_char_len(self) -> float:
        return statistics.mean(r.char_len for r in self.runs)

    @property
    def avg_est_tokens(self) -> float:
        return statistics.mean(r.est_tokens for r in self.runs)

    @property
    def char_len_stdev(self) -> float:
        """How much output length swings across repeated runs - the
        consistency signal. 0 when there's only one run to compare."""
        if len(self.runs) < 2:
            return 0.0
        return statistics.stdev(r.char_len for r in self.runs)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) - fine for relative comparison."""
    return max(1, len(text) // 4)


def run_comparison(
    client: LLMClient,
    variants: list[PromptVariant],
    user_input: str,
    runs: int = 1,
) -> list[VariantResult]:
    """
    Run the same user_input through each variant's system prompt and collect
    output + timing metrics. Task-agnostic: swap VARIANTS for a classification
    or rewriting task and this function needs no changes.

    Args:
        client: A constructed LLMClient. Its system_prompt attribute is
            reassigned per variant, then restored afterwards.
        variants: The competing prompt strategies to test.
        user_input: The single input every variant is run against.
        runs: How many times to repeat each variant. >1 reveals how much a
            prompt's output varies run to run, not just one sample of it.

    Returns:
        One VariantResult per variant, in the same order as `variants`.
    """
    original_system_prompt = client.system_prompt
    results: list[VariantResult] = []
    try:
        for variant in variants:
            client.system_prompt = variant.system_prompt
            run_results: list[RunResult] = []
            for _ in range(runs):
                start = time.perf_counter()
                try:
                    output = client.chat([{"role": "user", "content": user_input}])
                except RuntimeError as error:
                    output = f"[Error: {error}]"
                latency_s = time.perf_counter() - start
                run_results.append(
                    RunResult(
                        output=output,
                        latency_s=latency_s,
                        char_len=len(output),
                        est_tokens=estimate_tokens(output),
                    )
                )
            results.append(VariantResult(variant=variant, runs=run_results))
    finally:
        # Leave the shared client the way we found it for any caller code
        # that runs after the comparison.
        client.system_prompt = original_system_prompt
    return results


def print_comparison(results: list[VariantResult]) -> None:
    """Print a readable, dependency-free side-by-side comparison."""
    width = 78
    for result in results:
        print("=" * width)
        print(f"VARIANT: {result.variant.label}")
        print("-" * width)
        # Show the first run's full output; with multiple runs the point is
        # consistency (see stdev below), not re-reading near-duplicate text.
        print(result.runs[0].output.strip())
        print("-" * width)

        metrics = (
            f"latency: {result.avg_latency_s:.2f}s  |  "
            f"length: {result.avg_char_len:.0f} chars  |  "
            f"est. tokens: {result.avg_est_tokens:.0f}"
        )
        if len(result.runs) > 1:
            metrics += (
                f"  |  length stdev over {len(result.runs)} runs: "
                f"{result.char_len_stdev:.1f}"
            )
        print(metrics)
        print()


def load_text(argv: list[str]) -> str:
    """Read input text from a file path argument, or fall back to the sample."""
    if len(argv) > 1:
        return Path(argv[1]).read_text(encoding="utf-8")
    print("(no file given - using the hardcoded sample paragraph)\n")
    return SAMPLE_TEXT


def main() -> None:
    text = load_text(sys.argv)
    client = LLMClient(model=MODEL)

    # runs=3 by default so the demo actually exercises the consistency
    # signal (stdev), not just a single sample per variant.
    print(f"Comparing {len(VARIANTS)} prompt variants (3 runs each)...\n")
    results = run_comparison(client, VARIANTS, text, runs=3)
    print_comparison(results)


if __name__ == "__main__":
    main()
