"""
extract.py - Demo CLI for Day 2: schema-constrained structured extraction.

Run with: python extract.py [path/to/resume.txt]
If no path is given, a hardcoded messy sample resume is used instead.
"""

import sys
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from llm import LLMClient

MODEL = "claude-haiku-4-5"  # change this to switch Claude models

# Deliberately messy: inconsistent formatting, an obfuscated email, vague
# experience, and a wall of run-on prose - the kind of text you'd actually
# get from a pasted resume rather than a clean form submission.
SAMPLE_RESUME = """
JANE DOE -- Senior-ish Backend Engineer??
Contact: jane dot doe AT example DOT com (or jdoe2020@gmail.com, whichever works)
~7 yrs in the industry, give or take. Most recently grinding away as a
Staff Software Engineer @ Acme Corp (2021-present).
Tech I've touched: Python, Go, a little bit of Rust, Postgres, Redis, Docker,
Kubernetes, Terraform, and way too much Bash.
Before that: Backend Engineer at Globex (2018-2021), and a brief stint doing
freelance Django work back in 2017.
"""


class Resume(BaseModel):
    """The shape we want messy resume text squeezed into."""

    name: str
    email: Optional[str] = None
    years_experience: Optional[int] = None
    skills: list[str] = Field(default_factory=list)
    most_recent_role: Optional[str] = None


def load_text(argv: list[str]) -> str:
    """Read resume text from a file path argument, or fall back to the sample."""
    if len(argv) > 1:
        return Path(argv[1]).read_text(encoding="utf-8")
    print("(no file given - using the hardcoded sample resume)\n")
    return SAMPLE_RESUME


def main() -> None:
    text = load_text(sys.argv)
    llm = LLMClient(model=MODEL)

    print("Extracting structured data...\n")
    try:
        resume = llm.extract(text, Resume)
    except (RuntimeError, ValueError) as error:
        print(f"[Error: {error}]")
        sys.exit(1)

    print(resume.model_dump_json(indent=2))

    # This is the payoff: `resume` is a real Resume instance, not a string we
    # hope is valid JSON, so these attribute accesses are safe by construction.
    print(f"\n{len(resume.skills)} skill(s) parsed.")
    print(f"Email parsed OK: {resume.email}" if resume.email else "No email found in the text.")


if __name__ == "__main__":
    main()
