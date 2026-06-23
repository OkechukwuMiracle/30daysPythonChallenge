"""
llm.py - A small, reusable wrapper around the Anthropic (Claude) Python SDK.

Why this exists: Claude's API isn't OpenAI-schema-compatible (system prompts
are a top-level parameter, not a message; max_tokens is required; responses
are a list of content blocks), so this wrapper hides those details behind the
same chat()/stream() shape we'll reuse in every other day's project.

Day 4 adds count_tokens() and estimate_cost(): before you spend money calling
a model, you should be able to answer "how many tokens is this, and what will
it cost?" without guessing.
"""

import os
from typing import Generator, Optional

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

# Load variables from a .env file into the environment as soon as this
# module is imported, so any script that imports llm.py gets the key for free.
load_dotenv()

# USD price per MILLION tokens, by model. Source: https://claude.com/pricing
# (rates change - re-check before trusting these for a real budget).
#
# Note output is consistently ~5x the price of input across every tier. A
# reply is almost always shorter than its prompt in raw character count, but
# NOT always in token count once you factor in verbose answers, chain-of-
# thought, or generated code - so output length, not input length, is
# usually the lever that actually moves your bill.
PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-opus-4-8": {"input": 5.00, "output": 25.00},
}


class LLMClient:
    """A thin wrapper around Anthropic's Messages API."""

    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> None:
        """
        Set up the client.

        Args:
            model: Which Claude model to call. Defaults to the cheapest,
                fastest tier - plenty for a chat demo. Pass "claude-opus-4-8"
                for the most capable model, or "claude-sonnet-4-6" for
                something in between.
            system_prompt: Optional system prompt used to set the assistant's
                behavior/persona. Sent as a top-level field, not a message.
            max_tokens: Maximum tokens to generate per reply. Anthropic
                requires this on every request, unlike OpenAI.
        """
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            # Fail loudly and early rather than letting the SDK raise a
            # confusing auth error deep inside a network call.
            raise ValueError(
                "Missing API key. Create a .env file (see .env.example) and "
                "set ANTHROPIC_API_KEY=your-key-here."
            )

        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(api_key=api_key)

    def _request_kwargs(self, messages: list[dict]) -> dict:
        """Build the shared kwargs for create()/stream() calls.

        system is only included when set, since Anthropic rejects an empty
        string the same as a real prompt - omitting the key is the only way
        to say "no system prompt."
        """
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if self.system_prompt:
            kwargs["system"] = self.system_prompt
        return kwargs

    def chat(self, messages: list[dict]) -> str:
        """
        Send the full conversation and return the complete reply as a string.

        Args:
            messages: A list of {"role": ..., "content": ...} dicts representing
                the conversation so far (e.g. [{"role": "user", "content": "hi"}]).
                Only "user" and "assistant" roles belong here - the system
                prompt is set separately, via the constructor.

        Returns:
            The assistant's full reply text.
        """
        try:
            response = self.client.messages.create(**self._request_kwargs(messages))
            # content is a list of blocks (text, tool_use, etc.) - grab the
            # first text block rather than assuming index 0 is always text.
            for block in response.content:
                if block.type == "text":
                    return block.text
            return ""
        except Exception as error:
            raise RuntimeError(f"LLM request failed: {error}") from error

    def stream(self, messages: list[dict]) -> Generator[str, None, None]:
        """
        Send the full conversation and yield the reply one chunk at a time.

        Args:
            messages: Same format as chat(). The full history, not just the
                latest message, since the API has no memory of its own.

        Yields:
            Successive text chunks as they arrive from the API.
        """
        try:
            with self.client.messages.stream(**self._request_kwargs(messages)) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception as error:
            raise RuntimeError(f"LLM streaming request failed: {error}") from error

    def extract(self, text: str, schema: type[BaseModel]) -> BaseModel:
        """
        Extract structured data from free text, guaranteed to match `schema`.

        Why tool-use instead of "please reply with JSON": prompting for JSON
        still lets the model wrap output in prose, add trailing commentary,
        or drift from the exact field names/types you wanted. Defining a
        single tool whose input_schema *is* the Pydantic schema, then forcing
        the model to call it via tool_choice, makes the API constrain the
        output shape for us - we only need to validate the result, not coax
        the model into behaving.

        Args:
            text: The raw, messy text to pull structured fields out of.
            schema: A Pydantic model class describing the desired shape.

        Returns:
            A validated instance of `schema`.

        Raises:
            RuntimeError: The API call failed, or no tool_use block came back.
            ValueError: The model's output didn't satisfy the schema.
        """
        # Anthropic identifies tools by name, so derive one from the schema
        # rather than asking the caller to invent and pass one in.
        tool_name = f"extract_{schema.__name__.lower()}"
        tool = {
            "name": tool_name,
            "description": f"Extract {schema.__name__} fields from the given text.",
            "input_schema": schema.model_json_schema(),
        }

        kwargs = self._request_kwargs([{"role": "user", "content": text}])
        kwargs["tools"] = [tool]
        # Forcing tool_choice (rather than leaving it "auto") is what
        # guarantees a tool_use block comes back instead of a text reply.
        kwargs["tool_choice"] = {"type": "tool", "name": tool_name}

        try:
            response = self.client.messages.create(**kwargs)
        except Exception as error:
            raise RuntimeError(f"LLM extraction request failed: {error}") from error

        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                try:
                    return schema.model_validate(block.input)
                except ValidationError as error:
                    raise ValueError(
                        f"Model output didn't match {schema.__name__} schema:\n{error}"
                    ) from error

        raise RuntimeError(
            f"Expected a '{tool_name}' tool_use block in the response but found none."
        )

    def count_tokens(self, text: str) -> int:
        """
        Get the EXACT input-token count for `text` from Anthropic's own
        tokenizer, via a dedicated count_tokens endpoint.

        Why this matters: Claude's tokenizer isn't public, so any local
        estimate (e.g. chars/4) can drift from the real count - sometimes
        enough to matter when you're near a context limit or budgeting
        precisely. This costs one API round-trip but no generation tokens,
        so it's "free" in dollar terms, just not in latency.

        Args:
            text: The text to count, wrapped as a single user message - this
                mirrors how it would actually be billed in a real chat() call.

        Returns:
            The exact number of input tokens Claude would charge for this text
            (plus the fixed overhead of the message structure itself).

        Raises:
            RuntimeError: The API call failed (e.g. network issue, bad model name).
        """
        try:
            result = self.client.messages.count_tokens(
                model=self.model,
                messages=[{"role": "user", "content": text}],
            )
            return result.input_tokens
        except Exception as error:
            raise RuntimeError(f"Token count request failed: {error}") from error

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> dict:
        """
        Turn a token count into a dollar estimate using the PRICING table.

        Args:
            input_tokens: Tokens charged for the prompt (system + messages).
            output_tokens: Tokens charged for the generated reply.

        Returns:
            A dict with input_cost, output_cost, and total_cost in USD, plus
            the model and token counts used to compute them - so callers can
            print a full breakdown without re-deriving it.

        Raises:
            ValueError: self.model isn't in PRICING (add it there first).
        """
        if self.model not in PRICING:
            raise ValueError(
                f"No pricing data for model '{self.model}'. "
                f"Add it to PRICING in llm.py - known models: {list(PRICING)}"
            )

        rates = PRICING[self.model]
        # Rates are dollars per MILLION tokens, hence the / 1_000_000.
        input_cost = (input_tokens / 1_000_000) * rates["input"]
        output_cost = (output_tokens / 1_000_000) * rates["output"]

        return {
            "model": self.model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
        }
