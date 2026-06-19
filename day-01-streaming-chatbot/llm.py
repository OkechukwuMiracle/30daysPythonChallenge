"""
llm.py - A small, reusable wrapper around the Anthropic (Claude) Python SDK.

Why this exists: Claude's API isn't OpenAI-schema-compatible (system prompts
are a top-level parameter, not a message; max_tokens is required; responses
are a list of content blocks), so this wrapper hides those details behind the
same chat()/stream() shape we'll reuse in every other day's project.
"""

import os
from typing import Generator, Optional

import anthropic
from dotenv import load_dotenv

# Load variables from a .env file into the environment as soon as this
# module is imported, so any script that imports llm.py gets the key for free.
load_dotenv()


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
