"""LLM adapters.

The extractor talks to a `Protocol`, not a vendor. That keeps the tests offline
and key-free — the interesting logic here is the validation and repair, not the
HTTP call, and it should be testable without an API key or a network.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable


@runtime_checkable
class LLM(Protocol):
    def complete(self, prompt: str, *, schema: dict[str, Any]) -> str:
        """Return the model's raw response (a JSON string)."""


class ScriptedLLM:
    """Returns queued responses in order, then repeats the last one.

    Used by the tests, and useful for reproducing a specific failure without
    paying for it. Records every prompt it received so tests can assert what the
    repair attempt actually said.
    """

    def __init__(self, responses: Sequence[str]) -> None:
        if not responses:
            raise ValueError("need at least one response")
        self._responses = list(responses)
        self.prompts: list[str] = []
        self.schemas: list[dict[str, Any]] = []

    def complete(self, prompt: str, *, schema: dict[str, Any]) -> str:
        self.prompts.append(prompt)
        self.schemas.append(schema)
        index = min(len(self.prompts) - 1, len(self._responses) - 1)
        return self._responses[index]


class ClaudeLLM:
    """Real adapter using the Messages API with structured output.

    `output_config.format` constrains the response to the supplied JSON Schema,
    so the reply is always valid JSON with the right keys and types. That is
    exactly why the repair loop exists: the shape is guaranteed, the *content*
    is not, and a schema-valid response full of empty strings still needs a
    second attempt.

    Retries for 429 and 5xx are the SDK's own (`max_retries`), not hand-rolled —
    the SDK already does exponential backoff, and a second implementation on top
    just doubles the delay.
    """

    def __init__(
        self,
        *,
        model: str = "claude-opus-5-5",
        max_tokens: int = 4_000,
        max_retries: int = 3,
        client: Any | None = None,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            import anthropic  # imported lazily so tests need no dependency

            self._client = anthropic.Anthropic(max_retries=max_retries)

        self.model = model
        self.max_tokens = max_tokens

    def complete(self, prompt: str, *, schema: dict[str, Any]) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": schema,
                },
                # Extraction is not a reasoning-heavy task, so effort can sit
                # below the default ("medium" on this model) at no quality cost.
                # Raise it only if a field is genuinely ambiguous.
                "effort": "low",
            },
        )

        # With output_config.format the first text block is the JSON payload.
        for block in response.content:
            if block.type == "text":
                return block.text
        raise RuntimeError("model returned no text block")
