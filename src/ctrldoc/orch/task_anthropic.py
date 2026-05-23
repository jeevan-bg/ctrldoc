"""Anthropic backend for the stateless task primitive.

The rendered `CacheablePrefix` is the byte-stable system message
across every sub-task in a session. This wrapper sends it as a single
system block carrying `cache_control: ephemeral`, so Anthropic's
prompt cache keys on that prefix — N fresh sub-tasks reuse one
cached prefix instead of paying full price N times (SPEC §3.1
pillar 2).

The wrapper is a thin transport. The runner (S-060) keeps all
parsing and validation; this module only ships request shape and
returns the model's raw text.

SPEC-REF: §3.1, §4.5
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from anthropic import Anthropic


class AnthropicTaskClient:
    """TaskClient backed by Anthropic Messages with cache_control on the prefix."""

    def __init__(
        self,
        *,
        model: str = "claude-opus-4-7",
        max_output_tokens: int = 2048,
        client: Anthropic | None = None,
    ) -> None:
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._client = client

    def _ensure_client(self) -> Anthropic:
        if self._client is None:
            from anthropic import Anthropic

            self._client = Anthropic()
        return self._client

    def call(self, *, system: str, user: str) -> str:
        client = self._ensure_client()
        system_blocks: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        message = client.messages.create(
            model=self._model,
            max_tokens=self._max_output_tokens,
            system=cast(Any, system_blocks),
            messages=[{"role": "user", "content": user}],
        )
        return _extract_text(message)


def _extract_text(message: object) -> str:
    parts: list[str] = []
    for block in getattr(message, "content", []) or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


__all__ = ["AnthropicTaskClient"]
