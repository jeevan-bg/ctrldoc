"""Anthropic-backed claim decomposer with JSON-constrained output.

The model is instructed to emit a single JSON object of the shape
`{"claims": [<str>, ...]}` — nothing else. The wrapper validates the
shape, filters out non-string entries (a common model failure mode),
and strips surrounding whitespace.

SPEC-REF: §4.4
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anthropic import Anthropic


_SYSTEM_PROMPT = (
    "You decompose an answer into atomic, verifiable claims. Each claim "
    "must stand alone — a reader who hasn't seen the original answer "
    "should be able to check it independently. Avoid editorialising or "
    "inventing facts. Return one JSON object only, of the form:\n"
    '  {"claims": ["claim one.", "claim two.", ...]}\n'
    "No prose, no code fences, no commentary outside the JSON."
)


class AnthropicClaimDecomposer:
    """Decompose answers into atomic claims via the Anthropic Messages API."""

    def __init__(
        self,
        *,
        model: str = "claude-haiku-4-5",
        max_output_tokens: int = 512,
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

    def decompose(self, answer: str) -> list[str]:
        body = answer.strip()
        if not body:
            return []
        client = self._ensure_client()
        message = client.messages.create(
            model=self._model,
            max_tokens=self._max_output_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": body}],
        )
        return _parse_claims(_extract_text(message))


def _extract_text(message: object) -> str:
    parts: list[str] = []
    for block in getattr(message, "content", []) or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts).strip()


def _parse_claims(text: str) -> list[str]:
    try:
        payload: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"claim decomposer returned non-JSON: {text[:80]!r}") from exc
    if not isinstance(payload, dict) or "claims" not in payload:
        raise ValueError("claim decomposer response missing 'claims' field")
    raw_claims = payload["claims"]
    if not isinstance(raw_claims, list):
        raise ValueError("claim decomposer 'claims' must be a list")
    cleaned: list[str] = []
    for entry in raw_claims:
        if not isinstance(entry, str):
            continue
        stripped = entry.strip()
        if stripped:
            cleaned.append(stripped)
    return cleaned


__all__ = ["AnthropicClaimDecomposer"]
