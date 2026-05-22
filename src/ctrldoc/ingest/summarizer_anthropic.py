"""Anthropic-backed `Summarizer`.

Kept in a separate module so importing `ctrldoc.ingest.summarizer`
does not require the `anthropic` SDK to be installed — callers that
only want the heuristic reference avoid pulling it.

SPEC-REF: §4.1 (ingest step 7 — one LLM pass per section)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anthropic import Anthropic


_SYSTEM_PROMPT = (
    "You produce one or two factual sentences that summarise the section "
    "body the user gives you. Do not editorialise. Do not invent facts. "
    "Do not include markdown, citations, or headings. Output the summary "
    "text only — no preamble, no quotes."
)


class AnthropicSummarizer:
    """Per-section summariser using the Anthropic Messages API.

    The model name and max output tokens are explicit constructor
    arguments so callers can swap a cheaper haiku for the planner
    Opus when iterating. An `Anthropic` client can be passed in for
    tests / mocking; otherwise one is created lazily from the
    environment.
    """

    def __init__(
        self,
        *,
        model: str = "claude-haiku-4-5",
        max_output_tokens: int = 96,
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

    def summarize(self, text: str) -> str:
        body = text.strip()
        if not body:
            return ""
        client = self._ensure_client()
        message = client.messages.create(
            model=self._model,
            max_tokens=self._max_output_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": body}],
        )
        return _extract_text(message).strip()


def _extract_text(message: object) -> str:
    content = getattr(message, "content", [])
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


__all__ = ["AnthropicSummarizer"]
