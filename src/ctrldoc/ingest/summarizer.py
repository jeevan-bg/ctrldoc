"""Section summariser — protocol and dependency-free reference.

Each `Section` gets a 1 or 2 sentence summary that becomes its row in the
`doc_skeleton` consumed by every cacheable-prefix call. The protocol
keeps the summariser swap-out trivial: an LLM-backed implementation
satisfies the same surface as the regex-based heuristic.

SPEC-REF: §4.1 (ingest step 7), §3.1 (cacheable prefix)
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Protocol, runtime_checkable

from ctrldoc.models import Section

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@runtime_checkable
class Summarizer(Protocol):
    """Body text → short summary (target: 1 or 2 sentences)."""

    def summarize(self, text: str) -> str: ...


class HeuristicSummarizer:
    """Pick the first `max_sentences` sentences. No model, no network."""

    def __init__(self, *, max_sentences: int = 2) -> None:
        if max_sentences <= 0:
            raise ValueError("max_sentences must be positive")
        self._max_sentences = max_sentences

    def summarize(self, text: str) -> str:
        body = text.strip()
        if not body:
            return ""
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(body) if s.strip()]
        if not sentences:
            return body
        return " ".join(sentences[: self._max_sentences])


def summarize_sections(
    sections: Iterable[Section],
    *,
    body_for: Callable[[Section], str],
    summarizer: Summarizer,
) -> list[Section]:
    """Apply `summarizer` to each section body, returning updated `Section`s.

    `body_for` resolves a section to its full text — usually by
    concatenating the section's chunks. Structural fields (id,
    parent_id, title, chunk_ids) pass through unchanged so storage
    and skeleton rendering see the same tree shape.
    """
    out: list[Section] = []
    for section in sections:
        body = body_for(section)
        summary = summarizer.summarize(body)
        out.append(section.model_copy(update={"summary": summary}))
    return out


__all__ = [
    "HeuristicSummarizer",
    "Summarizer",
    "summarize_sections",
]
