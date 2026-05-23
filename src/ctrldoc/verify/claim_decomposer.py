"""Claim decomposer — protocol and heuristic reference.

`ClaimDecomposer.decompose(answer)` turns an answer text into a list
of atomic claim strings. The heuristic reference splits on sentence
terminators; an Anthropic-backed implementation lives in
`claim_decomposer_anthropic.py` and constrains output to JSON.

SPEC-REF: §4.4 (verifier step 1)
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@runtime_checkable
class ClaimDecomposer(Protocol):
    """Answer text → list of atomic claim strings."""

    def decompose(self, answer: str) -> list[str]: ...


class HeuristicClaimDecomposer:
    """Split on sentence terminators. Drops adjacent duplicates.

    Useful for unit tests of layers above the decomposer and as a
    cheap fallback when an LLM call is not desired.
    """

    def decompose(self, answer: str) -> list[str]:
        body = answer.strip()
        if not body:
            return []
        parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(body)]
        parts = [part for part in parts if part]
        if not parts:
            return [body]
        deduped: list[str] = []
        for part in parts:
            if not deduped or deduped[-1] != part:
                deduped.append(part)
        return deduped


__all__ = ["ClaimDecomposer", "HeuristicClaimDecomposer"]
