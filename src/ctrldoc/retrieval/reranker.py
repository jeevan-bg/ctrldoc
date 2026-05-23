"""Cross-encoder reranker — protocol and dependency-free references.

`Reranker.rerank(query, candidates, *, k)` rescores each candidate
against the query jointly and returns the top-k `(chunk_id, score)`
pairs. The MVP ships two references: `IdentityReranker` (passthrough,
truncates to k) and `LexicalReranker` (Jaccard token overlap). A
BGE-reranker-v2-m3 backend can satisfy the same protocol later.

SPEC-REF: §4.3 (reranker)
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict


class Candidate(BaseModel):
    """One row handed to the reranker — text plus the chunk id we're scoring."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    text: str


RerankHit = tuple[str, float]


@runtime_checkable
class Reranker(Protocol):
    """Score `(query, candidate.text)` jointly; return top-k by score."""

    def rerank(
        self,
        query: str,
        candidates: list[Candidate],
        *,
        k: int,
    ) -> list[RerankHit]: ...


class IdentityReranker:
    """Passthrough — preserve input order, truncate to `k`.

    Useful when the upstream ranking is already known to be good (or as
    a control in evals).
    """

    def rerank(
        self,
        query: str,
        candidates: list[Candidate],
        *,
        k: int,
    ) -> list[RerankHit]:
        if k < 0:
            raise ValueError("k must be non-negative")
        return [(c.chunk_id, 0.0) for c in candidates[:k]]


class LexicalReranker:
    """Score by Jaccard token overlap between query and candidate text.

    Deterministic and dependency-free — useful as a baseline and as a
    behavioural oracle for the production BGE cross-encoder.
    """

    def rerank(
        self,
        query: str,
        candidates: list[Candidate],
        *,
        k: int,
    ) -> list[RerankHit]:
        if k < 0:
            raise ValueError("k must be non-negative")
        if not candidates or k == 0:
            return []
        query_tokens = _tokenize(query)
        scored = [
            (idx, c.chunk_id, _jaccard(query_tokens, _tokenize(c.text)))
            for idx, c in enumerate(candidates)
        ]
        scored.sort(key=lambda item: (-item[2], item[0]))
        return [(chunk_id, score) for _, chunk_id, score in scored[:k]]


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


__all__ = [
    "Candidate",
    "IdentityReranker",
    "LexicalReranker",
    "RerankHit",
    "Reranker",
]
