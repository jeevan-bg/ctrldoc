"""Dense-vector index — protocol and pure-Python reference.

Defines the contract every dense-vector backend satisfies: pin a
dimension, accept `(chunk_id, embedding)` pairs, return top-k by
cosine similarity. The `InMemoryVectorIndex` reference is the
behavioural oracle for the production `sqlite-vec`-backed index.

SPEC-REF: §4.2 (dense vectors), §4.3 (retrieval)
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import Protocol, runtime_checkable


class VectorDimensionMismatchError(ValueError):
    """Raised when an embedding's length disagrees with the pinned dimension."""


VectorHit = tuple[str, float]


@runtime_checkable
class VectorIndex(Protocol):
    """Top-k cosine-similarity search over `(chunk_id, embedding)` pairs."""

    @property
    def dimension(self) -> int: ...

    def add(self, chunk_id: str, embedding: Sequence[float]) -> None: ...

    def remove(self, chunk_id: str) -> None: ...

    def search(self, query: Sequence[float], *, k: int) -> list[VectorHit]: ...

    def iter(self) -> Iterator[tuple[str, list[float]]]: ...


class InMemoryVectorIndex:
    """Reference dense-vector index for tests and downstream layers.

    Cosine similarity is computed in pure Python; this is intentionally
    O(N * d) per query because the goal here is contract fidelity, not
    throughput. A `sqlite-vec`-backed index replaces this for real
    corpora.
    """

    def __init__(self, *, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension
        self._vectors: dict[str, tuple[list[float], float]] = {}

    @property
    def dimension(self) -> int:
        return self._dimension

    def add(self, chunk_id: str, embedding: Sequence[float]) -> None:
        self._check_dim(embedding, "add")
        vec = list(map(float, embedding))
        self._vectors[chunk_id] = (vec, _norm(vec))

    def remove(self, chunk_id: str) -> None:
        self._vectors.pop(chunk_id, None)

    def search(self, query: Sequence[float], *, k: int) -> list[VectorHit]:
        if k < 0:
            raise ValueError("k must be non-negative")
        if k == 0 or not self._vectors:
            return []
        self._check_dim(query, "search")
        q = list(map(float, query))
        q_norm = _norm(q)
        hits: list[VectorHit] = []
        for chunk_id, (vec, vec_norm) in self._vectors.items():
            denom = q_norm * vec_norm
            score = 0.0 if denom == 0.0 else _dot(q, vec) / denom
            hits.append((chunk_id, score))
        hits.sort(key=lambda h: h[1], reverse=True)
        return hits[:k]

    def iter(self) -> Iterator[tuple[str, list[float]]]:
        for chunk_id, (vec, _norm_value) in self._vectors.items():
            yield chunk_id, list(vec)

    def _check_dim(self, vec: Sequence[float], op: str) -> None:
        if len(vec) != self._dimension:
            raise VectorDimensionMismatchError(
                f"{op} vector has dimension {len(vec)}; expected {self._dimension}"
            )


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


__all__ = [
    "InMemoryVectorIndex",
    "VectorDimensionMismatchError",
    "VectorHit",
    "VectorIndex",
]
