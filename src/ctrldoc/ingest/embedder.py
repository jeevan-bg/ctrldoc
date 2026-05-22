"""Embedder protocol and deterministic hash-based reference.

`Embedder` is the contract every embedding backend (the production
BGE-M3 via Ollama, or any future replacement) must satisfy.
`HashEmbedder` is a dependency-free reference that maps text →
unit-normalised float vector deterministically — useful in tests
and as a stand-in for downstream slices until the Ollama backend
lands.

SPEC-REF: §4.1 (ingest step 5 — embed), §4.2 (dense vectors)
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Text → dense vector. Backends pin `dimension` at construction."""

    @property
    def dimension(self) -> int: ...

    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Deterministic hash-based embedder.

    Each text is expanded to `dimension` floats by repeatedly hashing
    `(seed || counter || text)` with sha512 and unpacking the digest
    as little-endian signed int32s, then mapping each int32 to
    `[-1, 1]` by dividing by `2**31`. The vector is unit normalised.
    Empty input maps to the zero vector by convention.
    """

    _INT_DIVISOR = float(2**31)
    _INTS_PER_DIGEST = 16  # 64 bytes / 4 bytes per int32

    def __init__(self, *, dimension: int, seed: int = 0) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension
        self._seed = seed

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        if not text:
            return [0.0] * self._dimension
        encoded = text.encode("utf-8")
        seed_bytes = self._seed.to_bytes(8, "little", signed=False)
        floats: list[float] = []
        counter = 0
        while len(floats) < self._dimension:
            block = hashlib.sha512(
                seed_bytes + counter.to_bytes(4, "little", signed=False) + encoded,
            ).digest()
            chunk = struct.unpack("<16i", block)
            for raw in chunk:
                floats.append(raw / self._INT_DIVISOR)
                if len(floats) == self._dimension:
                    break
            counter += 1

        norm = math.sqrt(sum(x * x for x in floats))
        if norm == 0.0:
            return [0.0] * self._dimension
        return [x / norm for x in floats]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


__all__ = ["Embedder", "HashEmbedder"]
