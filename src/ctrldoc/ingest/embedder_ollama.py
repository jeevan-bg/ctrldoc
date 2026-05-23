"""BGE-M3 backend for the `Embedder` protocol via a local Ollama service.

Kept in a separate module so importing `ctrldoc.ingest.embedder`
does not require the `ollama` SDK unless the caller actually wants
the production embedder. The SDK client is constructed lazily on
first `embed*()` call and reused across calls.

SPEC-REF: §4.1 (ingest step 5 — embed), §4.2 (dense vectors)
"""

from __future__ import annotations

import math
from typing import Any

_BGE_M3_DIMENSION = 1024


class OllamaEmbedder:
    """Dense BGE-M3 embeddings via a local Ollama HTTP service.

    Vectors are L2-normalised to keep the downstream cosine
    similarity path numerically identical to `HashEmbedder` —
    callers can dot-product directly. Empty input maps to the
    zero vector by convention (same as `HashEmbedder`).
    """

    def __init__(
        self,
        *,
        model: str = "bge-m3",
        host: str = "http://127.0.0.1:11434",
        dimension: int = _BGE_M3_DIMENSION,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._model = model
        self._host = host
        self._dimension = dimension
        self._client: Any | None = None

    @property
    def dimension(self) -> int:
        return self._dimension

    def _ensure_client(self) -> Any:
        if self._client is None:
            import ollama

            self._client = ollama.Client(host=self._host)
        return self._client

    def embed(self, text: str) -> list[float]:
        if not text:
            return [0.0] * self._dimension
        client = self._ensure_client()
        response = client.embeddings(model=self._model, prompt=text)
        raw = list(response["embedding"])
        return _l2_normalise(raw, self._dimension)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def _l2_normalise(vec: list[float], expected_dim: int) -> list[float]:
    if len(vec) != expected_dim:
        raise ValueError(
            f"expected {expected_dim}-d embedding, got {len(vec)}-d (model dim mismatch)"
        )
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return [0.0] * expected_dim
    return [x / norm for x in vec]


__all__ = ["OllamaEmbedder"]
