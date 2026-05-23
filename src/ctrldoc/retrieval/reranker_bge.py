"""BGE-reranker-v2-m3 backend for the `Reranker` protocol.

Kept in a separate module so importing `ctrldoc.retrieval.reranker`
does not pull `torch` + `transformers` into memory unless the caller
actually wants the production cross-encoder. The model is loaded
lazily on first `rerank()` call.

SPEC-REF: §4.3 (reranker)
"""

from __future__ import annotations

from typing import Any

from ctrldoc.retrieval.reranker import Candidate, RerankHit


class BGEReranker:
    """Cross-encoder reranker using `BAAI/bge-reranker-v2-m3`.

    Each `(query, candidate.text)` pair is scored jointly by the
    cross-encoder, then candidates are returned in descending score
    order, truncated to `k`. Ties resolve by input order.
    """

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        max_length: int = 512,
    ) -> None:
        self._model_name = model_name
        self._max_length = max_length
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._tokenizer is None or self._model is None:
            from transformers import (  # type: ignore[import-untyped]
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
            model.eval()
            self._model = model
        return self._tokenizer, self._model

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
        import torch

        tokenizer, model = self._ensure_loaded()
        pairs = [[query, c.text] for c in candidates]
        with torch.no_grad():
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self._max_length,
            )
            logits = model(**inputs, return_dict=True).logits.view(-1)
            scores = logits.float().tolist()
        scored = [
            (idx, candidates[idx].chunk_id, float(scores[idx])) for idx in range(len(candidates))
        ]
        scored.sort(key=lambda item: (-item[2], item[0]))
        return [(chunk_id, score) for _, chunk_id, score in scored[:k]]


__all__ = ["BGEReranker"]
