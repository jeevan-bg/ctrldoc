"""GLiNER-backed `NERTagger`.

Kept in a separate module so importing `ctrldoc.ingest.ner` doesn't
pull the GLiNER (and transitively torch + transformers) install
unless the caller actually wants the real backend.

SPEC-REF: §4.1 (ingest step 3 — NER)
"""

from __future__ import annotations

from typing import Any

from ctrldoc.ingest.ner import EntityMention


class GLiNERTagger:
    """Zero-shot NER via the `urchade/gliner_*` family of models.

    The model is loaded lazily on first `tag()` call so that simply
    importing this module is cheap.
    """

    def __init__(
        self,
        *,
        model_name: str = "urchade/gliner_small-v2.1",
        score_threshold: float = 0.5,
    ) -> None:
        self._model_name = model_name
        self._score_threshold = score_threshold
        self._model: Any | None = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            from gliner import GLiNER  # type: ignore[import-untyped,import-not-found,unused-ignore]

            self._model = GLiNER.from_pretrained(self._model_name)
        return self._model

    def tag(self, text: str, *, labels: list[str]) -> list[EntityMention]:
        if not text.strip() or not labels:
            return []
        model = self._ensure_model()
        raw = model.predict_entities(text, labels=labels, threshold=self._score_threshold)
        mentions: list[EntityMention] = []
        for r in raw:
            mentions.append(
                EntityMention(
                    text=str(r["text"]),
                    label=str(r["label"]),
                    start=int(r["start"]),
                    end=int(r["end"]),
                    score=float(r["score"]),
                )
            )
        return mentions


__all__ = ["GLiNERTagger"]
