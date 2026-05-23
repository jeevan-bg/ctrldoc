"""DeBERTa-v3-large-mnli backend for the `NLIChecker` protocol.

Kept in a separate module so importing `ctrldoc.verify.nli` does
not pull `torch` + `transformers` into memory unless the caller
actually wants the production cross-encoder. The model is loaded
lazily on first `check()` call.

SPEC-REF: §4.4 (verifier step 3 — NLI check)
"""

from __future__ import annotations

from typing import Any

from ctrldoc.verify.nli import NLILabel, NLIResult


class DeBERTaNLIChecker:
    """Cross-encoder NLI using `cross-encoder/nli-deberta-v3-large`.

    The model emits three logits per `(premise, hypothesis)` pair;
    softmax-normalised probabilities yield the label + score. Labels
    follow the model's own `id2label` table — typically
    `{0: "contradiction", 1: "entailment", 2: "neutral"}` — and are
    mapped to the spec's three-label vocabulary one-for-one.
    """

    def __init__(
        self,
        *,
        model_name: str = "cross-encoder/nli-deberta-v3-large",
        max_length: int = 512,
    ) -> None:
        self._model_name = model_name
        self._max_length = max_length
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._label_map: dict[int, NLILabel] | None = None

    def _ensure_loaded(self) -> tuple[Any, Any, dict[int, NLILabel]]:
        if self._tokenizer is None or self._model is None or self._label_map is None:
            from transformers import (  # type: ignore[import-untyped]
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
            model.eval()
            self._model = model
            self._label_map = _normalise_label_map(model.config.id2label)
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._label_map is not None
        return self._tokenizer, self._model, self._label_map

    def check(self, premise: str, hypothesis: str) -> NLIResult:
        if not premise.strip() or not hypothesis.strip():
            return NLIResult(label="neutral", score=0.0)
        import torch

        tokenizer, model, label_map = self._ensure_loaded()
        with torch.no_grad():
            inputs = tokenizer(
                premise,
                hypothesis,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self._max_length,
            )
            logits = model(**inputs, return_dict=True).logits.view(-1)
            probs = torch.softmax(logits.float(), dim=-1).tolist()
        idx = max(range(len(probs)), key=lambda i: probs[i])
        return NLIResult(label=label_map[idx], score=float(probs[idx]))


_ALLOWED_LABELS: set[NLILabel] = {"entailment", "neutral", "contradiction"}


def _normalise_label_map(id2label: dict[int, str] | dict[str, str]) -> dict[int, NLILabel]:
    """Lower-case + validate the model's `id2label` against the spec's three-label vocabulary."""
    mapping: dict[int, NLILabel] = {}
    for raw_key, raw_value in id2label.items():
        key = int(raw_key)
        value = str(raw_value).lower()
        if value not in _ALLOWED_LABELS:
            raise ValueError(
                f"unexpected NLI label {raw_value!r} from model id2label; "
                f"expected one of {sorted(_ALLOWED_LABELS)}"
            )
        mapping[key] = value
    if set(mapping.values()) != _ALLOWED_LABELS:
        raise ValueError(
            f"model id2label is missing one of {sorted(_ALLOWED_LABELS)}; got {sorted(set(mapping.values()))}"
        )
    return mapping


__all__ = ["DeBERTaNLIChecker"]
