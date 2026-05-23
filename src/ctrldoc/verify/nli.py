"""Natural-language-inference checker — protocol and heuristic reference.

`NLIChecker.check(premise, hypothesis)` returns one of three labels —
`entailment`, `neutral`, `contradiction` — with a `[0, 1]` score. The
heuristic reference looks at lower-cased token overlap and classifies
`entailment` when every hypothesis token is present in the premise.
It does not detect contradiction; downstream callers can layer that
on (or wait for the DeBERTa backend in S-051b).

SPEC-REF: §4.4 (verifier step 3 — NLI check)
"""

from __future__ import annotations

import re
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from ctrldoc.models import UnitInterval

NLILabel = Literal["entailment", "neutral", "contradiction"]


class NLIResult(BaseModel):
    """One inference judgement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: NLILabel
    score: UnitInterval


@runtime_checkable
class NLIChecker(Protocol):
    """Premise + hypothesis → labelled inference result."""

    def check(self, premise: str, hypothesis: str) -> NLIResult: ...


class HeuristicNLIChecker:
    """Token-overlap heuristic. Deterministic, no dependencies.

    `score` is the fraction of (lower-cased) hypothesis tokens that
    also appear in the premise. The label is `entailment` when
    `score >= entailment_threshold`, otherwise `neutral`.
    `contradiction` is never produced — that's a job for the DeBERTa
    backend.
    """

    def __init__(self, *, entailment_threshold: float = 0.999) -> None:
        if not 0.0 <= entailment_threshold <= 1.0:
            raise ValueError("entailment_threshold must be in [0, 1]")
        self._threshold = entailment_threshold

    def check(self, premise: str, hypothesis: str) -> NLIResult:
        hyp_tokens = _tokenize(hypothesis)
        if not hyp_tokens:
            return NLIResult(label="neutral", score=0.0)
        prem_tokens = _tokenize(premise)
        if not prem_tokens:
            return NLIResult(label="neutral", score=0.0)
        overlap = sum(1 for token in hyp_tokens if token in prem_tokens)
        score = overlap / len(hyp_tokens)
        label: NLILabel = "entailment" if score >= self._threshold else "neutral"
        return NLIResult(label=label, score=score)


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


__all__ = ["HeuristicNLIChecker", "NLIChecker", "NLILabel", "NLIResult"]
