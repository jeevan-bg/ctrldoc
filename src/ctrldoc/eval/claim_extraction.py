"""claim_extraction_eval — universal claim tuple extractor scoring.

The eval set grades a `ClaimExtractor` on its ability to recover
universal-claim-tuple lists from individual sentences. The set-based
metric is precision / recall / F1 over a "core match" of normalized
subject / predicate / object plus exact-match polarity and modality;
the qualifier slot is enforced only when the gold tuple sets one.

Per §14 the release gate is `claim_F1 >= 0.85`.

The tuple shape is the v1 logic floor from §6.2:

    Claim = (subject, predicate, object, polarity, modality, qualifier,
             span_refs, confidence)

`span_refs` and `confidence` belong to the production graph; gold
tuples here exercise only the logical content, so those fields are
not part of the eval tuple. The fields scored are the six that decide
contradiction (polarity flip), strength (qualifier ordering), and
modal force (modality).

SPEC-REF: §6.2 (universal claim tuple), §14 (claim_F1 gate)
"""

from __future__ import annotations

import re
from typing import Literal, Protocol, TypeAlias, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ctrldoc.eval.harness import EvalResult

CLAIM_F1_THRESHOLD = 0.85

PolarityLiteral: TypeAlias = Literal["affirmative", "negative"]
ModalityLiteral: TypeAlias = Literal[
    "asserted",  # plain factual statement
    "obligatory",  # MUST / SHALL / required
    "recommended",  # SHOULD / encouraged
    "permitted",  # MAY / optional
    "prohibited",  # MUST NOT / SHALL NOT
    "hypothetical",  # could / might / would / conditional
]
DocTypeLiteral: TypeAlias = Literal[
    "spec",
    "runbook",
    "rfc",
    "legal",
    "academic",
    "narrative",
]

POLARITIES: tuple[PolarityLiteral, ...] = ("affirmative", "negative")
MODALITIES: tuple[ModalityLiteral, ...] = (
    "asserted",
    "obligatory",
    "recommended",
    "permitted",
    "prohibited",
    "hypothetical",
)
DOC_TYPES: tuple[DocTypeLiteral, ...] = (
    "spec",
    "runbook",
    "rfc",
    "legal",
    "academic",
    "narrative",
)


class ClaimTuple(BaseModel):
    """One universal claim tuple — the logic floor from SPEC §6.2."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    subject: str
    predicate: str
    object: str
    polarity: PolarityLiteral
    modality: ModalityLiteral
    qualifier: str = ""


_TRAILING_PUNCT = re.compile(r"[\.\?!,;:]+$")
_WHITESPACE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """Lowercase, strip outer whitespace, collapse runs of inner
    whitespace, and drop trailing sentence punctuation."""
    s = s.strip().lower()
    s = _WHITESPACE.sub(" ", s)
    s = _TRAILING_PUNCT.sub("", s)
    return s


def claim_tuple_matches(*, extracted: ClaimTuple, gold: ClaimTuple) -> bool:
    """Core-match equality on the six scored fields.

    Subject, predicate, object, and qualifier are compared after
    `normalize_text`; polarity and modality require exact-literal
    match. Qualifier is enforced only when the gold sets it — an
    empty gold qualifier means the test does not exercise the slot.
    """
    if normalize_text(extracted.subject) != normalize_text(gold.subject):
        return False
    if normalize_text(extracted.predicate) != normalize_text(gold.predicate):
        return False
    if normalize_text(extracted.object) != normalize_text(gold.object):
        return False
    if extracted.polarity != gold.polarity:
        return False
    if extracted.modality != gold.modality:
        return False
    gold_qual = normalize_text(gold.qualifier)
    return not (gold_qual and normalize_text(extracted.qualifier) != gold_qual)


def precision_recall_f1(*, extracted: list[ClaimTuple], gold: list[ClaimTuple]) -> dict[str, float]:
    """Set-based precision / recall / F1 over claim-tuple core match.

    Greedy one-to-one assignment: each gold tuple may match at most
    one extracted tuple, so duplicate extractions do not inflate the
    score. With no gold or no extracted tuples the metrics are all
    zero (gold-empty cases are caller-rejected; we still surface 0.0
    rather than raising).
    """
    if not gold or not extracted:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    matched_gold_idx: set[int] = set()
    tp = 0
    for e in extracted:
        for gi, g in enumerate(gold):
            if gi in matched_gold_idx:
                continue
            if claim_tuple_matches(extracted=e, gold=g):
                matched_gold_idx.add(gi)
                tp += 1
                break
    precision = tp / len(extracted)
    recall = tp / len(gold)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


@runtime_checkable
class ClaimExtractor(Protocol):
    """Sentence-level extractor under evaluation."""

    def extract(self, sentence: str) -> list[ClaimTuple]: ...


class ClaimExtractionEvalCase(BaseModel):
    """One row in the claim-extraction eval set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    doc_type: DocTypeLiteral
    sentence: str
    gold_tuples: list[ClaimTuple] = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)

    @field_validator("sentence")
    @classmethod
    def _sentence_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("sentence must not be blank")
        return v


class ClaimExtractionEvalRunner:
    """Adapt a `ClaimExtractor` into the harness `CaseRunner` shape."""

    def __init__(self, *, extractor: ClaimExtractor) -> None:
        self._extractor = extractor

    def run_case(self, case: ClaimExtractionEvalCase) -> EvalResult:
        extracted = self._extractor.extract(case.sentence)
        prf = precision_recall_f1(extracted=extracted, gold=case.gold_tuples)
        return EvalResult(
            case_id=case.id,
            passed=prf["f1"] >= CLAIM_F1_THRESHOLD,
            score=prf["f1"],
            metrics=prf,
            notes=(
                f"doc_type={case.doc_type}, gold={len(case.gold_tuples)}, "
                f"extracted={len(extracted)}, f1={prf['f1']:.3f}"
            ),
        )


__all__ = [
    "CLAIM_F1_THRESHOLD",
    "DOC_TYPES",
    "MODALITIES",
    "POLARITIES",
    "ClaimExtractionEvalCase",
    "ClaimExtractionEvalRunner",
    "ClaimExtractor",
    "ClaimTuple",
    "DocTypeLiteral",
    "ModalityLiteral",
    "PolarityLiteral",
    "claim_tuple_matches",
    "normalize_text",
    "precision_recall_f1",
]
