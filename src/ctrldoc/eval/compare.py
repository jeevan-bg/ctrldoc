"""compare_eval — per-concept-cluster verdict scoring.

The eval set grades a `CompareVerifier` on its ability to label
each concept cluster in a `(doc_a, doc_b)` pair with one of
{StrengthA, StrengthB, Gap}. This is the per-concept-cluster cost
summary from SPEC §6.6's `compare(A, B)` operation — asymmetric
transport in both directions where each cluster's net cost falls
into one of three buckets:

- `StrengthA`: A's claim about this concept is stronger than B's,
  either by modality ordering (MUST vs SHOULD) or qualifier scope.
- `StrengthB`: symmetric.
- `Gap`: the concept appears in only one doc — the asymmetric
  transport has no partner mass on the absent side.

The substrate pre-clusters claims (the verifier under test sees
labeled comparisons rather than raw claim lists). Clustering itself
is graded by other slices in the v1 arc — splitting concerns keeps
this eval focused on the verdict assignment. The runner strips
`gold_verdict` before invoking the verifier so the gold can't leak
into the verifier's input.

The metric is per-cluster accuracy on the 3-label space, with
per-class precision / recall surfaced so a verifier that collapses
to a single label can't hide behind a balanced-sounding number.

SPEC-REF: §6.6 (compare = per-concept-cluster strengths/weaknesses), §14
"""

from __future__ import annotations

from typing import Literal, Protocol, TypeAlias, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ctrldoc.eval.claim_extraction import ClaimTuple, DocTypeLiteral
from ctrldoc.eval.harness import EvalResult

COMPARE_VERDICT_THRESHOLD = 0.85

CompareVerdictLiteral: TypeAlias = Literal["StrengthA", "StrengthB", "Gap"]

COMPARE_VERDICTS: tuple[CompareVerdictLiteral, ...] = ("StrengthA", "StrengthB", "Gap")


class ConceptComparisonInput(BaseModel):
    """One concept cluster shown to a verifier — no gold attached."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    label: str
    a_claim: ClaimTuple | None = None
    b_claim: ClaimTuple | None = None

    @model_validator(mode="after")
    def _at_least_one_side(self) -> ConceptComparisonInput:
        if self.a_claim is None and self.b_claim is None:
            raise ValueError("ConceptComparisonInput must carry at least one claim (a or b)")
        return self


class ConceptComparison(BaseModel):
    """A `ConceptComparisonInput` paired with its gold verdict.

    `Gap` verdicts occur when exactly one side is `None`.
    `StrengthA` / `StrengthB` require both sides present — comparing
    strength is only meaningful when there is something to compare.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    label: str
    a_claim: ClaimTuple | None = None
    b_claim: ClaimTuple | None = None
    gold_verdict: CompareVerdictLiteral

    @model_validator(mode="after")
    def _at_least_one_side(self) -> ConceptComparison:
        if self.a_claim is None and self.b_claim is None:
            raise ValueError("ConceptComparison must carry at least one claim (a or b)")
        return self

    @model_validator(mode="after")
    def _verdict_consistent_with_sides(self) -> ConceptComparison:
        both_present = self.a_claim is not None and self.b_claim is not None
        if self.gold_verdict == "Gap" and both_present:
            raise ValueError(f"cluster {self.id!r}: Gap verdict requires exactly one side absent")
        if self.gold_verdict != "Gap" and not both_present:
            raise ValueError(
                f"cluster {self.id!r}: {self.gold_verdict} verdict requires both sides present"
            )
        return self

    def to_input(self) -> ConceptComparisonInput:
        return ConceptComparisonInput(
            id=self.id,
            label=self.label,
            a_claim=self.a_claim,
            b_claim=self.b_claim,
        )


class CompareEvalCase(BaseModel):
    """One compare case: a pair of typed docs with labeled clusters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    a_doc_type: DocTypeLiteral
    b_doc_type: DocTypeLiteral
    clusters: list[ConceptComparison] = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)

    @field_validator("clusters")
    @classmethod
    def _cluster_ids_unique(cls, v: list[ConceptComparison]) -> list[ConceptComparison]:
        ids = [c.id for c in v]
        if len(ids) != len(set(ids)):
            raise ValueError("clusters must have unique ids within a case")
        return v


def compare_accuracy(
    *,
    predicted: list[CompareVerdictLiteral],
    gold: list[CompareVerdictLiteral],
) -> dict[str, float]:
    """Per-cluster accuracy plus per-class precision / recall.

    The 3-label space is small enough that a constant-predictor can
    score deceptively well on imbalanced sets — surfacing per-class
    metrics makes that failure mode immediately visible in CI output.
    """
    if len(predicted) != len(gold):
        raise ValueError(
            f"predicted ({len(predicted)}) and gold ({len(gold)}) must align by length"
        )
    metrics: dict[str, float] = {"accuracy": 0.0}
    for label in COMPARE_VERDICTS:
        metrics[f"{label.lower()}_precision"] = 0.0
        metrics[f"{label.lower()}_recall"] = 0.0
    if not gold:
        return metrics

    correct = sum(1 for p, g in zip(predicted, gold, strict=True) if p == g)
    metrics["accuracy"] = correct / len(gold)
    for label in COMPARE_VERDICTS:
        pred_pos = sum(1 for p in predicted if p == label)
        gold_pos = sum(1 for g in gold if g == label)
        tp = sum(1 for p, g in zip(predicted, gold, strict=True) if p == g == label)
        metrics[f"{label.lower()}_precision"] = tp / pred_pos if pred_pos else 0.0
        metrics[f"{label.lower()}_recall"] = tp / gold_pos if gold_pos else 0.0
    return metrics


@runtime_checkable
class CompareVerifier(Protocol):
    """Verifier under evaluation.

    Receives the unlabeled cluster inputs in case order and returns
    one verdict per cluster in the same order. The runner strips the
    gold verdict before invoking the verifier so it can never read
    it directly.
    """

    def verdicts(
        self,
        *,
        clusters: list[ConceptComparisonInput],
    ) -> list[CompareVerdictLiteral]: ...


class CompareEvalRunner:
    """Adapt a `CompareVerifier` into the harness `CaseRunner` shape."""

    def __init__(self, *, verifier: CompareVerifier) -> None:
        self._verifier = verifier

    def run_case(self, case: CompareEvalCase) -> EvalResult:
        inputs = [c.to_input() for c in case.clusters]
        predicted = self._verifier.verdicts(clusters=inputs)
        if len(predicted) != len(case.clusters):
            raise ValueError(
                f"verifier returned {len(predicted)} verdicts for {len(case.clusters)} clusters"
            )
        gold: list[CompareVerdictLiteral] = [c.gold_verdict for c in case.clusters]
        metrics = compare_accuracy(predicted=predicted, gold=gold)
        return EvalResult(
            case_id=case.id,
            passed=metrics["accuracy"] >= COMPARE_VERDICT_THRESHOLD,
            score=metrics["accuracy"],
            metrics=metrics,
            notes=(
                f"a={case.a_doc_type}, b={case.b_doc_type}, "
                f"clusters={len(case.clusters)}, accuracy={metrics['accuracy']:.3f}"
            ),
        )


__all__ = [
    "COMPARE_VERDICTS",
    "COMPARE_VERDICT_THRESHOLD",
    "CompareEvalCase",
    "CompareEvalRunner",
    "CompareVerdictLiteral",
    "CompareVerifier",
    "ConceptComparison",
    "ConceptComparisonInput",
    "compare_accuracy",
]
