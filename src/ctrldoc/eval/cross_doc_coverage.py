"""cross_doc_coverage_eval — per-target-claim verdict scoring.

The eval set grades a `CrossDocCoverageVerifier` on its ability to
decide, per target claim, whether a source claim list covers that
target. This is the `coverage(A → B)` reduction from SPEC §6.6 —
min-cost transport of B's claim-mass onto A's claim-mass, where
unmoved mass is uncovered.

The metric is per-target-claim accuracy on the two-label space
{Covered, Missing}. Per-claim accuracy ≥ 0.85 is the release gate
the optimal-transport `coverage` operation must clear (the eval
substrate here pins that contract; the implementation lands later
in the v1 arc).

Per-class precision / recall are surfaced too — the two-label space
is often imbalanced, so a single-number accuracy can hide a verifier
that always answers Covered.

SPEC-REF: §6.6 (optimal-transport coverage), §14 (eval substrate)
"""

from __future__ import annotations

from typing import Literal, Protocol, TypeAlias, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ctrldoc.eval.claim_extraction import ClaimTuple, DocTypeLiteral
from ctrldoc.eval.harness import EvalResult

CROSS_DOC_COVERAGE_THRESHOLD = 0.85

CoverageVerdictLiteral: TypeAlias = Literal["Covered", "Missing"]

COVERAGE_VERDICTS: tuple[CoverageVerdictLiteral, ...] = ("Covered", "Missing")


class TargetClaim(BaseModel):
    """One target claim paired with its expected coverage verdict."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    claim: ClaimTuple
    gold_verdict: CoverageVerdictLiteral


class CrossDocCoverageEvalCase(BaseModel):
    """One cross-doc coverage case.

    `source_claims` may be empty — that case exercises a verifier's
    handling of a source with no extractable content (target claims
    cannot be supported and should all be `Missing`). `target_claims`
    must be non-empty; an empty target has nothing to grade.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    source_doc_type: DocTypeLiteral
    target_doc_type: DocTypeLiteral
    source_claims: list[ClaimTuple] = Field(default_factory=list)
    target_claims: list[TargetClaim] = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)

    @field_validator("target_claims")
    @classmethod
    def _target_ids_unique(cls, v: list[TargetClaim]) -> list[TargetClaim]:
        ids = [t.id for t in v]
        if len(ids) != len(set(ids)):
            raise ValueError("target_claims must have unique ids within a case")
        return v


def coverage_accuracy(
    *,
    predicted: list[CoverageVerdictLiteral],
    gold: list[CoverageVerdictLiteral],
) -> dict[str, float]:
    """Per-claim accuracy plus per-class precision / recall.

    The verifier returns one prediction per target claim, in order;
    `predicted` and `gold` must align. Per-class metrics make the
    behaviour legible in the common imbalanced case — a verifier that
    blindly answers `Covered` would score high accuracy on a
    coverage-heavy set but collapse `missing_recall` to zero.
    """
    if len(predicted) != len(gold):
        raise ValueError(
            f"predicted ({len(predicted)}) and gold ({len(gold)}) must align by length"
        )
    if not gold:
        return {
            "accuracy": 0.0,
            "covered_precision": 0.0,
            "covered_recall": 0.0,
            "missing_precision": 0.0,
            "missing_recall": 0.0,
        }
    correct = sum(1 for p, g in zip(predicted, gold, strict=True) if p == g)
    accuracy = correct / len(gold)

    def per_class(label: CoverageVerdictLiteral) -> tuple[float, float]:
        pred_pos = sum(1 for p in predicted if p == label)
        gold_pos = sum(1 for g in gold if g == label)
        tp = sum(1 for p, g in zip(predicted, gold, strict=True) if p == g == label)
        precision = tp / pred_pos if pred_pos else 0.0
        recall = tp / gold_pos if gold_pos else 0.0
        return precision, recall

    cov_p, cov_r = per_class("Covered")
    mis_p, mis_r = per_class("Missing")
    return {
        "accuracy": accuracy,
        "covered_precision": cov_p,
        "covered_recall": cov_r,
        "missing_precision": mis_p,
        "missing_recall": mis_r,
    }


@runtime_checkable
class CrossDocCoverageVerifier(Protocol):
    """Verifier under evaluation.

    Given a source's full claim list plus a target's claim list, the
    verifier returns one `Covered` / `Missing` verdict per target
    claim, in input order. Order alignment is part of the contract —
    the runner relies on it to score by position.
    """

    def verdicts(
        self,
        *,
        source: list[ClaimTuple],
        target: list[ClaimTuple],
    ) -> list[CoverageVerdictLiteral]: ...


class CrossDocCoverageEvalRunner:
    """Adapt a `CrossDocCoverageVerifier` into the harness `CaseRunner` shape."""

    def __init__(self, *, verifier: CrossDocCoverageVerifier) -> None:
        self._verifier = verifier

    def run_case(self, case: CrossDocCoverageEvalCase) -> EvalResult:
        target_tuples = [tc.claim for tc in case.target_claims]
        predicted = self._verifier.verdicts(
            source=list(case.source_claims),
            target=target_tuples,
        )
        if len(predicted) != len(target_tuples):
            raise ValueError(
                f"verifier returned {len(predicted)} verdicts for "
                f"{len(target_tuples)} target claims"
            )
        gold: list[CoverageVerdictLiteral] = [tc.gold_verdict for tc in case.target_claims]
        metrics = coverage_accuracy(predicted=predicted, gold=gold)
        return EvalResult(
            case_id=case.id,
            passed=metrics["accuracy"] >= CROSS_DOC_COVERAGE_THRESHOLD,
            score=metrics["accuracy"],
            metrics=metrics,
            notes=(
                f"source={case.source_doc_type}, target={case.target_doc_type}, "
                f"targets={len(target_tuples)}, accuracy={metrics['accuracy']:.3f}"
            ),
        )


__all__ = [
    "COVERAGE_VERDICTS",
    "CROSS_DOC_COVERAGE_THRESHOLD",
    "CoverageVerdictLiteral",
    "CrossDocCoverageEvalCase",
    "CrossDocCoverageEvalRunner",
    "CrossDocCoverageVerifier",
    "TargetClaim",
    "coverage_accuracy",
]
