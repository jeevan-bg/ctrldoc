"""analytical_eval — recall on seeded weaknesses.

Each case carries a `doc_type` to drive `AnalyticalReviewPlaybook`
and a list of `SeededWeakness` entries describing the issues the
review should surface. The runner runs the playbook and counts how
many seeded weaknesses match at least one emitted `Finding` (same
lens, substring match on the claim). Per §8.2 the threshold is
`recall ≥ 0.80`.

SPEC-REF: §8.1 (analytical_eval), §8.2 (analytical_review metrics)
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ctrldoc.eval.harness import EvalResult
from ctrldoc.models import Finding
from ctrldoc.ops.review import AnalyticalReviewPlaybook

WEAKNESS_RECALL_THRESHOLD = 0.80


class SeededWeakness(BaseModel):
    """One known issue the review should surface."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    lens: str
    claim_pattern: str


class AnalyticalEvalCase(BaseModel):
    """One row in analytical_eval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    tags: list[str] = []
    doc_type: str
    seeded_weaknesses: list[SeededWeakness]


def matches_seeded(finding: Finding, seeded: SeededWeakness) -> bool:
    """True when a finding plausibly addresses the seeded weakness.

    Match rule: same lens name (`Finding.ctrldoc == seeded.lens`) AND
    a case-insensitive substring of the seeded pattern appears in the
    finding's claim.
    """
    if finding.ctrldoc != seeded.lens:
        return False
    return seeded.claim_pattern.lower() in finding.claim.lower()


def weakness_recall(findings: list[Finding], seeded: list[SeededWeakness]) -> float:
    """Fraction of seeded weaknesses with at least one matching finding."""
    if not seeded:
        return 0.0
    matched = 0
    for entry in seeded:
        if any(matches_seeded(finding, entry) for finding in findings):
            matched += 1
    return matched / len(seeded)


class AnalyticalEvalRunner:
    """Drive `AnalyticalReviewPlaybook` per case, score recall."""

    def __init__(self, *, playbook: AnalyticalReviewPlaybook) -> None:
        self._playbook = playbook

    def run_case(self, case: AnalyticalEvalCase) -> EvalResult:
        report = self._playbook.run(case.doc_type)
        recall = weakness_recall(report.findings, case.seeded_weaknesses)
        return EvalResult(
            case_id=case.id,
            passed=recall >= WEAKNESS_RECALL_THRESHOLD,
            score=recall,
            metrics={"weakness_recall": recall},
            notes=(
                f"seeded={len(case.seeded_weaknesses)}, "
                f"findings={len(report.findings)}, recall={recall:.3f}"
            ),
        )


__all__ = [
    "WEAKNESS_RECALL_THRESHOLD",
    "AnalyticalEvalCase",
    "AnalyticalEvalRunner",
    "SeededWeakness",
    "matches_seeded",
    "weakness_recall",
]
