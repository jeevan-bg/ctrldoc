"""qa_eval — score `QAPlaybook` runs against labelled cases.

Each case is either **positive** (an in-doc question with a
`gold_chunk_ids` set the correct answer must cite) or **refusal**
(an out-of-doc question; the playbook must refuse). The runner
adapts a `QAPlaybook` into a `CaseRunner` and emits one metric per
case type:

  - positive: `citation_precision` — fraction of cited chunk ids
    that appear in the gold set (deduped).
  - refusal: `refusal_accuracy` — 1.0 if the playbook returned an
    empty answer or no verified claims, 0.0 otherwise.

Per §8.2 the thresholds are ≥0.95 (precision) and ≥0.90 (refusal
accuracy). A case passes iff its individual metric meets the
relevant threshold, so the aggregate `pass_rate` lines up with the
spec gate.

SPEC-REF: §8.1 (qa_eval), §8.2 (qa metrics)
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict

from ctrldoc.eval.harness import EvalResult
from ctrldoc.ops.qa import QAPlaybook

CITATION_PRECISION_THRESHOLD = 0.95
REFUSAL_ACCURACY_THRESHOLD = 0.90


class QAEvalCase(BaseModel):
    """One row in the qa_eval set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    tags: list[str] = []
    question: str
    gold_chunk_ids: list[str] = []
    should_refuse: bool = False


def citation_precision(cited_chunk_ids: Iterable[str], gold: set[str]) -> float:
    """Fraction of unique cited chunk ids that appear in `gold`.

    Returns 0.0 when no citations are present — a positive case that
    cites nothing has zero precision by convention here, since the
    gold set is non-empty.
    """
    unique = set(cited_chunk_ids)
    if not unique:
        return 0.0
    return len(unique & gold) / len(unique)


class QAEvalRunner:
    """Adapts `QAPlaybook` into a `CaseRunner` for the harness."""

    def __init__(self, *, playbook: QAPlaybook) -> None:
        self._playbook = playbook

    def run_case(self, case: QAEvalCase) -> EvalResult:
        report = self._playbook.run(case.question)
        verified_claims = [claim for claim in report.claims if claim.verified]
        refused = not verified_claims or not report.answer.strip()

        if case.should_refuse:
            accuracy = 1.0 if refused else 0.0
            return EvalResult(
                case_id=case.id,
                passed=accuracy >= REFUSAL_ACCURACY_THRESHOLD,
                score=accuracy,
                metrics={"refusal_accuracy": accuracy},
                notes=(
                    f"refused={refused}, verified_claims={len(verified_claims)}, "
                    f"answer_len={len(report.answer)}"
                ),
            )

        cited = [span.chunk_id for claim in verified_claims for span in claim.citations]
        precision = citation_precision(cited, set(case.gold_chunk_ids))
        return EvalResult(
            case_id=case.id,
            passed=precision >= CITATION_PRECISION_THRESHOLD,
            score=precision,
            metrics={"citation_precision": precision},
            notes=f"cited={sorted(set(cited))}, gold={sorted(case.gold_chunk_ids)}",
        )


__all__ = [
    "CITATION_PRECISION_THRESHOLD",
    "REFUSAL_ACCURACY_THRESHOLD",
    "QAEvalCase",
    "QAEvalRunner",
    "citation_precision",
]
