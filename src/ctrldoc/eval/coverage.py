"""coverage_eval — score `CoverageAuditPlaybook` runs against gold verdicts.

Each case carries its checklist items, the per-topic evidence spans
the playbook should retrieve, and a per-item `gold_verdict` from the
expert. The runner constructs a case-local retriever, drives the
playbook, and emits a `verdict_accuracy` metric in `[0, 1]`. Per
§8.2 the threshold is ≥0.90.

SPEC-REF: §8.1 (coverage_eval), §8.2 (coverage_audit metrics)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from ctrldoc.assembler import CacheablePrefix
from ctrldoc.eval.harness import EvalResult
from ctrldoc.models import EvidencePack, Span
from ctrldoc.ops.audit import (
    ChecklistItem,
    CoverageAuditPlaybook,
)
from ctrldoc.orch.batch import BatchedTaskRunner

VERDICT_ACCURACY_THRESHOLD = 0.90

VerdictLabel = Literal["Covered", "Partial", "NotCovered", "Ambiguous"]


class EvidenceSpan(BaseModel):
    """One labelled span the eval supplies as retrieval material."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    text: str


class CoverageEvalCase(BaseModel):
    """One row in coverage_eval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    tags: list[str] = []
    items: list[ChecklistItem]
    evidence_by_topic: dict[str, list[EvidenceSpan]]
    gold_verdicts: dict[str, VerdictLabel]

    @model_validator(mode="after")
    def _gold_covers_items(self) -> CoverageEvalCase:
        item_ids = {item.id for item in self.items}
        gold_ids = set(self.gold_verdicts)
        missing = item_ids - gold_ids
        if missing:
            raise ValueError(f"gold_verdicts missing entries for items: {sorted(missing)}")
        extra = gold_ids - item_ids
        if extra:
            raise ValueError(f"gold_verdicts has unknown item ids: {sorted(extra)}")
        return self


@dataclass
class _CaseLocalRetriever:
    """Returns the configured spans for the given topic key."""

    spans_by_topic: dict[str, list[EvidenceSpan]]

    def retrieve(self, topic_key: str) -> EvidencePack:
        spans = self.spans_by_topic.get(topic_key, [])
        return EvidencePack(
            query=topic_key,
            spans=[
                Span(chunk_id=s.chunk_id, char_start=0, char_end=len(s.text), text=s.text)
                for s in spans
            ],
            token_count=0,
            retrieval_plan=[],
        )


class CoverageEvalRunner:
    """Adapts a configurable `CoverageAuditPlaybook` to the harness."""

    def __init__(
        self,
        *,
        prefix: CacheablePrefix,
        batched_runner: BatchedTaskRunner,
    ) -> None:
        self._prefix = prefix
        self._batched_runner = batched_runner

    def run_case(self, case: CoverageEvalCase) -> EvalResult:
        retriever = _CaseLocalRetriever(spans_by_topic=case.evidence_by_topic)
        playbook = CoverageAuditPlaybook(
            prefix=self._prefix,
            retriever=retriever,
            batched_runner=self._batched_runner,
        )
        report = playbook.run(case.items)
        verdict_by_item = {verdict.item_id: verdict.verdict for verdict in report.verdicts}

        correct = 0
        for item_id, gold in case.gold_verdicts.items():
            if verdict_by_item.get(item_id) == gold:
                correct += 1
        accuracy = correct / len(case.gold_verdicts) if case.gold_verdicts else 0.0

        return EvalResult(
            case_id=case.id,
            passed=accuracy >= VERDICT_ACCURACY_THRESHOLD,
            score=accuracy,
            metrics={"verdict_accuracy": accuracy},
            notes=(
                f"correct={correct}/{len(case.gold_verdicts)}; emitted={sorted(verdict_by_item)}"
            ),
        )


__all__ = [
    "VERDICT_ACCURACY_THRESHOLD",
    "CoverageEvalCase",
    "CoverageEvalRunner",
    "EvidenceSpan",
    "VerdictLabel",
]
