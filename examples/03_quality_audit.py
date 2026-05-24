"""UC3 quality audit — generate criteria, delegate to coverage.

`HeuristicCriteriaGenerator` emits the canonical four-axis quality
checklist (clarity / completeness / safety / examples) scoped under
the slugged doc_type. The playbook then runs the same coverage
machinery as UC2 against the generated criteria.

Run:

    python examples/03_quality_audit.py

SPEC-REF: §5.3
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ctrldoc.assembler import CacheablePrefix
from ctrldoc.models import EvidencePack, Span
from ctrldoc.ops.audit import CoverageAuditPlaybook
from ctrldoc.ops.quality import (
    HeuristicCriteriaGenerator,
    QualityAuditPlaybook,
)
from ctrldoc.orch.batch import BatchedTaskRunner


@dataclass
class _UniformRetriever:
    def retrieve(self, topic_key: str) -> EvidencePack:
        return EvidencePack(
            query=topic_key,
            spans=[Span(chunk_id=f"c-{topic_key}", char_start=0, char_end=8, text="evidence")],
            token_count=0,
            retrieval_plan=[],
        )


@dataclass
class _CoveredJudgeClient:
    """Marks every item Covered with a single citation per cluster."""

    calls: list[tuple[str, str]] = field(default_factory=list)

    def call(self, *, system: str, user: str) -> str:
        self.calls.append((system, user))
        # The batched-task runner sends a JSON `items` array; parse the
        # ids out of the user message tail.
        payload = user[user.index("<items>") + 7 : user.index("</items>")].strip()
        items = json.loads(payload)
        return json.dumps(
            [
                {
                    "id": item["id"],
                    "output": {
                        "verdict": "Covered",
                        "confidence": 0.9,
                        "citation_chunk_ids": [],
                    },
                }
                for item in items
            ]
        )


def main() -> None:
    client = _CoveredJudgeClient()
    coverage = CoverageAuditPlaybook(
        prefix=CacheablePrefix(
            system_prompt="quality judge", doc_skeleton="# §1", entity_glossary=""
        ),
        retriever=_UniformRetriever(),
        batched_runner=BatchedTaskRunner(client=client),
    )
    playbook = QualityAuditPlaybook(
        criteria_generator=HeuristicCriteriaGenerator(),
        coverage_audit=coverage,
    )

    report = playbook.run("L0 kernel spec")
    print(
        json.dumps(
            {
                "doc_type": report.doc_type,
                "criteria": [
                    {"id": item.id, "text": item.text, "topic_key": item.topic_key}
                    for item in report.criteria
                ],
                "verdicts": [
                    {"item_id": v.item_id, "verdict": v.verdict, "confidence": v.confidence}
                    for v in report.coverage.verdicts
                ],
                "batched_calls": len(client.calls),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
