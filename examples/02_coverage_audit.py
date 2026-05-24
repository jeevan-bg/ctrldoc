"""UC2 coverage audit — checklist vs. doc with batched judging.

The checklist items are clustered by `topic_key`; the playbook
retrieves once per cluster and runs a batched LLM call that judges
every item in the cluster against the shared evidence pack.

Run:

    python examples/02_coverage_audit.py

SPEC-REF: §5.2
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ctrldoc.assembler import CacheablePrefix
from ctrldoc.models import EvidencePack, Span
from ctrldoc.ops.audit import ChecklistItem, CoverageAuditPlaybook
from ctrldoc.orch.batch import BatchedTaskRunner


@dataclass
class _ClusterRetriever:
    spans_by_topic: dict[str, list[tuple[str, str]]]

    def retrieve(self, topic_key: str) -> EvidencePack:
        spans = self.spans_by_topic.get(topic_key, [])
        return EvidencePack(
            query=topic_key,
            spans=[
                Span(chunk_id=cid, char_start=0, char_end=len(text), text=text)
                for cid, text in spans
            ],
            token_count=0,
            retrieval_plan=[],
        )


@dataclass
class _ScriptedClient:
    """Returns scripted batched-judging responses in submission order."""

    responses: list[str]
    calls: list[tuple[str, str]] = field(default_factory=list)

    def call(self, *, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.responses.pop(0)


def _batched_response(rows: list[tuple[str, str, float, list[str]]]) -> str:
    return json.dumps(
        [
            {
                "id": item_id,
                "output": {
                    "verdict": verdict,
                    "confidence": confidence,
                    "citation_chunk_ids": citations,
                },
            }
            for item_id, verdict, confidence, citations in rows
        ]
    )


def main() -> None:
    items = [
        ChecklistItem(
            id="r-hashing", text="Explains consistent-hashing partitioning.", topic_key="hashing"
        ),
        ChecklistItem(
            id="r-failover", text="Describes node-failure handling.", topic_key="failover"
        ),
        ChecklistItem(id="r-license", text="States the licence.", topic_key="license"),
    ]
    retriever = _ClusterRetriever(
        spans_by_topic={
            "hashing": [("c-h", "Aurora partitions the key space using consistent hashing.")],
            "failover": [
                ("c-f", "GossipBus detects node failures; ShardRing rebalances partitions.")
            ],
            "license": [],
        },
    )
    # One batched call per topic cluster.
    client = _ScriptedClient(
        responses=[
            _batched_response([("r-hashing", "Covered", 0.9, ["c-h"])]),
            _batched_response([("r-failover", "Covered", 0.85, ["c-f"])]),
            _batched_response([("r-license", "NotCovered", 0.95, [])]),
        ]
    )
    playbook = CoverageAuditPlaybook(
        prefix=CacheablePrefix(
            system_prompt="coverage judge", doc_skeleton="# §1", entity_glossary=""
        ),
        retriever=retriever,
        batched_runner=BatchedTaskRunner(client=client),
    )

    report = playbook.run(items)
    print(
        json.dumps(
            {
                "verdicts": [
                    {
                        "item_id": v.item_id,
                        "verdict": v.verdict,
                        "confidence": v.confidence,
                        "citations": [s.chunk_id for s in v.citations],
                    }
                    for v in report.verdicts
                ],
                "batched_calls": len(client.calls),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
