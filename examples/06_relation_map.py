"""UC6 concept relation map — extract → retrieve pair evidence → classify.

`RelationMapPlaybook` iterates the upper triangle of concept pairs.
For each pair it asks the retriever for co-occurrence evidence; if
the evidence is non-empty it asks the classifier for a relation
type. Pairs with empty evidence or a `None` classification yield no
edge — the playbook records only what's actually grounded in the
doc.

Run:

    python examples/06_relation_map.py

SPEC-REF: §5.6
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ctrldoc.models import EvidencePack, Span
from ctrldoc.playbooks.relations import (
    Concept,
    RelationClassification,
    RelationMapPlaybook,
)


@dataclass
class _StubExtractor:
    concepts: list[Concept]

    def extract(self) -> list[Concept]:
        return list(self.concepts)


@dataclass
class _PairRetriever:
    spans_by_key: dict[str, list[tuple[str, str]]]

    def retrieve(self, c_i: Concept, c_j: Concept) -> EvidencePack:
        # Canonical pair key — sorted ids so direction doesn't matter.
        a, b = sorted([c_i.id, c_j.id])
        key = f"{a}|{b}"
        spans = self.spans_by_key.get(key, [])
        return EvidencePack(
            query=key,
            spans=[Span(chunk_id=cid, char_start=0, char_end=len(t), text=t) for cid, t in spans],
            token_count=0,
            retrieval_plan=[],
        )


@dataclass
class _ScriptedClassifier:
    types_by_key: dict[str, str | None]

    def classify(
        self,
        c_i: Concept,
        c_j: Concept,
        evidence: EvidencePack,
    ) -> RelationClassification | None:
        a, b = sorted([c_i.id, c_j.id])
        key = f"{a}|{b}"
        if key not in self.types_by_key or self.types_by_key[key] is None:
            return None
        return RelationClassification(
            type=self.types_by_key[key],  # type: ignore[arg-type]
            citations=[evidence.spans[0]] if evidence.spans else [],
            confidence=0.9,
        )


def main() -> None:
    concepts = [
        Concept(id="shard-ring", name="ShardRing"),
        Concept(id="consistent-hashing", name="consistent hashing"),
        Concept(id="virtual-nodes", name="virtual nodes"),
    ]
    spans_by_key = {
        "consistent-hashing|shard-ring": [
            ("c-1", "ShardRing is the consistent-hash ring."),
        ],
        "shard-ring|virtual-nodes": [
            ("c-2", "ShardRing adds virtual nodes to reduce skew."),
        ],
        "consistent-hashing|virtual-nodes": [
            ("c-3", "Virtual nodes refine the classical consistent-hashing approach."),
        ],
    }
    types_by_key = {
        "consistent-hashing|shard-ring": "depends_on",
        "shard-ring|virtual-nodes": "depends_on",
        "consistent-hashing|virtual-nodes": "refines",
    }

    graph = RelationMapPlaybook(
        extractor=_StubExtractor(concepts=concepts),
        retriever=_PairRetriever(spans_by_key=spans_by_key),
        classifier=_ScriptedClassifier(types_by_key=types_by_key),
    ).run()

    print(
        json.dumps(
            {
                "nodes": [{"id": n.id, "name": n.name} for n in graph.nodes],
                "edges": [
                    {
                        "src": e.src_concept,
                        "dst": e.dst_concept,
                        "type": e.type,
                        "confidence": e.confidence,
                    }
                    for e in graph.edges
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
