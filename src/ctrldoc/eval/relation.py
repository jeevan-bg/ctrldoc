"""relation_eval — relation-type accuracy on gold concept pairs.

Each case carries an explicit concept list, per-pair evidence, and
a gold relation type per labelled pair (or `None` to mean "unrelated
in the doc"). The runner stitches a stub extractor + case-local
pair-retriever around the caller-supplied classifier, runs
`RelationMapPlaybook`, and scores accuracy over the gold pairs. Per
§8.2 the threshold is `≥0.80`.

A gold pair is "correct" when:

  - `gold_type is None` ⇒ the playbook emitted no edge for that pair
    (the classifier returned `None` or the evidence was empty).
  - `gold_type` is a relation literal ⇒ the playbook emitted an edge
    whose `type` matches.

Pair keys are canonicalised by sorting the two concept ids, so
upper-triangle iteration in the playbook never causes a key-order
mismatch with the gold table.

SPEC-REF: §8.1 (relation_eval), §8.2 (relation_map metrics)
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, model_validator

from ctrldoc.eval.harness import EvalResult
from ctrldoc.models import EvidencePack, RelationTypeLiteral, Span
from ctrldoc.playbooks.relations import (
    Concept,
    RelationClassifier,
    RelationMapPlaybook,
)

RELATION_TYPE_ACCURACY_THRESHOLD = 0.80


class EvidenceSpan(BaseModel):
    """One labelled span the eval supplies as retrieval material."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    text: str


class GoldPair(BaseModel):
    """Expert label for one concept pair.

    `gold_type=None` means the pair is intentionally unrelated in the
    doc — the playbook should emit no edge.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    src_concept_id: str
    dst_concept_id: str
    gold_type: RelationTypeLiteral | None = None


def pair_key(a_id: str, b_id: str) -> str:
    """Canonical sorted pair key — direction-agnostic."""
    lo, hi = sorted([a_id, b_id])
    return f"{lo}|{hi}"


class RelationEvalCase(BaseModel):
    """One row in relation_eval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    tags: list[str] = []
    concepts: list[Concept]
    evidence_by_pair: dict[str, list[EvidenceSpan]] = {}
    gold_pairs: list[GoldPair]

    @model_validator(mode="after")
    def _gold_pairs_reference_known_concepts(self) -> RelationEvalCase:
        ids = {c.id for c in self.concepts}
        for gold in self.gold_pairs:
            if gold.src_concept_id not in ids:
                raise ValueError(f"gold pair src {gold.src_concept_id!r} not in concept list")
            if gold.dst_concept_id not in ids:
                raise ValueError(f"gold pair dst {gold.dst_concept_id!r} not in concept list")
            if gold.src_concept_id == gold.dst_concept_id:
                raise ValueError(f"gold pair has identical src and dst: {gold.src_concept_id!r}")
        # Reject duplicate pair keys (canonical form).
        seen: set[str] = set()
        for gold in self.gold_pairs:
            key = pair_key(gold.src_concept_id, gold.dst_concept_id)
            if key in seen:
                raise ValueError(f"duplicate gold pair: {key!r}")
            seen.add(key)
        return self


def relation_type_accuracy(
    emitted_edges: list[tuple[str, str, RelationTypeLiteral]],
    gold_pairs: list[GoldPair],
) -> float:
    """Fraction of gold pairs whose emitted edge matches the gold label.

    Emitted edges arrive as `(src_id, dst_id, type)` tuples; pair keys
    are canonicalised so direction doesn't matter.
    """
    if not gold_pairs:
        return 0.0
    edges_by_key: dict[str, RelationTypeLiteral] = {
        pair_key(src, dst): edge_type for src, dst, edge_type in emitted_edges
    }
    correct = 0
    for gold in gold_pairs:
        key = pair_key(gold.src_concept_id, gold.dst_concept_id)
        emitted = edges_by_key.get(key)
        if gold.gold_type is None:
            if emitted is None:
                correct += 1
        else:
            if emitted == gold.gold_type:
                correct += 1
    return correct / len(gold_pairs)


@dataclass
class _StubExtractor:
    concepts: list[Concept]

    def extract(self) -> list[Concept]:
        return list(self.concepts)


@dataclass
class _CaseLocalRetriever:
    spans_by_key: dict[str, list[EvidenceSpan]]

    def retrieve(self, c_i: Concept, c_j: Concept) -> EvidencePack:
        key = pair_key(c_i.id, c_j.id)
        spans = self.spans_by_key.get(key, [])
        return EvidencePack(
            query=key,
            spans=[
                Span(chunk_id=s.chunk_id, char_start=0, char_end=len(s.text), text=s.text)
                for s in spans
            ],
            token_count=0,
            retrieval_plan=[],
        )


class RelationEvalRunner:
    """Adapt a `RelationClassifier` into a `CaseRunner`."""

    def __init__(self, *, classifier: RelationClassifier) -> None:
        self._classifier = classifier

    def run_case(self, case: RelationEvalCase) -> EvalResult:
        playbook = RelationMapPlaybook(
            extractor=_StubExtractor(concepts=case.concepts),
            retriever=_CaseLocalRetriever(spans_by_key=case.evidence_by_pair),
            classifier=self._classifier,
        )
        graph = playbook.run()
        emitted = [(edge.src_concept, edge.dst_concept, edge.type) for edge in graph.edges]
        accuracy = relation_type_accuracy(emitted, case.gold_pairs)
        return EvalResult(
            case_id=case.id,
            passed=accuracy >= RELATION_TYPE_ACCURACY_THRESHOLD,
            score=accuracy,
            metrics={"relation_type_accuracy": accuracy},
            notes=(
                f"gold_pairs={len(case.gold_pairs)}, emitted_edges={len(graph.edges)}, "
                f"accuracy={accuracy:.3f}"
            ),
        )


__all__ = [
    "RELATION_TYPE_ACCURACY_THRESHOLD",
    "EvidenceSpan",
    "GoldPair",
    "RelationEvalCase",
    "RelationEvalRunner",
    "pair_key",
    "relation_type_accuracy",
]
