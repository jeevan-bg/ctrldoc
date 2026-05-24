"""UC6 — concept-relation map playbook.

Walks the upper triangle of concept pairs, asks the retriever for
co-occurrence evidence, and lets the classifier emit a
`RelationEdge` for each related pair. Pairs the retriever returns
zero spans for are skipped before classification — saving an LLM
call when the doc never co-mentions the two concepts. Pairs the
classifier marks as `unrelated` (returns `None`) are dropped from
the graph rather than recorded with a sentinel type.

The slice ships the composition primitive plus three Protocols
(extractor / retriever / classifier). Production-grade implementations
of each — entity-glossary-plus-LLM concept enrichment, BM25+entity
co-occurrence retrieval, constrained-JSON Opus classification — can
land as follow-ups behind these seams.

SPEC-REF: §5.6 (UC6 relation_map)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from ctrldoc.models import (
    EvidencePack,
    RelationEdge,
    RelationTypeLiteral,
    Span,
    UnitInterval,
)


class Concept(BaseModel):
    """One node in the relation graph."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    name: str


class RelationClassification(BaseModel):
    """Output of one pair-classification call (when the pair is related)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: RelationTypeLiteral
    citations: list[Span]
    confidence: UnitInterval


class RelationGraph(BaseModel):
    """Final UC6 output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    nodes: list[Concept]
    edges: list[RelationEdge]


@runtime_checkable
class ConceptExtractor(Protocol):
    """Yields the concept list to map (entity glossary + LLM enrichment)."""

    def extract(self) -> list[Concept]: ...


@runtime_checkable
class CoOccurrenceRetriever(Protocol):
    """Fetches evidence spans co-mentioning two concepts."""

    def retrieve(self, c_i: Concept, c_j: Concept) -> EvidencePack: ...


@runtime_checkable
class RelationClassifier(Protocol):
    """Classifies the relation between two concepts given evidence.

    Returns `None` to mean "unrelated" — the playbook drops the pair
    rather than recording an edge of synthetic `unrelated` type.
    """

    def classify(
        self,
        c_i: Concept,
        c_j: Concept,
        evidence: EvidencePack,
    ) -> RelationClassification | None: ...


class RelationMapPlaybook:
    """Compose concept extraction, pair retrieval, relation classification."""

    def __init__(
        self,
        *,
        extractor: ConceptExtractor,
        retriever: CoOccurrenceRetriever,
        classifier: RelationClassifier,
    ) -> None:
        self._extractor = extractor
        self._retriever = retriever
        self._classifier = classifier

    def run(self) -> RelationGraph:
        concepts = self._extractor.extract()
        seen: set[str] = set()
        for concept in concepts:
            if concept.id in seen:
                raise ValueError(f"duplicate concept id: {concept.id!r}")
            seen.add(concept.id)

        edges: list[RelationEdge] = []
        for i, c_i in enumerate(concepts):
            for c_j in concepts[i + 1 :]:
                evidence = self._retriever.retrieve(c_i, c_j)
                if not evidence.spans:
                    continue
                classification = self._classifier.classify(c_i, c_j, evidence)
                if classification is None:
                    continue
                edges.append(
                    RelationEdge(
                        src_concept=c_i.id,
                        dst_concept=c_j.id,
                        type=classification.type,
                        citations=classification.citations,
                        confidence=classification.confidence,
                    )
                )

        return RelationGraph(nodes=concepts, edges=edges)


__all__ = [
    "CoOccurrenceRetriever",
    "Concept",
    "ConceptExtractor",
    "RelationClassification",
    "RelationClassifier",
    "RelationGraph",
    "RelationMapPlaybook",
]
