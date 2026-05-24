"""UC6 `relation_map` playbook — pair-wise concept relation extraction.

Per §5.6 the playbook walks the upper triangle of concept pairs,
asks the retriever for co-occurrence evidence (skipping pairs with
no evidence), and lets the classifier emit a `RelationEdge` for each
related pair. Pairs the classifier marks as `unrelated` are dropped
from the graph rather than recorded with a sentinel type. The
returned `RelationGraph` carries both the concept nodes (verbatim
from the extractor) and the discovered edges in deterministic order.

SPEC-REF: §5.6 (UC6 relation_map)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from pydantic import ValidationError

from ctrldoc.models import EvidencePack, Span
from ctrldoc.ops.map import (
    Concept,
    ConceptExtractor,
    CoOccurrenceRetriever,
    RelationClassification,
    RelationClassifier,
    RelationGraph,
    RelationMapPlaybook,
)

# --- fixtures ---


def _pack_with_spans(spans: list[tuple[str, str]]) -> EvidencePack:
    return EvidencePack(
        query="pair",
        spans=[
            Span(chunk_id=cid, char_start=0, char_end=len(text), text=text) for cid, text in spans
        ],
        token_count=10,
        retrieval_plan=["search('pair', view=dense)"],
    )


def _empty_pack() -> EvidencePack:
    return EvidencePack(query="pair", spans=[], token_count=0, retrieval_plan=[])


# --- stubs ---


@dataclass
class _StubExtractor:
    concepts: list[Concept]
    calls: int = 0

    def extract(self) -> list[Concept]:
        self.calls += 1
        return list(self.concepts)


@dataclass
class _StubRetriever:
    """Returns evidence keyed by frozenset({c_i.id, c_j.id})."""

    packs_by_pair: dict[frozenset[str], EvidencePack]
    calls: list[tuple[str, str]] = field(default_factory=list)

    def retrieve(self, c_i: Concept, c_j: Concept) -> EvidencePack:
        self.calls.append((c_i.id, c_j.id))
        return self.packs_by_pair.get(frozenset({c_i.id, c_j.id}), _empty_pack())


@dataclass
class _StubClassifier:
    """Returns classifications keyed by frozenset of concept ids."""

    classifications: dict[frozenset[str], RelationClassification | None]
    calls: list[tuple[str, str]] = field(default_factory=list)

    def classify(
        self,
        c_i: Concept,
        c_j: Concept,
        evidence: EvidencePack,
    ) -> RelationClassification | None:
        self.calls.append((c_i.id, c_j.id))
        return self.classifications.get(frozenset({c_i.id, c_j.id}))


def _span(chunk_id: str, text: str) -> Span:
    return Span(chunk_id=chunk_id, char_start=0, char_end=len(text), text=text)


# --- happy path ---


def test_emits_edges_for_pairs_with_evidence_and_a_relation_type() -> None:
    a = Concept(id="aurora", name="Aurora")
    b = Concept(id="hashing", name="consistent hashing")
    c = Concept(id="failover", name="failover")

    extractor = _StubExtractor(concepts=[a, b, c])
    retriever = _StubRetriever(
        packs_by_pair={
            frozenset({"aurora", "hashing"}): _pack_with_spans([("c1", "A→H evidence")]),
            frozenset({"aurora", "failover"}): _pack_with_spans([("c2", "A→F evidence")]),
            # The hashing/failover pair has no co-occurrence in the doc.
            frozenset({"hashing", "failover"}): _empty_pack(),
        }
    )
    classifier = _StubClassifier(
        classifications={
            frozenset({"aurora", "hashing"}): RelationClassification(
                type="depends_on",
                citations=[_span("c1", "A→H evidence")],
                confidence=0.9,
            ),
            frozenset({"aurora", "failover"}): RelationClassification(
                type="refines",
                citations=[_span("c2", "A→F evidence")],
                confidence=0.7,
            ),
        }
    )

    playbook = RelationMapPlaybook(
        extractor=extractor,
        retriever=retriever,
        classifier=classifier,
    )
    graph = playbook.run()

    assert graph.nodes == [a, b, c]
    assert len(graph.edges) == 2
    types = {(edge.src_concept, edge.dst_concept, edge.type) for edge in graph.edges}
    assert types == {
        ("aurora", "hashing", "depends_on"),
        ("aurora", "failover", "refines"),
    }
    # Upper-triangle iteration: (a,b), (a,c), (b,c).
    assert retriever.calls == [
        ("aurora", "hashing"),
        ("aurora", "failover"),
        ("hashing", "failover"),
    ]
    # Classifier never invoked on the empty-evidence pair.
    assert ("hashing", "failover") not in classifier.calls


# --- evidence gating ---


def test_pairs_with_empty_evidence_are_skipped_before_classification() -> None:
    a = Concept(id="a", name="A")
    b = Concept(id="b", name="B")
    extractor = _StubExtractor(concepts=[a, b])
    retriever = _StubRetriever(packs_by_pair={frozenset({"a", "b"}): _empty_pack()})
    classifier = _StubClassifier(classifications={})
    playbook = RelationMapPlaybook(extractor=extractor, retriever=retriever, classifier=classifier)

    graph = playbook.run()

    assert graph.edges == []
    assert retriever.calls == [("a", "b")]
    # Empty evidence ⇒ classifier was not invoked.
    assert classifier.calls == []


def test_classifier_returning_none_means_unrelated_and_skips_edge() -> None:
    a = Concept(id="a", name="A")
    b = Concept(id="b", name="B")
    extractor = _StubExtractor(concepts=[a, b])
    retriever = _StubRetriever(
        packs_by_pair={frozenset({"a", "b"}): _pack_with_spans([("c1", "ev")])},
    )
    classifier = _StubClassifier(classifications={frozenset({"a", "b"}): None})
    playbook = RelationMapPlaybook(extractor=extractor, retriever=retriever, classifier=classifier)

    graph = playbook.run()

    assert graph.edges == []
    # Both retriever and classifier were exercised; the playbook just
    # dropped the unrelated edge.
    assert classifier.calls == [("a", "b")]


# --- iteration shape ---


def test_upper_triangle_iteration_no_self_pairs_no_duplicates() -> None:
    """For M concepts the playbook should ask about exactly M*(M-1)/2 pairs."""
    concepts = [Concept(id=f"c-{i}", name=f"C{i}") for i in range(5)]
    extractor = _StubExtractor(concepts=concepts)
    retriever = _StubRetriever(packs_by_pair={})  # all empty
    classifier = _StubClassifier(classifications={})

    playbook = RelationMapPlaybook(extractor=extractor, retriever=retriever, classifier=classifier)
    playbook.run()

    pairs = retriever.calls
    # M * (M-1) / 2 = 10 pairs for M=5.
    assert len(pairs) == 10
    # All pairs are unique and ordered with i < j.
    assert len(pairs) == len(set(pairs))
    for a, b in pairs:
        assert a < b


# --- edge cases ---


def test_empty_concept_list_returns_empty_graph_without_calls() -> None:
    extractor = _StubExtractor(concepts=[])
    retriever = _StubRetriever(packs_by_pair={})
    classifier = _StubClassifier(classifications={})
    playbook = RelationMapPlaybook(extractor=extractor, retriever=retriever, classifier=classifier)

    graph = playbook.run()

    assert graph.nodes == []
    assert graph.edges == []
    assert extractor.calls == 1  # extract was called
    assert retriever.calls == []
    assert classifier.calls == []


def test_single_concept_emits_no_pairs() -> None:
    a = Concept(id="solo", name="Solo")
    extractor = _StubExtractor(concepts=[a])
    retriever = _StubRetriever(packs_by_pair={})
    classifier = _StubClassifier(classifications={})
    playbook = RelationMapPlaybook(extractor=extractor, retriever=retriever, classifier=classifier)

    graph = playbook.run()

    assert graph.nodes == [a]
    assert graph.edges == []
    assert retriever.calls == []


def test_concept_extractor_yielding_duplicate_ids_rejected() -> None:
    """A duplicate concept id would create ambiguous edges. Fail loud."""
    a = Concept(id="dup", name="A")
    b = Concept(id="dup", name="B")
    extractor = _StubExtractor(concepts=[a, b])
    retriever = _StubRetriever(packs_by_pair={})
    classifier = _StubClassifier(classifications={})
    playbook = RelationMapPlaybook(extractor=extractor, retriever=retriever, classifier=classifier)

    with pytest.raises(ValueError, match="duplicate"):
        playbook.run()


# --- model invariants ---


def test_concept_is_frozen() -> None:
    c = Concept(id="a", name="A")
    with pytest.raises(ValidationError):
        c.name = "B"  # type: ignore[misc]


def test_relation_graph_is_frozen() -> None:
    g = RelationGraph(nodes=[], edges=[])
    with pytest.raises(ValidationError):
        g.nodes = []  # type: ignore[misc]


def test_relation_classification_rejects_invalid_type_literal() -> None:
    with pytest.raises(ValidationError):
        RelationClassification(
            type="not_a_real_relation_type",  # type: ignore[arg-type]
            citations=[],
            confidence=0.5,
        )


# --- protocols ---


def test_concept_extractor_protocol_isinstance() -> None:
    assert isinstance(_StubExtractor(concepts=[]), ConceptExtractor)


def test_cooccurrence_retriever_protocol_isinstance() -> None:
    assert isinstance(_StubRetriever(packs_by_pair={}), CoOccurrenceRetriever)


def test_relation_classifier_protocol_isinstance() -> None:
    assert isinstance(_StubClassifier(classifications={}), RelationClassifier)


# --- edge confidence and citations carry through ---


def test_emitted_edge_carries_classifier_citations_and_confidence_verbatim() -> None:
    a = Concept(id="a", name="A")
    b = Concept(id="b", name="B")
    citations = [_span("c1", "ev1"), _span("c2", "ev2")]
    extractor = _StubExtractor(concepts=[a, b])
    retriever = _StubRetriever(
        packs_by_pair={frozenset({"a", "b"}): _pack_with_spans([("c1", "ev1")])}
    )
    classifier = _StubClassifier(
        classifications={
            frozenset({"a", "b"}): RelationClassification(
                type="contradicts",
                citations=citations,
                confidence=0.42,
            )
        }
    )
    playbook = RelationMapPlaybook(extractor=extractor, retriever=retriever, classifier=classifier)
    graph = playbook.run()
    assert len(graph.edges) == 1
    edge = graph.edges[0]
    assert edge.type == "contradicts"
    assert edge.citations == citations
    assert edge.confidence == pytest.approx(0.42)
