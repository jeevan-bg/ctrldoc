"""Tier-1 deterministic claim-graph extractor — heuristic edge induction.

The Tier-1 extractor is the §6.4 floor of the schema co-induction loop:
four pattern families (Hearst lexico-syntactic, heading-tree containment,
sliding-window PMI co-occurrence, lexical-identity coref) produce typed
edges between concept clusters with no LLM in the path. Edge confidence
is the fixed heuristic prior from §6.5 (~0.9). All edges cite their
producing span(s); the extractor is fully deterministic given identical
input.

SPEC-REF: §6.4 (schema co-induction floor), §6.5 (heuristic edge prior)
"""

from __future__ import annotations

import pytest

from ctrldoc.extract.tier1 import (
    HEURISTIC_CONFIDENCE,
    Tier1Config,
    Tier1Extraction,
    extract_tier1,
)
from ctrldoc.models import Chunk, Section


def _chunk(chunk_id: str, section_id: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        section_id=section_id,
        text=text,
        token_count=len(text.split()),
        char_start=0,
        char_end=len(text),
        embedding_id=f"emb-{chunk_id}",
    )


def _section(sid: str, parent: str | None, title: str, chunk_ids: list[str]) -> Section:
    return Section(
        id=sid,
        parent_id=parent,
        title=title,
        summary="",
        chunk_ids=chunk_ids,
    )


@pytest.mark.family_determinism
def test_extract_returns_tier1_extraction_dataclass() -> None:
    """A minimal call returns the public extraction shape with empty lists."""
    out = extract_tier1(sections=[], chunks=[])
    assert isinstance(out, Tier1Extraction)
    assert out.mentions == []
    assert out.concepts == []
    assert out.edges == []


@pytest.mark.family_determinism
def test_hearst_such_as_emits_example_of_edge() -> None:
    """`X such as Y` is the canonical Hearst pattern for `example_of(Y, X)`."""
    chunk = _chunk(
        "c0",
        "s0",
        "Mammals such as dogs are warm-blooded.",
    )
    sec = _section("s0", None, "Animals", ["c0"])

    out = extract_tier1(sections=[sec], chunks=[chunk])

    types = {(e.type, e.pattern) for e in out.edges}
    assert ("example_of", "hearst_such_as") in types

    edge = next(e for e in out.edges if e.pattern == "hearst_such_as")
    assert edge.confidence == HEURISTIC_CONFIDENCE
    assert edge.source == "heuristic"
    assert edge.citations  # carry their producing span
    src = next(c for c in out.concepts if c.id == edge.src_id)
    dst = next(c for c in out.concepts if c.id == edge.dst_id)
    assert src.canonical_name == "dogs"
    assert dst.canonical_name == "mammals"


@pytest.mark.family_determinism
def test_hearst_including_emits_example_of_edge() -> None:
    """`X including Y` is a Hearst variant — emits `example_of(Y, X)`."""
    chunk = _chunk("c0", "s0", "Several languages including Python are dynamic.")
    sec = _section("s0", None, "Languages", ["c0"])

    out = extract_tier1(sections=[sec], chunks=[chunk])
    patterns = {e.pattern for e in out.edges}
    assert "hearst_including" in patterns


@pytest.mark.family_determinism
def test_hearst_is_a_emits_is_a_edge() -> None:
    """`X is a Y` is a Hearst variant — emits `is_a(X, Y)`."""
    chunk = _chunk("c0", "s0", "Python is a programming language.")
    sec = _section("s0", None, "Python", ["c0"])

    out = extract_tier1(sections=[sec], chunks=[chunk])
    is_a_edges = [e for e in out.edges if e.type == "is_a" and e.pattern == "hearst_is_a"]
    assert is_a_edges, f"expected is_a edge, got {[e.pattern for e in out.edges]}"
    edge = is_a_edges[0]
    src = next(c for c in out.concepts if c.id == edge.src_id)
    dst = next(c for c in out.concepts if c.id == edge.dst_id)
    assert src.canonical_name == "python"
    assert dst.canonical_name == "programming language"


@pytest.mark.family_determinism
def test_heading_tree_emits_part_of_edges_for_child_sections() -> None:
    """A child section is `part_of` its parent — pure structural heuristic."""
    sec_root = _section("s_root", None, "Specification", [])
    sec_child = _section("s_child", "s_root", "Authentication", [])
    sec_grand = _section("s_grand", "s_child", "Token Lifecycle", [])

    out = extract_tier1(sections=[sec_root, sec_child, sec_grand], chunks=[])

    part_of_patterns = {(e.type, e.pattern) for e in out.edges if e.pattern == "heading_tree"}
    assert ("part_of", "heading_tree") in part_of_patterns
    # Exactly two parent-edges: child→root and grand→child.
    heading_edges = [e for e in out.edges if e.pattern == "heading_tree"]
    assert len(heading_edges) == 2
    canonical_pairs = {
        (
            next(c.canonical_name for c in out.concepts if c.id == e.src_id),
            next(c.canonical_name for c in out.concepts if c.id == e.dst_id),
        )
        for e in heading_edges
    }
    assert ("authentication", "specification") in canonical_pairs
    assert ("token lifecycle", "authentication") in canonical_pairs


@pytest.mark.family_determinism
def test_pmi_emits_related_to_edge_for_repeated_cooccurrence() -> None:
    """Tokens co-occurring in a sliding window above PMI threshold → `related_to`."""
    # `kafka` and `partition` co-occur in three sentences; `unrelated` never does.
    body = (
        "kafka partitions are immutable. "
        "kafka partition leaders elect via zookeeper. "
        "a kafka partition is the unit of parallelism. "
        "unrelated topics live elsewhere."
    )
    chunk = _chunk("c0", "s0", body)
    sec = _section("s0", None, "Streams", ["c0"])

    out = extract_tier1(
        sections=[sec],
        chunks=[chunk],
        config=Tier1Config(pmi_window_tokens=8, pmi_min_count=2, pmi_threshold=0.5),
    )
    related = [e for e in out.edges if e.pattern == "pmi_window"]
    assert related, "expected at least one PMI related_to edge"
    pairs = {
        tuple(
            sorted(
                (
                    next(c.canonical_name for c in out.concepts if c.id == e.src_id),
                    next(c.canonical_name for c in out.concepts if c.id == e.dst_id),
                )
            )
        )
        for e in related
    }
    assert ("kafka", "partition") in pairs or ("kafka", "partitions") in pairs


@pytest.mark.family_determinism
def test_coref_identity_emits_equivalent_to_edge_for_repeated_surface_form() -> None:
    """Two mentions sharing a normalized surface form → identity edge.

    Lexical coref is the deterministic-floor stand-in: in the same
    document, two occurrences of `Kafka` are mention-equivalent. They
    cluster into one Concept and an `equivalent_to` edge is emitted
    pointing the concept at itself so the audit trail records the
    identity collapse explicitly.
    """
    chunk0 = _chunk("c0", "s0", "Kafka stores partitions on disk.")
    chunk1 = _chunk("c1", "s0", "Kafka guarantees ordered delivery.")
    sec = _section("s0", None, "Kafka", ["c0", "c1"])

    out = extract_tier1(sections=[sec], chunks=[chunk0, chunk1])
    coref_edges = [e for e in out.edges if e.pattern == "coref_identity"]
    assert coref_edges, "expected a coref identity edge"
    edge = coref_edges[0]
    assert edge.type == "equivalent_to"
    # Both citations belong to the two Kafka mentions; same concept on both sides.
    assert edge.src_id == edge.dst_id
    concept = next(c for c in out.concepts if c.id == edge.src_id)
    assert concept.canonical_name == "kafka"
    assert len(concept.mention_ids) >= 2


@pytest.mark.family_determinism
def test_run_is_deterministic_across_invocations() -> None:
    """Repeated extraction on the same input produces byte-identical output."""
    chunk = _chunk(
        "c0",
        "s0",
        "Cats are mammals. Mammals such as cats and dogs are warm-blooded.",
    )
    sec = _section("s0", None, "Animals", ["c0"])

    out1 = extract_tier1(sections=[sec], chunks=[chunk])
    out2 = extract_tier1(sections=[sec], chunks=[chunk])
    assert out1.model_dump() == out2.model_dump()


@pytest.mark.family_determinism
def test_edges_carry_unit_interval_confidence_and_heuristic_source() -> None:
    """Every emitted edge respects §6.5: source=heuristic, confidence in (0, 1]."""
    chunk = _chunk("c0", "s0", "Birds such as sparrows fly.")
    sec_root = _section("s_root", None, "Animals", [])
    sec_birds = _section("s0", "s_root", "Birds", ["c0"])

    out = extract_tier1(sections=[sec_root, sec_birds], chunks=[chunk])
    assert out.edges, "expected at least one edge across the four heuristics"
    for edge in out.edges:
        assert edge.source == "heuristic"
        assert 0.0 < edge.confidence <= 1.0
        assert edge.citations or edge.pattern == "heading_tree"


@pytest.mark.family_determinism
def test_concept_ids_are_content_hashed_and_stable() -> None:
    """A concept's id is content-derived; same canonical_name → same id."""
    chunk_a = _chunk("c0", "s0", "Birds such as sparrows fly.")
    chunk_b = _chunk("c1", "s1", "Birds such as eagles soar.")
    sec_a = _section("s0", None, "Birds A", ["c0"])
    sec_b = _section("s1", None, "Birds B", ["c1"])

    out = extract_tier1(sections=[sec_a, sec_b], chunks=[chunk_a, chunk_b])
    birds = [c for c in out.concepts if c.canonical_name == "birds"]
    # The same surface "birds" across two chunks collapses to one Concept.
    assert len(birds) == 1, [c.canonical_name for c in out.concepts]
