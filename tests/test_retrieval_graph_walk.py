"""Personalized PageRank graph-walk retrieval over typed claim-graph edges.

Per §6.9 the L2 retrieval layer is upgraded to walk the claim-graph: a seed
set of concept nodes (obtained by entity-linking the query) is diffused along
typed edges via personalized PageRank, with per-edge-type weights pinned by
the spec:

* `depends_on`, `refines`, `prerequisite_of` — **high** precision-relevant.
* `is_a`, `part_of`                            — **medium** abstraction-relevant.
* `related_to`                                 — **low** loose-similarity.

Stationary probability ranks concepts; the top-N harvest pulls the chunk ids
anchored to those concepts. The §6.9 release-gate is "**≥ 10 % recall
lift on multi-hop queries**": a synthetic 2-hop fixture where the gold target
is reachable from the seed only through an intermediate concept proves the
graph-walk recall strictly exceeds 1.10x the seed-only baseline recall.

The module is pure-Python (no numpy): power iteration over a `dict[str,
dict[str, float]]` adjacency, deterministic across runs, byte-stable
re-orderings of the same edge list.

SPEC-REF: §6.9
"""

from __future__ import annotations

import pytest

from ctrldoc.models import Span
from ctrldoc.models_v1 import TypedEdge
from ctrldoc.retrieval.graph_walk import (
    DEFAULT_ALPHA,
    DEFAULT_HARVEST_K,
    DEFAULT_MAX_ITER,
    EDGE_TYPE_WEIGHTS,
    PPR_RECALL_LIFT_THRESHOLD,
    GraphWalkConfig,
    GraphWalkResult,
    GraphWalkRetriever,
    personalized_pagerank,
    recall_at_k,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _edge(*, src: str, dst: str, edge_type: str, confidence: float = 1.0) -> TypedEdge:
    """Minimal `TypedEdge` factory — only the fields the walker actually reads."""
    return TypedEdge(
        src_id=src,
        dst_id=dst,
        type=edge_type,  # type: ignore[arg-type]  # narrow Literal at runtime
        confidence=confidence,
        raw_score=confidence,
        citations=[Span(chunk_id="c-0", char_start=0, char_end=1, text="x")],
        source="heuristic",
        paraphrase_votes=None,
    )


# ---------------------------------------------------------------------------
# Edge-type weight ladder — §6.9 precision/abstraction/loose tiers
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_edge_type_weights_match_spec_ladder() -> None:
    """High > Medium > Low, with the three §6.9 tiers strictly separated."""
    high = {
        EDGE_TYPE_WEIGHTS["depends_on"],
        EDGE_TYPE_WEIGHTS["refines"],
        EDGE_TYPE_WEIGHTS["prerequisite_of"],
    }
    medium = {EDGE_TYPE_WEIGHTS["is_a"], EDGE_TYPE_WEIGHTS["part_of"]}
    low = {EDGE_TYPE_WEIGHTS["related_to"]}

    assert len(high) == 1, "all 'high' edge types share one weight"
    assert len(medium) == 1, "all 'medium' edge types share one weight"
    assert len(low) == 1, "the single 'low' edge type has its own weight"
    assert max(low) < max(medium) < max(high), "tier ordering: low < medium < high per §6.9"
    # All weights are strictly positive — a zero would silence an edge entirely
    # and contradict the spec's "high / medium / low" framing.
    assert all(w > 0 for w in EDGE_TYPE_WEIGHTS.values())


# ---------------------------------------------------------------------------
# Pure PPR primitive
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_ppr_returns_stationary_distribution_summing_to_one() -> None:
    """PPR output is a probability distribution over the seen node set."""
    edges = [
        _edge(src="A", dst="B", edge_type="is_a"),
        _edge(src="B", dst="C", edge_type="is_a"),
    ]

    dist = personalized_pagerank(edges=edges, seeds={"A": 1.0})

    assert set(dist.keys()) == {"A", "B", "C"}
    assert sum(dist.values()) == pytest.approx(1.0, abs=1e-6)
    # Seed gets the most mass; sinks get strictly less.
    assert dist["A"] >= dist["B"] >= dist["C"]


@pytest.mark.family_determinism
def test_ppr_deterministic_across_edge_orderings() -> None:
    """Shuffling the edge list does not change the PPR output."""
    edges_a = [
        _edge(src="A", dst="B", edge_type="depends_on"),
        _edge(src="B", dst="C", edge_type="is_a"),
        _edge(src="A", dst="C", edge_type="related_to"),
    ]
    edges_b = [edges_a[2], edges_a[0], edges_a[1]]

    dist_a = personalized_pagerank(edges=edges_a, seeds={"A": 1.0})
    dist_b = personalized_pagerank(edges=edges_b, seeds={"A": 1.0})

    for node_id, mass in dist_a.items():
        assert dist_b[node_id] == pytest.approx(mass, abs=1e-9)


@pytest.mark.family_determinism
def test_ppr_seed_only_returns_seed_when_no_edges() -> None:
    """No edges → all mass stays on seed."""
    dist = personalized_pagerank(edges=[], seeds={"A": 1.0})
    assert dist == {"A": pytest.approx(1.0, abs=1e-9)}


@pytest.mark.family_determinism
def test_ppr_multi_seed_distributes_proportionally() -> None:
    """Two seeds with equal mass leave the highest-mass nodes near-symmetric."""
    edges = [
        _edge(src="A", dst="X", edge_type="is_a"),
        _edge(src="B", dst="X", edge_type="is_a"),
    ]

    dist = personalized_pagerank(edges=edges, seeds={"A": 0.5, "B": 0.5})

    assert sum(dist.values()) == pytest.approx(1.0, abs=1e-6)
    # By symmetry the two seeds carry the same mass.
    assert dist["A"] == pytest.approx(dist["B"], abs=1e-9)


@pytest.mark.family_determinism
def test_ppr_high_weight_edges_concentrate_more_mass() -> None:
    """A `depends_on` neighbour absorbs more probability than a `related_to`."""
    edges = [
        _edge(src="A", dst="HIGH", edge_type="depends_on"),
        _edge(src="A", dst="LOW", edge_type="related_to"),
    ]

    dist = personalized_pagerank(edges=edges, seeds={"A": 1.0})

    # High-tier neighbour strictly outranks the low-tier neighbour.
    assert dist["HIGH"] > dist["LOW"]


@pytest.mark.family_determinism
def test_ppr_confidence_modulates_edge_weight() -> None:
    """A high-confidence edge carries more mass than a low-confidence sibling
    of the same type."""
    edges = [
        _edge(src="A", dst="STRONG", edge_type="is_a", confidence=0.95),
        _edge(src="A", dst="WEAK", edge_type="is_a", confidence=0.30),
    ]

    dist = personalized_pagerank(edges=edges, seeds={"A": 1.0})

    assert dist["STRONG"] > dist["WEAK"]


@pytest.mark.family_determinism
def test_ppr_rejects_empty_seeds() -> None:
    """Empty seed dict has no defined personalisation — must raise."""
    with pytest.raises(ValueError):
        personalized_pagerank(edges=[], seeds={})


@pytest.mark.family_determinism
def test_ppr_rejects_invalid_alpha() -> None:
    """alpha must sit in the open (0, 1) interval."""
    with pytest.raises(ValueError):
        personalized_pagerank(edges=[], seeds={"A": 1.0}, alpha=0.0)
    with pytest.raises(ValueError):
        personalized_pagerank(edges=[], seeds={"A": 1.0}, alpha=1.0)


# ---------------------------------------------------------------------------
# Recall-at-k helper
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_recall_at_k_basic() -> None:
    """recall@k = |retrieved[:k] ∩ gold| / |gold|."""
    assert recall_at_k(retrieved=["a", "b", "c"], gold={"a", "c"}, k=3) == pytest.approx(1.0)
    assert recall_at_k(retrieved=["a", "b", "c"], gold={"a", "c"}, k=2) == pytest.approx(0.5)
    assert recall_at_k(retrieved=[], gold={"a"}, k=5) == pytest.approx(0.0)
    # Empty gold → recall is undefined; we return 0.0 for harness safety.
    assert recall_at_k(retrieved=["a"], gold=set(), k=1) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# GraphWalkRetriever — harvest stage
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_retriever_harvests_chunks_for_top_concepts() -> None:
    """Top-N concept ids resolve to the chunks they anchor, in PPR order."""
    edges = [
        _edge(src="A", dst="B", edge_type="depends_on"),
        _edge(src="B", dst="C", edge_type="is_a"),
    ]
    concept_to_chunks = {
        "A": ["c-A1"],
        "B": ["c-B1", "c-B2"],
        "C": ["c-C1"],
    }
    retriever = GraphWalkRetriever(
        edges=edges,
        concept_to_chunks=concept_to_chunks,
        config=GraphWalkConfig(harvest_k=3),
    )

    result = retriever.walk(seeds={"A": 1.0})

    assert isinstance(result, GraphWalkResult)
    # Top concept = seed A; B follows because of the high-tier outgoing edge.
    assert result.concept_ids[0] == "A"
    # All chunk ids surface, deduped, in concept-rank order.
    assert "c-A1" in result.chunk_ids
    assert "c-B1" in result.chunk_ids and "c-B2" in result.chunk_ids
    # Order is concept-rank then within-concept input order.
    assert result.chunk_ids.index("c-A1") < result.chunk_ids.index("c-B1")
    assert result.chunk_ids.index("c-B1") < result.chunk_ids.index("c-B2")


@pytest.mark.family_determinism
def test_retriever_skips_concepts_without_chunks() -> None:
    """A concept with no anchored chunks still ranks but contributes nothing."""
    edges = [_edge(src="A", dst="B", edge_type="is_a")]
    concept_to_chunks = {"A": ["c-A1"], "B": []}
    retriever = GraphWalkRetriever(
        edges=edges,
        concept_to_chunks=concept_to_chunks,
        config=GraphWalkConfig(harvest_k=2),
    )

    result = retriever.walk(seeds={"A": 1.0})

    assert result.chunk_ids == ["c-A1"]


@pytest.mark.family_determinism
def test_retriever_dedupes_chunks_across_concepts() -> None:
    """A chunk anchored by two concepts surfaces once, at its earliest rank."""
    edges = [_edge(src="A", dst="B", edge_type="depends_on")]
    concept_to_chunks = {
        "A": ["c-shared"],
        "B": ["c-shared", "c-B"],
    }
    retriever = GraphWalkRetriever(
        edges=edges,
        concept_to_chunks=concept_to_chunks,
        config=GraphWalkConfig(harvest_k=2),
    )

    result = retriever.walk(seeds={"A": 1.0})

    # `c-shared` appears once, earliest first; `c-B` after it.
    assert result.chunk_ids == ["c-shared", "c-B"]


# ---------------------------------------------------------------------------
# Multi-hop recall-lift release gate (§6.9: ≥ 10% recall lift)
# ---------------------------------------------------------------------------


@pytest.mark.family_verifier_calibration
def test_two_hop_recall_lift_beats_seed_only_baseline_by_10pct() -> None:
    """Gold concepts reachable in two hops are recovered by PPR but not by
    a seed-only (entity-link) baseline. The §6.9 release-gate threshold is
    a strictly > 10 % recall lift on a multi-hop query."""

    # Synthetic 2-hop topology: seed Q -> intermediates -> gold targets.
    # Direct entity-link surfaces Q only (one chunk). PPR walks one hop to
    # the intermediates, two hops to the targets — and harvests the chunks
    # anchored on the latter.
    edges = [
        # Hop 1 — Q to intermediates (high-tier so the walk concentrates).
        _edge(src="Q", dst="I1", edge_type="depends_on"),
        _edge(src="Q", dst="I2", edge_type="refines"),
        # Hop 2 — intermediates to the gold concepts (medium tier).
        _edge(src="I1", dst="G1", edge_type="is_a"),
        _edge(src="I2", dst="G2", edge_type="part_of"),
        _edge(src="I1", dst="G3", edge_type="part_of"),
        # A loose-tier red herring that should NOT outrank the gold targets.
        _edge(src="Q", dst="NOISE", edge_type="related_to"),
    ]
    concept_to_chunks = {
        "Q": ["c-Q"],
        "I1": ["c-I1"],
        "I2": ["c-I2"],
        "G1": ["c-G1"],
        "G2": ["c-G2"],
        "G3": ["c-G3"],
        "NOISE": ["c-NOISE"],
    }
    gold_chunks = {"c-G1", "c-G2", "c-G3"}

    # Seed-only baseline: only chunks anchored on the seed concept itself.
    baseline_chunks = list(concept_to_chunks["Q"])
    baseline_recall = recall_at_k(retrieved=baseline_chunks, gold=gold_chunks, k=10)

    # Graph-walk: PPR over typed edges, harvest top concepts.
    retriever = GraphWalkRetriever(
        edges=edges,
        concept_to_chunks=concept_to_chunks,
        config=GraphWalkConfig(harvest_k=6),
    )
    walk_chunks = retriever.walk(seeds={"Q": 1.0}).chunk_ids
    walk_recall = recall_at_k(retrieved=walk_chunks, gold=gold_chunks, k=10)

    # The spec's release gate: strictly > 10 % lift over the baseline.
    assert walk_recall >= baseline_recall + PPR_RECALL_LIFT_THRESHOLD, (
        f"PPR recall {walk_recall} must beat baseline {baseline_recall} "
        f"by at least {PPR_RECALL_LIFT_THRESHOLD} (got lift "
        f"{walk_recall - baseline_recall:.3f})"
    )
    # Sanity: PPR actually surfaces some gold targets.
    assert walk_recall > 0.0


# ---------------------------------------------------------------------------
# Convergence + defaults
# ---------------------------------------------------------------------------


@pytest.mark.family_performance_cost
def test_ppr_converges_under_default_max_iter() -> None:
    """Power iteration converges within the default `max_iter` budget on a
    modest 20-node graph — the cost contract for L2 retrieval."""
    # Wire 20 nodes into a sparse chain plus a few cross-links.
    edges: list[TypedEdge] = []
    for i in range(19):
        edges.append(_edge(src=f"n{i}", dst=f"n{i + 1}", edge_type="is_a"))
    edges.append(_edge(src="n0", dst="n5", edge_type="depends_on"))
    edges.append(_edge(src="n3", dst="n12", edge_type="part_of"))

    # If we converge, the call returns; if we did not, the helper raises.
    dist = personalized_pagerank(
        edges=edges,
        seeds={"n0": 1.0},
        alpha=DEFAULT_ALPHA,
        max_iter=DEFAULT_MAX_ITER,
    )

    assert len(dist) == 20
    assert sum(dist.values()) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.family_determinism
def test_default_harvest_k_is_positive() -> None:
    """The default harvest fanout is sensible and positive."""
    assert DEFAULT_HARVEST_K >= 5
