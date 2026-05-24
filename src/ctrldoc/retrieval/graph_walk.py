"""Personalized PageRank over typed claim-graph edges — L2 graph-walk retrieval.

Pure vector retrieval cannot find chunks that are two hops away in the
concept graph: a query about a concept `Q` whose answer lives in a chunk
anchored on a sibling concept `G` reachable only via an intermediate
`I` is invisible to single-hop dense / lexical lookup. §6.9 upgrades L2 by
diffusing a seed set of concept nodes (the query's entity-linked anchors)
across the typed-edge graph with personalized PageRank, harvesting the
chunks anchored on the top-N concepts by stationary probability, then
fusing with the existing dense ⊕ BM25 ⊕ entity ranks via the established
reciprocal-rank-fusion step.

The §6.9 weight ladder is encoded once in `EDGE_TYPE_WEIGHTS`:

* `depends_on`, `refines`, `prerequisite_of` — **high** (precision-relevant).
* `is_a`, `part_of`                         — **medium** (abstraction-relevant).
* `related_to`                              — **low** (loose similarity).

Edges of other types in the `TypedEdgeTypeLiteral` alphabet are absent from
the walker — they describe propositional logic (`entails`, `contradicts`,
`equivalent_to`), cross-doc bridges (`aligned_with`, `entails_across`,
`contradicts_across`), or instantiations (`instantiates`, `example_of`,
`alternative_to`, `stronger_than`) that the retrieval layer does not climb.
Including them here would warp PPR mass distribution into directions the
spec explicitly excludes from L2 walks; the propositional types are the
domain of L3 inference and the cross-doc types are the domain of L2.5.

Edge weight per outgoing edge `e` from node `u`:

    w(e) = EDGE_TYPE_WEIGHTS[e.type] * e.confidence

PPR transition: from any node `u` with positive outgoing weight sum, the
walker hops to a neighbour with probability proportional to `w(e)`; from a
dangling node (no outgoing edges in the supported types) all mass teleports
to the seed distribution. With restart probability `1 - alpha` per step the
walker resets to the seed distribution — `alpha` defaults to 0.85 (the
PageRank original).

Determinism: PPR output is reproducible across reorderings of the input
edge list because the iteration consumes a sorted adjacency dict (node id,
then destination id). Power-iteration termination is L1 distance below
`tol`; convergence is guaranteed by the PageRank fixed-point because the
transition matrix is row-stochastic on supported nodes and the teleport
restart ensures aperiodicity.

Cost contract: each iteration is `O(|edges|)`. The default
`max_iter = 50` is the L2 retrieval cost ceiling — power iteration on the
sparse claim graph converges in well under that for any seed.

SPEC-REF: §6.9 (graph-walk retrieval — personalized PageRank along typed edges)
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from ctrldoc.models_v1 import TypedEdge, TypedEdgeTypeLiteral

# ---------------------------------------------------------------------------
# §6.9 edge-type weight ladder
# ---------------------------------------------------------------------------


# Strict tier separation — `low < medium < high` with daylight between tiers.
# Magnitudes are spaced so that a high-tier edge outweighs a low-tier edge
# even when the low-tier edge carries unit confidence and the high-tier edge
# carries the §6.5 calibration floor (≈ 0.5).
_HIGH_WEIGHT: float = 3.0
_MEDIUM_WEIGHT: float = 1.0
_LOW_WEIGHT: float = 0.25


EDGE_TYPE_WEIGHTS: dict[TypedEdgeTypeLiteral, float] = {
    "depends_on": _HIGH_WEIGHT,
    "refines": _HIGH_WEIGHT,
    "prerequisite_of": _HIGH_WEIGHT,
    "is_a": _MEDIUM_WEIGHT,
    "part_of": _MEDIUM_WEIGHT,
    "related_to": _LOW_WEIGHT,
}
"""Per-edge-type walker weights — the §6.9 ladder verbatim. Edge types
absent from this map are skipped by the walker (see module docstring)."""


# ---------------------------------------------------------------------------
# Tunable defaults
# ---------------------------------------------------------------------------


DEFAULT_ALPHA: float = 0.85
"""Random-walk persistence — the classic PageRank value. `1 - alpha` is the
per-step teleport-to-seed probability."""

DEFAULT_MAX_ITER: int = 50
"""Power-iteration ceiling. Convergence on the sparse claim graph is fast;
the cap is a runtime safety, not a per-call typical."""

DEFAULT_TOL: float = 1.0e-8
"""L1-distance termination tolerance. Two successive distributions whose L1
delta drops under this value are considered converged."""

DEFAULT_HARVEST_K: int = 10
"""Top-N concepts harvested by stationary probability — feeds chunks into the
RRF fusion step."""

PPR_RECALL_LIFT_THRESHOLD: float = 0.10
"""§6.9 release-gate: graph-walk recall must beat the seed-only baseline by
at least this much (absolute), on multi-hop queries the spec calls out."""


# ---------------------------------------------------------------------------
# Pure PPR primitive
# ---------------------------------------------------------------------------


def personalized_pagerank(
    *,
    edges: Iterable[TypedEdge],
    seeds: Mapping[str, float],
    alpha: float = DEFAULT_ALPHA,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
) -> dict[str, float]:
    """Compute the personalized-PageRank stationary distribution.

    Args:
        edges: Typed edges over the claim-graph. Only edges whose `type`
            appears in `EDGE_TYPE_WEIGHTS` contribute to walks.
        seeds: Restart distribution — `node_id -> mass`. Mass need not sum
            to one; the helper normalises it. Must be non-empty.
        alpha: Walk persistence in (0, 1). `1 - alpha` is the per-step
            teleport-to-seed probability.
        max_iter: Power-iteration ceiling.
        tol: L1 termination tolerance.

    Returns:
        A `dict[str, float]` over every node mentioned in `seeds` or
        `edges`, summing to 1.0 within `tol`.

    Raises:
        ValueError: empty seeds, or alpha not strictly in (0, 1).
    """
    if not seeds:
        raise ValueError("seeds must be non-empty")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in open (0, 1) (got {alpha})")

    # 1. Materialise nodes + weighted adjacency, deterministically.
    nodes: set[str] = set(seeds.keys())
    # `adjacency[u]` is a sorted list of (dst, weight) for stable iteration.
    raw_adj: dict[str, dict[str, float]] = {}
    for edge in edges:
        nodes.add(edge.src_id)
        nodes.add(edge.dst_id)
        weight = EDGE_TYPE_WEIGHTS.get(edge.type)
        if weight is None:
            continue
        bucket = raw_adj.setdefault(edge.src_id, {})
        # Multi-edges of the same (src, dst, type) cannot exist per §8 PK,
        # but two different supported types (e.g. is_a + part_of) on the same
        # endpoints sum their weights so the walker treats them as a single
        # composite link.
        bucket[edge.dst_id] = bucket.get(edge.dst_id, 0.0) + weight * edge.confidence

    # Sorted-key snapshot — iteration order is now byte-stable across runs.
    adjacency: dict[str, list[tuple[str, float]]] = {
        u: sorted(neighbours.items(), key=lambda kv: kv[0]) for u, neighbours in raw_adj.items()
    }
    node_order: list[str] = sorted(nodes)

    # 2. Normalise the seed (personalisation) vector.
    seed_total = sum(seeds.values())
    if seed_total <= 0.0:
        raise ValueError("seeds must carry strictly positive total mass")
    personalisation: dict[str, float] = dict.fromkeys(node_order, 0.0)
    for n, mass in seeds.items():
        personalisation[n] += mass / seed_total

    # 3. Initial distribution = personalisation vector.
    current: dict[str, float] = dict(personalisation)

    # 4. Power iteration.
    for _ in range(max_iter):
        nxt: dict[str, float] = dict.fromkeys(node_order, 0.0)
        # Random-jump term: every step has `1 - alpha` probability of
        # teleporting to the seed distribution.
        for n in node_order:
            nxt[n] += (1.0 - alpha) * personalisation[n]
        # Walk term.
        for u in node_order:
            mass = current[u]
            if mass <= 0.0:
                continue
            neighbours = adjacency.get(u)
            if not neighbours:
                # Dangling node — by convention its mass teleports to the
                # seed distribution (rather than being lost).
                for n in node_order:
                    nxt[n] += alpha * mass * personalisation[n]
                continue
            total = sum(w for _, w in neighbours)
            if total <= 0.0:
                # Should not happen — weights are positive by construction
                # — but guard against floating underflow in the worst case.
                for n in node_order:
                    nxt[n] += alpha * mass * personalisation[n]
                continue
            for dst, weight in neighbours:
                nxt[dst] += alpha * mass * (weight / total)

        # 5. Convergence check (L1).
        delta = sum(abs(nxt[n] - current[n]) for n in node_order)
        current = nxt
        if delta < tol:
            break

    return current


# ---------------------------------------------------------------------------
# Retriever — harvest stage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphWalkConfig:
    """Tunable knobs for the graph-walk retriever."""

    alpha: float = DEFAULT_ALPHA
    """Walk persistence. See `personalized_pagerank`."""

    max_iter: int = DEFAULT_MAX_ITER
    """Power-iteration ceiling."""

    tol: float = DEFAULT_TOL
    """L1 termination tolerance."""

    harvest_k: int = DEFAULT_HARVEST_K
    """Top-N concepts harvested from the stationary distribution."""


class GraphWalkResult(BaseModel):
    """Aggregate output of one `GraphWalkRetriever.walk` call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    concept_ids: list[str]
    """Top-`harvest_k` concept ids by stationary probability, in rank order."""

    chunk_ids: list[str]
    """Deduped chunk ids in concept-rank order then within-concept input
    order. Concepts with no anchored chunks contribute nothing."""

    distribution: dict[str, float]
    """The full PPR distribution over every node seen — kept on the result
    so callers can inspect lower-ranked concepts without re-walking."""


class GraphWalkRetriever:
    """Diffuse a seed concept set across typed edges; harvest anchored chunks.

    Construction is cheap; the PPR walk runs lazily on `walk(...)`. The
    retriever holds no per-call state, so it is safe to share across
    threads provided the input collections are not mutated externally.
    """

    def __init__(
        self,
        *,
        edges: Sequence[TypedEdge],
        concept_to_chunks: Mapping[str, Sequence[str]],
        config: GraphWalkConfig | None = None,
    ) -> None:
        self._edges = list(edges)
        # Defensive copy keeps the retriever insulated from caller mutation.
        self._concept_to_chunks: dict[str, list[str]] = {
            concept_id: list(chunk_ids) for concept_id, chunk_ids in concept_to_chunks.items()
        }
        self._config = config or GraphWalkConfig()

    def walk(self, *, seeds: Mapping[str, float]) -> GraphWalkResult:
        """Run PPR from `seeds`, harvest top concepts + their chunks."""
        distribution = personalized_pagerank(
            edges=self._edges,
            seeds=seeds,
            alpha=self._config.alpha,
            max_iter=self._config.max_iter,
            tol=self._config.tol,
        )
        # Rank concepts by descending stationary probability, lex tiebreak.
        ranked_concepts = sorted(distribution.items(), key=lambda kv: (-kv[1], kv[0]))
        top_concepts = [c for c, _ in ranked_concepts[: self._config.harvest_k]]

        # Harvest chunks in concept-rank order, dedupe at first occurrence.
        chunk_ids: list[str] = []
        seen: set[str] = set()
        for concept_id in top_concepts:
            for chunk_id in self._concept_to_chunks.get(concept_id, ()):
                if chunk_id not in seen:
                    seen.add(chunk_id)
                    chunk_ids.append(chunk_id)

        return GraphWalkResult(
            concept_ids=top_concepts,
            chunk_ids=chunk_ids,
            distribution=distribution,
        )


# ---------------------------------------------------------------------------
# Recall-at-k helper — for the §6.9 release-gate assertion
# ---------------------------------------------------------------------------


def recall_at_k(*, retrieved: Sequence[str], gold: set[str], k: int) -> float:
    """`recall@k = |retrieved[:k] ∩ gold| / |gold|`.

    Returns 0.0 when `gold` is empty (recall is undefined; the harness
    treats undefined as the floor so a degenerate fixture does not pass
    the gate silently).
    """
    if k < 0:
        raise ValueError(f"k must be non-negative (got {k})")
    if not gold:
        return 0.0
    topk = set(retrieved[:k])
    return len(topk & gold) / len(gold)


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_HARVEST_K",
    "DEFAULT_MAX_ITER",
    "DEFAULT_TOL",
    "EDGE_TYPE_WEIGHTS",
    "PPR_RECALL_LIFT_THRESHOLD",
    "GraphWalkConfig",
    "GraphWalkResult",
    "GraphWalkRetriever",
    "personalized_pagerank",
    "recall_at_k",
]
