"""Entity resolution (canonicalization) — blocking + LLM-judge ER.

The L1.5 substrate is built on canonical `Concept` nodes shared across
documents. This module is the §6.8 producer of those concepts: it
collapses noisy cross-document mentions into one canonical cluster per
underlying concept and emits Galois subsumption edges between them.

The recipe follows the standard ER four-step:

1. **Blocking.** For each pair of `ConceptMention` rows of the same
   `primitive_type`, compute the cosine similarity of their embeddings
   (any `Embedder` backend will do — production wiring uses the
   `OllamaEmbedder` for bge-m3 1024-d vectors). Pairs below the
   `tau_block` threshold (default 0.85 per §6.8) are dropped.
2. **LLM judge.** The blocked candidates are handed to an
   `ERJudge` one pair at a time. The judge returns a four-class verdict
   (`equivalent` / `subsumes` / `subsumed_by` / `incomparable`) with no
   doc body in context — only the mention surfaces and primitive types.
   The default `HeuristicERJudge` shipped here is a deterministic
   reference suitable for tests and the deterministic profile; the
   production LLM judges land in their own backend modules.
3. **Union-find.** Equivalence verdicts merge mentions into canonical
   clusters. Each cluster becomes one `Concept` row whose
   `canonical_name` is the most frequent (then lexicographically
   smallest) surface form among its members.
4. **Subsumption edges.** `subsumes(L, R)` and `subsumed_by(L, R)`
   verdicts translate into `is_a` typed edges between the **canonical
   concept ids** of L and R (not the raw mention ids). Direction:
   the more-specific concept is the source; the more-general concept
   is the destination. Duplicate edges that collapse onto the same
   canonical pair are kept once. Edges sort by `(src_id, dst_id)` for
   deterministic diffs.

`cluster_precision_recall` is the off-the-shelf pairwise scoring
helper used to gate the slice's release thresholds (precision ≥ 0.90
and recall ≥ 0.85 per §6.8). It compares any predicted partition to a
gold partition by counting same-cluster mention pairs.

SPEC-REF: §6.8 (entity resolution / canonicalization)
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, runtime_checkable

from pydantic import BaseModel, ConfigDict

from ctrldoc.ingest.embedder import Embedder
from ctrldoc.models_v1 import Concept, PrimitiveTypeLiteral, TypedEdge
from ctrldoc.versioning import content_hash

# ---------------------------------------------------------------------------
# Public release-gate constants (per §6.8)
# ---------------------------------------------------------------------------


DEFAULT_TAU_BLOCK: float = 0.85
"""Embedding-cosine threshold above which a pair survives blocking."""

ER_PRECISION_THRESHOLD: float = 0.90
"""§6.8 release gate — minimum pairwise precision on the gold fixture."""

ER_RECALL_THRESHOLD: float = 0.85
"""§6.8 release gate — minimum pairwise recall on the gold fixture."""


EntityResolutionVerdict: TypeAlias = Literal[
    "equivalent",
    "subsumes",
    "subsumed_by",
    "incomparable",
]
"""The four-class verdict an `ERJudge` returns per candidate pair."""


# ---------------------------------------------------------------------------
# Inputs / outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConceptMention:
    """One mention of a concept in a document — the input to ER.

    `id` must be unique across the input batch; the resolver uses it as
    the union-find key and as the basis for the canonical `Concept.id`.
    `claim_id` is the parent claim row the mention was extracted from
    (used to populate `Concept.mention_claim_ids`).
    """

    id: str
    mention_text: str
    primitive_type: PrimitiveTypeLiteral
    doc_id: str
    claim_id: str


class EntityResolution(BaseModel):
    """Aggregate output of one `EntityResolver.resolve` call.

    `concepts` is the canonical cluster list; one `Concept` per cluster.
    `subsumption_edges` carries the `is_a` edges Galois-derived from
    the judge's `subsumes` / `subsumed_by` verdicts. `clusters` is the
    parallel mention-id partition the caller needs to rebuild
    cluster-membership for scoring (the `Concept` row only persists
    claim ids; mention ids are this layer's transient bookkeeping).
    `judge_calls` is the actual number of judge invocations the
    resolver issued — never exceeds `C(N, 2)` after blocking.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    concepts: list[Concept]
    subsumption_edges: list[TypedEdge]
    clusters: list[list[str]]
    judge_calls: int


@dataclass(frozen=True)
class EntityResolutionConfig:
    """Tunable knobs for the entity resolver."""

    tau_block: float = DEFAULT_TAU_BLOCK
    """Cosine similarity below which a candidate pair is dropped."""


# ---------------------------------------------------------------------------
# Judge protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ERJudge(Protocol):
    """Decides equivalence / subsumption for one candidate mention pair."""

    def judge(
        self,
        *,
        left: ConceptMention,
        right: ConceptMention,
    ) -> EntityResolutionVerdict: ...


# ---------------------------------------------------------------------------
# Pairwise scoring helper — used by §6.8 release gates
# ---------------------------------------------------------------------------


def cluster_precision_recall(
    *,
    predicted: Sequence[set[str]],
    gold: Sequence[set[str]],
) -> tuple[float, float]:
    """Pairwise precision / recall of a predicted partition against a gold one.

    Both arguments are sequences of disjoint mention-id sets. The metric
    enumerates the unordered same-cluster pairs in each partition and
    computes:

    * precision = |predicted_pairs ∩ gold_pairs| / |predicted_pairs|.
    * recall    = |predicted_pairs ∩ gold_pairs| / |gold_pairs|.

    An empty `predicted_pairs` set yields precision 1.0 by convention
    (no false positives to count against). An empty `gold_pairs` set
    yields recall 1.0 (nothing to recall).
    """

    def _pairs(clusters: Sequence[set[str]]) -> set[frozenset[str]]:
        out: set[frozenset[str]] = set()
        for cluster in clusters:
            members = sorted(cluster)
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    out.add(frozenset({members[i], members[j]}))
        return out

    predicted_pairs = _pairs(predicted)
    gold_pairs = _pairs(gold)
    tp = len(predicted_pairs & gold_pairs)
    precision = 1.0 if not predicted_pairs else tp / len(predicted_pairs)
    recall = 1.0 if not gold_pairs else tp / len(gold_pairs)
    return precision, recall


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


class EntityResolver:
    """Run the four-step §6.8 ER recipe over a batch of mentions.

    Construction is cheap; embedder + judge calls happen lazily inside
    `resolve`. The resolver holds no per-call state — it is safe to
    share across threads as long as the embedder and judge are.
    """

    def __init__(
        self,
        *,
        embedder: Embedder,
        judge: ERJudge,
        config: EntityResolutionConfig | None = None,
    ) -> None:
        if config is not None and not (-1.0 <= config.tau_block <= 1.0):
            raise ValueError(f"tau_block must lie in [-1, 1] (got {config.tau_block})")
        self._embedder = embedder
        self._judge = judge
        self._config = config or EntityResolutionConfig()

    def resolve(self, mentions: Iterable[ConceptMention]) -> EntityResolution:
        """Block → judge → union-find → emit canonical concepts and edges."""
        mention_list = list(mentions)
        n = len(mention_list)
        if n == 0:
            return EntityResolution(
                concepts=[],
                subsumption_edges=[],
                clusters=[],
                judge_calls=0,
            )

        # 1. Single-mention short-circuit: no pairs to block, no judge calls.
        if n == 1:
            only = mention_list[0]
            concept = _build_concept(cluster=[only])
            return EntityResolution(
                concepts=[concept],
                subsumption_edges=[],
                clusters=[[only.id]],
                judge_calls=0,
            )

        # 2. Embed every mention once. Token-thrifty; reused for every
        #    blocked pair's cosine computation.
        embeddings: list[list[float]] = self._embedder.embed_batch(
            [m.mention_text for m in mention_list]
        )

        # 3. Blocking: emit (i, j) pairs with i < j, same primitive type,
        #    and cosine ≥ tau_block. Ordering by (i, j) is stable and
        #    drives deterministic judge-call order.
        blocked: list[tuple[int, int]] = []
        tau = self._config.tau_block
        for i in range(n):
            for j in range(i + 1, n):
                if mention_list[i].primitive_type != mention_list[j].primitive_type:
                    continue
                if _cosine(embeddings[i], embeddings[j]) < tau:
                    continue
                blocked.append((i, j))

        # 4. Judge each surviving pair once and collect equivalence /
        #    subsumption verdicts.
        equivalence_pairs: list[tuple[int, int]] = []
        subsumption_pairs: list[tuple[int, int, EntityResolutionVerdict]] = []
        for i, j in blocked:
            verdict = self._judge.judge(
                left=mention_list[i],
                right=mention_list[j],
            )
            if verdict == "equivalent":
                equivalence_pairs.append((i, j))
            elif verdict in ("subsumes", "subsumed_by"):
                subsumption_pairs.append((i, j, verdict))
            # `incomparable` is the explicit no-edge signal — drop.

        # 5. Union-find over equivalence verdicts → canonical cluster ids.
        parent = list(range(n))
        for i, j in equivalence_pairs:
            _union(parent, i, j)
        # Root each member explicitly to flatten paths.
        roots = [_find(parent, i) for i in range(n)]
        clusters_by_root: dict[int, list[int]] = {}
        for idx, root in enumerate(roots):
            clusters_by_root.setdefault(root, []).append(idx)

        # 6. Build one `Concept` per cluster. Cluster output order is
        #    keyed on the smallest mention id inside the cluster so the
        #    output is deterministic across runs.
        cluster_groups = sorted(
            clusters_by_root.values(),
            key=lambda members: min(mention_list[i].id for i in members),
        )
        concepts: list[Concept] = []
        concept_id_by_member_idx: dict[int, str] = {}
        cluster_id_lists: list[list[str]] = []
        for members in cluster_groups:
            cluster_mentions = [mention_list[i] for i in members]
            concept = _build_concept(cluster=cluster_mentions)
            concepts.append(concept)
            cluster_id_lists.append(sorted(m.id for m in cluster_mentions))
            for i in members:
                concept_id_by_member_idx[i] = concept.id

        # 7. Subsumption edges: rewrite each (mention, mention) verdict
        #    onto the canonical concept ids of its endpoints. Drop
        #    self-edges (when both mentions ended up in the same
        #    cluster) and deduplicate parallel edges.
        edges_seen: set[tuple[str, str]] = set()
        subsumption_edges: list[TypedEdge] = []
        for i, j, verdict in subsumption_pairs:
            if verdict == "subsumes":
                # left (i) is the parent → child is j; record j is_a i.
                child_idx, parent_idx = j, i
            else:  # subsumed_by
                # left (i) is the child → record i is_a j.
                child_idx, parent_idx = i, j
            src = concept_id_by_member_idx[child_idx]
            dst = concept_id_by_member_idx[parent_idx]
            if src == dst:
                continue
            if (src, dst) in edges_seen:
                continue
            edges_seen.add((src, dst))
            subsumption_edges.append(
                TypedEdge(
                    src_id=src,
                    dst_id=dst,
                    type="is_a",
                    confidence=1.0,
                    raw_score=1.0,
                    citations=[],
                    source="llm",
                    paraphrase_votes=None,
                )
            )

        subsumption_edges.sort(key=lambda e: (e.src_id, e.dst_id))
        return EntityResolution(
            concepts=concepts,
            subsumption_edges=subsumption_edges,
            clusters=cluster_id_lists,
            judge_calls=len(blocked),
        )


# ---------------------------------------------------------------------------
# Concept builder
# ---------------------------------------------------------------------------


def _build_concept(*, cluster: list[ConceptMention]) -> Concept:
    """Build one canonical `Concept` from a non-empty mention cluster.

    Canonical name = the most frequent surface form among the cluster's
    mention texts, with ties broken by lexicographic order so the
    result is deterministic. Aliases = every other distinct surface
    form in the cluster, sorted. Doc ids and claim ids are de-duplicated
    and sorted for stable output. The concept id is a content hash of
    the sorted mention id list so identical clusters produce identical
    ids across runs.
    """
    assert cluster, "cluster must be non-empty"
    primitive = cluster[0].primitive_type
    # Surface-form frequency table.
    counts: dict[str, int] = {}
    for m in cluster:
        counts[m.mention_text] = counts.get(m.mention_text, 0) + 1
    # Sort by (-count, surface) so the most frequent surface wins; ties
    # broken lexicographically.
    by_freq = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    canonical_name = by_freq[0][0]
    aliases = sorted(name for name, _ in by_freq[1:])
    sorted_ids = sorted(m.id for m in cluster)
    concept_id = "concept-" + content_hash("|".join(["concept-cluster", *sorted_ids]))
    mention_claim_ids = sorted({m.claim_id for m in cluster})
    doc_ids = sorted({m.doc_id for m in cluster})
    return Concept(
        id=concept_id,
        canonical_name=canonical_name,
        aliases=aliases,
        primitive_type=primitive,
        mention_claim_ids=mention_claim_ids,
        doc_ids=doc_ids,
    )


# ---------------------------------------------------------------------------
# Cosine + union-find internals
# ---------------------------------------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-dimension float vectors.

    Returns 0.0 if either vector is the zero vector (degenerate input).
    The embedder contract guarantees equal dimensions; a mismatch
    indicates a backend bug and raises explicitly.
    """
    if len(a) != len(b):
        raise ValueError(f"embedding dim mismatch: {len(a)} vs {len(b)}")
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _find(parent: list[int], x: int) -> int:
    """Union-find root lookup with path compression."""
    root = x
    while parent[root] != root:
        root = parent[root]
    # Path compression.
    while parent[x] != root:
        parent[x], x = root, parent[x]
    return root


def _union(parent: list[int], x: int, y: int) -> None:
    """Union by smaller-root-wins so root assignment is deterministic."""
    rx = _find(parent, x)
    ry = _find(parent, y)
    if rx == ry:
        return
    if rx < ry:
        parent[ry] = rx
    else:
        parent[rx] = ry


__all__ = [
    "DEFAULT_TAU_BLOCK",
    "ER_PRECISION_THRESHOLD",
    "ER_RECALL_THRESHOLD",
    "ConceptMention",
    "ERJudge",
    "EntityResolution",
    "EntityResolutionConfig",
    "EntityResolutionVerdict",
    "EntityResolver",
    "cluster_precision_recall",
]
