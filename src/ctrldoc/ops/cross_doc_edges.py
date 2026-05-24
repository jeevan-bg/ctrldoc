"""Workspace cross-doc edge inference — `aligned_with` / `entails_across` /
`contradicts_across` over N member docs.

A workspace shares one concept lattice across its docs (§6.7). Cross-doc edges
are the lazy, cached, **linear** bridge between two docs' claim graphs that the
optimal-transport ops (`coverage`, `compare`, `merge`, `list_check` — Phase
18) walk to align claims across docs. This module is the producer of those
edges from an NLI scorer; persistence into `cross_doc_edges` and replay live
with the transport engine.

The inferer is a workspace-shaped sibling of the per-doc Tier-2 NLI edge
inferer (`ctrldoc.extract.tier2_nli`): same `NLIScorer` protocol, same
token-overlap candidate ranker, same threshold-driven emission. What changes:

* The unit of work is an ordered pair of distinct docs `(A, B)`. For every
  claim `a` in A, the top-`k` claims in B are picked under token-overlap
  Jaccard and scored exactly once.
* Edge types are the cross-doc trio from `TypedEdgeTypeLiteral`:
  - `entails_across` when entailment crosses the entail threshold,
  - `contradicts_across` when contradiction crosses its threshold,
  - `aligned_with` when entailment sits in the soft band
    `[aligned_with_threshold, entails_across_threshold)` — paraphrase /
    near-equivalent claims that are not strictly entailing.
* Endpoint identity is the persisted `Claim.id` verbatim (not a rehash),
  because the workspace already has stable claim ids at the storage layer
  (§7) and the transport engine needs the same ids it reads back from
  `claims`.

Cost contract (§6.7): scorer calls grow linearly. For each ordered doc pair
the inferer issues at most `k * |A|` calls, so the total is bounded by
`k * sum(|d|) * (n_docs - 1)`. The default `k = 5` matches `Tier2NLIConfig`'s
fanout and the §6.5 `5N` envelope.

Determinism: edges sort by `(type, src_id, dst_id)`; tied Jaccard scores break
on lexicographic `claim_id` so the candidate set is reproducible across runs.

SPEC-REF: §6.7 (workspace cross-doc edges, lazy + cached + linear)
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from ctrldoc.eval.calibration import NLIScore
from ctrldoc.models import Span
from ctrldoc.models_v1 import Claim, TypedEdge

# ---------------------------------------------------------------------------
# Public thresholds + defaults
# ---------------------------------------------------------------------------


DEFAULT_K_CANDIDATES: int = 5
"""§6.7 candidate-retrieval fanout. Caps scorer calls at `k * |A|` per ordered pair."""

ENTAILS_ACROSS_THRESHOLD: float = 0.70
"""Top-label entailment confidence required to emit an `entails_across` edge."""

ALIGNED_WITH_THRESHOLD: float = 0.50
"""Soft-alignment lower bound. Pairs in `[this, entails_across)` emit `aligned_with`."""

CONTRADICTS_ACROSS_THRESHOLD: float = 0.70
"""Top-label contradiction confidence required to emit a `contradicts_across` edge."""


# ---------------------------------------------------------------------------
# Scorer protocol — re-used from the calibration substrate
# ---------------------------------------------------------------------------


@runtime_checkable
class NLIScorer(Protocol):
    """3-way NLI backend. Same shape as `CalibrationScorer` and `tier2_nli.NLIScorer`."""

    def score(self, *, premise: str, hypothesis: str) -> NLIScore: ...


# ---------------------------------------------------------------------------
# Config + result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrossDocEdgeConfig:
    """Tunable knobs for the workspace cross-doc edge inferer."""

    k_candidates: int = DEFAULT_K_CANDIDATES
    """Per-source-claim fanout into the target doc. Hard cap: `k * |A|` per pair."""

    entail_threshold: float = ENTAILS_ACROSS_THRESHOLD
    """Minimum top-label entailment probability for an `entails_across` edge."""

    aligned_threshold: float = ALIGNED_WITH_THRESHOLD
    """Minimum entailment probability for the soft `aligned_with` band."""

    contradict_threshold: float = CONTRADICTS_ACROSS_THRESHOLD
    """Minimum top-label contradiction probability for a `contradicts_across` edge."""


class CrossDocEdgeInference(BaseModel):
    """Aggregate output of one `CrossDocEdgeInferer.infer` call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    workspace_id: str
    edges: list[TypedEdge]
    scorer_calls: int
    """Bookkeeping: actual NLI calls issued. Must obey `≤ k * |A|` per ordered pair."""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


class CrossDocEdgeInferer:
    """Bridge two or more doc-graphs with cross-doc NLI edges (§6.7).

    Construction is cheap; the scorer is invoked lazily inside `infer`.
    The inferer holds no per-call state — it is safe to share across
    threads as long as the underlying scorer is.
    """

    def __init__(
        self,
        *,
        scorer: NLIScorer,
        config: CrossDocEdgeConfig | None = None,
    ) -> None:
        if config is not None:
            if config.k_candidates < 1:
                raise ValueError(f"k_candidates must be >= 1 (got {config.k_candidates})")
            if not 0.0 < config.aligned_threshold <= config.entail_threshold < 1.0:
                raise ValueError(
                    "thresholds must satisfy 0 < aligned_threshold <= "
                    f"entail_threshold < 1 (got aligned={config.aligned_threshold}, "
                    f"entail={config.entail_threshold})"
                )
            if not 0.0 < config.contradict_threshold < 1.0:
                raise ValueError(
                    f"contradict_threshold must be in (0, 1) (got {config.contradict_threshold})"
                )
        self._scorer = scorer
        self._config = config or CrossDocEdgeConfig()

    def infer(
        self,
        *,
        workspace_id: str,
        claims_by_doc: Mapping[str, Iterable[Claim]],
    ) -> CrossDocEdgeInference:
        """Score every ordered cross-doc candidate pair; emit cross-doc edges.

        `claims_by_doc` maps doc id → that doc's claim list. Docs with
        zero claims are skipped. With fewer than two non-empty docs no
        cross-doc pair exists and the inferer short-circuits to an empty
        inference.
        """
        # Materialise once; preserve insertion order so cross-doc edge
        # enumeration is reproducible given the same input mapping.
        materialised: list[tuple[str, list[Claim]]] = [
            (doc_id, list(claims)) for doc_id, claims in claims_by_doc.items()
        ]
        non_empty = [(doc_id, claims) for doc_id, claims in materialised if claims]
        if len(non_empty) < 2:
            return CrossDocEdgeInference(workspace_id=workspace_id, edges=[], scorer_calls=0)

        # Pre-compute token bags per claim per doc once. Token-overlap
        # ranking reads them for every (source-claim, target-doc) lookup
        # so caching once is the only sensible move.
        bags_by_doc: dict[str, list[frozenset[str]]] = {
            doc_id: [_tokens(c.text) for c in claims] for doc_id, claims in non_empty
        }

        k = self._config.k_candidates
        edges: list[TypedEdge] = []
        scorer_calls = 0
        # Iterate every ordered (source_doc, target_doc) pair where source != target.
        # Linear in `n_docs * (n_docs - 1)`; each pair contributes ≤ k * |source|
        # scorer calls so total stays linear in `sum(|d|) * (n_docs - 1)`.
        for src_doc, src_claims in non_empty:
            src_bags = bags_by_doc[src_doc]
            for dst_doc, dst_claims in non_empty:
                if dst_doc == src_doc:
                    continue
                dst_bags = bags_by_doc[dst_doc]
                # For each source claim, pick the top-k destination claims.
                for src_idx, src_claim in enumerate(src_claims):
                    candidates = _rank_targets(
                        anchor_bag=src_bags[src_idx],
                        target_bags=dst_bags,
                        target_ids=[c.id for c in dst_claims],
                    )
                    for dst_idx in candidates[:k]:
                        dst_claim = dst_claims[dst_idx]
                        scorer_calls += 1
                        score = self._scorer.score(
                            premise=src_claim.text,
                            hypothesis=dst_claim.text,
                        )
                        edge = _maybe_build_edge(
                            src=src_claim,
                            dst=dst_claim,
                            score=score,
                            config=self._config,
                        )
                        if edge is not None:
                            edges.append(edge)

        edges.sort(key=lambda e: (e.type, e.src_id, e.dst_id))
        return CrossDocEdgeInference(
            workspace_id=workspace_id,
            edges=edges,
            scorer_calls=scorer_calls,
        )


# ---------------------------------------------------------------------------
# Edge builder — maps an NLI verdict to an optional cross-doc TypedEdge
# ---------------------------------------------------------------------------


def _maybe_build_edge(
    *,
    src: Claim,
    dst: Claim,
    score: NLIScore,
    config: CrossDocEdgeConfig,
) -> TypedEdge | None:
    """Decide which (if any) cross-doc edge type fires for this NLI verdict.

    Decision order: contradiction → entailment → aligned. A pair never
    fires more than one edge type per call — the verdicts are mutually
    exclusive by construction because the entailment / contradiction
    bands sit on disjoint axes of the 3-way softmax and the soft-aligned
    band sits strictly below the entailment cutoff on the entailment
    axis.
    """
    top_label = score.argmax_label()
    top_conf = score.top_confidence()

    if top_label == "contradiction" and top_conf >= config.contradict_threshold:
        return _build_edge(src=src, dst=dst, edge_type="contradicts_across", confidence=top_conf)
    if top_label == "entailment":
        if top_conf >= config.entail_threshold:
            return _build_edge(src=src, dst=dst, edge_type="entails_across", confidence=top_conf)
        if top_conf >= config.aligned_threshold:
            return _build_edge(src=src, dst=dst, edge_type="aligned_with", confidence=top_conf)
    return None


def _build_edge(
    *,
    src: Claim,
    dst: Claim,
    edge_type: str,
    confidence: float,
) -> TypedEdge:
    """Materialise a cross-doc `TypedEdge` citing one span from each endpoint.

    The first `span_ref` from each `Claim` is the canonical citation —
    every persisted claim is required to have at least one span (§7),
    and the v1 trace renderer only needs one anchor per endpoint to
    point a reviewer at the source-doc and target-doc evidence.
    """
    citations = (
        [src.span_refs[0], dst.span_refs[0]]
        if src.span_refs and dst.span_refs
        else [
            # Fall back to synthesised spans so the edge still has provenance
            # in the (unexpected) absence of `span_refs`. Defense in depth —
            # the §7 contract should already guarantee non-empty `span_refs`.
            Span(
                chunk_id=f"cross-doc:{src.id}",
                char_start=0,
                char_end=len(src.text),
                text=src.text,
            ),
            Span(
                chunk_id=f"cross-doc:{dst.id}",
                char_start=0,
                char_end=len(dst.text),
                text=dst.text,
            ),
        ]
    )
    if edge_type == "entails_across":
        return TypedEdge(
            src_id=src.id,
            dst_id=dst.id,
            type="entails_across",
            confidence=confidence,
            raw_score=confidence,
            citations=citations,
            source="nli",
            paraphrase_votes=None,
        )
    if edge_type == "contradicts_across":
        return TypedEdge(
            src_id=src.id,
            dst_id=dst.id,
            type="contradicts_across",
            confidence=confidence,
            raw_score=confidence,
            citations=citations,
            source="nli",
            paraphrase_votes=None,
        )
    if edge_type == "aligned_with":
        return TypedEdge(
            src_id=src.id,
            dst_id=dst.id,
            type="aligned_with",
            confidence=confidence,
            raw_score=confidence,
            citations=citations,
            source="nli",
            paraphrase_votes=None,
        )
    raise ValueError(f"unsupported cross-doc edge_type {edge_type!r}")


# ---------------------------------------------------------------------------
# Candidate retrieval — token-overlap ranking across two docs
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> frozenset[str]:
    """Lower-cased word-token bag, single-character tokens dropped.

    Identical to `tier2_nli._tokens` — kept private here so the two
    layers stay decoupled (S-136 paraphrase voting will replace the
    ranker on the per-doc side without dragging the cross-doc layer
    along).
    """
    return frozenset(t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 2)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Symmetric Jaccard overlap. 0 when either bag is empty."""
    if not a or not b:
        return 0.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


def _rank_targets(
    *,
    anchor_bag: frozenset[str],
    target_bags: list[frozenset[str]],
    target_ids: list[str],
) -> list[int]:
    """Target indices sorted by (descending Jaccard, lexicographic id).

    Stable for repeated runs because the tiebreak on `target_id` totalises
    the ordering even when several target claims share the same Jaccard
    score against the anchor.
    """
    scored = [
        (_jaccard(anchor_bag, target_bags[j]), target_ids[j], j) for j in range(len(target_bags))
    ]
    scored.sort(key=lambda triple: (-triple[0], triple[1]))
    return [j for _, _, j in scored]


__all__ = [
    "ALIGNED_WITH_THRESHOLD",
    "CONTRADICTS_ACROSS_THRESHOLD",
    "DEFAULT_K_CANDIDATES",
    "ENTAILS_ACROSS_THRESHOLD",
    "CrossDocEdgeConfig",
    "CrossDocEdgeInference",
    "CrossDocEdgeInferer",
    "NLIScorer",
]
