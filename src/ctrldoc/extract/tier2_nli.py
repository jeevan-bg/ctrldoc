"""Tier-2 NLI edge inference — `entails` / `contradicts` over universal claims.

The Tier-2 NLI edge layer is the §6.5 producer of probabilistic edges
on the per-doc claim graph. Where the Tier-1 deterministic floor
(`tier1.py`) emits heuristic edges with a fixed prior (~0.9), and the
Tier-2 SVO extractor (`tier2_spacy.py`) lifts sentences into universal
`ClaimTuple` rows, this module is the bridge that asks an NLI scorer:
for each claim, which of its closest neighbours does it entail or
contradict?

The cost contract is §6.5's **candidate retrieval, `<= 5 * N` pairs**
rule. Quadratic enumeration over the claim list scales O(N^2) which is
fatal for any non-trivial document. Instead we score each claim
against its top-`k` neighbours under a cheap, deterministic
token-overlap ranker (Jaccard on lower-cased word tokens), which
keeps the NLI call budget strictly bounded at `k * N`. Default
`k = 5` matches the spec's `5N` envelope.

Edge confidence here is the raw top-label probability from the
scorer; the §6.5 calibration step (isotonic regression, paraphrase
voting) is the job of the Phase-17 slices (S-136, S-137). The
`raw_score` field on `TypedEdge` is persisted separately so that
calibration can be fitted later without losing the original signal.

Edges emit only when the top-label confidence crosses
`NLI_ENTAIL_THRESHOLD` (for `entails`) or `NLI_CONTRADICT_THRESHOLD`
(for `contradicts`). Neutral-dominated pairs produce no edge — the
empty result is the correct signal for an unrelated pair. Self-pairs
(a claim against itself) are never scored.

Output ordering: edges sort by `(type, src_id, dst_id)` so diffs are
reviewer-friendly and the inferer is deterministic given a
deterministic scorer. The `src_id` / `dst_id` of each emitted
`TypedEdge` is the content-hashed `claim_id` derived from the claim's
six logical fields — stable across runs and across processes.

SPEC-REF: §6.5 (probabilistic edges + calibration — Tier-2 NLI)
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from ctrldoc.eval.calibration import NLIScore
from ctrldoc.eval.claim_extraction import ClaimTuple
from ctrldoc.models import Span
from ctrldoc.models_v1 import TypedEdge
from ctrldoc.versioning import content_hash

# ---------------------------------------------------------------------------
# Public thresholds + defaults
# ---------------------------------------------------------------------------


DEFAULT_K_CANDIDATES: int = 5
"""§6.5 candidate-retrieval fanout. Caps NLI calls at `k * N` pairs."""

NLI_ENTAIL_THRESHOLD: float = 0.70
"""Top-label confidence required to emit an `entails` edge."""

NLI_CONTRADICT_THRESHOLD: float = 0.70
"""Top-label confidence required to emit a `contradicts` edge."""


# ---------------------------------------------------------------------------
# Scorer protocol — re-used from the calibration substrate
# ---------------------------------------------------------------------------


@runtime_checkable
class NLIScorer(Protocol):
    """3-way NLI backend. Same shape as `CalibrationScorer` from §6.5."""

    def score(self, *, premise: str, hypothesis: str) -> NLIScore: ...


# ---------------------------------------------------------------------------
# Config + result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tier2NLIConfig:
    """Tunable knobs for the Tier-2 NLI edge inferer."""

    k_candidates: int = DEFAULT_K_CANDIDATES
    """Per-claim neighbour fanout. Hard cap on scorer calls is `k * N`."""

    entail_threshold: float = NLI_ENTAIL_THRESHOLD
    """Minimum top-label probability for an `entails` edge."""

    contradict_threshold: float = NLI_CONTRADICT_THRESHOLD
    """Minimum top-label probability for a `contradicts` edge."""


class Tier2NLIExtraction(BaseModel):
    """Aggregate output of one `Tier2NLIEdgeInferer.infer` call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    edges: list[TypedEdge]
    scorer_calls: int
    """Bookkeeping: actual NLI calls issued. Must obey `≤ k * N`."""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


class Tier2NLIEdgeInferer:
    """Build `entails` / `contradicts` edges between universal claims via NLI.

    Construction is cheap; the scorer is invoked lazily inside `infer`.
    The inferer holds no per-call state — it is safe to share across
    threads as long as the underlying scorer is.
    """

    def __init__(
        self,
        *,
        scorer: NLIScorer,
        config: Tier2NLIConfig | None = None,
    ) -> None:
        if config is not None and config.k_candidates < 1:
            raise ValueError(f"k_candidates must be >= 1 (got {config.k_candidates})")
        self._scorer = scorer
        self._config = config or Tier2NLIConfig()

    def infer(self, claims: Iterable[ClaimTuple]) -> Tier2NLIExtraction:
        """Score each claim against its top-`k` neighbours; emit typed edges.

        Single-claim and empty inputs short-circuit to an empty
        extraction — there are no peers to pair with. For N ≥ 2 the
        scorer is asked about at most `k * N` ordered pairs.
        """
        claim_list = list(claims)
        n = len(claim_list)
        if n < 2:
            return Tier2NLIExtraction(edges=[], scorer_calls=0)

        # 1. Pre-compute rendered text + token bags + claim ids.
        rendered: list[str] = [render_claim_text(c) for c in claim_list]
        token_bags: list[frozenset[str]] = [_tokens(t) for t in rendered]
        claim_ids: list[str] = [claim_id(c) for c in claim_list]

        # 2. For every claim, pick top-`k` neighbours by token-overlap
        # Jaccard. Self is excluded; ties broken by stable claim-id
        # ordering so the ranking is fully deterministic.
        k = self._config.k_candidates
        scored_pairs: list[tuple[str, str, int, int]] = []
        # Track ordered (src_idx, dst_idx) pairs already enqueued so we
        # never double-score a pair under both `entails` and
        # `contradicts` — one NLI call decides both labels.
        seen_pairs: set[tuple[int, int]] = set()
        for i in range(n):
            ranked = _rank_neighbours(
                anchor_idx=i,
                token_bags=token_bags,
                claim_ids=claim_ids,
            )
            for j in ranked[:k]:
                if (i, j) in seen_pairs:
                    continue
                seen_pairs.add((i, j))
                scored_pairs.append((rendered[i], rendered[j], i, j))

        # 3. Score each candidate pair exactly once; emit edges when
        # the top-label confidence crosses the relevant threshold.
        edges: list[TypedEdge] = []
        for premise, hypothesis, i, j in scored_pairs:
            score = self._scorer.score(premise=premise, hypothesis=hypothesis)
            top_label = score.argmax_label()
            top_conf = score.top_confidence()
            if top_label == "entailment" and top_conf >= self._config.entail_threshold:
                edges.append(
                    _build_edge(
                        src_id=claim_ids[i],
                        dst_id=claim_ids[j],
                        edge_type="entails",
                        confidence=top_conf,
                        premise=premise,
                        hypothesis=hypothesis,
                    )
                )
            elif top_label == "contradiction" and top_conf >= self._config.contradict_threshold:
                edges.append(
                    _build_edge(
                        src_id=claim_ids[i],
                        dst_id=claim_ids[j],
                        edge_type="contradicts",
                        confidence=top_conf,
                        premise=premise,
                        hypothesis=hypothesis,
                    )
                )

        edges.sort(key=lambda e: (e.type, e.src_id, e.dst_id))
        return Tier2NLIExtraction(edges=edges, scorer_calls=len(scored_pairs))


# ---------------------------------------------------------------------------
# Rendering — `ClaimTuple` → natural-language surface for the NLI backend
# ---------------------------------------------------------------------------


_NEGATION_INSERT: dict[str, str] = {
    "is": "is not",
    "are": "are not",
    "was": "was not",
    "were": "were not",
    "has": "does not have",
    "have": "do not have",
    "had": "did not have",
}


def render_claim_text(claim: ClaimTuple) -> str:
    """Render a `ClaimTuple` as a natural-language sentence for NLI input.

    Polarity flips by either swapping the copula (`is` → `is not`) or
    splicing `does not` before the predicate for plain verbs — both
    forms read naturally to an NLI cross-encoder. Modality is left to
    surface in the predicate text already produced by the Tier-2 SVO
    extractor; injecting modal cues here would double-count what's in
    `predicate`.

    The qualifier slot, when present, trails the subject-predicate-
    object trunk so e.g. `"the leader is elected within five seconds"`
    reads naturally.
    """
    subject = claim.subject.strip()
    predicate = claim.predicate.strip()
    obj = claim.object.strip()
    qualifier = claim.qualifier.strip()

    if claim.polarity == "negative":
        pred_lower = predicate.lower()
        predicate = _NEGATION_INSERT.get(pred_lower, f"does not {predicate}")

    parts = [p for p in (subject, predicate, obj, qualifier) if p]
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Claim identity — content-hashed for stability across runs / processes
# ---------------------------------------------------------------------------


def claim_id(claim: ClaimTuple) -> str:
    """Content-hashed id for a `ClaimTuple` keyed on its six logical fields."""
    payload = "|".join(
        [
            "tier2-nli-claim",
            claim.subject,
            claim.predicate,
            claim.object,
            claim.polarity,
            claim.modality,
            claim.qualifier,
        ]
    )
    return content_hash(payload)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> frozenset[str]:
    """Lower-cased word-token bag, single-character tokens dropped."""
    return frozenset(t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 2)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Symmetric Jaccard overlap. 0 when either bag is empty."""
    if not a or not b:
        return 0.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


def _rank_neighbours(
    *,
    anchor_idx: int,
    token_bags: list[frozenset[str]],
    claim_ids: list[str],
) -> list[int]:
    """Indices of all other claims, sorted by (descending Jaccard, claim id).

    Self is excluded. Ties on Jaccard break on lexicographic `claim_id`
    so the ranking is total and deterministic.
    """
    anchor_bag = token_bags[anchor_idx]
    scored = [
        (_jaccard(anchor_bag, token_bags[j]), claim_ids[j], j)
        for j in range(len(token_bags))
        if j != anchor_idx
    ]
    # Sort descending on Jaccard; ascending on claim_id as the tiebreak.
    scored.sort(key=lambda triple: (-triple[0], triple[1]))
    return [j for _, _, j in scored]


def _build_edge(
    *,
    src_id: str,
    dst_id: str,
    edge_type: str,
    confidence: float,
    premise: str,
    hypothesis: str,
) -> TypedEdge:
    """Materialise a `TypedEdge` with synthetic premise / hypothesis spans.

    The Tier-2 NLI inferer does not own a chunk-anchored span for the
    rendered claim text (the span lives on the original `Claim` row
    in the storage layer), so we synthesise a pseudo-`Span` per
    endpoint carrying the rendered text. The `chunk_id` is a stable,
    inferer-prefixed pseudo id so the trace renderer can recognise
    these citations as Tier-2 NLI provenance markers.
    """
    citations = [
        Span(
            chunk_id=f"tier2-nli:{src_id}",
            char_start=0,
            char_end=len(premise),
            text=premise,
        ),
        Span(
            chunk_id=f"tier2-nli:{dst_id}",
            char_start=0,
            char_end=len(hypothesis),
            text=hypothesis,
        ),
    ]
    # mypy-narrowing helper — `edge_type` is a string literal at every
    # call site below but the function signature accepts plain `str`
    # so we can keep one builder for both edge labels.
    if edge_type == "entails":
        return TypedEdge(
            src_id=src_id,
            dst_id=dst_id,
            type="entails",
            confidence=confidence,
            raw_score=confidence,
            citations=citations,
            source="nli",
            paraphrase_votes=None,
        )
    if edge_type == "contradicts":
        return TypedEdge(
            src_id=src_id,
            dst_id=dst_id,
            type="contradicts",
            confidence=confidence,
            raw_score=confidence,
            citations=citations,
            source="nli",
            paraphrase_votes=None,
        )
    raise ValueError(f"unsupported edge_type {edge_type!r}")


__all__ = [
    "DEFAULT_K_CANDIDATES",
    "NLI_CONTRADICT_THRESHOLD",
    "NLI_ENTAIL_THRESHOLD",
    "NLIScorer",
    "Tier2NLIConfig",
    "Tier2NLIEdgeInferer",
    "Tier2NLIExtraction",
    "claim_id",
    "render_claim_text",
]
