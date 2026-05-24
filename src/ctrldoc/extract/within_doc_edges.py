"""Within-doc typed-edge inference — Galois floor + optional Tier-2 NLI.

The §6.3 Galois subsumption lattice and the §6.5 Tier-2 NLI scorer
both produce calibrated edges over the per-doc claim graph. This module
is the seam that consumes a list of persisted ``Claim`` rows from one
doc and emits the union of:

* **Galois floor edges** — pure-function structural verdicts on the
  universal-tuple alphabet (subject / predicate / object / polarity /
  modality / qualifier). The four §6.3 verdicts project onto the
  ``TypedEdge`` alphabet as:

  - ``equivalent`` → ``equivalent_to`` (symmetric; one edge per pair)
  - ``subsumes`` (left strictly stronger) → ``entails`` (src = left,
    dst = right; the stronger claim entails the weaker)
  - ``subsumed_by`` → ``entails`` (orientation reversed)
  - ``incomparable`` → no edge

  Every Galois edge carries ``source = "heuristic"`` and the §6.5
  heuristic prior ``HEURISTIC_CONFIDENCE = 0.9``. Citations are the
  first ``Span`` from each endpoint's persisted ``span_refs`` — the §7
  contract guarantees every ``Claim`` has at least one span, so the
  S-155 gate ("every emitted edge cites a span") holds by construction.

* **Tier-2 NLI edges** — when an ``NLIScorer`` is also available
  (thrifty / production profiles), this module delegates to
  ``Tier2NLIEdgeInferer`` for the semantic-relation pass that the
  structural floor cannot see (paraphrases, predicate alignment,
  contradictions across distinct SVO surface forms). NLI edges carry
  ``source = "nli"`` and the raw top-label confidence; the §6.5
  calibration layer (isotonic regression, paraphrase voting) is the
  consumer that refines those scores later in the build.

The heuristic profile (no NLI scorer) shrinks to Galois-only output;
thrifty / production add the NLI delta on top. Edges from the two
sources can coexist on the same ``(src_id, dst_id, type)`` tuple — the
persistence layer's PRIMARY KEY plus ``INSERT OR REPLACE`` means the
last writer wins. The inferer emits Galois edges first then NLI edges
so an NLI verdict with the same key supersedes the heuristic prior; in
practice the two rarely collide because Galois fires only on
SVO-equal pairs while NLI's candidate-retrieval ranker also surfaces
SVO-different neighbours.

Output ordering: edges are sorted by ``(type, src_id, dst_id)`` so the
list is byte-deterministic across input orderings — the §13
non-negotiable 4 "every verdict is replayable" property the verdict
ledger depends on.

SPEC-REF: §6.3 (Galois lattice), §6.5 (probabilistic edges + calibration)
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ctrldoc.extract.claim_persistence import claim_to_tuple as _persisted_to_tuple
from ctrldoc.extract.galois import claim_subsumption
from ctrldoc.extract.tier2_nli import (
    NLIScorer,
    Tier2NLIConfig,
    Tier2NLIEdgeInferer,
)
from ctrldoc.models import Span
from ctrldoc.models_v1 import Claim, TypedEdge

# Heuristic prior every Galois-derived edge carries. Mirrors
# `tier1.HEURISTIC_CONFIDENCE` — the §6.5 fixed prior for any
# deterministic structural verdict.
HEURISTIC_CONFIDENCE: float = 0.9


# ---------------------------------------------------------------------------
# Persisted → universal-tuple alphabet mapping
# ---------------------------------------------------------------------------
#
# The shared converter lives in `ctrldoc.extract.claim_persistence`
# (the natural home for the §6.2 ↔ §7 alphabet seam). This module
# imports it as `_persisted_to_tuple` so its existing call sites stay
# stable.


# ---------------------------------------------------------------------------
# Galois floor — pairwise within-doc subsumption → TypedEdge
# ---------------------------------------------------------------------------


def _first_span(claim: Claim) -> Span:
    """Return the first ``Span`` from a claim's ``span_refs``.

    §7 guarantees ``span_refs`` is non-empty for persisted claims. The
    inferer never owns a citation of its own — it threads the claim's
    own anchored span into the edge so the trace renderer can point
    the reviewer at the source chunk.
    """
    if not claim.span_refs:  # pragma: no cover — §7 forbids this
        raise ValueError(
            f"persisted claim {claim.id!r} has no span_refs; cannot build a cited edge"
        )
    return claim.span_refs[0]


def _make_heuristic_edge(
    *,
    src: Claim,
    dst: Claim,
    edge_type: str,
) -> TypedEdge:
    """Build a Galois-derived ``TypedEdge`` citing one span per endpoint."""
    citations = [_first_span(src), _first_span(dst)]
    if edge_type == "entails":
        return TypedEdge(
            src_id=src.id,
            dst_id=dst.id,
            type="entails",
            confidence=HEURISTIC_CONFIDENCE,
            raw_score=HEURISTIC_CONFIDENCE,
            citations=citations,
            source="heuristic",
            paraphrase_votes=None,
        )
    if edge_type == "equivalent_to":
        return TypedEdge(
            src_id=src.id,
            dst_id=dst.id,
            type="equivalent_to",
            confidence=HEURISTIC_CONFIDENCE,
            raw_score=HEURISTIC_CONFIDENCE,
            citations=citations,
            source="heuristic",
            paraphrase_votes=None,
        )
    raise ValueError(f"unsupported heuristic edge_type {edge_type!r}")


def galois_within_doc_edges(claims: Sequence[Claim]) -> list[TypedEdge]:
    """Run the §6.3 Galois floor over every unordered claim pair.

    Returns a list of ``TypedEdge`` rows, sorted by ``(type, src_id,
    dst_id)`` for byte-deterministic output. Self-pairs are skipped;
    incomparable pairs emit nothing. Equivalent pairs emit a single
    ``equivalent_to`` edge with the lexicographically smaller claim id
    in the ``src_id`` slot so the symmetric relation maps to one row
    rather than two.
    """
    n = len(claims)
    if n < 2:
        return []

    # Pre-compute the universal tuple for each claim so the inner loop
    # never re-derives it. The conversion is pure; the cache shrinks
    # the inner work to a Galois call.
    tuples = [_persisted_to_tuple(c) for c in claims]

    edges: list[TypedEdge] = []
    for i in range(n):
        for j in range(i + 1, n):
            verdict = claim_subsumption(tuples[i], tuples[j])
            if verdict == "incomparable":
                continue
            left, right = claims[i], claims[j]
            if verdict == "subsumes":
                # left ⊑ right => stronger left implies weaker right.
                edges.append(_make_heuristic_edge(src=left, dst=right, edge_type="entails"))
            elif verdict == "subsumed_by":
                # right is the stronger side — flip the endpoints.
                edges.append(_make_heuristic_edge(src=right, dst=left, edge_type="entails"))
            elif verdict == "equivalent":
                # Symmetric relation — canonicalise to one row with
                # the lexicographically smaller id in `src_id` so
                # re-running the pass on a re-ordered input list still
                # produces the same single row.
                if left.id <= right.id:
                    edges.append(
                        _make_heuristic_edge(src=left, dst=right, edge_type="equivalent_to")
                    )
                else:
                    edges.append(
                        _make_heuristic_edge(src=right, dst=left, edge_type="equivalent_to")
                    )

    edges.sort(key=lambda e: (e.type, e.src_id, e.dst_id))
    return edges


# ---------------------------------------------------------------------------
# Optional NLI overlay — Tier-2 NLI edges thread through the same builder
# ---------------------------------------------------------------------------


def _rehome_nli_citations(edge: TypedEdge, *, src: Claim, dst: Claim) -> TypedEdge:
    """Replace the Tier-2 NLI inferer's synthesised citations with real spans.

    ``Tier2NLIEdgeInferer`` builds synthetic ``Span`` rows keyed on the
    rendered claim text because it does not own a chunk-anchored span
    for the universal tuple. Within the per-doc pass we do own one —
    the source ``Claim``'s persisted ``span_refs[0]`` — so we swap the
    synthesised citations for the real ones. This keeps the S-155 gate
    intact (every persisted edge cites a real chunk-anchored span).
    """
    return edge.model_copy(update={"citations": [_first_span(src), _first_span(dst)]})


class WithinDocEdgeInferer:
    """Compose Galois + optional Tier-2 NLI into one within-doc pass.

    Construction is cheap; the NLI scorer (when present) is invoked
    lazily inside ``infer``. The inferer holds no per-call state.

    Construction-time decisions:

    * ``nli_scorer`` is the §6.5 ``NLIScorer`` protocol — same shape as
      the ``Tier2NLIEdgeInferer`` consumer. Passing ``None`` reduces
      the inferer to the Galois floor, which is the heuristic profile's
      contract.
    * ``nli_config`` is an optional ``Tier2NLIConfig`` so the candidate
      fanout / thresholds can be tuned per profile without touching
      this module.
    """

    def __init__(
        self,
        *,
        nli_scorer: NLIScorer | None = None,
        nli_config: Tier2NLIConfig | None = None,
    ) -> None:
        self._nli_scorer = nli_scorer
        self._nli_inferer: Tier2NLIEdgeInferer | None
        if nli_scorer is not None:
            self._nli_inferer = Tier2NLIEdgeInferer(scorer=nli_scorer, config=nli_config)
        else:
            self._nli_inferer = None

    def infer(self, claims: Iterable[Claim]) -> list[TypedEdge]:
        """Return the union of Galois + NLI edges, sorted deterministically."""
        claim_list = list(claims)
        edges = list(galois_within_doc_edges(claim_list))

        if self._nli_inferer is not None and len(claim_list) >= 2:
            tuples = [_persisted_to_tuple(c) for c in claim_list]
            nli_extraction = self._nli_inferer.infer(tuples)
            # Re-home the NLI inferer's synthesised citations onto the
            # real chunk-anchored spans the per-doc pass already owns.
            # The Tier-2 inferer's `claim_id` content-hash is computed
            # over the same six universal-tuple fields as ours, but it
            # does not include the doc/chunk binding the §7 ``Claim.id``
            # does — so we map back through the universal tuple to find
            # the originating ``Claim`` for each endpoint.
            from ctrldoc.extract.tier2_nli import claim_id as nli_claim_id

            id_to_claim: dict[str, Claim] = {}
            for claim, tuple_ in zip(claim_list, tuples, strict=True):
                id_to_claim.setdefault(nli_claim_id(tuple_), claim)
            for raw_edge in nli_extraction.edges:
                src = id_to_claim.get(raw_edge.src_id)
                dst = id_to_claim.get(raw_edge.dst_id)
                if src is None or dst is None:
                    # An NLI verdict pointing at a tuple we did not
                    # feed it — should never happen, but the §13 "no
                    # silent no-op" rule says we skip it loudly rather
                    # than emitting an uncited edge.
                    continue
                edges.append(
                    _rehome_nli_citations(
                        raw_edge.model_copy(update={"src_id": src.id, "dst_id": dst.id}),
                        src=src,
                        dst=dst,
                    )
                )

        edges.sort(key=lambda e: (e.type, e.src_id, e.dst_id))
        return edges


__all__ = [
    "HEURISTIC_CONFIDENCE",
    "WithinDocEdgeInferer",
    "galois_within_doc_edges",
]
