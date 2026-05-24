"""Within-doc typed-edge inference via Galois subsumption + optional Tier-2 NLI.

The Galois floor (§6.3) decides every within-doc claim pair on the
universal-tuple alphabet alone — `equivalent`, `subsumes`,
`subsumed_by`, `incomparable`. The first three project onto the
`TypedEdge` alphabet as `equivalent_to` (symmetric) and `entails`
(asymmetric, oriented from the stronger side to the weaker). The
`incomparable` verdict emits nothing — at the structural floor there
is no entailment to record.

When an `NLIScorer` is also available (thrifty / production profiles
ship one; the heuristic profile does not), the `Tier2NLIEdgeInferer`
adds `entails` / `contradicts` edges from semantic relations the
structural floor cannot see — paraphrases, polysemy, predicate
alignment. The two sources produce edges with different `source`
provenance tags (`"heuristic"` for Galois, `"nli"` for NLI), and
duplicates on the same `(src_id, dst_id, type)` key are deduped by
the persistence layer's PRIMARY KEY contract — last write wins.

Gate: every emitted edge carries at least one span in its `citations`
list, because every persisted `Claim` has `span_refs` (§7), and the
within-doc inferer threads at least one of each endpoint's spans
into the edge's `citations`.

SPEC-REF: §6.3, §6.5
"""

from __future__ import annotations

import pytest

from ctrldoc.eval.calibration import NLIScore
from ctrldoc.eval.claim_extraction import ClaimTuple as ClaimTupleType
from ctrldoc.extract.within_doc_edges import (
    WithinDocEdgeInferer,
    galois_within_doc_edges,
)
from ctrldoc.models import Span
from ctrldoc.models_v1 import Claim

pytestmark = [pytest.mark.family_referential_integrity]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _claim(
    *,
    claim_id: str,
    doc_id: str = "doc-a",
    subject: str = "system",
    predicate: str = "validate",
    obj: str = "inputs",
    polarity: str = "+",
    modality: str | None = "assert",
    qualifier: dict[str, object] | None = None,
    chunk_id: str = "chunk-1",
    section_id: str = "sec-1",
) -> Claim:
    text_body = f"{subject} {predicate} {obj}".strip()
    span = Span(
        chunk_id=chunk_id,
        char_start=0,
        char_end=len(text_body),
        text=text_body,
    )
    return Claim(
        id=claim_id,
        doc_id=doc_id,
        text=text_body,
        subject=subject,
        predicate=predicate,
        object=obj,
        polarity=polarity,  # type: ignore[arg-type]
        modality=modality,  # type: ignore[arg-type]
        qualifier=qualifier or {},
        span_refs=[span],
        section_id=section_id,
        concept_ids=[],
        typed_slots={},
        confidence=1.0,
    )


class _DictScorer:
    """`NLIScorer` keyed on `(premise, hypothesis)`; default neutral."""

    def __init__(self, table: dict[tuple[str, str], NLIScore]) -> None:
        self._table = table
        self.calls: list[tuple[str, str]] = []

    def score(self, *, premise: str, hypothesis: str) -> NLIScore:
        self.calls.append((premise, hypothesis))
        if (premise, hypothesis) in self._table:
            return self._table[(premise, hypothesis)]
        return NLIScore(entailment=0.20, contradiction=0.20, neutral=0.60)


# ---------------------------------------------------------------------------
# Galois floor — emits `entails` / `equivalent_to` from structural subsumption
# ---------------------------------------------------------------------------


def test_galois_emits_entails_when_left_subsumes_right() -> None:
    """`obligatory` subsumes `recommended` on the deontic chain → `entails`."""
    strong = _claim(claim_id="must-validate", modality="must")
    weak = _claim(claim_id="should-validate", modality="should")

    edges = galois_within_doc_edges([strong, weak])

    entails = [e for e in edges if e.type == "entails"]
    assert any(
        e.src_id == "must-validate" and e.dst_id == "should-validate" for e in entails
    ), f"expected an entails(must -> should) edge; got {[(e.type, e.src_id, e.dst_id) for e in edges]}"
    # Every emitted edge carries the heuristic provenance.
    for edge in edges:
        assert edge.source == "heuristic"


def test_galois_emits_equivalent_to_when_pair_is_equivalent() -> None:
    """Same SVO + same modality + same polarity → `equivalent_to` (one direction)."""
    a = _claim(claim_id="a", chunk_id="chunk-1")
    b = _claim(claim_id="b", chunk_id="chunk-2")  # same logical content, different chunk

    edges = galois_within_doc_edges([a, b])

    equiv = [e for e in edges if e.type == "equivalent_to"]
    assert equiv, "expected at least one equivalent_to edge for equivalent claims"
    # The edge endpoints cover both claim ids (symmetric relation).
    endpoint_pairs = {(e.src_id, e.dst_id) for e in equiv}
    assert ("a", "b") in endpoint_pairs or ("b", "a") in endpoint_pairs


def test_galois_emits_no_edge_for_incomparable_pairs() -> None:
    """Different SVO → `incomparable` → no edge."""
    a = _claim(claim_id="a", subject="alice", predicate="reads", obj="book")
    b = _claim(claim_id="b", subject="bob", predicate="writes", obj="poem")

    edges = galois_within_doc_edges([a, b])
    assert edges == [], f"expected zero edges; got {edges}"


def test_galois_emits_no_self_edge() -> None:
    """A claim never emits an edge with itself as both endpoints."""
    a = _claim(claim_id="a")
    edges = galois_within_doc_edges([a])
    assert edges == []


# ---------------------------------------------------------------------------
# Gate: every emitted edge carries at least one citation span
# ---------------------------------------------------------------------------


def test_every_galois_edge_carries_at_least_one_citation_span() -> None:
    strong = _claim(claim_id="must-validate", modality="must", chunk_id="chunk-1")
    weak = _claim(claim_id="should-validate", modality="should", chunk_id="chunk-2")

    edges = galois_within_doc_edges([strong, weak])

    assert edges, "fixture should produce at least one edge"
    for edge in edges:
        assert len(edge.citations) >= 1, f"edge {edge} has no citations"
        for span in edge.citations:
            # Citation spans must be real chunk-anchored Span objects,
            # not synthesised placeholders — the §7 contract that
            # Claim.span_refs is non-empty makes this enforceable.
            assert isinstance(span, Span)
            assert span.chunk_id, "citation span has no chunk_id"


# ---------------------------------------------------------------------------
# Within-doc inferer — Galois + optional NLI, sorted, deduped output
# ---------------------------------------------------------------------------


def test_within_doc_inferer_galois_only_when_no_scorer() -> None:
    """Heuristic profile (no NLI scorer) gives Galois-only output."""
    strong = _claim(claim_id="must-x", modality="must")
    weak = _claim(claim_id="should-x", modality="should")

    inferer = WithinDocEdgeInferer(nli_scorer=None)
    edges = inferer.infer([strong, weak])

    assert all(e.source == "heuristic" for e in edges)
    assert any(e.type == "entails" for e in edges)


def test_within_doc_inferer_adds_nli_edges_when_scorer_present() -> None:
    """Thrifty / production profile: NLI scorer adds semantic edges."""
    # Two claims with disjoint SVO (Galois floor returns incomparable) but
    # an NLI scorer confidently asserts entailment.
    a = _claim(claim_id="claim-a", subject="alice", predicate="reads", obj="book")
    b = _claim(claim_id="claim-b", subject="bob", predicate="reads", obj="novel")
    # Use the rendered claim text the inferer feeds the scorer.
    from ctrldoc.extract.tier2_nli import render_claim_text

    a_text = render_claim_text(_to_tuple(a))
    b_text = render_claim_text(_to_tuple(b))
    table = {
        (a_text, b_text): NLIScore(entailment=0.85, contradiction=0.05, neutral=0.10),
        (b_text, a_text): NLIScore(entailment=0.10, contradiction=0.05, neutral=0.85),
    }
    scorer = _DictScorer(table)
    inferer = WithinDocEdgeInferer(nli_scorer=scorer)

    edges = inferer.infer([a, b])

    nli_edges = [e for e in edges if e.source == "nli"]
    assert nli_edges, f"expected NLI edge; got sources {[e.source for e in edges]}"
    assert any(e.type == "entails" for e in nli_edges)
    # The scorer must have been called at least once.
    assert scorer.calls, "scorer was not invoked"


def test_within_doc_inferer_output_is_sorted_and_deterministic() -> None:
    """Edge list sorts by `(type, src_id, dst_id)` for byte-deterministic output."""
    strong = _claim(claim_id="z-must", modality="must")
    weak = _claim(claim_id="a-should", modality="should")

    inferer = WithinDocEdgeInferer(nli_scorer=None)
    edges_a = inferer.infer([strong, weak])
    edges_b = inferer.infer([weak, strong])

    # Two different input orderings produce the same sorted output.
    assert [(e.type, e.src_id, e.dst_id) for e in edges_a] == [
        (e.type, e.src_id, e.dst_id) for e in edges_b
    ]
    # Sorted property: tuples non-decreasing.
    keys = [(e.type, e.src_id, e.dst_id) for e in edges_a]
    assert keys == sorted(keys)


def test_within_doc_inferer_empty_and_single_claim_input_is_empty() -> None:
    inferer = WithinDocEdgeInferer(nli_scorer=None)
    assert inferer.infer([]) == []
    assert inferer.infer([_claim(claim_id="solo")]) == []


# ---------------------------------------------------------------------------
# Helper to convert a persisted `Claim` back to the universal `ClaimTuple`
# the §6.3 / Tier-2 NLI APIs consume. Mirrors the inverse of
# `claim_persistence.claim_from_tuple`. Only used in tests.
# ---------------------------------------------------------------------------


def _to_tuple(claim: Claim) -> ClaimTupleType:
    polarity = "affirmative" if claim.polarity == "+" else "negative"
    modality_map = {
        "assert": "asserted",
        "must": "obligatory",
        "should": "recommended",
        "may": "permitted",
        "neg": "prohibited",
        "shall": "obligatory",
    }
    modality = modality_map.get(claim.modality or "assert", "asserted")
    qualifier_text = ""
    if isinstance(claim.qualifier, dict):
        raw = claim.qualifier.get("text")
        if isinstance(raw, str):
            qualifier_text = raw
    return ClaimTupleType(
        subject=claim.subject or "",
        predicate=claim.predicate,
        object=claim.object or "",
        polarity=polarity,  # type: ignore[arg-type]
        modality=modality,  # type: ignore[arg-type]
        qualifier=qualifier_text,
    )
