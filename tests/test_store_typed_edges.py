"""SQLiteStore + InMemoryStore typed-edge CRUD.

The §8 schema provisioned `typed_edges` at S-125 but the v1 substrate
never grew the matching `append_typed_edge` / `iter_typed_edges` /
`iter_typed_edges_for_doc` methods. S-155 wires the within-doc edge
inferer into ingest, which needs these methods to persist Galois
floor + Tier-2 NLI verdicts.

Contract:

* `append_typed_edge` is idempotent by `(src_id, dst_id, type)` — the
  table's composite PRIMARY KEY plus `INSERT OR REPLACE` guarantee
  that re-running ingest never produces duplicates.
* `iter_typed_edges` yields every persisted edge in `(type, src_id,
  dst_id)` order so the read order is byte-deterministic across runs.
* `iter_typed_edges_for_doc(doc_id)` yields the subset whose `src_id`
  or `dst_id` belongs to the given doc (joined through `claims`).

SPEC-REF: §6.3, §6.5, §8
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ctrldoc.models import Span
from ctrldoc.models_v1 import Claim, TypedEdge
from ctrldoc.store.memory import InMemoryStore
from ctrldoc.store.sqlite import SQLiteStore

pytestmark = [pytest.mark.family_referential_integrity]


def _edge(
    *,
    src: str,
    dst: str,
    edge_type: str = "entails",
    confidence: float = 0.9,
    source: str = "heuristic",
) -> TypedEdge:
    span = Span(chunk_id="chunk-1", char_start=0, char_end=4, text="abcd")
    return TypedEdge(
        src_id=src,
        dst_id=dst,
        type=edge_type,  # type: ignore[arg-type]
        confidence=confidence,
        raw_score=confidence,
        citations=[span],
        source=source,  # type: ignore[arg-type]
        paraphrase_votes=None,
    )


def _claim(
    *,
    claim_id: str,
    doc_id: str,
    chunk_id: str = "chunk-1",
) -> Claim:
    span = Span(chunk_id=chunk_id, char_start=0, char_end=4, text="abcd")
    return Claim(
        id=claim_id,
        doc_id=doc_id,
        text="abcd",
        subject="s",
        predicate="p",
        object="o",
        polarity="+",
        modality="assert",
        qualifier={},
        span_refs=[span],
        section_id="sec-1",
        concept_ids=[],
        typed_slots={},
        confidence=1.0,
    )


# ---------------------------------------------------------------------------
# Core CRUD parity — both backends must agree on the contract
# ---------------------------------------------------------------------------


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path):  # type: ignore[no-untyped-def]
    if request.param == "memory":
        yield InMemoryStore()
    else:
        store = SQLiteStore(tmp_path / "t.db")
        try:
            yield store
        finally:
            store.close()


def test_append_typed_edge_then_iter_returns_it(store) -> None:  # type: ignore[no-untyped-def]
    e = _edge(src="a", dst="b")
    store.append_typed_edge(e)
    rows = list(store.iter_typed_edges())
    assert rows == [e]


def test_append_typed_edge_is_idempotent_by_primary_key(store) -> None:  # type: ignore[no-untyped-def]
    e = _edge(src="a", dst="b", confidence=0.5)
    e_updated = _edge(src="a", dst="b", confidence=0.99)
    store.append_typed_edge(e)
    store.append_typed_edge(e_updated)
    rows = list(store.iter_typed_edges())
    assert len(rows) == 1
    assert rows[0].confidence == pytest.approx(0.99)


def test_iter_typed_edges_is_sorted_by_type_src_dst(store) -> None:  # type: ignore[no-untyped-def]
    # Insert in a deliberately scrambled order.
    inserts = [
        _edge(src="z", dst="a", edge_type="entails"),
        _edge(src="a", dst="z", edge_type="equivalent_to"),
        _edge(src="a", dst="b", edge_type="entails"),
    ]
    for e in inserts:
        store.append_typed_edge(e)
    rows = list(store.iter_typed_edges())
    keys = [(r.type, r.src_id, r.dst_id) for r in rows]
    assert keys == sorted(keys)


def test_distinct_edge_types_on_same_endpoints_are_separate_rows(store) -> None:  # type: ignore[no-untyped-def]
    """The PK is `(src, dst, type)` — two types on the same pair coexist."""
    a = _edge(src="a", dst="b", edge_type="entails")
    b = _edge(src="a", dst="b", edge_type="equivalent_to")
    store.append_typed_edge(a)
    store.append_typed_edge(b)
    rows = list(store.iter_typed_edges())
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# Per-doc filter — joins through `claims.doc_id` for the SQLite backend;
# the in-memory implementation uses the parallel `_claims` dict.
# ---------------------------------------------------------------------------


def test_iter_typed_edges_for_doc_filters_by_endpoint_doc(store) -> None:  # type: ignore[no-untyped-def]
    # Two docs; each has one claim. One edge connects within doc-a;
    # another connects across the two docs.
    store.append_claim(_claim(claim_id="a1", doc_id="doc-a"))
    store.append_claim(_claim(claim_id="a2", doc_id="doc-a"))
    store.append_claim(_claim(claim_id="b1", doc_id="doc-b"))

    intra_a = _edge(src="a1", dst="a2")
    cross = _edge(src="a1", dst="b1")
    intra_b_only = _edge(src="b1", dst="b1", edge_type="equivalent_to")
    store.append_typed_edge(intra_a)
    store.append_typed_edge(cross)
    store.append_typed_edge(intra_b_only)

    rows_a = list(store.iter_typed_edges_for_doc("doc-a"))
    # `doc-a` sees both edges that touch one of its claims.
    keys = {(r.src_id, r.dst_id, r.type) for r in rows_a}
    assert ("a1", "a2", "entails") in keys
    assert ("a1", "b1", "entails") in keys

    rows_b = list(store.iter_typed_edges_for_doc("doc-b"))
    keys_b = {(r.src_id, r.dst_id, r.type) for r in rows_b}
    assert ("a1", "b1", "entails") in keys_b
    assert ("b1", "b1", "equivalent_to") in keys_b
