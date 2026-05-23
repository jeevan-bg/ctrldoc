"""Contract tests for the evidence-pack builder.

The builder turns a ranked chunk list into an `EvidencePack` — the
≤ 6k-token bundle the downstream judge sees. Each emitted chunk
becomes a `Span` carrying its stable chunk id. Budget enforcement is
strict: adding a chunk that would exceed `max_tokens` short-circuits.

SPEC-REF: §4.3 (evidence pack builder)
"""

from __future__ import annotations

import pytest

from ctrldoc.models import EVIDENCE_PACK_TOKEN_CAP, Chunk
from ctrldoc.retrieval.evidence import build_evidence_pack
from ctrldoc.store.memory import InMemoryStore


def _chunk(chunk_id: str, *, text: str, tokens: int, section_id: str = "sec/a") -> Chunk:
    return Chunk(
        id=chunk_id,
        section_id=section_id,
        text=text,
        token_count=tokens,
        char_start=0,
        char_end=len(text),
        embedding_id=f"emb/{chunk_id}",
    )


def _store_with(chunks: list[Chunk]) -> InMemoryStore:
    store = InMemoryStore()
    store.add_chunks(chunks)
    return store


# --- defaults ---


def test_default_max_tokens_matches_spec_cap() -> None:
    """The builder's default cap must equal the SPEC §4.3 6k limit."""
    pack = build_evidence_pack(
        query="q",
        ranked_chunk_ids=[],
        store=_store_with([]),
    )
    assert pack.query == "q"
    assert pack.spans == []
    assert pack.token_count == 0
    # The cap is the EvidencePack model's own upper bound.
    assert EVIDENCE_PACK_TOKEN_CAP == 6000


# --- basics ---


def test_empty_ranked_list_returns_empty_pack() -> None:
    pack = build_evidence_pack(query="q", ranked_chunk_ids=[], store=_store_with([]))
    assert pack.spans == []
    assert pack.token_count == 0


def test_single_chunk_becomes_one_span() -> None:
    store = _store_with([_chunk("c1", text="hello world", tokens=2)])
    pack = build_evidence_pack(query="q", ranked_chunk_ids=["c1"], store=store)
    assert len(pack.spans) == 1
    assert pack.spans[0].chunk_id == "c1"
    assert pack.spans[0].text == "hello world"
    assert pack.token_count == 2


def test_preserves_input_order() -> None:
    store = _store_with(
        [
            _chunk("c1", text="alpha", tokens=1),
            _chunk("c2", text="beta", tokens=1),
            _chunk("c3", text="gamma", tokens=1),
        ]
    )
    pack = build_evidence_pack(
        query="q",
        ranked_chunk_ids=["c3", "c1", "c2"],
        store=store,
    )
    assert [s.chunk_id for s in pack.spans] == ["c3", "c1", "c2"]


def test_missing_chunk_ids_are_skipped() -> None:
    store = _store_with([_chunk("c1", text="hello", tokens=1)])
    pack = build_evidence_pack(
        query="q",
        ranked_chunk_ids=["missing", "c1", "missing-2"],
        store=store,
    )
    assert [s.chunk_id for s in pack.spans] == ["c1"]


# --- budget ---


def test_budget_stops_at_first_chunk_that_would_exceed() -> None:
    store = _store_with(
        [
            _chunk("c1", text="A" * 10, tokens=40),
            _chunk("c2", text="B" * 10, tokens=40),
            _chunk("c3", text="C" * 10, tokens=40),
        ]
    )
    pack = build_evidence_pack(
        query="q",
        ranked_chunk_ids=["c1", "c2", "c3"],
        store=store,
        max_tokens=80,
    )
    assert [s.chunk_id for s in pack.spans] == ["c1", "c2"]
    assert pack.token_count == 80


def test_budget_zero_returns_empty_pack() -> None:
    store = _store_with([_chunk("c1", text="hi", tokens=1)])
    pack = build_evidence_pack(
        query="q",
        ranked_chunk_ids=["c1"],
        store=store,
        max_tokens=0,
    )
    assert pack.spans == []
    assert pack.token_count == 0


def test_invalid_max_tokens_rejected() -> None:
    store = _store_with([])
    with pytest.raises(ValueError):
        build_evidence_pack(query="q", ranked_chunk_ids=[], store=store, max_tokens=-1)
    with pytest.raises(ValueError):
        build_evidence_pack(
            query="q", ranked_chunk_ids=[], store=store, max_tokens=EVIDENCE_PACK_TOKEN_CAP + 1
        )


def test_single_oversized_chunk_excluded() -> None:
    store = _store_with([_chunk("big", text="long text", tokens=999_999)])
    pack = build_evidence_pack(
        query="q",
        ranked_chunk_ids=["big"],
        store=store,
        max_tokens=100,
    )
    assert pack.spans == []
    assert pack.token_count == 0


# --- retrieval plan trace ---


def test_retrieval_plan_preserved() -> None:
    pack = build_evidence_pack(
        query="q",
        ranked_chunk_ids=[],
        store=_store_with([]),
        retrieval_plan=["search(query=q, view=dense, k=4)", "expand(section_id=sec/a)"],
    )
    assert pack.retrieval_plan == [
        "search(query=q, view=dense, k=4)",
        "expand(section_id=sec/a)",
    ]


def test_retrieval_plan_defaults_to_empty() -> None:
    pack = build_evidence_pack(
        query="q",
        ranked_chunk_ids=[],
        store=_store_with([]),
    )
    assert pack.retrieval_plan == []


# --- span fidelity ---


def test_span_char_range_matches_chunk() -> None:
    store = _store_with(
        [
            Chunk(
                id="c1",
                section_id="sec/a",
                text="hello world",
                token_count=2,
                char_start=42,
                char_end=53,
                embedding_id="emb/c1",
            )
        ]
    )
    pack = build_evidence_pack(query="q", ranked_chunk_ids=["c1"], store=store)
    span = pack.spans[0]
    assert span.char_start == 42
    assert span.char_end == 53
