"""Contract tests for the retrieval-DSL executor.

The executor walks a `Plan` step-by-step against an injected Store,
VectorIndex, BM25Index, and Embedder. Each step produces a typed
`StepResult` carrying chunk_ids (and/or entity_ids for Neighbors).
Fusion across views is the next slice (S-042).

SPEC-REF: §4.3 (retrieval executor)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ctrldoc.ingest.embedder import HashEmbedder
from ctrldoc.models import Chunk, Entity, Section
from ctrldoc.retrieval.dsl import Expand, Neighbors, Plan, Search
from ctrldoc.retrieval.executor import PlanExecutor, StepResult
from ctrldoc.store.bm25 import TantivyBM25Index
from ctrldoc.store.memory import InMemoryStore
from ctrldoc.store.vectors import InMemoryVectorIndex


def _chunk(chunk_id: str, *, section_id: str = "sec/a", text: str = "body") -> Chunk:
    return Chunk(
        id=chunk_id,
        section_id=section_id,
        text=text,
        token_count=2,
        char_start=0,
        char_end=len(text),
        embedding_id=f"emb/{chunk_id}",
    )


@pytest.fixture
def kit(tmp_path: Path):  # type: ignore[no-untyped-def]
    store = InMemoryStore()
    vector_index = InMemoryVectorIndex(dimension=16)
    bm25_index = TantivyBM25Index(path=tmp_path / "bm25")
    embedder = HashEmbedder(dimension=16)
    return {
        "store": store,
        "vector_index": vector_index,
        "bm25_index": bm25_index,
        "embedder": embedder,
        "executor": PlanExecutor(
            store=store,
            vector_index=vector_index,
            bm25_index=bm25_index,
            embedder=embedder,
        ),
    }


def _seed_corpus(kit: dict) -> None:
    chunks = [
        _chunk("c1", section_id="sec/a", text="alpha cosmos hello"),
        _chunk("c2", section_id="sec/a", text="beta hello world"),
        _chunk("c3", section_id="sec/b", text="gamma quantum"),
        _chunk("c4", section_id="sec/b", text="delta black hole"),
    ]
    kit["store"].add_chunks(chunks)
    kit["store"].add_sections(
        [
            Section(id="sec/a", parent_id=None, title="A", summary="", chunk_ids=["c1", "c2"]),
            Section(id="sec/b", parent_id=None, title="B", summary="", chunk_ids=["c3", "c4"]),
        ]
    )
    for chunk in chunks:
        kit["vector_index"].add(chunk.id, kit["embedder"].embed(chunk.text))
        kit["bm25_index"].add(chunk.id, chunk.text)


# --- empty plan ---


def test_empty_plan_returns_no_results(kit: dict) -> None:
    assert kit["executor"].execute(Plan(steps=[])) == []


# --- Search ---


def test_search_dense_returns_top_k_chunks(kit: dict) -> None:
    _seed_corpus(kit)
    result = kit["executor"].execute(Plan(steps=[Search(query="hello world", view="dense", k=2)]))
    assert len(result) == 1
    step = result[0]
    assert step.op == "search"
    assert len(step.chunk_ids) == 2


def test_search_lexical_finds_keyword(kit: dict) -> None:
    _seed_corpus(kit)
    result = kit["executor"].execute(Plan(steps=[Search(query="cosmos", view="lexical", k=5)]))
    assert "c1" in result[0].chunk_ids


def test_search_lexical_no_match_returns_empty(kit: dict) -> None:
    _seed_corpus(kit)
    result = kit["executor"].execute(
        Plan(steps=[Search(query="zzz_nonexistent", view="lexical", k=5)])
    )
    assert result[0].chunk_ids == []


def test_search_entity_returns_mention_chunks(kit: dict) -> None:
    _seed_corpus(kit)
    kit["store"].add_entities(
        [
            Entity(
                id="ent/concept/cosmos",
                aliases=["cosmos"],
                type="concept",
                mention_chunk_ids=["c1", "c2"],
            )
        ]
    )
    result = kit["executor"].execute(
        Plan(steps=[Search(query="ent/concept/cosmos", view="entity", k=5)])
    )
    assert set(result[0].chunk_ids) == {"c1", "c2"}


def test_search_entity_unknown_returns_empty(kit: dict) -> None:
    _seed_corpus(kit)
    result = kit["executor"].execute(
        Plan(steps=[Search(query="ent/concept/missing", view="entity", k=5)])
    )
    assert result[0].chunk_ids == []


def test_search_entity_respects_k(kit: dict) -> None:
    _seed_corpus(kit)
    kit["store"].add_entities(
        [
            Entity(
                id="ent/c/x",
                aliases=["x"],
                type="concept",
                mention_chunk_ids=["c1", "c2", "c3"],
            )
        ]
    )
    result = kit["executor"].execute(Plan(steps=[Search(query="ent/c/x", view="entity", k=2)]))
    assert len(result[0].chunk_ids) == 2


# --- Expand ---


def test_expand_returns_chunks_of_section(kit: dict) -> None:
    _seed_corpus(kit)
    result = kit["executor"].execute(Plan(steps=[Expand(section_id="sec/a")]))
    assert set(result[0].chunk_ids) == {"c1", "c2"}


def test_expand_unknown_section_returns_empty(kit: dict) -> None:
    _seed_corpus(kit)
    result = kit["executor"].execute(Plan(steps=[Expand(section_id="sec/missing")]))
    assert result[0].chunk_ids == []


# --- Neighbors ---


def test_neighbors_one_hop_excludes_self(kit: dict) -> None:
    kit["store"].add_entities(
        [
            Entity(id="ent/a", aliases=["A"], type="x", mention_chunk_ids=["c1"]),
            Entity(id="ent/b", aliases=["B"], type="x", mention_chunk_ids=["c1"]),
        ]
    )
    result = kit["executor"].execute(Plan(steps=[Neighbors(entity_id="ent/a")]))
    assert result[0].entity_ids == ["ent/b"]
    assert result[0].chunk_ids == []


def test_neighbors_two_hops_reaches_further(kit: dict) -> None:
    # A and B share c1; B and C share c2; A and C are not directly connected.
    kit["store"].add_entities(
        [
            Entity(id="ent/a", aliases=["A"], type="x", mention_chunk_ids=["c1"]),
            Entity(id="ent/b", aliases=["B"], type="x", mention_chunk_ids=["c1", "c2"]),
            Entity(id="ent/c", aliases=["C"], type="x", mention_chunk_ids=["c2"]),
        ]
    )
    one_hop = kit["executor"].execute(Plan(steps=[Neighbors(entity_id="ent/a", hops=1)]))[0]
    assert set(one_hop.entity_ids) == {"ent/b"}
    two_hop = kit["executor"].execute(Plan(steps=[Neighbors(entity_id="ent/a", hops=2)]))[0]
    assert set(two_hop.entity_ids) == {"ent/b", "ent/c"}


def test_neighbors_unknown_entity_returns_empty(kit: dict) -> None:
    result = kit["executor"].execute(Plan(steps=[Neighbors(entity_id="ent/missing")]))
    assert result[0].entity_ids == []


# --- multi-step plan ---


def test_executor_returns_one_step_result_per_step_in_order(kit: dict) -> None:
    _seed_corpus(kit)
    plan = Plan(
        steps=[
            Search(query="cosmos", view="lexical", k=5),
            Expand(section_id="sec/b"),
        ]
    )
    results = kit["executor"].execute(plan)
    assert [r.op for r in results] == ["search", "expand"]
    assert isinstance(results[0], StepResult)
    assert set(results[1].chunk_ids) == {"c3", "c4"}
