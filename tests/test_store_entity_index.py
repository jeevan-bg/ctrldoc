"""Contract tests for the entity inverted-index lookup methods.

Every `Store` backend supports three lookups over the entity ↔ chunk
junction: chunks for an entity, entities for a chunk, and the
1-hop entity neighbourhood through shared chunks. The tests run
against both the in-memory reference and the persistent SQLite
backend so the two cannot drift.

SPEC-REF: §4.2 (entity index), §4.3 (`neighbors` DSL primitive)
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from ctrldoc.models import Entity
from ctrldoc.store import Store
from ctrldoc.store.memory import InMemoryStore
from ctrldoc.store.sqlite import SQLiteStore

StoreFactory = Callable[[], Iterator[Store]]


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Store]:
    backend = request.param
    if backend == "memory":
        yield InMemoryStore()
    else:
        with SQLiteStore(tmp_path / "ctrldoc.db") as s:
            yield s


def _entity(entity_id: str, chunk_ids: list[str], *, type_: str = "concept") -> Entity:
    return Entity(
        id=entity_id,
        aliases=[entity_id.upper()],
        type=type_,
        mention_chunk_ids=chunk_ids,
    )


# --- chunks_for_entity ---


def test_chunks_for_entity_returns_mentions(store: Store) -> None:
    store.add_entities([_entity("ent-a", ["c1", "c2", "c5"])])
    assert sorted(store.chunks_for_entity("ent-a")) == ["c1", "c2", "c5"]


def test_chunks_for_entity_unknown_returns_empty(store: Store) -> None:
    assert store.chunks_for_entity("missing") == []


def test_chunks_for_entity_empty_mentions_returns_empty(store: Store) -> None:
    store.add_entities([_entity("ent-a", [])])
    assert store.chunks_for_entity("ent-a") == []


# --- entities_for_chunk ---


def test_entities_for_chunk_returns_mentioning_entities(store: Store) -> None:
    store.add_entities(
        [
            _entity("ent-a", ["c1", "c2"]),
            _entity("ent-b", ["c2", "c3"]),
            _entity("ent-c", ["c4"]),
        ]
    )
    assert sorted(store.entities_for_chunk("c2")) == ["ent-a", "ent-b"]
    assert store.entities_for_chunk("c4") == ["ent-c"]


def test_entities_for_chunk_unknown_returns_empty(store: Store) -> None:
    assert store.entities_for_chunk("missing") == []


# --- entity_neighbors ---


def test_entity_neighbors_via_shared_chunks(store: Store) -> None:
    store.add_entities(
        [
            _entity("ent-a", ["c1", "c2"]),
            _entity("ent-b", ["c2", "c3"]),  # shares c2 with A
            _entity("ent-c", ["c3"]),  # shares c3 with B but not A
            _entity("ent-d", ["c9"]),  # disjoint
        ]
    )
    assert sorted(store.entity_neighbors("ent-a")) == ["ent-b"]
    assert sorted(store.entity_neighbors("ent-b")) == ["ent-a", "ent-c"]
    assert store.entity_neighbors("ent-d") == []


def test_entity_neighbors_excludes_self(store: Store) -> None:
    store.add_entities([_entity("ent-a", ["c1", "c2"]), _entity("ent-b", ["c1"])])
    assert "ent-a" not in store.entity_neighbors("ent-a")


def test_entity_neighbors_unknown_entity_returns_empty(store: Store) -> None:
    store.add_entities([_entity("ent-a", ["c1"])])
    assert store.entity_neighbors("missing") == []


def test_entity_neighbors_dedups_multi_chunk_overlap(store: Store) -> None:
    store.add_entities(
        [
            _entity("ent-a", ["c1", "c2", "c3"]),
            _entity("ent-b", ["c1", "c2", "c3"]),  # 3 shared chunks → one neighbor row
        ]
    )
    neighbors = store.entity_neighbors("ent-a")
    assert neighbors == ["ent-b"]


# --- re-add semantics propagate to inverted index ---


def test_re_adding_entity_updates_inverted_index(store: Store) -> None:
    store.add_entities([_entity("ent-a", ["c1", "c2"])])
    assert sorted(store.chunks_for_entity("ent-a")) == ["c1", "c2"]
    store.add_entities([_entity("ent-a", ["c9"])])
    assert sorted(store.chunks_for_entity("ent-a")) == ["c9"]
    assert store.entities_for_chunk("c1") == []
    assert sorted(store.entities_for_chunk("c9")) == ["ent-a"]
