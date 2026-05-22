"""Contract tests for the `Store` protocol and its in-memory reference.

The protocol is the single seam between L1 storage and the rest of
the stack — every concrete backend (SQLite, Qdrant, ...) must satisfy
it. The in-memory implementation in `ctrldoc.store.memory` doubles as
a unit-test fixture for the layers above.

SPEC-REF: §10 (storage abstraction), §13 (non-negotiable #5), §4.2
"""

from __future__ import annotations

import pytest

from ctrldoc.models import Chunk, Entity, Section
from ctrldoc.store import Store
from ctrldoc.store.memory import InMemoryStore
from ctrldoc.versioning import IndexVersions


def _chunk(**overrides: object) -> Chunk:
    defaults: dict[str, object] = {
        "id": "c1",
        "section_id": "s1",
        "text": "hello",
        "token_count": 1,
        "char_start": 0,
        "char_end": 5,
        "embedding_id": "e1",
    }
    defaults.update(overrides)
    return Chunk(**defaults)  # type: ignore[arg-type]


def _section(**overrides: object) -> Section:
    defaults: dict[str, object] = {
        "id": "s1",
        "parent_id": None,
        "title": "Intro",
        "summary": "Summary.",
        "chunk_ids": ["c1"],
    }
    defaults.update(overrides)
    return Section(**defaults)  # type: ignore[arg-type]


def _entity(**overrides: object) -> Entity:
    defaults: dict[str, object] = {
        "id": "ent-a",
        "aliases": ["A"],
        "type": "concept",
        "mention_chunk_ids": ["c1"],
    }
    defaults.update(overrides)
    return Entity(**defaults)  # type: ignore[arg-type]


# --- protocol conformance ---


def test_inmemory_store_satisfies_protocol() -> None:
    store: Store = InMemoryStore()
    assert isinstance(store, Store)


def test_versions_default_to_current() -> None:
    store = InMemoryStore()
    assert store.versions == IndexVersions.current()


def test_versions_can_be_pinned() -> None:
    pinned = IndexVersions(
        schema_version="1.2.3",
        index_version="9.9.9",
        embedding_model_version="custom-emb",
    )
    store = InMemoryStore(versions=pinned)
    assert store.versions == pinned


# --- chunks ---


def test_add_and_get_chunk() -> None:
    store = InMemoryStore()
    store.add_chunks([_chunk(id="c1"), _chunk(id="c2", text="world")])
    assert store.get_chunk("c1") == _chunk(id="c1")
    assert store.get_chunk("c2") == _chunk(id="c2", text="world")


def test_get_chunk_missing_returns_none() -> None:
    assert InMemoryStore().get_chunk("missing") is None


def test_iter_chunks_yields_all_added() -> None:
    store = InMemoryStore()
    chunks = [_chunk(id=f"c{i}") for i in range(5)]
    store.add_chunks(chunks)
    assert {c.id for c in store.iter_chunks()} == {f"c{i}" for i in range(5)}


def test_add_chunks_is_idempotent_by_id() -> None:
    store = InMemoryStore()
    store.add_chunks([_chunk(id="c1", text="old")])
    store.add_chunks([_chunk(id="c1", text="new")])
    got = store.get_chunk("c1")
    assert got is not None
    assert got.text == "new"
    assert len(list(store.iter_chunks())) == 1


# --- sections ---


def test_add_and_get_section() -> None:
    store = InMemoryStore()
    store.add_sections([_section(id="s1"), _section(id="s2", title="Two", chunk_ids=["c2"])])
    sec2 = store.get_section("s2")
    assert sec2 is not None
    assert sec2.title == "Two"


def test_iter_sections_yields_all() -> None:
    store = InMemoryStore()
    store.add_sections([_section(id="s1"), _section(id="s2", chunk_ids=[])])
    assert {s.id for s in store.iter_sections()} == {"s1", "s2"}


# --- entities ---


def test_add_and_get_entity() -> None:
    store = InMemoryStore()
    store.add_entities([_entity(id="e1"), _entity(id="e2", aliases=["B"], type="person")])
    e2 = store.get_entity("e2")
    assert e2 is not None
    assert e2.type == "person"


def test_iter_entities_yields_all() -> None:
    store = InMemoryStore()
    store.add_entities([_entity(id=f"e{i}") for i in range(3)])
    assert {e.id for e in store.iter_entities()} == {f"e{i}" for i in range(3)}


def test_glossary_built_from_iter_entities() -> None:
    store = InMemoryStore()
    store.add_entities([_entity(id="e1"), _entity(id="e2")])
    glossary = {e.id: e for e in store.iter_entities()}
    assert set(glossary.keys()) == {"e1", "e2"}


# --- isolation ---


def test_two_stores_are_independent() -> None:
    a = InMemoryStore()
    b = InMemoryStore()
    a.add_chunks([_chunk(id="c1")])
    assert b.get_chunk("c1") is None
    assert list(a.iter_chunks()) and not list(b.iter_chunks())


@pytest.mark.parametrize(
    "method,arg",
    [
        ("get_chunk", "missing"),
        ("get_section", "missing"),
        ("get_entity", "missing"),
    ],
)
def test_get_missing_returns_none_across_views(method: str, arg: str) -> None:
    store = InMemoryStore()
    assert getattr(store, method)(arg) is None
