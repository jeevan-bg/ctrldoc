"""Contract tests for the persistent SQLite-backed `Store`.

The SQLite backend must satisfy the same protocol as the in-memory
reference. On reopen it verifies the stored `IndexVersions` against
the runtime and runs `PRAGMA integrity_check`; any mismatch refuses
to proceed.

SPEC-REF: §4.2, §4.7 (versioning), §4.7 (index integrity)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ctrldoc.models import Chunk, Entity, Section
from ctrldoc.store import Store
from ctrldoc.store.sqlite import SQLiteStore
from ctrldoc.versioning import IndexVersionMismatchError, IndexVersions


def _chunk(**overrides: object) -> Chunk:
    defaults: dict[str, object] = {
        "id": "c1",
        "section_id": "s1",
        "text": "hello",
        "token_count": 1,
        "char_start": 0,
        "char_end": 5,
        "embedding_id": "e1",
        "metadata": {"source": "test", "lang": "en"},
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
        "aliases": ["A", "alpha"],
        "type": "concept",
        "mention_chunk_ids": ["c1", "c2"],
    }
    defaults.update(overrides)
    return Entity(**defaults)  # type: ignore[arg-type]


def _db(tmp_path: Path) -> Path:
    return tmp_path / "ctrldoc.db"


# --- protocol conformance ---


def test_satisfies_store_protocol(tmp_path: Path) -> None:
    with SQLiteStore(_db(tmp_path)) as store:
        assert isinstance(store, Store)


# --- bootstrap & versions ---


def test_new_file_writes_current_versions(tmp_path: Path) -> None:
    with SQLiteStore(_db(tmp_path)) as store:
        assert store.versions == IndexVersions.current()


def test_reopen_round_trips_versions(tmp_path: Path) -> None:
    path = _db(tmp_path)
    with SQLiteStore(path) as store:
        store.add_chunks([_chunk()])
    with SQLiteStore(path) as store:
        assert store.versions == IndexVersions.current()
        got = store.get_chunk("c1")
        assert got is not None
        assert got.text == "hello"


def test_reopen_with_mismatched_versions_raises(tmp_path: Path) -> None:
    path = _db(tmp_path)
    pinned = IndexVersions(
        schema_version="9.9.9",
        index_version="9.9.9",
        embedding_model_version="ancient",
    )
    with SQLiteStore(path, versions=pinned):
        pass
    with pytest.raises(IndexVersionMismatchError):
        SQLiteStore(path)


# --- CRUD: chunks ---


def test_chunk_round_trip(tmp_path: Path) -> None:
    path = _db(tmp_path)
    with SQLiteStore(path) as store:
        store.add_chunks([_chunk(id="c1"), _chunk(id="c2", text="world")])
    with SQLiteStore(path) as store:
        assert store.get_chunk("c1") == _chunk(id="c1")
        assert store.get_chunk("c2") == _chunk(id="c2", text="world")
        assert {c.id for c in store.iter_chunks()} == {"c1", "c2"}


def test_chunk_metadata_round_trip(tmp_path: Path) -> None:
    with SQLiteStore(_db(tmp_path)) as store:
        store.add_chunks([_chunk(metadata={"k": 1, "nested": [1, 2, 3]})])
        got = store.get_chunk("c1")
        assert got is not None
        assert got.metadata == {"k": 1, "nested": [1, 2, 3]}


def test_chunk_get_missing(tmp_path: Path) -> None:
    with SQLiteStore(_db(tmp_path)) as store:
        assert store.get_chunk("missing") is None


def test_chunk_add_is_idempotent_by_id(tmp_path: Path) -> None:
    with SQLiteStore(_db(tmp_path)) as store:
        store.add_chunks([_chunk(id="c1", text="old")])
        store.add_chunks([_chunk(id="c1", text="new")])
        got = store.get_chunk("c1")
        assert got is not None
        assert got.text == "new"
        assert len(list(store.iter_chunks())) == 1


# --- CRUD: sections ---


def test_section_round_trip(tmp_path: Path) -> None:
    path = _db(tmp_path)
    with SQLiteStore(path) as store:
        store.add_sections(
            [_section(id="s1"), _section(id="s2", parent_id="s1", chunk_ids=["c2", "c3"])]
        )
    with SQLiteStore(path) as store:
        got = store.get_section("s2")
        assert got is not None
        assert got.parent_id == "s1"
        assert got.chunk_ids == ["c2", "c3"]
        assert {s.id for s in store.iter_sections()} == {"s1", "s2"}


# --- CRUD: entities ---


def test_entity_round_trip(tmp_path: Path) -> None:
    path = _db(tmp_path)
    with SQLiteStore(path) as store:
        store.add_entities([_entity(id="e1"), _entity(id="e2", type="person")])
    with SQLiteStore(path) as store:
        got = store.get_entity("e2")
        assert got is not None
        assert got.type == "person"
        assert {e.id for e in store.iter_entities()} == {"e1", "e2"}


def test_mentions_table_links_entities_to_chunks(tmp_path: Path) -> None:
    """add_entities must populate the `mentions` junction so S-024 has the row pairs."""
    with SQLiteStore(_db(tmp_path)) as store:
        store.add_entities(
            [
                _entity(id="e1", mention_chunk_ids=["c1", "c2"]),
                _entity(id="e2", mention_chunk_ids=["c2"]),
            ]
        )
        cur = store._conn.execute(
            "SELECT entity_id, chunk_id FROM mentions ORDER BY entity_id, chunk_id"
        )
        rows = [tuple(row) for row in cur.fetchall()]
        assert rows == [("e1", "c1"), ("e1", "c2"), ("e2", "c2")]


def test_re_adding_entity_replaces_mentions(tmp_path: Path) -> None:
    with SQLiteStore(_db(tmp_path)) as store:
        store.add_entities([_entity(id="e1", mention_chunk_ids=["c1", "c2"])])
        store.add_entities([_entity(id="e1", mention_chunk_ids=["c9"])])
        cur = store._conn.execute(
            "SELECT chunk_id FROM mentions WHERE entity_id = 'e1' ORDER BY chunk_id"
        )
        assert [row[0] for row in cur.fetchall()] == ["c9"]


# --- integrity ---


def test_integrity_check_passes_after_writes(tmp_path: Path) -> None:
    path = _db(tmp_path)
    with SQLiteStore(path) as store:
        store.add_chunks([_chunk()])
        store.add_sections([_section()])
        store.add_entities([_entity()])
    with SQLiteStore(path) as store:
        result = store._conn.execute("PRAGMA integrity_check").fetchone()
        assert result[0] == "ok"


def test_corrupt_db_raises_on_open(tmp_path: Path) -> None:
    path = _db(tmp_path)
    path.write_bytes(b"not a sqlite file at all" + b"\0" * 1024)
    with pytest.raises(sqlite3.DatabaseError):
        SQLiteStore(path)
