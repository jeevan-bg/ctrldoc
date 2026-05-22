"""Contract tests for index integrity + backup before destructive ops.

PRAGMA integrity_check runs on every open (asserted in S-021); this
slice adds an on-demand `verify_integrity()` and the `backup()`
snapshot that any destructive operation must produce before mutating
the live index. `clear_all()` is the first destructive op and uses
the backup hook automatically.

SPEC-REF: §4.7 (index integrity)
"""

from __future__ import annotations

from pathlib import Path

from ctrldoc.models import Chunk, Entity, Section
from ctrldoc.store.sqlite import SQLiteStore


def _chunk(chunk_id: str = "c1") -> Chunk:
    return Chunk(
        id=chunk_id,
        section_id="s1",
        text=f"text-{chunk_id}",
        token_count=1,
        char_start=0,
        char_end=1,
        embedding_id=f"e-{chunk_id}",
    )


def _section(section_id: str = "s1") -> Section:
    return Section(
        id=section_id,
        parent_id=None,
        title=f"Title {section_id}",
        summary="Summary.",
        chunk_ids=[],
    )


def _entity(entity_id: str = "e1") -> Entity:
    return Entity(
        id=entity_id,
        aliases=[entity_id.upper()],
        type="concept",
        mention_chunk_ids=["c1"],
    )


def _populate(store: SQLiteStore) -> None:
    store.add_chunks([_chunk("c1"), _chunk("c2")])
    store.add_sections([_section("s1")])
    store.add_entities([_entity("e1")])


# --- verify_integrity ---


def test_verify_integrity_passes_on_healthy_store(tmp_path: Path) -> None:
    with SQLiteStore(tmp_path / "ctrldoc.db") as store:
        _populate(store)
        store.verify_integrity()  # must not raise


# --- backup ---


def test_backup_writes_alongside_db(tmp_path: Path) -> None:
    db = tmp_path / "ctrldoc.db"
    with SQLiteStore(db) as store:
        _populate(store)
        bak = store.backup()
    assert bak == db.with_suffix(db.suffix + ".bak")
    assert bak.exists()
    assert bak.stat().st_size > 0


def test_backup_is_a_valid_sqlite_store(tmp_path: Path) -> None:
    db = tmp_path / "ctrldoc.db"
    with SQLiteStore(db) as store:
        _populate(store)
        bak = store.backup()
    with SQLiteStore(bak) as restored:
        assert {c.id for c in restored.iter_chunks()} == {"c1", "c2"}
        assert {s.id for s in restored.iter_sections()} == {"s1"}
        assert {e.id for e in restored.iter_entities()} == {"e1"}


def test_backup_overwrites_previous_bak(tmp_path: Path) -> None:
    db = tmp_path / "ctrldoc.db"
    with SQLiteStore(db) as store:
        store.add_chunks([_chunk("c1")])
        first = store.backup()
        first_size = first.stat().st_size
        # Add more data and re-backup; the new bak must reflect the new state.
        store.add_chunks([_chunk(f"c{i}") for i in range(2, 30)])
        second = store.backup()
    assert second == first
    assert second.stat().st_size >= first_size
    with SQLiteStore(second) as restored:
        assert len(list(restored.iter_chunks())) == 29


# --- clear_all ---


def test_clear_all_takes_backup_first(tmp_path: Path) -> None:
    db = tmp_path / "ctrldoc.db"
    with SQLiteStore(db) as store:
        _populate(store)
        store.clear_all()
        assert list(store.iter_chunks()) == []
        assert list(store.iter_sections()) == []
        assert list(store.iter_entities()) == []
    bak = db.with_suffix(db.suffix + ".bak")
    assert bak.exists(), "clear_all must produce a .bak snapshot first"
    with SQLiteStore(bak) as restored:
        assert {c.id for c in restored.iter_chunks()} == {"c1", "c2"}


def test_clear_all_preserves_versions(tmp_path: Path) -> None:
    db = tmp_path / "ctrldoc.db"
    with SQLiteStore(db) as store:
        original_versions = store.versions
        _populate(store)
        store.clear_all()
        assert store.versions == original_versions


def test_clear_all_truncates_mentions(tmp_path: Path) -> None:
    db = tmp_path / "ctrldoc.db"
    with SQLiteStore(db) as store:
        _populate(store)
        # Pre-condition: mentions populated.
        rows_before = store._conn.execute("SELECT COUNT(*) FROM mentions").fetchone()[0]
        assert rows_before == 1
        store.clear_all()
        rows_after = store._conn.execute("SELECT COUNT(*) FROM mentions").fetchone()[0]
        assert rows_after == 0


def test_clear_all_on_empty_store_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "ctrldoc.db"
    with SQLiteStore(db) as store:
        store.clear_all()
        store.clear_all()  # second call must not raise
        assert list(store.iter_chunks()) == []
