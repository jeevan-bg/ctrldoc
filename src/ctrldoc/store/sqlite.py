"""SQLite-backed implementation of the `Store` protocol.

The structural layout is:

```
meta(key TEXT PRIMARY KEY, value TEXT)
chunks(id PK, section_id, text, token_count, char_start, char_end,
       embedding_id, metadata_json)
sections(id PK, parent_id, title, summary, chunk_ids_json)
entities(id PK, type, aliases_json)
mentions(entity_id, chunk_id, PRIMARY KEY (entity_id, chunk_id))
```

Dense-vector, BM25, and entity-inverted-index queries are added on
top of the same tables by other modules in the `store` package.

On open the backend runs `PRAGMA integrity_check` and, when an
existing file is opened, verifies the stored `IndexVersions`
against the runtime — a drift raises `IndexVersionMismatchError`
rather than silently migrating.

SPEC-REF: §4.2, §4.7 (versioning, index integrity)
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from pathlib import Path
from types import TracebackType

from ctrldoc.models import Chunk, Entity, Section
from ctrldoc.versioning import IndexVersions

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    section_id TEXT NOT NULL,
    text TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    embedding_id TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS sections (
    id TEXT PRIMARY KEY,
    parent_id TEXT,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    chunk_ids_json TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    aliases_json TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS mentions (
    entity_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    PRIMARY KEY (entity_id, chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks(section_id);
CREATE INDEX IF NOT EXISTS idx_mentions_chunk ON mentions(chunk_id);
"""


class SQLiteStore:
    """Persistent `Store` backed by a single SQLite file."""

    def __init__(
        self,
        path: str | Path,
        *,
        versions: IndexVersions | None = None,
    ) -> None:
        self._path = Path(path)
        is_new = not self._path.exists()
        self._conn = sqlite3.connect(self._path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._integrity_check()
        if is_new:
            self._init_schema()
            self._write_versions(versions or IndexVersions.current())
        else:
            self._init_schema()  # idempotent; CREATE TABLE IF NOT EXISTS
            stored = self._read_versions()
            stored.assert_compatible_with(versions or IndexVersions.current())
        self._versions = self._read_versions()

    # --- lifecycle ---

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SQLiteStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # --- protocol ---

    @property
    def versions(self) -> IndexVersions:
        return self._versions

    def add_chunks(self, chunks: Iterable[Chunk]) -> None:
        with self._conn:
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO chunks
                    (id, section_id, text, token_count, char_start, char_end,
                     embedding_id, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        c.id,
                        c.section_id,
                        c.text,
                        c.token_count,
                        c.char_start,
                        c.char_end,
                        c.embedding_id,
                        json.dumps(c.metadata),
                    )
                    for c in chunks
                ],
            )

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        row = self._conn.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        return _row_to_chunk(row) if row is not None else None

    def iter_chunks(self) -> Iterator[Chunk]:
        for row in self._conn.execute("SELECT * FROM chunks"):
            yield _row_to_chunk(row)

    def add_sections(self, sections: Iterable[Section]) -> None:
        with self._conn:
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO sections
                    (id, parent_id, title, summary, chunk_ids_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (s.id, s.parent_id, s.title, s.summary, json.dumps(s.chunk_ids))
                    for s in sections
                ],
            )

    def get_section(self, section_id: str) -> Section | None:
        row = self._conn.execute("SELECT * FROM sections WHERE id = ?", (section_id,)).fetchone()
        return _row_to_section(row) if row is not None else None

    def iter_sections(self) -> Iterator[Section]:
        for row in self._conn.execute("SELECT * FROM sections"):
            yield _row_to_section(row)

    def add_entities(self, entities: Iterable[Entity]) -> None:
        entities = list(entities)
        with self._conn:
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO entities (id, type, aliases_json)
                VALUES (?, ?, ?)
                """,
                [(e.id, e.type, json.dumps(e.aliases)) for e in entities],
            )
            # Replace the mention rows for each entity we just wrote so
            # the inverted-index junction always reflects the latest set.
            for entity in entities:
                self._conn.execute("DELETE FROM mentions WHERE entity_id = ?", (entity.id,))
                self._conn.executemany(
                    "INSERT INTO mentions (entity_id, chunk_id) VALUES (?, ?)",
                    [(entity.id, chunk_id) for chunk_id in entity.mention_chunk_ids],
                )

    def get_entity(self, entity_id: str) -> Entity | None:
        row = self._conn.execute("SELECT * FROM entities WHERE id = ?", (entity_id,)).fetchone()
        if row is None:
            return None
        return self._hydrate_entity(row)

    def iter_entities(self) -> Iterator[Entity]:
        for row in self._conn.execute("SELECT * FROM entities"):
            yield self._hydrate_entity(row)

    # --- entity inverted-index lookups ---

    def chunks_for_entity(self, entity_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT chunk_id FROM mentions WHERE entity_id = ? ORDER BY chunk_id",
            (entity_id,),
        ).fetchall()
        return [row[0] for row in rows]

    def entities_for_chunk(self, chunk_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT entity_id FROM mentions WHERE chunk_id = ? ORDER BY entity_id",
            (chunk_id,),
        ).fetchall()
        return [row[0] for row in rows]

    def entity_neighbors(self, entity_id: str) -> list[str]:
        rows = self._conn.execute(
            """
            SELECT DISTINCT m2.entity_id
            FROM mentions m1
            JOIN mentions m2 ON m1.chunk_id = m2.chunk_id
            WHERE m1.entity_id = ? AND m2.entity_id != m1.entity_id
            ORDER BY m2.entity_id
            """,
            (entity_id,),
        ).fetchall()
        return [row[0] for row in rows]

    # --- internals ---

    def _integrity_check(self) -> None:
        result = self._conn.execute("PRAGMA integrity_check").fetchone()
        if result is None or result[0] != "ok":
            raise sqlite3.DatabaseError(
                f"integrity_check failed for {self._path}: {result[0] if result else 'no result'}"
            )

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.executescript(_SCHEMA)

    def _write_versions(self, versions: IndexVersions) -> None:
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                [
                    ("schema_version", versions.schema_version),
                    ("index_version", versions.index_version),
                    ("embedding_model_version", versions.embedding_model_version),
                ],
            )

    def _read_versions(self) -> IndexVersions:
        rows = dict(self._conn.execute("SELECT key, value FROM meta").fetchall())
        return IndexVersions(
            schema_version=rows["schema_version"],
            index_version=rows["index_version"],
            embedding_model_version=rows["embedding_model_version"],
        )

    def _hydrate_entity(self, row: sqlite3.Row) -> Entity:
        chunk_ids = [
            r[0]
            for r in self._conn.execute(
                "SELECT chunk_id FROM mentions WHERE entity_id = ? ORDER BY chunk_id",
                (row["id"],),
            )
        ]
        return Entity(
            id=row["id"],
            type=row["type"],
            aliases=list(json.loads(row["aliases_json"])),
            mention_chunk_ids=chunk_ids,
        )


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        id=row["id"],
        section_id=row["section_id"],
        text=row["text"],
        token_count=row["token_count"],
        char_start=row["char_start"],
        char_end=row["char_end"],
        embedding_id=row["embedding_id"],
        metadata=dict(json.loads(row["metadata_json"])),
    )


def _row_to_section(row: sqlite3.Row) -> Section:
    return Section(
        id=row["id"],
        parent_id=row["parent_id"],
        title=row["title"],
        summary=row["summary"],
        chunk_ids=list(json.loads(row["chunk_ids_json"])),
    )


__all__ = ["SQLiteStore"]
