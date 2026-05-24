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

from ctrldoc.models import Chunk, Entity, Section, Span
from ctrldoc.models_v1 import Claim, Concept, Workspace
from ctrldoc.provenance import Provenance, now_iso
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

-- v2 claim-graph substrate (SPEC §8). New tables are provisioned by
-- the same idempotent CREATE-IF-NOT-EXISTS path; the schema_version
-- bump (0.1.0 → 0.2.0) is what gates a v0.3 index from opening here.

CREATE TABLE IF NOT EXISTS claims (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    text TEXT NOT NULL,
    subject TEXT,
    predicate TEXT NOT NULL,
    object TEXT,
    polarity TEXT NOT NULL CHECK (polarity IN ('+', '-')),
    modality TEXT,
    qualifier_json TEXT NOT NULL DEFAULT '{}',
    span_refs_json TEXT NOT NULL,
    section_id TEXT NOT NULL,
    concept_ids_json TEXT NOT NULL DEFAULT '[]',
    typed_slots_json TEXT NOT NULL DEFAULT '{}',
    confidence REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_claims_doc ON claims(doc_id);
CREATE INDEX IF NOT EXISTS idx_claims_section ON claims(section_id);

CREATE TABLE IF NOT EXISTS concepts (
    id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    aliases_json TEXT NOT NULL DEFAULT '[]',
    primitive_type TEXT NOT NULL,
    mention_claim_ids_json TEXT NOT NULL DEFAULT '[]',
    doc_ids_json TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS typed_edges (
    src_id TEXT NOT NULL,
    dst_id TEXT NOT NULL,
    type TEXT NOT NULL,
    confidence REAL NOT NULL,
    raw_score REAL NOT NULL,
    citations_json TEXT NOT NULL,
    source TEXT NOT NULL,
    paraphrase_votes INTEGER,
    PRIMARY KEY (src_id, dst_id, type)
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON typed_edges(src_id, type);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON typed_edges(dst_id, type);

CREATE TABLE IF NOT EXISTS workspaces (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    doc_ids_json TEXT NOT NULL,
    induced_schema_json TEXT NOT NULL DEFAULT '{}',
    provenance_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cross_doc_edges (
    workspace_id TEXT NOT NULL,
    src_claim_id TEXT NOT NULL,
    dst_claim_id TEXT NOT NULL,
    type TEXT NOT NULL,
    confidence REAL NOT NULL,
    raw_score REAL NOT NULL,
    citations_json TEXT NOT NULL,
    source TEXT NOT NULL,
    PRIMARY KEY (workspace_id, src_claim_id, dst_claim_id, type)
);

CREATE TABLE IF NOT EXISTS verdict_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    inputs_json TEXT NOT NULL,
    output_json TEXT NOT NULL,
    calibrated_confidence REAL NOT NULL,
    model_versions_json TEXT NOT NULL,
    paraphrase_votes_json TEXT,
    timestamp TEXT NOT NULL
);
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

    # --- integrity + backup (§4.7 index integrity) ---

    def verify_integrity(self) -> None:
        """Run `PRAGMA integrity_check` on demand. Raises on failure."""
        self._integrity_check()

    def backup(self) -> Path:
        """Snapshot the live database to `<path>.bak` using SQLite's online
        backup API. Returns the path of the snapshot. Overwrites a prior
        snapshot.
        """
        target = self._path.with_suffix(self._path.suffix + ".bak")
        if target.exists():
            target.unlink()
        with sqlite3.connect(target) as dst:
            self._conn.backup(dst)
        return target

    def clear_all(self) -> None:
        """Destructive: truncate every data table after taking a backup.

        The `meta` table (versions) is preserved so an emptied index
        remains openable without re-bootstrapping the schema.
        """
        self.backup()
        with self._conn:
            self._conn.execute("DELETE FROM chunks")
            self._conn.execute("DELETE FROM sections")
            self._conn.execute("DELETE FROM entities")
            self._conn.execute("DELETE FROM mentions")
            # v2 claim-graph substrate.
            self._conn.execute("DELETE FROM claims")
            self._conn.execute("DELETE FROM concepts")
            self._conn.execute("DELETE FROM typed_edges")
            self._conn.execute("DELETE FROM workspaces")
            self._conn.execute("DELETE FROM cross_doc_edges")
            self._conn.execute("DELETE FROM verdict_ledger")

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

    def delete_chunks_for_section(self, section_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT id FROM chunks WHERE section_id = ? ORDER BY id",
            (section_id,),
        ).fetchall()
        chunk_ids = [row[0] for row in rows]
        with self._conn:
            self._conn.execute("DELETE FROM chunks WHERE section_id = ?", (section_id,))
        return chunk_ids

    def delete_section(self, section_id: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM sections WHERE id = ?", (section_id,))

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

    # --- v2 workspace CRUD (§6.7) ---

    def add_workspace(self, workspace: Workspace) -> None:
        """Insert or replace a `Workspace` row.

        `INSERT OR REPLACE` mirrors the v0.3 chunk/section/entity
        semantics — same id ⇒ overwrite. Higher-level uniqueness
        (one workspace per name) is enforced by the `WorkspaceManager`
        facade and by the `UNIQUE` constraint on the `name` column.
        """
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO workspaces
                    (id, name, doc_ids_json, induced_schema_json,
                     provenance_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    workspace.id,
                    workspace.name,
                    json.dumps(list(workspace.doc_ids)),
                    json.dumps(dict(workspace.induced_schema)),
                    workspace.provenance.model_dump_json(),
                    now_iso(),
                ),
            )

    def get_workspace_by_id(self, workspace_id: str) -> Workspace | None:
        row = self._conn.execute(
            "SELECT * FROM workspaces WHERE id = ?", (workspace_id,)
        ).fetchone()
        return _row_to_workspace(row) if row is not None else None

    def get_workspace_by_name(self, name: str) -> Workspace | None:
        row = self._conn.execute("SELECT * FROM workspaces WHERE name = ?", (name,)).fetchone()
        return _row_to_workspace(row) if row is not None else None

    def iter_workspaces(self) -> Iterator[Workspace]:
        """Yield workspaces in their creation order.

        Primary order key is the sortable ISO `created_at` timestamp;
        SQLite's implicit monotonic `rowid` breaks sub-second ties so
        the iteration order matches insertion order even when several
        workspaces are created within the same wall-clock second.
        """
        for row in self._conn.execute("SELECT * FROM workspaces ORDER BY created_at, rowid"):
            yield _row_to_workspace(row)

    def update_workspace_doc_ids(self, workspace_id: str, doc_ids: Iterable[str]) -> None:
        """Replace the workspace's `doc_ids` list without touching other fields.

        Raises `KeyError` if the workspace does not exist — silent no-ops
        on a missing id would let `workspace add unknown` succeed and
        confuse the user.
        """
        new_doc_ids = list(doc_ids)
        with self._conn:
            cursor = self._conn.execute(
                "UPDATE workspaces SET doc_ids_json = ? WHERE id = ?",
                (json.dumps(new_doc_ids), workspace_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"workspace not found: {workspace_id!r}")

    # --- v2 claim CRUD (§6.2, §6.4 universal-tuple persistence) ---

    def append_claim(self, claim: Claim) -> None:
        """Insert or replace one `Claim` row.

        Idempotent on `claim.id` — the id is a content hash over the
        six logical slots plus the doc / chunk binding (§6.2), so
        same-id ⇒ overwrite mirrors the §4.1 "ingest is idempotent
        and cacheable" property. Re-ingesting the same doc never
        produces duplicate rows.
        """
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO claims
                    (id, doc_id, text, subject, predicate, object,
                     polarity, modality, qualifier_json, span_refs_json,
                     section_id, concept_ids_json, typed_slots_json,
                     confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    claim.id,
                    claim.doc_id,
                    claim.text,
                    claim.subject,
                    claim.predicate,
                    claim.object,
                    claim.polarity,
                    claim.modality,
                    json.dumps(dict(claim.qualifier)),
                    json.dumps([s.model_dump() for s in claim.span_refs]),
                    claim.section_id,
                    json.dumps(list(claim.concept_ids)),
                    json.dumps(dict(claim.typed_slots)),
                    float(claim.confidence),
                ),
            )

    def get_claim(self, claim_id: str) -> Claim | None:
        row = self._conn.execute("SELECT * FROM claims WHERE id = ?", (claim_id,)).fetchone()
        return _row_to_claim(row) if row is not None else None

    def iter_claims(self) -> Iterator[Claim]:
        """Yield every persisted claim in ascending id order.

        The `id` is a sha256 content hash so ordering is stable across
        runs — guards downstream determinism (§13 non-negotiable 4).
        """
        for row in self._conn.execute("SELECT * FROM claims ORDER BY id"):
            yield _row_to_claim(row)

    def iter_claims_for_doc(self, doc_id: str) -> Iterator[Claim]:
        """Yield claims belonging to one `doc_id`, in ascending id order."""
        for row in self._conn.execute(
            "SELECT * FROM claims WHERE doc_id = ? ORDER BY id",
            (doc_id,),
        ):
            yield _row_to_claim(row)

    # --- v2 concept CRUD (§6.7 shared concept lattice) ---

    def add_concepts(self, concepts: Iterable[Concept]) -> None:
        """Insert or replace a batch of `Concept` rows.

        Concepts are canonical-cluster nodes — the same id ⇒ overwrite
        path lets a downstream ER pass (S-130) widen `doc_ids` /
        `mention_claim_ids` without a separate update method.
        """
        with self._conn:
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO concepts
                    (id, canonical_name, aliases_json, primitive_type,
                     mention_claim_ids_json, doc_ids_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        c.id,
                        c.canonical_name,
                        json.dumps(list(c.aliases)),
                        c.primitive_type,
                        json.dumps(list(c.mention_claim_ids)),
                        json.dumps(list(c.doc_ids)),
                    )
                    for c in concepts
                ],
            )

    def get_concept(self, concept_id: str) -> Concept | None:
        row = self._conn.execute("SELECT * FROM concepts WHERE id = ?", (concept_id,)).fetchone()
        return _row_to_concept(row) if row is not None else None

    def iter_concepts(self) -> Iterator[Concept]:
        for row in self._conn.execute("SELECT * FROM concepts ORDER BY id"):
            yield _row_to_concept(row)

    def concepts_for_workspace(self, workspace_id: str) -> Iterator[Concept]:
        """Yield the shared-concept-lattice slice visible to a workspace.

        A concept belongs to the slice when its `doc_ids` intersects
        the workspace's `doc_ids` (§6.7). The filtering happens in
        Python after a single SQL scan: the intersection predicate is
        cheap, but the JSON `doc_ids` column is opaque to SQLite, and
        the v1 workspace cardinality (≤ a handful of docs per
        workspace until v2) makes the scan negligible.
        """
        ws = self.get_workspace_by_id(workspace_id)
        if ws is None:
            raise KeyError(f"workspace not found: {workspace_id!r}")
        workspace_docs = set(ws.doc_ids)
        if not workspace_docs:
            return
        for concept in self.iter_concepts():
            if workspace_docs.intersection(concept.doc_ids):
                yield concept

    # --- v2 verdict ledger (§6.5 replayable verdicts) ---

    def append_ledger_row(
        self,
        *,
        workspace_id: str,
        operation: str,
        inputs_json: str,
        output_json: str,
        calibrated_confidence: float,
        model_versions_json: str,
        paraphrase_votes_json: str | None,
        timestamp: str,
    ) -> int:
        """Insert one row into `verdict_ledger`; return the AUTOINCREMENT id.

        Pure append — there is no companion `update_ledger_row` /
        `delete_ledger_row` because §6.5 requires the ledger to be
        replayable from the historical record. Schema-level absence of
        a mutator is the cheapest enforcement of the append-only
        contract.
        """
        with self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO verdict_ledger
                    (workspace_id, operation, inputs_json, output_json,
                     calibrated_confidence, model_versions_json,
                     paraphrase_votes_json, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    workspace_id,
                    operation,
                    inputs_json,
                    output_json,
                    calibrated_confidence,
                    model_versions_json,
                    paraphrase_votes_json,
                    timestamp,
                ),
            )
        row_id = cursor.lastrowid
        if row_id is None:  # pragma: no cover — SQLite always assigns one.
            raise RuntimeError("verdict_ledger insert did not return a rowid")
        return int(row_id)

    def get_ledger_row(self, entry_id: int) -> sqlite3.Row | None:
        """Fetch one ledger row by AUTOINCREMENT id, or `None` if absent."""
        row = self._conn.execute(
            "SELECT * FROM verdict_ledger WHERE id = ?", (entry_id,)
        ).fetchone()
        if row is None:
            return None
        assert isinstance(row, sqlite3.Row)
        return row

    def iter_ledger_rows(self, *, workspace_id: str | None = None) -> Iterator[sqlite3.Row]:
        """Iterate ledger rows in append (== id) order.

        `workspace_id` narrows the result set to one workspace; `None`
        (the default) yields every row. SQLite's `ORDER BY id` is the
        cheapest ordering: AUTOINCREMENT guarantees monotonicity, so
        append order is the read order.
        """
        if workspace_id is None:
            sql = "SELECT * FROM verdict_ledger ORDER BY id"
            params: tuple[object, ...] = ()
        else:
            sql = "SELECT * FROM verdict_ledger WHERE workspace_id = ? ORDER BY id"
            params = (workspace_id,)
        yield from self._conn.execute(sql, params)

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


def _row_to_workspace(row: sqlite3.Row) -> Workspace:
    return Workspace(
        id=row["id"],
        name=row["name"],
        doc_ids=list(json.loads(row["doc_ids_json"])),
        induced_schema=dict(json.loads(row["induced_schema_json"])),
        provenance=Provenance.model_validate_json(row["provenance_json"]),
    )


def _row_to_claim(row: sqlite3.Row) -> Claim:
    return Claim(
        id=row["id"],
        doc_id=row["doc_id"],
        text=row["text"],
        subject=row["subject"],
        predicate=row["predicate"],
        object=row["object"],
        polarity=row["polarity"],
        modality=row["modality"],
        qualifier=dict(json.loads(row["qualifier_json"])),
        span_refs=[Span(**s) for s in json.loads(row["span_refs_json"])],
        section_id=row["section_id"],
        concept_ids=list(json.loads(row["concept_ids_json"])),
        typed_slots=dict(json.loads(row["typed_slots_json"])),
        confidence=float(row["confidence"]),
    )


def _row_to_concept(row: sqlite3.Row) -> Concept:
    return Concept(
        id=row["id"],
        canonical_name=row["canonical_name"],
        aliases=list(json.loads(row["aliases_json"])),
        primitive_type=row["primitive_type"],
        mention_claim_ids=list(json.loads(row["mention_claim_ids_json"])),
        doc_ids=list(json.loads(row["doc_ids_json"])),
    )


__all__ = ["SQLiteStore"]
