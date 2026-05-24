"""Schema v2: claim-graph storage tables provisioned by `SQLiteStore`.

The v1 substrate adds six tables on top of the v0.3 chunk/section/entity
schema: `claims`, `concepts`, `typed_edges`, `workspaces`,
`cross_doc_edges`, and `verdict_ledger`. They are created by the same
idempotent `_init_schema` path the v0.3 tables use, gated by the
`IndexVersions.schema_version` bump from `"0.1.0"` to `"0.2.0"` so a
v0.3 index refuses to open under v1 without an explicit re-ingest.

This test file pins the storage contract: column sets, primary keys,
indexes, and the version bump. CRUD methods land in later slices.

SPEC-REF: §8
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ctrldoc.provenance import SCHEMA_VERSION
from ctrldoc.store.sqlite import SQLiteStore
from ctrldoc.versioning import IndexVersionMismatchError, IndexVersions


def _db(tmp_path: Path) -> Path:
    return tmp_path / "ctrldoc.db"


def _columns(store: SQLiteStore, table: str) -> dict[str, dict[str, object]]:
    rows = store._conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {
        row["name"]: {
            "type": row["type"],
            "notnull": bool(row["notnull"]),
            "dflt_value": row["dflt_value"],
            "pk": int(row["pk"]),
        }
        for row in rows
    }


def _index_names(store: SQLiteStore, table: str) -> set[str]:
    rows = store._conn.execute(f"PRAGMA index_list({table})").fetchall()
    return {row["name"] for row in rows}


# --- version bump ---


def test_schema_version_bumped_to_v2() -> None:
    """Bump signals the v0.3 → v1 substrate transition (§8: indexes guarded)."""
    assert SCHEMA_VERSION == "0.2.0"
    assert IndexVersions.current().schema_version == "0.2.0"


def test_v0_3_index_refuses_to_open_under_v2_runtime(tmp_path: Path) -> None:
    """An index stamped with the v0.3 schema_version must refuse v1 open."""
    path = _db(tmp_path)
    legacy = IndexVersions(
        schema_version="0.1.0",
        index_version=IndexVersions.current().index_version,
        embedding_model_version=IndexVersions.current().embedding_model_version,
    )
    with SQLiteStore(path, versions=legacy):
        pass
    with pytest.raises(IndexVersionMismatchError) as info:
        SQLiteStore(path)
    assert "schema_version" in str(info.value)


# --- claims ---


def test_claims_table_columns(tmp_path: Path) -> None:
    with SQLiteStore(_db(tmp_path)) as store:
        cols = _columns(store, "claims")
    expected = {
        "id",
        "doc_id",
        "text",
        "subject",
        "predicate",
        "object",
        "polarity",
        "modality",
        "qualifier_json",
        "span_refs_json",
        "section_id",
        "concept_ids_json",
        "typed_slots_json",
        "confidence",
    }
    assert set(cols) == expected
    assert cols["id"]["pk"] == 1
    assert cols["doc_id"]["notnull"]
    assert cols["text"]["notnull"]
    assert cols["predicate"]["notnull"]
    assert cols["polarity"]["notnull"]
    assert cols["span_refs_json"]["notnull"]
    assert cols["section_id"]["notnull"]
    assert cols["confidence"]["notnull"]
    # Optional columns from §8 (subject/object/modality) stay nullable.
    assert not cols["subject"]["notnull"]
    assert not cols["object"]["notnull"]
    assert not cols["modality"]["notnull"]


def test_claims_polarity_check_constraint(tmp_path: Path) -> None:
    """§8 requires polarity ∈ {'+', '-'}."""
    with SQLiteStore(_db(tmp_path)) as store:
        store._conn.execute(
            "INSERT INTO claims "
            "(id, doc_id, text, predicate, polarity, span_refs_json, section_id, confidence) "
            "VALUES ('c1', 'd1', 't', 'p', '+', '[]', 's1', 0.9)"
        )
        # Negative polarity is allowed too.
        store._conn.execute(
            "INSERT INTO claims "
            "(id, doc_id, text, predicate, polarity, span_refs_json, section_id, confidence) "
            "VALUES ('c2', 'd1', 't', 'p', '-', '[]', 's1', 0.5)"
        )
        # Anything else must violate the CHECK.
        import sqlite3

        with pytest.raises(sqlite3.IntegrityError):
            store._conn.execute(
                "INSERT INTO claims "
                "(id, doc_id, text, predicate, polarity, span_refs_json, section_id, confidence) "
                "VALUES ('c3', 'd1', 't', 'p', '?', '[]', 's1', 0.5)"
            )


def test_claims_indexes(tmp_path: Path) -> None:
    with SQLiteStore(_db(tmp_path)) as store:
        names = _index_names(store, "claims")
    assert "idx_claims_doc" in names
    assert "idx_claims_section" in names


# --- concepts ---


def test_concepts_table_columns(tmp_path: Path) -> None:
    with SQLiteStore(_db(tmp_path)) as store:
        cols = _columns(store, "concepts")
    assert set(cols) == {
        "id",
        "canonical_name",
        "aliases_json",
        "primitive_type",
        "mention_claim_ids_json",
        "doc_ids_json",
    }
    assert cols["id"]["pk"] == 1
    assert cols["canonical_name"]["notnull"]
    assert cols["primitive_type"]["notnull"]


# --- typed_edges ---


def test_typed_edges_table_columns_and_pk(tmp_path: Path) -> None:
    with SQLiteStore(_db(tmp_path)) as store:
        cols = _columns(store, "typed_edges")
    assert set(cols) == {
        "src_id",
        "dst_id",
        "type",
        "confidence",
        "raw_score",
        "citations_json",
        "source",
        "paraphrase_votes",
    }
    # Composite primary key on (src_id, dst_id, type).
    pk_columns = {name: meta["pk"] for name, meta in cols.items() if meta["pk"]}
    assert set(pk_columns) == {"src_id", "dst_id", "type"}
    assert cols["confidence"]["notnull"]
    assert cols["raw_score"]["notnull"]
    assert cols["citations_json"]["notnull"]
    assert cols["source"]["notnull"]
    # paraphrase_votes is optional (NULL allowed for non-paraphrase sources).
    assert not cols["paraphrase_votes"]["notnull"]


def test_typed_edges_indexes(tmp_path: Path) -> None:
    with SQLiteStore(_db(tmp_path)) as store:
        names = _index_names(store, "typed_edges")
    assert "idx_edges_src" in names
    assert "idx_edges_dst" in names


# --- workspaces ---


def test_workspaces_table_columns(tmp_path: Path) -> None:
    with SQLiteStore(_db(tmp_path)) as store:
        cols = _columns(store, "workspaces")
    # `provenance_json` carries the §7 `Provenance` record the in-memory
    # `Workspace` model requires; persisted alongside the §8 base columns
    # so create / list / info round-trips reconstruct the full Pydantic shape.
    assert set(cols) == {
        "id",
        "name",
        "doc_ids_json",
        "induced_schema_json",
        "provenance_json",
        "created_at",
    }
    assert cols["id"]["pk"] == 1
    assert cols["name"]["notnull"]
    assert cols["doc_ids_json"]["notnull"]
    assert cols["created_at"]["notnull"]


def test_workspaces_name_is_unique(tmp_path: Path) -> None:
    """`WorkspaceManager.create` rejects duplicate names; the DB enforces it too."""
    import sqlite3

    with SQLiteStore(_db(tmp_path)) as store:
        store._conn.execute(
            "INSERT INTO workspaces (id, name, doc_ids_json, created_at) "
            "VALUES ('ws-1', 'audit-2026', '[]', '2026-05-24T00:00:00Z')"
        )
        with pytest.raises(sqlite3.IntegrityError):
            store._conn.execute(
                "INSERT INTO workspaces (id, name, doc_ids_json, created_at) "
                "VALUES ('ws-2', 'audit-2026', '[]', '2026-05-24T00:00:01Z')"
            )


# --- cross_doc_edges ---


def test_cross_doc_edges_table_columns_and_pk(tmp_path: Path) -> None:
    with SQLiteStore(_db(tmp_path)) as store:
        cols = _columns(store, "cross_doc_edges")
    assert set(cols) == {
        "workspace_id",
        "src_claim_id",
        "dst_claim_id",
        "type",
        "confidence",
        "raw_score",
        "citations_json",
        "source",
    }
    pk_columns = {name: meta["pk"] for name, meta in cols.items() if meta["pk"]}
    assert set(pk_columns) == {"workspace_id", "src_claim_id", "dst_claim_id", "type"}
    assert cols["confidence"]["notnull"]
    assert cols["raw_score"]["notnull"]


# --- verdict_ledger ---


def test_verdict_ledger_table_columns(tmp_path: Path) -> None:
    with SQLiteStore(_db(tmp_path)) as store:
        cols = _columns(store, "verdict_ledger")
    assert set(cols) == {
        "id",
        "workspace_id",
        "operation",
        "inputs_json",
        "output_json",
        "calibrated_confidence",
        "model_versions_json",
        "paraphrase_votes_json",
        "timestamp",
    }
    # AUTOINCREMENT primary key.
    assert cols["id"]["pk"] == 1
    assert cols["workspace_id"]["notnull"]
    assert cols["operation"]["notnull"]
    assert cols["inputs_json"]["notnull"]
    assert cols["output_json"]["notnull"]
    assert cols["calibrated_confidence"]["notnull"]
    assert cols["model_versions_json"]["notnull"]
    assert cols["timestamp"]["notnull"]
    # paraphrase_votes_json is optional — only set when the operation
    # produced paraphrase-vote confidence shipments (§6.5).
    assert not cols["paraphrase_votes_json"]["notnull"]


def test_verdict_ledger_id_autoincrements(tmp_path: Path) -> None:
    """AUTOINCREMENT ⇒ rowids are monotonically assigned, never reused."""
    with SQLiteStore(_db(tmp_path)) as store:
        store._conn.execute(
            "INSERT INTO verdict_ledger "
            "(workspace_id, operation, inputs_json, output_json, "
            " calibrated_confidence, model_versions_json, timestamp) "
            "VALUES ('w1', 'coverage', '{}', '{}', 0.9, '{}', '2026-05-24T00:00:00Z')"
        )
        store._conn.execute(
            "INSERT INTO verdict_ledger "
            "(workspace_id, operation, inputs_json, output_json, "
            " calibrated_confidence, model_versions_json, timestamp) "
            "VALUES ('w1', 'compare', '{}', '{}', 0.7, '{}', '2026-05-24T00:00:01Z')"
        )
        rows = store._conn.execute(
            "SELECT id, operation FROM verdict_ledger ORDER BY id"
        ).fetchall()
        ids = [row["id"] for row in rows]
        assert ids == sorted(ids)
        assert len(set(ids)) == 2
        assert all(isinstance(i, int) for i in ids)


# --- whole-schema sanity ---


def test_v2_tables_all_present(tmp_path: Path) -> None:
    with SQLiteStore(_db(tmp_path)) as store:
        rows = store._conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    names = {row["name"] for row in rows}
    expected = {
        # v0.3 tables.
        "meta",
        "chunks",
        "sections",
        "entities",
        "mentions",
        # v2 additions.
        "claims",
        "concepts",
        "typed_edges",
        "workspaces",
        "cross_doc_edges",
        "verdict_ledger",
    }
    assert expected.issubset(names)


def test_clear_all_truncates_v2_tables(tmp_path: Path) -> None:
    """`SQLiteStore.clear_all` is the documented destructive reset; it must
    truncate the v2 substrate tables as well as the v0.3 ones."""
    with SQLiteStore(_db(tmp_path)) as store:
        store._conn.execute(
            "INSERT INTO claims "
            "(id, doc_id, text, predicate, polarity, span_refs_json, section_id, confidence) "
            "VALUES ('c1', 'd1', 't', 'p', '+', '[]', 's1', 0.9)"
        )
        store._conn.execute(
            "INSERT INTO concepts (id, canonical_name, primitive_type) "
            "VALUES ('k1', 'thing', 'entity')"
        )
        store._conn.execute(
            "INSERT INTO typed_edges "
            "(src_id, dst_id, type, confidence, raw_score, citations_json, source) "
            "VALUES ('a', 'b', 'entails', 0.9, 0.8, '[]', 'nli')"
        )
        store._conn.execute(
            "INSERT INTO workspaces (id, name, doc_ids_json, created_at) "
            "VALUES ('w1', 'demo', '[]', '2026-05-24T00:00:00Z')"
        )
        store._conn.execute(
            "INSERT INTO cross_doc_edges "
            "(workspace_id, src_claim_id, dst_claim_id, type, "
            " confidence, raw_score, citations_json, source) "
            "VALUES ('w1', 'a', 'b', 'aligned_with', 0.8, 0.7, '[]', 'nli')"
        )
        store._conn.execute(
            "INSERT INTO verdict_ledger "
            "(workspace_id, operation, inputs_json, output_json, "
            " calibrated_confidence, model_versions_json, timestamp) "
            "VALUES ('w1', 'coverage', '{}', '{}', 0.9, '{}', '2026-05-24T00:00:00Z')"
        )
        store._conn.commit()
        store.clear_all()
        for table in (
            "claims",
            "concepts",
            "typed_edges",
            "workspaces",
            "cross_doc_edges",
            "verdict_ledger",
        ):
            count = store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            assert count == 0, f"{table} not truncated"


def test_init_schema_is_idempotent_on_reopen(tmp_path: Path) -> None:
    """Reopening the same DB must succeed (CREATE TABLE IF NOT EXISTS)."""
    path = _db(tmp_path)
    with SQLiteStore(path):
        pass
    # Second open exercises _init_schema against an existing v2 file.
    with SQLiteStore(path) as store:
        rows = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='claims'"
        ).fetchall()
        assert len(rows) == 1
