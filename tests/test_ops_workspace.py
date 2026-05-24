"""High-level workspace operations: `create` / `add` / `list` / `info`.

The L2.5 workspace primitive is a typed collection of doc-graphs that
shares one concept lattice (§6.7). `WorkspaceManager` is the small
facade the v1 CLI and the v1 Python API both call: it owns id-shape
discipline (`ws-<sha256(name)[:16]>`), name-uniqueness, provenance
stamping, and the shared-concept-lattice view per workspace. Heavy
state (concepts, claims, cross-doc edges) lives in the SQLite store;
the manager is the seam playbook code and the CLI talk through.

`WorkspaceInfo` is the read-only view `workspace info <name>` returns:
the underlying `Workspace`, the document count, and the concept count
in the shared lattice slice. The fuller per-doc breakdown lands when
the L2 retrieval layer arrives (S-135 onwards).

SPEC-REF: §6.7 (workspace), §9 (CLI: workspace create/add/list/info)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ctrldoc.models_v1 import Concept
from ctrldoc.ops.workspace import (
    WorkspaceAlreadyExistsError,
    WorkspaceInfo,
    WorkspaceManager,
    WorkspaceNotFoundError,
)
from ctrldoc.store.sqlite import SQLiteStore


def _store(tmp_path: Path) -> SQLiteStore:
    return SQLiteStore(tmp_path / "ctrldoc.db")


def _concept(*, id: str, name: str, doc_ids: list[str]) -> Concept:
    return Concept(
        id=id,
        canonical_name=name,
        aliases=[],
        primitive_type="Entity",
        mention_claim_ids=[],
        doc_ids=doc_ids,
    )


# --- create ---


def test_create_persists_workspace_with_deterministic_id(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        manager = WorkspaceManager(store=store)
        first = manager.create("audit-2026")
        # The id is a pure function of the name so reloading another
        # process sees the same identity.
        with _store(tmp_path) as reopened:
            assert reopened.get_workspace_by_id(first.id) is not None
        assert first.name == "audit-2026"
        assert first.id.startswith("ws-")
        assert len(first.id) == len("ws-") + 16  # sha256[:16]


def test_create_with_blank_name_rejected(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        manager = WorkspaceManager(store=store)
        with pytest.raises(ValueError):
            manager.create("   ")


def test_create_with_duplicate_name_raises(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        manager = WorkspaceManager(store=store)
        manager.create("audit-2026")
        with pytest.raises(WorkspaceAlreadyExistsError):
            manager.create("audit-2026")


def test_create_stamps_provenance_with_workspace_playbook(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        manager = WorkspaceManager(store=store)
        ws = manager.create("audit-2026")
    assert ws.provenance.playbook == "workspace"
    assert ws.provenance.schema_version == "0.2.0"


# --- add ---


def test_add_appends_doc_id_and_dedupes(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        manager = WorkspaceManager(store=store)
        manager.create("audit-2026")
        manager.add("audit-2026", "doc-a")
        manager.add("audit-2026", "doc-b")
        # Re-adding the same doc is a no-op; the doc list stays unique
        # so downstream cross-doc-edge enumeration over (|A|·k) is honest.
        ws_after = manager.add("audit-2026", "doc-a")
    assert ws_after.doc_ids == ["doc-a", "doc-b"]


def test_add_preserves_insertion_order(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        manager = WorkspaceManager(store=store)
        manager.create("audit-2026")
        for doc_id in ("doc-c", "doc-a", "doc-b"):
            manager.add("audit-2026", doc_id)
        ws = manager.get("audit-2026")
    assert ws.doc_ids == ["doc-c", "doc-a", "doc-b"]


def test_add_to_missing_workspace_raises(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        manager = WorkspaceManager(store=store)
        with pytest.raises(WorkspaceNotFoundError):
            manager.add("missing", "doc-a")


def test_add_blank_doc_id_rejected(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        manager = WorkspaceManager(store=store)
        manager.create("audit-2026")
        with pytest.raises(ValueError):
            manager.add("audit-2026", "   ")


# --- list ---


def test_list_returns_all_workspaces_in_creation_order(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        manager = WorkspaceManager(store=store)
        manager.create("a-first")
        manager.create("b-second")
        manager.create("c-third")
        names = [ws.name for ws in manager.list()]
    assert names == ["a-first", "b-second", "c-third"]


def test_list_on_empty_store_returns_empty(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        manager = WorkspaceManager(store=store)
        assert manager.list() == []


# --- info ---


def test_info_aggregates_shared_concept_lattice(tmp_path: Path) -> None:
    """`workspace info` exposes the shared-lattice slice the docs co-induce.

    The `concept_count` is the size of the slice §6.7 calls the shared
    concept lattice — concepts whose `doc_ids` intersect the
    workspace's member docs. Lattice rows whose members live in
    sibling docs outside the workspace are excluded.
    """
    with _store(tmp_path) as store:
        manager = WorkspaceManager(store=store)
        manager.create("audit-2026")
        manager.add("audit-2026", "doc-a")
        manager.add("audit-2026", "doc-b")
        store.add_concepts(
            [
                _concept(id="k-1", name="OAuth2", doc_ids=["doc-a"]),
                _concept(id="k-2", name="JWT", doc_ids=["doc-a", "doc-b"]),
                _concept(id="k-3", name="OpenID", doc_ids=["doc-b"]),
                _concept(id="k-4", name="LDAP", doc_ids=["doc-c"]),  # outside
            ]
        )
        info = manager.info("audit-2026")
    assert isinstance(info, WorkspaceInfo)
    assert info.workspace.name == "audit-2026"
    assert info.doc_count == 2
    assert info.concept_count == 3
    assert sorted(info.shared_concept_ids) == ["k-1", "k-2", "k-3"]


def test_info_on_empty_workspace_reports_zero_counts(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        manager = WorkspaceManager(store=store)
        manager.create("audit-2026")
        info = manager.info("audit-2026")
    assert info.doc_count == 0
    assert info.concept_count == 0
    assert info.shared_concept_ids == []


def test_info_on_missing_workspace_raises(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        manager = WorkspaceManager(store=store)
        with pytest.raises(WorkspaceNotFoundError):
            manager.info("missing")


def test_get_returns_persisted_workspace(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        manager = WorkspaceManager(store=store)
        manager.create("audit-2026")
        ws = manager.get("audit-2026")
    assert ws.name == "audit-2026"
    assert ws.doc_ids == []


def test_get_on_missing_workspace_raises(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        manager = WorkspaceManager(store=store)
        with pytest.raises(WorkspaceNotFoundError):
            manager.get("missing")
