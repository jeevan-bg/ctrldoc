"""Workspace and concept persistence on `SQLiteStore`.

S-125 provisioned the v2 tables; S-134 adds the CRUD seams the L2.5
workspace primitive and the L1.5 concept lattice need. The shape
mirrors the v0.3 `add_chunks` / `add_entities` / `iter_*` surface:
typed-model in, typed-model out, JSON round-trip stable.

The five workspace-row helpers (`add_workspace`, `get_workspace_by_id`,
`get_workspace_by_name`, `iter_workspaces`, `update_workspace_doc_ids`)
are what the workspace manager calls. The four concept-row helpers
(`add_concepts`, `get_concept`, `iter_concepts`, `concepts_for_workspace`)
back the shared concept lattice (§6.7) — concepts canonicalise into
one row per cluster, with `doc_ids_json` listing every member doc so
a workspace query can slice the lattice without re-walking the claim
table.

SPEC-REF: §6.7 (workspace = shared latent ontology), §8 (storage)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ctrldoc.models_v1 import Concept, Workspace
from ctrldoc.provenance import Provenance
from ctrldoc.store.sqlite import SQLiteStore


def _db(tmp_path: Path) -> Path:
    return tmp_path / "ctrldoc.db"


def _provenance() -> Provenance:
    return Provenance.create(
        playbook="workspace",
        playbook_version="1.0.0",
        index_hash="sha256:deadbeef",
        models={"judge": "qwen2.5:7b-instruct"},
    )


def _workspace(
    *,
    id: str = "ws-abc",
    name: str = "audit-2026",
    doc_ids: list[str] | None = None,
    induced_schema: dict[str, object] | None = None,
) -> Workspace:
    return Workspace(
        id=id,
        name=name,
        doc_ids=doc_ids if doc_ids is not None else [],
        induced_schema=induced_schema if induced_schema is not None else {},
        provenance=_provenance(),
    )


def _concept(
    *,
    id: str,
    canonical_name: str,
    doc_ids: list[str],
    primitive_type: str = "Entity",
    aliases: list[str] | None = None,
    mention_claim_ids: list[str] | None = None,
) -> Concept:
    return Concept(
        id=id,
        canonical_name=canonical_name,
        aliases=aliases if aliases is not None else [],
        primitive_type=primitive_type,  # type: ignore[arg-type]
        mention_claim_ids=mention_claim_ids if mention_claim_ids is not None else [],
        doc_ids=doc_ids,
    )


# --- workspace CRUD ---


def test_add_then_get_workspace_round_trips(tmp_path: Path) -> None:
    with SQLiteStore(_db(tmp_path)) as store:
        ws = _workspace(
            id="ws-1",
            name="audit-2026",
            doc_ids=["doc-a", "doc-b"],
            induced_schema={"nodes": ["Obligation"], "version": 1},
        )
        store.add_workspace(ws)
        got = store.get_workspace_by_id("ws-1")
    assert got == ws


def test_get_workspace_by_name_matches_by_unique_label(tmp_path: Path) -> None:
    with SQLiteStore(_db(tmp_path)) as store:
        store.add_workspace(_workspace(id="ws-1", name="audit-2026"))
        store.add_workspace(_workspace(id="ws-2", name="spec-vs-impl"))
        got = store.get_workspace_by_name("spec-vs-impl")
    assert got is not None
    assert got.id == "ws-2"


def test_get_workspace_missing_returns_none(tmp_path: Path) -> None:
    with SQLiteStore(_db(tmp_path)) as store:
        assert store.get_workspace_by_id("nope") is None
        assert store.get_workspace_by_name("nope") is None


def test_iter_workspaces_yields_in_creation_order(tmp_path: Path) -> None:
    with SQLiteStore(_db(tmp_path)) as store:
        store.add_workspace(_workspace(id="ws-1", name="a-first"))
        store.add_workspace(_workspace(id="ws-2", name="b-second"))
        store.add_workspace(_workspace(id="ws-3", name="c-third"))
        ids = [w.id for w in store.iter_workspaces()]
    assert ids == ["ws-1", "ws-2", "ws-3"]


def test_add_workspace_with_duplicate_id_replaces(tmp_path: Path) -> None:
    """INSERT OR REPLACE matches the v0.3 chunk/section/entity semantics."""
    with SQLiteStore(_db(tmp_path)) as store:
        store.add_workspace(_workspace(id="ws-1", name="audit-2026", doc_ids=["a"]))
        store.add_workspace(_workspace(id="ws-1", name="audit-2026", doc_ids=["a", "b"]))
        got = store.get_workspace_by_id("ws-1")
    assert got is not None
    assert got.doc_ids == ["a", "b"]


def test_update_workspace_doc_ids_appends_without_rewriting_other_fields(
    tmp_path: Path,
) -> None:
    """`workspace add` must mutate doc_ids only; name + schema + provenance stay pinned."""
    with SQLiteStore(_db(tmp_path)) as store:
        original = _workspace(
            id="ws-1",
            name="audit-2026",
            doc_ids=["doc-a"],
            induced_schema={"nodes": ["Obligation"]},
        )
        store.add_workspace(original)
        store.update_workspace_doc_ids("ws-1", ["doc-a", "doc-b"])
        got = store.get_workspace_by_id("ws-1")
    assert got is not None
    assert got.doc_ids == ["doc-a", "doc-b"]
    assert got.name == "audit-2026"
    assert got.induced_schema == {"nodes": ["Obligation"]}
    assert got.provenance == original.provenance


def test_update_workspace_doc_ids_on_missing_workspace_raises(tmp_path: Path) -> None:
    with SQLiteStore(_db(tmp_path)) as store, pytest.raises(KeyError):
        store.update_workspace_doc_ids("missing", ["doc-a"])


# --- concept CRUD ---


def test_add_then_get_concept_round_trips(tmp_path: Path) -> None:
    with SQLiteStore(_db(tmp_path)) as store:
        concept = _concept(
            id="k-1",
            canonical_name="OAuth2",
            doc_ids=["doc-a"],
            aliases=["OAuth 2.0", "RFC6749"],
            mention_claim_ids=["c-1", "c-2"],
            primitive_type="Entity",
        )
        store.add_concepts([concept])
        got = store.get_concept("k-1")
    assert got == concept


def test_add_concepts_is_idempotent_on_id(tmp_path: Path) -> None:
    """Re-adding under the same id replaces the row (INSERT OR REPLACE)."""
    with SQLiteStore(_db(tmp_path)) as store:
        store.add_concepts([_concept(id="k-1", canonical_name="OAuth2", doc_ids=["doc-a"])])
        store.add_concepts(
            [_concept(id="k-1", canonical_name="OAuth2", doc_ids=["doc-a", "doc-b"])]
        )
        got = store.get_concept("k-1")
    assert got is not None
    assert got.doc_ids == ["doc-a", "doc-b"]


def test_iter_concepts_yields_all_rows(tmp_path: Path) -> None:
    with SQLiteStore(_db(tmp_path)) as store:
        store.add_concepts(
            [
                _concept(id="k-1", canonical_name="OAuth2", doc_ids=["doc-a"]),
                _concept(id="k-2", canonical_name="JWT", doc_ids=["doc-b"]),
            ]
        )
        ids = sorted(c.id for c in store.iter_concepts())
    assert ids == ["k-1", "k-2"]


def test_concepts_for_workspace_returns_shared_lattice_slice(tmp_path: Path) -> None:
    """The workspace's concept lattice is the union of its members' concepts.

    Any concept whose `doc_ids` intersects the workspace's `doc_ids`
    appears in the slice — that is what §6.7 calls the *shared
    concept lattice*. Concepts unique to docs outside the workspace
    are filtered out.
    """
    with SQLiteStore(_db(tmp_path)) as store:
        ws = _workspace(id="ws-1", name="audit-2026", doc_ids=["doc-a", "doc-b"])
        store.add_workspace(ws)
        store.add_concepts(
            [
                _concept(id="k-1", canonical_name="OAuth2", doc_ids=["doc-a"]),
                _concept(id="k-2", canonical_name="JWT", doc_ids=["doc-a", "doc-b"]),
                _concept(id="k-3", canonical_name="OpenID", doc_ids=["doc-b"]),
                _concept(id="k-4", canonical_name="LDAP", doc_ids=["doc-c"]),  # outside ws
            ]
        )
        ids = sorted(c.id for c in store.concepts_for_workspace("ws-1"))
    assert ids == ["k-1", "k-2", "k-3"]


def test_concepts_for_workspace_on_empty_workspace_returns_empty(tmp_path: Path) -> None:
    with SQLiteStore(_db(tmp_path)) as store:
        store.add_workspace(_workspace(id="ws-1", name="empty", doc_ids=[]))
        store.add_concepts([_concept(id="k-1", canonical_name="OAuth2", doc_ids=["doc-a"])])
        assert list(store.concepts_for_workspace("ws-1")) == []


def test_concepts_for_workspace_on_missing_workspace_raises(tmp_path: Path) -> None:
    with SQLiteStore(_db(tmp_path)) as store, pytest.raises(KeyError):
        list(store.concepts_for_workspace("missing"))
