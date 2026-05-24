"""L2.5 workspace primitive — CRUD facade over the SQLite store.

A `Workspace` is a typed collection of doc-graphs that shares one
concept lattice (§6.7). `WorkspaceManager` is the small read/write
seam every callsite — CLI, Python API, MCP server — goes through to
create one, attach docs, list them, and inspect the shared lattice.
Heavy state (concepts, claims, cross-doc edges) lives in
`SQLiteStore`; the manager is the unit of API stability.

The id is content-derived (`ws-<sha256(name)[:16]>`) so the same name
yields the same id across processes — a property the verdict ledger
(§6.5) depends on for replay determinism. Name uniqueness is enforced
both here (clearer error message) and at the SQL layer (`UNIQUE`
constraint, defense in depth).

`WorkspaceInfo` is the read-only view `workspace info <name>` returns:
the underlying `Workspace`, the document count, and the size of the
shared concept lattice slice — the count of `Concept` rows whose
`doc_ids` intersect the workspace's member docs. Richer per-doc
breakdowns land alongside cross-doc edges (S-135).

SPEC-REF: §6.7 (workspace = shared latent ontology), §9 (CLI surface)
"""

from __future__ import annotations

import builtins
import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ctrldoc.models_v1 import Claim, Concept, Workspace
from ctrldoc.ops.cross_doc_edges import CrossDocEdgeInferer
from ctrldoc.provenance import Provenance
from ctrldoc.store.sqlite import SQLiteStore

_WORKSPACE_PLAYBOOK = "workspace"
_WORKSPACE_PLAYBOOK_VERSION = "1.0.0"
_WORKSPACE_ID_PREFIX = "ws-"
_WORKSPACE_ID_HASH_LEN = 16


class WorkspaceNotFoundError(KeyError):
    """Raised when a lookup hits an unknown workspace name or id."""


class WorkspaceAlreadyExistsError(ValueError):
    """Raised when `create()` is called with a name that already exists."""


@runtime_checkable
class DocResolver(Protocol):
    """Reads claims + concepts for a given `doc_id` from per-doc storage.

    The workspace store is a separate SQLite file from the per-doc
    indexes (`runs/indexes/<doc_hash>.db`). When `workspace add` lands
    a doc, the manager needs to bridge the doc's concepts into the
    workspace store and feed its claims to the cross-doc edge inferer.
    This Protocol is the seam — in production the CLI plugs in an
    adapter that opens the matching per-doc SQLiteStore; in unit tests
    a stub returns canned lists.
    """

    def claims_for_doc(self, doc_id: str) -> Iterable[Claim]: ...
    def concepts_for_doc(self, doc_id: str) -> Iterable[Concept]: ...


@dataclass(frozen=True)
class WorkspaceInfo:
    """Aggregate view returned by `WorkspaceManager.info`.

    `shared_concept_ids` is the slice of the concept lattice visible
    to this workspace per §6.7. `concept_count = len(shared_concept_ids)`
    is exposed separately so callers that only want the rollup don't
    have to materialize the list.
    """

    workspace: Workspace
    doc_count: int
    concept_count: int
    shared_concept_ids: list[str]


def _derive_workspace_id(name: str) -> str:
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return f"{_WORKSPACE_ID_PREFIX}{digest[:_WORKSPACE_ID_HASH_LEN]}"


def _validate_name(name: str) -> str:
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("workspace name must not be blank")
    return cleaned


def _validate_doc_id(doc_id: str) -> str:
    cleaned = doc_id.strip()
    if not cleaned:
        raise ValueError("doc_id must not be blank")
    return cleaned


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


class WorkspaceManager:
    """CRUD facade over `SQLiteStore` for the v1 `Workspace` primitive.

    The optional `doc_resolver` + `cross_doc_inferer` plug into
    `add()` to bridge §6.7 cross-doc state — concepts are mirrored
    into the workspace store so the shared-lattice rollup works, and
    `CrossDocEdgeInferer` runs over the incoming doc plus every
    existing member to populate the `cross_doc_edges` table. Both are
    opt-in: legacy call sites that pre-date S-156 pass neither and
    keep the unchanged CRUD-only surface.
    """

    def __init__(
        self,
        *,
        store: SQLiteStore,
        doc_resolver: DocResolver | None = None,
        cross_doc_inferer: CrossDocEdgeInferer | None = None,
    ) -> None:
        self._store = store
        self._doc_resolver = doc_resolver
        self._cross_doc_inferer = cross_doc_inferer

    # --- create ---

    def create(self, name: str) -> Workspace:
        """Create a new workspace; raise if the name is already in use."""
        cleaned = _validate_name(name)
        existing = self._store.get_workspace_by_name(cleaned)
        if existing is not None:
            raise WorkspaceAlreadyExistsError(f"workspace already exists: {cleaned!r}")
        workspace = Workspace(
            id=_derive_workspace_id(cleaned),
            name=cleaned,
            doc_ids=[],
            induced_schema={},
            provenance=Provenance.create(
                playbook=_WORKSPACE_PLAYBOOK,
                playbook_version=_WORKSPACE_PLAYBOOK_VERSION,
                index_hash="",
                models={},
            ),
        )
        self._store.add_workspace(workspace)
        return workspace

    # --- add ---

    def add(self, name: str, doc_id: str) -> Workspace:
        """Attach a doc to the workspace; idempotent and order-preserving.

        Re-adding the same doc is a no-op for the doc-list update so
        callers can replay the command without first checking
        membership. Document order is preserved across additions
        because cross-doc-edge enumeration walks pairs in a stable
        order to keep verdict-ledger replay deterministic (§6.5).

        When a `doc_resolver` is wired into the manager, the new doc's
        concepts are mirrored into the workspace store so
        `WorkspaceInfo.shared_concept_ids` surfaces them. When both a
        resolver AND a `cross_doc_inferer` are wired AND the workspace
        already had at least one prior member, the inferer runs over
        the new doc + every existing member and persists every emitted
        `TypedEdge` into `cross_doc_edges`. Per §6.7 the call budget
        stays linear in `|new_claims| * existing_doc_count * k`.
        """
        cleaned_doc = _validate_doc_id(doc_id)
        workspace = self._require_workspace(name)
        prior_doc_ids = list(workspace.doc_ids)
        new_doc_ids = _dedupe_preserve_order([*prior_doc_ids, cleaned_doc])
        doc_list_changed = new_doc_ids != prior_doc_ids
        if doc_list_changed:
            self._store.update_workspace_doc_ids(workspace.id, new_doc_ids)
            updated = self._store.get_workspace_by_id(workspace.id)
            assert updated is not None  # just updated; cannot have vanished.
        else:
            updated = workspace

        # Bridge concepts + run cross-doc edge inference. The bridge
        # runs on every call (idempotent on Concept.id) so a replay
        # picks up concepts produced by a later per-doc ER pass; the
        # cross-doc inference runs against the prior member list so
        # the very first add is a no-op even on replay.
        if self._doc_resolver is not None:
            self._bridge_concepts(cleaned_doc)
            if self._cross_doc_inferer is not None and prior_doc_ids:
                self._infer_cross_doc_edges(
                    workspace_id=updated.id,
                    new_doc_id=cleaned_doc,
                    existing_doc_ids=prior_doc_ids,
                )

        return updated

    # --- §6.7 helpers (concept bridge + cross-doc edge inference) ---

    def _bridge_concepts(self, doc_id: str) -> None:
        """Mirror the per-doc concepts into the workspace store.

        `Store.add_concepts` is idempotent on `Concept.id` so
        re-bridging is safe; concepts that already exist with the same
        id are overwritten by the freshest copy from the per-doc
        resolver.
        """
        resolver = self._doc_resolver
        assert resolver is not None
        concepts = list(resolver.concepts_for_doc(doc_id))
        if concepts:
            self._store.add_concepts(concepts)

    def _infer_cross_doc_edges(
        self,
        *,
        workspace_id: str,
        new_doc_id: str,
        existing_doc_ids: list[str],
    ) -> None:
        """Run `CrossDocEdgeInferer` over the new doc + prior members and persist.

        The inferer reads `claims_by_doc` and emits `TypedEdge` rows;
        we persist each row through `Store.append_cross_doc_edge`
        scoped to `workspace_id` so the SQL PRIMARY KEY
        `(workspace_id, src_claim_id, dst_claim_id, type)` keeps
        replays idempotent.
        """
        resolver = self._doc_resolver
        inferer = self._cross_doc_inferer
        assert resolver is not None
        assert inferer is not None
        claims_by_doc: dict[str, list[Claim]] = {
            new_doc_id: list(resolver.claims_for_doc(new_doc_id)),
        }
        for existing_doc in existing_doc_ids:
            claims_by_doc[existing_doc] = list(resolver.claims_for_doc(existing_doc))
        inference = inferer.infer(workspace_id=workspace_id, claims_by_doc=claims_by_doc)
        for edge in inference.edges:
            self._store.append_cross_doc_edge(workspace_id=workspace_id, edge=edge)

    # --- list ---

    def list(self) -> list[Workspace]:
        """Return every workspace in creation order."""
        return list(self._store.iter_workspaces())

    # --- info ---

    def info(self, name: str) -> WorkspaceInfo:
        """Return the read-only aggregate view of one workspace."""
        workspace = self._require_workspace(name)
        shared = sorted(c.id for c in self._store.concepts_for_workspace(workspace.id))
        return WorkspaceInfo(
            workspace=workspace,
            doc_count=len(workspace.doc_ids),
            concept_count=len(shared),
            shared_concept_ids=shared,
        )

    # --- get ---

    def get(self, name: str) -> Workspace:
        """Return the persisted `Workspace` or raise `WorkspaceNotFoundError`."""
        return self._require_workspace(name)

    # --- concept lattice slice (§6.7) ---

    def concepts_for_workspace(self, name: str) -> builtins.list[Concept]:
        """Return the workspace-visible concept-lattice slice (§6.7).

        A concept belongs to the slice when its `doc_ids` intersects
        the workspace's `doc_ids`. The list is sorted by concept id
        for byte-deterministic output across runs — the §6.7
        cross-doc-edge enumeration walks this exact order, so a stable
        sort here keeps the verdict-ledger replay deterministic
        (§6.5). Returns the empty list when no member doc has
        contributed concepts yet.
        """
        workspace = self._require_workspace(name)
        concepts = builtins.list(self._store.concepts_for_workspace(workspace.id))
        concepts.sort(key=lambda c: c.id)
        return concepts

    # --- internals ---

    def _require_workspace(self, name: str) -> Workspace:
        cleaned = _validate_name(name)
        existing = self._store.get_workspace_by_name(cleaned)
        if existing is None:
            raise WorkspaceNotFoundError(f"workspace not found: {cleaned!r}")
        return existing


__all__ = [
    "DocResolver",
    "WorkspaceAlreadyExistsError",
    "WorkspaceInfo",
    "WorkspaceManager",
    "WorkspaceNotFoundError",
]
