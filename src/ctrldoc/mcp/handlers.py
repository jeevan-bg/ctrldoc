"""Pure-Python + storage-backed MCP handler factory for the §6.10 tool surface.

The L4 tool surface in `ctrldoc.orch.tools` is a registry of input /
output schemas; engines plug in via
`dispatcher.register_handler(name, fn)`. The MCP server in
`ctrldoc.mcp.server` reuses that dispatcher verbatim — handlers wired
into the dispatcher are reachable over the JSON-RPC 2.0 stdio
transport described in §11.

This module ships the **pure-Python** + **storage-backed** waves of
handlers — the six tools whose engines need only structural
primitives or a per-doc SQLite store, no LLM call, no network.

Pure-Python handlers
--------------------

* ``subsumes`` → `claim_subsumption` from
  :mod:`ctrldoc.extract.galois`. The Galois lattice (§6.3) is a
  deterministic structural floor over the universal claim tuple
  (§6.2); Tier-2 NLI / LLM layers escalate from here. The handler
  reports ``confidence = 1.0`` because the structural verdict is
  exact at this layer — uncertainty enters only when the
  semantic-equivalence escalation runs.

* ``optimal_transport`` → `min_cost_transport` from
  :mod:`ctrldoc.ops.transport`. Pure stdlib min-cost flow over the
  caller-supplied marginals + cost matrix. The ``cost_fn_tag`` slot
  on the input schema is a verbatim passthrough so the verdict ledger
  (§6.5) can replay the same call deterministically.

* ``calibration`` → `fit_per_backend_ece` from
  :mod:`ctrldoc.extract.isotonic_calibration`. Iterates a per-backend
  mapping of ``(raw_scores, correct)`` pairs and reports the
  held-out ECE plus the held-out sample size per backend. With no
  data injected, the handler returns a valid but empty result — a
  faithful "no backends fit yet" answer rather than a refusal.

Storage-backed handlers
-----------------------

* ``get_claim`` → resolves a `claim_id` into the persisted `Claim`
  (§7) via an injected ``claim_record_lookup`` closure. A missing id
  raises `LookupError` so the MCP server lifts it into an
  ``isError=true`` envelope without fabricating a verdict (§13
  non-negotiable 3).

* ``lookup_concept`` → returns the `Concept` id whose
  ``canonical_name`` matches the supplied surface form, or `None` if
  no concept carries that name. The "None" branch is an explicit
  answer rather than a refusal — the schema surfaces
  ``concept_id: str | None`` precisely so a host can render
  "no concept by that name" without an extra round-trip.

* ``traverse`` → walks the typed-edge graph from the seed node
  filtered to a single ``edge_type`` and ``direction``, then harvests
  the top-`hops` reachable node ids by personalized-PageRank
  stationary probability. Backed by
  :class:`ctrldoc.retrieval.graph_walk.GraphWalkRetriever`. The seed
  itself is trimmed from the output so the caller sees only the
  nodes the walk reached.

Wiring policy
-------------

`register_default_handlers(dispatcher, deps)` registers exactly the
handlers whose dependencies are satisfied:

* ``optimal_transport`` — wires unconditionally (no deps).
* ``calibration`` — wires unconditionally (empty data is a valid
  degenerate answer; the host gets ``{ece_per_backend: {},
  sample_sizes: {}}``, which is a faithful "no backends fit yet"
  answer rather than refusal).
* ``subsumes`` — wires only when ``deps.claim_lookup`` is set. Without
  a lookup the handler cannot turn an id back into a `ClaimTuple`,
  so we leave it unregistered — the dispatcher then refuses the call
  with `ToolNotImplementedError`, honouring §13 non-negotiable 3
  ("every claim cited or refused").
* ``get_claim`` — wires only when ``deps.claim_record_lookup`` is set.
* ``lookup_concept`` — wires only when ``deps.concept_name_lookup`` is set.
* ``traverse`` — wires only when ``deps.typed_edges_supplier`` is set.

The factory returns the `frozenset` of wired tool names so callers
can log the surface they actually exposed.

`build_store_backed_deps(runs_path)` is the convenience factory that
opens every ``<runs_path>/indexes/*.db`` as a `SQLiteStore`, unions
the `Claim` / `Concept` / `TypedEdge` rows across stores, and returns
an `MCPHandlerDeps` ready to plug into the dispatcher. The runtime
cost is linear in the number of per-doc stores; the v1 workspace
cardinality (handful of docs) makes the scan negligible.

The downstream waves of MCP handlers ship in S-159 / S-160 (OT-backed:
``coverage`` / ``list_check`` / ``compare`` / ``merge``) and S-161
(LLM-backed: ``entails`` / ``qa`` / ``map``). Each will plug in via
the same `register_handler` seam — this module does not own those.

SPEC-REF: §6.10 (tool-using orchestrator), §11 (MCP server)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel

from ctrldoc.eval.claim_extraction import ClaimTuple
from ctrldoc.extract.galois import claim_subsumption
from ctrldoc.extract.isotonic_calibration import fit_per_backend_ece
from ctrldoc.models_v1 import Claim, Concept, TypedEdge
from ctrldoc.ops.transport import TransportProblem, min_cost_transport
from ctrldoc.orch.tools import (
    CalibrationInput,
    CalibrationOutput,
    GetClaimInput,
    GetClaimOutput,
    LookupConceptInput,
    LookupConceptOutput,
    OptimalTransportInput,
    OptimalTransportOutput,
    SubsumesInput,
    SubsumesOutput,
    ToolDispatcher,
    ToolHandler,
    TraverseInput,
    TraverseOutput,
)
from ctrldoc.retrieval.graph_walk import (
    GraphWalkConfig,
    GraphWalkRetriever,
)

# ---------------------------------------------------------------------------
# Dependency container
# ---------------------------------------------------------------------------


_STRUCTURAL_CONFIDENCE: float = 1.0
"""Confidence the Galois floor reports on every verdict.

The structural lattice is deterministic — the verdict is exact under
the §6.2 universal tuple. Probabilistic confidence only enters when
the Tier-2 NLI / LLM escalation runs (the storage-backed and
LLM-backed handler waves in S-158 / S-161 surface that). Pinning the
constant here keeps the magic out of the handler body.
"""


ClaimLookup = Callable[[str], ClaimTuple]
"""Resolve a `claim_id` string into the universal `ClaimTuple` shape.

Implementations may pull from the SQLite store, an in-memory dict, or
any other adapter — the handler treats this as an opaque function and
lets `LookupError` (or any other exception the lookup raises)
propagate so the MCP server lifts it into an `isError=true` envelope.
"""

CalibrationData = Mapping[str, tuple[Sequence[float], Sequence[int]]]
"""Per-backend `(raw_scores, correct)` pairs ready for `fit_per_backend_ece`.

The mapping key is the backend name (e.g. ``"ollama-qwen"``,
``"anthropic-sonnet"``) — surfaced verbatim in the
`CalibrationOutput.ece_per_backend` map so a host can route calls by
backend.
"""

ClaimRecordLookup = Callable[[str], Claim]
"""Resolve a `claim_id` string into the persisted `Claim` (§7).

Implementations may pull from a SQLite store, an in-memory dict, or
any other adapter — the handler treats this as an opaque function and
lets `LookupError` (or any other exception the lookup raises)
propagate so the MCP server lifts it into an `isError=true` envelope.
"""

ConceptNameLookup = Callable[[str], Concept | None]
"""Resolve a canonical name to its persisted `Concept`, or `None`.

The lookup must return `None` on a miss rather than raise — the
schema surfaces `concept_id: str | None` so the host can render
"no concept by that name" without an extra round-trip.
"""

TypedEdgesSupplier = Callable[[], Sequence[TypedEdge]]
"""Yield every persisted `TypedEdge` (§7) in the index.

Called once per `traverse` invocation. Implementations should be
cheap (a snapshot of an in-memory list) or cacheable (a `list(...)`
materialisation of a SQLite cursor) — the handler treats the result
as opaque and walks it via :class:`GraphWalkRetriever`.
"""


@dataclass(frozen=True)
class MCPHandlerDeps:
    """Injected dependencies for the pure-Python + storage-backed handler factory.

    Each field is optional — the factory registers exactly the handlers
    whose deps are satisfied. Missing deps means the dispatcher
    surfaces `ToolNotImplementedError` for that tool, which the MCP
    server lifts into an `isError=true` envelope.
    """

    claim_lookup: ClaimLookup | None = None
    """If set, the `subsumes` handler resolves `claim_a_id` / `claim_b_id`
    through this function and runs the Galois floor on the resulting
    `ClaimTuple` pair. If `None`, `subsumes` stays unwired."""

    calibration_data: CalibrationData | None = field(default=None)
    """If set, the `calibration` handler fits one `IsotonicCalibrator`
    per (backend, (raw, correct)) entry and reports the held-out ECE
    per backend. If `None` (or empty), the handler still wires and
    returns an empty result — a faithful "no backends fit yet"
    answer."""

    claim_record_lookup: ClaimRecordLookup | None = None
    """If set, the `get_claim` handler returns the persisted `Claim`
    (§7) for the supplied id. If `None`, `get_claim` stays unwired."""

    concept_name_lookup: ConceptNameLookup | None = None
    """If set, the `lookup_concept` handler resolves a canonical name to
    its concept id. If `None`, `lookup_concept` stays unwired."""

    typed_edges_supplier: TypedEdgesSupplier | None = None
    """If set, the `traverse` handler walks the typed-edge graph yielded
    by this supplier. If `None`, `traverse` stays unwired."""


# ---------------------------------------------------------------------------
# Individual handler factories — each returns a closure over its deps
# ---------------------------------------------------------------------------


def _make_subsumes_handler(claim_lookup: ClaimLookup) -> ToolHandler:
    """Bind `claim_subsumption` to a per-id lookup function.

    The handler signature on the dispatcher is `Callable[[BaseModel],
    Any]`; we narrow to `SubsumesInput` inside via `isinstance` so the
    typing remains sound. The dispatcher pre-validates the input
    against `SubsumesInput` before invoking us, so the runtime check is
    a defensive belt that also doubles as the static-narrowing claim.
    """

    def _handler(inp: BaseModel) -> SubsumesOutput:
        assert isinstance(inp, SubsumesInput), inp
        left = claim_lookup(inp.claim_a_id)
        right = claim_lookup(inp.claim_b_id)
        verdict = claim_subsumption(left, right)
        return SubsumesOutput(verdict=verdict, confidence=_STRUCTURAL_CONFIDENCE)

    return _handler


def _optimal_transport_handler(inp: BaseModel) -> OptimalTransportOutput:
    """Solve the transportation problem exactly. No deps required.

    The handler builds a `TransportProblem`, runs `min_cost_transport`,
    and lifts the resulting `TransportPlan` into the output schema.
    The `cost_fn_tag` slot is not used here — it travels verbatim
    through the input schema so the verdict ledger (§6.5) can replay
    the same call later.
    """
    assert isinstance(inp, OptimalTransportInput), inp
    problem = TransportProblem(
        source_weights=list(inp.source_weights),
        target_weights=list(inp.target_weights),
        cost_matrix=[list(row) for row in inp.cost_matrix],
    )
    plan = min_cost_transport(problem)
    return OptimalTransportOutput(
        flow=[list(row) for row in plan.flow],
        total_cost=plan.total_cost,
    )


def _make_get_claim_handler(claim_record_lookup: ClaimRecordLookup) -> ToolHandler:
    """Bind `Store.get_claim` to a per-id lookup function.

    The handler returns the persisted `Claim` (§7) unchanged. A miss
    raises `LookupError` so the MCP server lifts it into an
    `isError=true` envelope rather than fabricating a record (§13
    non-negotiable 3).
    """

    def _handler(inp: BaseModel) -> GetClaimOutput:
        assert isinstance(inp, GetClaimInput), inp
        claim = claim_record_lookup(inp.claim_id)
        return GetClaimOutput(claim=claim)

    return _handler


def _make_lookup_concept_handler(concept_name_lookup: ConceptNameLookup) -> ToolHandler:
    """Bind `lookup_concept` to a canonical-name resolver.

    A miss surfaces as `concept_id=None` rather than a refusal — the
    schema is explicit about this branch so callers can distinguish
    "no concept by that name" from "tool unavailable".
    """

    def _handler(inp: BaseModel) -> LookupConceptOutput:
        assert isinstance(inp, LookupConceptInput), inp
        concept = concept_name_lookup(inp.name)
        return LookupConceptOutput(concept_id=concept.id if concept is not None else None)

    return _handler


def _make_traverse_handler(typed_edges_supplier: TypedEdgesSupplier) -> ToolHandler:
    """Bind `GraphWalkRetriever.walk` to a typed-edge supplier.

    The handler filters the edge supply to the requested ``edge_type``
    and orients it by ``direction`` (forward / reverse / both) before
    constructing the retriever. The PPR walker yields a stationary
    distribution over every node reachable from the seed along the
    requested edge type; the handler then narrows to the nodes within
    ``hops`` BFS jumps so the caller's "depth-bounded neighbourhood"
    intent is honoured — and ranks them by PPR mass (descending,
    lex tiebreak) so the rendering is deterministic.

    The seed node itself is trimmed from the returned `node_ids` so
    the caller sees only the nodes the walk reached.
    """

    def _handler(inp: BaseModel) -> TraverseOutput:
        assert isinstance(inp, TraverseInput), inp
        all_edges = list(typed_edges_supplier())
        # Keep only edges of the requested type. The PPR walker already
        # filters by `EDGE_TYPE_WEIGHTS` membership, but pre-filtering
        # keeps it focused on the single edge type the caller asked
        # about — otherwise other supported types would dilute the
        # stationary mass.
        same_type = [e for e in all_edges if e.type == inp.edge_type]
        oriented: list[TypedEdge] = []
        if inp.direction in ("forward", "both"):
            oriented.extend(same_type)
        if inp.direction in ("reverse", "both"):
            # Reverse-direction walk: flip every edge's endpoints so the
            # walker treats `dst -> src` as a forward step. The flipped
            # edges keep the original type / confidence / citations so
            # `EDGE_TYPE_WEIGHTS` still applies.
            for e in same_type:
                oriented.append(
                    TypedEdge(
                        src_id=e.dst_id,
                        dst_id=e.src_id,
                        type=e.type,
                        confidence=e.confidence,
                        raw_score=e.raw_score,
                        citations=list(e.citations),
                        source=e.source,
                        paraphrase_votes=e.paraphrase_votes,
                    )
                )

        # First narrow to the strict hop-bounded reachable set via BFS
        # on the oriented edge list — the spec's `hops` argument is a
        # depth bound, not a top-N harvest size.
        reachable = _bfs_within_hops(seed=inp.node_id, edges=oriented, hops=inp.hops)
        if not reachable:
            return TraverseOutput(node_ids=[])

        # Now rank the reachable set by PPR stationary mass so the
        # rendering is deterministic and the "more-walked" neighbours
        # surface first. Harvest the full reachable set (the retriever
        # caps internally if it is smaller than `harvest_k`).
        retriever = GraphWalkRetriever(
            edges=oriented,
            concept_to_chunks={},  # we only need node ids, not chunks.
            config=GraphWalkConfig(harvest_k=len(reachable) + 1),
        )
        ranked = retriever.walk(seeds={inp.node_id: 1.0}).concept_ids
        node_ids = [n for n in ranked if n in reachable]
        # Append any reachable node the walker did not surface — guards
        # against pathological PPR mass distributions where a reachable
        # node sits below the harvest cap.
        for n in sorted(reachable):
            if n not in node_ids:
                node_ids.append(n)
        return TraverseOutput(node_ids=node_ids)

    return _handler


def _bfs_within_hops(*, seed: str, edges: Sequence[TypedEdge], hops: int) -> set[str]:
    """Set of distinct non-seed nodes reachable within `hops` jumps
    along the oriented edge list.

    Pure BFS — the spec semantics for `traverse(node_id, edge_type,
    direction, hops)` are "every node within `hops` steps along the
    requested edge type". The PPR ranker layered on top of this gives
    a deterministic surface order but does not change membership.
    """
    if hops <= 0:
        return set()
    adjacency: dict[str, list[str]] = {}
    for e in edges:
        adjacency.setdefault(e.src_id, []).append(e.dst_id)
    visited: set[str] = {seed}
    frontier: set[str] = {seed}
    for _ in range(hops):
        next_frontier: set[str] = set()
        for u in frontier:
            for v in adjacency.get(u, ()):
                if v not in visited:
                    visited.add(v)
                    next_frontier.add(v)
        if not next_frontier:
            break
        frontier = next_frontier
    visited.discard(seed)
    return visited


def _make_calibration_handler(
    calibration_data: CalibrationData | None,
) -> ToolHandler:
    """Bind `fit_per_backend_ece` to a per-backend labelled-batch mapping.

    Returns a handler that iterates the mapping in insertion order
    (deterministic for `dict` since Python 3.7) and emits one ECE
    plus held-out sample size per backend. A `None` or empty mapping
    surfaces as ``{ece_per_backend: {}, sample_sizes: {}}`` — a valid
    answer the host treats as "no backends fit yet".
    """

    def _handler(inp: BaseModel) -> CalibrationOutput:
        assert isinstance(inp, CalibrationInput), inp
        ece_per_backend: dict[str, float] = {}
        sample_sizes: dict[str, int] = {}
        if calibration_data:
            for backend, (raw_scores, correct) in calibration_data.items():
                # `fit_per_backend_ece` validates lengths and minimum size.
                ece, _calibrator = fit_per_backend_ece(
                    raw_scores=list(raw_scores),
                    correct=list(correct),
                )
                ece_per_backend[backend] = ece
                # Held-out half size matches the slice the helper evaluates on.
                half = len(raw_scores) // 2
                sample_sizes[backend] = len(raw_scores) - half
        return CalibrationOutput(
            ece_per_backend=ece_per_backend,
            sample_sizes=sample_sizes,
        )

    return _handler


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def register_default_handlers(
    *,
    dispatcher: ToolDispatcher,
    deps: MCPHandlerDeps,
) -> frozenset[str]:
    """Wire every pure-Python handler whose deps are satisfied.

    Returns the `frozenset` of tool names actually registered so the
    caller can log the surface it exposed. Tools whose deps are absent
    are intentionally left unregistered — the dispatcher then refuses
    those calls with `ToolNotImplementedError`, which the MCP server
    lifts into a structured `isError=true` envelope.

    The wiring policy:

    * ``optimal_transport`` — always wires. Pure stdlib, no deps.
    * ``calibration`` — always wires. Empty data => empty result.
    * ``subsumes`` — wires only if ``deps.claim_lookup`` is set.
    * ``get_claim`` — wires only if ``deps.claim_record_lookup`` is set.
    * ``lookup_concept`` — wires only if ``deps.concept_name_lookup`` is set.
    * ``traverse`` — wires only if ``deps.typed_edges_supplier`` is set.

    Re-registering replaces the previous handler (the dispatcher
    documents this), so callers can layer richer LLM-backed handlers
    (S-161) on top of this factory's pure-Python + storage-backed
    floor.
    """
    wired: set[str] = set()

    dispatcher.register_handler("optimal_transport", _optimal_transport_handler)
    wired.add("optimal_transport")

    dispatcher.register_handler("calibration", _make_calibration_handler(deps.calibration_data))
    wired.add("calibration")

    if deps.claim_lookup is not None:
        dispatcher.register_handler("subsumes", _make_subsumes_handler(deps.claim_lookup))
        wired.add("subsumes")

    if deps.claim_record_lookup is not None:
        dispatcher.register_handler("get_claim", _make_get_claim_handler(deps.claim_record_lookup))
        wired.add("get_claim")

    if deps.concept_name_lookup is not None:
        dispatcher.register_handler(
            "lookup_concept", _make_lookup_concept_handler(deps.concept_name_lookup)
        )
        wired.add("lookup_concept")

    if deps.typed_edges_supplier is not None:
        dispatcher.register_handler("traverse", _make_traverse_handler(deps.typed_edges_supplier))
        wired.add("traverse")

    return frozenset(wired)


# ---------------------------------------------------------------------------
# Per-installation factory — open every `<runs_path>/indexes/*.db`
# ---------------------------------------------------------------------------


def build_store_backed_deps(*, runs_path: Path) -> MCPHandlerDeps:
    """Open every per-doc SQLite store under `runs_path/indexes/` and
    return an `MCPHandlerDeps` wired with the three storage-backed
    closures.

    The per-installation layout (set up by `cli._build_per_doc_backends`)
    persists one SQLite file per doc at
    ``<runs_path>/indexes/<doc_hash>.db``; this factory walks that
    directory, opens each `.db` file as a `SQLiteStore`, and aggregates
    the row-level views the handlers need.

    Materialisation strategy:

    * `claim_record_lookup` keeps a dict-of-stores cache so a hit
      avoids reopening the SQLite connection on every call. A miss in
      every store raises `LookupError` — the MCP server lifts that into
      an `isError=true` envelope.
    * `concept_name_lookup` iterates the union snapshot in canonical
      order; exact `canonical_name` match short-circuits.
    * `typed_edges_supplier` returns the cached `TypedEdge` list
      materialised once at construction; the v1 workspace cardinality
      makes this trivial.

    When the `indexes/` directory is absent or empty, the closures
    still wire (so the dispatcher can serve them) but report empty /
    miss results — a faithful "no docs ingested yet" answer rather
    than a refusal.
    """
    indexes_dir = Path(runs_path) / "indexes"
    db_paths: list[Path] = (
        sorted(p for p in indexes_dir.glob("*.db") if p.is_file()) if indexes_dir.is_dir() else []
    )
    # One open SQLiteStore per per-doc file. Held for the lifetime of
    # the returned closures; the typical caller is the MCP serve_stdio
    # entry point, so we accept the open-fd footprint for the duration
    # of the stdio loop.
    from ctrldoc.store.sqlite import SQLiteStore

    stores: list[SQLiteStore] = []
    materialised_edges: list[TypedEdge] = []
    materialised_concepts: list[Concept] = []
    for db_path in db_paths:
        store = SQLiteStore(db_path)
        stores.append(store)
        materialised_edges.extend(store.iter_typed_edges())
        materialised_concepts.extend(store.iter_concepts())

    def _claim_record_lookup(claim_id: str) -> Claim:
        for store in stores:
            claim = store.get_claim(claim_id)
            if claim is not None:
                return claim
        raise LookupError(f"unknown claim id across {len(stores)} store(s): {claim_id!r}")

    def _concept_name_lookup(name: str) -> Concept | None:
        for concept in materialised_concepts:
            if concept.canonical_name == name:
                return concept
        return None

    def _typed_edges_supplier() -> list[TypedEdge]:
        return list(materialised_edges)

    return MCPHandlerDeps(
        claim_record_lookup=_claim_record_lookup,
        concept_name_lookup=_concept_name_lookup,
        typed_edges_supplier=_typed_edges_supplier,
    )


__all__ = [
    "CalibrationData",
    "ClaimLookup",
    "ClaimRecordLookup",
    "ConceptNameLookup",
    "MCPHandlerDeps",
    "TypedEdgesSupplier",
    "build_store_backed_deps",
    "register_default_handlers",
]
