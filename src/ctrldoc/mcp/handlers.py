"""Pure-Python + storage-backed + OT-backed MCP handler factory for the §6.10 tool surface.

The L4 tool surface in `ctrldoc.orch.tools` is a registry of input /
output schemas; engines plug in via
`dispatcher.register_handler(name, fn)`. The MCP server in
`ctrldoc.mcp.server` reuses that dispatcher verbatim — handlers wired
into the dispatcher are reachable over the JSON-RPC 2.0 stdio
transport described in §11.

This module ships the **pure-Python**, **storage-backed**, and
**OT-backed** waves of handlers — the eight tools whose engines need
only structural primitives, a per-doc SQLite store, or the
optimal-transport core in `ctrldoc.ops.coverage`; no LLM call, no
network.

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

OT-backed handlers
------------------

* ``coverage`` → `ops.coverage.coverage` over the persisted `Claim`
  rows of two docs in a workspace. The handler resolves
  `target_doc_id` and `source_doc_id` to lists of `Claim` via the
  injected ``claims_for_doc_supplier``, converts each `Claim` back to
  the §6.2 universal tuple via
  :func:`ctrldoc.extract.claim_persistence.claim_to_tuple`, runs the
  §6.6 transport reduction with the injected ``nli_scorer``, then
  lifts the per-target `Covered` / `Missing` verdicts into a full §7
  ``CoverageReport`` — pinned to the workspace id, target id, source
  id, with one `CoverageVerdict` per target claim carrying the
  aligned source-claim ids, the transport cost, and the calibrated
  confidence.

* ``list_check`` → `ops.coverage.list_check` with the items list
  parsed as a tiny target doc per §6.6 framing
  (``list_check(items, D) == coverage(items → D)``). Each
  `ListCheckItem` becomes a `ClaimTuple` whose `subject` slot carries
  the item text; the persisted doc claims act as sources. Verdicts
  surface as the four-class partition shared with `coverage` —
  S-159 surfaces `Covered` / `Missing` only; richer partials land
  with the calibrated edge layer in later slices.

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
* ``coverage`` / ``list_check`` / ``compare`` / ``merge`` — each
  wires only when both ``deps.claims_for_doc_supplier`` and
  ``deps.nli_scorer`` are set. Either dep missing leaves all four
  tools unregistered so the dispatcher refuses the call rather than
  fabricating a verdict.

The factory returns the `frozenset` of wired tool names so callers
can log the surface they actually exposed.

`build_store_backed_deps(runs_path)` is the convenience factory that
opens every ``<runs_path>/indexes/*.db`` as a `SQLiteStore`, unions
the `Claim` / `Concept` / `TypedEdge` rows across stores, and returns
an `MCPHandlerDeps` ready to plug into the dispatcher. The runtime
cost is linear in the number of per-doc stores; the v1 workspace
cardinality (handful of docs) makes the scan negligible.

The OT-backed `compare` / `merge` handlers reduce to
:func:`ctrldoc.ops.compare.compare` and
:func:`ctrldoc.ops.merge.merge` respectively. Compare resolves every
input `doc_id` to its persisted `Claim` list, forms per-concept
clusters by matching universal `(subject, predicate, object)`
triplets across pairs, and emits one row per cluster per `(doc_i,
doc_j)` pair with `i < j`. Merge resolves every input `doc_id`
identically, lifts each row into an `InputClaim`, and runs the §6.6
union-find + Galois-join reduction over the union — the output's
`MergedDoc` envelope carries one `cluster_id` per cluster and one
representative claim id per cluster (the first member id in input
order, matching the engine's deterministic tiebreak).

The final wave (`entails` / `qa` / `map`) lands in S-161; each will
plug in via the same `register_handler` seam — this module does not
own those.

SPEC-REF: §6.10 (tool-using orchestrator), §11 (MCP server)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ctrldoc.eval.claim_extraction import ClaimTuple, DocTypeLiteral
from ctrldoc.eval.compare import ConceptComparisonInput
from ctrldoc.eval.merge import InputClaim, MergedCluster
from ctrldoc.extract.claim_persistence import claim_to_tuple
from ctrldoc.extract.galois import claim_subsumption
from ctrldoc.extract.isotonic_calibration import fit_per_backend_ece
from ctrldoc.models_v1 import (
    Claim,
    Concept,
    CoverageReport,
    CoverageSummary,
    CoverageVerdict,
    ProofTrace,
    TypedEdge,
    VerdictLiteral,
)
from ctrldoc.ops.compare import CompareConfig
from ctrldoc.ops.compare import (
    compare as ops_compare,
)
from ctrldoc.ops.coverage import (
    CoverageConfig,
    CoverageResult,
    NLIScorer,
)
from ctrldoc.ops.coverage import (
    coverage as ops_coverage,
)
from ctrldoc.ops.coverage import (
    list_check as ops_list_check,
)
from ctrldoc.ops.merge import MergeConfig
from ctrldoc.ops.merge import (
    merge as ops_merge,
)
from ctrldoc.ops.transport import TransportProblem, min_cost_transport
from ctrldoc.orch.tools import (
    CalibrationInput,
    CalibrationOutput,
    CompareInput,
    CompareOutput,
    CompareReport,
    CoverageInput,
    CoverageOutput,
    GetClaimInput,
    GetClaimOutput,
    ListCheckInput,
    ListCheckItem,
    ListCheckOutput,
    ListCheckVerdict,
    LookupConceptInput,
    LookupConceptOutput,
    MergedDoc,
    MergeInput,
    MergeOutput,
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

ClaimsForDocSupplier = Callable[[str], Sequence[Claim]]
"""Yield the persisted `Claim` rows belonging to one `doc_id`.

Called by the OT-backed `coverage` / `list_check` handlers — each
runs at most twice per invocation (once per doc). Implementations
may pull from a SQLite store, an in-memory dict, or any other
adapter. An unknown doc id should surface as an empty sequence (the
faithful "no claims for this doc" answer); the handler treats that
case as the all-Missing degenerate path.
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

    claims_for_doc_supplier: ClaimsForDocSupplier | None = None
    """If set together with `nli_scorer`, the OT-backed `coverage` and
    `list_check` handlers resolve per-doc claim lists through this
    function. Either dep missing leaves both tools unwired."""

    nli_scorer: NLIScorer | None = None
    """If set together with `claims_for_doc_supplier`, the OT-backed
    `coverage` and `list_check` handlers consume this scorer as the
    §6.6 entailment backend. Either dep missing leaves both tools
    unwired."""


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


# ---------------------------------------------------------------------------
# OT-backed `coverage` / `list_check` handlers — §6.6 transport reduction
# ---------------------------------------------------------------------------


# Calibrated confidence for §7 `CoverageVerdict` rows derived from the
# transport plan. The slack/real mass split is the natural calibrated
# probability the §6.6 reduction emits: `real_mass` for `Covered`
# (the column's mass came from real sources), `slack_mass` for
# `Missing` (the slack column dominated). Pre-isotonic; the §6.5
# calibration layer refines these scores later.
_VERDICT_TO_CALIBRATED_MASS: dict[VerdictLiteral, str] = {
    "Covered": "real_mass",
    "Missing": "slack_mass",
}


def _build_coverage_report(
    *,
    workspace_id: str,
    target_doc_id: str,
    source_doc_id: str,
    target_claims: Sequence[Claim],
    source_claims: Sequence[Claim],
    result: CoverageResult,
) -> CoverageReport:
    """Lift a `CoverageResult` plus its inputs into a §7 `CoverageReport`.

    Per-claim row carries the target's persisted id, the verdict, the
    aligned source-claim ids (resolved through the source list via the
    transport plan's per-column readout), the transport cost, and the
    calibrated confidence (the column's real or slack mass depending
    on the verdict). The summary's four rates partition the target
    claim count.

    Empty-target short-circuit: every rate is zero except `covered_rate`
    which inherits the vacuous 1.0 — `CoverageSummary` enforces the
    sum-to-one invariant, so the all-covered degenerate is the only
    valid "no targets" surface. Empty-source: `assignments` carries
    one all-Missing row per target (every target is uncovered), which
    is what the assembled report reflects.
    """
    per_claim: list[CoverageVerdict] = []
    counts = {"Covered": 0, "Partial": 0, "Missing": 0, "Contradicted": 0}
    for j, target in enumerate(target_claims):
        verdict = result.verdicts[j]
        assignment = result.assignments[j]
        aligned = [source_claims[i].id for i in assignment.aligned_source_indices]
        mass_attr = _VERDICT_TO_CALIBRATED_MASS.get(verdict)
        calibrated = (
            getattr(assignment, mass_attr) if mass_attr is not None else assignment.real_mass
        )
        # Clamp to the unit interval — guards against IEEE-754 drift
        # in the plan's marginals (the engine balances within 1e-9).
        calibrated = max(0.0, min(1.0, calibrated))
        per_claim.append(
            CoverageVerdict(
                target_claim_id=target.id,
                verdict=verdict,
                aligned_source_claims=aligned,
                transport_cost=assignment.transport_cost,
                calibrated_confidence=calibrated,
                trace=ProofTrace(steps=["coverage", "render_claim", "nli_entail", "transport"]),
            )
        )
        counts[verdict] += 1

    total = max(1, len(target_claims))  # avoid div-by-zero on empty target
    if not target_claims:
        summary = CoverageSummary(
            covered_rate=1.0,
            partial_rate=0.0,
            missing_rate=0.0,
            contradicted_rate=0.0,
        )
    else:
        summary = CoverageSummary(
            covered_rate=counts["Covered"] / total,
            partial_rate=counts["Partial"] / total,
            missing_rate=counts["Missing"] / total,
            contradicted_rate=counts["Contradicted"] / total,
        )

    return CoverageReport(
        workspace_id=workspace_id,
        target_doc_id=target_doc_id,
        source_doc_id=source_doc_id,
        per_claim=per_claim,
        summary=summary,
    )


def _make_coverage_handler(
    *,
    claims_for_doc_supplier: ClaimsForDocSupplier,
    nli_scorer: NLIScorer,
) -> ToolHandler:
    """Bind `ops.coverage.coverage` to the §6.10 `coverage` schema.

    Resolves `target_doc_id` and `source_doc_id` to persisted `Claim`
    lists via the supplier, converts each to the §6.2 universal tuple,
    runs the transport reduction, and lifts the result into a
    `CoverageReport` pinned to the call's workspace id.
    """

    def _handler(inp: BaseModel) -> CoverageOutput:
        assert isinstance(inp, CoverageInput), inp
        target_claims = list(claims_for_doc_supplier(inp.target_doc_id))
        source_claims = list(claims_for_doc_supplier(inp.source_doc_id))
        target_tuples = [claim_to_tuple(c) for c in target_claims]
        source_tuples = [claim_to_tuple(c) for c in source_claims]
        result = ops_coverage(
            source=source_tuples,
            target=target_tuples,
            scorer=nli_scorer,
            config=CoverageConfig(),
        )
        report = _build_coverage_report(
            workspace_id=inp.workspace_id,
            target_doc_id=inp.target_doc_id,
            source_doc_id=inp.source_doc_id,
            target_claims=target_claims,
            source_claims=source_claims,
            result=result,
        )
        return CoverageOutput(report=report)

    return _handler


def _items_to_target_tuples(items: Sequence[ListCheckItem]) -> list[ClaimTuple]:
    """Convert `ListCheckItem` rows into §6.2 tuples — subject = item text.

    The §6.6 framing of `list_check(items, D) == coverage(items → D)`
    treats each item as a degenerate claim: the rendered surface is the
    item text. The cleanest mapping is to put the text in the
    `subject` slot and leave `predicate` / `object` empty, so the
    `ops.coverage._render_claim` helper produces the verbatim text.
    """
    tuples: list[ClaimTuple] = []
    for item in items:
        tuples.append(
            ClaimTuple(
                subject=item.text,
                predicate="",
                object="",
                polarity="affirmative",
                modality="asserted",
                qualifier="",
            )
        )
    return tuples


def _make_list_check_handler(
    *,
    claims_for_doc_supplier: ClaimsForDocSupplier,
    nli_scorer: NLIScorer,
) -> ToolHandler:
    """Bind `ops.coverage.list_check` to the §6.10 `list_check` schema.

    Each `ListCheckItem` becomes a `ClaimTuple` with the item text as
    the `subject` slot (predicate / object empty); the persisted doc
    claims act as sources. Output is one `ListCheckVerdict` per item in
    input order, verdict from the shared four-class partition and
    confidence equal to the column's real-mass split for `Covered`
    targets, slack-mass for `Missing` ones.
    """

    def _handler(inp: BaseModel) -> ListCheckOutput:
        assert isinstance(inp, ListCheckInput), inp
        target_tuples = _items_to_target_tuples(inp.items)
        source_claims = list(claims_for_doc_supplier(inp.doc_id))
        source_tuples = [claim_to_tuple(c) for c in source_claims]
        result = ops_list_check(
            items=target_tuples,
            doc=source_tuples,
            scorer=nli_scorer,
            config=CoverageConfig(),
        )
        verdicts: list[ListCheckVerdict] = []
        for j, item in enumerate(inp.items):
            verdict = result.verdicts[j]
            assignment = result.assignments[j]
            mass_attr = _VERDICT_TO_CALIBRATED_MASS.get(verdict)
            calibrated = (
                getattr(assignment, mass_attr) if mass_attr is not None else assignment.real_mass
            )
            calibrated = max(0.0, min(1.0, calibrated))
            verdicts.append(
                ListCheckVerdict(
                    item_id=item.item_id,
                    verdict=verdict,
                    confidence=calibrated,
                )
            )
        return ListCheckOutput(verdicts=verdicts)

    return _handler


# ---------------------------------------------------------------------------
# OT-backed `compare` / `merge` handlers — §6.6 transport reduction
# ---------------------------------------------------------------------------


def _claims_to_svo_key(claim: Claim) -> tuple[str, str, str]:
    """Cluster key over the universal `(subject, predicate, object)` triplet.

    Two persisted claims with identical SVO from different docs form
    one §6.6 compare cluster — modality / qualifier / polarity
    differences then surface as StrengthA / StrengthB via the Galois
    floor. Claims whose SVO is not shared by the other doc form a Gap
    singleton. Empty subject / object fields collapse to the empty
    string (the `claim_to_tuple` convention) so the key remains
    well-typed even for degenerate persisted rows.
    """
    return (claim.subject or "", claim.predicate, claim.object or "")


def _build_compare_pair_rows(
    *,
    a_doc_id: str,
    b_doc_id: str,
    a_claims: Sequence[Claim],
    b_claims: Sequence[Claim],
    scorer: NLIScorer,
) -> list[dict[str, Any]]:
    """One row per cluster across the doc pair, in deterministic order.

    Build the per-cluster `ConceptComparisonInput` list keyed on the
    universal SVO triplet. Pairs present in both docs surface as a
    two-sided cluster (the Galois / NLI fallback decides StrengthA vs
    StrengthB); pairs present in only one doc surface as a one-sided
    Gap cluster. Cluster order is fixed by the union of SVO keys in
    `a_claims` order followed by `b_claims` order — that guarantees
    byte-deterministic rows across runs even when the underlying dicts
    iterate differently.
    """
    a_by_key: dict[tuple[str, str, str], Claim] = {}
    a_order: list[tuple[str, str, str]] = []
    for c in a_claims:
        key = _claims_to_svo_key(c)
        # First-claim-with-this-key wins — matches the eval substrate's
        # deterministic input-order tiebreak (`ops.compare` itself does
        # not collapse intra-doc duplicates, so the first persisted row
        # is the canonical one for the cluster).
        if key not in a_by_key:
            a_by_key[key] = c
            a_order.append(key)
    b_by_key: dict[tuple[str, str, str], Claim] = {}
    b_order: list[tuple[str, str, str]] = []
    for c in b_claims:
        key = _claims_to_svo_key(c)
        if key not in b_by_key:
            b_by_key[key] = c
            b_order.append(key)

    # Union of keys in deterministic order — a's first, then b's
    # additions. The eval substrate's cluster ids are caller-supplied;
    # here we synthesise stable ids from the pair label and a 0-based
    # ordinal so re-runs over identical inputs land identical rows.
    seen_keys: set[tuple[str, str, str]] = set()
    cluster_keys: list[tuple[str, str, str]] = []
    for key in a_order:
        if key not in seen_keys:
            seen_keys.add(key)
            cluster_keys.append(key)
    for key in b_order:
        if key not in seen_keys:
            seen_keys.add(key)
            cluster_keys.append(key)

    clusters: list[ConceptComparisonInput] = []
    paired_claim_ids: list[tuple[str | None, str | None]] = []
    for ordinal, key in enumerate(cluster_keys):
        a_claim = a_by_key.get(key)
        b_claim = b_by_key.get(key)
        clusters.append(
            ConceptComparisonInput(
                id=f"cluster-{a_doc_id}-{b_doc_id}-{ordinal}",
                # Label is the rendered SVO triplet so the host has a
                # human-readable handle without re-rendering.
                label=" ".join(p for p in key if p),
                a_claim=claim_to_tuple(a_claim) if a_claim is not None else None,
                b_claim=claim_to_tuple(b_claim) if b_claim is not None else None,
            )
        )
        paired_claim_ids.append(
            (
                a_claim.id if a_claim is not None else None,
                b_claim.id if b_claim is not None else None,
            )
        )

    if not clusters:
        return []

    result = ops_compare(clusters=clusters, scorer=scorer, config=CompareConfig())
    rows: list[dict[str, Any]] = []
    for cluster, verdict, (a_id, b_id) in zip(
        clusters, result.verdicts, paired_claim_ids, strict=True
    ):
        row: dict[str, Any] = {
            "cluster_id": cluster.id,
            "label": cluster.label,
            "verdict": verdict,
            "a_doc_id": a_doc_id,
            "b_doc_id": b_doc_id,
        }
        if a_id is not None:
            row["a_claim_id"] = a_id
        if b_id is not None:
            row["b_claim_id"] = b_id
        rows.append(row)
    return rows


def _make_compare_handler(
    *,
    claims_for_doc_supplier: ClaimsForDocSupplier,
    nli_scorer: NLIScorer,
) -> ToolHandler:
    """Bind `ops.compare.compare` to the §6.10 `compare` schema.

    Resolves every `doc_id` in the input to its persisted `Claim` list,
    forms per-concept clusters by matching universal SVO triplets, and
    runs the §6.6 reduction per `(doc_i, doc_j)` pair with `i < j`. For
    two docs the row set is the direct reduction; for `N > 2` docs the
    rows surface pairwise comparisons in declaration order.
    """

    def _handler(inp: BaseModel) -> CompareOutput:
        assert isinstance(inp, CompareInput), inp
        doc_ids = list(inp.doc_ids)
        per_doc_claims: dict[str, list[Claim]] = {
            doc_id: list(claims_for_doc_supplier(doc_id)) for doc_id in doc_ids
        }
        rows: list[dict[str, Any]] = []
        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                rows.extend(
                    _build_compare_pair_rows(
                        a_doc_id=doc_ids[i],
                        b_doc_id=doc_ids[j],
                        a_claims=per_doc_claims[doc_ids[i]],
                        b_claims=per_doc_claims[doc_ids[j]],
                        scorer=nli_scorer,
                    )
                )
        report = CompareReport(workspace_id=inp.workspace_id, doc_ids=doc_ids, rows=rows)
        return CompareOutput(report=report)

    return _handler


# Default `doc_type` slot stamped on every `InputClaim` the §6.6 merge
# engine consumes. The persisted `Claim` row (§7) does not carry a
# doc-type tag, but `InputClaim` enforces a `DocTypeLiteral` for
# schema completeness; the engine itself never reads the slot — its
# union-find + Galois-join reduction keys on the claim tuple alone.
_MERGE_DEFAULT_DOC_TYPE: DocTypeLiteral = "spec"


def _build_merge_input_claims(
    doc_id_to_claims: Mapping[str, Sequence[Claim]],
    doc_ids: Sequence[str],
) -> list[InputClaim]:
    """Lift persisted `Claim` rows into `InputClaim` rows for the §6.6 engine.

    Order is `(doc_id_position, intra-doc input order)` so the cluster
    member ordering the engine emits is stable across runs.
    """
    inputs: list[InputClaim] = []
    for doc_id in doc_ids:
        for claim in doc_id_to_claims.get(doc_id, ()):
            inputs.append(
                InputClaim(
                    id=claim.id,
                    doc_id=doc_id,
                    doc_type=_MERGE_DEFAULT_DOC_TYPE,
                    claim=claim_to_tuple(claim),
                )
            )
    return inputs


def _build_merge_output_clusters_for_invariant(
    *,
    doc_id_to_claims: Mapping[str, Sequence[Claim]],
    doc_ids: Sequence[str],
    nli_scorer: NLIScorer,
) -> list[MergedCluster]:
    """Test seam: re-run the §6.6 merge engine and surface its
    `MergedCluster` view so callers can assert the §13 loss invariant
    against the persisted member sets.

    Kept module-private; only the handler's release-gate test imports
    it. The actual `MergeOutput` envelope the MCP handler surfaces is
    the §10 schema (cluster ids + representative ids) — the full
    cluster shape stays internal to avoid leaking the engine's
    `MergedCluster` model out of the §6.10 surface.
    """
    inputs = _build_merge_input_claims(doc_id_to_claims, doc_ids)
    if not inputs:
        return []
    result = ops_merge(input_claims=inputs, scorer=nli_scorer, config=MergeConfig())
    return list(result.output.clusters)


def _make_merge_handler(
    *,
    claims_for_doc_supplier: ClaimsForDocSupplier,
    nli_scorer: NLIScorer,
) -> ToolHandler:
    """Bind `ops.merge.merge` to the §6.10 `merge` schema.

    Resolves every `doc_id` in the input to its persisted `Claim`
    list, lifts each row into an `InputClaim`, and runs the §6.6
    union-find + Galois-join reduction over the union. Output is a
    `MergedDoc` carrying one `cluster_id` per cluster and one
    representative claim id per cluster — the first member id in
    input order, matching the engine's deterministic input-order
    tiebreak for the Galois-join surface representative.
    """

    def _handler(inp: BaseModel) -> MergeOutput:
        assert isinstance(inp, MergeInput), inp
        doc_ids = list(inp.doc_ids)
        per_doc_claims: dict[str, list[Claim]] = {
            doc_id: list(claims_for_doc_supplier(doc_id)) for doc_id in doc_ids
        }
        inputs = _build_merge_input_claims(per_doc_claims, doc_ids)
        if not inputs:
            merged = MergedDoc(
                workspace_id=inp.workspace_id,
                cluster_ids=[],
                representative_claim_ids=[],
            )
            return MergeOutput(merged=merged)
        result = ops_merge(input_claims=inputs, scorer=nli_scorer, config=MergeConfig())
        cluster_ids: list[str] = []
        representative_claim_ids: list[str] = []
        for cluster in result.output.clusters:
            cluster_ids.append(cluster.id)
            # First member id is the deterministic input-order
            # representative — `MergedCluster.member_claim_ids` is
            # ordered by input position, so member_claim_ids[0] is
            # the canonical pick.
            representative_claim_ids.append(cluster.member_claim_ids[0])
        merged = MergedDoc(
            workspace_id=inp.workspace_id,
            cluster_ids=cluster_ids,
            representative_claim_ids=representative_claim_ids,
        )
        return MergeOutput(merged=merged)

    return _handler


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
    * ``coverage`` / ``list_check`` / ``compare`` / ``merge`` — each
      wires only if BOTH ``deps.claims_for_doc_supplier`` and
      ``deps.nli_scorer`` are set. Either dep missing leaves all four
      tools unwired.

    Re-registering replaces the previous handler (the dispatcher
    documents this), so callers can layer richer LLM-backed handlers
    (S-161) on top of this factory's pure-Python + storage-backed +
    OT-backed floor.
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

    if deps.claims_for_doc_supplier is not None and deps.nli_scorer is not None:
        dispatcher.register_handler(
            "coverage",
            _make_coverage_handler(
                claims_for_doc_supplier=deps.claims_for_doc_supplier,
                nli_scorer=deps.nli_scorer,
            ),
        )
        wired.add("coverage")
        dispatcher.register_handler(
            "list_check",
            _make_list_check_handler(
                claims_for_doc_supplier=deps.claims_for_doc_supplier,
                nli_scorer=deps.nli_scorer,
            ),
        )
        wired.add("list_check")
        dispatcher.register_handler(
            "compare",
            _make_compare_handler(
                claims_for_doc_supplier=deps.claims_for_doc_supplier,
                nli_scorer=deps.nli_scorer,
            ),
        )
        wired.add("compare")
        dispatcher.register_handler(
            "merge",
            _make_merge_handler(
                claims_for_doc_supplier=deps.claims_for_doc_supplier,
                nli_scorer=deps.nli_scorer,
            ),
        )
        wired.add("merge")

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
    "ClaimsForDocSupplier",
    "ConceptNameLookup",
    "MCPHandlerDeps",
    "TypedEdgesSupplier",
    "build_store_backed_deps",
    "register_default_handlers",
]
