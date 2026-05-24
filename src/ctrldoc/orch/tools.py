"""L4 tool surface — fixed Pydantic schemas + forced-tool-call dispatcher.

SPEC §6.10 fixes the L4 orchestrator's exposed surface at exactly 13
tools. The orchestrator (and the §11 MCP server) never reasons in
free-form text; it picks a tool, fills its input schema, and ships a
structured object back. This module is the substrate every callsite
goes through to do that.

It has two responsibilities:

1. **Schemas.** One Pydantic `input_model` and one `output_model` per
   tool. The schemas live here (not at each engine's site) so the
   surface is one stable contract — `ctrldoc mcp serve` (§11) and the
   in-process Python API read the same Pydantic types.
2. **Dispatch.** `ToolDispatcher` validates raw inputs, routes to the
   registered handler, and validates the handler's output back into a
   typed model. Unknown tool names raise `UnknownToolError`; malformed
   payloads raise `ToolValidationError`; tools without a wired handler
   raise `ToolNotImplementedError` — never a silent no-op, because that
   would violate §13 non-negotiable 3 ("every claim cited or refused").

Why a registry instead of a thin function call per tool? Because the
MCP server (§11) and the verdict ledger (§6.5) both want to enumerate
the surface and version it (§13 non-negotiable 14). `TOOL_SURFACE` is
the single source of truth; `TOOL_SURFACE_VERSION` is the semver pin
on the schema set as a whole.

Handler bodies for engines that have not yet shipped (the
optimal-transport ops in Phase 18, paraphrase voting in S-136,
calibration in S-137, the verdict ledger in S-143, the MCP server in
S-144) are intentionally NOT wired here. Each downstream slice will
call `dispatcher.register_handler(name, fn)` to plug its engine in.
Until then, calls to unwired tools raise `ToolNotImplementedError` —
the dispatcher refuses to fabricate verdicts.

SPEC-REF: §6.10 (tool-using orchestrator, forced tool calls only)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ctrldoc.models import UnitInterval
from ctrldoc.models_v1 import (
    Claim,
    CoverageReport,
    TypedEdgeTypeLiteral,
)

# ---------------------------------------------------------------------------
# Surface version (§13 non-negotiable 14)
# ---------------------------------------------------------------------------


TOOL_SURFACE_VERSION: str = "1.0.0"
"""Semver pin on the tool-surface schemas as a whole.

Bump MAJOR for breaking schema changes (renamed/removed fields, type
changes). Bump MINOR for additive changes (new tools, new optional
fields). Bump PATCH for docstring or description-only changes. The
MCP server (§11) ships this version verbatim so clients can detect
incompatible servers without parsing every schema.
"""


# ---------------------------------------------------------------------------
# Shared literal aliases
# ---------------------------------------------------------------------------


TraversalDirection = Literal["forward", "reverse", "both"]
"""Direction of an edge walk relative to the source node."""

EntailmentVerdict = Literal["entailment", "contradiction", "neutral"]
"""3-way NLI verdict surfaced by `entails()`."""

SubsumptionVerdict = Literal["subsumes", "subsumed_by", "equivalent", "incomparable"]
"""Galois subsumption verdict (§6.3) surfaced by `subsumes()`."""


# ---------------------------------------------------------------------------
# Strict-base mix-in — every tool I/O schema forbids extras and freezes.
# ---------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# 1. lookup_concept
# ---------------------------------------------------------------------------


class LookupConceptInput(_Strict):
    """Inputs for `lookup_concept(name)`."""

    name: str = Field(min_length=1)


class LookupConceptOutput(_Strict):
    """Output: `ConceptId | None`."""

    concept_id: str | None


# ---------------------------------------------------------------------------
# 2. get_claim
# ---------------------------------------------------------------------------


class GetClaimInput(_Strict):
    """Inputs for `get_claim(claim_id)`."""

    claim_id: str = Field(min_length=1)


class GetClaimOutput(_Strict):
    """Output: the full `Claim` row (§7)."""

    claim: Claim


# ---------------------------------------------------------------------------
# 3. traverse
# ---------------------------------------------------------------------------


class TraverseInput(_Strict):
    """Inputs for `traverse(node_id, edge_type, direction, hops)`."""

    node_id: str = Field(min_length=1)
    edge_type: TypedEdgeTypeLiteral
    direction: TraversalDirection
    hops: int = Field(ge=1, le=10)


class TraverseOutput(_Strict):
    """Output: ordered list of node ids reached within `hops` along `edge_type`."""

    node_ids: list[str]


# ---------------------------------------------------------------------------
# 4. entails
# ---------------------------------------------------------------------------


class EntailsInput(_Strict):
    """Inputs for `entails(claim_a, claim_b)`."""

    claim_a_id: str = Field(min_length=1)
    claim_b_id: str = Field(min_length=1)


class EntailsOutput(_Strict):
    """Output: `{verdict, confidence}` for `claim_a ⇒ claim_b`."""

    verdict: EntailmentVerdict
    confidence: UnitInterval


# ---------------------------------------------------------------------------
# 5. subsumes
# ---------------------------------------------------------------------------


class SubsumesInput(_Strict):
    """Inputs for `subsumes(claim_a, claim_b)`."""

    claim_a_id: str = Field(min_length=1)
    claim_b_id: str = Field(min_length=1)


class SubsumesOutput(_Strict):
    """Output: Galois `{verdict, confidence}` for `claim_a` vs `claim_b`."""

    verdict: SubsumptionVerdict
    confidence: UnitInterval


# ---------------------------------------------------------------------------
# 6. optimal_transport
# ---------------------------------------------------------------------------


class OptimalTransportInput(_Strict):
    """Inputs for `optimal_transport(distribution_a, distribution_b, cost_fn)`.

    The cost matrix is materialised by the caller — `cost_fn` is a tag
    that names which cost function was used (e.g. `"1-NLI_entail"`) so
    the verdict ledger can replay the same call deterministically.
    """

    source_weights: list[float]
    target_weights: list[float]
    cost_matrix: list[list[float]]
    cost_fn_tag: str = Field(min_length=1)


class OptimalTransportOutput(_Strict):
    """Output: a `TransportPlan` lifted to a Pydantic shape (§6.6)."""

    flow: list[list[float]]
    total_cost: float


# ---------------------------------------------------------------------------
# 7. coverage
# ---------------------------------------------------------------------------


class CoverageInput(_Strict):
    """Inputs for `coverage(workspace, target_doc_id, source_doc_id)`."""

    workspace_id: str = Field(min_length=1)
    target_doc_id: str = Field(min_length=1)
    source_doc_id: str = Field(min_length=1)


class CoverageOutput(_Strict):
    """Output: the persisted `CoverageReport` (§7)."""

    report: CoverageReport


# ---------------------------------------------------------------------------
# 8. compare
# ---------------------------------------------------------------------------


class CompareInput(_Strict):
    """Inputs for `compare(workspace, doc_ids)`."""

    workspace_id: str = Field(min_length=1)
    doc_ids: list[str] = Field(min_length=2)


class CompareReport(_Strict):
    """Per-doc strengths / weaknesses / gaps surfaced by `compare()` (§6.6).

    The schema mirrors the eval substrate in
    `ctrldoc.eval.compare`: each row carries a verdict from
    `{StrengthA, StrengthB, Gap}` plus the calibrated transport-derived
    confidence. The detailed body of `compare()` lands in Phase 18.
    """

    workspace_id: str
    doc_ids: list[str]
    rows: list[dict[str, Any]]


class CompareOutput(_Strict):
    """Output: a `CompareReport`."""

    report: CompareReport


# ---------------------------------------------------------------------------
# 9. merge
# ---------------------------------------------------------------------------


class MergeInput(_Strict):
    """Inputs for `merge(workspace, doc_ids)`."""

    workspace_id: str = Field(min_length=1)
    doc_ids: list[str] = Field(min_length=1)


class MergedDoc(_Strict):
    """A merged-doc envelope. Phase 18 fleshes out the cluster shape."""

    workspace_id: str
    cluster_ids: list[str]
    representative_claim_ids: list[str]


class MergeOutput(_Strict):
    """Output: a `MergedDoc`."""

    merged: MergedDoc


# ---------------------------------------------------------------------------
# 10. list_check
# ---------------------------------------------------------------------------


class ListCheckItem(_Strict):
    """One item to verify against the doc — text plus an opaque caller-supplied id."""

    item_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class ListCheckInput(_Strict):
    """Inputs for `list_check(items, doc_id)`."""

    items: list[ListCheckItem] = Field(min_length=1)
    doc_id: str = Field(min_length=1)


class ListCheckVerdict(_Strict):
    """Per-item verdict — the four-class partition shared with `coverage`."""

    item_id: str
    verdict: Literal["Covered", "Partial", "Missing", "Contradicted"]
    confidence: UnitInterval


class ListCheckOutput(_Strict):
    """Output: per-item verdicts in input order."""

    verdicts: list[ListCheckVerdict]


# ---------------------------------------------------------------------------
# 11. map
# ---------------------------------------------------------------------------


class MapInput(_Strict):
    """Inputs for `map(doc_id, filters)`.

    `filters` is a free-form dict so callers can restrict by concept
    primitive type, edge type, or section. The map renderer in
    `ctrldoc.cli_map` is the source of truth for which keys are honoured.
    """

    doc_id: str = Field(min_length=1)
    filters: dict[str, Any] = Field(default_factory=dict)


class MapOutput(_Strict):
    """Output: a Mermaid graph string and the rows that produced it."""

    mermaid: str
    node_ids: list[str]
    edge_count: int


# ---------------------------------------------------------------------------
# 12. qa
# ---------------------------------------------------------------------------


class QAInput(_Strict):
    """Inputs for `qa(doc_id_or_workspace, query)`.

    `target` is the doc id or workspace id; the receiving handler
    decides which by registry lookup so the tool surface stays uniform.
    """

    target: str = Field(min_length=1)
    query: str = Field(min_length=1)


class AnswerWithTrace(_Strict):
    """Q/A reply: a free-text answer plus its proof trace (§13 non-negotiable 4)."""

    answer: str
    citations: list[str]
    trace_steps: list[str]
    confidence: UnitInterval


class QAOutput(_Strict):
    """Output: an `AnswerWithTrace`."""

    reply: AnswerWithTrace


# ---------------------------------------------------------------------------
# 13. calibration
# ---------------------------------------------------------------------------


class CalibrationInput(_Strict):
    """`calibration()` takes no arguments."""


class CalibrationOutput(_Strict):
    """Output: `{ECE_per_backend, sample_sizes}` keyed by backend name."""

    ece_per_backend: dict[str, UnitInterval]
    sample_sizes: dict[str, int]


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    """One row in the tool surface — the contract for a single L4 tool."""

    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]


def _build_surface() -> dict[str, ToolSpec]:
    """Construct the immutable §6.10 tool surface in declaration order."""
    rows: list[ToolSpec] = [
        ToolSpec(
            name="lookup_concept",
            description="Look up a canonical concept id by surface name.",
            input_model=LookupConceptInput,
            output_model=LookupConceptOutput,
        ),
        ToolSpec(
            name="get_claim",
            description="Fetch a persisted Claim by id.",
            input_model=GetClaimInput,
            output_model=GetClaimOutput,
        ),
        ToolSpec(
            name="traverse",
            description="Walk the typed-edge graph from node_id under one edge type.",
            input_model=TraverseInput,
            output_model=TraverseOutput,
        ),
        ToolSpec(
            name="entails",
            description="Score the directed entailment claim_a -> claim_b.",
            input_model=EntailsInput,
            output_model=EntailsOutput,
        ),
        ToolSpec(
            name="subsumes",
            description="Score the Galois subsumption verdict between two claims (§6.3).",
            input_model=SubsumesInput,
            output_model=SubsumesOutput,
        ),
        ToolSpec(
            name="optimal_transport",
            description="Solve the transportation problem on a claim-pair cost matrix.",
            input_model=OptimalTransportInput,
            output_model=OptimalTransportOutput,
        ),
        ToolSpec(
            name="coverage",
            description="Per-claim coverage verdicts for target_doc vs source_doc in a workspace.",
            input_model=CoverageInput,
            output_model=CoverageOutput,
        ),
        ToolSpec(
            name="compare",
            description="Strengths / weaknesses / gaps across N docs in a workspace.",
            input_model=CompareInput,
            output_model=CompareOutput,
        ),
        ToolSpec(
            name="merge",
            description="Lossless synthesis of N docs into one merged-doc envelope.",
            input_model=MergeInput,
            output_model=MergeOutput,
        ),
        ToolSpec(
            name="list_check",
            description="Per-item verdicts of a list against one doc.",
            input_model=ListCheckInput,
            output_model=ListCheckOutput,
        ),
        ToolSpec(
            name="map",
            description="Render a doc's concept graph as Mermaid plus node/edge bookkeeping.",
            input_model=MapInput,
            output_model=MapOutput,
        ),
        ToolSpec(
            name="qa",
            description="Answer a query over a doc or workspace with an embedded proof trace.",
            input_model=QAInput,
            output_model=QAOutput,
        ),
        ToolSpec(
            name="calibration",
            description="Surface shipped ECE-per-backend and the sample size each was fit on.",
            input_model=CalibrationInput,
            output_model=CalibrationOutput,
        ),
    ]
    return {row.name: row for row in rows}


TOOL_SURFACE: dict[str, ToolSpec] = _build_surface()
"""The fixed §6.10 tool surface. Keys are stable tool names."""


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


class UnknownToolError(KeyError):
    """Raised when a tool name is not in `TOOL_SURFACE`."""


class ToolValidationError(ValueError):
    """Raised when an input or output payload fails Pydantic validation."""


class ToolNotImplementedError(NotImplementedError):
    """Raised when a tool has no handler wired — never silently no-ops."""


ToolHandler = Callable[[BaseModel], Any]
"""Handler signature: takes a validated input model, returns a model or dict."""


class ToolDispatcher:
    """Forced-tool-call dispatcher over the §6.10 surface.

    The dispatcher is intentionally dumb: it owns the registry, the
    validation rules, and the routing table — nothing else. Engines
    plug in via `register_handler`. Callers (the MCP server, the
    Python API, the test harness) invoke `dispatch`.

    Per-instance handler maps mean test isolation is free: every
    `default_dispatcher()` call returns a fresh routing table even
    though all instances share the same immutable schema registry.
    """

    def __init__(self, surface: dict[str, ToolSpec] | None = None) -> None:
        self._surface: dict[str, ToolSpec] = dict(surface or TOOL_SURFACE)
        self._handlers: dict[str, ToolHandler] = {}

    def tool_names(self) -> list[str]:
        """Return the registered tool names in declaration order."""
        return list(self._surface.keys())

    def spec(self, tool_name: str) -> ToolSpec:
        """Look up the schema spec for one tool. Raises `UnknownToolError`."""
        try:
            return self._surface[tool_name]
        except KeyError as exc:
            raise UnknownToolError(f"unknown tool: {tool_name!r}") from exc

    def register_handler(self, tool_name: str, handler: ToolHandler) -> None:
        """Plug an engine in. Re-registering replaces the previous handler."""
        # Surface-membership check first so wire-up typos fail loudly.
        self.spec(tool_name)
        self._handlers[tool_name] = handler

    def dispatch(self, *, tool_name: str, raw_input: dict[str, Any]) -> BaseModel:
        """Validate, route, re-validate. The single forced-call entry point."""
        spec = self.spec(tool_name)

        # 1. Input validation.
        try:
            validated_input = spec.input_model.model_validate(raw_input)
        except ValidationError as exc:
            raise ToolValidationError(f"input for {tool_name!r} failed validation: {exc}") from exc

        # 2. Handler lookup. Missing handler is an explicit refusal,
        #    not a silent fallback.
        handler = self._handlers.get(tool_name)
        if handler is None:
            raise ToolNotImplementedError(f"tool {tool_name!r} has no handler registered")

        # 3. Dispatch.
        raw_output = handler(validated_input)

        # 4. Output validation. Accept either a model instance of the
        #    expected type (pass-through) or a raw dict (validate it).
        if isinstance(raw_output, spec.output_model):
            return raw_output
        try:
            return spec.output_model.model_validate(raw_output)
        except ValidationError as exc:
            raise ToolValidationError(f"output for {tool_name!r} failed validation: {exc}") from exc


def default_dispatcher() -> ToolDispatcher:
    """Build a fresh dispatcher over the canonical surface with no handlers wired."""
    return ToolDispatcher()


__all__ = [
    "TOOL_SURFACE",
    "TOOL_SURFACE_VERSION",
    "AnswerWithTrace",
    "CalibrationInput",
    "CalibrationOutput",
    "CompareInput",
    "CompareOutput",
    "CompareReport",
    "CoverageInput",
    "CoverageOutput",
    "EntailmentVerdict",
    "EntailsInput",
    "EntailsOutput",
    "GetClaimInput",
    "GetClaimOutput",
    "ListCheckInput",
    "ListCheckItem",
    "ListCheckOutput",
    "ListCheckVerdict",
    "LookupConceptInput",
    "LookupConceptOutput",
    "MapInput",
    "MapOutput",
    "MergeInput",
    "MergeOutput",
    "MergedDoc",
    "OptimalTransportInput",
    "OptimalTransportOutput",
    "QAInput",
    "QAOutput",
    "SubsumesInput",
    "SubsumesOutput",
    "SubsumptionVerdict",
    "ToolDispatcher",
    "ToolHandler",
    "ToolNotImplementedError",
    "ToolSpec",
    "ToolValidationError",
    "TraversalDirection",
    "TraverseInput",
    "TraverseOutput",
    "UnknownToolError",
    "default_dispatcher",
]
