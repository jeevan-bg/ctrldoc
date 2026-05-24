"""Pure-Python MCP handler factory for the §6.10 tool surface.

The L4 tool surface in `ctrldoc.orch.tools` is a registry of input /
output schemas; engines plug in via
`dispatcher.register_handler(name, fn)`. The MCP server in
`ctrldoc.mcp.server` reuses that dispatcher verbatim — handlers wired
into the dispatcher are reachable over the JSON-RPC 2.0 stdio
transport described in §11.

This module ships the **pure-Python** wave of handlers — the three
tools whose engines need no storage layer, no LLM call, no network:

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

The factory returns the `frozenset` of wired tool names so callers
can log the surface they actually exposed.

The downstream waves of MCP handlers ship in S-158 (storage-backed:
``get_claim`` / ``lookup_concept`` / ``traverse``), S-159 / S-160
(OT-backed: ``coverage`` / ``list_check`` / ``compare`` / ``merge``),
and S-161 (LLM-backed: ``entails`` / ``qa`` / ``map``). Each will
plug in via the same `register_handler` seam - this module does not
own those.

SPEC-REF: §6.10 (tool-using orchestrator), §11 (MCP server)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel

from ctrldoc.eval.claim_extraction import ClaimTuple
from ctrldoc.extract.galois import claim_subsumption
from ctrldoc.extract.isotonic_calibration import fit_per_backend_ece
from ctrldoc.ops.transport import TransportProblem, min_cost_transport
from ctrldoc.orch.tools import (
    CalibrationInput,
    CalibrationOutput,
    OptimalTransportInput,
    OptimalTransportOutput,
    SubsumesInput,
    SubsumesOutput,
    ToolDispatcher,
    ToolHandler,
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


@dataclass(frozen=True)
class MCPHandlerDeps:
    """Injected dependencies for the pure-Python handler factory.

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

    Re-registering replaces the previous handler (the dispatcher
    documents this), so callers can layer richer wave-S-158+ handlers
    on top of this factory's pure-Python floor.
    """
    wired: set[str] = set()

    dispatcher.register_handler("optimal_transport", _optimal_transport_handler)
    wired.add("optimal_transport")

    dispatcher.register_handler("calibration", _make_calibration_handler(deps.calibration_data))
    wired.add("calibration")

    if deps.claim_lookup is not None:
        dispatcher.register_handler("subsumes", _make_subsumes_handler(deps.claim_lookup))
        wired.add("subsumes")

    return frozenset(wired)


__all__ = [
    "CalibrationData",
    "ClaimLookup",
    "MCPHandlerDeps",
    "register_default_handlers",
]
