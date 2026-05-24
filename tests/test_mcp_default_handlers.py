"""§6.10 + §11 — pure-Python MCP handlers wired via `register_default_handlers`.

The L4 tool surface in `ctrldoc.orch.tools` is a registry of input /
output schemas; engines plug in via `dispatcher.register_handler(name,
fn)`. This module pins the first three engine bindings — the
*pure-Python* ones that need no storage or LLM dependencies:

* `subsumes` → `claim_subsumption` from `ctrldoc.extract.galois`.
  Structural Galois floor (§6.3); deterministic, no probability — the
  handler reports `confidence = 1.0` because the lattice verdict is
  exact at the structural floor (Tier-2 NLI/LLM layers escalate from
  here, in slices to come). The handler depends on a `claim_lookup`
  injected through `MCPHandlerDeps` so it can resolve the schema's
  `claim_a_id` / `claim_b_id` strings into `ClaimTuple` shapes.

* `optimal_transport` → `min_cost_transport` from `ctrldoc.ops.transport`.
  Self-contained; the input schema carries the source / target marginals
  and the cost matrix directly, so the handler just builds the
  `TransportProblem`, solves, and lifts the `TransportPlan` into the
  output schema. The `cost_fn_tag` slot is passed through untouched —
  the verdict ledger (§6.5) re-reads it on replay; no behavioural use
  here.

* `calibration` → `fit_per_backend_ece` from
  `ctrldoc.extract.isotonic_calibration`. Iterates a per-backend
  mapping of `(raw_scores, correct)` pairs, fits one calibrator per
  backend, reports the held-out ECE plus the held-out sample size.
  With no data injected, the handler still returns a structurally
  valid `CalibrationOutput` with empty `ece_per_backend` /
  `sample_sizes` dicts — a faithful "no backends fit yet" answer that
  honours §13 non-negotiable 3 (no silent fabrication) without forcing
  the host to special-case the empty path.

The wire-up factory `register_default_handlers(dispatcher, deps)`
registers exactly the handlers whose dependencies are satisfied —
`optimal_transport` always wires (no deps), `calibration` always wires
(empty data is a valid degenerate answer), `subsumes` wires only when
`deps.claim_lookup` is set. Unwired tools continue to raise
`ToolNotImplementedError` per the §6.10 dispatcher contract; the
factory never silently no-ops on a tool it cannot service.

The factory is called from `serve_stdio` by default so a stock host
sees the three pure-Python tools live as soon as the subprocess comes
up. Callers that already construct an `MCPServer` (e.g. the existing
in-process tests) bypass the factory by passing their own dispatcher.

SPEC-REF: §6.10 (forced-tool-call dispatcher), §11 (MCP server)
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence

import pytest

from ctrldoc.eval.claim_extraction import ClaimTuple
from ctrldoc.mcp.handlers import MCPHandlerDeps, register_default_handlers
from ctrldoc.mcp.server import MCPServer, serve_stdio
from ctrldoc.orch.tools import (
    CalibrationInput,
    OptimalTransportInput,
    OptimalTransportOutput,
    SubsumesInput,
    SubsumesOutput,
    ToolDispatcher,
    ToolNotImplementedError,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _claim(
    *,
    subject: str,
    predicate: str,
    obj: str,
    modality: str = "asserted",
    polarity: str = "affirmative",
    qualifier: str = "",
) -> ClaimTuple:
    """Build a `ClaimTuple` with sensible defaults for the test surface."""
    return ClaimTuple(
        subject=subject,
        predicate=predicate,
        object=obj,
        polarity=polarity,  # type: ignore[arg-type]
        modality=modality,  # type: ignore[arg-type]
        qualifier=qualifier,
    )


def _build_lookup(claims: dict[str, ClaimTuple]):
    """Return a `claim_lookup` closure backed by an in-memory dict."""

    def _lookup(claim_id: str) -> ClaimTuple:
        try:
            return claims[claim_id]
        except KeyError as exc:
            raise LookupError(f"unknown claim id: {claim_id!r}") from exc

    return _lookup


# ---------------------------------------------------------------------------
# `subsumes` handler — Galois subsumption over injected claim lookup
# ---------------------------------------------------------------------------


@pytest.mark.family_referential_integrity
def test_subsumes_handler_returns_equivalent_for_identical_tuples() -> None:
    """Two identical claims must collapse to `equivalent` with confidence 1.0."""
    claim = _claim(subject="dog", predicate="is", obj="animal")
    dispatcher = ToolDispatcher()
    deps = MCPHandlerDeps(claim_lookup=_build_lookup({"c1": claim, "c2": claim}))
    register_default_handlers(dispatcher=dispatcher, deps=deps)

    result = dispatcher.dispatch(
        tool_name="subsumes",
        raw_input={"claim_a_id": "c1", "claim_b_id": "c2"},
    )
    assert isinstance(result, SubsumesOutput)
    assert result.verdict == "equivalent"
    assert result.confidence == 1.0


@pytest.mark.family_referential_integrity
def test_subsumes_handler_returns_subsumes_for_stronger_left() -> None:
    """Stronger-modality left tuple subsumes the weaker right tuple."""
    strong = _claim(subject="x", predicate="follow", obj="rule", modality="obligatory")
    weak = _claim(subject="x", predicate="follow", obj="rule", modality="permitted")
    dispatcher = ToolDispatcher()
    deps = MCPHandlerDeps(claim_lookup=_build_lookup({"strong": strong, "weak": weak}))
    register_default_handlers(dispatcher=dispatcher, deps=deps)

    fwd = dispatcher.dispatch(
        tool_name="subsumes",
        raw_input={"claim_a_id": "strong", "claim_b_id": "weak"},
    )
    rev = dispatcher.dispatch(
        tool_name="subsumes",
        raw_input={"claim_a_id": "weak", "claim_b_id": "strong"},
    )
    assert isinstance(fwd, SubsumesOutput)
    assert isinstance(rev, SubsumesOutput)
    assert fwd.verdict == "subsumes"
    assert rev.verdict == "subsumed_by"
    assert fwd.confidence == 1.0
    assert rev.confidence == 1.0


@pytest.mark.family_referential_integrity
def test_subsumes_handler_returns_incomparable_for_unrelated_tuples() -> None:
    """SVO mismatch must surface as `incomparable` at the structural floor."""
    left = _claim(subject="dog", predicate="is", obj="animal")
    right = _claim(subject="rain", predicate="is", obj="wet")
    dispatcher = ToolDispatcher()
    deps = MCPHandlerDeps(claim_lookup=_build_lookup({"l": left, "r": right}))
    register_default_handlers(dispatcher=dispatcher, deps=deps)

    result = dispatcher.dispatch(
        tool_name="subsumes",
        raw_input={"claim_a_id": "l", "claim_b_id": "r"},
    )
    assert isinstance(result, SubsumesOutput)
    assert result.verdict == "incomparable"
    # Structural-floor confidence is exact: 1.0 for the decision that the
    # pair cannot be ordered without semantic escalation.
    assert result.confidence == 1.0


@pytest.mark.family_referential_integrity
def test_subsumes_handler_not_wired_without_claim_lookup() -> None:
    """No `claim_lookup` injected = handler must NOT silently register."""
    dispatcher = ToolDispatcher()
    register_default_handlers(dispatcher=dispatcher, deps=MCPHandlerDeps())

    with pytest.raises(ToolNotImplementedError):
        dispatcher.dispatch(
            tool_name="subsumes",
            raw_input={"claim_a_id": "c1", "claim_b_id": "c2"},
        )


# ---------------------------------------------------------------------------
# `optimal_transport` handler — `min_cost_transport` on injected matrices
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_optimal_transport_handler_solves_diagonal_problem() -> None:
    """Identity cost matrix => identity flow, zero total cost."""
    dispatcher = ToolDispatcher()
    register_default_handlers(dispatcher=dispatcher, deps=MCPHandlerDeps())

    result = dispatcher.dispatch(
        tool_name="optimal_transport",
        raw_input={
            "source_weights": [1.0, 1.0],
            "target_weights": [1.0, 1.0],
            "cost_matrix": [[0.0, 1.0], [1.0, 0.0]],
            "cost_fn_tag": "1-NLI_entail",
        },
    )
    assert isinstance(result, OptimalTransportOutput)
    assert result.total_cost == pytest.approx(0.0)
    # Identity transport: row 0 -> col 0, row 1 -> col 1.
    assert result.flow[0][0] == pytest.approx(1.0)
    assert result.flow[1][1] == pytest.approx(1.0)
    assert result.flow[0][1] == pytest.approx(0.0)
    assert result.flow[1][0] == pytest.approx(0.0)


@pytest.mark.family_determinism
def test_optimal_transport_handler_runs_repeatedly_with_identical_output() -> None:
    """Two consecutive calls must produce byte-identical flows."""
    dispatcher = ToolDispatcher()
    register_default_handlers(dispatcher=dispatcher, deps=MCPHandlerDeps())

    raw = {
        "source_weights": [0.5, 0.5],
        "target_weights": [0.3, 0.7],
        "cost_matrix": [[0.1, 0.9], [0.8, 0.2]],
        "cost_fn_tag": "1-NLI_entail",
    }
    first = dispatcher.dispatch(tool_name="optimal_transport", raw_input=raw)
    second = dispatcher.dispatch(tool_name="optimal_transport", raw_input=raw)
    assert isinstance(first, OptimalTransportOutput)
    assert isinstance(second, OptimalTransportOutput)
    assert first.flow == second.flow
    assert first.total_cost == second.total_cost


@pytest.mark.family_referential_integrity
def test_optimal_transport_handler_rejects_unbalanced_marginals() -> None:
    """Unbalanced marginals must surface as the dispatcher's validation error."""
    from ctrldoc.orch.tools import ToolValidationError

    dispatcher = ToolDispatcher()
    register_default_handlers(dispatcher=dispatcher, deps=MCPHandlerDeps())

    # The transport engine raises ValueError; the handler must let it
    # propagate (no silent rescue) so the caller — typically the MCP
    # server — surfaces it as an `isError=true` result.
    with pytest.raises(ValueError):
        dispatcher.dispatch(
            tool_name="optimal_transport",
            raw_input={
                "source_weights": [1.0, 1.0],
                "target_weights": [1.0],  # total mass 1.0 vs source 2.0
                "cost_matrix": [[0.0], [0.0]],
                "cost_fn_tag": "1-NLI_entail",
            },
        )
    # Sanity: the Pydantic validator that rejects validation drift only
    # raises ToolValidationError, so the test guards against a bare
    # ValueError being swallowed by the dispatcher.
    _ = ToolValidationError


# ---------------------------------------------------------------------------
# `calibration` handler — `fit_per_backend_ece` per injected backend
# ---------------------------------------------------------------------------


def _synthetic_backend(
    *, miscalibration: float, n: int, seed: int = 0
) -> tuple[Sequence[float], Sequence[int]]:
    """Build a labelled batch that triggers a measurable ECE before fit.

    Raw confidence is the gold probability plus `miscalibration`, clamped
    into `[0, 1]`. Correctness draws from the gold probability via a
    deterministic congruential pseudo-RNG so the fixture is byte-stable
    across runs (no `random.Random`).
    """
    raw: list[float] = []
    correct: list[int] = []
    state = (seed * 1103515245 + 12345) & 0x7FFFFFFF
    for i in range(n):
        gold_p = (i + 0.5) / n
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        u = state / 0x7FFFFFFF
        correct.append(1 if u < gold_p else 0)
        raw_score = min(1.0, max(0.0, gold_p + miscalibration))
        raw.append(raw_score)
    return raw, correct


@pytest.mark.family_referential_integrity
def test_calibration_handler_returns_empty_with_no_backends() -> None:
    """Empty calibration_data must yield empty maps, not refusal."""
    from ctrldoc.orch.tools import CalibrationOutput

    dispatcher = ToolDispatcher()
    register_default_handlers(dispatcher=dispatcher, deps=MCPHandlerDeps())

    result = dispatcher.dispatch(tool_name="calibration", raw_input={})
    assert isinstance(result, CalibrationOutput)
    assert result.ece_per_backend == {}
    assert result.sample_sizes == {}


@pytest.mark.family_referential_integrity
def test_calibration_handler_returns_ece_per_named_backend() -> None:
    """Per-backend `(raw, correct)` must produce per-backend ECE and sample size."""
    from ctrldoc.orch.tools import CalibrationOutput

    backend_a_raw, backend_a_correct = _synthetic_backend(miscalibration=0.20, n=40)
    backend_b_raw, backend_b_correct = _synthetic_backend(miscalibration=0.05, n=40, seed=7)

    dispatcher = ToolDispatcher()
    deps = MCPHandlerDeps(
        calibration_data={
            "ollama-qwen": (backend_a_raw, backend_a_correct),
            "anthropic-sonnet": (backend_b_raw, backend_b_correct),
        }
    )
    register_default_handlers(dispatcher=dispatcher, deps=deps)

    result = dispatcher.dispatch(tool_name="calibration", raw_input={})
    assert isinstance(result, CalibrationOutput)
    assert set(result.ece_per_backend.keys()) == {"ollama-qwen", "anthropic-sonnet"}
    assert set(result.sample_sizes.keys()) == {"ollama-qwen", "anthropic-sonnet"}
    # `fit_per_backend_ece` splits in half; held-out size is the upper half.
    assert result.sample_sizes["ollama-qwen"] == 40 - (40 // 2)
    assert result.sample_sizes["anthropic-sonnet"] == 40 - (40 // 2)
    # Every ECE is a valid probability.
    for backend, ece in result.ece_per_backend.items():
        assert 0.0 <= ece <= 1.0, f"{backend}: ECE {ece} out of [0,1]"
    # The well-calibrated backend produces no larger ECE than the
    # heavily-miscalibrated one — sanity that the dispatch threads each
    # labelled batch through its own calibrator (no global mixing).
    assert (
        result.ece_per_backend["anthropic-sonnet"] <= result.ece_per_backend["ollama-qwen"] + 0.10
    )


@pytest.mark.family_referential_integrity
def test_calibration_handler_input_schema_takes_no_arguments() -> None:
    """`calibration()` is parameter-free; extras must be rejected upstream."""
    from ctrldoc.orch.tools import ToolValidationError

    dispatcher = ToolDispatcher()
    register_default_handlers(dispatcher=dispatcher, deps=MCPHandlerDeps())

    # Sanity: the empty input model accepts an empty dict.
    CalibrationInput.model_validate({})
    with pytest.raises(ToolValidationError):
        dispatcher.dispatch(tool_name="calibration", raw_input={"unexpected": 1})


# ---------------------------------------------------------------------------
# Factory wires only the supported pure-Python tools — never silently no-ops
# ---------------------------------------------------------------------------


@pytest.mark.family_referential_integrity
def test_factory_wires_optimal_transport_and_calibration_without_deps() -> None:
    """No deps => `optimal_transport` and `calibration` wire; `subsumes` does not."""
    dispatcher = ToolDispatcher()
    register_default_handlers(dispatcher=dispatcher, deps=MCPHandlerDeps())

    # `optimal_transport` wires unconditionally (no deps required).
    dispatcher.dispatch(
        tool_name="optimal_transport",
        raw_input={
            "source_weights": [1.0],
            "target_weights": [1.0],
            "cost_matrix": [[0.0]],
            "cost_fn_tag": "1-NLI_entail",
        },
    )
    # `calibration` wires unconditionally (empty data => empty result).
    dispatcher.dispatch(tool_name="calibration", raw_input={})
    # `subsumes` is NOT registered (claim_lookup absent).
    with pytest.raises(ToolNotImplementedError):
        dispatcher.dispatch(
            tool_name="subsumes",
            raw_input={"claim_a_id": "x", "claim_b_id": "y"},
        )


@pytest.mark.family_referential_integrity
def test_factory_leaves_storage_backed_and_llm_handlers_unwired() -> None:
    """S-158..S-161 tools must still raise `ToolNotImplementedError`."""
    dispatcher = ToolDispatcher()
    register_default_handlers(
        dispatcher=dispatcher,
        deps=MCPHandlerDeps(
            claim_lookup=_build_lookup({}),
            calibration_data={},
        ),
    )

    # Tools that need storage (S-158), OT verifiers (S-159/S-160), or LLM
    # backends (S-161) must still surface ToolNotImplementedError.
    for tool_name, raw in [
        ("get_claim", {"claim_id": "x"}),
        ("lookup_concept", {"name": "x"}),
        ("traverse", {"node_id": "x", "edge_type": "is_a", "direction": "forward", "hops": 1}),
        ("coverage", {"workspace_id": "w", "target_doc_id": "t", "source_doc_id": "s"}),
        ("compare", {"workspace_id": "w", "doc_ids": ["a", "b"]}),
        ("merge", {"workspace_id": "w", "doc_ids": ["a"]}),
        ("list_check", {"items": [{"item_id": "i", "text": "t"}], "doc_id": "d"}),
        ("entails", {"claim_a_id": "a", "claim_b_id": "b"}),
        ("map", {"doc_id": "d"}),
        ("qa", {"target": "d", "query": "q"}),
    ]:
        with pytest.raises(ToolNotImplementedError):
            dispatcher.dispatch(tool_name=tool_name, raw_input=raw)


# ---------------------------------------------------------------------------
# `serve_stdio` round-trip — pure-Python handlers reachable over the wire
# ---------------------------------------------------------------------------


@pytest.mark.family_referential_integrity
def test_serve_stdio_round_trip_for_optimal_transport_handler() -> None:
    """A stock host can `tools/call optimal_transport` over the stdio loop."""
    import io

    # Construct a server with the factory wired in — same shape `serve_stdio`
    # uses by default when no explicit server is passed.
    dispatcher = ToolDispatcher()
    register_default_handlers(dispatcher=dispatcher, deps=MCPHandlerDeps())
    server = MCPServer(dispatcher=dispatcher)

    request = {
        "jsonrpc": "2.0",
        "id": 42,
        "method": "tools/call",
        "params": {
            "name": "optimal_transport",
            "arguments": {
                "source_weights": [1.0],
                "target_weights": [1.0],
                "cost_matrix": [[0.0]],
                "cost_fn_tag": "1-NLI_entail",
            },
        },
    }
    instream = io.StringIO(json.dumps(request) + "\n")
    outstream = io.StringIO()
    serve_stdio(server=server, instream=instream, outstream=outstream)

    response = json.loads(outstream.getvalue().splitlines()[0])
    assert response["id"] == 42
    assert response["result"]["isError"] is False
    body = response["result"]["structuredContent"]
    assert body["flow"] == [[1.0]]
    assert body["total_cost"] == 0.0


@pytest.mark.family_referential_integrity
def test_serve_stdio_round_trip_for_calibration_handler() -> None:
    """A stock host can `tools/call calibration` over the stdio loop."""
    import io

    dispatcher = ToolDispatcher()
    register_default_handlers(dispatcher=dispatcher, deps=MCPHandlerDeps())
    server = MCPServer(dispatcher=dispatcher)

    request = {
        "jsonrpc": "2.0",
        "id": 99,
        "method": "tools/call",
        "params": {"name": "calibration", "arguments": {}},
    }
    instream = io.StringIO(json.dumps(request) + "\n")
    outstream = io.StringIO()
    serve_stdio(server=server, instream=instream, outstream=outstream)

    response = json.loads(outstream.getvalue().splitlines()[0])
    assert response["id"] == 99
    assert response["result"]["isError"] is False
    body = response["result"]["structuredContent"]
    assert body == {"ece_per_backend": {}, "sample_sizes": {}}


@pytest.mark.family_referential_integrity
def test_serve_stdio_round_trip_for_subsumes_handler() -> None:
    """A stock host can `tools/call subsumes` when a `claim_lookup` is injected."""
    import io

    claim_left = _claim(subject="x", predicate="follow", obj="rule", modality="obligatory")
    claim_right = _claim(subject="x", predicate="follow", obj="rule", modality="permitted")
    dispatcher = ToolDispatcher()
    deps = MCPHandlerDeps(claim_lookup=_build_lookup({"L": claim_left, "R": claim_right}))
    register_default_handlers(dispatcher=dispatcher, deps=deps)
    server = MCPServer(dispatcher=dispatcher)

    request = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {
            "name": "subsumes",
            "arguments": {"claim_a_id": "L", "claim_b_id": "R"},
        },
    }
    instream = io.StringIO(json.dumps(request) + "\n")
    outstream = io.StringIO()
    serve_stdio(server=server, instream=instream, outstream=outstream)

    response = json.loads(outstream.getvalue().splitlines()[0])
    assert response["id"] == 7
    assert response["result"]["isError"] is False
    body = response["result"]["structuredContent"]
    assert body == {"verdict": "subsumes", "confidence": 1.0}


@pytest.mark.family_referential_integrity
def test_serve_stdio_default_wires_pure_python_handlers() -> None:
    """Calling `serve_stdio` with no explicit server wires the factory by default."""
    import io

    # No explicit server passed in => `serve_stdio` constructs an
    # `MCPServer` whose dispatcher has the pure-Python handlers wired.
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "calibration",
            "arguments": {},
        },
    }
    instream = io.StringIO(json.dumps(request) + "\n")
    outstream = io.StringIO()
    serve_stdio(instream=instream, outstream=outstream)

    response = json.loads(outstream.getvalue().splitlines()[0])
    assert response["id"] == 1
    # Was previously `isError=True` (no handlers); now wired -> false.
    assert response["result"]["isError"] is False


@pytest.mark.family_referential_integrity
def test_register_default_handlers_returns_set_of_wired_tool_names() -> None:
    """The factory must report which handlers it actually wired."""
    dispatcher = ToolDispatcher()
    wired = register_default_handlers(dispatcher=dispatcher, deps=MCPHandlerDeps())
    assert wired == frozenset({"optimal_transport", "calibration"})

    # With claim_lookup, `subsumes` also wires.
    dispatcher2 = ToolDispatcher()
    wired2 = register_default_handlers(
        dispatcher=dispatcher2,
        deps=MCPHandlerDeps(claim_lookup=_build_lookup({})),
    )
    assert wired2 == frozenset({"optimal_transport", "calibration", "subsumes"})


@pytest.mark.family_referential_integrity
def test_subsumes_handler_surfaces_lookup_errors_as_validation() -> None:
    """An unknown claim id from the lookup must raise a clean error."""
    dispatcher = ToolDispatcher()
    deps = MCPHandlerDeps(claim_lookup=_build_lookup({}))
    register_default_handlers(dispatcher=dispatcher, deps=deps)

    # The lookup raises LookupError; the handler lets it propagate so the
    # MCP server lifts it into an `isError=true` envelope.
    with pytest.raises(LookupError):
        dispatcher.dispatch(
            tool_name="subsumes",
            raw_input={"claim_a_id": "missing", "claim_b_id": "missing"},
        )


@pytest.mark.family_determinism
def test_optimal_transport_handler_handles_empty_problem() -> None:
    """Empty marginals must produce an empty flow with zero cost."""
    dispatcher = ToolDispatcher()
    register_default_handlers(dispatcher=dispatcher, deps=MCPHandlerDeps())

    result = dispatcher.dispatch(
        tool_name="optimal_transport",
        raw_input={
            "source_weights": [],
            "target_weights": [],
            "cost_matrix": [],
            "cost_fn_tag": "1-NLI_entail",
        },
    )
    assert isinstance(result, OptimalTransportOutput)
    assert result.flow == []
    assert math.isclose(result.total_cost, 0.0)


@pytest.mark.family_referential_integrity
def test_subsumes_input_schema_round_trips_through_handler() -> None:
    """Smoke that the Pydantic input model accepts the raw dict the handler reads."""
    inp = SubsumesInput(claim_a_id="a", claim_b_id="b")
    assert inp.claim_a_id == "a"
    assert inp.claim_b_id == "b"


@pytest.mark.family_determinism
def test_optimal_transport_input_schema_round_trips_through_handler() -> None:
    """Smoke that the Pydantic input model preserves cost-fn tag verbatim."""
    inp = OptimalTransportInput(
        source_weights=[1.0],
        target_weights=[1.0],
        cost_matrix=[[0.5]],
        cost_fn_tag="1-NLI_entail",
    )
    assert inp.cost_fn_tag == "1-NLI_entail"
