"""§6.10 + §6.6 — OT-backed MCP handlers (`coverage`, `list_check`).

The pure-Python wave (S-157) shipped `subsumes` / `optimal_transport`
/ `calibration`; the storage-backed wave (S-158) added `get_claim`,
`lookup_concept`, and `traverse`. This module pins the next wave —
the optimal-transport-backed `coverage` and `list_check` handlers
defined in §6.10 and reduced to the §6.6 transport core.

* ``coverage`` resolves `target_doc_id` and `source_doc_id` to the
  persisted `Claim` lists each doc carries (via an injected
  ``claims_for_doc_supplier``), converts each `Claim` back to the §6.2
  universal tuple, and runs `ctrldoc.ops.coverage.coverage` over them
  with the injected `NLIScorer`. The handler lifts the per-target
  `Covered` / `Missing` verdicts into a full §7 `CoverageReport` —
  pinned to the workspace id, target id, source id, with one
  `CoverageVerdict` per target claim and the aggregate `CoverageSummary`
  rates the dashboard renders.

* ``list_check`` parses its `items` payload (one `(item_id, text)` row
  per list entry) as a tiny target doc — each item becomes a `ClaimTuple`
  whose `subject` slot carries the item text — then runs the same §6.6
  transport reduction with the persisted doc claims as source. Output
  is one `ListCheckVerdict` per item in input order, verdict from the
  shared `{Covered, Partial, Missing, Contradicted}` four-class
  partition and confidence pinned by the transport reduction (S-159
  surfaces `Covered`/`Missing` only; richer partials land with the
  calibrated edge layer in later slices).

The release gate is the reflexive identity: a doc's own claims must
cover the doc 100 % when scored by an entailment-perfect oracle. The
fixture exercises this end-to-end through the dispatcher and through
the stdio round-trip.

SPEC-REF: §6.6 (optimal-transport core), §6.10 (tool-using orchestrator),
§11 (MCP server)
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable

import pytest

from ctrldoc.eval.calibration import NLIScore
from ctrldoc.mcp.handlers import MCPHandlerDeps, register_default_handlers
from ctrldoc.mcp.server import MCPServer, serve_stdio
from ctrldoc.models import Span
from ctrldoc.models_v1 import Claim
from ctrldoc.orch.tools import (
    CoverageOutput,
    ListCheckOutput,
    ToolDispatcher,
    ToolNotImplementedError,
    ToolValidationError,
)

# ---------------------------------------------------------------------------
# Fixtures — Claim builders + perfect-oracle NLI scorer
# ---------------------------------------------------------------------------


def _claim(
    cid: str,
    *,
    doc_id: str,
    subject: str,
    predicate: str,
    obj: str,
    polarity: str = "+",
    modality: str | None = "assert",
) -> Claim:
    """Build a persisted `Claim` with a single chunk-anchored span."""
    span = Span(chunk_id=f"{doc_id}-chunk-1", char_start=0, char_end=1, text="x")
    return Claim(
        id=cid,
        doc_id=doc_id,
        text=f"{subject} {predicate} {obj}",
        subject=subject,
        predicate=predicate,
        object=obj,
        polarity=polarity,  # type: ignore[arg-type]
        modality=modality,  # type: ignore[arg-type]
        qualifier={},
        span_refs=[span],
        section_id="sec-1",
        concept_ids=[],
        typed_slots={},
        confidence=0.9,
    )


class _IdentityEntailScorer:
    """Perfect-oracle NLI scorer: identical premise/hypothesis ⇒ 100 % entailment.

    Non-identical pairs collapse to a low-entail score that the §6.6
    slack column beats — the transport engine then routes the target's
    mass to slack, surfacing `Missing`. This isolates the OT reduction's
    correctness from any real backend's quality (the same gold-aligned
    oracle pattern `test_ops_coverage.py` uses for the reflexive-identity
    family).
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def score(self, *, premise: str, hypothesis: str) -> NLIScore:
        self.calls.append((premise, hypothesis))
        if premise == hypothesis:
            return NLIScore(entailment=0.99, contradiction=0.005, neutral=0.005)
        return NLIScore(entailment=0.10, contradiction=0.10, neutral=0.80)


def _doc_a_claims() -> list[Claim]:
    return [
        _claim(
            "c-a1",
            doc_id="doc-A",
            subject="the proxy",
            predicate="caches",
            obj="responses",
        ),
        _claim(
            "c-a2",
            doc_id="doc-A",
            subject="the proxy",
            predicate="drops",
            obj="idle connections",
        ),
    ]


def _doc_b_claims() -> list[Claim]:
    return [
        _claim(
            "c-b1",
            doc_id="doc-B",
            subject="the gateway",
            predicate="rotates",
            obj="api keys",
        ),
    ]


def _make_supplier(
    doc_to_claims: dict[str, list[Claim]],
) -> Callable[[str], list[Claim]]:
    """Build a `claims_for_doc_supplier` closure backed by an in-memory dict.

    An unknown doc id resolves to an empty list — that surfaces to the
    handler as "no claims for this doc", which the §6.6 reduction
    treats as the all-Missing degenerate case (zero scorer calls). A
    missing-doc raise would force every caller to special-case before
    dispatching; the empty-list answer is the faithful one.
    """

    def _supplier(doc_id: str) -> list[Claim]:
        return list(doc_to_claims.get(doc_id, []))

    return _supplier


# ---------------------------------------------------------------------------
# `coverage` handler — reflexive identity = 100 % covered (the release gate)
# ---------------------------------------------------------------------------


@pytest.mark.family_verifier_calibration
def test_coverage_handler_reflexive_identity_is_fully_covered() -> None:
    """A doc's own claims must cover the doc 100 % — the slice's release gate."""
    claims = _doc_a_claims()
    scorer = _IdentityEntailScorer()
    deps = MCPHandlerDeps(
        claims_for_doc_supplier=_make_supplier({"doc-A": claims}),
        nli_scorer=scorer,
    )
    dispatcher = ToolDispatcher()
    register_default_handlers(dispatcher=dispatcher, deps=deps)

    result = dispatcher.dispatch(
        tool_name="coverage",
        raw_input={
            "workspace_id": "ws-1",
            "target_doc_id": "doc-A",
            "source_doc_id": "doc-A",
        },
    )
    assert isinstance(result, CoverageOutput)
    report = result.report
    assert report.workspace_id == "ws-1"
    assert report.target_doc_id == "doc-A"
    assert report.source_doc_id == "doc-A"
    # Every target claim has exactly one row in the per-claim verdict list.
    assert [v.target_claim_id for v in report.per_claim] == [c.id for c in claims]
    # Reflexive identity gate: every verdict is `Covered`.
    for verdict in report.per_claim:
        assert verdict.verdict == "Covered", verdict
        # Every verdict cites at least one source claim from the same doc.
        assert verdict.aligned_source_claims, verdict
        # Calibrated confidence is the post-transport probability mass.
        assert 0.0 <= verdict.calibrated_confidence <= 1.0
    # Summary aggregates: 100 % covered, 0 % otherwise.
    assert report.summary.covered_rate == 1.0
    assert report.summary.missing_rate == 0.0
    assert report.summary.partial_rate == 0.0
    assert report.summary.contradicted_rate == 0.0


@pytest.mark.family_verifier_calibration
def test_coverage_handler_disjoint_docs_surface_all_missing() -> None:
    """Disjoint source/target docs ⇒ every target claim is `Missing`."""
    deps = MCPHandlerDeps(
        claims_for_doc_supplier=_make_supplier(
            {"doc-A": _doc_a_claims(), "doc-B": _doc_b_claims()}
        ),
        nli_scorer=_IdentityEntailScorer(),
    )
    dispatcher = ToolDispatcher()
    register_default_handlers(dispatcher=dispatcher, deps=deps)

    result = dispatcher.dispatch(
        tool_name="coverage",
        raw_input={
            "workspace_id": "ws-1",
            "target_doc_id": "doc-A",
            "source_doc_id": "doc-B",
        },
    )
    assert isinstance(result, CoverageOutput)
    for verdict in result.report.per_claim:
        assert verdict.verdict == "Missing", verdict
        # Missing verdicts list zero aligned source claims.
        assert verdict.aligned_source_claims == [], verdict
    assert result.report.summary.missing_rate == 1.0
    assert result.report.summary.covered_rate == 0.0


@pytest.mark.family_verifier_calibration
def test_coverage_handler_handles_empty_target_doc() -> None:
    """A target doc with no claims surfaces an empty per-claim list."""
    deps = MCPHandlerDeps(
        claims_for_doc_supplier=_make_supplier({"doc-A": _doc_a_claims(), "doc-empty": []}),
        nli_scorer=_IdentityEntailScorer(),
    )
    dispatcher = ToolDispatcher()
    register_default_handlers(dispatcher=dispatcher, deps=deps)

    result = dispatcher.dispatch(
        tool_name="coverage",
        raw_input={
            "workspace_id": "ws-1",
            "target_doc_id": "doc-empty",
            "source_doc_id": "doc-A",
        },
    )
    assert isinstance(result, CoverageOutput)
    assert result.report.per_claim == []
    # Empty target: summary rates default to all-Covered (zero of zero is
    # vacuously 1.0; the four rates must still sum to 1.0).
    rates = result.report.summary
    total = rates.covered_rate + rates.partial_rate + rates.missing_rate + rates.contradicted_rate
    assert abs(total - 1.0) < 1e-6


@pytest.mark.family_verifier_calibration
def test_coverage_handler_handles_empty_source_doc() -> None:
    """An empty source doc routes every target to slack ⇒ all `Missing`."""
    deps = MCPHandlerDeps(
        claims_for_doc_supplier=_make_supplier({"doc-A": _doc_a_claims(), "doc-empty": []}),
        nli_scorer=_IdentityEntailScorer(),
    )
    dispatcher = ToolDispatcher()
    register_default_handlers(dispatcher=dispatcher, deps=deps)

    result = dispatcher.dispatch(
        tool_name="coverage",
        raw_input={
            "workspace_id": "ws-1",
            "target_doc_id": "doc-A",
            "source_doc_id": "doc-empty",
        },
    )
    assert isinstance(result, CoverageOutput)
    for verdict in result.report.per_claim:
        assert verdict.verdict == "Missing"
        assert verdict.aligned_source_claims == []


# ---------------------------------------------------------------------------
# `coverage` handler — wiring policy
# ---------------------------------------------------------------------------


@pytest.mark.family_referential_integrity
def test_coverage_handler_not_wired_without_supplier() -> None:
    """No `claims_for_doc_supplier` injected ⇒ tool must NOT silently register."""
    dispatcher = ToolDispatcher()
    # Provide the scorer but not the supplier — the factory must still leave
    # coverage unwired so the dispatcher refuses with `ToolNotImplementedError`.
    register_default_handlers(
        dispatcher=dispatcher,
        deps=MCPHandlerDeps(nli_scorer=_IdentityEntailScorer()),
    )
    with pytest.raises(ToolNotImplementedError):
        dispatcher.dispatch(
            tool_name="coverage",
            raw_input={
                "workspace_id": "ws-1",
                "target_doc_id": "doc-A",
                "source_doc_id": "doc-B",
            },
        )


@pytest.mark.family_referential_integrity
def test_coverage_handler_not_wired_without_scorer() -> None:
    """No `nli_scorer` injected ⇒ tool must NOT silently register."""
    dispatcher = ToolDispatcher()
    register_default_handlers(
        dispatcher=dispatcher,
        deps=MCPHandlerDeps(
            claims_for_doc_supplier=_make_supplier({"doc-A": _doc_a_claims()}),
        ),
    )
    with pytest.raises(ToolNotImplementedError):
        dispatcher.dispatch(
            tool_name="coverage",
            raw_input={
                "workspace_id": "ws-1",
                "target_doc_id": "doc-A",
                "source_doc_id": "doc-A",
            },
        )


@pytest.mark.family_referential_integrity
def test_coverage_handler_rejects_malformed_input() -> None:
    """Missing required fields ⇒ `ToolValidationError` before any scorer call."""
    scorer = _IdentityEntailScorer()
    deps = MCPHandlerDeps(
        claims_for_doc_supplier=_make_supplier({"doc-A": _doc_a_claims()}),
        nli_scorer=scorer,
    )
    dispatcher = ToolDispatcher()
    register_default_handlers(dispatcher=dispatcher, deps=deps)
    with pytest.raises(ToolValidationError):
        dispatcher.dispatch(
            tool_name="coverage",
            raw_input={"workspace_id": "ws-1", "target_doc_id": "doc-A"},  # missing source
        )
    # No scorer call may have escaped validation.
    assert scorer.calls == []


# ---------------------------------------------------------------------------
# `list_check` handler — items as a tiny target doc
# ---------------------------------------------------------------------------


@pytest.mark.family_verifier_calibration
def test_list_check_handler_reflexive_item_text_is_covered() -> None:
    """Items whose text matches a doc claim's rendered surface clear coverage."""
    # The §6.6 reduction renders a `ClaimTuple` as
    # `f"{subject} {predicate} {object} {qualifier}".strip()`; for
    # `list_check` the item.text becomes the tuple's `subject` slot
    # (predicate/object empty), so the rendered surface is exactly the
    # text. We pre-populate the doc with claims whose rendered surface
    # equals the item text so the perfect-oracle scorer fires Covered.
    matching_doc_claim = _claim(
        "c-d1",
        doc_id="doc-D",
        subject="ssh tunnels rotate every 30 minutes",
        predicate="",
        obj="",
    )
    other_doc_claim = _claim(
        "c-d2",
        doc_id="doc-D",
        subject="metrics ship over otlp",
        predicate="",
        obj="",
    )
    deps = MCPHandlerDeps(
        claims_for_doc_supplier=_make_supplier({"doc-D": [matching_doc_claim, other_doc_claim]}),
        nli_scorer=_IdentityEntailScorer(),
    )
    dispatcher = ToolDispatcher()
    register_default_handlers(dispatcher=dispatcher, deps=deps)

    result = dispatcher.dispatch(
        tool_name="list_check",
        raw_input={
            "items": [
                {"item_id": "i-1", "text": "ssh tunnels rotate every 30 minutes"},
                {"item_id": "i-2", "text": "this requirement is nowhere in the doc"},
            ],
            "doc_id": "doc-D",
        },
    )
    assert isinstance(result, ListCheckOutput)
    assert [v.item_id for v in result.verdicts] == ["i-1", "i-2"]
    assert result.verdicts[0].verdict == "Covered"
    assert result.verdicts[1].verdict == "Missing"
    # Confidences sit in the unit interval.
    for v in result.verdicts:
        assert 0.0 <= v.confidence <= 1.0


@pytest.mark.family_referential_integrity
def test_list_check_handler_not_wired_without_supplier_or_scorer() -> None:
    """list_check needs both deps — neither alone should wire."""
    dispatcher_a = ToolDispatcher()
    register_default_handlers(
        dispatcher=dispatcher_a,
        deps=MCPHandlerDeps(nli_scorer=_IdentityEntailScorer()),
    )
    with pytest.raises(ToolNotImplementedError):
        dispatcher_a.dispatch(
            tool_name="list_check",
            raw_input={"items": [{"item_id": "i", "text": "t"}], "doc_id": "d"},
        )

    dispatcher_b = ToolDispatcher()
    register_default_handlers(
        dispatcher=dispatcher_b,
        deps=MCPHandlerDeps(
            claims_for_doc_supplier=_make_supplier({"doc-D": _doc_a_claims()}),
        ),
    )
    with pytest.raises(ToolNotImplementedError):
        dispatcher_b.dispatch(
            tool_name="list_check",
            raw_input={"items": [{"item_id": "i", "text": "t"}], "doc_id": "d"},
        )


# ---------------------------------------------------------------------------
# Factory return — names + wave-isolation guard
# ---------------------------------------------------------------------------


@pytest.mark.family_referential_integrity
def test_register_default_handlers_wires_ot_handlers_when_deps_set() -> None:
    """coverage + list_check must wire when both supplier and scorer are set."""
    dispatcher = ToolDispatcher()
    wired = register_default_handlers(
        dispatcher=dispatcher,
        deps=MCPHandlerDeps(
            claims_for_doc_supplier=_make_supplier({"doc-A": _doc_a_claims()}),
            nli_scorer=_IdentityEntailScorer(),
        ),
    )
    assert "coverage" in wired
    assert "list_check" in wired


@pytest.mark.family_referential_integrity
def test_register_default_handlers_leaves_remaining_waves_unwired() -> None:
    """`compare`, `merge`, and the LLM-backed tools must stay unwired here."""
    dispatcher = ToolDispatcher()
    register_default_handlers(
        dispatcher=dispatcher,
        deps=MCPHandlerDeps(
            claims_for_doc_supplier=_make_supplier({"doc-A": _doc_a_claims()}),
            nli_scorer=_IdentityEntailScorer(),
        ),
    )
    for tool_name, raw in [
        ("compare", {"workspace_id": "w", "doc_ids": ["a", "b"]}),
        ("merge", {"workspace_id": "w", "doc_ids": ["a"]}),
        ("entails", {"claim_a_id": "a", "claim_b_id": "b"}),
        ("map", {"doc_id": "d"}),
        ("qa", {"target": "d", "query": "q"}),
    ]:
        with pytest.raises(ToolNotImplementedError):
            dispatcher.dispatch(tool_name=tool_name, raw_input=raw)


# ---------------------------------------------------------------------------
# Stdio round-trip — `coverage` reachable over the wire
# ---------------------------------------------------------------------------


@pytest.mark.family_referential_integrity
def test_serve_stdio_round_trip_for_coverage_handler() -> None:
    """A stock host can `tools/call coverage` over the stdio loop."""
    claims = _doc_a_claims()
    dispatcher = ToolDispatcher()
    register_default_handlers(
        dispatcher=dispatcher,
        deps=MCPHandlerDeps(
            claims_for_doc_supplier=_make_supplier({"doc-A": claims}),
            nli_scorer=_IdentityEntailScorer(),
        ),
    )
    server = MCPServer(dispatcher=dispatcher)

    request = {
        "jsonrpc": "2.0",
        "id": 11,
        "method": "tools/call",
        "params": {
            "name": "coverage",
            "arguments": {
                "workspace_id": "ws-1",
                "target_doc_id": "doc-A",
                "source_doc_id": "doc-A",
            },
        },
    }
    instream = io.StringIO(json.dumps(request) + "\n")
    outstream = io.StringIO()
    serve_stdio(server=server, instream=instream, outstream=outstream)

    response = json.loads(outstream.getvalue().splitlines()[0])
    assert response["id"] == 11
    assert response["result"]["isError"] is False
    body = response["result"]["structuredContent"]
    assert body["report"]["target_doc_id"] == "doc-A"
    assert body["report"]["source_doc_id"] == "doc-A"
    # Reflexive identity ⇒ summary.covered_rate is 1.0.
    assert body["report"]["summary"]["covered_rate"] == 1.0


@pytest.mark.family_referential_integrity
def test_serve_stdio_round_trip_for_list_check_handler() -> None:
    """A stock host can `tools/call list_check` over the stdio loop."""
    doc_claim = _claim(
        "c-1",
        doc_id="doc-D",
        subject="ssh tunnels rotate every 30 minutes",
        predicate="",
        obj="",
    )
    dispatcher = ToolDispatcher()
    register_default_handlers(
        dispatcher=dispatcher,
        deps=MCPHandlerDeps(
            claims_for_doc_supplier=_make_supplier({"doc-D": [doc_claim]}),
            nli_scorer=_IdentityEntailScorer(),
        ),
    )
    server = MCPServer(dispatcher=dispatcher)

    request = {
        "jsonrpc": "2.0",
        "id": 12,
        "method": "tools/call",
        "params": {
            "name": "list_check",
            "arguments": {
                "items": [
                    {"item_id": "i-1", "text": "ssh tunnels rotate every 30 minutes"},
                    {"item_id": "i-2", "text": "this is nowhere in the doc"},
                ],
                "doc_id": "doc-D",
            },
        },
    }
    instream = io.StringIO(json.dumps(request) + "\n")
    outstream = io.StringIO()
    serve_stdio(server=server, instream=instream, outstream=outstream)

    response = json.loads(outstream.getvalue().splitlines()[0])
    assert response["id"] == 12
    assert response["result"]["isError"] is False
    body = response["result"]["structuredContent"]
    verdicts = body["verdicts"]
    assert [v["item_id"] for v in verdicts] == ["i-1", "i-2"]
    assert verdicts[0]["verdict"] == "Covered"
    assert verdicts[1]["verdict"] == "Missing"


# ---------------------------------------------------------------------------
# Determinism — repeat calls must produce identical reports
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_coverage_handler_is_repeatedly_deterministic() -> None:
    """Two identical calls must produce byte-identical CoverageReport rows."""
    deps = MCPHandlerDeps(
        claims_for_doc_supplier=_make_supplier({"doc-A": _doc_a_claims()}),
        nli_scorer=_IdentityEntailScorer(),
    )
    dispatcher = ToolDispatcher()
    register_default_handlers(dispatcher=dispatcher, deps=deps)

    raw = {
        "workspace_id": "ws-1",
        "target_doc_id": "doc-A",
        "source_doc_id": "doc-A",
    }
    first = dispatcher.dispatch(tool_name="coverage", raw_input=raw)
    second = dispatcher.dispatch(tool_name="coverage", raw_input=raw)
    assert isinstance(first, CoverageOutput)
    assert isinstance(second, CoverageOutput)
    assert first.model_dump_json() == second.model_dump_json()
