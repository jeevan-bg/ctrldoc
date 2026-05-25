"""§6.10 + §11 — LLM-backed MCP handlers for `entails`, `qa`, and `map`.

The S-160 wave wired the OT-backed `compare` / `merge` tools onto
`ctrldoc.mcp.handlers`. This module pins the final wave — the three
LLM- or LLM-tier-adjacent handlers that close the §6.10 surface:

* ``entails`` → wraps an injected `NLIScorer` (typically the Tier-2
  DeBERTa backend, optionally calibrated via
  `CalibratedNLIScorer`). The handler resolves the input's
  `claim_a_id` / `claim_b_id` strings into `ClaimTuple` rows through
  the existing `claim_lookup` dep, renders both sides with
  `render_claim_text`, calls `scorer.score(premise=A, hypothesis=B)`,
  and returns `{verdict: argmax_label, confidence: top_confidence}`.

* ``qa`` → wraps an injected `qa_runner` closure. The factory keeps
  the QA playbook's transitive deps (retriever, decomposer, task
  runner, evidence rendering) opaque so the handlers module does not
  have to import the full LLM stack; the slice that owns the runtime
  wiring builds the closure and passes it through `MCPHandlerDeps`.
  Gated on the LLM profile being available — the closure is the
  contract, not the playbook itself.

* ``map`` → typed-edge graph rendering. The handler resolves the
  input's `doc_id` to the doc's persisted `TypedEdge` rows via a
  `typed_edges_for_doc_supplier`, renders them as a deterministic
  Mermaid `graph LR` block (one node per distinct endpoint, one
  labelled arrow per edge), and surfaces the node ids + edge count
  alongside the rendered string. Empty doc surfaces a valid empty
  Mermaid block — a faithful "no edges yet" answer rather than
  refusal.

Wiring policy honours §13 non-negotiable 3: every LLM handler stays
unregistered unless its dep is set. `entails` needs both
`claim_lookup` and `nli_scorer`; `qa` needs `qa_runner`; `map` needs
`typed_edges_for_doc_supplier`. Each missing dep leaves its tool
surfaced as `ToolNotImplementedError` so the dispatcher refuses
rather than fabricating verdicts.

SPEC-REF: §6.10 (tool-using orchestrator — `entails` / `qa` / `map`)
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Sequence

import pytest

from ctrldoc.eval.calibration import NLIScore
from ctrldoc.eval.claim_extraction import ClaimTuple
from ctrldoc.mcp.handlers import MCPHandlerDeps, register_default_handlers
from ctrldoc.mcp.server import MCPServer, serve_stdio
from ctrldoc.models import Span
from ctrldoc.models_v1 import TypedEdge
from ctrldoc.orch.tools import (
    AnswerWithTrace,
    EntailsOutput,
    MapOutput,
    QAOutput,
    ToolDispatcher,
    ToolNotImplementedError,
    ToolValidationError,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _claim(
    *,
    subject: str,
    predicate: str,
    obj: str,
    polarity: str = "affirmative",
    modality: str = "asserted",
    qualifier: str = "",
) -> ClaimTuple:
    """Build a `ClaimTuple` with §6.2 universal-tuple defaults."""
    return ClaimTuple(
        subject=subject,
        predicate=predicate,
        object=obj,
        polarity=polarity,  # type: ignore[arg-type]
        modality=modality,  # type: ignore[arg-type]
        qualifier=qualifier,
    )


def _build_claim_lookup(
    claims: dict[str, ClaimTuple],
) -> Callable[[str], ClaimTuple]:
    """Return a `claim_lookup` closure backed by an in-memory dict."""

    def _lookup(claim_id: str) -> ClaimTuple:
        try:
            return claims[claim_id]
        except KeyError as exc:
            raise LookupError(f"unknown claim id: {claim_id!r}") from exc

    return _lookup


class _StubNLIScorer:
    """In-memory scorer: returns the canned `NLIScore` for any pair.

    A perfectly deterministic test seam — used to pin the handler's
    contract without dragging the DeBERTa backend (or any other real
    NLI model) into the fast test gate.
    """

    def __init__(self, *, score: NLIScore) -> None:
        self._score = score
        self.calls: list[tuple[str, str]] = []

    def score(self, *, premise: str, hypothesis: str) -> NLIScore:
        self.calls.append((premise, hypothesis))
        return self._score


def _typed_edge(
    *,
    src_id: str,
    dst_id: str,
    edge_type: str = "entails",
    confidence: float = 0.9,
) -> TypedEdge:
    """Build a `TypedEdge` with a single synthetic citation span per endpoint."""
    return TypedEdge(
        src_id=src_id,
        dst_id=dst_id,
        type=edge_type,  # type: ignore[arg-type]
        confidence=confidence,
        raw_score=confidence,
        citations=[
            Span(chunk_id=f"{src_id}-chunk", char_start=0, char_end=1, text="x"),
            Span(chunk_id=f"{dst_id}-chunk", char_start=0, char_end=1, text="y"),
        ],
        source="nli",
        paraphrase_votes=None,
    )


# ---------------------------------------------------------------------------
# `entails` handler — Tier-2 NLI on injected claim lookup + scorer
# ---------------------------------------------------------------------------


@pytest.mark.family_verifier_calibration
def test_entails_handler_returns_entailment_verdict_and_top_confidence() -> None:
    """A scorer that returns argmax=entailment must surface as `entailment` + top mass."""
    left = _claim(subject="dog", predicate="is", obj="animal")
    right = _claim(subject="dog", predicate="is", obj="animal")
    scorer = _StubNLIScorer(score=NLIScore(entailment=0.84, contradiction=0.08, neutral=0.08))
    dispatcher = ToolDispatcher()
    deps = MCPHandlerDeps(
        claim_lookup=_build_claim_lookup({"a": left, "b": right}),
        nli_scorer=scorer,
    )
    register_default_handlers(dispatcher=dispatcher, deps=deps)

    result = dispatcher.dispatch(
        tool_name="entails",
        raw_input={"claim_a_id": "a", "claim_b_id": "b"},
    )
    assert isinstance(result, EntailsOutput)
    assert result.verdict == "entailment"
    assert result.confidence == pytest.approx(0.84)
    # Exactly one scorer call — the handler must not double-score.
    assert len(scorer.calls) == 1


@pytest.mark.family_verifier_calibration
def test_entails_handler_surfaces_contradiction_argmax() -> None:
    """Contradiction-dominated distribution must flip the verdict accordingly."""
    left = _claim(subject="x", predicate="is", obj="alive")
    right = _claim(subject="x", predicate="is", obj="dead", polarity="affirmative")
    scorer = _StubNLIScorer(score=NLIScore(entailment=0.05, contradiction=0.91, neutral=0.04))
    dispatcher = ToolDispatcher()
    deps = MCPHandlerDeps(
        claim_lookup=_build_claim_lookup({"a": left, "b": right}),
        nli_scorer=scorer,
    )
    register_default_handlers(dispatcher=dispatcher, deps=deps)

    result = dispatcher.dispatch(
        tool_name="entails",
        raw_input={"claim_a_id": "a", "claim_b_id": "b"},
    )
    assert isinstance(result, EntailsOutput)
    assert result.verdict == "contradiction"
    assert result.confidence == pytest.approx(0.91)


@pytest.mark.family_verifier_calibration
def test_entails_handler_surfaces_neutral_argmax() -> None:
    """Neutral-dominated distribution must surface `neutral`."""
    left = _claim(subject="a", predicate="is", obj="b")
    right = _claim(subject="x", predicate="is", obj="y")
    scorer = _StubNLIScorer(score=NLIScore(entailment=0.10, contradiction=0.10, neutral=0.80))
    dispatcher = ToolDispatcher()
    deps = MCPHandlerDeps(
        claim_lookup=_build_claim_lookup({"a": left, "b": right}),
        nli_scorer=scorer,
    )
    register_default_handlers(dispatcher=dispatcher, deps=deps)

    result = dispatcher.dispatch(
        tool_name="entails",
        raw_input={"claim_a_id": "a", "claim_b_id": "b"},
    )
    assert isinstance(result, EntailsOutput)
    assert result.verdict == "neutral"
    assert result.confidence == pytest.approx(0.80)


@pytest.mark.family_referential_integrity
def test_entails_handler_directional_premise_then_hypothesis() -> None:
    """Premise is `claim_a`, hypothesis is `claim_b` — the order matters."""
    left = _claim(subject="dog", predicate="is", obj="mammal")
    right = _claim(subject="mammal", predicate="is", obj="animal")
    scorer = _StubNLIScorer(score=NLIScore(entailment=0.92, contradiction=0.04, neutral=0.04))
    dispatcher = ToolDispatcher()
    deps = MCPHandlerDeps(
        claim_lookup=_build_claim_lookup({"a": left, "b": right}),
        nli_scorer=scorer,
    )
    register_default_handlers(dispatcher=dispatcher, deps=deps)

    dispatcher.dispatch(
        tool_name="entails",
        raw_input={"claim_a_id": "a", "claim_b_id": "b"},
    )
    assert len(scorer.calls) == 1
    premise, hypothesis = scorer.calls[0]
    # `a` body must come first (premise).
    assert "dog" in premise and "mammal" in premise
    # `b` body must come second (hypothesis).
    assert "mammal" in hypothesis and "animal" in hypothesis


@pytest.mark.family_referential_integrity
def test_entails_handler_not_wired_without_claim_lookup() -> None:
    """No `claim_lookup` ⇒ `entails` must NOT silently register."""
    scorer = _StubNLIScorer(score=NLIScore(entailment=1.0, contradiction=0.0, neutral=0.0))
    dispatcher = ToolDispatcher()
    register_default_handlers(
        dispatcher=dispatcher,
        deps=MCPHandlerDeps(nli_scorer=scorer),
    )
    with pytest.raises(ToolNotImplementedError):
        dispatcher.dispatch(
            tool_name="entails",
            raw_input={"claim_a_id": "a", "claim_b_id": "b"},
        )


@pytest.mark.family_referential_integrity
def test_entails_handler_not_wired_without_nli_scorer() -> None:
    """No `nli_scorer` ⇒ `entails` must NOT silently register."""
    dispatcher = ToolDispatcher()
    register_default_handlers(
        dispatcher=dispatcher,
        deps=MCPHandlerDeps(claim_lookup=_build_claim_lookup({})),
    )
    with pytest.raises(ToolNotImplementedError):
        dispatcher.dispatch(
            tool_name="entails",
            raw_input={"claim_a_id": "a", "claim_b_id": "b"},
        )


@pytest.mark.family_referential_integrity
def test_entails_handler_propagates_lookup_error() -> None:
    """An unknown claim id from the lookup must surface as `LookupError`."""
    scorer = _StubNLIScorer(score=NLIScore(entailment=1.0, contradiction=0.0, neutral=0.0))
    dispatcher = ToolDispatcher()
    register_default_handlers(
        dispatcher=dispatcher,
        deps=MCPHandlerDeps(
            claim_lookup=_build_claim_lookup({}),
            nli_scorer=scorer,
        ),
    )
    with pytest.raises(LookupError):
        dispatcher.dispatch(
            tool_name="entails",
            raw_input={"claim_a_id": "missing", "claim_b_id": "missing"},
        )


# ---------------------------------------------------------------------------
# `qa` handler — wraps an injected runner closure
# ---------------------------------------------------------------------------


def _identity_qa_runner(reply: AnswerWithTrace) -> Callable[[str, str], AnswerWithTrace]:
    """Return a constant runner that captures the call payload."""

    captured: dict[str, str] = {}

    def _run(target: str, query: str) -> AnswerWithTrace:
        captured["target"] = target
        captured["query"] = query
        return reply

    _run.captured = captured  # type: ignore[attr-defined]
    return _run


@pytest.mark.family_referential_integrity
def test_qa_handler_returns_runner_reply_unchanged() -> None:
    """The runner's `AnswerWithTrace` must surface verbatim through `QAOutput`."""
    reply = AnswerWithTrace(
        answer="Bishop says CNNs use weight sharing.",
        citations=["bishop-ch5-1"],
        trace_steps=["retrieve", "generate", "decompose", "verify"],
        confidence=0.87,
    )
    runner = _identity_qa_runner(reply)
    dispatcher = ToolDispatcher()
    register_default_handlers(
        dispatcher=dispatcher,
        deps=MCPHandlerDeps(qa_runner=runner),
    )

    result = dispatcher.dispatch(
        tool_name="qa",
        raw_input={"target": "doc-bishop", "query": "What is weight sharing?"},
    )
    assert isinstance(result, QAOutput)
    assert result.reply == reply
    assert runner.captured == {"target": "doc-bishop", "query": "What is weight sharing?"}  # type: ignore[attr-defined]


@pytest.mark.family_referential_integrity
def test_qa_handler_not_wired_without_runner() -> None:
    """No `qa_runner` ⇒ `qa` must NOT silently register."""
    dispatcher = ToolDispatcher()
    register_default_handlers(dispatcher=dispatcher, deps=MCPHandlerDeps())
    with pytest.raises(ToolNotImplementedError):
        dispatcher.dispatch(
            tool_name="qa",
            raw_input={"target": "x", "query": "y"},
        )


@pytest.mark.family_referential_integrity
def test_qa_handler_clamps_runner_reply_confidence() -> None:
    """Runner confidence outside [0,1] must surface as a clean validation error.

    Pydantic's `UnitInterval` alias enforces the range — the handler
    cannot fabricate a clamp without altering the runner's intent.
    """
    with pytest.raises(ValueError):
        AnswerWithTrace(
            answer="x",
            citations=[],
            trace_steps=[],
            confidence=1.5,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# `map` handler — typed-edge → Mermaid rendering
# ---------------------------------------------------------------------------


@pytest.mark.family_referential_integrity
def test_map_handler_renders_mermaid_graph_lr_block() -> None:
    """One edge ⇒ one labelled arrow plus two node declarations."""
    edges = [
        _typed_edge(src_id="claim-a", dst_id="claim-b", edge_type="entails"),
    ]

    def _supplier(doc_id: str) -> Sequence[TypedEdge]:
        assert doc_id == "doc-1"
        return edges

    dispatcher = ToolDispatcher()
    register_default_handlers(
        dispatcher=dispatcher,
        deps=MCPHandlerDeps(typed_edges_for_doc_supplier=_supplier),
    )
    result = dispatcher.dispatch(
        tool_name="map",
        raw_input={"doc_id": "doc-1"},
    )
    assert isinstance(result, MapOutput)
    assert result.edge_count == 1
    assert sorted(result.node_ids) == ["claim-a", "claim-b"]
    # `graph LR` header + node declarations + edge arrow + closing fence
    assert "graph LR" in result.mermaid
    assert "-- entails -->" in result.mermaid


@pytest.mark.family_determinism
def test_map_handler_orders_nodes_and_edges_deterministically() -> None:
    """Two runs over the same edge list ⇒ byte-identical Mermaid blocks."""
    edges = [
        _typed_edge(src_id="b", dst_id="a", edge_type="entails"),
        _typed_edge(src_id="a", dst_id="c", edge_type="contradicts"),
        _typed_edge(src_id="a", dst_id="b", edge_type="entails"),
    ]

    def _supplier(_doc_id: str) -> Sequence[TypedEdge]:
        return edges

    dispatcher = ToolDispatcher()
    register_default_handlers(
        dispatcher=dispatcher,
        deps=MCPHandlerDeps(typed_edges_for_doc_supplier=_supplier),
    )
    first = dispatcher.dispatch(tool_name="map", raw_input={"doc_id": "doc-1"})
    second = dispatcher.dispatch(tool_name="map", raw_input={"doc_id": "doc-1"})
    assert isinstance(first, MapOutput)
    assert isinstance(second, MapOutput)
    assert first.mermaid == second.mermaid
    assert first.node_ids == second.node_ids
    assert first.edge_count == second.edge_count == 3


@pytest.mark.family_referential_integrity
def test_map_handler_empty_edges_renders_empty_placeholder() -> None:
    """Doc with zero typed edges ⇒ valid Mermaid empty-graph placeholder."""

    def _supplier(_doc_id: str) -> Sequence[TypedEdge]:
        return []

    dispatcher = ToolDispatcher()
    register_default_handlers(
        dispatcher=dispatcher,
        deps=MCPHandlerDeps(typed_edges_for_doc_supplier=_supplier),
    )
    result = dispatcher.dispatch(tool_name="map", raw_input={"doc_id": "empty-doc"})
    assert isinstance(result, MapOutput)
    assert result.edge_count == 0
    assert result.node_ids == []
    assert "graph LR" in result.mermaid


@pytest.mark.family_referential_integrity
def test_map_handler_not_wired_without_supplier() -> None:
    """No `typed_edges_for_doc_supplier` ⇒ `map` must NOT silently register."""
    dispatcher = ToolDispatcher()
    register_default_handlers(dispatcher=dispatcher, deps=MCPHandlerDeps())
    with pytest.raises(ToolNotImplementedError):
        dispatcher.dispatch(tool_name="map", raw_input={"doc_id": "d"})


@pytest.mark.family_referential_integrity
def test_map_handler_filters_argument_is_accepted_passthrough() -> None:
    """The schema's `filters` slot must be accepted (default empty dict)."""
    edges = [_typed_edge(src_id="x", dst_id="y", edge_type="entails")]

    def _supplier(_doc_id: str) -> Sequence[TypedEdge]:
        return edges

    dispatcher = ToolDispatcher()
    register_default_handlers(
        dispatcher=dispatcher,
        deps=MCPHandlerDeps(typed_edges_for_doc_supplier=_supplier),
    )
    # Default `filters` = {} branch.
    result = dispatcher.dispatch(tool_name="map", raw_input={"doc_id": "d"})
    assert isinstance(result, MapOutput)
    # Explicit empty filters branch.
    result2 = dispatcher.dispatch(tool_name="map", raw_input={"doc_id": "d", "filters": {}})
    assert isinstance(result2, MapOutput)
    assert result.mermaid == result2.mermaid


# ---------------------------------------------------------------------------
# Wiring policy + factory return value
# ---------------------------------------------------------------------------


@pytest.mark.family_referential_integrity
def test_register_default_handlers_wires_entails_qa_map_when_deps_present() -> None:
    """The factory must report every LLM handler whose dep is satisfied."""
    scorer = _StubNLIScorer(score=NLIScore(entailment=1.0, contradiction=0.0, neutral=0.0))
    runner = _identity_qa_runner(
        AnswerWithTrace(answer="x", citations=[], trace_steps=[], confidence=0.5)
    )
    dispatcher = ToolDispatcher()
    deps = MCPHandlerDeps(
        claim_lookup=_build_claim_lookup({}),
        nli_scorer=scorer,
        qa_runner=runner,
        typed_edges_for_doc_supplier=lambda _doc_id: [],
    )
    wired = register_default_handlers(dispatcher=dispatcher, deps=deps)
    assert {"entails", "qa", "map"}.issubset(wired)


@pytest.mark.family_referential_integrity
def test_factory_leaves_llm_handlers_unwired_when_no_llm_deps() -> None:
    """Empty `MCPHandlerDeps` ⇒ every LLM tool stays unregistered."""
    dispatcher = ToolDispatcher()
    register_default_handlers(dispatcher=dispatcher, deps=MCPHandlerDeps())
    for tool_name, raw in [
        ("entails", {"claim_a_id": "a", "claim_b_id": "b"}),
        ("qa", {"target": "d", "query": "q"}),
        ("map", {"doc_id": "d"}),
    ]:
        with pytest.raises(ToolNotImplementedError):
            dispatcher.dispatch(tool_name=tool_name, raw_input=raw)


@pytest.mark.family_referential_integrity
def test_entails_input_schema_round_trips_through_handler() -> None:
    """Sanity: input schema accepts the dict the handler reads."""
    from ctrldoc.orch.tools import EntailsInput

    inp = EntailsInput(claim_a_id="a", claim_b_id="b")
    assert inp.claim_a_id == "a"
    assert inp.claim_b_id == "b"


@pytest.mark.family_referential_integrity
def test_qa_handler_rejects_empty_target_or_query() -> None:
    """The input schema's `min_length=1` constraint must surface as validation."""
    runner = _identity_qa_runner(
        AnswerWithTrace(answer="x", citations=[], trace_steps=[], confidence=0.5)
    )
    dispatcher = ToolDispatcher()
    register_default_handlers(dispatcher=dispatcher, deps=MCPHandlerDeps(qa_runner=runner))

    with pytest.raises(ToolValidationError):
        dispatcher.dispatch(tool_name="qa", raw_input={"target": "", "query": "q"})
    with pytest.raises(ToolValidationError):
        dispatcher.dispatch(tool_name="qa", raw_input={"target": "t", "query": ""})


# ---------------------------------------------------------------------------
# `serve_stdio` round-trip — every LLM handler reachable over the wire
# ---------------------------------------------------------------------------


@pytest.mark.family_referential_integrity
def test_serve_stdio_round_trip_for_entails_handler() -> None:
    """A stock host can `tools/call entails` over the stdio loop."""
    left = _claim(subject="dog", predicate="is", obj="animal")
    right = _claim(subject="dog", predicate="is", obj="animal")
    scorer = _StubNLIScorer(score=NLIScore(entailment=0.75, contradiction=0.10, neutral=0.15))
    dispatcher = ToolDispatcher()
    register_default_handlers(
        dispatcher=dispatcher,
        deps=MCPHandlerDeps(
            claim_lookup=_build_claim_lookup({"L": left, "R": right}),
            nli_scorer=scorer,
        ),
    )
    server = MCPServer(dispatcher=dispatcher)

    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "entails",
            "arguments": {"claim_a_id": "L", "claim_b_id": "R"},
        },
    }
    instream = io.StringIO(json.dumps(request) + "\n")
    outstream = io.StringIO()
    serve_stdio(server=server, instream=instream, outstream=outstream)

    response = json.loads(outstream.getvalue().splitlines()[0])
    assert response["id"] == 1
    assert response["result"]["isError"] is False
    body = response["result"]["structuredContent"]
    assert body == {"verdict": "entailment", "confidence": pytest.approx(0.75)}


@pytest.mark.family_referential_integrity
def test_serve_stdio_round_trip_for_qa_handler() -> None:
    """A stock host can `tools/call qa` over the stdio loop."""
    reply = AnswerWithTrace(
        answer="weight sharing reduces parameter count",
        citations=["bishop-ch5-1"],
        trace_steps=["retrieve", "generate"],
        confidence=0.92,
    )
    runner = _identity_qa_runner(reply)
    dispatcher = ToolDispatcher()
    register_default_handlers(dispatcher=dispatcher, deps=MCPHandlerDeps(qa_runner=runner))
    server = MCPServer(dispatcher=dispatcher)

    request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "qa",
            "arguments": {"target": "doc-bishop", "query": "What is weight sharing?"},
        },
    }
    instream = io.StringIO(json.dumps(request) + "\n")
    outstream = io.StringIO()
    serve_stdio(server=server, instream=instream, outstream=outstream)

    response = json.loads(outstream.getvalue().splitlines()[0])
    assert response["id"] == 2
    assert response["result"]["isError"] is False
    body = response["result"]["structuredContent"]
    assert body["reply"]["answer"] == reply.answer
    assert body["reply"]["citations"] == reply.citations
    assert body["reply"]["confidence"] == pytest.approx(0.92)


@pytest.mark.family_referential_integrity
def test_serve_stdio_round_trip_for_map_handler() -> None:
    """A stock host can `tools/call map` over the stdio loop."""
    edges = [
        _typed_edge(src_id="alpha", dst_id="beta", edge_type="entails"),
    ]

    def _supplier(_doc_id: str) -> Sequence[TypedEdge]:
        return edges

    dispatcher = ToolDispatcher()
    register_default_handlers(
        dispatcher=dispatcher,
        deps=MCPHandlerDeps(typed_edges_for_doc_supplier=_supplier),
    )
    server = MCPServer(dispatcher=dispatcher)

    request = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "map",
            "arguments": {"doc_id": "doc-x"},
        },
    }
    instream = io.StringIO(json.dumps(request) + "\n")
    outstream = io.StringIO()
    serve_stdio(server=server, instream=instream, outstream=outstream)

    response = json.loads(outstream.getvalue().splitlines()[0])
    assert response["id"] == 3
    assert response["result"]["isError"] is False
    body = response["result"]["structuredContent"]
    assert body["edge_count"] == 1
    assert sorted(body["node_ids"]) == ["alpha", "beta"]
    assert "graph LR" in body["mermaid"]


# ---------------------------------------------------------------------------
# Wave-isolation guard — earlier waves stay green after S-161 wiring
# ---------------------------------------------------------------------------


@pytest.mark.family_referential_integrity
def test_llm_wiring_does_not_break_pure_python_wave() -> None:
    """Pure-Python + OT handlers must still wire when LLM deps are present."""
    scorer = _StubNLIScorer(score=NLIScore(entailment=1.0, contradiction=0.0, neutral=0.0))
    runner = _identity_qa_runner(
        AnswerWithTrace(answer="x", citations=[], trace_steps=[], confidence=0.5)
    )
    dispatcher = ToolDispatcher()
    wired = register_default_handlers(
        dispatcher=dispatcher,
        deps=MCPHandlerDeps(
            claim_lookup=_build_claim_lookup({}),
            nli_scorer=scorer,
            qa_runner=runner,
            typed_edges_for_doc_supplier=lambda _doc_id: [],
        ),
    )
    # Pure-Python wave (S-157).
    assert "optimal_transport" in wired
    assert "calibration" in wired
    assert "subsumes" in wired
    # LLM wave.
    assert "entails" in wired
    assert "qa" in wired
    assert "map" in wired
