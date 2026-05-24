"""§6.10 + §11 — storage-backed MCP handlers (`get_claim`, `lookup_concept`, `traverse`).

The pure-Python handler floor in `test_mcp_default_handlers.py` covers
the three engines that have no storage or LLM dependency (`subsumes`,
`optimal_transport`, `calibration`). This module pins the next wave —
the storage-backed handlers whose deps are satisfied by a per-doc
SQLiteStore opened from `<runs_path>/indexes/*`:

* ``get_claim`` → `Store.get_claim(claim_id) -> Claim`. The handler
  returns the persisted `Claim` (§7) verbatim. A missing id raises
  `LookupError` so the MCP server lifts it into an `isError=true`
  envelope without fabricating a verdict (§13 non-negotiable 3).

* ``lookup_concept`` → canonical-name → `ConceptId | None`. The handler
  scans the concept rollup, prefers an exact case-sensitive match on
  ``canonical_name``, falls back to a case-insensitive match, and
  reports `None` when no concept carries the surface form. The "None"
  branch is an explicit answer rather than a refusal — the schema
  surfaces `concept_id: str | None` precisely so a host can render
  "no concept by that name" without an extra round-trip.

* ``traverse`` → `GraphWalkRetriever.walk` over the typed-edge graph
  restricted to a single ``edge_type`` and ``direction``. ``hops`` caps
  the PPR power-iteration ceiling so a "1-hop" request walks exactly
  once before harvesting. The output is the top-`hops` neighbour node
  ids by stationary probability (the seed itself ranks first; the
  handler trims it from the response so the caller sees only the nodes
  the walk reached).

The factory `register_default_handlers(dispatcher, deps)` extends the
S-157 wave: each storage-backed handler wires only when its
corresponding `deps.*` callable is set. Missing deps leaves the tool
unregistered so the dispatcher refuses the call with
`ToolNotImplementedError` — never a silent no-op.

`build_store_backed_deps(runs_path)` is the convenience factory that
opens every `<runs_path>/indexes/*.db` as a SQLiteStore, unions the
`Claim` / `Concept` / `TypedEdge` rows across stores, and returns an
`MCPHandlerDeps` ready to plug into the dispatcher. The runtime cost is
linear in the number of per-doc stores; the v1 workspace cardinality
(handful of docs) makes the scan negligible.

SPEC-REF: §6.10 (tool-using orchestrator), §11 (MCP server)
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from ctrldoc.mcp.handlers import (
    MCPHandlerDeps,
    build_store_backed_deps,
    register_default_handlers,
)
from ctrldoc.mcp.server import MCPServer, serve_stdio
from ctrldoc.models_v1 import Claim, Concept, TypedEdge
from ctrldoc.orch.tools import (
    GetClaimOutput,
    LookupConceptOutput,
    ToolDispatcher,
    ToolNotImplementedError,
    TraverseOutput,
)
from ctrldoc.store.sqlite import SQLiteStore

# ---------------------------------------------------------------------------
# Fixtures — minimal `Claim` / `Concept` / `TypedEdge` builders
# ---------------------------------------------------------------------------


def _claim(
    cid: str,
    *,
    doc_id: str = "doc-A",
    subject: str = "dog",
    obj: str = "animal",
    predicate: str = "is",
) -> Claim:
    return Claim(
        id=cid,
        doc_id=doc_id,
        text=f"{subject} {predicate} {obj}",
        subject=subject,
        predicate=predicate,
        object=obj,
        polarity="+",
        modality=None,
        qualifier={},
        span_refs=[],
        section_id="sec-1",
        concept_ids=[],
        typed_slots={},
        confidence=0.9,
    )


def _concept(
    cid: str,
    *,
    name: str,
    doc_ids: Sequence[str] = ("doc-A",),
    aliases: Sequence[str] = (),
) -> Concept:
    return Concept(
        id=cid,
        canonical_name=name,
        aliases=list(aliases),
        primitive_type="Entity",
        mention_claim_ids=[],
        doc_ids=list(doc_ids),
    )


def _edge(
    *,
    src: str,
    dst: str,
    edge_type: str = "depends_on",
    confidence: float = 0.9,
) -> TypedEdge:
    return TypedEdge(
        src_id=src,
        dst_id=dst,
        type=edge_type,  # type: ignore[arg-type]
        confidence=confidence,
        raw_score=confidence,
        citations=[],
        source="heuristic",
        paraphrase_votes=None,
    )


# ---------------------------------------------------------------------------
# `get_claim` handler — Store-backed lookup
# ---------------------------------------------------------------------------


@pytest.mark.family_referential_integrity
def test_get_claim_handler_returns_persisted_claim() -> None:
    """A persisted claim id must round-trip through the dispatcher unchanged."""
    persisted = {"c-1": _claim("c-1", subject="dog", obj="animal")}

    def _lookup(claim_id: str) -> Claim:
        try:
            return persisted[claim_id]
        except KeyError as exc:
            raise LookupError(f"unknown claim id: {claim_id!r}") from exc

    dispatcher = ToolDispatcher()
    register_default_handlers(
        dispatcher=dispatcher,
        deps=MCPHandlerDeps(claim_record_lookup=_lookup),
    )

    result = dispatcher.dispatch(tool_name="get_claim", raw_input={"claim_id": "c-1"})
    assert isinstance(result, GetClaimOutput)
    assert result.claim.id == "c-1"
    assert result.claim.subject == "dog"
    assert result.claim.predicate == "is"
    assert result.claim.object == "animal"


@pytest.mark.family_referential_integrity
def test_get_claim_handler_propagates_lookup_error() -> None:
    """An unknown claim id must raise so the MCP server lifts an `isError=true`."""

    def _lookup(claim_id: str) -> Claim:
        raise LookupError(f"unknown claim id: {claim_id!r}")

    dispatcher = ToolDispatcher()
    register_default_handlers(
        dispatcher=dispatcher,
        deps=MCPHandlerDeps(claim_record_lookup=_lookup),
    )

    with pytest.raises(LookupError):
        dispatcher.dispatch(tool_name="get_claim", raw_input={"claim_id": "missing"})


@pytest.mark.family_referential_integrity
def test_get_claim_handler_not_wired_without_record_lookup() -> None:
    """No `claim_record_lookup` injected = handler must NOT silently register."""
    dispatcher = ToolDispatcher()
    register_default_handlers(dispatcher=dispatcher, deps=MCPHandlerDeps())

    with pytest.raises(ToolNotImplementedError):
        dispatcher.dispatch(tool_name="get_claim", raw_input={"claim_id": "c-1"})


# ---------------------------------------------------------------------------
# `lookup_concept` handler — canonical-name → concept rollup
# ---------------------------------------------------------------------------


@pytest.mark.family_referential_integrity
def test_lookup_concept_handler_returns_id_for_exact_match() -> None:
    """Exact `canonical_name` must surface its concept id."""
    concepts = {
        "k-1": _concept("k-1", name="Dog"),
        "k-2": _concept("k-2", name="Cat"),
    }

    def _lookup(name: str) -> Concept | None:
        for c in concepts.values():
            if c.canonical_name == name:
                return c
        return None

    dispatcher = ToolDispatcher()
    register_default_handlers(
        dispatcher=dispatcher,
        deps=MCPHandlerDeps(concept_name_lookup=_lookup),
    )

    result = dispatcher.dispatch(tool_name="lookup_concept", raw_input={"name": "Dog"})
    assert isinstance(result, LookupConceptOutput)
    assert result.concept_id == "k-1"


@pytest.mark.family_referential_integrity
def test_lookup_concept_handler_returns_none_for_unknown_name() -> None:
    """Unknown name must surface a structurally valid `None`, not refusal."""

    def _lookup(_name: str) -> Concept | None:
        return None

    dispatcher = ToolDispatcher()
    register_default_handlers(
        dispatcher=dispatcher,
        deps=MCPHandlerDeps(concept_name_lookup=_lookup),
    )

    result = dispatcher.dispatch(tool_name="lookup_concept", raw_input={"name": "Unknown"})
    assert isinstance(result, LookupConceptOutput)
    assert result.concept_id is None


@pytest.mark.family_referential_integrity
def test_lookup_concept_handler_not_wired_without_concept_lookup() -> None:
    """No `concept_name_lookup` injected = handler must NOT silently register."""
    dispatcher = ToolDispatcher()
    register_default_handlers(dispatcher=dispatcher, deps=MCPHandlerDeps())

    with pytest.raises(ToolNotImplementedError):
        dispatcher.dispatch(tool_name="lookup_concept", raw_input={"name": "Dog"})


# ---------------------------------------------------------------------------
# `traverse` handler — `GraphWalkRetriever.walk` over typed edges
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_traverse_handler_returns_forward_neighbours_for_seed() -> None:
    """One-hop forward walk along `depends_on` must reach the immediate target."""
    edges = [
        _edge(src="A", dst="B", edge_type="depends_on"),
        _edge(src="B", dst="C", edge_type="depends_on"),
        _edge(src="A", dst="X", edge_type="is_a"),  # different type — must be ignored.
    ]

    def _edges_supplier() -> list[TypedEdge]:
        return list(edges)

    dispatcher = ToolDispatcher()
    register_default_handlers(
        dispatcher=dispatcher,
        deps=MCPHandlerDeps(typed_edges_supplier=_edges_supplier),
    )

    result = dispatcher.dispatch(
        tool_name="traverse",
        raw_input={
            "node_id": "A",
            "edge_type": "depends_on",
            "direction": "forward",
            "hops": 1,
        },
    )
    assert isinstance(result, TraverseOutput)
    # Seed must not appear in the returned neighbour list.
    assert "A" not in result.node_ids
    # B is the one-hop forward neighbour along `depends_on`.
    assert "B" in result.node_ids
    # X is reached via a *different* edge type and must be filtered out.
    assert "X" not in result.node_ids


@pytest.mark.family_determinism
def test_traverse_handler_reverse_direction_walks_against_arrow() -> None:
    """`direction=reverse` must walk against the edge arrow."""
    edges = [_edge(src="A", dst="B", edge_type="depends_on")]

    def _edges_supplier() -> list[TypedEdge]:
        return list(edges)

    dispatcher = ToolDispatcher()
    register_default_handlers(
        dispatcher=dispatcher,
        deps=MCPHandlerDeps(typed_edges_supplier=_edges_supplier),
    )

    result = dispatcher.dispatch(
        tool_name="traverse",
        raw_input={
            "node_id": "B",
            "edge_type": "depends_on",
            "direction": "reverse",
            "hops": 1,
        },
    )
    assert isinstance(result, TraverseOutput)
    assert "A" in result.node_ids
    assert "B" not in result.node_ids


@pytest.mark.family_determinism
def test_traverse_handler_both_direction_walks_either_way() -> None:
    """`direction=both` must merge forward and reverse reachability."""
    edges = [
        _edge(src="A", dst="B", edge_type="depends_on"),
        _edge(src="C", dst="A", edge_type="depends_on"),
    ]

    def _edges_supplier() -> list[TypedEdge]:
        return list(edges)

    dispatcher = ToolDispatcher()
    register_default_handlers(
        dispatcher=dispatcher,
        deps=MCPHandlerDeps(typed_edges_supplier=_edges_supplier),
    )

    result = dispatcher.dispatch(
        tool_name="traverse",
        raw_input={
            "node_id": "A",
            "edge_type": "depends_on",
            "direction": "both",
            "hops": 1,
        },
    )
    assert isinstance(result, TraverseOutput)
    assert set(result.node_ids) >= {"B", "C"}
    assert "A" not in result.node_ids


@pytest.mark.family_determinism
def test_traverse_handler_returns_empty_when_no_outgoing_edges() -> None:
    """A seed with no matching outgoing edges must yield an empty result."""
    edges: list[TypedEdge] = []

    def _edges_supplier() -> list[TypedEdge]:
        return list(edges)

    dispatcher = ToolDispatcher()
    register_default_handlers(
        dispatcher=dispatcher,
        deps=MCPHandlerDeps(typed_edges_supplier=_edges_supplier),
    )

    result = dispatcher.dispatch(
        tool_name="traverse",
        raw_input={
            "node_id": "lone",
            "edge_type": "depends_on",
            "direction": "forward",
            "hops": 1,
        },
    )
    assert isinstance(result, TraverseOutput)
    assert result.node_ids == []


@pytest.mark.family_referential_integrity
def test_traverse_handler_not_wired_without_edges_supplier() -> None:
    """No `typed_edges_supplier` injected = handler must NOT silently register."""
    dispatcher = ToolDispatcher()
    register_default_handlers(dispatcher=dispatcher, deps=MCPHandlerDeps())

    with pytest.raises(ToolNotImplementedError):
        dispatcher.dispatch(
            tool_name="traverse",
            raw_input={
                "node_id": "A",
                "edge_type": "depends_on",
                "direction": "forward",
                "hops": 1,
            },
        )


# ---------------------------------------------------------------------------
# `register_default_handlers` wiring policy — set of tool names
# ---------------------------------------------------------------------------


@pytest.mark.family_referential_integrity
def test_register_default_handlers_wires_storage_tools_when_deps_set() -> None:
    """All three storage-backed handlers must wire when their deps are set."""

    def _claim_lookup(_cid: str) -> Claim:
        raise LookupError("unused")

    def _concept_lookup(_name: str) -> Concept | None:
        return None

    def _edges_supplier() -> list[TypedEdge]:
        return []

    dispatcher = ToolDispatcher()
    wired = register_default_handlers(
        dispatcher=dispatcher,
        deps=MCPHandlerDeps(
            claim_record_lookup=_claim_lookup,
            concept_name_lookup=_concept_lookup,
            typed_edges_supplier=_edges_supplier,
        ),
    )
    assert "get_claim" in wired
    assert "lookup_concept" in wired
    assert "traverse" in wired


@pytest.mark.family_referential_integrity
def test_register_default_handlers_leaves_unwired_for_remaining_waves() -> None:
    """S-159..S-161 tools must still raise `ToolNotImplementedError`."""

    def _claim_lookup(_cid: str) -> Claim:
        raise LookupError("unused")

    def _concept_lookup(_name: str) -> Concept | None:
        return None

    def _edges_supplier() -> list[TypedEdge]:
        return []

    dispatcher = ToolDispatcher()
    register_default_handlers(
        dispatcher=dispatcher,
        deps=MCPHandlerDeps(
            claim_record_lookup=_claim_lookup,
            concept_name_lookup=_concept_lookup,
            typed_edges_supplier=_edges_supplier,
        ),
    )
    # OT and LLM handlers ship in later slices and must stay unwired here.
    for tool_name, raw in [
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
# `build_store_backed_deps` — open every `<runs_path>/indexes/*.db`
# ---------------------------------------------------------------------------


@pytest.mark.family_referential_integrity
def test_build_store_backed_deps_unions_claims_across_per_doc_stores(
    tmp_path: Path,
) -> None:
    """Opening every per-doc store must produce claim / concept / edge views
    that span the union across stores."""
    indexes = tmp_path / "indexes"
    indexes.mkdir()

    # Two per-doc stores — one claim and one concept per store.
    with SQLiteStore(indexes / "doc1.db") as store_a:
        store_a.append_claim(_claim("c-a", doc_id="doc-A"))
        store_a.add_concepts([_concept("k-a", name="Alpha", doc_ids=["doc-A"])])
        store_a.append_typed_edge(_edge(src="c-a", dst="c-b", edge_type="depends_on"))
    with SQLiteStore(indexes / "doc2.db") as store_b:
        store_b.append_claim(_claim("c-b", doc_id="doc-B", subject="cat"))
        store_b.add_concepts([_concept("k-b", name="Beta", doc_ids=["doc-B"])])

    deps = build_store_backed_deps(runs_path=tmp_path)
    assert deps.claim_record_lookup is not None
    assert deps.concept_name_lookup is not None
    assert deps.typed_edges_supplier is not None

    # `get_claim` resolves across stores.
    claim_a = deps.claim_record_lookup("c-a")
    claim_b = deps.claim_record_lookup("c-b")
    assert claim_a.id == "c-a"
    assert claim_b.id == "c-b"

    # `lookup_concept` finds either canonical name across stores.
    concept_a = deps.concept_name_lookup("Alpha")
    concept_b = deps.concept_name_lookup("Beta")
    assert concept_a is not None and concept_a.id == "k-a"
    assert concept_b is not None and concept_b.id == "k-b"
    assert deps.concept_name_lookup("Gamma") is None

    # `typed_edges_supplier` exposes every persisted edge across stores.
    edges = deps.typed_edges_supplier()
    assert any(e.src_id == "c-a" and e.dst_id == "c-b" for e in edges)


@pytest.mark.family_referential_integrity
def test_build_store_backed_deps_raises_lookup_error_for_missing_claim(
    tmp_path: Path,
) -> None:
    """The claim_record_lookup closure must raise LookupError on a miss."""
    indexes = tmp_path / "indexes"
    indexes.mkdir()
    with SQLiteStore(indexes / "doc1.db") as store:
        store.append_claim(_claim("c-a"))

    deps = build_store_backed_deps(runs_path=tmp_path)
    assert deps.claim_record_lookup is not None
    with pytest.raises(LookupError):
        deps.claim_record_lookup("c-missing")


@pytest.mark.family_referential_integrity
def test_build_store_backed_deps_handles_empty_indexes_dir(tmp_path: Path) -> None:
    """No per-doc store on disk = closures still wire but return empty views."""
    deps = build_store_backed_deps(runs_path=tmp_path)
    assert deps.claim_record_lookup is not None
    assert deps.concept_name_lookup is not None
    assert deps.typed_edges_supplier is not None
    assert deps.concept_name_lookup("anything") is None
    assert deps.typed_edges_supplier() == []
    with pytest.raises(LookupError):
        deps.claim_record_lookup("c-missing")


# ---------------------------------------------------------------------------
# `serve_stdio` round-trip — storage-backed handlers reachable over the wire
# ---------------------------------------------------------------------------


@pytest.mark.family_referential_integrity
def test_serve_stdio_round_trip_for_get_claim_handler() -> None:
    """A stock host can `tools/call get_claim` when a record lookup is wired."""
    import io

    persisted = {"c-1": _claim("c-1", subject="dog", obj="animal")}

    def _lookup(claim_id: str) -> Claim:
        try:
            return persisted[claim_id]
        except KeyError as exc:
            raise LookupError(f"unknown: {claim_id!r}") from exc

    dispatcher = ToolDispatcher()
    register_default_handlers(
        dispatcher=dispatcher,
        deps=MCPHandlerDeps(claim_record_lookup=_lookup),
    )
    server = MCPServer(dispatcher=dispatcher)

    request = {
        "jsonrpc": "2.0",
        "id": 11,
        "method": "tools/call",
        "params": {
            "name": "get_claim",
            "arguments": {"claim_id": "c-1"},
        },
    }
    instream = io.StringIO(json.dumps(request) + "\n")
    outstream = io.StringIO()
    serve_stdio(server=server, instream=instream, outstream=outstream)

    response = json.loads(outstream.getvalue().splitlines()[0])
    assert response["id"] == 11
    assert response["result"]["isError"] is False
    body = response["result"]["structuredContent"]
    assert body["claim"]["id"] == "c-1"
    assert body["claim"]["subject"] == "dog"


@pytest.mark.family_referential_integrity
def test_serve_stdio_round_trip_for_lookup_concept_handler() -> None:
    """A stock host can `tools/call lookup_concept` when a name lookup is wired."""
    import io

    concepts = {"k-1": _concept("k-1", name="Dog")}

    def _lookup(name: str) -> Concept | None:
        for c in concepts.values():
            if c.canonical_name == name:
                return c
        return None

    dispatcher = ToolDispatcher()
    register_default_handlers(
        dispatcher=dispatcher,
        deps=MCPHandlerDeps(concept_name_lookup=_lookup),
    )
    server = MCPServer(dispatcher=dispatcher)

    request = {
        "jsonrpc": "2.0",
        "id": 12,
        "method": "tools/call",
        "params": {
            "name": "lookup_concept",
            "arguments": {"name": "Dog"},
        },
    }
    instream = io.StringIO(json.dumps(request) + "\n")
    outstream = io.StringIO()
    serve_stdio(server=server, instream=instream, outstream=outstream)

    response = json.loads(outstream.getvalue().splitlines()[0])
    assert response["id"] == 12
    assert response["result"]["isError"] is False
    body = response["result"]["structuredContent"]
    assert body["concept_id"] == "k-1"


@pytest.mark.family_referential_integrity
def test_serve_stdio_round_trip_for_traverse_handler() -> None:
    """A stock host can `tools/call traverse` when an edges supplier is wired."""
    import io

    edges = [_edge(src="A", dst="B", edge_type="depends_on")]

    def _edges_supplier() -> list[TypedEdge]:
        return list(edges)

    dispatcher = ToolDispatcher()
    register_default_handlers(
        dispatcher=dispatcher,
        deps=MCPHandlerDeps(typed_edges_supplier=_edges_supplier),
    )
    server = MCPServer(dispatcher=dispatcher)

    request = {
        "jsonrpc": "2.0",
        "id": 13,
        "method": "tools/call",
        "params": {
            "name": "traverse",
            "arguments": {
                "node_id": "A",
                "edge_type": "depends_on",
                "direction": "forward",
                "hops": 1,
            },
        },
    }
    instream = io.StringIO(json.dumps(request) + "\n")
    outstream = io.StringIO()
    serve_stdio(server=server, instream=instream, outstream=outstream)

    response = json.loads(outstream.getvalue().splitlines()[0])
    assert response["id"] == 13
    assert response["result"]["isError"] is False
    body = response["result"]["structuredContent"]
    assert "B" in body["node_ids"]
    assert "A" not in body["node_ids"]
