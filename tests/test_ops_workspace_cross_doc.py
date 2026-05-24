"""Workspace cross-doc edge wiring on `workspace add`.

When `WorkspaceManager.add(name, doc_id)` lands a new member, the
manager triggers the §6.7 cross-doc bridge over the new doc + the
existing members:

1. **Concept bridge.** Concepts persisted for the incoming doc in its
   per-doc store are mirrored into the workspace store so the shared
   concept-lattice rollup (`WorkspaceInfo.shared_concept_ids`) is
   non-empty after a doc with concepts joins — the §6.7 lattice slice
   has to surface at least one concept per member doc.
2. **Cross-doc edge inference.** When an `NLIScorer` is wired in and
   at least one prior member already lives in the workspace, the
   manager fans out `CrossDocEdgeInferer` over `{existing_doc_ids:
   claims, new_doc_id: new_claims}` and persists every emitted
   `TypedEdge` into the workspace store's `cross_doc_edges` table.

The resolver is injected so unit tests can stub the per-doc lookup;
in production the CLI plugs in the per-doc SQLiteStore opener.

The cost contract from `CrossDocEdgeInferer` propagates: at most
`k * |new_doc_claims|` NLI calls per existing member, so adding a doc
to a workspace of size N issues at most `k * |new| * N` calls — linear
in `N`, never quadratic across the full member set.

SPEC-REF: §6.7 (workspace cross-doc edges, lazy + cached + linear)
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest

from ctrldoc.eval.calibration import NLIScore
from ctrldoc.models import Span
from ctrldoc.models_v1 import Claim, Concept
from ctrldoc.ops.cross_doc_edges import (
    CrossDocEdgeConfig,
    CrossDocEdgeInferer,
)
from ctrldoc.ops.workspace import (
    DocResolver,
    WorkspaceManager,
)
from ctrldoc.store.sqlite import SQLiteStore

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _StubResolver:
    """In-memory `DocResolver` — keys claim + concept lists by doc_id."""

    def __init__(
        self,
        *,
        claims: dict[str, list[Claim]] | None = None,
        concepts: dict[str, list[Concept]] | None = None,
    ) -> None:
        self._claims = claims or {}
        self._concepts = concepts or {}
        self.claims_calls: list[str] = []
        self.concepts_calls: list[str] = []

    def claims_for_doc(self, doc_id: str) -> Iterable[Claim]:
        self.claims_calls.append(doc_id)
        return list(self._claims.get(doc_id, []))

    def concepts_for_doc(self, doc_id: str) -> Iterable[Concept]:
        self.concepts_calls.append(doc_id)
        return list(self._concepts.get(doc_id, []))


class _DictScorer:
    """A deterministic `NLIScorer` keyed on `(premise, hypothesis)`."""

    def __init__(self, table: dict[tuple[str, str], NLIScore]) -> None:
        self._table = table
        self.calls: list[tuple[str, str]] = []

    def score(self, *, premise: str, hypothesis: str) -> NLIScore:
        self.calls.append((premise, hypothesis))
        return self._table.get(
            (premise, hypothesis),
            NLIScore(entailment=0.20, contradiction=0.20, neutral=0.60),
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _store(tmp_path: Path) -> SQLiteStore:
    return SQLiteStore(tmp_path / "workspaces.db")


def _claim(*, claim_id: str, doc_id: str, text: str) -> Claim:
    return Claim(
        id=claim_id,
        doc_id=doc_id,
        text=text,
        subject=None,
        predicate=text,
        object=None,
        polarity="+",
        modality=None,
        qualifier={},
        span_refs=[Span(chunk_id=f"{doc_id}:chunk-0", char_start=0, char_end=len(text), text=text)],
        section_id=f"{doc_id}:sec-0",
        concept_ids=[],
        typed_slots={},
        confidence=1.0,
    )


def _concept(*, id: str, name: str, doc_ids: list[str]) -> Concept:
    return Concept(
        id=id,
        canonical_name=name,
        aliases=[],
        primitive_type="Entity",
        mention_claim_ids=[],
        doc_ids=doc_ids,
    )


# ---------------------------------------------------------------------------
# DocResolver protocol exists with the documented surface
# ---------------------------------------------------------------------------


@pytest.mark.family_referential_integrity
def test_doc_resolver_protocol_admits_stub() -> None:
    """`_StubResolver` satisfies the `DocResolver` Protocol structurally."""
    stub = _StubResolver()
    assert isinstance(stub, DocResolver)


# ---------------------------------------------------------------------------
# Concept bridging — concepts from per-doc resolver land in workspace store
# ---------------------------------------------------------------------------


@pytest.mark.family_referential_integrity
def test_add_bridges_concepts_into_workspace_store(tmp_path: Path) -> None:
    """After `add`, the new doc's concepts surface via `workspace info`."""
    resolver = _StubResolver(
        concepts={
            "doc-a": [_concept(id="concept-1", name="OAuth", doc_ids=["doc-a"])],
        }
    )
    with _store(tmp_path) as store:
        manager = WorkspaceManager(store=store, doc_resolver=resolver)
        manager.create("audit-2026")
        manager.add("audit-2026", "doc-a")

        info = manager.info("audit-2026")
    assert info.doc_count == 1
    assert info.concept_count >= 1
    assert "concept-1" in info.shared_concept_ids


@pytest.mark.family_referential_integrity
def test_add_two_docs_yields_non_zero_shared_concept_rollup(tmp_path: Path) -> None:
    """S-156 gate: `workspace info` after a 2-doc add reports a
    non-zero shared-concept rollup. Each doc contributes its own
    concept; both surface in the workspace lattice slice.
    """
    resolver = _StubResolver(
        concepts={
            "doc-a": [_concept(id="concept-a1", name="OAuth", doc_ids=["doc-a"])],
            "doc-b": [_concept(id="concept-b1", name="JWT", doc_ids=["doc-b"])],
        }
    )
    with _store(tmp_path) as store:
        manager = WorkspaceManager(store=store, doc_resolver=resolver)
        manager.create("audit-2026")
        manager.add("audit-2026", "doc-a")
        manager.add("audit-2026", "doc-b")

        info = manager.info("audit-2026")
    assert info.doc_count == 2
    assert info.concept_count >= 2
    assert "concept-a1" in info.shared_concept_ids
    assert "concept-b1" in info.shared_concept_ids


@pytest.mark.family_referential_integrity
def test_add_without_resolver_skips_bridging(tmp_path: Path) -> None:
    """The manager works without a resolver — concept bridge is opt-in.

    Existing call sites that pre-date S-156 keep the same surface;
    no resolver means no concept bridge, no cross-doc edges, no
    behaviour change.
    """
    with _store(tmp_path) as store:
        manager = WorkspaceManager(store=store)
        manager.create("audit-2026")
        manager.add("audit-2026", "doc-a")
        info = manager.info("audit-2026")
    # No resolver wired → no concepts bridged → rollup stays empty.
    assert info.concept_count == 0
    assert info.shared_concept_ids == []


@pytest.mark.family_determinism
def test_concept_bridge_is_idempotent_on_repeated_add(tmp_path: Path) -> None:
    """Re-adding the same doc does not duplicate concept rows."""
    resolver = _StubResolver(
        concepts={"doc-a": [_concept(id="concept-1", name="OAuth", doc_ids=["doc-a"])]}
    )
    with _store(tmp_path) as store:
        manager = WorkspaceManager(store=store, doc_resolver=resolver)
        manager.create("audit-2026")
        manager.add("audit-2026", "doc-a")
        manager.add("audit-2026", "doc-a")  # idempotent no-op on docs
        info = manager.info("audit-2026")
    # The concept set never duplicates because `add_concepts` is
    # idempotent on id.
    assert info.shared_concept_ids == ["concept-1"]


# ---------------------------------------------------------------------------
# Cross-doc edge inference + persistence
# ---------------------------------------------------------------------------


@pytest.mark.family_verifier_calibration
def test_cross_doc_inference_triggered_when_existing_member_present(
    tmp_path: Path,
) -> None:
    """Adding the second doc triggers `CrossDocEdgeInferer` and persists
    every emitted edge into the workspace store's `cross_doc_edges` table.
    """
    a = _claim(claim_id="cA1", doc_id="doc-a", text="the system uses TLS 1.3")
    b = _claim(claim_id="cB1", doc_id="doc-b", text="the system uses TLS")
    resolver = _StubResolver(
        claims={"doc-a": [a], "doc-b": [b]},
        concepts={"doc-a": [], "doc-b": []},
    )
    scorer = _DictScorer(
        {
            (a.text, b.text): NLIScore(entailment=0.92, contradiction=0.02, neutral=0.06),
            (b.text, a.text): NLIScore(entailment=0.91, contradiction=0.02, neutral=0.07),
        }
    )
    inferer = CrossDocEdgeInferer(scorer=scorer)

    with _store(tmp_path) as store:
        manager = WorkspaceManager(store=store, doc_resolver=resolver, cross_doc_inferer=inferer)
        ws = manager.create("audit-2026")
        manager.add("audit-2026", "doc-a")  # first add: nothing to bridge to
        assert scorer.calls == []
        manager.add("audit-2026", "doc-b")  # second add: triggers inference

        edges = list(store.iter_cross_doc_edges_for_workspace(ws.id))
    # Both directions cross the entail threshold so we expect at least
    # one persisted `entails_across` edge.
    types = [e.type for e in edges]
    assert "entails_across" in types
    # Endpoint identity uses the persisted claim ids verbatim.
    entail = next(e for e in edges if e.type == "entails_across")
    assert {entail.src_id, entail.dst_id} == {"cA1", "cB1"}


@pytest.mark.family_performance_cost
def test_cross_doc_inference_skipped_on_first_member(tmp_path: Path) -> None:
    """Adding the first doc never invokes the scorer — no pair exists."""
    a = _claim(claim_id="cA1", doc_id="doc-a", text="alpha")
    resolver = _StubResolver(claims={"doc-a": [a]})
    scorer = _DictScorer({})
    inferer = CrossDocEdgeInferer(scorer=scorer)

    with _store(tmp_path) as store:
        manager = WorkspaceManager(store=store, doc_resolver=resolver, cross_doc_inferer=inferer)
        ws = manager.create("audit-2026")
        manager.add("audit-2026", "doc-a")
        edges = list(store.iter_cross_doc_edges_for_workspace(ws.id))
    assert edges == []
    assert scorer.calls == []


@pytest.mark.family_performance_cost
def test_cross_doc_inference_only_runs_when_inferer_is_wired(tmp_path: Path) -> None:
    """Without `cross_doc_inferer`, edges are never persisted even with a resolver."""
    a = _claim(claim_id="cA1", doc_id="doc-a", text="alpha")
    b = _claim(claim_id="cB1", doc_id="doc-b", text="beta")
    resolver = _StubResolver(claims={"doc-a": [a], "doc-b": [b]})

    with _store(tmp_path) as store:
        manager = WorkspaceManager(store=store, doc_resolver=resolver)
        ws = manager.create("audit-2026")
        manager.add("audit-2026", "doc-a")
        manager.add("audit-2026", "doc-b")
        edges = list(store.iter_cross_doc_edges_for_workspace(ws.id))
    assert edges == []


@pytest.mark.family_performance_cost
def test_cross_doc_inference_budget_stays_linear(tmp_path: Path) -> None:
    """Per-pair scorer budget is `k * |new_doc_claims|`, so total stays linear in
    workspace size after each new doc joins.
    """
    a_claims = [_claim(claim_id=f"cA{i}", doc_id="doc-a", text=f"alpha {i}") for i in range(3)]
    b_claims = [_claim(claim_id=f"cB{i}", doc_id="doc-b", text=f"beta {i}") for i in range(3)]
    c_claims = [_claim(claim_id=f"cC{i}", doc_id="doc-c", text=f"gamma {i}") for i in range(3)]
    resolver = _StubResolver(claims={"doc-a": a_claims, "doc-b": b_claims, "doc-c": c_claims})
    scorer = _DictScorer({})
    k = 2
    inferer = CrossDocEdgeInferer(scorer=scorer, config=CrossDocEdgeConfig(k_candidates=k))

    with _store(tmp_path) as store:
        manager = WorkspaceManager(store=store, doc_resolver=resolver, cross_doc_inferer=inferer)
        manager.create("ws-big")
        manager.add("ws-big", "doc-a")
        manager.add("ws-big", "doc-b")
        calls_after_two = len(scorer.calls)
        manager.add("ws-big", "doc-c")
        calls_after_three = len(scorer.calls)
    # Cost contract: `CrossDocEdgeInferer.infer` over N docs of M claims each
    # issues ≤ k * M * N * (N - 1) scorer calls — one ordered-pair walk over
    # every (src_doc, dst_doc) with k targets per source claim. The
    # quadratic baseline would be `(N * M) * (N * M - 1)`.
    # After adding doc-b → N=2, M=3 → ≤ 12 calls; quadratic = 30.
    assert calls_after_two <= 2 * 1 * 3 * k
    assert calls_after_two < 6 * 5
    # After adding doc-c → infer runs over {a, b, c}: N=3, M=3 → ≤ 36 calls;
    # quadratic = 9 * 8 = 72.
    delta = calls_after_three - calls_after_two
    assert delta <= 3 * 2 * 3 * k
    assert delta < 9 * 8


@pytest.mark.family_determinism
def test_cross_doc_edges_persist_across_store_reopen(tmp_path: Path) -> None:
    """Persisted cross-doc edges survive process restart — the table is durable."""
    a = _claim(claim_id="cA1", doc_id="doc-a", text="the proxy forwards requests")
    b = _claim(claim_id="cB1", doc_id="doc-b", text="the proxy forwards requests")
    resolver = _StubResolver(claims={"doc-a": [a], "doc-b": [b]})
    scorer = _DictScorer(
        {
            (a.text, b.text): NLIScore(entailment=0.95, contradiction=0.02, neutral=0.03),
            (b.text, a.text): NLIScore(entailment=0.95, contradiction=0.02, neutral=0.03),
        }
    )
    inferer = CrossDocEdgeInferer(scorer=scorer)
    db_path = tmp_path / "workspaces.db"

    with SQLiteStore(db_path) as store:
        manager = WorkspaceManager(store=store, doc_resolver=resolver, cross_doc_inferer=inferer)
        ws = manager.create("ws-durable")
        manager.add("ws-durable", "doc-a")
        manager.add("ws-durable", "doc-b")

    with SQLiteStore(db_path) as reopened:
        edges = list(reopened.iter_cross_doc_edges_for_workspace(ws.id))
    assert edges, "cross-doc edges must survive reopen"
    # Every persisted edge must round-trip through the SQLite layer cleanly.
    for edge in edges:
        assert edge.source == "nli"
        assert edge.confidence > 0.0
        assert edge.citations, "every edge must carry at least one citation"


@pytest.mark.family_determinism
def test_cross_doc_edges_idempotent_on_repeated_add(tmp_path: Path) -> None:
    """Re-adding the second doc never duplicates persisted edges.

    `add` is documented as idempotent; the cross-doc edge layer must
    inherit that contract so a replayed `workspace add` produces zero
    drift.
    """
    a = _claim(claim_id="cA1", doc_id="doc-a", text="the system uses TLS 1.3")
    b = _claim(claim_id="cB1", doc_id="doc-b", text="the system uses TLS")
    resolver = _StubResolver(claims={"doc-a": [a], "doc-b": [b]})
    scorer = _DictScorer(
        {
            (a.text, b.text): NLIScore(entailment=0.92, contradiction=0.02, neutral=0.06),
            (b.text, a.text): NLIScore(entailment=0.91, contradiction=0.02, neutral=0.07),
        }
    )
    inferer = CrossDocEdgeInferer(scorer=scorer)
    with _store(tmp_path) as store:
        manager = WorkspaceManager(store=store, doc_resolver=resolver, cross_doc_inferer=inferer)
        ws = manager.create("ws-idem")
        manager.add("ws-idem", "doc-a")
        manager.add("ws-idem", "doc-b")
        first = list(store.iter_cross_doc_edges_for_workspace(ws.id))
        # Re-add — no-op for doc list; cross-doc inferer should also re-run
        # but `INSERT OR REPLACE` keeps the row count fixed.
        manager.add("ws-idem", "doc-b")
        second = list(store.iter_cross_doc_edges_for_workspace(ws.id))
    assert len(first) == len(second)
    assert [(e.src_id, e.dst_id, e.type) for e in first] == [
        (e.src_id, e.dst_id, e.type) for e in second
    ]


# ---------------------------------------------------------------------------
# Resolver contract — claims are loaded for the new doc + every existing member
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Store-level parity — `InMemoryStore` mirrors the SQLite behaviour
# ---------------------------------------------------------------------------


@pytest.mark.family_referential_integrity
def test_in_memory_store_cross_doc_edge_round_trips() -> None:
    """`InMemoryStore` round-trips a cross-doc edge — backend parity contract."""
    from ctrldoc.models_v1 import TypedEdge
    from ctrldoc.store.memory import InMemoryStore

    store = InMemoryStore()
    edge = TypedEdge(
        src_id="cA1",
        dst_id="cB1",
        type="entails_across",
        confidence=0.92,
        raw_score=0.92,
        citations=[
            Span(chunk_id="doc-a:chunk-0", char_start=0, char_end=5, text="alpha"),
            Span(chunk_id="doc-b:chunk-0", char_start=0, char_end=4, text="beta"),
        ],
        source="nli",
        paraphrase_votes=None,
    )
    store.append_cross_doc_edge(workspace_id="ws-test", edge=edge)
    out = list(store.iter_cross_doc_edges_for_workspace("ws-test"))
    assert len(out) == 1
    assert out[0].src_id == "cA1"
    assert out[0].dst_id == "cB1"
    assert out[0].type == "entails_across"
    # Different workspace id → no edges surface (PK includes workspace_id).
    assert list(store.iter_cross_doc_edges_for_workspace("ws-other")) == []


@pytest.mark.family_referential_integrity
def test_resolver_is_called_for_every_existing_member_on_second_add(
    tmp_path: Path,
) -> None:
    """When the second doc joins, the resolver fetches claims for the
    incoming doc AND for every existing member so the inferer can build
    the full `claims_by_doc` mapping.
    """
    a = _claim(claim_id="cA1", doc_id="doc-a", text="alpha")
    b = _claim(claim_id="cB1", doc_id="doc-b", text="beta")
    resolver = _StubResolver(claims={"doc-a": [a], "doc-b": [b]})
    scorer = _DictScorer({})
    inferer = CrossDocEdgeInferer(scorer=scorer)

    with _store(tmp_path) as store:
        manager = WorkspaceManager(store=store, doc_resolver=resolver, cross_doc_inferer=inferer)
        manager.create("ws-resolve")
        manager.add("ws-resolve", "doc-a")
        manager.add("ws-resolve", "doc-b")

    # claims_for_doc must have been called for both members.
    assert "doc-a" in resolver.claims_calls
    assert "doc-b" in resolver.claims_calls
