"""Concept-population wiring — `EntityResolver` over store mentions + claims.

The L1.5 substrate needs `Concept` rows in the store before a workspace
can surface a shared-concept-lattice slice (§6.7). The §6.8 recipe is
already implemented in `extract.entity_resolution.EntityResolver`; this
slice plugs it into the ingest substrate by adapting persisted
`Entity` (GLiNER + claim-augmented mentions) and `Claim` (universal-
tuple subjects / objects) rows into `ConceptMention` inputs, running
the resolver, and persisting the resulting `Concept` cluster nodes via
`store.add_concepts`.

The deterministic / heuristic profile substitutes a `HeuristicERJudge`
keyed on the normalised canonical name — no LLM required — and the
`HashEmbedder` provides the blocking-time cosine ("falls back to hash
distance" per the slice's ROADMAP row). Production profiles plug in
the real LLM judge + Ollama embedder.

After population, `WorkspaceManager.concepts_for_workspace(name)` is
non-empty whenever any member document contributed at least one
mention to the underlying store — the per-doc concept rows carry the
document id, the workspace's `doc_ids` filters them in.

SPEC-REF: §6.8 (entity resolution / canonicalization)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ctrldoc.extract.concept_population import (
    HeuristicERJudge,
    mentions_from_store,
    populate_concepts_for_doc,
)
from ctrldoc.extract.entity_resolution import (
    ConceptMention,
    EntityResolutionConfig,
)
from ctrldoc.ingest.embedder import HashEmbedder
from ctrldoc.models import Entity
from ctrldoc.models_v1 import Claim
from ctrldoc.ops.workspace import WorkspaceManager
from ctrldoc.store.memory import InMemoryStore
from ctrldoc.store.sqlite import SQLiteStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _claim(
    cid: str,
    *,
    doc_id: str,
    subject: str | None,
    obj: str | None,
    predicate: str = "is",
) -> Claim:
    return Claim(
        id=cid,
        doc_id=doc_id,
        text=f"{subject or ''} {predicate} {obj or ''}".strip(),
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


def _entity(eid: str, *, type_: str, aliases: list[str], chunk_id: str) -> Entity:
    return Entity(
        id=eid,
        aliases=aliases,
        type=type_,
        mention_chunk_ids=[chunk_id],
    )


# ---------------------------------------------------------------------------
# HeuristicERJudge — canonical-name equivalence
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_heuristic_judge_equivalent_on_identical_canonical_name() -> None:
    judge = HeuristicERJudge()
    left = ConceptMention(
        id="m-1",
        mention_text="Bishop",
        primitive_type="Entity",
        doc_id="doc-1",
        claim_id="c-1",
    )
    right = ConceptMention(
        id="m-2",
        mention_text="bishop",
        primitive_type="Entity",
        doc_id="doc-2",
        claim_id="c-2",
    )
    assert judge.judge(left=left, right=right) == "equivalent"


@pytest.mark.family_determinism
def test_heuristic_judge_equivalent_with_whitespace_normalisation() -> None:
    judge = HeuristicERJudge()
    left = ConceptMention(
        id="m-1",
        mention_text="  convolution  network ",
        primitive_type="Entity",
        doc_id="doc-1",
        claim_id="c-1",
    )
    right = ConceptMention(
        id="m-2",
        mention_text="Convolution Network",
        primitive_type="Entity",
        doc_id="doc-2",
        claim_id="c-2",
    )
    assert judge.judge(left=left, right=right) == "equivalent"


@pytest.mark.family_determinism
def test_heuristic_judge_incomparable_on_distinct_text() -> None:
    judge = HeuristicERJudge()
    left = ConceptMention(
        id="m-1",
        mention_text="bishop",
        primitive_type="Entity",
        doc_id="doc-1",
        claim_id="c-1",
    )
    right = ConceptMention(
        id="m-2",
        mention_text="rook",
        primitive_type="Entity",
        doc_id="doc-1",
        claim_id="c-1",
    )
    assert judge.judge(left=left, right=right) == "incomparable"


# ---------------------------------------------------------------------------
# mentions_from_store — adapts Entity + Claim rows into ConceptMention
# ---------------------------------------------------------------------------


@pytest.mark.family_referential_integrity
def test_mentions_from_store_pulls_from_entities_and_claims() -> None:
    store = InMemoryStore()
    store.add_entities(
        [
            _entity("ent/person/bishop", type_="person", aliases=["Bishop"], chunk_id="chk-1"),
        ]
    )
    store.append_claim(_claim("clm-1", doc_id="doc-1", subject="Bishop", obj="neural network"))

    mentions = mentions_from_store(store, doc_id="doc-1")

    surfaces = sorted(m.mention_text for m in mentions)
    assert "Bishop" in surfaces
    assert "neural network" in surfaces
    assert all(m.doc_id == "doc-1" for m in mentions)
    assert all(m.id and m.claim_id for m in mentions)


@pytest.mark.family_referential_integrity
def test_mentions_from_store_skips_blank_subject_object() -> None:
    store = InMemoryStore()
    store.append_claim(_claim("clm-1", doc_id="doc-1", subject="", obj="   "))
    store.append_claim(_claim("clm-2", doc_id="doc-1", subject="Bishop", obj=None))

    mentions = mentions_from_store(store, doc_id="doc-1")

    surfaces = [m.mention_text for m in mentions]
    assert surfaces == ["Bishop"]


@pytest.mark.family_referential_integrity
def test_mentions_from_store_filters_other_docs() -> None:
    store = InMemoryStore()
    store.append_claim(_claim("clm-1", doc_id="doc-1", subject="A", obj="B"))
    store.append_claim(_claim("clm-2", doc_id="doc-2", subject="C", obj="D"))

    mentions = mentions_from_store(store, doc_id="doc-1")
    surfaces = sorted(m.mention_text for m in mentions)
    assert surfaces == ["A", "B"]


@pytest.mark.family_determinism
def test_mentions_from_store_deterministic_id_assignment() -> None:
    store = InMemoryStore()
    store.add_entities(
        [
            _entity("ent/person/bishop", type_="person", aliases=["Bishop"], chunk_id="chk-1"),
        ]
    )
    store.append_claim(_claim("clm-1", doc_id="doc-1", subject="Bishop", obj="x"))

    first = mentions_from_store(store, doc_id="doc-1")
    second = mentions_from_store(store, doc_id="doc-1")
    assert [m.id for m in first] == [m.id for m in second]


# ---------------------------------------------------------------------------
# populate_concepts_for_doc — end-to-end with InMemoryStore + HashEmbedder
# ---------------------------------------------------------------------------


@pytest.mark.family_referential_integrity
def test_populate_concepts_persists_concept_rows() -> None:
    store = InMemoryStore()
    store.add_entities(
        [_entity("ent/person/bishop", type_="person", aliases=["Bishop"], chunk_id="chk-1")]
    )
    store.append_claim(_claim("clm-1", doc_id="doc-1", subject="Bishop", obj="network"))

    outcome = populate_concepts_for_doc(
        store=store,
        doc_id="doc-1",
        embedder=HashEmbedder(dimension=32),
    )

    persisted = list(store.iter_concepts())
    assert len(persisted) == len(outcome.concepts)
    assert len(outcome.concepts) >= 2
    assert all(c.doc_ids == ["doc-1"] for c in persisted)
    assert all(c.mention_claim_ids for c in persisted)


@pytest.mark.family_referential_integrity
def test_populate_concepts_merges_identical_text_across_sources() -> None:
    store = InMemoryStore()
    # Same canonical surface form contributed by both NER and a claim.
    store.add_entities(
        [_entity("ent/person/bishop", type_="person", aliases=["Bishop"], chunk_id="chk-1")]
    )
    store.append_claim(_claim("clm-1", doc_id="doc-1", subject="Bishop", obj="network"))

    outcome = populate_concepts_for_doc(
        store=store,
        doc_id="doc-1",
        embedder=HashEmbedder(dimension=32),
    )

    bishop_clusters = [c for c in store.iter_concepts() if c.canonical_name.lower() == "bishop"]
    assert len(bishop_clusters) == 1, "identical-text mentions must collapse into one cluster"
    # Both the NER entity row and the claim row contributed.
    assert "clm-1" in bishop_clusters[0].mention_claim_ids
    assert len(outcome.concepts) >= 1


@pytest.mark.family_determinism
def test_populate_concepts_empty_store_short_circuits() -> None:
    store = InMemoryStore()
    outcome = populate_concepts_for_doc(
        store=store,
        doc_id="doc-1",
        embedder=HashEmbedder(dimension=32),
    )
    assert len(outcome.concepts) == 0
    assert list(store.iter_concepts()) == []


@pytest.mark.family_determinism
def test_populate_concepts_idempotent_across_reruns() -> None:
    store = InMemoryStore()
    store.append_claim(_claim("clm-1", doc_id="doc-1", subject="Bishop", obj="net"))
    embedder = HashEmbedder(dimension=32)

    first = populate_concepts_for_doc(store=store, doc_id="doc-1", embedder=embedder)
    ids_after_first = sorted(c.id for c in store.iter_concepts())
    second = populate_concepts_for_doc(store=store, doc_id="doc-1", embedder=embedder)
    ids_after_second = sorted(c.id for c in store.iter_concepts())

    assert len(first.concepts) == len(second.concepts)
    assert ids_after_first == ids_after_second


@pytest.mark.family_referential_integrity
def test_populate_concepts_default_embedder_when_none_supplied() -> None:
    store = InMemoryStore()
    store.append_claim(_claim("clm-1", doc_id="doc-1", subject="Bishop", obj="net"))

    outcome = populate_concepts_for_doc(store=store, doc_id="doc-1")
    assert len(outcome.concepts) >= 1


@pytest.mark.family_referential_integrity
def test_populate_concepts_respects_tau_block_config() -> None:
    store = InMemoryStore()
    store.append_claim(_claim("clm-1", doc_id="doc-1", subject="Bishop", obj="net"))
    # A tau above 1.0 would reject every pair; here we just verify the
    # config plumbs through without crashing and yields well-formed
    # singleton concepts.
    outcome = populate_concepts_for_doc(
        store=store,
        doc_id="doc-1",
        embedder=HashEmbedder(dimension=32),
        config=EntityResolutionConfig(tau_block=0.99),
    )
    assert len(outcome.concepts) >= 1


# ---------------------------------------------------------------------------
# InMemoryStore concept CRUD — parity with SQLiteStore
# ---------------------------------------------------------------------------


@pytest.mark.family_referential_integrity
def test_inmemory_store_add_and_iter_concepts() -> None:
    from ctrldoc.models_v1 import Concept

    store = InMemoryStore()
    concept = Concept(
        id="concept-x",
        canonical_name="Bishop",
        aliases=[],
        primitive_type="Entity",
        mention_claim_ids=["clm-1"],
        doc_ids=["doc-1"],
    )
    store.add_concepts([concept])
    assert store.get_concept("concept-x") == concept
    assert list(store.iter_concepts()) == [concept]


@pytest.mark.family_referential_integrity
def test_inmemory_store_concepts_for_workspace_intersection() -> None:
    from ctrldoc.models_v1 import Concept

    store = InMemoryStore()
    store.add_concepts(
        [
            Concept(
                id="concept-a",
                canonical_name="A",
                aliases=[],
                primitive_type="Entity",
                mention_claim_ids=["c1"],
                doc_ids=["doc-1"],
            ),
            Concept(
                id="concept-b",
                canonical_name="B",
                aliases=[],
                primitive_type="Entity",
                mention_claim_ids=["c2"],
                doc_ids=["doc-2"],
            ),
        ]
    )
    visible = list(store.concepts_for_workspace_docs(["doc-1"]))
    assert [c.id for c in visible] == ["concept-a"]


# ---------------------------------------------------------------------------
# WorkspaceManager — end-to-end through SQLiteStore
# ---------------------------------------------------------------------------


@pytest.mark.family_referential_integrity
def test_workspace_manager_concepts_for_workspace_after_add(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "ctrldoc.db")
    # Simulate ingest: claims persisted then concepts populated for doc-1.
    store.append_claim(_claim("clm-1", doc_id="doc-1", subject="Bishop", obj="network"))
    populate_concepts_for_doc(
        store=store,
        doc_id="doc-1",
        embedder=HashEmbedder(dimension=32),
    )

    manager = WorkspaceManager(store=store)
    manager.create("ws-bishop")
    manager.add("ws-bishop", "doc-1")

    visible = manager.concepts_for_workspace("ws-bishop")
    assert visible, "workspace must surface a non-empty concept-lattice slice"
    assert all(c.id.startswith("concept-") for c in visible)


@pytest.mark.family_referential_integrity
def test_workspace_manager_concepts_for_workspace_empty_without_member_docs(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(tmp_path / "ctrldoc.db")
    store.append_claim(_claim("clm-1", doc_id="doc-1", subject="Bishop", obj="x"))
    populate_concepts_for_doc(
        store=store,
        doc_id="doc-1",
        embedder=HashEmbedder(dimension=32),
    )

    manager = WorkspaceManager(store=store)
    manager.create("empty-ws")
    # No doc attached — workspace must not surface unrelated concepts.
    assert manager.concepts_for_workspace("empty-ws") == []
