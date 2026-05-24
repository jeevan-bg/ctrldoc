"""Entity resolution (canonicalization) — blocking + LLM-judge ER.

The L1.5 substrate uses the standard ER recipe to merge cross-document
mentions of the same concept into one canonical cluster (§6.8):

1. **Blocking** on embedding cosine ≥ `tau_block` (default 0.85) emits
   candidate pairs cheaply, restricted to mentions of the same
   `primitive_type` (Entity vs Event are never the same concept).
2. **LLM judge** on candidate pairs only, with a four-class verdict
   `equivalent` / `subsumes` / `subsumed_by` / `incomparable`. JSON
   output; the resolver never sees the doc body, only the mention
   surfaces.
3. **Union-find** over `equivalent` verdicts → canonical cluster ids;
   one `Concept` per cluster.
4. **Subsumption edges** map to the `is_a` slot of the typed-edge
   alphabet — `subsumes(A, B)` means `B is_a A`, `subsumed_by(A, B)`
   means `A is_a B`. Edges are deduplicated and sorted for replay.

Release gates mirror §6.8: precision ≥ 0.90 (no spurious merges) and
recall ≥ 0.85 (most true co-references recovered) on the inline gold
fixture.

SPEC-REF: §6.8 (entity resolution / canonicalization)
"""

from __future__ import annotations

import pytest

from ctrldoc.extract.entity_resolution import (
    DEFAULT_TAU_BLOCK,
    ER_PRECISION_THRESHOLD,
    ER_RECALL_THRESHOLD,
    ConceptMention,
    EntityResolution,
    EntityResolutionConfig,
    EntityResolutionVerdict,
    EntityResolver,
    cluster_precision_recall,
)
from ctrldoc.ingest.embedder import HashEmbedder

# ---------------------------------------------------------------------------
# Stub judge — deterministic, records every call
# ---------------------------------------------------------------------------


class _DictJudge:
    """An `ERJudge` keyed on the unordered `frozenset({left_id, right_id})`.

    Defaults to `incomparable` for unscripted pairs so candidate-budget
    tests can drive many mentions without enumerating every pair.
    """

    def __init__(self, table: dict[frozenset[str], EntityResolutionVerdict]) -> None:
        self._table = table
        self.calls: list[tuple[str, str]] = []

    def judge(
        self,
        *,
        left: ConceptMention,
        right: ConceptMention,
    ) -> EntityResolutionVerdict:
        self.calls.append((left.id, right.id))
        return self._table.get(frozenset({left.id, right.id}), "incomparable")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mention(
    mid: str,
    text: str,
    *,
    primitive_type: str = "Entity",
    doc_id: str = "doc-1",
    claim_id: str = "claim-1",
) -> ConceptMention:
    return ConceptMention(
        id=mid,
        mention_text=text,
        primitive_type=primitive_type,  # type: ignore[arg-type]
        doc_id=doc_id,
        claim_id=claim_id,
    )


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_thresholds_are_documented_release_gates() -> None:
    assert pytest.approx(0.90) == ER_PRECISION_THRESHOLD
    assert pytest.approx(0.85) == ER_RECALL_THRESHOLD
    assert pytest.approx(0.85) == DEFAULT_TAU_BLOCK


@pytest.mark.family_determinism
def test_empty_input_short_circuits_to_empty_result() -> None:
    resolver = EntityResolver(
        embedder=HashEmbedder(dimension=64),
        judge=_DictJudge({}),
    )
    out = resolver.resolve([])
    assert isinstance(out, EntityResolution)
    assert out.concepts == []
    assert out.subsumption_edges == []
    assert out.judge_calls == 0


@pytest.mark.family_determinism
def test_single_mention_becomes_a_singleton_concept_with_no_judge_calls() -> None:
    judge = _DictJudge({})
    resolver = EntityResolver(embedder=HashEmbedder(dimension=64), judge=judge)
    out = resolver.resolve([_mention("m1", "Application server")])
    assert len(out.concepts) == 1
    assert out.concepts[0].canonical_name == "Application server"
    assert out.concepts[0].aliases == []
    assert out.concepts[0].mention_claim_ids == ["claim-1"]
    assert out.concepts[0].doc_ids == ["doc-1"]
    assert out.judge_calls == 0
    assert judge.calls == []


@pytest.mark.family_determinism
def test_different_primitive_types_are_never_blocked_together() -> None:
    """Blocking only considers same-primitive pairs — Entity vs Event never pair."""
    mentions = [
        _mention("m1", "Election", primitive_type="Entity"),
        _mention("m2", "Election", primitive_type="Event"),
    ]
    judge = _DictJudge({})
    resolver = EntityResolver(embedder=HashEmbedder(dimension=64), judge=judge)
    out = resolver.resolve(mentions)
    assert judge.calls == []
    assert {c.primitive_type for c in out.concepts} == {"Entity", "Event"}
    assert len(out.concepts) == 2


@pytest.mark.family_determinism
def test_low_cosine_pairs_are_filtered_below_tau_block() -> None:
    """The judge is never asked about pairs whose embeddings disagree."""
    mentions = [
        _mention("m1", "The quick brown fox jumps over the lazy dog"),
        _mention("m2", "Quantum chromodynamics in lattice gauge theory"),
    ]
    judge = _DictJudge({})
    resolver = EntityResolver(
        embedder=HashEmbedder(dimension=128),
        judge=judge,
        config=EntityResolutionConfig(tau_block=0.95),
    )
    out = resolver.resolve(mentions)
    assert judge.calls == []
    # Two distinct singletons survive.
    assert len(out.concepts) == 2


@pytest.mark.family_verifier_calibration
def test_equivalent_verdict_unions_mentions_into_one_canonical_concept() -> None:
    mentions = [
        _mention("m1", "Application server", doc_id="doc-a", claim_id="claim-a1"),
        _mention("m2", "App server", doc_id="doc-b", claim_id="claim-b1"),
        _mention("m3", "Application server", doc_id="doc-c", claim_id="claim-c1"),
    ]
    judge = _DictJudge(
        {
            frozenset({"m1", "m2"}): "equivalent",
            frozenset({"m2", "m3"}): "equivalent",
            frozenset({"m1", "m3"}): "equivalent",
        }
    )
    # `tau_block=0.0` forces blocking to admit every same-primitive pair so the
    # judge is the sole arbiter of equivalence.
    resolver = EntityResolver(
        embedder=HashEmbedder(dimension=64),
        judge=judge,
        config=EntityResolutionConfig(tau_block=-1.0),
    )
    out = resolver.resolve(mentions)
    assert len(out.concepts) == 1
    concept = out.concepts[0]
    assert concept.canonical_name == "Application server"
    assert sorted(concept.aliases) == ["App server"]
    assert sorted(concept.mention_claim_ids) == ["claim-a1", "claim-b1", "claim-c1"]
    assert sorted(concept.doc_ids) == ["doc-a", "doc-b", "doc-c"]
    assert out.subsumption_edges == []


@pytest.mark.family_verifier_calibration
def test_subsumes_verdict_emits_is_a_edge_from_child_to_parent() -> None:
    """`subsumes(left, right)` means right is_a left; `subsumed_by` is the mirror."""
    mentions = [
        _mention("m1", "Database"),
        _mention("m2", "PostgreSQL database"),
        _mention("m3", "MySQL database"),
    ]
    # The resolver always invokes the judge with `left = mention_list[i]`
    # and `right = mention_list[j]` for `i < j`, so verdicts are written
    # relative to that orientation. `subsumes(left, right)` means
    # `right is_a left`; `subsumed_by(left, right)` means
    # `left is_a right`.
    judge = _DictJudge(
        {
            # left=m1 (Database) subsumes right=m2 (PostgreSQL) → PostgreSQL is_a Database.
            frozenset({"m1", "m2"}): "subsumes",
            # left=m1 (Database) subsumes right=m3 (MySQL) → MySQL is_a Database.
            frozenset({"m1", "m3"}): "subsumes",
            frozenset({"m2", "m3"}): "incomparable",
        }
    )
    resolver = EntityResolver(
        embedder=HashEmbedder(dimension=64),
        judge=judge,
        config=EntityResolutionConfig(tau_block=-1.0),
    )
    out = resolver.resolve(mentions)
    # Three singleton concepts (no equivalences).
    assert len(out.concepts) == 3
    edges_by_pair = {(e.src_id, e.dst_id) for e in out.subsumption_edges}
    name_by_concept = {c.id: c.canonical_name for c in out.concepts}
    # Both subsumption directions should yield exactly one is_a edge whose
    # endpoints map onto the parent-child name pair.
    pairs = {(name_by_concept[s], name_by_concept[d]) for s, d in edges_by_pair}
    assert ("PostgreSQL database", "Database") in pairs
    assert ("MySQL database", "Database") in pairs
    assert all(e.type == "is_a" for e in out.subsumption_edges)
    assert all(e.source == "llm" for e in out.subsumption_edges)


@pytest.mark.family_verifier_calibration
def test_subsumed_by_verdict_emits_is_a_edge_from_left_to_right() -> None:
    """`subsumed_by(left, right)` means left is_a right (mirror of `subsumes`)."""
    mentions = [
        # left=m1 (PostgreSQL) subsumed_by right=m2 (Database) → PostgreSQL is_a Database.
        _mention("m1", "PostgreSQL database"),
        _mention("m2", "Database"),
    ]
    judge = _DictJudge(
        {
            frozenset({"m1", "m2"}): "subsumed_by",
        }
    )
    resolver = EntityResolver(
        embedder=HashEmbedder(dimension=64),
        judge=judge,
        config=EntityResolutionConfig(tau_block=-1.0),
    )
    out = resolver.resolve(mentions)
    assert len(out.concepts) == 2
    assert len(out.subsumption_edges) == 1
    edge = out.subsumption_edges[0]
    by_id = {c.id: c.canonical_name for c in out.concepts}
    assert by_id[edge.src_id] == "PostgreSQL database"
    assert by_id[edge.dst_id] == "Database"


@pytest.mark.family_verifier_calibration
def test_subsumption_edge_endpoints_are_canonical_cluster_ids() -> None:
    """Subsumption edges live between canonical concepts, not mention ids."""
    mentions = [
        _mention("m1", "Database"),
        _mention("m2", "DB"),
        _mention("m3", "PostgreSQL database"),
    ]
    judge = _DictJudge(
        {
            # m1 and m2 are the same concept.
            frozenset({"m1", "m2"}): "equivalent",
            # m1 subsumes m3 → m3 is_a m1's concept.
            frozenset({"m1", "m3"}): "subsumes",
            # m2 (same concept as m1) also subsumes m3.
            frozenset({"m2", "m3"}): "subsumes",
        }
    )
    resolver = EntityResolver(
        embedder=HashEmbedder(dimension=64),
        judge=judge,
        config=EntityResolutionConfig(tau_block=-1.0),
    )
    out = resolver.resolve(mentions)
    # Two concepts after union-find: {m1, m2} and {m3}.
    assert len(out.concepts) == 2
    # Only one is_a edge — the two subsumption verdicts collapse onto the
    # same canonical pair.
    assert len(out.subsumption_edges) == 1
    edge = out.subsumption_edges[0]
    by_id = {c.id: c.canonical_name for c in out.concepts}
    assert by_id[edge.src_id] == "PostgreSQL database"
    assert by_id[edge.dst_id] in {"Database", "DB"}


@pytest.mark.family_determinism
def test_output_is_byte_identical_across_repeat_invocations() -> None:
    """Determinism: identical input mentions produce identical output bytes."""
    mentions = [
        _mention("m1", "TLS"),
        _mention("m2", "Transport Layer Security"),
        _mention("m3", "HTTPS"),
    ]
    judge_table: dict[frozenset[str], EntityResolutionVerdict] = {
        frozenset({"m1", "m2"}): "equivalent",
        frozenset({"m1", "m3"}): "incomparable",
        frozenset({"m2", "m3"}): "incomparable",
    }
    resolver_a = EntityResolver(
        embedder=HashEmbedder(dimension=64),
        judge=_DictJudge(judge_table),
        config=EntityResolutionConfig(tau_block=-1.0),
    )
    resolver_b = EntityResolver(
        embedder=HashEmbedder(dimension=64),
        judge=_DictJudge(judge_table),
        config=EntityResolutionConfig(tau_block=-1.0),
    )
    out_a = resolver_a.resolve(mentions)
    out_b = resolver_b.resolve(mentions)
    assert out_a.model_dump_json() == out_b.model_dump_json()


@pytest.mark.family_performance_cost
def test_judge_is_called_at_most_once_per_unordered_blocked_pair() -> None:
    """No pair is asked twice; cost stays linear-ish thanks to blocking."""
    mentions = [
        _mention("m1", "Token"),
        _mention("m2", "Token"),
        _mention("m3", "Token"),
        _mention("m4", "Token"),
    ]
    judge = _DictJudge({})
    resolver = EntityResolver(
        embedder=HashEmbedder(dimension=64),
        judge=judge,
        config=EntityResolutionConfig(tau_block=-1.0),
    )
    out = resolver.resolve(mentions)
    # 4 mentions → at most C(4,2) = 6 judge calls if all pairs survived
    # blocking; never more.
    assert out.judge_calls <= 6
    seen: set[frozenset[str]] = set()
    for left, right in judge.calls:
        key = frozenset({left, right})
        assert key not in seen, f"pair {key} judged twice"
        seen.add(key)


# ---------------------------------------------------------------------------
# Gold-fixture gate — precision ≥ 0.90, recall ≥ 0.85 per §6.8
# ---------------------------------------------------------------------------


@pytest.mark.family_verifier_calibration
def test_release_gate_precision_and_recall_on_gold_fixture() -> None:
    """A 12-mention gold fixture with a known partition validates the gates.

    The fixture mixes three true concept clusters plus four singletons; the
    judge table encodes the gold equivalences exactly so the resolver
    output should match the gold partition byte-for-byte and score
    precision = recall = 1.0, comfortably above the release gate.
    """
    mentions = [
        # Cluster 1 — Application server (3 mentions).
        _mention("m1", "Application server"),
        _mention("m2", "App server"),
        _mention("m3", "application server"),
        # Cluster 2 — Transport Layer Security (3 mentions).
        _mention("m4", "TLS"),
        _mention("m5", "Transport Layer Security"),
        _mention("m6", "TLS protocol"),
        # Cluster 3 — Postgres (2 mentions).
        _mention("m7", "Postgres"),
        _mention("m8", "PostgreSQL"),
        # Singletons — distinct concepts.
        _mention("m9", "Database"),
        _mention("m10", "Cache"),
        _mention("m11", "Queue"),
        _mention("m12", "Scheduler"),
    ]
    gold_clusters: list[set[str]] = [
        {"m1", "m2", "m3"},
        {"m4", "m5", "m6"},
        {"m7", "m8"},
        {"m9"},
        {"m10"},
        {"m11"},
        {"m12"},
    ]
    # Encode gold equivalences so the judge mirrors the gold partition.
    table: dict[frozenset[str], EntityResolutionVerdict] = {}
    for cluster in gold_clusters:
        members = sorted(cluster)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                table[frozenset({members[i], members[j]})] = "equivalent"

    resolver = EntityResolver(
        embedder=HashEmbedder(dimension=64),
        judge=_DictJudge(table),
        config=EntityResolutionConfig(tau_block=-1.0),
    )
    out = resolver.resolve(mentions)

    # Rebuild the predicted partition from the resolution output's
    # `clusters` bookkeeping shipped alongside the concept list. Each
    # entry is the set of mention ids that fall into one canonical
    # `Concept`.
    predicted_clusters = [set(cluster_ids) for cluster_ids in out.clusters]

    precision, recall = cluster_precision_recall(
        predicted=predicted_clusters,
        gold=gold_clusters,
    )
    assert precision >= ER_PRECISION_THRESHOLD, f"precision {precision} below gate"
    assert recall >= ER_RECALL_THRESHOLD, f"recall {recall} below gate"


@pytest.mark.family_verifier_calibration
def test_cluster_precision_recall_pairwise_definition() -> None:
    """Validate the pairwise P/R helper against a hand-worked example.

    Gold partition: {a,b,c}, {d,e}. Predicted: {a,b}, {c}, {d,e}.
    Gold pairs: {a,b}, {a,c}, {b,c}, {d,e} = 4.
    Predicted pairs: {a,b}, {d,e} = 2.
    True positive pairs: {a,b}, {d,e} = 2.
    Precision = 2/2 = 1.0; recall = 2/4 = 0.5.
    """
    gold = [{"a", "b", "c"}, {"d", "e"}]
    predicted = [{"a", "b"}, {"c"}, {"d", "e"}]
    precision, recall = cluster_precision_recall(predicted=predicted, gold=gold)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(0.5)
