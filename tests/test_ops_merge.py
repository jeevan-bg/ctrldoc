"""L5 merge operation via optimal transport + Galois join.

§6.6 frames `merge({A, B, C})` as:

> *Centroid distribution over the union; per-cluster strongest claim
> (Galois join) emitted; topological order via `depends_on` +
> `part_of`. Loss invariant: every input claim ID maps to exactly one
> output cluster.*

The hard `loss_invariant_satisfied` gate is the §6.6 release contract:
the merger's output partition must cover every input id exactly
once. This module ships `TransportMerger` — the `Merger`-protocol-
shaped consumer wiring the Galois floor (`claim_subsumption`) plus
NLI-driven equivalence detection into a union-find partition with
per-cluster Galois-join representatives — and a functional `merge`
surface for the §9 CLI.

The reduction:

* Build the upper-triangular pairwise judgement matrix over input
  claims. For each pair `(i, j)` (i < j):
    1. Galois floor first via `claim_subsumption(i, j)`. If
       `equivalent` / `subsumes` / `subsumed_by` → record as
       *mergeable* without any NLI call.
    2. Otherwise consult the NLI scorer in both directions. If both
       directions clear `equivalence_threshold` → the pair is a
       paraphrase-style semantic equivalence → record as mergeable.
* Union-find collapses every mergeable pair into one cluster.
  Singletons (no merge partner) emit as their own cluster — that's
  what guarantees the §6.6 loss invariant by construction.
* Per cluster, pick the *strongest* representative by reducing
  `claim_meet` across members; on any incomparable step, fall back
  to the first member (deterministic, input-order tiebreak). The
  meet is the greatest lower bound at the Galois floor — for
  modality-driven clusters it surfaces the most-binding claim (the
  `MUST` over the `SHOULD`).

The release gate is the hard `loss_invariant_satisfied` binary plus
the soft `MERGE_PARTITION_THRESHOLD = 0.85` pairwise accuracy gate
on the shipped 6-case fixture.

SPEC-REF: §6.6 (merge = partition + Galois join with loss invariant)
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ctrldoc.eval.calibration import NLIScore
from ctrldoc.eval.claim_extraction import ClaimTuple
from ctrldoc.eval.harness import load_jsonl_cases, run_eval
from ctrldoc.eval.merge import (
    MERGE_PARTITION_THRESHOLD,
    InputClaim,
    MergeEvalCase,
    MergeEvalRunner,
    Merger,
    loss_invariant_satisfied,
    pairwise_partition_accuracy,
)
from ctrldoc.ops.merge import (
    MergeConfig,
    MergeResult,
    TransportMerger,
    merge,
)

MERGE_EVAL_PATH = Path(__file__).parent / "eval" / "merge_eval.jsonl"


# ---------------------------------------------------------------------------
# Stub scorers — deterministic, recordable
# ---------------------------------------------------------------------------


class _DictScorer:
    """`NLIScorer` keyed on `(premise, hypothesis)` strings."""

    def __init__(
        self,
        table: dict[tuple[str, str], NLIScore],
        *,
        default: NLIScore | None = None,
    ) -> None:
        self._table = table
        self._default = default or NLIScore(entailment=0.10, contradiction=0.10, neutral=0.80)
        self.calls: list[tuple[str, str]] = []

    def score(self, *, premise: str, hypothesis: str) -> NLIScore:
        self.calls.append((premise, hypothesis))
        return self._table.get((premise, hypothesis), self._default)


def _high_entail() -> NLIScore:
    return NLIScore(entailment=0.92, contradiction=0.03, neutral=0.05)


def _low_entail() -> NLIScore:
    return NLIScore(entailment=0.15, contradiction=0.10, neutral=0.75)


def _claim(
    subject: str = "the system",
    predicate: str = "uses",
    object_: str = "consistent hashing",
    polarity: str = "affirmative",
    modality: str = "asserted",
    qualifier: str = "",
) -> ClaimTuple:
    return ClaimTuple(
        subject=subject,
        predicate=predicate,
        object=object_,
        polarity=polarity,  # type: ignore[arg-type]
        modality=modality,  # type: ignore[arg-type]
        qualifier=qualifier,
    )


def _input(
    id_: str,
    doc_id: str = "docA",
    claim: ClaimTuple | None = None,
) -> InputClaim:
    return InputClaim(id=id_, doc_id=doc_id, doc_type="spec", claim=claim or _claim())


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_config_default_equivalence_threshold_in_unit_interval() -> None:
    cfg = MergeConfig()
    assert 0.0 < cfg.equivalence_threshold < 1.0


@pytest.mark.family_determinism
def test_config_rejects_threshold_at_boundary_or_outside() -> None:
    with pytest.raises(ValueError):
        MergeConfig(equivalence_threshold=0.0)
    with pytest.raises(ValueError):
        MergeConfig(equivalence_threshold=1.0)
    with pytest.raises(ValueError):
        MergeConfig(equivalence_threshold=-0.1)


# ---------------------------------------------------------------------------
# MergeResult shape
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_result_is_frozen_and_strict() -> None:
    """`MergeResult` rejects extra fields and pinned attributes are read-only."""
    inputs = [_input("a1")]
    r = merge(input_claims=inputs, scorer=_DictScorer({}))
    with pytest.raises(ValidationError):
        r.scorer_calls = 5  # type: ignore[misc]
    with pytest.raises(ValidationError):
        MergeResult(output=r.output, scorer_calls=0, stray="oops")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Empty / singleton input — degenerate but well-defined
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_empty_input_yields_empty_output() -> None:
    """No input claims → no clusters, no scorer calls. Loss invariant
    is vacuously satisfied."""
    scorer = _DictScorer({})
    result = merge(input_claims=[], scorer=scorer)
    assert result.output.clusters == []
    assert result.scorer_calls == 0
    assert loss_invariant_satisfied(input_ids=[], output_clusters=result.output.clusters)


@pytest.mark.family_determinism
def test_single_claim_input_yields_singleton_cluster() -> None:
    """A single input claim → one singleton cluster. No NLI call needed."""
    scorer = _DictScorer({})
    ic = _input("a1")
    result = merge(input_claims=[ic], scorer=scorer)
    assert len(result.output.clusters) == 1
    assert result.output.clusters[0].member_claim_ids == ["a1"]
    assert result.output.clusters[0].strongest_claim == ic.claim
    assert result.scorer_calls == 0


# ---------------------------------------------------------------------------
# Loss invariant — every input id appears in exactly one cluster
# ---------------------------------------------------------------------------


@pytest.mark.family_referential_integrity
def test_loss_invariant_holds_for_all_singletons() -> None:
    """Disjoint claims → singleton clusters → invariant holds."""
    inputs = [
        _input("a1", claim=_claim(subject="x", predicate="p", object_="1")),
        _input("a2", claim=_claim(subject="x", predicate="p", object_="2")),
        _input("a3", claim=_claim(subject="x", predicate="p", object_="3")),
    ]
    scorer = _DictScorer({})
    result = merge(input_claims=inputs, scorer=scorer)
    assert loss_invariant_satisfied(
        input_ids=["a1", "a2", "a3"], output_clusters=result.output.clusters
    )


@pytest.mark.family_referential_integrity
def test_loss_invariant_holds_when_galois_merges_cluster() -> None:
    """Galois-equivalent claims collapse into one cluster — invariant holds."""
    same = _claim(subject="x", predicate="p", object_="y")
    inputs = [
        _input("a1", claim=same),
        _input("a2", claim=same),  # identical → Galois equivalent
        _input("a3", claim=_claim(subject="z", predicate="q", object_="w")),
    ]
    scorer = _DictScorer({})
    result = merge(input_claims=inputs, scorer=scorer)
    assert loss_invariant_satisfied(
        input_ids=["a1", "a2", "a3"], output_clusters=result.output.clusters
    )
    # a1 and a2 belong to the same cluster.
    member_lookup = {
        cid: i
        for i, cluster in enumerate(result.output.clusters)
        for cid in cluster.member_claim_ids
    }
    assert member_lookup["a1"] == member_lookup["a2"]
    assert member_lookup["a1"] != member_lookup["a3"]


@pytest.mark.family_referential_integrity
def test_loss_invariant_holds_when_nli_merges_cluster() -> None:
    """NLI bidirectional equivalence → cluster merge — invariant still holds."""
    a = _claim(subject="users", predicate="authenticate", object_="via oauth")
    b = _claim(subject="users", predicate="sign in", object_="with oauth")
    c = _claim(subject="logs", predicate="rotate", object_="daily")
    a_text = "users authenticate via oauth"
    b_text = "users sign in with oauth"
    scorer = _DictScorer(
        {
            (a_text, b_text): _high_entail(),
            (b_text, a_text): _high_entail(),
        }
    )
    inputs = [_input("a1", claim=a), _input("a2", claim=b), _input("a3", claim=c)]
    result = merge(input_claims=inputs, scorer=scorer)
    assert loss_invariant_satisfied(
        input_ids=["a1", "a2", "a3"], output_clusters=result.output.clusters
    )
    member_lookup = {
        cid: i
        for i, cluster in enumerate(result.output.clusters)
        for cid in cluster.member_claim_ids
    }
    assert member_lookup["a1"] == member_lookup["a2"]
    assert member_lookup["a1"] != member_lookup["a3"]


@pytest.mark.family_referential_integrity
def test_output_emits_no_extra_or_missing_ids() -> None:
    """The §6.6 invariant rejects extras and missing — the merger must
    emit exactly the input ids."""
    inputs = [_input(f"i{n}") for n in range(5)]
    scorer = _DictScorer({})
    result = merge(input_claims=inputs, scorer=scorer)
    emitted = sorted(cid for cluster in result.output.clusters for cid in cluster.member_claim_ids)
    assert emitted == sorted(ic.id for ic in inputs)


# ---------------------------------------------------------------------------
# Galois join — strongest representative
# ---------------------------------------------------------------------------


@pytest.mark.family_verifier_calibration
def test_galois_join_picks_strongest_modality_as_representative() -> None:
    """A cluster of {MUST, SHOULD} merges and emits MUST as the strongest claim."""
    must = _claim(subject="x", predicate="p", object_="y", modality="obligatory")
    should = _claim(subject="x", predicate="p", object_="y", modality="recommended")
    inputs = [_input("a1", claim=should), _input("a2", claim=must)]
    scorer = _DictScorer({})
    result = merge(input_claims=inputs, scorer=scorer)
    assert len(result.output.clusters) == 1
    cluster = result.output.clusters[0]
    assert set(cluster.member_claim_ids) == {"a1", "a2"}
    assert cluster.strongest_claim == must
    # Galois floor short-circuits — no NLI call.
    assert result.scorer_calls == 0


@pytest.mark.family_verifier_calibration
def test_galois_join_picks_unscoped_over_qualifier_scoped() -> None:
    """An empty qualifier is strictly stronger than a scoped one."""
    universal = _claim(subject="x", predicate="p", object_="y", qualifier="")
    scoped = _claim(subject="x", predicate="p", object_="y", qualifier="for new users")
    inputs = [_input("a1", claim=scoped), _input("a2", claim=universal)]
    scorer = _DictScorer({})
    result = merge(input_claims=inputs, scorer=scorer)
    assert len(result.output.clusters) == 1
    cluster = result.output.clusters[0]
    assert set(cluster.member_claim_ids) == {"a1", "a2"}
    assert cluster.strongest_claim == universal


# ---------------------------------------------------------------------------
# Cost contract — NLI calls bounded by pairs that survive the Galois floor
# ---------------------------------------------------------------------------


@pytest.mark.family_performance_cost
def test_galois_resolvable_pairs_skip_nli() -> None:
    """When all pairs are Galois-resolvable (modality-driven), zero NLI cost."""
    must = _claim(subject="x", predicate="p", object_="y", modality="obligatory")
    should = _claim(subject="x", predicate="p", object_="y", modality="recommended")
    may = _claim(subject="x", predicate="p", object_="y", modality="permitted")
    inputs = [_input("a1", claim=must), _input("a2", claim=should), _input("a3", claim=may)]
    scorer = _DictScorer({})
    result = merge(input_claims=inputs, scorer=scorer)
    assert result.scorer_calls == 0
    # All three collapse into one cluster (all on the deontic axis).
    assert len(result.output.clusters) == 1


@pytest.mark.family_performance_cost
def test_nli_only_runs_on_incomparable_pairs() -> None:
    """NLI is asked exactly twice per (i < j) pair the Galois floor refused."""
    # 3-claim layout, three (i < j) pairs:
    #   (a1, a3): same SVO+polarity, modality obligatory vs recommended →
    #             Galois `subsumes` (deontic chain) → 0 NLI calls.
    #   (a1, a2): different SVO → Galois `incomparable` → 2 NLI calls.
    #   (a2, a3): different SVO → Galois `incomparable` → 2 NLI calls.
    a1_claim = _claim(subject="X", predicate="p", object_="1", modality="obligatory")
    a2_claim = _claim(subject="Y", predicate="q", object_="2")
    a3_claim = _claim(subject="X", predicate="p", object_="1", modality="recommended")
    inputs = [
        _input("a1", claim=a1_claim),
        _input("a2", claim=a2_claim),
        _input("a3", claim=a3_claim),
    ]
    scorer = _DictScorer({})
    merge(input_claims=inputs, scorer=scorer)
    # Total: 0 + 2 + 2 = 4 calls.
    assert len(scorer.calls) == 4


# ---------------------------------------------------------------------------
# Determinism — identical input → identical output
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_repeat_calls_produce_identical_partitions() -> None:
    inputs = [
        _input("a1", claim=_claim(subject="x", predicate="p", object_="1")),
        _input("a2", claim=_claim(subject="x", predicate="p", object_="2")),
        _input("a3", claim=_claim(subject="x", predicate="p", object_="1")),
    ]
    scorer1 = _DictScorer({})
    scorer2 = _DictScorer({})
    r1 = merge(input_claims=inputs, scorer=scorer1)
    r2 = merge(input_claims=inputs, scorer=scorer2)
    # Compare partitions by member-set (cluster order is deterministic too).
    p1 = sorted(tuple(sorted(c.member_claim_ids)) for c in r1.output.clusters)
    p2 = sorted(tuple(sorted(c.member_claim_ids)) for c in r2.output.clusters)
    assert p1 == p2


# ---------------------------------------------------------------------------
# Merger shape — implements `Merger` protocol
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_transport_merger_implements_merger_protocol() -> None:
    """`TransportMerger` satisfies `Merger`."""
    merger: Merger = TransportMerger(scorer=_DictScorer({}))
    out = merger.merge(input_claims=[_input("a1")])
    assert len(out.clusters) == 1
    assert out.clusters[0].member_claim_ids == ["a1"]


# ---------------------------------------------------------------------------
# Release-gate eval — loss invariant + pairwise accuracy ≥ 0.85 on fixture
# ---------------------------------------------------------------------------


class _MergeGoldOracle:
    """Deterministic NLI oracle aligned to the merge eval fixture.

    For every gold cluster with size > 1 whose member pairs are
    Galois-`incomparable`, the oracle returns high bidirectional
    entailment on every within-cluster pair and the default (low
    entailment) for between-cluster pairs. This isolates the §6.6
    transport reduction's partition behaviour from any real NLI
    backend; the release-gate constants are being asserted *of the
    reduction*, not of the model.
    """

    def __init__(self, cases: list[MergeEvalCase]) -> None:
        from ctrldoc.extract.galois import claim_subsumption
        from ctrldoc.ops.merge import _render_claim

        self._table: dict[tuple[str, str], NLIScore] = {}
        self.calls: list[tuple[str, str]] = []
        for case in cases:
            by_id = {ic.id: ic for ic in case.input_claims}
            for gold in case.gold_clusters:
                members = list(gold.member_claim_ids)
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        a = by_id[members[i]].claim
                        b = by_id[members[j]].claim
                        verdict = claim_subsumption(a, b)
                        if verdict != "incomparable":
                            continue  # Galois floor handles it.
                        a_text = _render_claim(a)
                        b_text = _render_claim(b)
                        self._table[(a_text, b_text)] = _high_entail()
                        self._table[(b_text, a_text)] = _high_entail()

    def score(self, *, premise: str, hypothesis: str) -> NLIScore:
        self.calls.append((premise, hypothesis))
        return self._table.get((premise, hypothesis), _low_entail())


@pytest.mark.family_verifier_calibration
def test_transport_merge_clears_release_gate_on_eval_fixture() -> None:
    """§6.6 loss invariant + pairwise accuracy ≥ 0.85 on the 6-case fixture."""
    cases = load_jsonl_cases(MERGE_EVAL_PATH, case_model=MergeEvalCase)
    oracle = _MergeGoldOracle(cases)
    merger = TransportMerger(scorer=oracle)
    runner = MergeEvalRunner(merger=merger)

    report = run_eval(
        set_name="ops_merge_release_gate",
        cases=cases,
        runner=runner,
        thresholds={"pairwise_accuracy": MERGE_PARTITION_THRESHOLD},
    )
    # Per-case checks: every case must satisfy the loss invariant AND
    # clear the pairwise-accuracy threshold.
    for case in cases:
        out = merger.merge(input_claims=list(case.input_claims))
        input_ids = [c.id for c in case.input_claims]
        assert loss_invariant_satisfied(
            input_ids=input_ids, output_clusters=out.clusters
        ), f"case {case.id} violates §6.6 loss invariant"
        pairwise = pairwise_partition_accuracy(
            predicted=out.clusters,
            gold=list(case.gold_clusters),
            input_ids=input_ids,
        )
        assert (
            pairwise >= MERGE_PARTITION_THRESHOLD
        ), f"case {case.id} pairwise_accuracy={pairwise:.3f} < {MERGE_PARTITION_THRESHOLD}"

    assert report.passed is True
    assert len(report.results) == len(cases)
