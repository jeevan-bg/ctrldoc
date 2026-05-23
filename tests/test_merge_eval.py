"""merge_eval — partition + Galois-join verdict scoring.

The runner takes a `Merger` and grades it on three axes: the §6.6
loss invariant (hard gate — every input claim id appears in exactly
one output cluster), pairwise partition accuracy on the input id
pairs (soft gate at 0.85), and representative-claim match rate
(reported but not gated). Per §6.6 the merger contract is binary
on the invariant; soft on the partition shape.

The starter dataset at `tests/eval/merge_eval.jsonl` ships 6 cases
spanning no-overlap singletons (every input claim becomes its own
cluster), full-overlap multi-member clusters (two identical specs),
modality-driven Galois join (recommended + obligatory → obligatory),
three-doc partial overlap with mixed cluster sizes, paraphrase
clustering (different surface forms, one merged cluster), and
cross-doc-type topic-disjoint singletons.

SPEC-REF: §6.6 (merge = partition + Galois join with loss invariant), §14
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

from ctrldoc.eval.claim_extraction import ClaimTuple
from ctrldoc.eval.harness import load_jsonl_cases, run_eval
from ctrldoc.eval.merge import (
    MERGE_PARTITION_THRESHOLD,
    InputClaim,
    MergedCluster,
    MergeEvalCase,
    MergeEvalRunner,
    MergeOutput,
    Merger,
    loss_invariant_satisfied,
    pairwise_partition_accuracy,
    representative_match_rate,
)

MERGE_EVAL_PATH = Path(__file__).parent / "eval" / "merge_eval.jsonl"


def _cases() -> list[MergeEvalCase]:
    return load_jsonl_cases(MERGE_EVAL_PATH, case_model=MergeEvalCase)


def _claim(
    subject: str = "x",
    predicate: str = "is",
    object_: str = "y",
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


def _input(id_: str, claim: ClaimTuple, doc_id: str = "docA") -> InputClaim:
    return InputClaim(id=id_, doc_id=doc_id, doc_type="spec", claim=claim)


def _cluster(id_: str, members: list[str], strongest: ClaimTuple | None = None) -> MergedCluster:
    return MergedCluster(
        id=id_,
        member_claim_ids=members,
        strongest_claim=strongest or _claim(),
    )


# --- InputClaim contract ---


def test_input_claim_is_frozen() -> None:
    ic = _input("a1", _claim())
    with pytest.raises(ValidationError):
        ic.id = "a2"  # type: ignore[misc]


def test_input_claim_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        InputClaim(
            id="a1",
            doc_id="docA",
            doc_type="spec",
            claim=_claim(),
            stray="oops",  # type: ignore[call-arg]
        )


def test_input_claim_rejects_unknown_doc_type() -> None:
    with pytest.raises(ValidationError):
        InputClaim(
            id="a1",
            doc_id="docA",
            doc_type="poetry",  # type: ignore[arg-type]
            claim=_claim(),
        )


# --- MergedCluster contract ---


def test_cluster_rejects_empty_members() -> None:
    with pytest.raises(ValidationError):
        MergedCluster(id="c1", member_claim_ids=[], strongest_claim=_claim())


def test_cluster_rejects_duplicate_members() -> None:
    with pytest.raises(ValidationError, match="must be unique"):
        MergedCluster(id="c1", member_claim_ids=["a1", "a1"], strongest_claim=_claim())


def test_cluster_is_frozen() -> None:
    c = _cluster("c1", ["a1"])
    with pytest.raises(ValidationError):
        c.id = "c2"  # type: ignore[misc]


# --- MergeOutput contract ---


def test_output_rejects_duplicate_cluster_ids() -> None:
    with pytest.raises(ValidationError, match="cluster ids must be unique"):
        MergeOutput(
            clusters=[_cluster("c1", ["a1"]), _cluster("c1", ["a2"])],
        )


def test_output_rejects_empty_clusters_list() -> None:
    with pytest.raises(ValidationError):
        MergeOutput(clusters=[])


# --- MergeEvalCase loss invariant on gold ---


def test_case_rejects_gold_missing_input_id() -> None:
    """If gold doesn't cover every input claim id, the case is malformed."""
    with pytest.raises(ValidationError, match="loss invariant"):
        MergeEvalCase(
            id="bad",
            input_claims=[_input("a1", _claim()), _input("a2", _claim())],
            gold_clusters=[_cluster("c1", ["a1"])],  # a2 missing
        )


def test_case_rejects_gold_extra_id() -> None:
    """If gold names an id not in input, the case is malformed."""
    with pytest.raises(ValidationError, match="loss invariant"):
        MergeEvalCase(
            id="bad",
            input_claims=[_input("a1", _claim())],
            gold_clusters=[_cluster("c1", ["a1", "rogue"])],
        )


def test_case_rejects_gold_duplicate_assignment() -> None:
    """If gold assigns the same input id to two clusters, malformed."""
    with pytest.raises(ValidationError, match="loss invariant"):
        MergeEvalCase(
            id="bad",
            input_claims=[_input("a1", _claim())],
            gold_clusters=[_cluster("c1", ["a1"]), _cluster("c2", ["a1"])],
        )


def test_case_rejects_duplicate_input_ids() -> None:
    with pytest.raises(ValidationError, match="unique ids"):
        MergeEvalCase(
            id="dup",
            input_claims=[_input("a1", _claim()), _input("a1", _claim())],
            gold_clusters=[_cluster("c1", ["a1"])],
        )


def test_case_accepts_singleton_partition() -> None:
    case = MergeEvalCase(
        id="ok",
        input_claims=[_input("a1", _claim()), _input("a2", _claim(subject="b"))],
        gold_clusters=[
            _cluster("c1", ["a1"]),
            _cluster("c2", ["a2"], strongest=_claim(subject="b")),
        ],
    )
    assert len(case.gold_clusters) == 2


# --- loss_invariant_satisfied ---


def test_invariant_holds_when_perfect() -> None:
    assert (
        loss_invariant_satisfied(
            input_ids=["a1", "a2", "a3"],
            output_clusters=[
                _cluster("c1", ["a1", "a2"]),
                _cluster("c2", ["a3"]),
            ],
        )
        is True
    )


def test_invariant_fails_on_missing_id() -> None:
    assert (
        loss_invariant_satisfied(
            input_ids=["a1", "a2"],
            output_clusters=[_cluster("c1", ["a1"])],
        )
        is False
    )


def test_invariant_fails_on_extra_id() -> None:
    assert (
        loss_invariant_satisfied(
            input_ids=["a1"],
            output_clusters=[_cluster("c1", ["a1", "ghost"])],
        )
        is False
    )


def test_invariant_fails_on_duplicate_assignment() -> None:
    assert (
        loss_invariant_satisfied(
            input_ids=["a1", "a2"],
            output_clusters=[_cluster("c1", ["a1"]), _cluster("c2", ["a1", "a2"])],
        )
        is False
    )


# --- pairwise_partition_accuracy ---


def test_pairwise_identical_partition_is_one() -> None:
    clusters = [_cluster("c1", ["a1", "a2"]), _cluster("c2", ["a3"])]
    acc = pairwise_partition_accuracy(
        predicted=clusters, gold=clusters, input_ids=["a1", "a2", "a3"]
    )
    assert acc == pytest.approx(1.0)


def test_pairwise_singletons_vs_one_blob_is_zero() -> None:
    """Three claims: all-singleton vs all-in-one disagrees on every pair."""
    singletons = [_cluster("c1", ["a1"]), _cluster("c2", ["a2"]), _cluster("c3", ["a3"])]
    one_blob = [_cluster("c1", ["a1", "a2", "a3"])]
    acc = pairwise_partition_accuracy(
        predicted=one_blob, gold=singletons, input_ids=["a1", "a2", "a3"]
    )
    # 3 pairs total, all disagree
    assert acc == pytest.approx(0.0)


def test_pairwise_partial_agreement() -> None:
    """{a1,a2} | {a3} vs {a1} | {a2,a3} — pair (a1,a2): disagree; pair (a1,a3): agree (different in both); pair (a2,a3): disagree. 1/3 agree."""
    pred = [_cluster("c1", ["a1", "a2"]), _cluster("c2", ["a3"])]
    gold = [_cluster("c1", ["a1"]), _cluster("c2", ["a2", "a3"])]
    acc = pairwise_partition_accuracy(predicted=pred, gold=gold, input_ids=["a1", "a2", "a3"])
    assert acc == pytest.approx(1 / 3)


def test_pairwise_handles_fewer_than_two_inputs() -> None:
    assert pairwise_partition_accuracy(predicted=[], gold=[], input_ids=[]) == pytest.approx(1.0)
    assert pairwise_partition_accuracy(
        predicted=[_cluster("c1", ["a1"])],
        gold=[_cluster("c1", ["a1"])],
        input_ids=["a1"],
    ) == pytest.approx(1.0)


def test_pairwise_missing_in_predicted_counts_as_disagreement() -> None:
    """Predicted drops an id; the pair is scored disagree rather than skipped."""
    pred = [_cluster("c1", ["a1"])]
    gold = [_cluster("c1", ["a1", "a2"])]
    acc = pairwise_partition_accuracy(predicted=pred, gold=gold, input_ids=["a1", "a2"])
    # 1 pair; pred has no a2 → disagree
    assert acc == pytest.approx(0.0)


# --- representative_match_rate ---


def test_representative_perfect_match() -> None:
    pred = [_cluster("c1", ["a1", "a2"], strongest=_claim(modality="obligatory"))]
    gold = [_cluster("c1", ["a1", "a2"], strongest=_claim(modality="obligatory"))]
    assert representative_match_rate(predicted=pred, gold=gold) == pytest.approx(1.0)


def test_representative_wrong_strongest_scores_zero() -> None:
    pred = [_cluster("c1", ["a1", "a2"], strongest=_claim(modality="recommended"))]
    gold = [_cluster("c1", ["a1", "a2"], strongest=_claim(modality="obligatory"))]
    assert representative_match_rate(predicted=pred, gold=gold) == pytest.approx(0.0)


def test_representative_uses_best_overlap_for_matching() -> None:
    pred = [_cluster("c1", ["a1", "a2"], strongest=_claim(subject="alpha"))]
    gold = [
        _cluster("g1", ["a1"], strongest=_claim(subject="beta")),
        _cluster("g2", ["a2"], strongest=_claim(subject="alpha")),
    ]
    # pred overlaps g1 by 1, g2 by 1 → first-best wins (g1) → predicted strongest
    # is "alpha" but gold g1 strongest is "beta" → mismatch → 0
    assert representative_match_rate(predicted=pred, gold=gold) == pytest.approx(0.0)


def test_representative_no_overlap_with_any_gold_scores_zero() -> None:
    pred = [_cluster("c1", ["rogue"], strongest=_claim())]
    gold = [_cluster("g1", ["real"], strongest=_claim())]
    assert representative_match_rate(predicted=pred, gold=gold) == pytest.approx(0.0)


def test_representative_empty_predicted_is_one() -> None:
    assert representative_match_rate(predicted=[], gold=[_cluster("g1", ["a1"])]) == 1.0


# --- Merger Protocol conformance ---


def test_oracle_merger_satisfies_protocol() -> None:
    @dataclass
    class _Oracle:
        canned: MergeOutput

        def merge(self, *, input_claims: list[InputClaim]) -> MergeOutput:
            return self.canned

    assert isinstance(_Oracle(canned=MergeOutput(clusters=[_cluster("c1", ["a1"])])), Merger)


# --- Runner ---


@dataclass
class _GoldOracle:
    """Returns the case's gold clusters as the predicted output."""

    gold_by_case: dict[str, list[MergedCluster]]
    current_case_id: str = ""

    def merge(self, *, input_claims: list[InputClaim]) -> MergeOutput:
        return MergeOutput(clusters=list(self.gold_by_case[self.current_case_id]))


def _gold_lookup(cases: list[MergeEvalCase]) -> dict[str, list[MergedCluster]]:
    return {c.id: list(c.gold_clusters) for c in cases}


def test_runner_oracle_passes_starter_case() -> None:
    cases = _cases()
    case = cases[0]
    runner = MergeEvalRunner(
        merger=_GoldOracle(gold_by_case=_gold_lookup(cases), current_case_id=case.id),
    )
    result = runner.run_case(case)
    assert result.metrics["loss_invariant_satisfied"] == 1.0
    assert result.metrics["pairwise_accuracy"] == pytest.approx(1.0)
    assert result.metrics["representative_match_rate"] == pytest.approx(1.0)
    assert result.passed is True


def test_runner_invariant_failure_zeros_score() -> None:
    """A merger that drops an input claim fails the hard gate."""
    cases = _cases()
    case = next(c for c in cases if len(c.input_claims) > 1)

    @dataclass
    class _Dropper:
        def merge(self, *, input_claims: list[InputClaim]) -> MergeOutput:
            # drop the last claim entirely
            kept = input_claims[:-1]
            return MergeOutput(clusters=[_cluster("c1", [c.id for c in kept])])

    result = MergeEvalRunner(merger=_Dropper()).run_case(case)
    assert result.metrics["loss_invariant_satisfied"] == 0.0
    assert result.score == pytest.approx(0.0)
    assert result.passed is False


def test_runner_one_blob_partition_fails_threshold_on_mixed_case() -> None:
    """A merger that lumps everything into one cluster fails partition on cases that have multiple gold clusters."""
    cases = _cases()
    multi_gold = next(c for c in cases if len(c.gold_clusters) >= 3)

    @dataclass
    class _OneBlob:
        def merge(self, *, input_claims: list[InputClaim]) -> MergeOutput:
            return MergeOutput(clusters=[_cluster("c1", [c.id for c in input_claims])])

    result = MergeEvalRunner(merger=_OneBlob()).run_case(multi_gold)
    # Loss invariant still satisfied (all ids assigned, no extras), but
    # partition is wrong on most pairs.
    assert result.metrics["loss_invariant_satisfied"] == 1.0
    assert result.metrics["pairwise_accuracy"] < MERGE_PARTITION_THRESHOLD
    assert result.passed is False


def test_runner_singleton_merger_passes_no_overlap_case() -> None:
    """A merger that produces singletons matches gold on no-overlap cases."""
    cases = _cases()
    singletons_case = next(c for c in cases if "all-singletons" in c.tags)

    @dataclass
    class _Singletons:
        def merge(self, *, input_claims: list[InputClaim]) -> MergeOutput:
            return MergeOutput(
                clusters=[
                    _cluster(f"c{i}", [c.id], strongest=c.claim) for i, c in enumerate(input_claims)
                ]
            )

    result = MergeEvalRunner(merger=_Singletons()).run_case(singletons_case)
    assert result.passed is True
    assert result.metrics["pairwise_accuracy"] == pytest.approx(1.0)


def test_runner_emits_three_named_metrics() -> None:
    cases = _cases()
    case = cases[0]
    runner = MergeEvalRunner(
        merger=_GoldOracle(gold_by_case=_gold_lookup(cases), current_case_id=case.id),
    )
    result = runner.run_case(case)
    assert set(result.metrics.keys()) == {
        "loss_invariant_satisfied",
        "pairwise_accuracy",
        "representative_match_rate",
    }


# --- Dataset invariants ---


def test_dataset_has_exactly_six_cases() -> None:
    assert len(_cases()) == 6


def test_dataset_ids_are_unique() -> None:
    ids = [c.id for c in _cases()]
    assert len(ids) == len(set(ids))


def test_dataset_every_gold_satisfies_loss_invariant() -> None:
    """Pinned by the model validator, but check it on the loaded set too."""
    for case in _cases():
        input_ids = [c.id for c in case.input_claims]
        assert loss_invariant_satisfied(
            input_ids=input_ids, output_clusters=list(case.gold_clusters)
        )


def test_dataset_includes_singleton_and_multi_member_clusters() -> None:
    seen_singleton = False
    seen_multi = False
    for case in _cases():
        for cl in case.gold_clusters:
            if len(cl.member_claim_ids) == 1:
                seen_singleton = True
            elif len(cl.member_claim_ids) >= 2:
                seen_multi = True
    assert seen_singleton, "expected at least one singleton cluster"
    assert seen_multi, "expected at least one multi-member cluster"


def test_dataset_includes_three_doc_case() -> None:
    """At least one case merges claims from three or more input docs."""
    assert any(len({c.doc_id for c in case.input_claims}) >= 3 for case in _cases())


def test_dataset_includes_modality_join_case() -> None:
    assert any("galois" in c.tags or "modality-join" in c.tags for c in _cases())


def test_dataset_includes_paraphrase_case() -> None:
    assert any("paraphrase" in c.tags for c in _cases())


def test_dataset_includes_cross_doc_type_case() -> None:
    """At least one case mixes doc types across the input claims."""
    for case in _cases():
        types = {c.doc_type for c in case.input_claims}
        if len(types) >= 2:
            return
    raise AssertionError("expected at least one cross-doc-type case")


def test_dataset_every_id_has_mrg_prefix() -> None:
    for case in _cases():
        assert case.id.startswith("mrg-"), f"case id {case.id!r} should start with 'mrg-'"


# --- End-to-end through the harness ---


def test_starter_set_passes_threshold_with_oracle_merger() -> None:
    cases = _cases()
    gold_lookup = _gold_lookup(cases)

    @dataclass
    class _DispatchingRunner:
        def run_case(self, case: MergeEvalCase):  # type: ignore[no-untyped-def]
            return MergeEvalRunner(
                merger=_GoldOracle(gold_by_case=gold_lookup, current_case_id=case.id),
            ).run_case(case)

    report = run_eval(
        set_name="merge_eval",
        cases=cases,
        runner=_DispatchingRunner(),
        thresholds={"pairwise_accuracy": MERGE_PARTITION_THRESHOLD},
    )
    assert report.passed is True
    assert report.aggregate["pairwise_accuracy"] == pytest.approx(1.0)
    assert report.aggregate["loss_invariant_satisfied"] == pytest.approx(1.0)


def test_starter_set_fails_threshold_with_one_blob_merger() -> None:
    cases = _cases()

    @dataclass
    class _OneBlob:
        def merge(self, *, input_claims: list[InputClaim]) -> MergeOutput:
            return MergeOutput(clusters=[_cluster("c1", [c.id for c in input_claims])])

    @dataclass
    class _DispatchingRunner:
        def run_case(self, case: MergeEvalCase):  # type: ignore[no-untyped-def]
            return MergeEvalRunner(merger=_OneBlob()).run_case(case)

    report = run_eval(
        set_name="merge_eval",
        cases=cases,
        runner=_DispatchingRunner(),
        thresholds={"pairwise_accuracy": MERGE_PARTITION_THRESHOLD},
    )
    assert report.passed is False
    assert report.aggregate["pairwise_accuracy"] < MERGE_PARTITION_THRESHOLD
