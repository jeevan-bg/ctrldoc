"""compare_eval — per-concept-cluster verdict scoring.

The runner takes a `CompareVerifier` and grades it on the
{StrengthA, StrengthB, Gap} 3-label space. The metric is per-cluster
accuracy on the case's clusters, with per-class precision / recall
surfaced for legibility. Per §6.6 the gate is per-cluster accuracy
≥ 0.85 — matching the per-claim gate used by `coverage` since both
operations reduce to the same optimal-transport substrate.

The starter dataset at `tests/eval/compare_eval.jsonl` ships 8 doc-pair
strengths/weaknesses/gaps tuples that exercise modality ordering
(MUST > SHOULD > MAY), polarity-aware strength comparison
(SHALL NOT vs SHOULD NOT), asserted-vs-hypothetical strength
(academic claims), one-sided gaps in both directions, and disjoint
spec-vs-runbook concept spaces.

SPEC-REF: §6.6 (compare = per-concept-cluster strengths/weaknesses), §14
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

from ctrldoc.eval.claim_extraction import ClaimTuple
from ctrldoc.eval.compare import (
    COMPARE_VERDICT_THRESHOLD,
    COMPARE_VERDICTS,
    CompareEvalCase,
    CompareEvalRunner,
    CompareVerdictLiteral,
    CompareVerifier,
    ConceptComparison,
    ConceptComparisonInput,
    compare_accuracy,
)
from ctrldoc.eval.harness import load_jsonl_cases, run_eval

CMP_EVAL_PATH = Path(__file__).parent / "eval" / "compare_eval.jsonl"


def _cases() -> list[CompareEvalCase]:
    return load_jsonl_cases(CMP_EVAL_PATH, case_model=CompareEvalCase)


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


# --- ConceptComparisonInput contract ---


def test_input_requires_at_least_one_side() -> None:
    with pytest.raises(ValidationError):
        ConceptComparisonInput(id="c1", label="x", a_claim=None, b_claim=None)


def test_input_is_frozen() -> None:
    c = ConceptComparisonInput(id="c1", label="x", a_claim=_claim(), b_claim=None)
    with pytest.raises(ValidationError):
        c.id = "c2"  # type: ignore[misc]


def test_input_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ConceptComparisonInput(
            id="c1",
            label="x",
            a_claim=_claim(),
            b_claim=None,
            stray="oops",  # type: ignore[call-arg]
        )


# --- ConceptComparison contract ---


def test_cluster_rejects_gap_with_both_sides() -> None:
    with pytest.raises(ValidationError, match="exactly one side absent"):
        ConceptComparison(
            id="c1",
            label="x",
            a_claim=_claim(),
            b_claim=_claim(),
            gold_verdict="Gap",
        )


def test_cluster_rejects_strength_with_only_one_side() -> None:
    for verdict in ("StrengthA", "StrengthB"):
        with pytest.raises(ValidationError, match="both sides present"):
            ConceptComparison(
                id="c1",
                label="x",
                a_claim=_claim(),
                b_claim=None,
                gold_verdict=verdict,  # type: ignore[arg-type]
            )


def test_cluster_accepts_gap_with_only_a() -> None:
    c = ConceptComparison(
        id="c1",
        label="x",
        a_claim=_claim(),
        b_claim=None,
        gold_verdict="Gap",
    )
    assert c.gold_verdict == "Gap"


def test_cluster_accepts_gap_with_only_b() -> None:
    c = ConceptComparison(
        id="c1",
        label="x",
        a_claim=None,
        b_claim=_claim(),
        gold_verdict="Gap",
    )
    assert c.gold_verdict == "Gap"


def test_cluster_verdict_literal_enforced() -> None:
    with pytest.raises(ValidationError):
        ConceptComparison(
            id="c1",
            label="x",
            a_claim=_claim(),
            b_claim=_claim(),
            gold_verdict="Equivalent",  # type: ignore[arg-type]
        )


def test_cluster_to_input_strips_gold() -> None:
    c = ConceptComparison(
        id="c1",
        label="x",
        a_claim=_claim(),
        b_claim=_claim(modality="obligatory"),
        gold_verdict="StrengthB",
    )
    inp = c.to_input()
    assert isinstance(inp, ConceptComparisonInput)
    assert inp.id == "c1"
    assert inp.a_claim == c.a_claim
    assert inp.b_claim == c.b_claim
    assert not hasattr(inp, "gold_verdict")


# --- CompareEvalCase contract ---


def test_case_rejects_empty_cluster_list() -> None:
    with pytest.raises(ValidationError):
        CompareEvalCase(
            id="bad",
            a_doc_type="spec",
            b_doc_type="spec",
            clusters=[],
        )


def test_case_rejects_duplicate_cluster_ids() -> None:
    with pytest.raises(ValidationError):
        CompareEvalCase(
            id="dup",
            a_doc_type="spec",
            b_doc_type="spec",
            clusters=[
                ConceptComparison(
                    id="c1",
                    label="x",
                    a_claim=_claim(),
                    b_claim=None,
                    gold_verdict="Gap",
                ),
                ConceptComparison(
                    id="c1",
                    label="y",
                    a_claim=None,
                    b_claim=_claim(),
                    gold_verdict="Gap",
                ),
            ],
        )


def test_case_rejects_unknown_doc_type() -> None:
    with pytest.raises(ValidationError):
        CompareEvalCase(
            id="bad-dt",
            a_doc_type="poetry",  # type: ignore[arg-type]
            b_doc_type="spec",
            clusters=[
                ConceptComparison(
                    id="c1",
                    label="x",
                    a_claim=_claim(),
                    b_claim=None,
                    gold_verdict="Gap",
                )
            ],
        )


# --- compare_accuracy ---


def test_accuracy_all_correct() -> None:
    metrics = compare_accuracy(
        predicted=["StrengthA", "StrengthB", "Gap"],
        gold=["StrengthA", "StrengthB", "Gap"],
    )
    assert metrics["accuracy"] == pytest.approx(1.0)
    for label in COMPARE_VERDICTS:
        assert metrics[f"{label.lower()}_precision"] == pytest.approx(1.0)
        assert metrics[f"{label.lower()}_recall"] == pytest.approx(1.0)


def test_accuracy_all_wrong_is_zero() -> None:
    metrics = compare_accuracy(
        predicted=["StrengthB", "Gap", "StrengthA"],
        gold=["StrengthA", "StrengthB", "Gap"],
    )
    assert metrics["accuracy"] == pytest.approx(0.0)


def test_accuracy_constant_predictor_collapses_recall() -> None:
    """A verifier that always returns `Gap` cannot pass on a mixed set."""
    metrics = compare_accuracy(
        predicted=["Gap", "Gap", "Gap", "Gap"],
        gold=["StrengthA", "StrengthB", "Gap", "Gap"],
    )
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["strengtha_recall"] == pytest.approx(0.0)
    assert metrics["strengthb_recall"] == pytest.approx(0.0)
    assert metrics["gap_recall"] == pytest.approx(1.0)


def test_accuracy_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="must align"):
        compare_accuracy(predicted=["Gap"], gold=["Gap", "StrengthA"])


def test_accuracy_empty_inputs_return_zero() -> None:
    metrics = compare_accuracy(predicted=[], gold=[])
    assert metrics["accuracy"] == pytest.approx(0.0)
    for label in COMPARE_VERDICTS:
        assert metrics[f"{label.lower()}_precision"] == pytest.approx(0.0)
        assert metrics[f"{label.lower()}_recall"] == pytest.approx(0.0)


def test_accuracy_emits_seven_named_metrics() -> None:
    metrics = compare_accuracy(predicted=["Gap"], gold=["Gap"])
    expected = {"accuracy"} | {
        f"{label.lower()}_{kind}" for label in COMPARE_VERDICTS for kind in ("precision", "recall")
    }
    assert set(metrics.keys()) == expected


# --- CompareVerifier Protocol conformance ---


def test_oracle_verifier_satisfies_protocol() -> None:
    @dataclass
    class _Oracle:
        canned: list[CompareVerdictLiteral]

        def verdicts(
            self,
            *,
            clusters: list[ConceptComparisonInput],
        ) -> list[CompareVerdictLiteral]:
            return list(self.canned[: len(clusters)])

    assert isinstance(_Oracle(canned=[]), CompareVerifier)


# --- Runner ---


@dataclass
class _GoldOracle:
    """Looks each cluster up in a per-case gold map by case id."""

    gold_by_case: dict[str, list[CompareVerdictLiteral]]
    current_case_id: str = ""

    def verdicts(
        self,
        *,
        clusters: list[ConceptComparisonInput],
    ) -> list[CompareVerdictLiteral]:
        return list(self.gold_by_case[self.current_case_id])


def _gold_lookup(cases: list[CompareEvalCase]) -> dict[str, list[CompareVerdictLiteral]]:
    return {c.id: [cl.gold_verdict for cl in c.clusters] for c in cases}


def test_runner_oracle_clears_threshold_on_starter_case() -> None:
    cases = _cases()
    case = cases[0]
    runner = CompareEvalRunner(
        verifier=_GoldOracle(gold_by_case=_gold_lookup(cases), current_case_id=case.id),
    )
    result = runner.run_case(case)
    assert result.metrics["accuracy"] == pytest.approx(1.0)
    assert result.passed is True


def test_runner_strips_gold_before_verifier_sees_clusters() -> None:
    """The verifier must not be able to read gold_verdict from inputs."""
    cases = _cases()
    case = cases[0]
    seen_inputs: list[ConceptComparisonInput] = []

    @dataclass
    class _Spy:
        def verdicts(
            self,
            *,
            clusters: list[ConceptComparisonInput],
        ) -> list[CompareVerdictLiteral]:
            seen_inputs.extend(clusters)
            return ["Gap"] * len(clusters)

    CompareEvalRunner(verifier=_Spy()).run_case(case)
    for inp in seen_inputs:
        assert not hasattr(inp, "gold_verdict")


def test_runner_constant_predictor_fails_threshold_on_mixed_case() -> None:
    cases = _cases()
    mixed = next(c for c in cases if len({cl.gold_verdict for cl in c.clusters}) >= 2)

    @dataclass
    class _AlwaysGap:
        def verdicts(
            self,
            *,
            clusters: list[ConceptComparisonInput],
        ) -> list[CompareVerdictLiteral]:
            return ["Gap"] * len(clusters)

    result = CompareEvalRunner(verifier=_AlwaysGap()).run_case(mixed)
    assert result.metrics["accuracy"] < COMPARE_VERDICT_THRESHOLD
    assert result.passed is False


def test_runner_length_mismatch_raises() -> None:
    case = _cases()[0]

    @dataclass
    class _Short:
        def verdicts(
            self,
            *,
            clusters: list[ConceptComparisonInput],
        ) -> list[CompareVerdictLiteral]:
            return ["Gap"]

    with pytest.raises(ValueError, match="verdicts for"):
        CompareEvalRunner(verifier=_Short()).run_case(case)


def test_runner_emits_accuracy_and_per_class_metrics() -> None:
    cases = _cases()
    case = cases[0]
    runner = CompareEvalRunner(
        verifier=_GoldOracle(gold_by_case=_gold_lookup(cases), current_case_id=case.id),
    )
    result = runner.run_case(case)
    expected_keys = {"accuracy"} | {
        f"{label.lower()}_{kind}" for label in COMPARE_VERDICTS for kind in ("precision", "recall")
    }
    assert expected_keys.issubset(result.metrics.keys())


# --- Dataset invariants ---


def test_dataset_has_exactly_eight_cases() -> None:
    assert len(_cases()) == 8


def test_dataset_ids_are_unique() -> None:
    ids = [c.id for c in _cases()]
    assert len(ids) == len(set(ids))


def test_dataset_all_three_verdicts_present() -> None:
    seen: set[CompareVerdictLiteral] = set()
    for case in _cases():
        for cl in case.clusters:
            seen.add(cl.gold_verdict)
    assert seen == set(COMPARE_VERDICTS)


def test_dataset_each_verdict_used_at_least_four_times() -> None:
    counts: dict[CompareVerdictLiteral, int] = dict.fromkeys(COMPARE_VERDICTS, 0)
    for case in _cases():
        for cl in case.clusters:
            counts[cl.gold_verdict] += 1
    for verdict, count in counts.items():
        assert count >= 4, f"verdict {verdict!r} used {count} times (expected >= 4)"


def test_dataset_modality_driven_case_present() -> None:
    """At least one case exercises pure modality-ordering strength."""
    assert any("modality-driven" in c.tags for c in _cases())


def test_dataset_gap_heavy_case_present() -> None:
    assert any("gap-heavy" in c.tags or "disjoint" in c.tags for c in _cases())


def test_dataset_hypothetical_vs_asserted_case_present() -> None:
    assert any("hypothetical-vs-asserted" in c.tags for c in _cases())


def test_dataset_both_gap_directions_exercised() -> None:
    """Across the set there must be Gap clusters with only-A and only-B sides."""
    seen_only_a = False
    seen_only_b = False
    for case in _cases():
        for cl in case.clusters:
            if cl.gold_verdict != "Gap":
                continue
            if cl.a_claim is not None and cl.b_claim is None:
                seen_only_a = True
            if cl.a_claim is None and cl.b_claim is not None:
                seen_only_b = True
    assert seen_only_a, "expected a Gap cluster with only a_claim present"
    assert seen_only_b, "expected a Gap cluster with only b_claim present"


def test_dataset_doc_type_diversity() -> None:
    """At least three distinct doc types appear across the (a, b) pairs."""
    doc_types = set()
    for case in _cases():
        doc_types.add(case.a_doc_type)
        doc_types.add(case.b_doc_type)
    assert len(doc_types) >= 3


def test_dataset_every_id_has_cmp_prefix() -> None:
    for case in _cases():
        assert case.id.startswith("cmp-"), f"case id {case.id!r} should start with 'cmp-'"


def test_dataset_cluster_ids_unique_within_case() -> None:
    for case in _cases():
        ids = [cl.id for cl in case.clusters]
        assert len(ids) == len(set(ids)), f"case {case.id!r} has duplicate cluster ids"


# --- End-to-end through the harness ---


def test_starter_set_passes_threshold_with_oracle_verifier() -> None:
    cases = _cases()
    gold_lookup = _gold_lookup(cases)

    @dataclass
    class _DispatchingRunner:
        def run_case(self, case: CompareEvalCase):  # type: ignore[no-untyped-def]
            return CompareEvalRunner(
                verifier=_GoldOracle(gold_by_case=gold_lookup, current_case_id=case.id),
            ).run_case(case)

    report = run_eval(
        set_name="compare_eval",
        cases=cases,
        runner=_DispatchingRunner(),
        thresholds={"accuracy": COMPARE_VERDICT_THRESHOLD},
    )
    assert report.passed is True
    assert report.aggregate["accuracy"] == pytest.approx(1.0)


def test_starter_set_fails_threshold_with_always_gap() -> None:
    cases = _cases()

    @dataclass
    class _AlwaysGap:
        def verdicts(
            self,
            *,
            clusters: list[ConceptComparisonInput],
        ) -> list[CompareVerdictLiteral]:
            return ["Gap"] * len(clusters)

    @dataclass
    class _DispatchingRunner:
        def run_case(self, case: CompareEvalCase):  # type: ignore[no-untyped-def]
            return CompareEvalRunner(verifier=_AlwaysGap()).run_case(case)

    report = run_eval(
        set_name="compare_eval",
        cases=cases,
        runner=_DispatchingRunner(),
        thresholds={"accuracy": COMPARE_VERDICT_THRESHOLD},
    )
    assert report.passed is False
    assert report.aggregate["accuracy"] < COMPARE_VERDICT_THRESHOLD
