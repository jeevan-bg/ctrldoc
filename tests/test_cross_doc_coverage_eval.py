"""cross_doc_coverage_eval — per-target-claim verdict scoring.

The runner takes a `CrossDocCoverageVerifier` and grades it on the
{Covered, Missing} two-label space. The metric is per-target-claim
accuracy, with per-class precision / recall surfaced for legibility.
Per §6.6 the optimal-transport `coverage` operation must clear a
per-claim accuracy ≥ 0.85 gate; the eval substrate here pins that
contract.

The starter dataset at `tests/eval/cross_doc_coverage_eval.jsonl`
ships 12 hand-curated target/source/per-claim-verdict tuples that
exercise paraphrase coverage, modality ordering (stronger-covers-weaker
and weaker-misses-stronger), polarity-flip contradiction, topic-disjoint
sources, many-to-one transport, empty-source handling, qualifier
divergence, and per-doc-type-pair diversity.

SPEC-REF: §6.6 (optimal-transport coverage), §14 (eval substrate)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

from ctrldoc.eval.claim_extraction import ClaimTuple
from ctrldoc.eval.cross_doc_coverage import (
    COVERAGE_VERDICTS,
    CROSS_DOC_COVERAGE_THRESHOLD,
    CoverageVerdictLiteral,
    CrossDocCoverageEvalCase,
    CrossDocCoverageEvalRunner,
    CrossDocCoverageVerifier,
    TargetClaim,
    coverage_accuracy,
)
from ctrldoc.eval.harness import load_jsonl_cases, run_eval

CCOV_EVAL_PATH = Path(__file__).parent / "eval" / "cross_doc_coverage_eval.jsonl"


def _cases() -> list[CrossDocCoverageEvalCase]:
    return load_jsonl_cases(CCOV_EVAL_PATH, case_model=CrossDocCoverageEvalCase)


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


# --- TargetClaim model contract ---


def test_target_claim_is_frozen() -> None:
    t = TargetClaim(id="t1", claim=_claim(), gold_verdict="Covered")
    with pytest.raises(ValidationError):
        t.id = "t2"  # type: ignore[misc]


def test_target_claim_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        TargetClaim(
            id="t1",
            claim=_claim(),
            gold_verdict="Covered",
            stray="oops",  # type: ignore[call-arg]
        )


def test_target_claim_verdict_literal_enforced() -> None:
    with pytest.raises(ValidationError):
        TargetClaim(
            id="t1",
            claim=_claim(),
            gold_verdict="MaybeCovered",  # type: ignore[arg-type]
        )


# --- CrossDocCoverageEvalCase contract ---


def test_case_rejects_blank_target_list() -> None:
    with pytest.raises(ValidationError):
        CrossDocCoverageEvalCase(
            id="bad",
            source_doc_type="spec",
            target_doc_type="spec",
            source_claims=[_claim()],
            target_claims=[],
        )


def test_case_allows_empty_source_claims() -> None:
    case = CrossDocCoverageEvalCase(
        id="empty-src",
        source_doc_type="narrative",
        target_doc_type="spec",
        source_claims=[],
        target_claims=[TargetClaim(id="t1", claim=_claim(), gold_verdict="Missing")],
    )
    assert case.source_claims == []


def test_case_rejects_duplicate_target_ids() -> None:
    with pytest.raises(ValidationError):
        CrossDocCoverageEvalCase(
            id="dup",
            source_doc_type="spec",
            target_doc_type="spec",
            source_claims=[_claim()],
            target_claims=[
                TargetClaim(id="t1", claim=_claim(), gold_verdict="Covered"),
                TargetClaim(id="t1", claim=_claim(predicate="rejects"), gold_verdict="Missing"),
            ],
        )


def test_case_rejects_unknown_doc_type() -> None:
    with pytest.raises(ValidationError):
        CrossDocCoverageEvalCase(
            id="bad-dt",
            source_doc_type="poetry",  # type: ignore[arg-type]
            target_doc_type="spec",
            source_claims=[_claim()],
            target_claims=[TargetClaim(id="t1", claim=_claim(), gold_verdict="Covered")],
        )


# --- coverage_accuracy ---


def test_accuracy_all_correct() -> None:
    metrics = coverage_accuracy(
        predicted=["Covered", "Missing", "Covered"],
        gold=["Covered", "Missing", "Covered"],
    )
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["covered_precision"] == pytest.approx(1.0)
    assert metrics["covered_recall"] == pytest.approx(1.0)
    assert metrics["missing_precision"] == pytest.approx(1.0)
    assert metrics["missing_recall"] == pytest.approx(1.0)


def test_accuracy_all_wrong_is_zero() -> None:
    metrics = coverage_accuracy(
        predicted=["Missing", "Covered"],
        gold=["Covered", "Missing"],
    )
    assert metrics["accuracy"] == pytest.approx(0.0)


def test_accuracy_partial() -> None:
    metrics = coverage_accuracy(
        predicted=["Covered", "Covered", "Missing", "Missing"],
        gold=["Covered", "Missing", "Missing", "Covered"],
    )
    # 2 correct out of 4
    assert metrics["accuracy"] == pytest.approx(0.5)


def test_accuracy_always_covered_collapses_missing_recall() -> None:
    """A verifier that answers Covered every time scores `missing_recall` 0."""
    metrics = coverage_accuracy(
        predicted=["Covered", "Covered", "Covered", "Covered"],
        gold=["Covered", "Missing", "Covered", "Missing"],
    )
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["missing_recall"] == pytest.approx(0.0)
    assert metrics["covered_recall"] == pytest.approx(1.0)


def test_accuracy_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="must align"):
        coverage_accuracy(predicted=["Covered"], gold=["Covered", "Missing"])


def test_accuracy_empty_inputs_return_zero() -> None:
    metrics = coverage_accuracy(predicted=[], gold=[])
    assert metrics["accuracy"] == pytest.approx(0.0)


def test_accuracy_emits_five_named_metrics() -> None:
    metrics = coverage_accuracy(predicted=["Covered"], gold=["Covered"])
    assert set(metrics.keys()) == {
        "accuracy",
        "covered_precision",
        "covered_recall",
        "missing_precision",
        "missing_recall",
    }


# --- CrossDocCoverageVerifier Protocol conformance ---


def test_oracle_verifier_satisfies_protocol() -> None:
    @dataclass
    class _Oracle:
        verdicts_by_target_id: dict[int, CoverageVerdictLiteral]

        def verdicts(
            self,
            *,
            source: list[ClaimTuple],
            target: list[ClaimTuple],
        ) -> list[CoverageVerdictLiteral]:
            return ["Covered" for _ in target]

    assert isinstance(_Oracle(verdicts_by_target_id={}), CrossDocCoverageVerifier)


# --- Runner ---


@dataclass
class _OracleVerifier:
    """Looks each target claim up in a per-case gold map."""

    gold_by_case: dict[str, list[CoverageVerdictLiteral]]
    current_case_id: str = ""

    def verdicts(
        self,
        *,
        source: list[ClaimTuple],
        target: list[ClaimTuple],
    ) -> list[CoverageVerdictLiteral]:
        return list(self.gold_by_case[self.current_case_id])


def _gold_lookup(
    cases: list[CrossDocCoverageEvalCase],
) -> dict[str, list[CoverageVerdictLiteral]]:
    return {c.id: [tc.gold_verdict for tc in c.target_claims] for c in cases}


def test_runner_oracle_clears_threshold_on_starter_case() -> None:
    cases = _cases()
    case = cases[0]
    verifier = _OracleVerifier(gold_by_case=_gold_lookup(cases), current_case_id=case.id)
    runner = CrossDocCoverageEvalRunner(verifier=verifier)
    result = runner.run_case(case)
    assert result.metrics["accuracy"] == pytest.approx(1.0)
    assert result.passed is True


def test_runner_always_covered_fails_threshold_on_mixed_case() -> None:
    cases = _cases()
    mixed = next(
        c for c in cases if {tc.gold_verdict for tc in c.target_claims} == set(COVERAGE_VERDICTS)
    )

    @dataclass
    class _AlwaysCovered:
        def verdicts(
            self,
            *,
            source: list[ClaimTuple],
            target: list[ClaimTuple],
        ) -> list[CoverageVerdictLiteral]:
            return ["Covered" for _ in target]

    runner = CrossDocCoverageEvalRunner(verifier=_AlwaysCovered())
    result = runner.run_case(mixed)
    # A mixed-verdict case cannot be solved by always-Covered above 0.85.
    assert result.metrics["accuracy"] < CROSS_DOC_COVERAGE_THRESHOLD
    assert result.passed is False
    assert result.metrics["missing_recall"] == pytest.approx(0.0)


def test_runner_length_mismatch_raises() -> None:
    case = _cases()[0]

    @dataclass
    class _ShortVerifier:
        def verdicts(
            self,
            *,
            source: list[ClaimTuple],
            target: list[ClaimTuple],
        ) -> list[CoverageVerdictLiteral]:
            return ["Covered"]

    runner = CrossDocCoverageEvalRunner(verifier=_ShortVerifier())
    with pytest.raises(ValueError, match="verdicts for"):
        runner.run_case(case)


def test_runner_emits_accuracy_and_per_class_metrics() -> None:
    cases = _cases()
    case = cases[0]
    runner = CrossDocCoverageEvalRunner(
        verifier=_OracleVerifier(gold_by_case=_gold_lookup(cases), current_case_id=case.id)
    )
    result = runner.run_case(case)
    assert {
        "accuracy",
        "covered_precision",
        "covered_recall",
        "missing_precision",
        "missing_recall",
    }.issubset(result.metrics.keys())


# --- Dataset invariants ---


def test_dataset_has_exactly_twelve_cases() -> None:
    assert len(_cases()) == 12


def test_dataset_ids_are_unique() -> None:
    ids = [c.id for c in _cases()]
    assert len(ids) == len(set(ids))


def test_dataset_both_verdicts_present() -> None:
    seen: set[CoverageVerdictLiteral] = set()
    for case in _cases():
        for tc in case.target_claims:
            seen.add(tc.gold_verdict)
    assert seen == set(COVERAGE_VERDICTS)


def test_dataset_each_verdict_used_at_least_four_times() -> None:
    """A two-label space is easy to game; require both labels appear ≥ 4 times."""
    counts: dict[CoverageVerdictLiteral, int] = dict.fromkeys(COVERAGE_VERDICTS, 0)
    for case in _cases():
        for tc in case.target_claims:
            counts[tc.gold_verdict] += 1
    for verdict, count in counts.items():
        assert count >= 4, f"verdict {verdict!r} used {count} times (expected >= 4)"


def test_dataset_many_to_one_case_present() -> None:
    """At least one case exercises many-to-one transport (target jointly supported)."""
    assert any("many-to-one" in c.tags for c in _cases())


def test_dataset_empty_source_case_present() -> None:
    """At least one case has an empty source claim list."""
    assert any(not c.source_claims for c in _cases())


def test_dataset_qualifier_divergence_case_present() -> None:
    """At least one case exercises qualifier-based coverage discrimination."""
    assert any("qualifier-divergence" in c.tags for c in _cases())


def test_dataset_contradiction_case_present() -> None:
    """At least one case has a polarity-flip contradiction."""
    assert any("contradiction" in c.tags or "negation-mismatch" in c.tags for c in _cases())


def test_dataset_modality_ordering_cases_present() -> None:
    """Both stronger-covers-weaker and weaker-misses-stronger exist."""
    tags = {tag for c in _cases() for tag in c.tags}
    assert "stronger-than-covers-weaker" in tags
    assert "weaker-than-misses-stronger" in tags


def test_dataset_doc_type_diversity() -> None:
    """The 12 cases cover at least three distinct source doc types."""
    src_types = {c.source_doc_type for c in _cases()}
    assert len(src_types) >= 3


def test_dataset_every_id_has_ccov_prefix() -> None:
    for case in _cases():
        assert case.id.startswith("ccov-"), f"case id {case.id!r} should start with 'ccov-'"


def test_dataset_target_ids_unique_within_case() -> None:
    for case in _cases():
        ids = [tc.id for tc in case.target_claims]
        assert len(ids) == len(set(ids)), f"case {case.id!r} has duplicate target ids"


# --- End-to-end through the harness ---


def test_starter_set_passes_threshold_with_oracle_verifier() -> None:
    cases = _cases()
    gold_lookup = _gold_lookup(cases)

    @dataclass
    class _DispatchingRunner:
        def run_case(self, case: CrossDocCoverageEvalCase):  # type: ignore[no-untyped-def]
            return CrossDocCoverageEvalRunner(
                verifier=_OracleVerifier(gold_by_case=gold_lookup, current_case_id=case.id),
            ).run_case(case)

    report = run_eval(
        set_name="cross_doc_coverage_eval",
        cases=cases,
        runner=_DispatchingRunner(),
        thresholds={"accuracy": CROSS_DOC_COVERAGE_THRESHOLD},
    )
    assert report.passed is True
    assert report.aggregate["accuracy"] == pytest.approx(1.0)


def test_starter_set_fails_threshold_with_always_covered() -> None:
    cases = _cases()

    @dataclass
    class _AlwaysCovered:
        def verdicts(
            self,
            *,
            source: list[ClaimTuple],
            target: list[ClaimTuple],
        ) -> list[CoverageVerdictLiteral]:
            return ["Covered" for _ in target]

    @dataclass
    class _DispatchingRunner:
        def run_case(self, case: CrossDocCoverageEvalCase):  # type: ignore[no-untyped-def]
            return CrossDocCoverageEvalRunner(verifier=_AlwaysCovered()).run_case(case)

    report = run_eval(
        set_name="cross_doc_coverage_eval",
        cases=cases,
        runner=_DispatchingRunner(),
        thresholds={"accuracy": CROSS_DOC_COVERAGE_THRESHOLD},
    )
    assert report.passed is False
    assert report.aggregate["accuracy"] < CROSS_DOC_COVERAGE_THRESHOLD
