"""claim_extraction_eval — universal claim tuple extractor scoring.

The runner takes a `ClaimExtractor` and grades it sentence-by-sentence
against gold tuples. The metric is set-based precision / recall / F1
on a "core match" of normalized subject / predicate / object plus
exact-match polarity and modality. Per §14 the release gate is
`claim_F1 >= 0.85`.

The starter dataset at `tests/eval/claim_extraction_eval.jsonl` ships
120 hand-curated cases — 20 each across six doc types (spec, runbook,
rfc, legal, academic, narrative) — exercising every modality, both
polarities, single- and multi-claim sentences, and the qualifier slot.

SPEC-REF: §6.2 (universal claim tuple), §14 (claim_F1 gate)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic import ValidationError

from ctrldoc.eval.claim_extraction import (
    CLAIM_F1_THRESHOLD,
    DOC_TYPES,
    MODALITIES,
    POLARITIES,
    ClaimExtractionEvalCase,
    ClaimExtractionEvalRunner,
    ClaimExtractor,
    ClaimTuple,
    claim_tuple_matches,
    normalize_text,
    precision_recall_f1,
)
from ctrldoc.eval.harness import load_jsonl_cases, run_eval

CLAIM_EVAL_PATH = Path(__file__).parent / "eval" / "claim_extraction_eval.jsonl"


def _cases() -> list[ClaimExtractionEvalCase]:
    return load_jsonl_cases(CLAIM_EVAL_PATH, case_model=ClaimExtractionEvalCase)


# --- ClaimTuple model contract ---


def test_claim_tuple_is_frozen() -> None:
    t = ClaimTuple(
        subject="x", predicate="is", object="y", polarity="affirmative", modality="asserted"
    )
    with pytest.raises(ValidationError):
        t.subject = "z"  # type: ignore[misc]


def test_claim_tuple_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ClaimTuple(
            subject="x",
            predicate="is",
            object="y",
            polarity="affirmative",
            modality="asserted",
            unknown="oops",  # type: ignore[call-arg]
        )


def test_claim_tuple_polarity_literal_enforced() -> None:
    with pytest.raises(ValidationError):
        ClaimTuple(
            subject="x",
            predicate="is",
            object="y",
            polarity="maybe",  # type: ignore[arg-type]
            modality="asserted",
        )


def test_claim_tuple_modality_literal_enforced() -> None:
    with pytest.raises(ValidationError):
        ClaimTuple(
            subject="x",
            predicate="is",
            object="y",
            polarity="affirmative",
            modality="opinion",  # type: ignore[arg-type]
        )


def test_claim_tuple_qualifier_defaults_to_empty() -> None:
    t = ClaimTuple(
        subject="x", predicate="is", object="y", polarity="affirmative", modality="asserted"
    )
    assert t.qualifier == ""


# --- normalize_text ---


def test_normalize_text_lowercases_and_strips() -> None:
    assert normalize_text("  The System  ") == "the system"


def test_normalize_text_strips_trailing_punctuation() -> None:
    assert normalize_text("the system.") == "the system"
    assert normalize_text("the system?") == "the system"
    assert normalize_text("the system!") == "the system"


def test_normalize_text_collapses_internal_whitespace() -> None:
    assert normalize_text("the   system   logs") == "the system logs"


def test_normalize_text_handles_empty() -> None:
    assert normalize_text("") == ""
    assert normalize_text("   ") == ""


# --- claim_tuple_matches ---


def _t(
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


def test_match_identical() -> None:
    assert claim_tuple_matches(extracted=_t(), gold=_t()) is True


def test_match_normalizes_subject_case() -> None:
    a = _t(subject="The System")
    b = _t(subject="the system.")
    assert claim_tuple_matches(extracted=a, gold=b) is True


def test_match_requires_polarity_exact() -> None:
    a = _t(polarity="affirmative")
    b = _t(polarity="negative")
    assert claim_tuple_matches(extracted=a, gold=b) is False


def test_match_requires_modality_exact() -> None:
    a = _t(modality="asserted")
    b = _t(modality="obligatory")
    assert claim_tuple_matches(extracted=a, gold=b) is False


def test_match_qualifier_ignored_when_gold_blank() -> None:
    a = _t(qualifier="always")
    b = _t(qualifier="")
    assert claim_tuple_matches(extracted=a, gold=b) is True


def test_match_qualifier_required_when_gold_set() -> None:
    a = _t(qualifier="")
    b = _t(qualifier="always")
    assert claim_tuple_matches(extracted=a, gold=b) is False


def test_match_qualifier_normalised() -> None:
    a = _t(qualifier="ALWAYS.")
    b = _t(qualifier="always")
    assert claim_tuple_matches(extracted=a, gold=b) is True


def test_match_predicate_or_object_differs() -> None:
    assert (
        claim_tuple_matches(extracted=_t(predicate="uses"), gold=_t(predicate="rejects")) is False
    )
    assert claim_tuple_matches(extracted=_t(object_="bm25"), gold=_t(object_="dense")) is False


# --- precision_recall_f1 ---


def test_prf1_all_correct() -> None:
    gold = [_t(), _t(subject="other")]
    extracted = [_t(), _t(subject="other")]
    prf = precision_recall_f1(extracted=extracted, gold=gold)
    assert prf == {
        "precision": pytest.approx(1.0),
        "recall": pytest.approx(1.0),
        "f1": pytest.approx(1.0),
    }


def test_prf1_no_extracted_is_zero() -> None:
    gold = [_t()]
    prf = precision_recall_f1(extracted=[], gold=gold)
    assert prf["precision"] == pytest.approx(0.0)
    assert prf["recall"] == pytest.approx(0.0)
    assert prf["f1"] == pytest.approx(0.0)


def test_prf1_empty_gold_returns_zero() -> None:
    prf = precision_recall_f1(extracted=[_t()], gold=[])
    assert prf == {
        "precision": pytest.approx(0.0),
        "recall": pytest.approx(0.0),
        "f1": pytest.approx(0.0),
    }


def test_prf1_partial_overlap() -> None:
    gold = [_t(subject="a"), _t(subject="b"), _t(subject="c")]
    extracted = [_t(subject="a"), _t(subject="b"), _t(subject="z")]
    prf = precision_recall_f1(extracted=extracted, gold=gold)
    assert prf["precision"] == pytest.approx(2 / 3)
    assert prf["recall"] == pytest.approx(2 / 3)
    assert prf["f1"] == pytest.approx(2 / 3)


def test_prf1_extracted_superset() -> None:
    gold = [_t(subject="a")]
    extracted = [_t(subject="a"), _t(subject="b")]
    prf = precision_recall_f1(extracted=extracted, gold=gold)
    # precision = 1/2 (one wrong), recall = 1/1
    assert prf["precision"] == pytest.approx(0.5)
    assert prf["recall"] == pytest.approx(1.0)
    assert prf["f1"] == pytest.approx(2 * 0.5 * 1.0 / (0.5 + 1.0))


def test_prf1_duplicate_extracted_only_counts_once() -> None:
    """A duplicate extraction does not inflate precision or recall."""
    gold = [_t(subject="a")]
    extracted = [_t(subject="a"), _t(subject="a")]
    prf = precision_recall_f1(extracted=extracted, gold=gold)
    # The second duplicate does not match an unmatched gold; it is a
    # false positive against the set.
    assert prf["precision"] == pytest.approx(0.5)
    assert prf["recall"] == pytest.approx(1.0)


def test_prf1_returns_three_metrics() -> None:
    prf = precision_recall_f1(extracted=[], gold=[])
    assert set(prf.keys()) == {"precision", "recall", "f1"}


# --- Protocol conformance ---


def test_oracle_extractor_satisfies_protocol() -> None:
    @dataclass
    class _Oracle:
        gold_by_sentence: dict[str, list[ClaimTuple]] = field(default_factory=dict)

        def extract(self, sentence: str) -> list[ClaimTuple]:
            return list(self.gold_by_sentence.get(sentence, []))

    assert isinstance(_Oracle(), ClaimExtractor)


# --- Runner ---


@dataclass
class _OracleExtractor:
    gold_by_sentence: dict[str, list[ClaimTuple]]

    def extract(self, sentence: str) -> list[ClaimTuple]:
        return list(self.gold_by_sentence.get(sentence, []))


def _gold_lookup(cases: list[ClaimExtractionEvalCase]) -> dict[str, list[ClaimTuple]]:
    return {c.sentence: list(c.gold_tuples) for c in cases}


def test_runner_oracle_clears_threshold_on_starter_case() -> None:
    case = _cases()[0]
    runner = ClaimExtractionEvalRunner(
        extractor=_OracleExtractor(gold_by_sentence={case.sentence: list(case.gold_tuples)})
    )
    result = runner.run_case(case)
    assert result.metrics["f1"] == pytest.approx(1.0)
    assert result.passed is True


def test_runner_empty_extractor_fails_threshold() -> None:
    case = _cases()[0]

    @dataclass
    class _Empty:
        def extract(self, sentence: str) -> list[ClaimTuple]:
            return []

    runner = ClaimExtractionEvalRunner(extractor=_Empty())
    result = runner.run_case(case)
    assert result.metrics["f1"] == pytest.approx(0.0)
    assert result.passed is False


def test_runner_wrong_modality_lowers_f1() -> None:
    case = _cases()[0]
    bad_tuples = [
        t.model_copy(update={"modality": "asserted" if t.modality != "asserted" else "obligatory"})
        for t in case.gold_tuples
    ]
    runner = ClaimExtractionEvalRunner(
        extractor=_OracleExtractor(gold_by_sentence={case.sentence: bad_tuples})
    )
    result = runner.run_case(case)
    assert result.metrics["f1"] == pytest.approx(0.0)
    assert result.passed is False


def test_runner_emits_precision_recall_and_f1() -> None:
    case = _cases()[0]
    runner = ClaimExtractionEvalRunner(
        extractor=_OracleExtractor(gold_by_sentence={case.sentence: list(case.gold_tuples)})
    )
    result = runner.run_case(case)
    assert {"precision", "recall", "f1"}.issubset(result.metrics.keys())


# --- ClaimExtractionEvalCase contract ---


def test_eval_case_rejects_unknown_doc_type() -> None:
    with pytest.raises(ValidationError):
        ClaimExtractionEvalCase(
            id="bad-1",
            doc_type="poetry",  # type: ignore[arg-type]
            sentence="x",
            gold_tuples=[_t()],
        )


def test_eval_case_rejects_blank_sentence() -> None:
    with pytest.raises(ValidationError):
        ClaimExtractionEvalCase(
            id="bad-2",
            doc_type="spec",
            sentence="   ",
            gold_tuples=[_t()],
        )


def test_eval_case_requires_at_least_one_gold_tuple() -> None:
    with pytest.raises(ValidationError):
        ClaimExtractionEvalCase(
            id="bad-3",
            doc_type="spec",
            sentence="x",
            gold_tuples=[],
        )


# --- Dataset invariants ---


def test_dataset_has_exactly_120_cases() -> None:
    assert len(_cases()) == 120


def test_dataset_has_20_cases_per_doc_type() -> None:
    cases = _cases()
    counts: dict[str, int] = dict.fromkeys(DOC_TYPES, 0)
    for case in cases:
        counts[case.doc_type] += 1
    for dt, count in counts.items():
        assert count == 20, f"doc_type {dt!r} has {count} cases (expected 20)"


def test_dataset_ids_are_unique() -> None:
    ids = [c.id for c in _cases()]
    assert len(ids) == len(set(ids))


def test_dataset_every_modality_used_at_least_five_times() -> None:
    cases = _cases()
    counts = dict.fromkeys(MODALITIES, 0)
    for case in cases:
        for t in case.gold_tuples:
            counts[t.modality] += 1
    for modality, count in counts.items():
        assert count >= 5, f"modality {modality!r} used {count} times (expected >= 5)"


def test_dataset_both_polarities_present_in_every_doc_type() -> None:
    cases = _cases()
    seen: dict[str, set[str]] = {dt: set() for dt in DOC_TYPES}
    for case in cases:
        for t in case.gold_tuples:
            seen[case.doc_type].add(t.polarity)
    for dt, polarities in seen.items():
        assert polarities == set(POLARITIES), (
            f"doc_type {dt!r} polarities = {polarities} (expected both)"
        )


def test_dataset_multi_claim_sentences_present() -> None:
    """At least 6 cases (1 per doc type on average) have >1 gold tuple."""
    multi = [c for c in _cases() if len(c.gold_tuples) > 1]
    assert len(multi) >= 6


def test_dataset_qualifier_slot_exercised() -> None:
    """At least 10 gold tuples have a non-empty qualifier."""
    qual_count = 0
    for case in _cases():
        for t in case.gold_tuples:
            if t.qualifier:
                qual_count += 1
    assert qual_count >= 10


def test_dataset_every_case_tagged_with_doc_type() -> None:
    for case in _cases():
        assert case.doc_type in case.tags, (
            f"case {case.id!r} missing doc_type tag {case.doc_type!r}"
        )


def test_dataset_every_id_has_doc_type_prefix() -> None:
    """A case's id starts with its doc_type slug — makes scan-by-eye fast."""
    for case in _cases():
        assert case.id.startswith(case.doc_type), (
            f"case id {case.id!r} should start with doc_type {case.doc_type!r}"
        )


# --- End-to-end through the harness ---


def test_starter_set_passes_threshold_with_oracle_extractor() -> None:
    cases = _cases()

    @dataclass
    class _DispatchingRunner:
        gold_lookup: dict[str, list[ClaimTuple]]

        def run_case(self, case: ClaimExtractionEvalCase):  # type: ignore[no-untyped-def]
            return ClaimExtractionEvalRunner(
                extractor=_OracleExtractor(gold_by_sentence=self.gold_lookup)
            ).run_case(case)

    report = run_eval(
        set_name="claim_extraction_eval",
        cases=cases,
        runner=_DispatchingRunner(gold_lookup=_gold_lookup(cases)),
        thresholds={"f1": CLAIM_F1_THRESHOLD},
    )
    assert report.passed is True
    assert report.aggregate["f1"] == pytest.approx(1.0)


def test_starter_set_fails_threshold_with_empty_extractor() -> None:
    cases = _cases()

    @dataclass
    class _Empty:
        def extract(self, sentence: str) -> list[ClaimTuple]:
            return []

    @dataclass
    class _DispatchingRunner:
        def run_case(self, case: ClaimExtractionEvalCase):  # type: ignore[no-untyped-def]
            return ClaimExtractionEvalRunner(extractor=_Empty()).run_case(case)

    report = run_eval(
        set_name="claim_extraction_eval",
        cases=cases,
        runner=_DispatchingRunner(),
        thresholds={"f1": CLAIM_F1_THRESHOLD},
    )
    assert report.passed is False
    assert report.aggregate["f1"] == pytest.approx(0.0)
