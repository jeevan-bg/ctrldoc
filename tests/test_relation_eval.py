"""relation_eval — runner + 3-case starter set scoring relation-type accuracy.

The runner stitches a stub extractor + case-local pair retriever
around the caller-supplied classifier so the eval is end-to-end
hermetic. Each gold pair is graded as correct iff the playbook
emitted an edge with the matching type — or, for `gold_type=None`
pairs, iff no edge was emitted.

SPEC-REF: §8.1 (relation_eval), §8.2 (relation_map metrics)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic import ValidationError

from ctrldoc.eval.harness import load_jsonl_cases, run_eval
from ctrldoc.eval.relation import (
    RELATION_TYPE_ACCURACY_THRESHOLD,
    GoldPair,
    RelationEvalCase,
    RelationEvalRunner,
    pair_key,
    relation_type_accuracy,
)
from ctrldoc.models import EvidencePack, Span
from ctrldoc.ops.map import (
    Concept,
    RelationClassification,
)

RELATION_EVAL_PATH = Path(__file__).parent / "eval" / "relation_eval.jsonl"


def _cases() -> list[RelationEvalCase]:
    return load_jsonl_cases(RELATION_EVAL_PATH, case_model=RelationEvalCase)


# --- pair_key ---


def test_pair_key_canonicalises_direction() -> None:
    assert pair_key("a", "b") == pair_key("b", "a")


def test_pair_key_distinct_for_different_pairs() -> None:
    assert pair_key("a", "b") != pair_key("a", "c")


# --- relation_type_accuracy helper ---


def test_accuracy_all_correct_returns_one() -> None:
    emitted = [
        ("a", "b", "depends_on"),
        ("b", "c", "refines"),
    ]
    gold = [
        GoldPair(src_concept_id="a", dst_concept_id="b", gold_type="depends_on"),
        GoldPair(src_concept_id="b", dst_concept_id="c", gold_type="refines"),
    ]
    assert relation_type_accuracy(emitted, gold) == pytest.approx(1.0)


def test_accuracy_handles_direction_canonicalisation() -> None:
    """Emitted edge in reverse direction should still count as a match."""
    emitted = [("b", "a", "depends_on")]
    gold = [GoldPair(src_concept_id="a", dst_concept_id="b", gold_type="depends_on")]
    assert relation_type_accuracy(emitted, gold) == pytest.approx(1.0)


def test_accuracy_unrelated_gold_correct_when_no_edge_emitted() -> None:
    emitted: list[tuple[str, str, str]] = []  # type: ignore[var-annotated]
    gold = [GoldPair(src_concept_id="a", dst_concept_id="b", gold_type=None)]
    assert relation_type_accuracy(emitted, gold) == pytest.approx(1.0)  # type: ignore[arg-type]


def test_accuracy_unrelated_gold_wrong_when_edge_emitted() -> None:
    emitted = [("a", "b", "depends_on")]
    gold = [GoldPair(src_concept_id="a", dst_concept_id="b", gold_type=None)]
    assert relation_type_accuracy(emitted, gold) == pytest.approx(0.0)


def test_accuracy_wrong_type_counts_as_incorrect() -> None:
    emitted = [("a", "b", "contradicts")]
    gold = [GoldPair(src_concept_id="a", dst_concept_id="b", gold_type="depends_on")]
    assert relation_type_accuracy(emitted, gold) == pytest.approx(0.0)


def test_accuracy_partial_match() -> None:
    emitted = [
        ("a", "b", "depends_on"),
        ("b", "c", "contradicts"),  # wrong type
    ]
    gold = [
        GoldPair(src_concept_id="a", dst_concept_id="b", gold_type="depends_on"),
        GoldPair(src_concept_id="b", dst_concept_id="c", gold_type="refines"),
        GoldPair(src_concept_id="a", dst_concept_id="c", gold_type=None),
    ]
    # a-b correct, b-c wrong, a-c correct (no edge, none expected) → 2/3
    assert relation_type_accuracy(emitted, gold) == pytest.approx(2 / 3)


def test_accuracy_empty_gold_returns_zero() -> None:
    assert relation_type_accuracy([], []) == pytest.approx(0.0)


# --- dataset invariants ---


def test_relation_eval_set_has_three_cases() -> None:
    assert len(_cases()) == 3


def test_every_case_has_concepts_and_gold_pairs() -> None:
    for case in _cases():
        assert case.concepts, f"case {case.id!r} has no concepts"
        assert case.gold_pairs, f"case {case.id!r} has no gold pairs"


def test_gold_pairs_reference_known_concepts() -> None:
    """Schema invariant enforced by the model validator."""
    with pytest.raises(ValidationError, match="not in concept list"):
        RelationEvalCase(
            id="bad-1",
            concepts=[Concept(id="a", name="A")],
            evidence_by_pair={},
            gold_pairs=[
                GoldPair(src_concept_id="a", dst_concept_id="missing", gold_type="depends_on"),
            ],
        )


def test_self_pair_rejected() -> None:
    with pytest.raises(ValidationError, match="identical src and dst"):
        RelationEvalCase(
            id="bad-2",
            concepts=[Concept(id="a", name="A")],
            evidence_by_pair={},
            gold_pairs=[
                GoldPair(src_concept_id="a", dst_concept_id="a", gold_type="depends_on"),
            ],
        )


def test_duplicate_pair_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate gold pair"):
        RelationEvalCase(
            id="bad-3",
            concepts=[
                Concept(id="a", name="A"),
                Concept(id="b", name="B"),
            ],
            evidence_by_pair={},
            gold_pairs=[
                GoldPair(src_concept_id="a", dst_concept_id="b", gold_type="depends_on"),
                # Same pair in reverse direction → same canonical key.
                GoldPair(src_concept_id="b", dst_concept_id="a", gold_type="refines"),
            ],
        )


# --- runner: oracle classifier ---


def _evidence_pack_has_spans(pack: EvidencePack) -> bool:
    return bool(pack.spans)


@dataclass
class _OracleClassifier:
    """Returns the gold relation type for every queried pair, using
    a caller-supplied gold lookup keyed by canonical pair key."""

    gold_by_key: dict[str, str | None]
    calls: list[tuple[str, str]] = field(default_factory=list)

    def classify(
        self,
        c_i: Concept,
        c_j: Concept,
        evidence: EvidencePack,
    ) -> RelationClassification | None:
        self.calls.append((c_i.id, c_j.id))
        key = pair_key(c_i.id, c_j.id)
        if key not in self.gold_by_key:
            return None
        relation_type = self.gold_by_key[key]
        if relation_type is None:
            return None
        return RelationClassification(
            type=relation_type,  # type: ignore[arg-type]
            citations=[
                Span(
                    chunk_id=evidence.spans[0].chunk_id,
                    char_start=0,
                    char_end=1,
                    text=evidence.spans[0].text[:1],
                )
            ]
            if _evidence_pack_has_spans(evidence)
            else [],
            confidence=0.9,
        )


def _gold_lookup(case: RelationEvalCase) -> dict[str, str | None]:
    return {pair_key(g.src_concept_id, g.dst_concept_id): g.gold_type for g in case.gold_pairs}


def test_runner_oracle_classifier_clears_threshold() -> None:
    case = _cases()[0]
    runner = RelationEvalRunner(classifier=_OracleClassifier(gold_by_key=_gold_lookup(case)))
    result = runner.run_case(case)
    assert result.metrics["relation_type_accuracy"] == pytest.approx(1.0)
    assert result.passed is True


def test_runner_skips_pair_when_no_evidence_to_classify() -> None:
    """A pair with no evidence in the case never reaches the classifier;
    if the gold says "unrelated" that's a correct skip."""

    @dataclass
    class _CountingClassifier:
        calls: int = 0

        def classify(self, c_i, c_j, evidence):  # type: ignore[no-untyped-def]
            self.calls += 1
            return None

    case = RelationEvalCase(
        id="r-skip",
        concepts=[
            Concept(id="a", name="A"),
            Concept(id="b", name="B"),
        ],
        evidence_by_pair={},  # no evidence anywhere
        gold_pairs=[
            GoldPair(src_concept_id="a", dst_concept_id="b", gold_type=None),
        ],
    )
    classifier = _CountingClassifier()
    runner = RelationEvalRunner(classifier=classifier)
    result = runner.run_case(case)
    # Empty evidence ⇒ playbook skipped classifier; gold says unrelated ⇒ correct.
    assert classifier.calls == 0
    assert result.metrics["relation_type_accuracy"] == pytest.approx(1.0)


def test_runner_wrong_classification_lowers_score() -> None:
    case = _cases()[0]
    # Build a "wrong-on-everything" gold by mapping each pair to the
    # wrong type ("contradicts" for everything).
    wrong_gold = dict.fromkeys(_gold_lookup(case), "contradicts")
    runner = RelationEvalRunner(classifier=_OracleClassifier(gold_by_key=wrong_gold))
    result = runner.run_case(case)
    # Every pair has a gold type but the classifier emits "contradicts"
    # ⇒ 0/3 correct.
    assert result.metrics["relation_type_accuracy"] == pytest.approx(0.0)
    assert result.passed is False


# --- end-to-end via the harness ---


def test_starter_set_passes_threshold_with_oracle_classifier() -> None:
    @dataclass
    class _DispatchingRunner:
        def run_case(self, case: RelationEvalCase):  # type: ignore[no-untyped-def]
            return RelationEvalRunner(
                classifier=_OracleClassifier(gold_by_key=_gold_lookup(case))
            ).run_case(case)

    report = run_eval(
        set_name="relation_eval",
        cases=_cases(),
        runner=_DispatchingRunner(),
        thresholds={"relation_type_accuracy": RELATION_TYPE_ACCURACY_THRESHOLD},
    )
    assert report.passed is True
    assert report.aggregate["relation_type_accuracy"] == pytest.approx(1.0)


def test_starter_set_fails_threshold_with_never_classify_classifier() -> None:
    """A classifier that always returns None means every related-pair
    gold counts as wrong (no edge emitted)."""

    @dataclass
    class _NeverClassifier:
        def classify(self, c_i, c_j, evidence):  # type: ignore[no-untyped-def]
            return None

    @dataclass
    class _DispatchingRunner:
        def run_case(self, case: RelationEvalCase):  # type: ignore[no-untyped-def]
            return RelationEvalRunner(classifier=_NeverClassifier()).run_case(case)

    report = run_eval(
        set_name="relation_eval",
        cases=_cases(),
        runner=_DispatchingRunner(),
        thresholds={"relation_type_accuracy": RELATION_TYPE_ACCURACY_THRESHOLD},
    )
    assert report.passed is False
    # The aggregate is the mean of per-case accuracies (not pooled
    # across all pairs). rel-001 has zero `gold_type=None` entries
    # → 0/3 correct. rel-002 and rel-003 each have one
    # `gold_type=None` entry → 1/3 correct. Mean: (0 + 1/3 + 1/3)/3.
    assert report.aggregate["relation_type_accuracy"] == pytest.approx(2 / 9)
