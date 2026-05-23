"""qa_refusal — dataset-level invariants for the 30-case refusal set.

Per §8.1 the qa_refusal set carries 30 out-of-doc questions; every
case must be refused by `QAPlaybook` to keep refusal accuracy at or
above the §8.2 0.90 threshold. The runner schema already lives in
`ctrldoc.eval.qa` (S-081); this dataset slice adds the cases and
asserts the structural invariants every refusal entry must hold.

SPEC-REF: §8.1 (qa_refusal), §8.2 (refusal accuracy)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from ctrldoc.eval.harness import load_jsonl_cases, run_eval
from ctrldoc.eval.qa import (
    REFUSAL_ACCURACY_THRESHOLD,
    QAEvalCase,
)
from ctrldoc.models import Claim

QA_REFUSAL_PATH = Path(__file__).parent / "eval" / "qa_refusal.jsonl"


def _cases() -> list[QAEvalCase]:
    return load_jsonl_cases(QA_REFUSAL_PATH, case_model=QAEvalCase)


# --- dataset invariants ---


def test_qa_refusal_set_has_thirty_cases() -> None:
    """Per §8.1: 30 (q where answer not in doc) cases."""
    cases = _cases()
    assert len(cases) == 30


def test_every_case_is_marked_should_refuse() -> None:
    for case in _cases():
        assert case.should_refuse, f"case {case.id!r} is not marked should_refuse"


def test_every_case_has_empty_gold_chunk_ids() -> None:
    """Refusal cases never carry gold citations — there's nothing to cite."""
    for case in _cases():
        assert case.gold_chunk_ids == [], (
            f"case {case.id!r} has gold chunk ids but is a refusal case"
        )


def test_case_ids_are_unique() -> None:
    cases = _cases()
    ids = [case.id for case in cases]
    assert len(set(ids)) == len(ids), "duplicate case ids in qa_refusal"


def test_every_question_is_non_blank() -> None:
    for case in _cases():
        assert case.question.strip(), f"case {case.id!r} has an empty question"


def test_every_case_carries_at_least_one_tag() -> None:
    """Tags drive triage and per-axis breakdowns in the regression report."""
    for case in _cases():
        assert case.tags, f"case {case.id!r} has no tags"


def test_every_case_tagged_out_of_doc() -> None:
    """The `out-of-doc` tag is the dataset's load-bearing semantic invariant."""
    for case in _cases():
        assert "out-of-doc" in case.tags, f"case {case.id!r} is missing the 'out-of-doc' tag"


def test_refusal_set_covers_diverse_axes() -> None:
    """The set should exercise many refusal axes, not the same one 30 times.

    The §8.2 0.90 threshold is meaningful only if the refusal questions
    span enough surface area that a system can't pass by memorising one
    refusal pattern."""
    all_tags: set[str] = set()
    for case in _cases():
        all_tags.update(case.tags)
    non_marker_tags = {tag for tag in all_tags if tag != "out-of-doc"}
    assert len(non_marker_tags) >= 10, (
        f"refusal set covers only {len(non_marker_tags)} axes: {sorted(non_marker_tags)}"
    )


# --- end-to-end through the harness ---


@dataclass
class _AlwaysRefuseRunner:
    """Stand-in for a refusal-correct playbook driver."""

    def run_case(self, case: QAEvalCase):  # type: ignore[no-untyped-def]
        from ctrldoc.eval.qa import QAEvalRunner
        from ctrldoc.playbooks.qa import AnswerReport

        # Build an AnswerReport that always carries a single unverified
        # claim — i.e. the playbook tried to answer but the verifier
        # refused. That's the "correct refusal" shape.
        @dataclass
        class _StubPlaybook:
            def run(self, question: str) -> AnswerReport:
                return AnswerReport(
                    query=question,
                    answer="The document does not state.",
                    claims=[
                        Claim(
                            text="I cannot determine that from the document.",
                            citations=[],
                            verified=False,
                            confidence=0.0,
                            nli_score=0.0,
                            judge_score=0.0,
                        )
                    ],
                )

        return QAEvalRunner(playbook=_StubPlaybook()).run_case(case)  # type: ignore[arg-type]


def test_refusal_set_passes_threshold_with_a_refusal_correct_playbook() -> None:
    """End-to-end sanity: a playbook that always refuses should hit 100%
    refusal accuracy on this set, well above the §8.2 0.90 floor."""
    report = run_eval(
        set_name="qa_refusal",
        cases=_cases(),
        runner=_AlwaysRefuseRunner(),
        thresholds={"refusal_accuracy": REFUSAL_ACCURACY_THRESHOLD},
    )
    assert report.passed is True
    assert report.aggregate["refusal_accuracy"] == pytest.approx(1.0)
    assert report.aggregate["pass_rate"] == pytest.approx(1.0)


def test_refusal_set_fails_threshold_with_a_fabricating_playbook() -> None:
    """If a playbook regression makes it answer one out-of-doc question,
    the harness must flag the regression."""

    @dataclass
    class _FabricatingRunner:
        def run_case(self, case: QAEvalCase):  # type: ignore[no-untyped-def]
            from ctrldoc.eval.qa import QAEvalRunner
            from ctrldoc.models import Span
            from ctrldoc.playbooks.qa import AnswerReport

            # Pretend the playbook fabricates a confident answer on every
            # case — every refusal case should now show 0.0 accuracy.
            @dataclass
            class _StubPlaybook:
                def run(self, question: str) -> AnswerReport:
                    return AnswerReport(
                        query=question,
                        answer="A confident but wrong answer.",
                        claims=[
                            Claim(
                                text="Confident fabricated claim.",
                                citations=[
                                    Span(
                                        chunk_id="c-fab",
                                        char_start=0,
                                        char_end=4,
                                        text="ev",
                                    )
                                ],
                                verified=True,
                                confidence=0.9,
                                nli_score=0.9,
                                judge_score=0.9,
                            )
                        ],
                    )

            return QAEvalRunner(playbook=_StubPlaybook()).run_case(case)  # type: ignore[arg-type]

    report = run_eval(
        set_name="qa_refusal",
        cases=_cases(),
        runner=_FabricatingRunner(),
        thresholds={"refusal_accuracy": REFUSAL_ACCURACY_THRESHOLD},
    )
    assert report.passed is False
    assert report.aggregate["refusal_accuracy"] == pytest.approx(0.0)
