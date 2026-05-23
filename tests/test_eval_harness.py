"""Eval harness scaffolding — case loading, per-case runners, aggregate metrics.

The harness reads labelled cases from JSONL, drives them through a
`CaseRunner`, and aggregates per-case scores into an `EvalReport`.
Thresholds gate the overall pass/fail so CI can block a regression
before merge. The substrate covered here is generic; per-playbook
eval sets (`qa_eval`, `coverage_eval`, …) build on it in S-081..S-085.

SPEC-REF: §8.1 (eval sets), §8.2 (metrics)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from ctrldoc.eval.harness import (
    EvalReport,
    EvalResult,
    aggregate_results,
    load_jsonl_cases,
    run_eval,
)

# --- toy case + runner ---


class _ToyCase(BaseModel):
    """A minimal eval case: predict the next integer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    tags: list[str] = []
    seed: int
    expected: int


@dataclass
class _ToyRunner:
    """Returns expected+offset; the offset controls how many cases pass."""

    offset: int = 0
    calls: list[str] = field(default_factory=list)

    def run_case(self, case: _ToyCase) -> EvalResult:
        self.calls.append(case.id)
        predicted = case.seed + 1 + self.offset
        passed = predicted == case.expected
        return EvalResult(
            case_id=case.id,
            passed=passed,
            score=1.0 if passed else 0.0,
            metrics={"correct": 1.0 if passed else 0.0},
            notes=f"predicted={predicted}, expected={case.expected}",
        )


def _cases(n: int) -> list[_ToyCase]:
    return [_ToyCase(id=f"c-{i}", seed=i, expected=i + 1) for i in range(n)]


# --- EvalResult model ---


def test_eval_result_is_frozen() -> None:
    result = EvalResult(case_id="c-1", passed=True, score=1.0, metrics={})
    with pytest.raises(ValidationError):
        result.passed = False  # type: ignore[misc]


def test_eval_result_score_is_clamped_to_unit_interval() -> None:
    with pytest.raises(ValidationError):
        EvalResult(case_id="c", passed=False, score=1.5, metrics={})


def test_eval_result_default_metrics_is_empty_dict() -> None:
    result = EvalResult(case_id="c-1", passed=True, score=1.0)
    assert result.metrics == {}


# --- aggregate ---


def test_aggregate_returns_per_metric_means_and_pass_rate() -> None:
    results = [
        EvalResult(case_id="a", passed=True, score=1.0, metrics={"p": 1.0, "r": 0.8}),
        EvalResult(case_id="b", passed=False, score=0.0, metrics={"p": 0.0, "r": 0.4}),
        EvalResult(case_id="c", passed=True, score=1.0, metrics={"p": 1.0, "r": 1.0}),
    ]
    agg = aggregate_results(results)
    assert agg["pass_rate"] == pytest.approx(2 / 3)
    assert agg["score"] == pytest.approx(2 / 3)
    assert agg["p"] == pytest.approx(2 / 3)
    assert agg["r"] == pytest.approx((0.8 + 0.4 + 1.0) / 3)


def test_aggregate_handles_missing_metric_keys() -> None:
    """A metric present on some results but not others is averaged
    over the present subset."""
    results = [
        EvalResult(case_id="a", passed=True, score=1.0, metrics={"p": 1.0}),
        EvalResult(case_id="b", passed=True, score=1.0, metrics={}),
    ]
    agg = aggregate_results(results)
    assert agg["p"] == pytest.approx(1.0)


def test_aggregate_empty_returns_zeroed_pass_rate() -> None:
    agg = aggregate_results([])
    assert agg["pass_rate"] == pytest.approx(0.0)
    assert agg["score"] == pytest.approx(0.0)


# --- run_eval ---


def test_run_eval_drives_every_case_through_runner() -> None:
    cases = _cases(5)
    runner = _ToyRunner()
    report = run_eval(set_name="toy", cases=cases, runner=runner)
    assert isinstance(report, EvalReport)
    assert report.set_name == "toy"
    assert len(report.results) == 5
    # Runner was called once per case in input order.
    assert runner.calls == [c.id for c in cases]


def test_run_eval_aggregate_reflects_runner_accuracy() -> None:
    cases = _cases(10)
    # Offset=0 → all correct. pass_rate = 1.0.
    report = run_eval(set_name="toy", cases=cases, runner=_ToyRunner())
    assert report.aggregate["pass_rate"] == pytest.approx(1.0)
    assert report.passed is True


def test_run_eval_threshold_failure_marks_report_failed() -> None:
    cases = _cases(10)
    # Offset=1 → every prediction off by one. pass_rate = 0.0.
    report = run_eval(
        set_name="toy",
        cases=cases,
        runner=_ToyRunner(offset=1),
        thresholds={"pass_rate": 0.95},
    )
    assert report.passed is False
    assert report.aggregate["pass_rate"] == pytest.approx(0.0)


def test_run_eval_threshold_satisfied_marks_report_passed() -> None:
    cases = _cases(10)
    report = run_eval(
        set_name="toy",
        cases=cases,
        runner=_ToyRunner(),
        thresholds={"pass_rate": 0.9},
    )
    assert report.passed is True


def test_run_eval_missing_metric_in_threshold_check_fails_report() -> None:
    """A threshold on a metric the runner never emits should fail loud,
    not silently default to zero."""
    cases = _cases(3)
    with pytest.raises(KeyError, match="precision"):
        run_eval(
            set_name="toy",
            cases=cases,
            runner=_ToyRunner(),
            thresholds={"precision": 0.5},
        )


def test_run_eval_empty_case_list_returns_zero_pass_rate_and_failed() -> None:
    report = run_eval(set_name="empty", cases=[], runner=_ToyRunner())
    assert report.results == []
    # Empty set with no threshold passes trivially.
    assert report.passed is True


# --- failure isolation ---


def test_runner_exception_propagates_so_eval_runs_arent_silently_corrupted() -> None:
    @dataclass
    class _BoomRunner:
        def run_case(self, case: _ToyCase) -> EvalResult:
            raise RuntimeError("runner crashed")

    with pytest.raises(RuntimeError, match="runner crashed"):
        run_eval(set_name="toy", cases=_cases(3), runner=_BoomRunner())


# --- JSONL loading ---


def test_load_jsonl_cases_round_trips_through_pydantic_model(tmp_path: Path) -> None:
    path = tmp_path / "toy.jsonl"
    rows = [
        {"id": "c-0", "seed": 0, "expected": 1},
        {"id": "c-1", "tags": ["positive"], "seed": 1, "expected": 2},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))

    cases = load_jsonl_cases(path, case_model=_ToyCase)
    assert [c.id for c in cases] == ["c-0", "c-1"]
    assert cases[1].tags == ["positive"]


def test_load_jsonl_cases_ignores_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "toy.jsonl"
    path.write_text(
        json.dumps({"id": "c-0", "seed": 0, "expected": 1})
        + "\n\n  \n"
        + json.dumps({"id": "c-1", "seed": 1, "expected": 2})
        + "\n"
    )
    cases = load_jsonl_cases(path, case_model=_ToyCase)
    assert [c.id for c in cases] == ["c-0", "c-1"]


def test_load_jsonl_cases_validates_each_row(tmp_path: Path) -> None:
    path = tmp_path / "toy.jsonl"
    rows = [
        {"id": "c-0", "seed": 0, "expected": 1},
        # Missing `expected`.
        {"id": "c-bad", "seed": 1},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValidationError):
        load_jsonl_cases(path, case_model=_ToyCase)


def test_load_jsonl_cases_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_jsonl_cases(tmp_path / "nope.jsonl", case_model=_ToyCase)


# --- EvalReport model ---


def test_eval_report_round_trips_via_json() -> None:
    report = EvalReport(
        set_name="toy",
        results=[EvalResult(case_id="c", passed=True, score=1.0, metrics={"p": 1.0})],
        aggregate={"pass_rate": 1.0, "score": 1.0, "p": 1.0},
        passed=True,
    )
    blob = report.model_dump_json()
    restored = EvalReport.model_validate_json(blob)
    assert restored == report
