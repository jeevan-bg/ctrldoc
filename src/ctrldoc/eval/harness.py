"""Eval harness substrate.

The harness loads labelled cases from JSONL, drives them through a
`CaseRunner`, aggregates per-case scores into an `EvalReport`, and
gates the overall pass/fail against caller-supplied thresholds.

The runner is dependency-injected — a per-playbook subclass drives
the underlying primitives — so the same harness can grade every
playbook in §8.1 with no harness-side changes.

SPEC-REF: §8.1 (eval sets), §8.2 (per-playbook metrics)
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ctrldoc.models import UnitInterval

T = TypeVar("T", bound=BaseModel)


class EvalResult(BaseModel):
    """One case's verdict from the runner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    passed: bool
    score: UnitInterval
    metrics: dict[str, float] = Field(default_factory=dict)
    notes: str = ""


class EvalReport(BaseModel):
    """Aggregate report from one eval run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    set_name: str
    results: list[EvalResult]
    aggregate: dict[str, float]
    passed: bool


@runtime_checkable
class CaseRunner(Protocol):
    """Drives one playbook against one case, returns its `EvalResult`."""

    def run_case(self, case: Any) -> EvalResult: ...


def aggregate_results(results: list[EvalResult]) -> dict[str, float]:
    """Compute aggregate metrics across per-case results.

    Always emits `pass_rate` (fraction of cases marked `passed=True`)
    and `score` (mean of per-case `score`). Other metrics are averaged
    over the subset of results that include them — a metric absent
    from some cases is not penalised by zeros.
    """
    aggregate: dict[str, float] = {}
    if not results:
        return {"pass_rate": 0.0, "score": 0.0}

    aggregate["pass_rate"] = sum(1 for r in results if r.passed) / len(results)
    aggregate["score"] = sum(r.score for r in results) / len(results)

    counts: dict[str, int] = {}
    totals: dict[str, float] = {}
    for result in results:
        for key, value in result.metrics.items():
            counts[key] = counts.get(key, 0) + 1
            totals[key] = totals.get(key, 0.0) + value
    for key, total in totals.items():
        aggregate[key] = total / counts[key]
    return aggregate


def run_eval(
    *,
    set_name: str,
    cases: list[Any],
    runner: CaseRunner,
    thresholds: Mapping[str, float] | None = None,
) -> EvalReport:
    """Drive `runner` over every case and assemble the report.

    Thresholds gate the overall `passed` flag: every named metric must
    meet or exceed its threshold for the report to pass. A threshold
    referencing a metric the runner never emits is a configuration bug;
    we raise `KeyError` so it surfaces immediately rather than silently
    defaulting.
    """
    results = [runner.run_case(case) for case in cases]
    aggregate = aggregate_results(results)

    passed = True
    if thresholds:
        for metric_name, threshold in thresholds.items():
            if metric_name not in aggregate:
                raise KeyError(
                    f"threshold names metric {metric_name!r} which the runner did not emit"
                )
            if aggregate[metric_name] < threshold:
                passed = False
                break

    return EvalReport(set_name=set_name, results=results, aggregate=aggregate, passed=passed)


def load_jsonl_cases(path: Path, *, case_model: type[T]) -> list[T]:
    """Load and validate one case per non-blank line in a JSONL file."""
    if not path.exists():
        raise FileNotFoundError(f"eval case file not found: {path}")
    cases: list[T] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        cases.append(case_model.model_validate(json.loads(line)))
    return cases


__all__ = [
    "CaseRunner",
    "EvalReport",
    "EvalResult",
    "aggregate_results",
    "load_jsonl_cases",
    "run_eval",
]
