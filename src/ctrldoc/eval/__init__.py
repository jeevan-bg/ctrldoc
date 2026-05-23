"""Evaluation harness — drives playbooks against labelled cases."""

from __future__ import annotations

from ctrldoc.eval.harness import (
    CaseRunner,
    EvalReport,
    EvalResult,
    aggregate_results,
    load_jsonl_cases,
    run_eval,
)

__all__ = [
    "CaseRunner",
    "EvalReport",
    "EvalResult",
    "aggregate_results",
    "load_jsonl_cases",
    "run_eval",
]
