"""Performance + cost gating helpers.

Per-playbook latency and cost budgets, per-call token caps, and
prompt-cache hit-rate floors are aggregated from `TraceRecord`
streams (S-006) and checked against the §8.4 baseline table.

SPEC-REF: §8.4, §8.6 family 11
"""

from __future__ import annotations

from ctrldoc.perf.baseline import (
    DEFAULT_MAX_INPUT_TOKENS_PER_CALL,
    DEFAULT_MIN_CACHE_HIT_RATE,
    DEFAULT_REGRESSION_TOLERANCE,
    SPEC_BASELINES,
    BaselineViolation,
    PlaybookBaseline,
    PlaybookStats,
    aggregate_stats,
    check_against_baseline,
)

__all__ = [
    "DEFAULT_MAX_INPUT_TOKENS_PER_CALL",
    "DEFAULT_MIN_CACHE_HIT_RATE",
    "DEFAULT_REGRESSION_TOLERANCE",
    "SPEC_BASELINES",
    "BaselineViolation",
    "PlaybookBaseline",
    "PlaybookStats",
    "aggregate_stats",
    "check_against_baseline",
]
