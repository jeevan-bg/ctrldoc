"""Per-playbook cost / latency baselines and regression gates.

The §8.4 table fixes nominal cost + wall-clock targets for the
five playbooks (qa, coverage_audit, quality_audit,
analytical_review, anomaly_scan). The §8.6 family-11 family adds
two more universal gates: every LLM call must stay under 16k input
tokens, and the prompt-cache hit rate must be ≥ 90% on cacheable
calls.

`aggregate_stats(records, playbook)` rolls a `TraceRecord` stream
into a `PlaybookStats` summary. `check_against_baseline(stats,
baseline)` returns a list of `BaselineViolation`s — empty when the
run is healthy. CI fails the build when any violation crosses the
20% regression tolerance.

The baselines themselves are versioned constants in `SPEC_BASELINES`;
they're the canonical source of truth for the §8.4 table. Updating
them is an explicit commit-time act.

SPEC-REF: §8.4, §8.6 family 11
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, NonNegativeInt

from ctrldoc.trace import TraceRecord

DEFAULT_MAX_INPUT_TOKENS_PER_CALL = 16_000
DEFAULT_MIN_CACHE_HIT_RATE = 0.90
DEFAULT_REGRESSION_TOLERANCE = 0.20


class PlaybookBaseline(BaseModel):
    """Cost + latency + token + cache budget for one playbook."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    playbook: str
    max_cost_usd: NonNegativeFloat
    max_wall_clock_s: NonNegativeFloat
    max_input_tokens_per_call: NonNegativeInt = Field(
        default=DEFAULT_MAX_INPUT_TOKENS_PER_CALL,
    )
    min_cache_hit_rate: float = Field(default=DEFAULT_MIN_CACHE_HIT_RATE, ge=0.0, le=1.0)


# §8.4 table — canonical.
SPEC_BASELINES: dict[str, PlaybookBaseline] = {
    "qa": PlaybookBaseline(
        playbook="qa",
        max_cost_usd=0.10,
        max_wall_clock_s=30.0,
    ),
    "coverage_audit": PlaybookBaseline(
        playbook="coverage_audit",
        max_cost_usd=5.0,
        max_wall_clock_s=300.0,
    ),
    "quality_audit": PlaybookBaseline(
        playbook="quality_audit",
        max_cost_usd=3.0,
        max_wall_clock_s=180.0,
    ),
    "analytical_review": PlaybookBaseline(
        playbook="analytical_review",
        max_cost_usd=10.0,
        max_wall_clock_s=600.0,
    ),
    "anomaly_scan": PlaybookBaseline(
        playbook="anomaly_scan",
        max_cost_usd=2.0,
        max_wall_clock_s=120.0,
    ),
}


class PlaybookStats(BaseModel):
    """Aggregated metrics for one playbook's call stream."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    playbook: str
    n_calls: NonNegativeInt
    total_cost_usd: NonNegativeFloat
    total_wall_clock_s: NonNegativeFloat
    max_tokens_in: NonNegativeInt
    cache_hits: NonNegativeInt
    cacheable_calls: NonNegativeInt

    @property
    def cache_hit_rate(self) -> float:
        """Hits / cacheable_calls; 1.0 when there are zero cacheable calls.

        The "zero cacheable calls" convention is generous on purpose:
        a playbook that legitimately makes no Anthropic calls (all
        local) has no cache to miss, and forcing the metric to 0.0
        would penalise it for that.
        """
        if self.cacheable_calls == 0:
            return 1.0
        return self.cache_hits / self.cacheable_calls


def aggregate_stats(records: Iterable[TraceRecord], *, playbook: str) -> PlaybookStats:
    """Roll a `TraceRecord` stream into a `PlaybookStats` for one playbook.

    Records belonging to other playbooks are ignored. Cacheable calls
    are inferred as those that *could* hit cache (any record where
    `cache_hit` is True or False — i.e. all of them; non-cacheable
    backends would mark `cache_hit=False` and still count toward the
    denominator). In practice every LLM-backed playbook call is
    treated as cacheable for the §8.6 family-11 hit-rate gate.
    """
    n_calls = 0
    total_cost = 0.0
    total_latency_ms = 0
    max_tokens_in = 0
    hits = 0
    cacheable = 0
    for record in records:
        if record.playbook != playbook:
            continue
        n_calls += 1
        total_cost += record.cost_usd
        total_latency_ms += record.latency_ms
        max_tokens_in = max(max_tokens_in, record.tokens_in)
        cacheable += 1
        if record.cache_hit:
            hits += 1
    return PlaybookStats(
        playbook=playbook,
        n_calls=n_calls,
        total_cost_usd=total_cost,
        total_wall_clock_s=total_latency_ms / 1000.0,
        max_tokens_in=max_tokens_in,
        cache_hits=hits,
        cacheable_calls=cacheable,
    )


@dataclass(frozen=True)
class BaselineViolation:
    """One gate that the run failed to clear."""

    metric: str
    baseline: float
    actual: float
    tolerance: float
    reason: str


def check_against_baseline(
    stats: PlaybookStats,
    baseline: PlaybookBaseline,
    *,
    regression_tolerance: float = DEFAULT_REGRESSION_TOLERANCE,
) -> list[BaselineViolation]:
    """Return one violation per gate the run failed.

    Cost and wall-clock gates allow up to `regression_tolerance`
    headroom over the baseline (default 20%, matching §8.4). Token
    cap and cache hit-rate gates are absolute — they don't get the
    regression budget because they're invariants, not perf targets.
    """
    violations: list[BaselineViolation] = []
    if stats.playbook != baseline.playbook:
        raise ValueError(
            f"playbook mismatch: stats={stats.playbook!r} baseline={baseline.playbook!r}"
        )

    cost_cap = baseline.max_cost_usd * (1.0 + regression_tolerance)
    if stats.total_cost_usd > cost_cap:
        violations.append(
            BaselineViolation(
                metric="cost_usd",
                baseline=baseline.max_cost_usd,
                actual=stats.total_cost_usd,
                tolerance=regression_tolerance,
                reason=f"total cost {stats.total_cost_usd:.3f} > cap {cost_cap:.3f}",
            )
        )

    latency_cap = baseline.max_wall_clock_s * (1.0 + regression_tolerance)
    if stats.total_wall_clock_s > latency_cap:
        violations.append(
            BaselineViolation(
                metric="wall_clock_s",
                baseline=baseline.max_wall_clock_s,
                actual=stats.total_wall_clock_s,
                tolerance=regression_tolerance,
                reason=(
                    f"total wall clock {stats.total_wall_clock_s:.1f}s > cap {latency_cap:.1f}s"
                ),
            )
        )

    if stats.max_tokens_in > baseline.max_input_tokens_per_call:
        violations.append(
            BaselineViolation(
                metric="tokens_per_call",
                baseline=float(baseline.max_input_tokens_per_call),
                actual=float(stats.max_tokens_in),
                tolerance=0.0,
                reason=(
                    f"max input tokens {stats.max_tokens_in} > "
                    f"per-call cap {baseline.max_input_tokens_per_call}"
                ),
            )
        )

    if stats.cache_hit_rate < baseline.min_cache_hit_rate:
        violations.append(
            BaselineViolation(
                metric="cache_hit_rate",
                baseline=baseline.min_cache_hit_rate,
                actual=stats.cache_hit_rate,
                tolerance=0.0,
                reason=(
                    f"cache hit rate {stats.cache_hit_rate:.3f} < "
                    f"floor {baseline.min_cache_hit_rate:.3f}"
                ),
            )
        )

    return violations


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
