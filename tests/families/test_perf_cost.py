"""Family-11 invariants — cost / latency / token / cache gates.

The §8.4 table pins per-playbook cost + wall-clock targets; §8.6
family 11 adds two universal gates (≤16k input tokens per call,
≥90% cache hit rate). This family asserts the gate machinery:

  - `SPEC_BASELINES` matches the §8.4 numbers verbatim.
  - `aggregate_stats` rolls a `TraceRecord` stream into the right
    cost / latency / token / cache totals, ignoring foreign
    playbooks.
  - `check_against_baseline` returns the right violations at the
    boundary (no headroom abuse on cost / latency, hard caps on
    tokens / cache hit rate).
  - Empty traces are well-defined (zero stats, vacuous-pass cache
    rate).

SPEC-REF: §8.4, §8.6 family 11
"""

from __future__ import annotations

import pytest

from ctrldoc.perf.baseline import (
    DEFAULT_MAX_INPUT_TOKENS_PER_CALL,
    DEFAULT_MIN_CACHE_HIT_RATE,
    DEFAULT_REGRESSION_TOLERANCE,
    SPEC_BASELINES,
    PlaybookBaseline,
    aggregate_stats,
    check_against_baseline,
)
from ctrldoc.trace import TraceRecord


def _record(
    *,
    playbook: str = "qa",
    cost_usd: float = 0.01,
    latency_ms: int = 100,
    tokens_in: int = 100,
    tokens_out: int = 50,
    cache_hit: bool = False,
    task_id: str = "t-1",
) -> TraceRecord:
    return TraceRecord(
        run_id="run-1",
        task_id=task_id,
        playbook=playbook,
        model="claude-opus-4-7",
        prompt_hash="h",
        response_hash="r",
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        cache_hit=cache_hit,
        error=None,
    )


# --- §8.4 baselines table ---


@pytest.mark.family_performance_cost
def test_spec_baselines_match_section_84_table() -> None:
    """The pinned constants are the canonical §8.4 numbers; this test
    is the canary that protects against silent edits to the table."""
    assert SPEC_BASELINES["qa"].max_cost_usd == pytest.approx(0.10)
    assert SPEC_BASELINES["qa"].max_wall_clock_s == pytest.approx(30.0)
    assert SPEC_BASELINES["coverage_audit"].max_cost_usd == pytest.approx(5.0)
    assert SPEC_BASELINES["coverage_audit"].max_wall_clock_s == pytest.approx(300.0)
    assert SPEC_BASELINES["quality_audit"].max_cost_usd == pytest.approx(3.0)
    assert SPEC_BASELINES["quality_audit"].max_wall_clock_s == pytest.approx(180.0)
    assert SPEC_BASELINES["analytical_review"].max_cost_usd == pytest.approx(10.0)
    assert SPEC_BASELINES["analytical_review"].max_wall_clock_s == pytest.approx(600.0)
    assert SPEC_BASELINES["anomaly_scan"].max_cost_usd == pytest.approx(2.0)
    assert SPEC_BASELINES["anomaly_scan"].max_wall_clock_s == pytest.approx(120.0)


@pytest.mark.family_performance_cost
def test_spec_baselines_cover_every_uc_playbook() -> None:
    """Exactly the five UC playbooks have baselines pinned. UC6
    (relation_map) is intentionally not in §8.4 — its cost profile is
    open-ended and will be added once an eval set anchors it."""
    expected = {"qa", "coverage_audit", "quality_audit", "analytical_review", "anomaly_scan"}
    assert set(SPEC_BASELINES) == expected


@pytest.mark.family_performance_cost
def test_universal_gates_match_section_86_family_11() -> None:
    assert DEFAULT_MAX_INPUT_TOKENS_PER_CALL == 16_000
    assert pytest.approx(0.90) == DEFAULT_MIN_CACHE_HIT_RATE
    assert pytest.approx(0.20) == DEFAULT_REGRESSION_TOLERANCE


# --- aggregate_stats ---


@pytest.mark.family_performance_cost
def test_aggregate_stats_sums_cost_and_latency() -> None:
    records = [
        _record(cost_usd=0.03, latency_ms=1000),
        _record(cost_usd=0.04, latency_ms=2000),
        _record(cost_usd=0.02, latency_ms=500),
    ]
    stats = aggregate_stats(records, playbook="qa")
    assert stats.n_calls == 3
    assert stats.total_cost_usd == pytest.approx(0.09)
    assert stats.total_wall_clock_s == pytest.approx(3.5)


@pytest.mark.family_performance_cost
def test_aggregate_stats_tracks_max_tokens_in() -> None:
    records = [
        _record(tokens_in=500),
        _record(tokens_in=14_000),
        _record(tokens_in=3_000),
    ]
    stats = aggregate_stats(records, playbook="qa")
    assert stats.max_tokens_in == 14_000


@pytest.mark.family_performance_cost
def test_aggregate_stats_cache_hit_rate_computed_over_cacheable_calls() -> None:
    records = [
        _record(cache_hit=True),
        _record(cache_hit=True),
        _record(cache_hit=False),
        _record(cache_hit=True),
    ]
    stats = aggregate_stats(records, playbook="qa")
    assert stats.cache_hits == 3
    assert stats.cacheable_calls == 4
    assert stats.cache_hit_rate == pytest.approx(0.75)


@pytest.mark.family_performance_cost
def test_aggregate_stats_ignores_records_from_other_playbooks() -> None:
    records = [
        _record(playbook="qa", cost_usd=0.05),
        _record(playbook="coverage_audit", cost_usd=2.0),
        _record(playbook="qa", cost_usd=0.05),
    ]
    qa_stats = aggregate_stats(records, playbook="qa")
    assert qa_stats.n_calls == 2
    assert qa_stats.total_cost_usd == pytest.approx(0.10)


@pytest.mark.family_performance_cost
def test_aggregate_stats_empty_stream_is_well_defined() -> None:
    stats = aggregate_stats([], playbook="qa")
    assert stats.n_calls == 0
    assert stats.total_cost_usd == 0.0
    assert stats.total_wall_clock_s == 0.0
    assert stats.max_tokens_in == 0
    # Vacuous pass: no cacheable calls means there's no cache miss to
    # penalise.
    assert stats.cache_hit_rate == pytest.approx(1.0)


# --- check_against_baseline ---


@pytest.mark.family_performance_cost
def test_check_within_baseline_returns_no_violations() -> None:
    """A run that fits comfortably under every gate is clean."""
    records = [_record(cost_usd=0.05, latency_ms=10_000, tokens_in=2_000, cache_hit=True)]
    stats = aggregate_stats(records, playbook="qa")
    violations = check_against_baseline(stats, SPEC_BASELINES["qa"])
    assert violations == []


@pytest.mark.family_performance_cost
def test_check_within_regression_tolerance_passes() -> None:
    """A 20% overshoot (default tolerance) is allowed — only over-20%
    fails. qa cap is $0.10; 1.20x = $0.12 must still pass."""
    records = [_record(cost_usd=0.119, latency_ms=10_000, cache_hit=True)]
    stats = aggregate_stats(records, playbook="qa")
    violations = check_against_baseline(stats, SPEC_BASELINES["qa"])
    cost_violations = [v for v in violations if v.metric == "cost_usd"]
    assert cost_violations == []


@pytest.mark.family_performance_cost
def test_check_just_over_regression_tolerance_fails() -> None:
    """qa cap = $0.10 → cap x 1.20 = $0.12. Anything strictly above
    that fails the gate."""
    records = [_record(cost_usd=0.13, latency_ms=10_000, cache_hit=True)]
    stats = aggregate_stats(records, playbook="qa")
    violations = check_against_baseline(stats, SPEC_BASELINES["qa"])
    cost_violations = [v for v in violations if v.metric == "cost_usd"]
    assert len(cost_violations) == 1
    assert cost_violations[0].actual == pytest.approx(0.13)
    assert cost_violations[0].baseline == pytest.approx(0.10)


@pytest.mark.family_performance_cost
def test_check_latency_overshoot_above_tolerance_fails() -> None:
    """qa cap = 30s → cap x 1.20 = 36s. 40s overshoot fails."""
    records = [_record(latency_ms=40_000, cache_hit=True)]
    stats = aggregate_stats(records, playbook="qa")
    violations = check_against_baseline(stats, SPEC_BASELINES["qa"])
    latency_violations = [v for v in violations if v.metric == "wall_clock_s"]
    assert len(latency_violations) == 1


@pytest.mark.family_performance_cost
def test_check_token_cap_is_hard_no_headroom() -> None:
    """Per §8.6 family 11 the 16k token cap is an invariant — not a
    perf target. A single 16_001-token call fails even though the
    regression budget would technically allow up to 19_200."""
    records = [_record(tokens_in=16_001, cache_hit=True)]
    stats = aggregate_stats(records, playbook="qa")
    violations = check_against_baseline(stats, SPEC_BASELINES["qa"])
    token_violations = [v for v in violations if v.metric == "tokens_per_call"]
    assert len(token_violations) == 1
    assert token_violations[0].tolerance == 0.0


@pytest.mark.family_performance_cost
def test_check_cache_hit_rate_below_floor_fails() -> None:
    """Per §8.6 family 11 the cache-hit floor (≥90%) is also an
    invariant. 80% hit rate fails."""
    records = [
        _record(cache_hit=True),
        _record(cache_hit=True),
        _record(cache_hit=True),
        _record(cache_hit=True),
        _record(cache_hit=False),
    ]
    stats = aggregate_stats(records, playbook="qa")
    violations = check_against_baseline(stats, SPEC_BASELINES["qa"])
    cache_violations = [v for v in violations if v.metric == "cache_hit_rate"]
    assert len(cache_violations) == 1
    assert cache_violations[0].actual == pytest.approx(0.80)
    assert cache_violations[0].baseline == pytest.approx(0.90)


@pytest.mark.family_performance_cost
def test_check_multiple_violations_surface_independently() -> None:
    """A run that blows several gates simultaneously must report each
    one separately so triage can address them in parallel."""
    records = [
        _record(cost_usd=2.0, latency_ms=120_000, tokens_in=20_000, cache_hit=False),
        _record(cost_usd=2.0, latency_ms=120_000, tokens_in=20_000, cache_hit=False),
    ]
    stats = aggregate_stats(records, playbook="qa")
    violations = check_against_baseline(stats, SPEC_BASELINES["qa"])
    metrics = {v.metric for v in violations}
    assert metrics == {"cost_usd", "wall_clock_s", "tokens_per_call", "cache_hit_rate"}


@pytest.mark.family_performance_cost
def test_check_mismatched_playbook_raises() -> None:
    """Stats and baseline must agree on which playbook they describe —
    a silent mismatch would make the gate compare apples to oranges."""
    stats = aggregate_stats([_record(playbook="qa")], playbook="qa")
    with pytest.raises(ValueError, match="playbook mismatch"):
        check_against_baseline(stats, SPEC_BASELINES["coverage_audit"])


# --- custom baselines ---


@pytest.mark.family_performance_cost
def test_custom_baseline_accepted() -> None:
    """Callers can supply a non-spec baseline — useful for per-doc
    overrides and for evaluating future playbooks before they land
    in `SPEC_BASELINES`."""
    custom = PlaybookBaseline(
        playbook="experimental",
        max_cost_usd=1.0,
        max_wall_clock_s=10.0,
    )
    stats = aggregate_stats(
        [_record(playbook="experimental", cost_usd=0.5, latency_ms=5000, cache_hit=True)],
        playbook="experimental",
    )
    assert check_against_baseline(stats, custom) == []


@pytest.mark.family_performance_cost
def test_tighter_regression_tolerance_catches_smaller_overshoots() -> None:
    """At tolerance=0.05, a $0.11 run (10% over $0.10 cap) fails even
    though the default 20% budget would allow it."""
    records = [_record(cost_usd=0.11, latency_ms=10_000, cache_hit=True)]
    stats = aggregate_stats(records, playbook="qa")
    strict = check_against_baseline(stats, SPEC_BASELINES["qa"], regression_tolerance=0.05)
    relaxed = check_against_baseline(stats, SPEC_BASELINES["qa"], regression_tolerance=0.20)
    assert any(v.metric == "cost_usd" for v in strict)
    assert all(v.metric != "cost_usd" for v in relaxed)
