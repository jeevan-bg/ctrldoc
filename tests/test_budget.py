"""Contract tests for the budget guard / hard kill switch.

The guard tracks cumulative cost, refuses oversized per-call requests
(in tokens), and bounds wall-clock time per playbook run. Breach
raises `BudgetExceededError` so the orchestrator can abort cleanly
without silent overspend.

SPEC-REF: §4.7 (cost / budget guard)
"""

from __future__ import annotations

import threading
import time

import pytest

from ctrldoc.budget import BudgetExceededError, BudgetGuard
from ctrldoc.config import BudgetsConfig


def test_add_cost_accumulates_under_limit() -> None:
    g = BudgetGuard(max_cost_usd=1.0)
    g.add_cost(0.10)
    g.add_cost(0.40)
    assert g.cost_spent_usd == pytest.approx(0.50)
    assert not g.is_exhausted


def test_add_cost_raises_on_overrun() -> None:
    g = BudgetGuard(max_cost_usd=1.0)
    g.add_cost(0.80)
    with pytest.raises(BudgetExceededError) as info:
        g.add_cost(0.30)
    msg = str(info.value)
    assert "cost" in msg.lower()


def test_overrun_increment_is_still_recorded() -> None:
    g = BudgetGuard(max_cost_usd=1.0)
    g.add_cost(0.80)
    with pytest.raises(BudgetExceededError):
        g.add_cost(0.30)
    # Caller must see what was actually spent.
    assert g.cost_spent_usd == pytest.approx(1.10)
    assert g.is_exhausted


def test_negative_cost_rejected() -> None:
    g = BudgetGuard(max_cost_usd=1.0)
    with pytest.raises(ValueError):
        g.add_cost(-0.01)


def test_per_call_tokens_under_limit() -> None:
    g = BudgetGuard(max_cost_usd=1.0, max_tokens_per_call=1000)
    g.check_per_call_tokens(999)


def test_per_call_tokens_at_limit_ok() -> None:
    g = BudgetGuard(max_cost_usd=1.0, max_tokens_per_call=1000)
    g.check_per_call_tokens(1000)


def test_per_call_tokens_over_limit_raises() -> None:
    g = BudgetGuard(max_cost_usd=1.0, max_tokens_per_call=1000)
    with pytest.raises(BudgetExceededError):
        g.check_per_call_tokens(1001)


def test_per_call_tokens_unconfigured_never_raises() -> None:
    g = BudgetGuard(max_cost_usd=1.0)
    g.check_per_call_tokens(10**9)


def test_wall_clock_guard() -> None:
    g = BudgetGuard(max_cost_usd=1.0, max_wall_clock_min=1)
    g.start_wall_clock()
    g.check_wall_clock()  # plenty of time left
    # Simulate elapsed by reaching into start time.
    g._wall_clock_started_at -= 120  # type: ignore[attr-defined]
    with pytest.raises(BudgetExceededError):
        g.check_wall_clock()


def test_wall_clock_unconfigured_never_raises() -> None:
    g = BudgetGuard(max_cost_usd=1.0)
    g.start_wall_clock()
    time.sleep(0.01)
    g.check_wall_clock()


def test_from_budgets_config_round_trips() -> None:
    cfg = BudgetsConfig(
        max_cost_usd=5.0,
        max_tokens_per_call=2000,
        max_wall_clock_min=30,
    )
    g = BudgetGuard.from_config(cfg)
    assert g.max_cost_usd == 5.0
    assert g.max_tokens_per_call == 2000
    assert g.max_wall_clock_min == 30


def test_zero_or_negative_max_cost_rejected() -> None:
    with pytest.raises(ValueError):
        BudgetGuard(max_cost_usd=0.0)
    with pytest.raises(ValueError):
        BudgetGuard(max_cost_usd=-1.0)


def test_concurrent_add_cost_is_thread_safe() -> None:
    g = BudgetGuard(max_cost_usd=1000.0)

    def adder() -> None:
        for _ in range(1000):
            g.add_cost(0.01)

    threads = [threading.Thread(target=adder) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert g.cost_spent_usd == pytest.approx(8 * 1000 * 0.01)
