"""Budget guard — hard kill switch on cost, per-call tokens, and wall clock.

The orchestrator records cost after every LLM call; the guard refuses
to advance once the total breaches the configured ceiling and raises
`BudgetExceededError` so a playbook aborts cleanly without silent
overspend.

SPEC-REF: §4.7 (cost / budget guard)
"""

from __future__ import annotations

import threading
import time

from ctrldoc.config import BudgetsConfig


class BudgetExceededError(RuntimeError):
    """Raised when a budget dimension is breached."""


class BudgetGuard:
    """Thread-safe accumulator for run-level cost, with optional caps on
    per-call token count and wall-clock time.
    """

    def __init__(
        self,
        *,
        max_cost_usd: float,
        max_tokens_per_call: int | None = None,
        max_wall_clock_min: int | None = None,
    ) -> None:
        if max_cost_usd <= 0:
            raise ValueError("max_cost_usd must be positive")
        self.max_cost_usd = float(max_cost_usd)
        self.max_tokens_per_call = max_tokens_per_call
        self.max_wall_clock_min = max_wall_clock_min
        self._cost_spent_usd = 0.0
        self._lock = threading.Lock()
        self._wall_clock_started_at: float | None = None

    @classmethod
    def from_config(cls, cfg: BudgetsConfig) -> BudgetGuard:
        return cls(
            max_cost_usd=cfg.max_cost_usd,
            max_tokens_per_call=cfg.max_tokens_per_call,
            max_wall_clock_min=cfg.max_wall_clock_min,
        )

    @property
    def cost_spent_usd(self) -> float:
        with self._lock:
            return self._cost_spent_usd

    @property
    def is_exhausted(self) -> bool:
        return self.cost_spent_usd >= self.max_cost_usd

    def add_cost(self, usd: float) -> None:
        """Record `usd` of additional spend. Raises if the new total exceeds
        the ceiling — the increment is recorded first so callers can observe
        what was actually spent.
        """
        if usd < 0:
            raise ValueError("cost increments must be non-negative")
        with self._lock:
            self._cost_spent_usd += usd
            spent = self._cost_spent_usd
        if spent > self.max_cost_usd:
            raise BudgetExceededError(
                f"cost limit breached: spent ${spent:.4f} of ${self.max_cost_usd:.2f}"
            )

    def check_per_call_tokens(self, tokens: int) -> None:
        if self.max_tokens_per_call is None:
            return
        if tokens > self.max_tokens_per_call:
            raise BudgetExceededError(
                f"per-call token limit breached: {tokens} > {self.max_tokens_per_call}"
            )

    def start_wall_clock(self) -> None:
        self._wall_clock_started_at = time.monotonic()

    def check_wall_clock(self) -> None:
        if self.max_wall_clock_min is None or self._wall_clock_started_at is None:
            return
        elapsed_min = (time.monotonic() - self._wall_clock_started_at) / 60.0
        if elapsed_min > self.max_wall_clock_min:
            raise BudgetExceededError(
                f"wall-clock limit breached: ran {elapsed_min:.2f} min "
                f"> {self.max_wall_clock_min} min"
            )


__all__ = ["BudgetExceededError", "BudgetGuard"]
