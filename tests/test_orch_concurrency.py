"""Concurrency policy — semaphore-bounded async fan-out.

`bounded_gather` runs N coroutines concurrently but never lets more
than `max_concurrent` execute at once. It preserves input order in
the result list, releases the semaphore on exception (so a failing
sub-task can't lock a slot), and propagates the first exception while
cancelling in-flight peers.

The accompanying family-14 invariants exercise: cap enforcement,
order preservation, error isolation, semaphore-release-on-failure,
and that parallel sub-tasks see no shared mutable state.

SPEC-REF: §4.7 (concurrency), §8.6 family 14
"""

from __future__ import annotations

import asyncio

import pytest

from ctrldoc.orch.concurrency import (
    DEFAULT_ANTHROPIC_CONCURRENT,
    DEFAULT_EMBEDDER_CONCURRENT,
    DEFAULT_OLLAMA_7B_CONCURRENT,
    DEFAULT_RERANKER_CONCURRENT,
    bounded_gather,
)

# --- basic shape ---


async def test_empty_input_returns_empty_result() -> None:
    result = await bounded_gather([], max_concurrent=4)
    assert result == []


async def test_single_coro_round_trips_its_result() -> None:
    async def one() -> int:
        return 7

    result = await bounded_gather([one()], max_concurrent=1)
    assert result == [7]


async def test_results_preserve_input_order_even_when_finish_out_of_order() -> None:
    """Slow tasks at the head must still appear at the head of the result."""

    async def task(idx: int, delay: float) -> int:
        await asyncio.sleep(delay)
        return idx

    coros = [task(0, 0.03), task(1, 0.01), task(2, 0.02)]
    result = await bounded_gather(coros, max_concurrent=4)
    assert result == [0, 1, 2]


# --- cap enforcement ---


@pytest.mark.family_concurrency
async def test_cap_is_observed_at_every_instant() -> None:
    """At no point are more than `max_concurrent` tasks in flight."""
    in_flight = 0
    peak = 0
    lock = asyncio.Lock()

    async def tracked() -> None:
        nonlocal in_flight, peak
        async with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        async with lock:
            in_flight -= 1

    cap = 3
    await bounded_gather([tracked() for _ in range(20)], max_concurrent=cap)
    assert peak <= cap
    assert peak == cap, "with 20 tasks and cap=3 we should saturate"


@pytest.mark.family_concurrency
async def test_cap_of_one_serialises_execution() -> None:
    """Cap=1 forces strict serialisation."""
    order: list[int] = []

    async def append(i: int) -> int:
        order.append(i)
        await asyncio.sleep(0)  # yield to scheduler
        return i

    result = await bounded_gather([append(i) for i in range(5)], max_concurrent=1)
    assert result == [0, 1, 2, 3, 4]
    # With cap=1, scheduler ordering matches submission order.
    assert order == [0, 1, 2, 3, 4]


# --- exception handling ---


async def test_exception_propagates_to_caller() -> None:
    async def boom() -> int:
        raise RuntimeError("kaboom")

    with pytest.raises(RuntimeError, match="kaboom"):
        await bounded_gather([boom()], max_concurrent=1)


@pytest.mark.family_concurrency
async def test_semaphore_releases_on_exception() -> None:
    """A failing sub-task must release its slot — otherwise a later
    gather under the same scope would deadlock."""

    async def boom() -> None:
        raise RuntimeError("boom")

    async def ok() -> int:
        return 42

    # First batch — one of many fails. Catch the first failure.
    with pytest.raises(RuntimeError):
        await bounded_gather([boom() for _ in range(3)], max_concurrent=2)

    # Second batch under the same event loop succeeds.
    result = await bounded_gather([ok() for _ in range(5)], max_concurrent=2)
    assert result == [42, 42, 42, 42, 42]


# --- validation ---


async def test_non_positive_max_concurrent_raises_value_error() -> None:
    async def noop() -> None:
        return

    coro = noop()
    try:
        with pytest.raises(ValueError, match="max_concurrent"):
            await bounded_gather([coro], max_concurrent=0)
    finally:
        coro.close()


# --- family-14: no shared mutable state ---


@pytest.mark.family_concurrency
async def test_parallel_tasks_have_no_shared_mutable_state() -> None:
    """Each coro should observe only its own inputs — no leakage from siblings.

    We seed each coro with a unique value and check the output is one-to-one
    with the input.
    """

    async def echo(value: int) -> int:
        # Yield to the scheduler so siblings can interleave.
        await asyncio.sleep(0)
        return value

    inputs = list(range(50))
    result = await bounded_gather([echo(v) for v in inputs], max_concurrent=8)
    assert result == inputs


# --- end-to-end: TaskClient via to_thread ---


@pytest.mark.family_concurrency
async def test_bounded_gather_wraps_sync_task_clients_via_to_thread() -> None:
    """The sync TaskClient seam (S-060) is wrapped with asyncio.to_thread
    inside the coro — `bounded_gather` itself stays generic."""
    from dataclasses import dataclass, field

    @dataclass
    class _Client:
        responses: list[str]
        calls: list[tuple[str, str]] = field(default_factory=list)

        def call(self, *, system: str, user: str) -> str:
            self.calls.append((system, user))
            # Simulate I/O.
            return self.responses.pop(0)

    client = _Client(responses=[f"r{i}" for i in range(10)])

    async def one(idx: int) -> str:
        return await asyncio.to_thread(client.call, system="s", user=f"u{idx}")

    result = await bounded_gather([one(i) for i in range(10)], max_concurrent=4)
    # Order preserved across worker thread completion.
    assert result == [f"r{i}" for i in range(10)]
    # All 10 calls went through.
    assert len(client.calls) == 10


# --- defaults from §4.7 ---


def test_default_concurrency_caps_match_spec_table() -> None:
    """The §4.7 table values are codified as module-level defaults so they
    survive a config-file refactor."""
    assert DEFAULT_ANTHROPIC_CONCURRENT == 8
    assert DEFAULT_OLLAMA_7B_CONCURRENT == 1
    assert DEFAULT_EMBEDDER_CONCURRENT == 2
    assert DEFAULT_RERANKER_CONCURRENT == 2
