"""Streaming progress events — schema, emitters, tracker.

The orchestrator emits one JSON-shaped event per sub-task transition
(`task_started`, `task_completed`, `task_failed`) so a user-side
process bar or callback can show liveness within the §4.7 "never
block silently longer than 5s" budget. ETA is computed from the
elapsed wall-clock divided by completed-so-far; running cost is the
sum of per-task `cost_usd` arguments. The tracker is thread-safe so
async fan-out via `bounded_gather` (S-064) can fire events from
multiple worker threads without garbling counts.

SPEC-REF: §4.7 (streaming)
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field

import pytest
from pydantic import ValidationError

from ctrldoc.orch.progress import (
    CallableProgressEmitter,
    ProgressEvent,
    ProgressTracker,
    StdoutProgressEmitter,
)

# --- ProgressEvent model ---


def test_progress_event_round_trip() -> None:
    ev = ProgressEvent(
        event="task_started",
        task_id="t-1",
        progress="0/10",
        eta_seconds=None,
        cost_so_far_usd=0.0,
    )
    assert ev.event == "task_started"
    assert ev.task_id == "t-1"
    payload = json.loads(ev.model_dump_json())
    assert payload["event"] == "task_started"


def test_progress_event_rejects_unknown_event_label() -> None:
    with pytest.raises(ValidationError):
        ProgressEvent(event="something_else", task_id="t", progress="0/1")  # type: ignore[arg-type]


def test_progress_event_is_frozen() -> None:
    ev = ProgressEvent(event="task_completed", task_id="t", progress="1/1")
    with pytest.raises(ValidationError):
        ev.task_id = "u"  # type: ignore[misc]


def test_progress_event_error_only_on_failure_events() -> None:
    """`error` is optional and only meaningful on `task_failed`; we don't
    forbid it elsewhere (Pydantic strictness handles the schema), but
    the tracker only emits it on `fail()`."""
    ev = ProgressEvent(event="task_failed", task_id="t", progress="1/1", error="oops")
    assert ev.error == "oops"


# --- callable emitter ---


@dataclass
class _Recorder:
    events: list[ProgressEvent] = field(default_factory=list)

    def __call__(self, event: ProgressEvent) -> None:
        self.events.append(event)


def test_callable_emitter_forwards_to_function() -> None:
    rec = _Recorder()
    emitter = CallableProgressEmitter(rec)
    ev = ProgressEvent(event="task_started", task_id="x", progress="0/1")
    emitter.emit(ev)
    assert rec.events == [ev]


# --- stdout emitter ---


def test_stdout_emitter_writes_json_lines(capsys: pytest.CaptureFixture[str]) -> None:
    emitter = StdoutProgressEmitter()
    emitter.emit(ProgressEvent(event="task_started", task_id="x", progress="0/3"))
    emitter.emit(ProgressEvent(event="task_completed", task_id="x", progress="1/3"))
    out = capsys.readouterr().out.strip().split("\n")
    assert len(out) == 2
    payloads = [json.loads(line) for line in out]
    assert payloads[0]["event"] == "task_started"
    assert payloads[1]["event"] == "task_completed"


# --- ProgressTracker basic flow ---


def _tracker(*, total: int, recorder: _Recorder, start_time: float = 0.0) -> ProgressTracker:
    clock_state = {"now": start_time}

    def clock() -> float:
        return clock_state["now"]

    tracker = ProgressTracker(
        total=total,
        emitter=CallableProgressEmitter(recorder),
        clock=clock,
    )
    tracker._clock_state = clock_state  # type: ignore[attr-defined]
    return tracker


def test_start_emits_task_started_with_zero_progress() -> None:
    rec = _Recorder()
    tracker = _tracker(total=5, recorder=rec)
    tracker.start("t-1")
    assert len(rec.events) == 1
    ev = rec.events[0]
    assert ev.event == "task_started"
    assert ev.task_id == "t-1"
    assert ev.progress == "0/5"
    assert ev.eta_seconds is None  # cannot estimate yet
    assert ev.cost_so_far_usd == 0.0


def test_complete_advances_progress_and_accumulates_cost() -> None:
    rec = _Recorder()
    tracker = _tracker(total=3, recorder=rec)
    tracker.start("t-1")
    tracker.complete("t-1", cost_usd=0.05)
    tracker.start("t-2")
    tracker.complete("t-2", cost_usd=0.07)
    # Events: started-1, completed-1, started-2, completed-2.
    progress = [(e.event, e.progress, e.cost_so_far_usd) for e in rec.events]
    assert progress == [
        ("task_started", "0/3", 0.0),
        ("task_completed", "1/3", 0.05),
        ("task_started", "1/3", 0.05),
        ("task_completed", "2/3", pytest.approx(0.12)),
    ]


def test_fail_emits_task_failed_with_error_and_progress() -> None:
    rec = _Recorder()
    tracker = _tracker(total=2, recorder=rec)
    tracker.start("t-1")
    tracker.fail("t-1", error="boom", cost_usd=0.01)
    failed = rec.events[-1]
    assert failed.event == "task_failed"
    assert failed.error == "boom"
    assert failed.progress == "1/2"  # fail counts toward completion for ETA purposes


def test_fail_advances_completed_counter_for_eta() -> None:
    """Failed tasks count as 'done' so ETA reflects actual throughput."""
    rec = _Recorder()
    tracker = _tracker(total=4, recorder=rec, start_time=0.0)
    tracker.start("a")
    tracker._clock_state["now"] = 1.0  # type: ignore[attr-defined]
    tracker.fail("a", error="x")
    tracker.start("b")
    tracker._clock_state["now"] = 2.0  # type: ignore[attr-defined]
    tracker.complete("b")
    # Two of four are done in 2 seconds → ETA = 2 more seconds.
    eta = rec.events[-1].eta_seconds
    assert eta is not None
    assert eta == pytest.approx(2.0)


# --- ETA derivation ---


def test_eta_is_none_until_first_completion() -> None:
    rec = _Recorder()
    tracker = _tracker(total=3, recorder=rec)
    tracker.start("a")
    assert rec.events[-1].eta_seconds is None


def test_eta_is_zero_when_everything_done() -> None:
    rec = _Recorder()
    tracker = _tracker(total=2, recorder=rec, start_time=0.0)
    tracker.start("a")
    tracker._clock_state["now"] = 1.0  # type: ignore[attr-defined]
    tracker.complete("a")
    tracker.start("b")
    tracker._clock_state["now"] = 2.0  # type: ignore[attr-defined]
    tracker.complete("b")
    assert rec.events[-1].eta_seconds == pytest.approx(0.0)
    assert rec.events[-1].progress == "2/2"


def test_eta_derived_from_throughput() -> None:
    rec = _Recorder()
    tracker = _tracker(total=10, recorder=rec, start_time=0.0)
    tracker.start("t-1")
    tracker._clock_state["now"] = 5.0  # type: ignore[attr-defined]
    tracker.complete("t-1")
    # Completed 1 of 10 in 5s → 5s/task → 9 remaining → ETA = 45s.
    assert rec.events[-1].eta_seconds == pytest.approx(45.0)


# --- construction invariants ---


def test_total_must_be_positive() -> None:
    rec = _Recorder()
    with pytest.raises(ValueError, match="total"):
        ProgressTracker(total=0, emitter=CallableProgressEmitter(rec))


# --- thread safety ---


def test_tracker_is_thread_safe_under_concurrent_completes() -> None:
    rec = _Recorder()
    # No clock injection — wall-clock works fine for this test; we only
    # care about counter consistency.
    tracker = ProgressTracker(total=200, emitter=CallableProgressEmitter(rec))

    def worker(i: int) -> None:
        tracker.start(f"t-{i}")
        tracker.complete(f"t-{i}", cost_usd=0.001)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(200)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 200 started + 200 completed = 400 events.
    assert len(rec.events) == 400
    final = [e for e in rec.events if e.event == "task_completed"][-1]
    assert final.progress == "200/200"
    assert final.cost_so_far_usd == pytest.approx(0.2)


# --- emitter exceptions do not corrupt counters ---


def test_emitter_failure_does_not_drop_progress_counters() -> None:
    """If the emitter raises (e.g. stdout closed), the tracker should
    propagate but its internal counters must remain consistent so a
    follow-up emit reports the right numbers."""

    class _BoomEmitter:
        def __init__(self) -> None:
            self.calls = 0
            self.recorder = _Recorder()

        def emit(self, event: ProgressEvent) -> None:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("stdout closed")
            self.recorder.events.append(event)

    boom = _BoomEmitter()
    tracker = ProgressTracker(total=3, emitter=boom)
    with pytest.raises(RuntimeError):
        tracker.start("a")
    # The counter increment that should have happened for the failed
    # emit must still be in place — otherwise the next event has the
    # wrong "N/M" denominator. The denominator here is total, which is
    # unchanged; the numerator must reflect that one start was
    # consumed (no — start doesn't advance numerator). Use complete
    # to verify: after the failed start the next complete should
    # report 1/3.
    tracker.complete("a")
    assert boom.recorder.events[-1].progress == "1/3"
