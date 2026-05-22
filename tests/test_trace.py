"""Contract tests for the JSONL trace logger.

Every LLM call writes one append-only JSONL record to
`traces/{run_id}.jsonl` so that any run is fully replayable and any
regression in cost, latency, or cache-hit rate is grep-able from disk.

SPEC-REF: §4.7 (observability / tracing)
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from pydantic import ValidationError

from ctrldoc.trace import TraceLogger, TraceRecord, read_trace


def _record(**overrides: object) -> TraceRecord:
    defaults: dict[str, object] = {
        "run_id": "20260522T000000Z-deadbeef",
        "task_id": "t-001",
        "playbook": "qa",
        "model": "claude-opus-4-7",
        "prompt_hash": "sha256:" + "a" * 64,
        "response_hash": "sha256:" + "b" * 64,
        "tokens_in": 1024,
        "tokens_out": 256,
        "cost_usd": 0.0125,
        "latency_ms": 1500,
        "cache_hit": True,
        "error": None,
    }
    defaults.update(overrides)
    return TraceRecord(**defaults)  # type: ignore[arg-type]


def test_record_has_spec_fields() -> None:
    expected = {
        "run_id",
        "task_id",
        "playbook",
        "model",
        "prompt_hash",
        "response_hash",
        "tokens_in",
        "tokens_out",
        "cost_usd",
        "latency_ms",
        "cache_hit",
        "error",
    }
    assert expected == set(TraceRecord.model_fields)


def test_record_negative_tokens_rejected() -> None:
    with pytest.raises(ValidationError):
        _record(tokens_in=-1)


def test_record_negative_cost_rejected() -> None:
    with pytest.raises(ValidationError):
        _record(cost_usd=-0.01)


def test_record_negative_latency_rejected() -> None:
    with pytest.raises(ValidationError):
        _record(latency_ms=-1)


def test_record_round_trip() -> None:
    r = _record()
    assert TraceRecord.model_validate(r.model_dump()) == r


def test_logger_creates_file_at_traces_run_id_jsonl(tmp_path: Path) -> None:
    logger = TraceLogger(traces_dir=tmp_path, run_id="20260522T000000Z-deadbeef")
    logger.write(_record())
    expected = tmp_path / "20260522T000000Z-deadbeef.jsonl"
    assert expected.exists()
    line = expected.read_text(encoding="utf-8").splitlines()[0]
    payload = json.loads(line)
    assert payload["task_id"] == "t-001"


def test_logger_appends_in_order(tmp_path: Path) -> None:
    logger = TraceLogger(traces_dir=tmp_path, run_id="r1")
    logger.write(_record(task_id="t-1"))
    logger.write(_record(task_id="t-2"))
    logger.write(_record(task_id="t-3"))
    records = list(read_trace(tmp_path / "r1.jsonl"))
    assert [r.task_id for r in records] == ["t-1", "t-2", "t-3"]


def test_logger_produces_valid_jsonl(tmp_path: Path) -> None:
    logger = TraceLogger(traces_dir=tmp_path, run_id="r1")
    for i in range(5):
        logger.write(_record(task_id=f"t-{i}"))
    lines = (tmp_path / "r1.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    for line in lines:
        json.loads(line)  # must parse


def test_logger_concurrent_writes_do_not_interleave(tmp_path: Path) -> None:
    logger = TraceLogger(traces_dir=tmp_path, run_id="r1")

    def writer(start: int) -> None:
        for i in range(50):
            logger.write(_record(task_id=f"t-{start}-{i}"))

    threads = [threading.Thread(target=writer, args=(s,)) for s in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = (tmp_path / "r1.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4 * 50
    for line in lines:
        payload = json.loads(line)
        assert payload["task_id"].startswith("t-")


def test_logger_creates_traces_directory(tmp_path: Path) -> None:
    nested = tmp_path / "deeper" / "traces"
    logger = TraceLogger(traces_dir=nested, run_id="r1")
    logger.write(_record())
    assert (nested / "r1.jsonl").exists()


def test_read_trace_yields_typed_records(tmp_path: Path) -> None:
    path = tmp_path / "r1.jsonl"
    payload = _record().model_dump()
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    records = list(read_trace(path))
    assert len(records) == 1
    assert isinstance(records[0], TraceRecord)


def test_error_field_optional_and_string(tmp_path: Path) -> None:
    logger = TraceLogger(traces_dir=tmp_path, run_id="r1")
    logger.write(_record(error="HTTP 429: rate limited", cache_hit=False))
    records = list(read_trace(tmp_path / "r1.jsonl"))
    assert records[0].error == "HTTP 429: rate limited"
