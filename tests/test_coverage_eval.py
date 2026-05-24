"""coverage_eval — runner + 3-case starter set against the gold doc.

Each case is self-contained: its checklist items, the per-topic
evidence spans the playbook will retrieve, and the per-item gold
verdicts all live inline. The runner builds a case-local retriever,
drives `CoverageAuditPlaybook`, and emits `verdict_accuracy`.

SPEC-REF: §8.1 (coverage_eval), §8.2 (coverage_audit metrics)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic import ValidationError

from ctrldoc.assembler import CacheablePrefix
from ctrldoc.eval.coverage import (
    VERDICT_ACCURACY_THRESHOLD,
    CoverageEvalCase,
    CoverageEvalRunner,
    EvidenceSpan,
)
from ctrldoc.eval.harness import load_jsonl_cases, run_eval
from ctrldoc.ops.audit import ChecklistItem
from ctrldoc.orch.batch import BatchedTaskRunner

COVERAGE_EVAL_PATH = Path(__file__).parent / "eval" / "coverage_eval.jsonl"


def _cases() -> list[CoverageEvalCase]:
    return load_jsonl_cases(COVERAGE_EVAL_PATH, case_model=CoverageEvalCase)


def _prefix() -> CacheablePrefix:
    return CacheablePrefix(
        system_prompt="coverage judge",
        doc_skeleton="# §1",
        entity_glossary="- **e/1** [concept]",
    )


# --- scripted task client ---


@dataclass
class _ScriptedClient:
    """Returns scripted batched-response payloads in FIFO order."""

    responses: list[str]
    calls: list[tuple[str, str]] = field(default_factory=list)

    def call(self, *, system: str, user: str) -> str:
        self.calls.append((system, user))
        if not self.responses:
            raise AssertionError("scripted client exhausted")
        return self.responses.pop(0)


def _batched_response(entries: list[tuple[str, str, float, list[str]]]) -> str:
    return json.dumps(
        [
            {
                "id": item_id,
                "output": {
                    "verdict": verdict,
                    "confidence": confidence,
                    "citation_chunk_ids": citations,
                },
            }
            for item_id, verdict, confidence, citations in entries
        ]
    )


# --- dataset invariants ---


def test_coverage_eval_set_loads_under_case_schema() -> None:
    cases = _cases()
    assert len(cases) >= 3
    assert any("aurora" in case.tags for case in cases)


def test_every_gold_verdict_id_matches_an_item() -> None:
    """Schema-level invariant: gold verdicts cover exactly the items.

    The model validator already enforces this; the test pins the
    invariant against accidental schema regressions."""
    for case in _cases():
        ids = {item.id for item in case.items}
        assert set(case.gold_verdicts.keys()) == ids


def test_case_with_missing_gold_for_item_rejected() -> None:
    with pytest.raises(ValidationError, match="missing"):
        CoverageEvalCase(
            id="bad-1",
            items=[
                ChecklistItem(id="a", text="x", topic_key="t"),
                ChecklistItem(id="b", text="y", topic_key="t"),
            ],
            evidence_by_topic={"t": []},
            gold_verdicts={"a": "Covered"},  # `b` missing
        )


def test_case_with_extra_gold_id_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown"):
        CoverageEvalCase(
            id="bad-2",
            items=[ChecklistItem(id="a", text="x", topic_key="t")],
            evidence_by_topic={"t": []},
            gold_verdicts={"a": "Covered", "ghost": "NotCovered"},
        )


# --- runner: all-correct verdicts ---


def test_runner_all_correct_yields_accuracy_one() -> None:
    case = CoverageEvalCase(
        id="cov-x",
        items=[
            ChecklistItem(id="r-1", text="Doc covers X.", topic_key="t"),
            ChecklistItem(id="r-2", text="Doc covers Y.", topic_key="t"),
        ],
        evidence_by_topic={"t": [EvidenceSpan(chunk_id="c1", text="evidence")]},
        gold_verdicts={"r-1": "Covered", "r-2": "Covered"},
    )
    client = _ScriptedClient(
        responses=[
            _batched_response([("r-1", "Covered", 0.9, ["c1"]), ("r-2", "Covered", 0.9, ["c1"])])
        ]
    )
    runner = CoverageEvalRunner(
        prefix=_prefix(),
        batched_runner=BatchedTaskRunner(client=client),
    )
    result = runner.run_case(case)
    assert result.metrics["verdict_accuracy"] == pytest.approx(1.0)
    assert result.passed is True


# --- runner: partial accuracy ---


def test_runner_partial_accuracy_below_threshold_fails() -> None:
    case = CoverageEvalCase(
        id="cov-y",
        items=[
            ChecklistItem(id="r-1", text="X", topic_key="t"),
            ChecklistItem(id="r-2", text="Y", topic_key="t"),
            ChecklistItem(id="r-3", text="Z", topic_key="t"),
        ],
        evidence_by_topic={"t": [EvidenceSpan(chunk_id="c1", text="ev")]},
        gold_verdicts={"r-1": "Covered", "r-2": "Covered", "r-3": "NotCovered"},
    )
    # The playbook emits Covered/Covered/Covered — third verdict is wrong.
    # 2/3 = 0.667, below the 0.90 threshold.
    client = _ScriptedClient(
        responses=[
            _batched_response(
                [
                    ("r-1", "Covered", 0.9, ["c1"]),
                    ("r-2", "Covered", 0.9, ["c1"]),
                    ("r-3", "Covered", 0.9, ["c1"]),
                ]
            )
        ]
    )
    runner = CoverageEvalRunner(
        prefix=_prefix(),
        batched_runner=BatchedTaskRunner(client=client),
    )
    result = runner.run_case(case)
    assert result.metrics["verdict_accuracy"] == pytest.approx(2 / 3)
    assert result.passed is False


# --- multi-topic case ---


def test_runner_runs_multiple_clusters_in_one_case() -> None:
    case = CoverageEvalCase(
        id="cov-z",
        items=[
            ChecklistItem(id="r-h", text="hashing", topic_key="hashing"),
            ChecklistItem(id="r-f", text="failover", topic_key="failover"),
        ],
        evidence_by_topic={
            "hashing": [EvidenceSpan(chunk_id="c-h", text="hashing evidence")],
            "failover": [EvidenceSpan(chunk_id="c-f", text="failover evidence")],
        },
        gold_verdicts={"r-h": "Covered", "r-f": "Partial"},
    )
    client = _ScriptedClient(
        responses=[
            _batched_response([("r-h", "Covered", 0.9, ["c-h"])]),
            _batched_response([("r-f", "Partial", 0.5, ["c-f"])]),
        ]
    )
    runner = CoverageEvalRunner(
        prefix=_prefix(),
        batched_runner=BatchedTaskRunner(client=client),
    )
    result = runner.run_case(case)
    assert result.metrics["verdict_accuracy"] == pytest.approx(1.0)
    # Two clusters ⇒ two batched calls.
    assert len(client.calls) == 2


# --- end-to-end through the harness ---


def test_starter_set_passes_with_an_oracle_runner() -> None:
    """A runner that always emits the gold verdict clears every case."""

    @dataclass
    class _OracleRunner:
        def run_case(self, case: CoverageEvalCase):  # type: ignore[no-untyped-def]
            scripted_by_topic: dict[str, list[tuple[str, str, float, list[str]]]] = {}
            for item in case.items:
                gold = case.gold_verdicts[item.id]
                citations: list[str] = []
                if case.evidence_by_topic.get(item.topic_key):
                    citations = [case.evidence_by_topic[item.topic_key][0].chunk_id]
                scripted_by_topic.setdefault(item.topic_key, []).append(
                    (item.id, gold, 0.9, citations)
                )
            responses = [_batched_response(rows) for rows in scripted_by_topic.values()]
            client = _ScriptedClient(responses=responses)
            return CoverageEvalRunner(
                prefix=_prefix(),
                batched_runner=BatchedTaskRunner(client=client),
            ).run_case(case)

    report = run_eval(
        set_name="coverage_eval",
        cases=_cases(),
        runner=_OracleRunner(),
        thresholds={"verdict_accuracy": VERDICT_ACCURACY_THRESHOLD},
    )
    assert report.passed is True
    assert report.aggregate["verdict_accuracy"] == pytest.approx(1.0)
