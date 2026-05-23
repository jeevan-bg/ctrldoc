"""UC2 `coverage_audit` playbook — cluster, batched-judge, aggregate.

Per §5.2 each checklist item carries a `topic_key`. Items with the
same key share one retrieval (one cluster's evidence pack) and one
batched API call (BatchedTaskRunner from S-063). The playbook
returns a `CoverageReport` whose verdicts are in original input
order — regardless of how the items got clustered internally.

SPEC-REF: §5.2 (UC2 coverage_audit)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest
from pydantic import ValidationError

from ctrldoc.assembler import CacheablePrefix
from ctrldoc.models import EvidencePack, Span, Verdict
from ctrldoc.orch.batch import BatchedTaskRunner
from ctrldoc.playbooks.coverage import (
    ChecklistItem,
    CoverageAuditPlaybook,
    CoverageReport,
    CoverageRetriever,
)

# --- stubs ---


@dataclass
class _StubRetriever:
    packs_by_topic: dict[str, EvidencePack]
    calls: list[str] = field(default_factory=list)

    def retrieve(self, topic_key: str) -> EvidencePack:
        self.calls.append(topic_key)
        return self.packs_by_topic[topic_key]


@dataclass
class _StubTaskClient:
    """Returns a scripted response per call (FIFO)."""

    responses: list[str]
    calls: list[tuple[str, str]] = field(default_factory=list)

    def call(self, *, system: str, user: str) -> str:
        self.calls.append((system, user))
        if not self.responses:
            raise AssertionError("stub exhausted")
        return self.responses.pop(0)


# --- fixtures ---


def _prefix() -> CacheablePrefix:
    return CacheablePrefix(
        system_prompt="You are an audit judge.",
        doc_skeleton="# §1 Aurora\n\nintroduces consistent hashing.",
        entity_glossary="- **aurora** [system]",
    )


def _pack(topic: str, spans: list[tuple[str, str]]) -> EvidencePack:
    return EvidencePack(
        query=topic,
        spans=[
            Span(chunk_id=cid, char_start=0, char_end=len(text), text=text) for cid, text in spans
        ],
        token_count=20,
        retrieval_plan=[f"search('{topic}', view=dense)"],
    )


def _verdict_response(entries: list[tuple[str, str, float, list[str]]]) -> str:
    """Build a batched response: list of {id, output}."""
    return json.dumps(
        [
            {
                "id": item_id,
                "output": {
                    "verdict": verdict,
                    "confidence": confidence,
                    "citation_chunk_ids": chunk_ids,
                },
            }
            for item_id, verdict, confidence, chunk_ids in entries
        ]
    )


def _playbook(
    *,
    packs_by_topic: dict[str, EvidencePack],
    responses: list[str],
) -> tuple[CoverageAuditPlaybook, _StubRetriever, _StubTaskClient]:
    retriever = _StubRetriever(packs_by_topic=packs_by_topic)
    client = _StubTaskClient(responses=responses)
    playbook = CoverageAuditPlaybook(
        prefix=_prefix(),
        retriever=retriever,
        batched_runner=BatchedTaskRunner(client=client),
    )
    return playbook, retriever, client


# --- happy path ---


def test_single_cluster_runs_one_batched_call_with_all_items() -> None:
    items = [
        ChecklistItem(id="r-1", text="Doc explains consistent hashing.", topic_key="hashing"),
        ChecklistItem(id="r-2", text="Doc explains failover.", topic_key="hashing"),
    ]
    playbook, retriever, client = _playbook(
        packs_by_topic={
            "hashing": _pack(
                "hashing",
                [
                    ("c1", "Aurora uses consistent hashing across nodes."),
                    ("c2", "Failover triggers reshard on heartbeat loss."),
                ],
            ),
        },
        responses=[
            _verdict_response(
                [
                    ("r-1", "Covered", 0.9, ["c1"]),
                    ("r-2", "Partial", 0.5, ["c2"]),
                ]
            ),
        ],
    )

    report = playbook.run(items)

    assert isinstance(report, CoverageReport)
    assert [v.item_id for v in report.verdicts] == ["r-1", "r-2"]
    assert report.verdicts[0].verdict == "Covered"
    assert report.verdicts[1].verdict == "Partial"
    assert report.verdicts[0].confidence == pytest.approx(0.9)
    # Cluster ⇒ one retrieve + one batched API call.
    assert retriever.calls == ["hashing"]
    assert len(client.calls) == 1


def test_citations_are_resolved_from_chunk_ids_to_spans() -> None:
    items = [ChecklistItem(id="r-1", text="x", topic_key="t")]
    playbook, _, _ = _playbook(
        packs_by_topic={
            "t": _pack(
                "t",
                [
                    ("c-a", "first span"),
                    ("c-b", "second span"),
                    ("c-c", "third span"),
                ],
            ),
        },
        responses=[
            _verdict_response([("r-1", "Covered", 0.9, ["c-a", "c-c"])]),
        ],
    )
    report = playbook.run(items)
    cited = report.verdicts[0].citations
    assert [s.chunk_id for s in cited] == ["c-a", "c-c"]
    assert [s.text for s in cited] == ["first span", "third span"]


def test_unknown_citation_chunk_ids_are_filtered() -> None:
    """Model occasionally hallucinates ids — the playbook silently
    drops citations the evidence pack doesn't contain."""
    items = [ChecklistItem(id="r-1", text="x", topic_key="t")]
    playbook, _, _ = _playbook(
        packs_by_topic={"t": _pack("t", [("c-a", "real span")])},
        responses=[
            _verdict_response([("r-1", "Covered", 0.9, ["c-a", "c-ghost"])]),
        ],
    )
    report = playbook.run(items)
    assert [s.chunk_id for s in report.verdicts[0].citations] == ["c-a"]


# --- multiple clusters ---


def test_multiple_clusters_each_get_their_own_batched_call() -> None:
    items = [
        ChecklistItem(id="r-1", text="hashing item", topic_key="hashing"),
        ChecklistItem(id="r-2", text="auth item", topic_key="auth"),
        ChecklistItem(id="r-3", text="another hashing item", topic_key="hashing"),
    ]
    playbook, retriever, client = _playbook(
        packs_by_topic={
            "hashing": _pack("hashing", [("c-h", "hashing evidence")]),
            "auth": _pack("auth", [("c-a", "auth evidence")]),
        },
        responses=[
            # Order depends on cluster iteration — items r-1 + r-3 share the
            # first cluster, r-2 the second.
            _verdict_response(
                [
                    ("r-1", "Covered", 0.9, ["c-h"]),
                    ("r-3", "Partial", 0.5, ["c-h"]),
                ]
            ),
            _verdict_response([("r-2", "NotCovered", 0.0, [])]),
        ],
    )

    report = playbook.run(items)

    # Two retrieves, two API calls — but verdicts are returned in input order.
    assert sorted(retriever.calls) == ["auth", "hashing"]
    assert len(client.calls) == 2
    assert [v.item_id for v in report.verdicts] == ["r-1", "r-2", "r-3"]
    assert [v.verdict for v in report.verdicts] == ["Covered", "NotCovered", "Partial"]


def test_cluster_evidence_passed_to_correct_batched_call() -> None:
    """Each cluster's user message carries the evidence text matching
    that topic — no cross-contamination between clusters."""
    items = [
        ChecklistItem(id="r-1", text="x", topic_key="t1"),
        ChecklistItem(id="r-2", text="y", topic_key="t2"),
    ]
    playbook, _, client = _playbook(
        packs_by_topic={
            "t1": _pack("t1", [("c1", "T1-EVIDENCE-MARKER")]),
            "t2": _pack("t2", [("c2", "T2-EVIDENCE-MARKER")]),
        },
        responses=[
            _verdict_response([("r-1", "Covered", 0.9, ["c1"])]),
            _verdict_response([("r-2", "Covered", 0.9, ["c2"])]),
        ],
    )
    playbook.run(items)
    # The first call carries t1's evidence and not t2's, and vice versa.
    users = [user for _, user in client.calls]
    by_topic = sorted(users, key=lambda u: u.find("MARKER"))
    assert "T1-EVIDENCE-MARKER" in by_topic[0]
    assert "T2-EVIDENCE-MARKER" not in by_topic[0]
    assert "T2-EVIDENCE-MARKER" in by_topic[1]
    assert "T1-EVIDENCE-MARKER" not in by_topic[1]


# --- empty input ---


def test_empty_items_returns_empty_report_without_dependencies() -> None:
    playbook, retriever, client = _playbook(packs_by_topic={}, responses=[])
    report = playbook.run([])
    assert report.verdicts == []
    assert retriever.calls == []
    assert client.calls == []


# --- input validation ---


def test_checklist_item_is_frozen() -> None:
    item = ChecklistItem(id="r-1", text="x", topic_key="t")
    with pytest.raises(ValidationError):
        item.text = "y"  # type: ignore[misc]


def test_duplicate_item_ids_across_clusters_rejected() -> None:
    items = [
        ChecklistItem(id="dup", text="x", topic_key="a"),
        ChecklistItem(id="dup", text="y", topic_key="b"),
    ]
    playbook, _, _ = _playbook(packs_by_topic={}, responses=[])
    with pytest.raises(ValueError, match="duplicate"):
        playbook.run(items)


def test_invalid_verdict_literal_from_model_raises() -> None:
    """The constrained Pydantic model rejects an unknown verdict label —
    silent fall-through would land in the final report."""
    items = [ChecklistItem(id="r-1", text="x", topic_key="t")]
    playbook, _, _ = _playbook(
        packs_by_topic={"t": _pack("t", [("c1", "evidence")])},
        responses=[_verdict_response([("r-1", "MaybeCovered", 0.5, ["c1"])])],
    )
    with pytest.raises(Exception):  # noqa: B017 — TaskOutputError wraps ValidationError
        playbook.run(items)


# --- CoverageReport model ---


def test_coverage_report_is_frozen() -> None:
    report = CoverageReport(verdicts=[])
    with pytest.raises(ValidationError):
        report.verdicts = []  # type: ignore[misc]


def test_coverage_report_round_trips_verdicts() -> None:
    span = Span(chunk_id="c1", char_start=0, char_end=5, text="hello")
    verdict = Verdict(item_id="r-1", verdict="Covered", citations=[span], confidence=0.9)
    report = CoverageReport(verdicts=[verdict])
    assert report.verdicts == [verdict]


# --- protocol ---


def test_coverage_retriever_protocol_isinstance() -> None:
    stub = _StubRetriever(packs_by_topic={})
    assert isinstance(stub, CoverageRetriever)


# --- cluster determinism ---


def test_clusters_are_iterated_in_first_appearance_order() -> None:
    """Topic clusters are visited in the order their first item appears,
    so two runs of the same input produce the same retriever call sequence."""
    items = [
        ChecklistItem(id="r-1", text="x", topic_key="z"),
        ChecklistItem(id="r-2", text="y", topic_key="a"),
        ChecklistItem(id="r-3", text="z", topic_key="z"),
    ]
    playbook, retriever, _ = _playbook(
        packs_by_topic={
            "z": _pack("z", [("cz", "z evidence")]),
            "a": _pack("a", [("ca", "a evidence")]),
        },
        responses=[
            _verdict_response(
                [
                    ("r-1", "Covered", 0.9, ["cz"]),
                    ("r-3", "Covered", 0.9, ["cz"]),
                ]
            ),
            _verdict_response([("r-2", "Covered", 0.9, ["ca"])]),
        ],
    )
    playbook.run(items)
    # `z` appears first in input → retrieved first.
    assert retriever.calls == ["z", "a"]
