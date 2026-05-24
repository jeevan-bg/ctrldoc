"""UC3 `quality_audit` playbook — generate criteria, delegate to coverage.

Per §5.3 quality audit is "coverage audit + criteria generation":
an LLM enumerates a checklist from `doc_type`, a human (in
production) approves it, and the checklist is then fed straight
into the UC2 coverage audit pipeline. The slice covers the
`CriteriaGenerator` Protocol, a deterministic heuristic reference,
and a `QualityAuditPlaybook` that composes the generator with an
existing `CoverageAuditPlaybook`.

SPEC-REF: §5.3 (UC3 quality_audit)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest
from pydantic import ValidationError

from ctrldoc.assembler import CacheablePrefix
from ctrldoc.models import EvidencePack, Span
from ctrldoc.ops.audit import (
    ChecklistItem,
    CoverageAuditPlaybook,
    CoverageReport,
)
from ctrldoc.ops.quality import (
    CriteriaGenerator,
    HeuristicCriteriaGenerator,
    QualityAuditPlaybook,
    QualityReport,
)
from ctrldoc.orch.batch import BatchedTaskRunner

# --- stubs ---


@dataclass
class _StubGenerator:
    """Returns the configured list regardless of doc_type."""

    items: list[ChecklistItem]
    calls: list[str] = field(default_factory=list)

    def generate(self, doc_type: str) -> list[ChecklistItem]:
        self.calls.append(doc_type)
        return list(self.items)


@dataclass
class _StubRetriever:
    packs_by_topic: dict[str, EvidencePack]
    calls: list[str] = field(default_factory=list)

    def retrieve(self, topic_key: str) -> EvidencePack:
        self.calls.append(topic_key)
        return self.packs_by_topic[topic_key]


@dataclass
class _StubTaskClient:
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
        system_prompt="You are a quality audit judge.",
        doc_skeleton="# §1 Aurora\n\nintroduces consistent hashing.",
        entity_glossary="- **aurora** [system]",
    )


def _pack(spans: list[tuple[str, str]]) -> EvidencePack:
    return EvidencePack(
        query="topic",
        spans=[
            Span(chunk_id=cid, char_start=0, char_end=len(text), text=text) for cid, text in spans
        ],
        token_count=20,
        retrieval_plan=["search('topic', view=dense)"],
    )


def _verdict_response(entries: list[tuple[str, str, float, list[str]]]) -> str:
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


def _coverage_playbook(
    *,
    packs_by_topic: dict[str, EvidencePack],
    responses: list[str],
) -> tuple[CoverageAuditPlaybook, _StubRetriever, _StubTaskClient]:
    retriever = _StubRetriever(packs_by_topic=packs_by_topic)
    client = _StubTaskClient(responses=responses)
    return (
        CoverageAuditPlaybook(
            prefix=_prefix(),
            retriever=retriever,
            batched_runner=BatchedTaskRunner(client=client),
        ),
        retriever,
        client,
    )


# --- happy path ---


def test_quality_audit_generates_then_delegates_to_coverage() -> None:
    items = [
        ChecklistItem(id="q-1", text="Explain consistent hashing.", topic_key="t"),
        ChecklistItem(id="q-2", text="Discuss failover.", topic_key="t"),
    ]
    coverage, retriever, client = _coverage_playbook(
        packs_by_topic={"t": _pack([("c1", "evidence")])},
        responses=[
            _verdict_response(
                [
                    ("q-1", "Covered", 0.9, ["c1"]),
                    ("q-2", "Partial", 0.5, ["c1"]),
                ]
            ),
        ],
    )
    generator = _StubGenerator(items=items)
    playbook = QualityAuditPlaybook(criteria_generator=generator, coverage_audit=coverage)

    report = playbook.run("L0 kernel spec")

    assert isinstance(report, QualityReport)
    assert report.doc_type == "L0 kernel spec"
    assert report.criteria == items
    assert [v.item_id for v in report.coverage.verdicts] == ["q-1", "q-2"]
    assert report.coverage.verdicts[0].verdict == "Covered"
    # The generator saw the doc_type; the coverage audit saw the criteria.
    assert generator.calls == ["L0 kernel spec"]
    assert retriever.calls == ["t"]
    assert len(client.calls) == 1


def test_criteria_round_trip_preserves_input_order() -> None:
    items = [
        ChecklistItem(id="q-3", text="x", topic_key="t"),
        ChecklistItem(id="q-1", text="y", topic_key="t"),
        ChecklistItem(id="q-2", text="z", topic_key="t"),
    ]
    coverage, _, _ = _coverage_playbook(
        packs_by_topic={"t": _pack([("c1", "e")])},
        responses=[
            _verdict_response(
                [
                    ("q-3", "Covered", 0.9, ["c1"]),
                    ("q-1", "Covered", 0.9, ["c1"]),
                    ("q-2", "Covered", 0.9, ["c1"]),
                ]
            ),
        ],
    )
    playbook = QualityAuditPlaybook(
        criteria_generator=_StubGenerator(items=items),
        coverage_audit=coverage,
    )
    report = playbook.run("any")
    assert [c.id for c in report.criteria] == ["q-3", "q-1", "q-2"]
    assert [v.item_id for v in report.coverage.verdicts] == ["q-3", "q-1", "q-2"]


# --- edge cases ---


def test_empty_doc_type_rejected() -> None:
    coverage, _, _ = _coverage_playbook(packs_by_topic={}, responses=[])
    playbook = QualityAuditPlaybook(
        criteria_generator=_StubGenerator(items=[]),
        coverage_audit=coverage,
    )
    with pytest.raises(ValueError, match="doc_type"):
        playbook.run("   ")


def test_empty_criteria_yields_empty_coverage_report() -> None:
    """A generator that returns no criteria → coverage audit short-circuits."""
    coverage, retriever, client = _coverage_playbook(packs_by_topic={}, responses=[])
    playbook = QualityAuditPlaybook(
        criteria_generator=_StubGenerator(items=[]),
        coverage_audit=coverage,
    )
    report = playbook.run("any")
    assert report.criteria == []
    assert report.coverage.verdicts == []
    # The coverage audit never invoked its dependencies.
    assert retriever.calls == []
    assert client.calls == []


# --- model invariants ---


def test_quality_report_is_frozen() -> None:
    report = QualityReport(doc_type="x", criteria=[], coverage=CoverageReport(verdicts=[]))
    with pytest.raises(ValidationError):
        report.doc_type = "y"  # type: ignore[misc]


# --- protocol ---


def test_criteria_generator_protocol_isinstance() -> None:
    stub = _StubGenerator(items=[])
    assert isinstance(stub, CriteriaGenerator)


# --- heuristic reference ---


def test_heuristic_generator_returns_deterministic_items_for_same_doc_type() -> None:
    gen = HeuristicCriteriaGenerator()
    a = gen.generate("L0 kernel spec")
    b = gen.generate("L0 kernel spec")
    assert a == b
    assert all(isinstance(item, ChecklistItem) for item in a)
    # Some non-trivial number of criteria so playbook tests have material.
    assert len(a) >= 3


def test_heuristic_generator_differentiates_doc_types() -> None:
    gen = HeuristicCriteriaGenerator()
    spec = gen.generate("L0 kernel spec")
    rfc = gen.generate("RFC")
    # The doc_type appears in the topic_key naming so different inputs
    # produce structurally distinct outputs.
    spec_keys = {item.topic_key for item in spec}
    rfc_keys = {item.topic_key for item in rfc}
    assert spec_keys != rfc_keys


def test_heuristic_generator_rejects_blank_doc_type() -> None:
    gen = HeuristicCriteriaGenerator()
    with pytest.raises(ValueError, match="doc_type"):
        gen.generate("   ")


def test_heuristic_generator_ids_are_unique_for_one_call() -> None:
    items = HeuristicCriteriaGenerator().generate("L0 kernel spec")
    ids = [item.id for item in items]
    assert len(ids) == len(set(ids))


# --- end-to-end with the heuristic generator ---


def test_quality_audit_with_heuristic_generator_runs_coverage_per_topic() -> None:
    """Use the heuristic generator and a stub retriever sized to its
    topic_keys to confirm the full pipeline composes."""
    gen = HeuristicCriteriaGenerator()
    items = gen.generate("L0 kernel spec")
    topics = {item.topic_key for item in items}

    packs = {topic: _pack([(f"c-{topic}", f"evidence for {topic}")]) for topic in topics}
    # One batched response per cluster, marking every item Covered.
    responses: list[str] = []
    seen: list[str] = []
    by_topic: dict[str, list[ChecklistItem]] = {}
    for item in items:
        by_topic.setdefault(item.topic_key, []).append(item)
    for topic, cluster_items in by_topic.items():
        responses.append(
            _verdict_response([(item.id, "Covered", 0.9, [f"c-{topic}"]) for item in cluster_items])
        )
        seen.append(topic)

    coverage, _, _ = _coverage_playbook(packs_by_topic=packs, responses=responses)
    playbook = QualityAuditPlaybook(criteria_generator=gen, coverage_audit=coverage)
    report = playbook.run("L0 kernel spec")

    assert len(report.coverage.verdicts) == len(items)
    assert all(v.verdict == "Covered" for v in report.coverage.verdicts)
