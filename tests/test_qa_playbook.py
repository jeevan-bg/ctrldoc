"""UC1 `qa` playbook — retrieve → generate → decompose → verify.

The playbook stitches together existing primitives (retriever,
stateless task runner, claim decomposer, claim verifier) into a
single `run(query)` call that returns an `AnswerReport`. The test
suite swaps each dependency for a stub so we can observe the exact
composition without touching the network.

SPEC-REF: §5.1 (UC1 qa playbook)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest
from pydantic import ValidationError

from ctrldoc.assembler import CacheablePrefix
from ctrldoc.models import Claim, EvidencePack, Span
from ctrldoc.orch.task import StatelessTaskRunner, TaskInput
from ctrldoc.playbooks.qa import AnswerReport, QAPlaybook, QARetriever

# --- stubs ---


@dataclass
class _StubRetriever:
    pack: EvidencePack
    calls: list[str] = field(default_factory=list)

    def retrieve(self, query: str) -> EvidencePack:
        self.calls.append(query)
        return self.pack


@dataclass
class _StubTaskClient:
    response: str
    calls: list[tuple[str, str]] = field(default_factory=list)

    def call(self, *, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.response


@dataclass
class _StubDecomposer:
    """Returns the configured per-answer claim list."""

    claims_by_answer: dict[str, list[str]]
    calls: list[str] = field(default_factory=list)

    def decompose(self, text: str) -> list[str]:
        self.calls.append(text)
        return list(self.claims_by_answer.get(text, []))


@dataclass
class _StubVerifier:
    """Verifies claims using a fixed per-claim verdict table."""

    verdicts: dict[str, Claim]
    calls: list[str] = field(default_factory=list)

    def verify(self, claim_text: str) -> Claim:
        self.calls.append(claim_text)
        if claim_text in self.verdicts:
            return self.verdicts[claim_text]
        # Default: refuse claims we have no fixture for.
        return Claim(
            text=claim_text,
            citations=[],
            verified=False,
            confidence=0.0,
            nli_score=0.0,
            judge_score=0.0,
        )


# --- fixtures ---


def _prefix() -> CacheablePrefix:
    return CacheablePrefix(
        system_prompt="You are a QA writer.",
        doc_skeleton="# §1 Aurora\n\nintroduces consistent hashing.",
        entity_glossary="- **aurora** [system]",
    )


def _evidence_pack() -> EvidencePack:
    return EvidencePack(
        query="anything",
        spans=[
            Span(
                chunk_id="c1",
                char_start=0,
                char_end=42,
                text="Aurora uses consistent hashing across nodes.",
            ),
            Span(chunk_id="c2", char_start=0, char_end=20, text="Sections lack rollback guidance."),
        ],
        token_count=20,
        retrieval_plan=["search(query, view=dense, k=8)"],
    )


def _verified_claim(text: str) -> Claim:
    return Claim(
        text=text,
        citations=[
            Span(
                chunk_id="c1",
                char_start=0,
                char_end=42,
                text="Aurora uses consistent hashing across nodes.",
            )
        ],
        verified=True,
        confidence=0.92,
        nli_score=0.95,
        judge_score=0.9,
    )


def _refused_claim(text: str) -> Claim:
    return Claim(
        text=text,
        citations=[],
        verified=False,
        confidence=0.0,
        nli_score=0.1,
        judge_score=0.2,
    )


def _playbook(
    *,
    retriever: QARetriever,
    task_client_response: str,
    claims_by_answer: dict[str, list[str]],
    verdicts: dict[str, Claim],
) -> tuple[QAPlaybook, _StubTaskClient, _StubDecomposer, _StubVerifier]:
    task_client = _StubTaskClient(response=task_client_response)
    decomposer = _StubDecomposer(claims_by_answer=claims_by_answer)
    verifier = _StubVerifier(verdicts=verdicts)
    runner = StatelessTaskRunner(client=task_client)
    playbook = QAPlaybook(
        prefix=_prefix(),
        retriever=retriever,
        task_runner=runner,
        decomposer=decomposer,
        verifier=verifier,
    )
    return playbook, task_client, decomposer, verifier


# --- happy path ---


def test_qa_runs_retrieve_generate_decompose_verify() -> None:
    answer = "Aurora uses consistent hashing. Sections lack rollback guidance."
    retriever = _StubRetriever(pack=_evidence_pack())
    playbook, task_client, decomposer, verifier = _playbook(
        retriever=retriever,
        task_client_response=json.dumps({"answer": answer}),
        claims_by_answer={
            answer: [
                "Aurora uses consistent hashing.",
                "Sections lack rollback guidance.",
            ]
        },
        verdicts={
            "Aurora uses consistent hashing.": _verified_claim("Aurora uses consistent hashing."),
            "Sections lack rollback guidance.": _refused_claim("Sections lack rollback guidance."),
        },
    )

    report = playbook.run("What does Aurora use?")

    assert isinstance(report, AnswerReport)
    assert report.query == "What does Aurora use?"
    assert report.answer == answer
    assert len(report.claims) == 2
    assert report.claims[0].verified is True
    assert report.claims[1].verified is False

    # Pipeline composition:
    assert retriever.calls == ["What does Aurora use?"]
    assert len(task_client.calls) == 1
    assert decomposer.calls == [answer]
    assert verifier.calls == [
        "Aurora uses consistent hashing.",
        "Sections lack rollback guidance.",
    ]


# --- prompt layout ---


def test_generation_step_passes_evidence_text_and_query_to_task_client() -> None:
    answer = "An answer."
    retriever = _StubRetriever(pack=_evidence_pack())
    playbook, task_client, _, _ = _playbook(
        retriever=retriever,
        task_client_response=json.dumps({"answer": answer}),
        claims_by_answer={answer: []},
        verdicts={},
    )

    playbook.run("Query-MARKER")

    system, user = task_client.calls[0]
    # System is the rendered cacheable prefix.
    assert system == _prefix().render()
    # User carries the query and the evidence text.
    assert "Query-MARKER" in user
    assert "Aurora uses consistent hashing across nodes." in user
    assert "Sections lack rollback guidance." in user


def test_evidence_rendering_uses_stable_span_handles() -> None:
    """Each span appears once and carries its chunk id so the model can cite."""
    retriever = _StubRetriever(pack=_evidence_pack())
    playbook, task_client, _, _ = _playbook(
        retriever=retriever,
        task_client_response=json.dumps({"answer": ""}),
        claims_by_answer={"": []},
        verdicts={},
    )
    playbook.run("q")
    user = task_client.calls[0][1]
    # chunk handles surface as ASCII markers.
    assert "c1" in user
    assert "c2" in user


# --- short-circuits ---


def test_empty_query_short_circuits_without_dependencies() -> None:
    """A blank query must never burn LLM tokens — short-circuit cleanly."""
    retriever = _StubRetriever(pack=_evidence_pack())
    playbook, task_client, decomposer, verifier = _playbook(
        retriever=retriever,
        task_client_response="should not be called",
        claims_by_answer={},
        verdicts={},
    )

    report = playbook.run("   ")

    assert report.answer == ""
    assert report.claims == []
    assert retriever.calls == []
    assert task_client.calls == []
    assert decomposer.calls == []
    assert verifier.calls == []


def test_empty_answer_skips_decompose_and_verify() -> None:
    """If the generator returns an empty answer string there is nothing
    to decompose or verify — the playbook still returns a valid report."""
    retriever = _StubRetriever(pack=_evidence_pack())
    playbook, task_client, decomposer, verifier = _playbook(
        retriever=retriever,
        task_client_response=json.dumps({"answer": ""}),
        claims_by_answer={"": []},
        verdicts={},
    )

    report = playbook.run("q")

    assert report.answer == ""
    assert report.claims == []
    assert len(task_client.calls) == 1  # generation step still fired
    assert decomposer.calls == [""]
    assert verifier.calls == []


# --- AnswerReport model ---


def test_answer_report_is_frozen() -> None:
    report = AnswerReport(query="q", answer="a", claims=[])
    with pytest.raises(ValidationError):
        report.query = "mutated"  # type: ignore[misc]


def test_answer_report_claims_list_round_trip() -> None:
    claim = _verified_claim("x")
    report = AnswerReport(query="q", answer="x", claims=[claim])
    assert report.claims == [claim]


# --- determinism ---


def test_identical_query_produces_identical_task_client_args() -> None:
    """Cache stability across two runs of the same query (same retrieval result)."""
    retriever = _StubRetriever(pack=_evidence_pack())
    playbook, task_client, _, _ = _playbook(
        retriever=retriever,
        task_client_response=json.dumps({"answer": "a"}),
        claims_by_answer={"a": []},
        verdicts={},
    )
    playbook.run("q")
    playbook.run("q")
    assert task_client.calls[0] == task_client.calls[1]


# --- contracts ---


def test_qa_retriever_protocol_isinstance_check() -> None:
    stub = _StubRetriever(pack=_evidence_pack())
    assert isinstance(stub, QARetriever)


# --- direct runner injection ---


def test_qa_playbook_accepts_any_task_runner_client_combo() -> None:
    """The playbook holds a StatelessTaskRunner reference, so callers can
    swap the underlying TaskClient (Anthropic, Ollama, stub, tier-router)
    without touching playbook code."""
    retriever = _StubRetriever(pack=_evidence_pack())
    task_client = _StubTaskClient(response=json.dumps({"answer": "a"}))
    runner = StatelessTaskRunner(client=task_client)
    decomposer = _StubDecomposer(claims_by_answer={"a": []})
    verifier = _StubVerifier(verdicts={})
    playbook = QAPlaybook(
        prefix=_prefix(),
        retriever=retriever,
        task_runner=runner,
        decomposer=decomposer,
        verifier=verifier,
    )
    report = playbook.run("q")
    assert report.answer == "a"


# --- TaskInput shape ---


def test_generation_task_input_carries_prefix_and_query_only() -> None:
    """The TaskInput passed to the runner should not leak retrieval-internal
    structure (Plan trace, span IDs) beyond what's in the evidence text."""
    retriever = _StubRetriever(pack=_evidence_pack())
    task_client = _StubTaskClient(response=json.dumps({"answer": "a"}))
    runner = StatelessTaskRunner(client=task_client)

    playbook = QAPlaybook(
        prefix=_prefix(),
        retriever=retriever,
        task_runner=runner,
        decomposer=_StubDecomposer(claims_by_answer={"a": []}),
        verifier=_StubVerifier(verdicts={}),
    )
    playbook.run("q")

    user = task_client.calls[0][1]
    # No raw Plan DSL ("search(...)" appears in the retrieval_plan list,
    # but not in the prompt — the model sees evidence, not the plan).
    assert "search(query" not in user


# --- TaskInput type sanity ---


def test_task_input_type_is_used_consistently() -> None:
    """Ensure the playbook constructs a TaskInput (not a dict or other shape)."""
    retriever = _StubRetriever(pack=_evidence_pack())
    captured: list[TaskInput] = []

    class _CapturingRunner(StatelessTaskRunner):
        def run(self, task: TaskInput, *, output_model):  # type: ignore[override]
            captured.append(task)
            return output_model.model_validate({"answer": "ok"})

    playbook = QAPlaybook(
        prefix=_prefix(),
        retriever=retriever,
        task_runner=_CapturingRunner(client=_StubTaskClient(response="{}")),
        decomposer=_StubDecomposer(claims_by_answer={"ok": []}),
        verifier=_StubVerifier(verdicts={}),
    )
    playbook.run("q")
    assert len(captured) == 1
    assert isinstance(captured[0], TaskInput)
    assert captured[0].task_input == "q"
