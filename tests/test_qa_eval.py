"""qa_eval — runner that scores `QAPlaybook` runs through the harness.

Each case carries either a `gold_chunk_ids` set (positive case) or a
`should_refuse=True` flag (refusal case, §8.1 `qa_refusal` style).
Positive cases emit `citation_precision`; refusal cases emit
`refusal_accuracy`. Per §8.2 the thresholds are ≥0.95 and ≥0.90
respectively. The runner is fully tested with stubbed `QAPlaybook`
dependencies; the JSONL set in `tests/eval/qa_eval.jsonl` is a
starter scaffold that the §8.1 100-case curation task will extend.

SPEC-REF: §8.1 (qa_eval), §8.2 (qa metrics)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from ctrldoc.assembler import CacheablePrefix
from ctrldoc.eval.harness import EvalResult, load_jsonl_cases, run_eval
from ctrldoc.eval.qa import (
    QAEvalCase,
    QAEvalRunner,
    citation_precision,
)
from ctrldoc.models import Claim, EvidencePack, Span
from ctrldoc.ops.qa import QAPlaybook
from ctrldoc.orch.task import StatelessTaskRunner

# --- stubbed QAPlaybook ---


@dataclass
class _StubRetriever:
    pack: EvidencePack

    def retrieve(self, query: str) -> EvidencePack:
        return self.pack


@dataclass
class _StubClient:
    response: str

    def call(self, *, system: str, user: str) -> str:
        return self.response


@dataclass
class _StubDecomposer:
    claims_by_answer: dict[str, list[str]]

    def decompose(self, text: str) -> list[str]:
        return list(self.claims_by_answer.get(text, []))


@dataclass
class _StubVerifier:
    verdicts: dict[str, Claim] = field(default_factory=dict)
    fallback_refused: bool = False

    def verify(self, claim_text: str) -> Claim:
        if claim_text in self.verdicts:
            return self.verdicts[claim_text]
        return Claim(
            text=claim_text,
            citations=[],
            verified=not self.fallback_refused,
            confidence=0.0 if self.fallback_refused else 0.5,
            nli_score=0.0 if self.fallback_refused else 0.5,
            judge_score=0.0 if self.fallback_refused else 0.5,
        )


def _prefix() -> CacheablePrefix:
    return CacheablePrefix(
        system_prompt="qa",
        doc_skeleton="# §1",
        entity_glossary="- **e/1** [concept]",
    )


def _pack(spans: list[tuple[str, str]]) -> EvidencePack:
    return EvidencePack(
        query="any",
        spans=[
            Span(chunk_id=cid, char_start=0, char_end=len(text), text=text) for cid, text in spans
        ],
        token_count=20,
        retrieval_plan=[],
    )


def _verified(text: str, chunk_id: str) -> Claim:
    return Claim(
        text=text,
        citations=[Span(chunk_id=chunk_id, char_start=0, char_end=len(text), text=text)],
        verified=True,
        confidence=0.9,
        nli_score=0.9,
        judge_score=0.9,
    )


def _refused(text: str) -> Claim:
    return Claim(
        text=text,
        citations=[],
        verified=False,
        confidence=0.0,
        nli_score=0.0,
        judge_score=0.0,
    )


def _playbook(
    *,
    pack: EvidencePack,
    answer: str,
    claims: list[str],
    verdicts: dict[str, Claim],
) -> QAPlaybook:
    return QAPlaybook(
        prefix=_prefix(),
        retriever=_StubRetriever(pack=pack),
        task_runner=StatelessTaskRunner(
            client=_StubClient(response=json.dumps({"answer": answer}))
        ),
        decomposer=_StubDecomposer(claims_by_answer={answer: claims}),
        verifier=_StubVerifier(verdicts=verdicts),
    )


# --- citation_precision helper ---


def test_citation_precision_pure_hit_returns_one() -> None:
    cited = ["c1", "c2"]
    gold = {"c1", "c2"}
    assert citation_precision(cited, gold) == pytest.approx(1.0)


def test_citation_precision_mixed_hit_returns_fraction() -> None:
    cited = ["c1", "c2", "c-ghost"]
    gold = {"c1", "c2"}
    assert citation_precision(cited, gold) == pytest.approx(2 / 3)


def test_citation_precision_no_citations_returns_zero() -> None:
    assert citation_precision([], {"c1"}) == pytest.approx(0.0)


def test_citation_precision_dedupes_repeated_cited_ids() -> None:
    """A claim that cites the same chunk twice shouldn't double-count."""
    cited = ["c1", "c1", "c2"]
    gold = {"c1"}
    assert citation_precision(cited, gold) == pytest.approx(0.5)


# --- runner: positive case ---


def test_positive_case_with_clean_citations_emits_precision_one() -> None:
    playbook = _playbook(
        pack=_pack([("c1", "Aurora hashing")]),
        answer="Aurora uses consistent hashing.",
        claims=["Aurora uses consistent hashing."],
        verdicts={
            "Aurora uses consistent hashing.": _verified("Aurora uses consistent hashing.", "c1")
        },
    )
    runner = QAEvalRunner(playbook=playbook)
    case = QAEvalCase(
        id="qa-1",
        question="Does Aurora use consistent hashing?",
        gold_chunk_ids=["c1"],
        should_refuse=False,
    )
    result = runner.run_case(case)
    assert isinstance(result, EvalResult)
    assert result.case_id == "qa-1"
    assert result.metrics["citation_precision"] == pytest.approx(1.0)
    assert result.passed is True


def test_positive_case_with_hallucinated_citation_lowers_precision() -> None:
    playbook = _playbook(
        pack=_pack([("c1", "Aurora hashing"), ("c-ghost", "irrelevant")]),
        answer="Aurora uses consistent hashing.",
        claims=["Aurora uses consistent hashing.", "Aurora ships next quarter."],
        verdicts={
            "Aurora uses consistent hashing.": _verified("Aurora uses consistent hashing.", "c1"),
            "Aurora ships next quarter.": _verified("Aurora ships next quarter.", "c-ghost"),
        },
    )
    runner = QAEvalRunner(playbook=playbook)
    case = QAEvalCase(
        id="qa-2",
        question="What is Aurora?",
        gold_chunk_ids=["c1"],
        should_refuse=False,
    )
    result = runner.run_case(case)
    assert result.metrics["citation_precision"] == pytest.approx(0.5)
    # Below the §8.2 0.95 threshold ⇒ case does not pass.
    assert result.passed is False


def test_positive_case_unverified_claims_excluded_from_citations() -> None:
    """Only verified claims contribute citations — that's the §4.4 rule."""
    playbook = _playbook(
        pack=_pack([("c1", "real"), ("c-bad", "wrong")]),
        answer="Two-claim answer.",
        claims=["good claim", "bad claim"],
        verdicts={
            "good claim": _verified("good claim", "c1"),
            "bad claim": _refused("bad claim"),  # never reaches the citation set
        },
    )
    runner = QAEvalRunner(playbook=playbook)
    case = QAEvalCase(
        id="qa-3",
        question="q",
        gold_chunk_ids=["c1"],
        should_refuse=False,
    )
    result = runner.run_case(case)
    assert result.metrics["citation_precision"] == pytest.approx(1.0)


# --- runner: refusal case ---


def test_refusal_case_correctly_refused_passes() -> None:
    playbook = _playbook(
        pack=_pack([("c1", "unrelated")]),
        answer="Some out-of-doc answer.",
        claims=["fabricated claim"],
        verdicts={"fabricated claim": _refused("fabricated claim")},
    )
    runner = QAEvalRunner(playbook=playbook)
    case = QAEvalCase(
        id="qa-r1",
        question="Out-of-doc trivia",
        gold_chunk_ids=[],
        should_refuse=True,
    )
    result = runner.run_case(case)
    assert result.metrics["refusal_accuracy"] == pytest.approx(1.0)
    assert result.passed is True


def test_refusal_case_with_verified_claim_fails() -> None:
    """A refusal case that yielded *any* verified claim is a fabrication."""
    playbook = _playbook(
        pack=_pack([("c1", "evidence")]),
        answer="Confident wrong answer.",
        claims=["fabricated claim"],
        verdicts={"fabricated claim": _verified("fabricated claim", "c1")},
    )
    runner = QAEvalRunner(playbook=playbook)
    case = QAEvalCase(
        id="qa-r2",
        question="Out-of-doc trivia",
        gold_chunk_ids=[],
        should_refuse=True,
    )
    result = runner.run_case(case)
    assert result.metrics["refusal_accuracy"] == pytest.approx(0.0)
    assert result.passed is False


def test_refusal_case_empty_answer_counts_as_refusal() -> None:
    """The QAPlaybook short-circuits empty queries → empty answer + claims.
    That shape is itself a valid refusal."""
    playbook = _playbook(
        pack=_pack([]),
        answer="",
        claims=[],
        verdicts={},
    )
    runner = QAEvalRunner(playbook=playbook)
    case = QAEvalCase(
        id="qa-r3",
        question="anything",
        gold_chunk_ids=[],
        should_refuse=True,
    )
    result = runner.run_case(case)
    assert result.metrics["refusal_accuracy"] == pytest.approx(1.0)
    assert result.passed is True


# --- metric isolation between case types ---


def test_positive_and_refusal_metrics_are_disjoint_in_each_result() -> None:
    """Per-case results only carry the metric appropriate to the case
    type — the aggregate then averages each over its own subset."""
    pos_playbook = _playbook(
        pack=_pack([("c1", "x")]),
        answer="a",
        claims=["a"],
        verdicts={"a": _verified("a", "c1")},
    )
    ref_playbook = _playbook(
        pack=_pack([]),
        answer="",
        claims=[],
        verdicts={},
    )
    pos = QAEvalRunner(playbook=pos_playbook).run_case(
        QAEvalCase(id="qa-pos", question="q", gold_chunk_ids=["c1"], should_refuse=False)
    )
    ref = QAEvalRunner(playbook=ref_playbook).run_case(
        QAEvalCase(id="qa-ref", question="q", gold_chunk_ids=[], should_refuse=True)
    )
    assert "citation_precision" in pos.metrics and "refusal_accuracy" not in pos.metrics
    assert "refusal_accuracy" in ref.metrics and "citation_precision" not in ref.metrics


# --- end-to-end via the harness ---


def test_eval_aggregate_mixes_positive_and_refusal_cases() -> None:
    """Run both case types through the harness and check the aggregate
    surfaces precision and refusal-accuracy independently."""

    @dataclass
    class _DispatchRunner:
        def run_case(self, case: QAEvalCase) -> EvalResult:
            if case.should_refuse:
                pb = _playbook(pack=_pack([]), answer="", claims=[], verdicts={})
            else:
                pb = _playbook(
                    pack=_pack([("c1", "x")]),
                    answer="a",
                    claims=["a"],
                    verdicts={"a": _verified("a", "c1")},
                )
            return QAEvalRunner(playbook=pb).run_case(case)

    cases = [
        QAEvalCase(id="p-1", question="q", gold_chunk_ids=["c1"], should_refuse=False),
        QAEvalCase(id="p-2", question="q", gold_chunk_ids=["c1"], should_refuse=False),
        QAEvalCase(id="r-1", question="q", gold_chunk_ids=[], should_refuse=True),
        QAEvalCase(id="r-2", question="q", gold_chunk_ids=[], should_refuse=True),
    ]
    report = run_eval(
        set_name="qa_eval",
        cases=cases,
        runner=_DispatchRunner(),
        thresholds={"citation_precision": 0.95, "refusal_accuracy": 0.90},
    )
    assert report.passed is True
    assert report.aggregate["citation_precision"] == pytest.approx(1.0)
    assert report.aggregate["refusal_accuracy"] == pytest.approx(1.0)


# --- JSONL set loads ---


def test_starter_jsonl_set_loads_under_qa_eval_case_schema() -> None:
    path = Path(__file__).parent / "eval" / "qa_eval.jsonl"
    cases = load_jsonl_cases(path, case_model=QAEvalCase)
    # Sanity: a non-trivial number of cases, mix of positive + refusal.
    assert len(cases) >= 4
    assert any(c.should_refuse for c in cases)
    assert any(not c.should_refuse for c in cases)


def test_case_schema_is_frozen_with_extra_forbid() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        QAEvalCase(
            id="x",
            question="q",
            gold_chunk_ids=[],
            should_refuse=False,
            extra_field="bad",  # type: ignore[call-arg]
        )

    case = QAEvalCase(id="x", question="q", gold_chunk_ids=[], should_refuse=False)
    with pytest.raises(ValidationError):
        case.question = "y"  # type: ignore[misc]
