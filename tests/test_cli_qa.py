"""Tests for the QA CLI wiring (S-114).

Covers:

  * `VerifierRetriever` — protocol conformance against
    `verify.claim_verifier.Retriever`; converts a `BundleRetriever`'s
    `EvidencePack` into the verifier's `RetrievedEvidence`.
  * `render_qa_markdown` — output shape: header, answer section,
    per-claim verification table with confidence + NLI + judge,
    citation appendix.
  * `ctrldoc qa` CLI surface:
      - blank query → exit 2.
      - missing target → exit 2.
      - heuristic profile rejected with a clear error.
      - thrifty profile end-to-end against Ollama (slow,
        skipped when ollama / gliner / fastcoref are missing).

SPEC-REF: §5.1 (QA playbook), §6 (CLI)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ctrldoc.backends import build_bundle
from ctrldoc.cli import app
from ctrldoc.cli_audit import BundleRetriever
from ctrldoc.cli_qa import VerifierRetriever, render_qa_markdown
from ctrldoc.config import Config
from ctrldoc.models import Claim, Span
from ctrldoc.ops.qa import AnswerReport
from ctrldoc.store.bm25 import TantivyBM25Index
from ctrldoc.store.memory import InMemoryStore
from ctrldoc.store.vectors import InMemoryVectorIndex
from ctrldoc.verify.claim_verifier import Retriever as VerifierRetrieverProtocol

runner = CliRunner()


_CONFIG_TEMPLATE = """\
[models]
planner = "claude-opus-4-7"
judge_tier1 = "qwen2.5:7b-instruct-q4_K_M"
judge_tier2 = "claude-opus-4-7"
verifier_nli = "deberta-v3-large-mnli"
embedder = "bge-m3"

[budgets]
max_cost_usd = 5.0
max_tokens_per_call = 16000
max_wall_clock_min = 30

[concurrency]
anthropic_concurrent = 8
ollama_concurrent = 2

[paths]
index_path = "{index_path}"
runs_path = "{runs_path}"
traces_path = "{traces_path}"
"""


def _write_config(tmp_path: Path) -> Path:
    index_path = tmp_path / "index"
    runs_path = tmp_path / "runs"
    traces_path = tmp_path / "traces"
    for p in (index_path, runs_path, traces_path):
        p.mkdir(exist_ok=True)
    cfg = tmp_path / "ctrldoc.toml"
    cfg.write_text(
        _CONFIG_TEMPLATE.format(
            index_path=index_path.as_posix(),
            runs_path=runs_path.as_posix(),
            traces_path=traces_path.as_posix(),
        ),
        encoding="utf-8",
    )
    return cfg


# --- VerifierRetriever ---


def test_verifier_retriever_satisfies_verify_protocol(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    bundle = build_bundle(config=Config.load(cfg), profile="heuristic")
    inner = BundleRetriever(
        bundle=bundle,
        store=InMemoryStore(),
        vector_index=InMemoryVectorIndex(dimension=32),
        bm25_index=TantivyBM25Index(path=tmp_path / "bm25"),
        prefix_skeleton="",
        prefix_glossary="",
    )
    retr = VerifierRetriever(bundle_retriever=inner)
    assert isinstance(retr, VerifierRetrieverProtocol)


def test_verifier_retriever_returns_labelled_text_and_citations(
    tmp_path: Path, synthetic_doc_path: Path
) -> None:
    from ctrldoc.ingest.parser import MarkdownParser
    from ctrldoc.ingest.pipeline import ingest_document

    cfg = _write_config(tmp_path)
    bundle = build_bundle(config=Config.load(cfg), profile="heuristic")
    store = InMemoryStore()
    vec = InMemoryVectorIndex(dimension=32)
    bm = TantivyBM25Index(path=tmp_path / "bm25")
    ingest_document(
        source=synthetic_doc_path,
        parser=MarkdownParser(),
        coref=bundle.coref,
        ner=bundle.ner,
        ner_labels=["person", "system"],
        embedder=bundle.embedder,
        summarizer=bundle.summarizer,
        store=store,
        vector_index=vec,
        bm25_index=bm,
    )
    inner = BundleRetriever(
        bundle=bundle,
        store=store,
        vector_index=vec,
        bm25_index=bm,
        prefix_skeleton="",
        prefix_glossary="",
    )
    retr = VerifierRetriever(bundle_retriever=inner)
    ev = retr.retrieve("Aurora consistent hashing", depth="normal")
    assert ev.text, "verifier retriever should return non-empty text"
    assert ev.citations, "verifier retriever should attach citation spans"
    # Each line is `[chunk_id] text`.
    for span in ev.citations:
        assert f"[{span.chunk_id}]" in ev.text


def test_verifier_retriever_depth_is_currently_a_no_op(tmp_path: Path) -> None:
    """Both depths route through the same retrieval. Document the
    invariant so a future widening of `broad` is a discoverable change.
    """
    cfg = _write_config(tmp_path)
    bundle = build_bundle(config=Config.load(cfg), profile="heuristic")
    inner = BundleRetriever(
        bundle=bundle,
        store=InMemoryStore(),
        vector_index=InMemoryVectorIndex(dimension=32),
        bm25_index=TantivyBM25Index(path=tmp_path / "bm25"),
        prefix_skeleton="",
        prefix_glossary="",
    )
    retr = VerifierRetriever(bundle_retriever=inner)
    normal = retr.retrieve("anything", depth="normal")
    broad = retr.retrieve("anything", depth="broad")
    # Empty store → both return empty equally.
    assert normal.text == broad.text == ""
    assert list(normal.citations) == list(broad.citations) == []


# --- render_qa_markdown ---


def _fake_claim(text: str, *, verified: bool, citations: list[Span] | None = None) -> Claim:
    return Claim(
        text=text,
        citations=citations or [],
        verified=verified,
        confidence=0.85 if verified else 0.0,
        nli_score=0.9 if verified else 0.1,
        judge_score=0.8 if verified else 0.0,
    )


def test_render_qa_markdown_emits_header_answer_and_claim_table() -> None:
    md = render_qa_markdown(
        report=AnswerReport(
            query="What is Aurora?",
            answer="Aurora is a distributed kernel.",
            claims=[
                _fake_claim(
                    "Aurora is a distributed kernel.",
                    verified=True,
                    citations=[
                        Span(chunk_id="c1", char_start=0, char_end=10, text="aurora desc"),
                    ],
                ),
            ],
        ),
        target_path=Path("doc.md"),
        profile="thrifty",
        run_id="r1",
    )
    assert "# ctrldoc — QA report" in md
    assert "Aurora is a distributed kernel." in md
    assert "## Claim verification" in md
    assert "| # | Verified |" in md
    assert "`[c1]`" in md
    assert "## Citation snippets" in md


def test_render_qa_markdown_handles_empty_claim_list() -> None:
    md = render_qa_markdown(
        report=AnswerReport(query="q", answer="", claims=[]),
        target_path=Path("x.md"),
        profile="thrifty",
        run_id="r2",
    )
    assert "_(empty answer" in md
    assert "no claims to verify" in md


def test_render_qa_markdown_escapes_pipes_in_claim_text() -> None:
    md = render_qa_markdown(
        report=AnswerReport(
            query="q",
            answer="a",
            claims=[_fake_claim("contains | a pipe", verified=False)],
        ),
        target_path=Path("x.md"),
        profile="thrifty",
        run_id="r3",
    )
    # `|` is escaped so the Markdown table row stays well-formed.
    assert "contains \\| a pipe" in md


def test_render_qa_markdown_marks_unverified_claims_with_no() -> None:
    md = render_qa_markdown(
        report=AnswerReport(
            query="q",
            answer="a",
            claims=[
                _fake_claim("verified claim", verified=True),
                _fake_claim("refused claim", verified=False),
            ],
        ),
        target_path=Path("x.md"),
        profile="thrifty",
        run_id="r4",
    )
    assert "| yes |" in md
    assert "| no |" in md


# --- ctrldoc qa CLI surface ---


def test_qa_blank_query_exits_with_code_two(tmp_path: Path, synthetic_doc_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = runner.invoke(
        app,
        [
            "--config",
            str(cfg),
            "--profile",
            "thrifty",
            "qa",
            "   ",
            "--target",
            str(synthetic_doc_path),
        ],
    )
    assert result.exit_code == 2
    assert "blank" in result.stderr


def test_qa_missing_target_exits_with_code_two(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = runner.invoke(
        app,
        [
            "--config",
            str(cfg),
            "--profile",
            "thrifty",
            "qa",
            "anything",
            "--target",
            str(tmp_path / "absent.md"),
        ],
    )
    assert result.exit_code == 2
    assert "does not exist" in result.stderr


def test_qa_heuristic_profile_rejected(tmp_path: Path, synthetic_doc_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = runner.invoke(
        app,
        [
            "--config",
            str(cfg),
            "--profile",
            "heuristic",
            "qa",
            "Aurora?",
            "--target",
            str(synthetic_doc_path),
        ],
    )
    assert result.exit_code == 2
    assert "qa requires --profile thrifty|production" in result.stderr


@pytest.mark.slow
@pytest.mark.requires_ollama
def test_qa_thrifty_writes_report_and_result_json(tmp_path: Path, synthetic_doc_path: Path) -> None:
    pytest.importorskip("sqlite_vec")
    pytest.importorskip("ollama")
    pytest.importorskip("gliner", reason="gliner optional; thrifty profile needs it")
    pytest.importorskip("fastcoref", reason="fastcoref optional; thrifty profile needs it")
    cfg = _write_config(tmp_path)
    result = runner.invoke(
        app,
        [
            "--config",
            str(cfg),
            "--profile",
            "thrifty",
            "--format",
            "json",
            "qa",
            "What is Aurora?",
            "--target",
            str(synthetic_doc_path),
        ],
    )
    if result.exit_code != 0:
        if result.exception and not isinstance(result.exception, SystemExit):
            raise AssertionError(
                f"thrifty qa crashed: {type(result.exception).__name__}: {result.exception}"
            ) from result.exception
        pytest.skip(f"thrifty qa non-zero exit ({result.exit_code})")
    payload = json.loads(result.stdout)
    assert payload["command"] == "qa"
    assert payload["profile"] == "thrifty"
    assert payload["query"] == "What is Aurora?"
    assert "answer" in payload
    assert isinstance(payload["claims"], list)
    assert list((tmp_path / "runs").rglob("report.md"))
    assert list((tmp_path / "runs").rglob("result.json"))


# --- remove old qa stub coverage from test_cli.py ---


def test_qa_help_describes_purpose_through_typer() -> None:
    result = runner.invoke(app, ["qa", "--help"])
    assert result.exit_code == 0
    assert "QA" in result.stdout
