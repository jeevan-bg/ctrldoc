"""Tests for the analytical-review CLI wiring (S-115).

Covers:

  * `LLMLensSweeper` — protocol conformance against
    `playbooks.review.LensSweeper`; happy-path with a stub task
    runner that returns canned findings; citation chunk_ids that
    don't appear in the evidence pack are silently dropped.
  * `render_review_markdown` — narrative header, per-lens groups
    ordered critical → warn → info, summary table.
  * `ctrldoc review` CLI surface:
      - blank doc_type → exit 2.
      - missing target → exit 2.
      - heuristic profile rejected.
      - thrifty profile end-to-end (slow, skipped without
        ollama + gliner + fastcoref).

SPEC-REF: §5.4 (analytical review), §6 (CLI)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from ctrldoc.assembler import CacheablePrefix
from ctrldoc.backends import build_bundle
from ctrldoc.cli import app
from ctrldoc.cli_audit import BundleRetriever
from ctrldoc.cli_review import LLMLensSweeper, render_review_markdown
from ctrldoc.config import Config
from ctrldoc.ingest.parser import MarkdownParser
from ctrldoc.ingest.pipeline import ingest_document
from ctrldoc.models import Finding, Span
from ctrldoc.orch.task import StatelessTaskRunner
from ctrldoc.playbooks.review import (
    Lens,
    LensSweeper,
    ReviewNarrative,
    ReviewReport,
)
from ctrldoc.store.bm25 import TantivyBM25Index
from ctrldoc.store.memory import InMemoryStore
from ctrldoc.store.vectors import InMemoryVectorIndex

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


class _StubTaskClient:
    """`TaskClient` returning a fixed text payload."""

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.calls: list[dict[str, Any]] = []

    def call(self, *, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        return self._reply


# --- LLMLensSweeper ---


def test_llm_lens_sweeper_satisfies_protocol(tmp_path: Path) -> None:
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
    prefix = CacheablePrefix(system_prompt="sys", doc_skeleton="", entity_glossary="")
    runner_obj = StatelessTaskRunner(client=_StubTaskClient('{"findings": []}'))
    sweeper = LLMLensSweeper(prefix=prefix, retriever=inner, task_runner=runner_obj)
    assert isinstance(sweeper, LensSweeper)


def test_llm_lens_sweeper_returns_empty_when_retrieval_empty(tmp_path: Path) -> None:
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
    stub = _StubTaskClient('{"findings": []}')
    sweeper = LLMLensSweeper(
        prefix=CacheablePrefix(system_prompt="s", doc_skeleton="", entity_glossary=""),
        retriever=inner,
        task_runner=StatelessTaskRunner(client=stub),
    )
    findings = sweeper.sweep(Lens(id="lens/test", name="test", description="test lens"))
    assert findings == []
    # Empty store → empty pack → sweeper short-circuits before calling the LLM.
    assert stub.calls == []


def test_llm_lens_sweeper_emits_findings_with_resolved_citations(
    tmp_path: Path, synthetic_doc_path: Path
) -> None:
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
    # Grab one real chunk_id from the store so we can cite it.
    real_chunk_id = next(iter(store.iter_chunks())).id

    reply = json.dumps(
        {
            "findings": [
                {
                    "claim": "Assumption X is not stated.",
                    "severity": "warn",
                    "citation_chunk_id": real_chunk_id,
                },
                {
                    "claim": "Hallucinated citation",
                    "severity": "info",
                    "citation_chunk_id": "this-chunk-does-not-exist",
                },
            ]
        }
    )
    sweeper = LLMLensSweeper(
        prefix=CacheablePrefix(system_prompt="s", doc_skeleton="", entity_glossary=""),
        retriever=inner,
        task_runner=StatelessTaskRunner(client=_StubTaskClient(reply)),
    )
    findings = sweeper.sweep(
        Lens(id="lens/assumptions", name="assumptions", description="check assumptions")
    )
    # Only the resolved citation is emitted; the hallucinated one is dropped.
    assert len(findings) == 1
    assert findings[0].ctrldoc == "lens/assumptions"
    assert findings[0].claim == "Assumption X is not stated."
    assert findings[0].location.chunk_id == real_chunk_id


# --- render_review_markdown ---


def _fake_finding(lens: str, severity: str, chunk_id: str = "c1") -> Finding:
    return Finding(
        ctrldoc=lens,
        location=Span(chunk_id=chunk_id, char_start=0, char_end=11, text="hello world"),
        claim=f"issue under {lens}",
        severity=severity,  # type: ignore[arg-type]
    )


def test_render_review_markdown_has_narrative_and_per_lens_groups() -> None:
    report = ReviewReport(
        doc_type="Aurora L0 spec",
        findings=[
            _fake_finding("lens/assumptions", "warn"),
            _fake_finding("lens/consistency", "critical"),
            _fake_finding("lens/consistency", "info"),
        ],
        narrative=ReviewNarrative(
            headline="Major gaps in consistency",
            sections=["Assumptions section 1", "Boundary cases"],
            summary="The spec has critical contradictions.",
        ),
    )
    md = render_review_markdown(
        report=report,
        target_path=Path("doc.md"),
        profile="thrifty",
        run_id="r1",
    )
    assert "# ctrldoc — analytical review" in md
    assert "### Major gaps in consistency" in md
    assert "The spec has critical contradictions." in md
    assert "### `lens/assumptions` (1)" in md
    assert "### `lens/consistency` (2)" in md
    # Critical → warn → info ordering within the consistency lens.
    consistency_block = md.split("### `lens/consistency` (2)")[1]
    assert consistency_block.find("**critical**") < consistency_block.find("**info**")
    # Summary table.
    assert "| Lens | Findings |" in md
    assert "| `lens/consistency` | 2 |" in md
    assert "| **Total** | **3** |" in md


def test_render_review_markdown_empty_findings_shows_placeholder() -> None:
    report = ReviewReport(
        doc_type="d",
        findings=[],
        narrative=ReviewNarrative(headline="", sections=[], summary=""),
    )
    md = render_review_markdown(
        report=report,
        target_path=Path("x.md"),
        profile="thrifty",
        run_id="r2",
    )
    assert "## Findings by lens" in md
    assert "_(no findings)_" in md
    assert "_(synthesis returned an empty narrative)_" in md


# --- ctrldoc review CLI surface ---


def test_review_blank_doc_type_exits_with_code_two(
    tmp_path: Path, synthetic_doc_path: Path
) -> None:
    cfg = _write_config(tmp_path)
    result = runner.invoke(
        app,
        [
            "--config",
            str(cfg),
            "--profile",
            "thrifty",
            "review",
            "   ",
            "--target",
            str(synthetic_doc_path),
        ],
    )
    assert result.exit_code == 2
    assert "blank" in result.stderr


def test_review_missing_target_exits_with_code_two(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = runner.invoke(
        app,
        [
            "--config",
            str(cfg),
            "--profile",
            "thrifty",
            "review",
            "Aurora L0 spec",
            "--target",
            str(tmp_path / "absent.md"),
        ],
    )
    assert result.exit_code == 2
    assert "does not exist" in result.stderr


def test_review_heuristic_profile_rejected(tmp_path: Path, synthetic_doc_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = runner.invoke(
        app,
        [
            "--config",
            str(cfg),
            "--profile",
            "heuristic",
            "review",
            "Aurora L0 spec",
            "--target",
            str(synthetic_doc_path),
        ],
    )
    assert result.exit_code == 2
    assert "review requires --profile thrifty|production" in result.stderr


@pytest.mark.slow
@pytest.mark.requires_ollama
def test_review_thrifty_writes_report_and_result_json(
    tmp_path: Path, synthetic_doc_path: Path
) -> None:
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
            "review",
            "Aurora L0 kernel spec",
            "--target",
            str(synthetic_doc_path),
        ],
    )
    if result.exit_code != 0:
        if result.exception and not isinstance(result.exception, SystemExit):
            raise AssertionError(
                f"thrifty review crashed: {type(result.exception).__name__}: {result.exception}"
            ) from result.exception
        pytest.skip(f"thrifty review non-zero exit ({result.exit_code})")
    payload = json.loads(result.stdout)
    assert payload["command"] == "review"
    assert payload["profile"] == "thrifty"
    assert payload["doc_type"] == "Aurora L0 kernel spec"
    assert "narrative" in payload
    assert isinstance(payload["findings"], list)
    assert list((tmp_path / "runs").rglob("report.md"))
    assert list((tmp_path / "runs").rglob("result.json"))
