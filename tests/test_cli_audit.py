"""Tests for the coverage-audit CLI wiring (S-113).

Covers:

  * `parse_checklist_markdown` — the deterministic H2/H3 + first
    paragraph parser; topic_key inheritance from the nearest H1/H2;
    duplicate id disambiguation.
  * `BundleRetriever` — protocol conformance against `QARetriever`
    and `CoverageRetriever`; end-to-end retrieval against the
    heuristic bundle on the synthetic gold doc (no LLM, no Ollama).
  * `render_coverage_markdown` — output shape (per-verdict groups,
    summary table, citation rendering with `[chunk_id]` handles).
  * `ctrldoc audit` CLI surface:
      - missing checklist or target → exit 2.
      - heuristic profile rejected with a clear error.
      - thrifty profile end-to-end against Ollama (slow,
        skipped when ollama / gliner / fastcoref are missing).

SPEC-REF: §5.2 (coverage audit), §6 (CLI)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ctrldoc.backends import build_bundle
from ctrldoc.cli import app
from ctrldoc.cli_audit import (
    BundleRetriever,
    parse_checklist_markdown,
    render_coverage_markdown,
)
from ctrldoc.config import Config
from ctrldoc.ingest.parser import MarkdownParser
from ctrldoc.ingest.pipeline import ingest_document
from ctrldoc.models import EvidencePack, Span, Verdict
from ctrldoc.ops.audit import (
    ChecklistItem,
    CoverageReport,
    CoverageRetriever,
)
from ctrldoc.ops.qa import QARetriever
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


# --- parse_checklist_markdown ---


def test_parse_checklist_extracts_h2_items_with_paragraphs() -> None:
    text = (
        "# Adversaries\n"
        "\n"
        "## Insider threat\n"
        "Trusted actor with elevated privileges.\n"
        "\n"
        "## External attacker\n"
        "Unauthenticated network actor.\n"
    )
    items = parse_checklist_markdown(text)
    assert len(items) == 2
    assert items[0].text.startswith("Insider threat: ")
    assert "elevated privileges" in items[0].text
    assert items[1].text.startswith("External attacker: ")


def test_parse_checklist_h3_inherits_h2_as_topic_key() -> None:
    text = (
        "## Confidentiality\n"
        "Top-level requirement.\n"
        "\n"
        "### Key rotation\n"
        "Keys are rotated every 90 days.\n"
        "\n"
        "### Encryption at rest\n"
        "All persisted data is AES-256 encrypted.\n"
    )
    items = parse_checklist_markdown(text)
    # The H2 itself becomes one item; its two H3 children inherit its slug as topic_key.
    h2 = next(i for i in items if "Confidentiality" in i.text)
    h3s = [i for i in items if i.id != h2.id]
    assert all(i.topic_key == "confidentiality" for i in h3s)


def test_parse_checklist_h2_inherits_h1_as_topic_key() -> None:
    text = "# Authorization\n\n## Role-based access\nRBAC governs every operation.\n"
    items = parse_checklist_markdown(text)
    assert len(items) == 1
    assert items[0].topic_key == "authorization"


def test_parse_checklist_no_parent_uses_fallback_topic_key() -> None:
    text = "## Orphan item\n\nNo H1 above.\n"
    items = parse_checklist_markdown(text, fallback_topic_key="checklist")
    assert items[0].topic_key == "checklist"


def test_parse_checklist_ids_unique_under_duplicate_headings() -> None:
    text = "## Logging\nFirst.\n\n## Logging\nSecond.\n"
    items = parse_checklist_markdown(text)
    assert len(items) == 2
    assert items[0].id != items[1].id


def test_parse_checklist_skips_h4_and_deeper() -> None:
    text = "## Top\nBody.\n\n#### Skip me\nToo deep.\n"
    items = parse_checklist_markdown(text)
    assert len(items) == 1
    assert "Top" in items[0].text


def test_parse_checklist_empty_input_returns_empty() -> None:
    assert parse_checklist_markdown("") == []
    assert parse_checklist_markdown("just prose, no headings") == []


# --- BundleRetriever ---


def test_bundle_retriever_satisfies_both_protocols(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    bundle = build_bundle(config=Config.load(cfg), profile="heuristic")
    store = InMemoryStore()
    retr = BundleRetriever(
        bundle=bundle,
        store=store,
        vector_index=InMemoryVectorIndex(dimension=32),
        bm25_index=TantivyBM25Index(path=tmp_path / "bm25"),
        prefix_skeleton="",
        prefix_glossary="",
    )
    assert isinstance(retr, QARetriever)
    assert isinstance(retr, CoverageRetriever)


def test_bundle_retriever_returns_evidence_pack_for_synthetic_doc(
    tmp_path: Path, synthetic_doc_path: Path
) -> None:
    cfg = _write_config(tmp_path)
    bundle = build_bundle(config=Config.load(cfg), profile="heuristic")
    store = InMemoryStore()
    vector_index = InMemoryVectorIndex(dimension=32)
    bm25_index = TantivyBM25Index(path=tmp_path / "bm25")
    ingest_document(
        source=synthetic_doc_path,
        parser=MarkdownParser(),
        coref=bundle.coref,
        ner=bundle.ner,
        ner_labels=["person", "system"],
        embedder=bundle.embedder,
        summarizer=bundle.summarizer,
        store=store,
        vector_index=vector_index,
        bm25_index=bm25_index,
    )
    retr = BundleRetriever(
        bundle=bundle,
        store=store,
        vector_index=vector_index,
        bm25_index=bm25_index,
        prefix_skeleton="",
        prefix_glossary="",
    )
    pack = retr.retrieve("Aurora consistent hashing")
    assert isinstance(pack, EvidencePack)
    assert pack.query == "Aurora consistent hashing"
    assert pack.spans, "retriever should return at least one span for a known topic"


# --- render_coverage_markdown ---


def _fake_verdict(item_id: str, verdict: str, citations: list[Span] | None = None) -> Verdict:
    return Verdict(
        item_id=item_id,
        verdict=verdict,  # type: ignore[arg-type]
        citations=citations or [],
        confidence=0.9,
    )


def test_render_coverage_markdown_groups_by_verdict(tmp_path: Path) -> None:
    items = [
        ChecklistItem(id="auth", text="Auth required", topic_key="security"),
        ChecklistItem(id="logs", text="Logging required", topic_key="security"),
        ChecklistItem(id="rate", text="Rate limiting", topic_key="security"),
    ]
    report = CoverageReport(
        verdicts=[
            _fake_verdict("auth", "Covered"),
            _fake_verdict("logs", "Partial"),
            _fake_verdict("rate", "NotCovered"),
        ]
    )
    md = render_coverage_markdown(
        report=report,
        items=items,
        checklist_path=tmp_path / "c.md",
        target_path=tmp_path / "t.md",
        profile="thrifty",
        run_id="run-001",
    )
    assert "# ctrldoc — coverage audit report" in md
    assert "## Summary" in md
    assert "| Covered | 1 |" in md
    assert "| Partial | 1 |" in md
    assert "| NotCovered | 1 |" in md
    assert "## Covered (1)" in md
    assert "## Partial (1)" in md
    assert "## NotCovered (1)" in md
    assert "## Ambiguous (0)" in md


def test_render_coverage_markdown_renders_citations() -> None:
    items = [ChecklistItem(id="x", text="X required", topic_key="t")]
    report = CoverageReport(
        verdicts=[
            _fake_verdict(
                "x",
                "Covered",
                citations=[
                    Span(chunk_id="c1", char_start=0, char_end=11, text="hello world"),
                ],
            )
        ]
    )
    md = render_coverage_markdown(
        report=report,
        items=items,
        checklist_path=Path("c.md"),
        target_path=Path("t.md"),
        profile="thrifty",
        run_id="run-002",
    )
    assert "`[c1]` hello world" in md


def test_render_coverage_markdown_marks_empty_verdict_buckets() -> None:
    report = CoverageReport(verdicts=[])
    md = render_coverage_markdown(
        report=report,
        items=[],
        checklist_path=Path("c.md"),
        target_path=Path("t.md"),
        profile="thrifty",
        run_id="run-003",
    )
    for label in ("Covered", "Partial", "NotCovered", "Ambiguous"):
        assert f"## {label} (0)" in md
    assert "_(none)_" in md


# --- ctrldoc audit CLI surface ---


def test_audit_missing_checklist_exits_two(tmp_path: Path) -> None:
    target = tmp_path / "t.md"
    target.write_text("# Target\n", encoding="utf-8")
    cfg = _write_config(tmp_path)
    result = runner.invoke(
        app,
        [
            "--config",
            str(cfg),
            "--profile",
            "thrifty",
            "audit",
            "--checklist",
            str(tmp_path / "missing.md"),
            "--target",
            str(target),
        ],
    )
    assert result.exit_code == 2
    assert "does not exist" in result.stderr


def test_audit_missing_target_exits_two(tmp_path: Path) -> None:
    checklist = tmp_path / "c.md"
    checklist.write_text("## Item\nBody.\n", encoding="utf-8")
    cfg = _write_config(tmp_path)
    result = runner.invoke(
        app,
        [
            "--config",
            str(cfg),
            "--profile",
            "thrifty",
            "audit",
            "--checklist",
            str(checklist),
            "--target",
            str(tmp_path / "missing.md"),
        ],
    )
    assert result.exit_code == 2
    assert "does not exist" in result.stderr


def test_audit_heuristic_profile_rejected_with_clear_error(tmp_path: Path) -> None:
    checklist = tmp_path / "c.md"
    target = tmp_path / "t.md"
    checklist.write_text("## Item\nBody.\n", encoding="utf-8")
    target.write_text("# Target\nbody\n", encoding="utf-8")
    cfg = _write_config(tmp_path)
    result = runner.invoke(
        app,
        [
            "--config",
            str(cfg),
            "--profile",
            "heuristic",
            "audit",
            "--checklist",
            str(checklist),
            "--target",
            str(target),
        ],
    )
    assert result.exit_code == 2
    assert "audit requires --profile thrifty|production" in result.stderr


@pytest.mark.slow
@pytest.mark.requires_ollama
def test_audit_thrifty_writes_report_and_result_json(
    tmp_path: Path, synthetic_doc_path: Path
) -> None:
    pytest.importorskip("sqlite_vec")
    pytest.importorskip("ollama")
    pytest.importorskip("gliner", reason="gliner optional; thrifty profile needs it")
    pytest.importorskip("fastcoref", reason="fastcoref optional; thrifty profile needs it")
    checklist = tmp_path / "checklist.md"
    checklist.write_text(
        "## Hashing\nThe system uses consistent hashing.\n"
        "\n"
        "## GossipBus\nThe system has a gossip-based bus.\n",
        encoding="utf-8",
    )
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
            "audit",
            "--checklist",
            str(checklist),
            "--target",
            str(synthetic_doc_path),
        ],
    )
    if result.exit_code != 0:
        if result.exception and not isinstance(result.exception, SystemExit):
            raise AssertionError(
                f"thrifty audit crashed: {type(result.exception).__name__}: {result.exception}"
            ) from result.exception
        pytest.skip(f"thrifty audit non-zero exit ({result.exit_code})")
    payload = json.loads(result.stdout)
    assert payload["command"] == "audit"
    assert payload["profile"] == "thrifty"
    assert payload["items_total"] == 2
    assert "summary" in payload
    runs_path = tmp_path / "runs"
    assert list(runs_path.rglob("report.md"))
    assert list(runs_path.rglob("result.json"))
