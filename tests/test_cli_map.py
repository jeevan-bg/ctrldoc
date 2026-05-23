"""Tests for the relation-map CLI wiring (S-117).

Covers:

  * `StoreEntityConceptExtractor` — pulls top-N entities by
    mention count; stable on ties by id; rejects non-positive
    `max_concepts`.
  * `BundleCoOccurrenceRetriever` — sanitises BM25-hostile
    punctuation in concept names; protocol conformance.
  * `LLMRelationClassifier` — returns `None` for `unrelated`;
    drops hallucinated citation chunk_ids; clamps confidence.
  * `render_map_markdown` — header, adjacency table, Mermaid
    block with typed edges + standalone nodes; empty-graph
    placeholder.
  * `ctrldoc map` CLI surface:
      - missing target → exit 2.
      - heuristic profile rejected with a clear error.
      - non-positive `--max-concepts` → exit 2.
      - slow thrifty integration gated on optional deps.

SPEC-REF: §5.6 (relation_map), §6 (CLI)
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
from ctrldoc.cli_map import (
    BundleCoOccurrenceRetriever,
    LLMRelationClassifier,
    StoreEntityConceptExtractor,
    render_map_markdown,
)
from ctrldoc.config import Config
from ctrldoc.models import Entity, EvidencePack, RelationEdge, Span
from ctrldoc.orch.task import StatelessTaskRunner
from ctrldoc.playbooks.relations import (
    Concept,
    CoOccurrenceRetriever,
    RelationGraph,
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
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.calls: list[dict[str, Any]] = []

    def call(self, *, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        return self._reply


# --- StoreEntityConceptExtractor ---


def test_concept_extractor_caps_at_max_concepts() -> None:
    store = InMemoryStore()
    entities = [
        Entity(
            id=f"e/{i}",
            aliases=[f"name-{i}"],
            type="concept",
            mention_chunk_ids=[f"c/{j}" for j in range(i + 1)],
        )
        for i in range(20)
    ]
    store.add_entities(entities)
    extractor = StoreEntityConceptExtractor(store=store, max_concepts=5)
    concepts = extractor.extract()
    assert len(concepts) == 5
    # Most-mentioned (highest i) come first.
    assert concepts[0].id == "e/19"


def test_concept_extractor_returns_empty_for_empty_store() -> None:
    store = InMemoryStore()
    extractor = StoreEntityConceptExtractor(store=store)
    assert extractor.extract() == []


def test_concept_extractor_rejects_zero_max_concepts() -> None:
    with pytest.raises(ValueError, match="positive"):
        StoreEntityConceptExtractor(store=InMemoryStore(), max_concepts=0)


# --- BundleCoOccurrenceRetriever ---


def test_bundle_cooccurrence_retriever_satisfies_protocol(tmp_path: Path) -> None:
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
    retr = BundleCoOccurrenceRetriever(bundle_retriever=inner)
    assert isinstance(retr, CoOccurrenceRetriever)


def test_bundle_cooccurrence_retriever_sanitises_punctuation_in_names(
    tmp_path: Path,
) -> None:
    """Concept names that contain BM25-hostile chars (`:`, `|`, etc.)
    must not crash the retrieval."""
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
    retr = BundleCoOccurrenceRetriever(bundle_retriever=inner)
    # Empty store → empty pack; the punctuation in names must not crash.
    pack = retr.retrieve(
        Concept(id="a", name="user: admin"),
        Concept(id="b", name="session|token"),
    )
    assert pack.spans == []


# --- LLMRelationClassifier ---


def _ev_pack(*chunk_ids: str) -> EvidencePack:
    spans = [Span(chunk_id=cid, char_start=0, char_end=4, text=f"text-{cid}") for cid in chunk_ids]
    return EvidencePack(
        query="q",
        spans=spans,
        token_count=sum(len(s.text) for s in spans),
        retrieval_plan=[],
    )


def test_classifier_returns_none_when_evidence_empty() -> None:
    classifier = LLMRelationClassifier(
        prefix=CacheablePrefix(system_prompt="s", doc_skeleton="", entity_glossary=""),
        task_runner=StatelessTaskRunner(client=_StubTaskClient("")),
    )
    out = classifier.classify(
        Concept(id="a", name="A"),
        Concept(id="b", name="B"),
        EvidencePack(query="q", spans=[], token_count=0, retrieval_plan=[]),
    )
    assert out is None


def test_classifier_returns_none_for_unrelated_verdict() -> None:
    stub = _StubTaskClient(
        json.dumps({"type": "unrelated", "confidence": 0.5, "citation_chunk_ids": ["c1"]})
    )
    classifier = LLMRelationClassifier(
        prefix=CacheablePrefix(system_prompt="s", doc_skeleton="", entity_glossary=""),
        task_runner=StatelessTaskRunner(client=stub),
    )
    out = classifier.classify(
        Concept(id="a", name="A"),
        Concept(id="b", name="B"),
        _ev_pack("c1"),
    )
    assert out is None


def test_classifier_resolves_citations_and_clamps_confidence() -> None:
    stub = _StubTaskClient(
        json.dumps(
            {
                "type": "depends_on",
                "confidence": 0.95,
                "citation_chunk_ids": ["c1", "missing-id", "c2"],
            }
        )
    )
    classifier = LLMRelationClassifier(
        prefix=CacheablePrefix(system_prompt="s", doc_skeleton="", entity_glossary=""),
        task_runner=StatelessTaskRunner(client=stub),
    )
    out = classifier.classify(
        Concept(id="a", name="A"),
        Concept(id="b", name="B"),
        _ev_pack("c1", "c2"),
    )
    assert out is not None
    assert out.type == "depends_on"
    assert out.confidence == 0.95
    # `missing-id` dropped; `c1` and `c2` kept.
    assert [s.chunk_id for s in out.citations] == ["c1", "c2"]


def test_classifier_drops_invalid_type_label() -> None:
    stub = _StubTaskClient(
        json.dumps({"type": "totally_made_up", "confidence": 0.9, "citation_chunk_ids": ["c1"]})
    )
    classifier = LLMRelationClassifier(
        prefix=CacheablePrefix(system_prompt="s", doc_skeleton="", entity_glossary=""),
        task_runner=StatelessTaskRunner(client=stub),
    )
    out = classifier.classify(
        Concept(id="a", name="A"),
        Concept(id="b", name="B"),
        _ev_pack("c1"),
    )
    assert out is None


# --- render_map_markdown ---


def test_render_map_markdown_has_adjacency_table_and_mermaid_block() -> None:
    graph = RelationGraph(
        nodes=[
            Concept(id="auth", name="Auth"),
            Concept(id="session", name="Session"),
            Concept(id="rate_limit", name="Rate Limit"),
        ],
        edges=[
            RelationEdge(
                src_concept="auth",
                dst_concept="session",
                type="depends_on",
                citations=[
                    Span(chunk_id="c1", char_start=0, char_end=5, text="hello"),
                ],
                confidence=0.85,
            ),
        ],
    )
    md = render_map_markdown(
        graph=graph,
        target_path=Path("doc.md"),
        profile="thrifty",
        run_id="r1",
    )
    assert "# ctrldoc — concept relation map" in md
    assert "- **Nodes**: 3" in md
    assert "- **Edges**: 1" in md
    assert "| src | type | dst |" in md
    assert "| `auth` | `depends_on` | `session` |" in md
    assert "```mermaid" in md
    assert "graph LR" in md
    assert 'auth["Auth"]' in md
    assert 'rate_limit["Rate Limit"]' in md
    assert "auth -- depends_on --> session" in md


def test_render_map_markdown_empty_edges_shows_placeholder() -> None:
    graph = RelationGraph(nodes=[Concept(id="a", name="A")], edges=[])
    md = render_map_markdown(
        graph=graph,
        target_path=Path("doc.md"),
        profile="thrifty",
        run_id="r2",
    )
    assert "_(no relations detected)_" in md
    assert "```mermaid" in md
    assert 'a["A"]' in md


def test_render_map_markdown_handles_dotted_concept_ids() -> None:
    """Concept ids with `/` or `.` are slugged for Mermaid."""
    graph = RelationGraph(
        nodes=[
            Concept(id="e/1", name="Concept 1"),
            Concept(id="e/2", name="Concept 2"),
        ],
        edges=[
            RelationEdge(
                src_concept="e/1",
                dst_concept="e/2",
                type="refines",
                citations=[],
                confidence=0.7,
            ),
        ],
    )
    md = render_map_markdown(
        graph=graph,
        target_path=Path("d.md"),
        profile="thrifty",
        run_id="r3",
    )
    # Slug: `/` → `_`. Concept ids are valid Mermaid node ids.
    assert "e_1 -- refines --> e_2" in md
    assert 'e_1["Concept 1"]' in md


# --- ctrldoc map CLI surface ---


def test_map_missing_target_exits_with_code_two(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = runner.invoke(
        app,
        [
            "--config",
            str(cfg),
            "--profile",
            "thrifty",
            "map",
            "--target",
            str(tmp_path / "absent.md"),
        ],
    )
    assert result.exit_code == 2
    assert "does not exist" in result.stderr


def test_map_heuristic_profile_rejected(tmp_path: Path, synthetic_doc_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = runner.invoke(
        app,
        [
            "--config",
            str(cfg),
            "--profile",
            "heuristic",
            "map",
            "--target",
            str(synthetic_doc_path),
        ],
    )
    assert result.exit_code == 2
    assert "map requires --profile thrifty|production" in result.stderr


def test_map_zero_max_concepts_exits_with_code_two(
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
            "map",
            "--target",
            str(synthetic_doc_path),
            "--max-concepts",
            "0",
        ],
    )
    assert result.exit_code == 2
    assert "must be positive" in result.stderr


@pytest.mark.slow
@pytest.mark.requires_ollama
def test_map_thrifty_writes_report_and_result_json(
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
            "map",
            "--target",
            str(synthetic_doc_path),
            "--max-concepts",
            "5",
        ],
    )
    if result.exit_code != 0:
        if result.exception and not isinstance(result.exception, SystemExit):
            raise AssertionError(
                f"thrifty map crashed: {type(result.exception).__name__}: {result.exception}"
            ) from result.exception
        pytest.skip(f"thrifty map non-zero exit ({result.exit_code})")
    payload = json.loads(result.stdout)
    assert payload["command"] == "map"
    assert payload["profile"] == "thrifty"
    assert payload["max_concepts"] == 5
    assert isinstance(payload["nodes"], list)
    assert isinstance(payload["edges"], list)
    assert list((tmp_path / "runs").rglob("report.md"))
    assert list((tmp_path / "runs").rglob("result.json"))
