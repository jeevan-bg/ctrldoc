"""`BackendBundle` factory — typed wiring for the three runtime profiles.

The factory reads a parsed `Config` and returns a frozen bundle of
every protocol-conformant backend the playbooks need (embedder,
vector_index, bm25_index, store, coref, ner, reranker, planner,
task_client_router, claim_decomposer, nli_checker, llm_judge,
summarizer).

Three profiles per the build plan:

  - **heuristic** — every reference impl, no LLM, no model loading.
    Right for unit tests and the CI lane that must stay free.
  - **thrifty** — production retrieval + verifier backends, but the
    LLM seams route per-item / per-claim calls to the local Qwen2.5-7B
    via Ollama. `task_client_router` still wires the `opus` tier to
    Anthropic for the single synthesis call per playbook run; the
    planner stays heuristic. Right for cheap end-to-end audits.
  - **production** — Anthropic everywhere a frontier model is called
    for: planner, claim decomposer, summarizer, escalating judge.

SPEC-REF: §4.5 (orchestrator — tiered routing), §4.7 (configuration)
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from ctrldoc.backends import (
    PROFILES,
    BackendBundle,
    build_bundle,
    build_bundle_from_toml,
)
from ctrldoc.config import Config
from ctrldoc.ingest.coref import CorefResolver, IdentityCorefResolver
from ctrldoc.ingest.embedder import Embedder, HashEmbedder
from ctrldoc.ingest.ner import NERTagger, StubNERTagger
from ctrldoc.ingest.summarizer import HeuristicSummarizer, Summarizer
from ctrldoc.retrieval.planner import HeuristicPlanner, Planner
from ctrldoc.retrieval.reranker import IdentityReranker, Reranker
from ctrldoc.store import Store
from ctrldoc.store.bm25 import BM25Index, TantivyBM25Index
from ctrldoc.store.memory import InMemoryStore
from ctrldoc.store.vectors import InMemoryVectorIndex, VectorIndex
from ctrldoc.verify.claim_decomposer import ClaimDecomposer, HeuristicClaimDecomposer
from ctrldoc.verify.judge import HeuristicLLMJudge, LLMJudge
from ctrldoc.verify.nli import HeuristicNLIChecker, NLIChecker

_TOML_TEMPLATE = """\
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
        p.mkdir()
    cfg_path = tmp_path / "ctrldoc.toml"
    cfg_path.write_text(
        _TOML_TEMPLATE.format(
            index_path=index_path.as_posix(),
            runs_path=runs_path.as_posix(),
            traces_path=traces_path.as_posix(),
        ),
        encoding="utf-8",
    )
    return cfg_path


# --- profile enumeration ---


def test_PROFILES_constant_matches_supported_set() -> None:
    assert set(PROFILES) == {"heuristic", "thrifty", "production"}


def test_unknown_profile_raises(tmp_path: Path) -> None:
    config = Config.load(_write_config(tmp_path))
    with pytest.raises(ValueError, match="unknown profile"):
        build_bundle(config=config, profile="bogus")  # type: ignore[arg-type]


# --- heuristic profile ---


def test_heuristic_bundle_carries_reference_impls(tmp_path: Path) -> None:
    config = Config.load(_write_config(tmp_path))
    bundle = build_bundle(config=config, profile="heuristic")
    assert bundle.profile == "heuristic"
    assert isinstance(bundle.embedder, HashEmbedder)
    assert isinstance(bundle.vector_index, InMemoryVectorIndex)
    assert isinstance(bundle.bm25_index, TantivyBM25Index)
    assert isinstance(bundle.store, InMemoryStore)
    assert isinstance(bundle.coref, IdentityCorefResolver)
    assert isinstance(bundle.ner, StubNERTagger)
    assert isinstance(bundle.reranker, IdentityReranker)
    assert isinstance(bundle.planner, HeuristicPlanner)
    assert isinstance(bundle.claim_decomposer, HeuristicClaimDecomposer)
    assert isinstance(bundle.nli_checker, HeuristicNLIChecker)
    assert isinstance(bundle.llm_judge, HeuristicLLMJudge)
    assert isinstance(bundle.summarizer, HeuristicSummarizer)


def test_heuristic_bundle_has_no_task_client_router(tmp_path: Path) -> None:
    """Heuristic mode never calls an LLM — the router stays unset to
    make accidental LLM use loud rather than silent."""
    config = Config.load(_write_config(tmp_path))
    bundle = build_bundle(config=config, profile="heuristic")
    assert bundle.task_client_router is None


def test_heuristic_bundle_satisfies_protocols(tmp_path: Path) -> None:
    config = Config.load(_write_config(tmp_path))
    bundle = build_bundle(config=config, profile="heuristic")
    assert isinstance(bundle.embedder, Embedder)
    assert isinstance(bundle.vector_index, VectorIndex)
    assert isinstance(bundle.bm25_index, BM25Index)
    assert isinstance(bundle.store, Store)
    assert isinstance(bundle.coref, CorefResolver)
    assert isinstance(bundle.ner, NERTagger)
    assert isinstance(bundle.reranker, Reranker)
    assert isinstance(bundle.planner, Planner)
    assert isinstance(bundle.claim_decomposer, ClaimDecomposer)
    assert isinstance(bundle.nli_checker, NLIChecker)
    assert isinstance(bundle.llm_judge, LLMJudge)
    assert isinstance(bundle.summarizer, Summarizer)


def test_heuristic_bundle_is_frozen(tmp_path: Path) -> None:
    config = Config.load(_write_config(tmp_path))
    bundle = build_bundle(config=config, profile="heuristic")
    with pytest.raises(FrozenInstanceError):
        bundle.profile = "thrifty"  # type: ignore[misc]


def test_heuristic_embedder_dimension_matches_vector_index(tmp_path: Path) -> None:
    config = Config.load(_write_config(tmp_path))
    bundle = build_bundle(config=config, profile="heuristic")
    # round-trip embed → index to verify dimensions line up
    vec = bundle.embedder.embed("hello world")
    bundle.vector_index.add("c1", vec)
    hits = bundle.vector_index.search(vec, k=1)
    assert hits and hits[0][0] == "c1"


# --- from-toml convenience ---


def test_build_bundle_from_toml_round_trips(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)
    bundle = build_bundle_from_toml(cfg_path, profile="heuristic")
    assert isinstance(bundle, BackendBundle)
    assert bundle.profile == "heuristic"


def test_build_bundle_from_toml_propagates_loader_errors(tmp_path: Path) -> None:
    missing = tmp_path / "absent.toml"
    with pytest.raises(FileNotFoundError):
        build_bundle_from_toml(missing, profile="heuristic")


# --- thrifty profile ---


def test_thrifty_bundle_uses_local_per_item_backends(tmp_path: Path) -> None:
    """Thrifty path: production retrieval/verifier infra but keep
    every per-item / per-claim LLM seam routed to Qwen via Ollama
    and the planner heuristic; only the synthesis call burns Opus.
    """
    pytest.importorskip("sqlite_vec")
    pytest.importorskip("ollama")
    from ctrldoc.ingest.coref_fastcoref import FastCorefResolver
    from ctrldoc.ingest.embedder_ollama import OllamaEmbedder
    from ctrldoc.ingest.ner_gliner import GLiNERTagger
    from ctrldoc.orch.routing import TaskClientRouter
    from ctrldoc.orch.task_anthropic import AnthropicTaskClient
    from ctrldoc.orch.task_ollama import OllamaTaskClient
    from ctrldoc.retrieval.reranker_bge import BGEReranker
    from ctrldoc.store.sqlite import SQLiteStore
    from ctrldoc.store.vectors_sqlite_vec import SqliteVecVectorIndex
    from ctrldoc.verify.judge_ollama import OllamaLLMJudge
    from ctrldoc.verify.nli_deberta import DeBERTaNLIChecker

    config = Config.load(_write_config(tmp_path))
    bundle = build_bundle(config=config, profile="thrifty")
    assert bundle.profile == "thrifty"
    # production retrieval / verifier infra
    assert isinstance(bundle.embedder, OllamaEmbedder)
    assert isinstance(bundle.vector_index, SqliteVecVectorIndex)
    assert isinstance(bundle.bm25_index, TantivyBM25Index)
    assert isinstance(bundle.store, SQLiteStore)
    assert isinstance(bundle.coref, FastCorefResolver)
    assert isinstance(bundle.ner, GLiNERTagger)
    assert isinstance(bundle.reranker, BGEReranker)
    assert isinstance(bundle.nli_checker, DeBERTaNLIChecker)
    # per-item / per-claim LLM seams: stay cheap
    assert isinstance(bundle.planner, HeuristicPlanner)
    assert isinstance(bundle.claim_decomposer, HeuristicClaimDecomposer)
    assert isinstance(bundle.summarizer, HeuristicSummarizer)
    # tier-1 only — escalation breaks budget for per-claim verification
    assert isinstance(bundle.llm_judge, OllamaLLMJudge)
    # task_client_router has BOTH tiers wired: opus reserved for synthesis
    assert isinstance(bundle.task_client_router, TaskClientRouter)
    assert isinstance(bundle.task_client_router.for_tier("local"), OllamaTaskClient)
    assert isinstance(bundle.task_client_router.for_tier("opus"), AnthropicTaskClient)


# --- production profile ---


def test_production_bundle_uses_frontier_backends_for_reasoning(tmp_path: Path) -> None:
    pytest.importorskip("sqlite_vec")
    pytest.importorskip("ollama")
    from ctrldoc.ingest.summarizer_anthropic import AnthropicSummarizer
    from ctrldoc.orch.routing import TaskClientRouter
    from ctrldoc.retrieval.planner_anthropic import AnthropicPlanner
    from ctrldoc.verify.claim_decomposer_anthropic import AnthropicClaimDecomposer
    from ctrldoc.verify.judge_escalating import EscalatingLLMJudge

    config = Config.load(_write_config(tmp_path))
    bundle = build_bundle(config=config, profile="production")
    assert bundle.profile == "production"
    assert isinstance(bundle.planner, AnthropicPlanner)
    assert isinstance(bundle.claim_decomposer, AnthropicClaimDecomposer)
    assert isinstance(bundle.summarizer, AnthropicSummarizer)
    assert isinstance(bundle.llm_judge, EscalatingLLMJudge)
    assert isinstance(bundle.task_client_router, TaskClientRouter)


def test_production_bundle_shares_retrieval_backends_with_thrifty(tmp_path: Path) -> None:
    """The cost split between thrifty and production is in the LLM
    seams, not the retrieval/verifier infra — confirm those line up."""
    pytest.importorskip("sqlite_vec")
    pytest.importorskip("ollama")
    from ctrldoc.ingest.coref_fastcoref import FastCorefResolver
    from ctrldoc.ingest.embedder_ollama import OllamaEmbedder
    from ctrldoc.ingest.ner_gliner import GLiNERTagger
    from ctrldoc.retrieval.reranker_bge import BGEReranker
    from ctrldoc.store.sqlite import SQLiteStore
    from ctrldoc.store.vectors_sqlite_vec import SqliteVecVectorIndex
    from ctrldoc.verify.nli_deberta import DeBERTaNLIChecker

    config = Config.load(_write_config(tmp_path))
    prod = build_bundle(config=config, profile="production")
    assert isinstance(prod.embedder, OllamaEmbedder)
    assert isinstance(prod.vector_index, SqliteVecVectorIndex)
    assert isinstance(prod.bm25_index, TantivyBM25Index)
    assert isinstance(prod.store, SQLiteStore)
    assert isinstance(prod.coref, FastCorefResolver)
    assert isinstance(prod.ner, GLiNERTagger)
    assert isinstance(prod.reranker, BGEReranker)
    assert isinstance(prod.nli_checker, DeBERTaNLIChecker)
