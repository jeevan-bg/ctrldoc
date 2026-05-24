"""Typed bundle of every backend a playbook depends on.

`build_bundle(config, profile)` returns a frozen `BackendBundle`
wired for one of three runtime profiles:

  - **heuristic** — every reference impl. No LLM, no model loading,
    no `task_client_router`. Right for unit tests and CI.
  - **thrifty** — production retrieval / verifier infra (Ollama
    embedder, sqlite-vec, BGE reranker, fastcoref, GLiNER, DeBERTa
    NLI) but every per-item / per-claim LLM seam routes to the
    local Qwen2.5-7B via Ollama. The `task_client_router` still
    holds an Anthropic client on the `opus` tier so a playbook
    can spend exactly one synthesis call per run. The planner,
    claim decomposer, and section summarizer stay heuristic so a
    realistic end-to-end audit fits in a few dollars.
  - **production** — Anthropic everywhere a frontier model
    actually moves the needle: planner, claim decomposer,
    summarizer, escalating judge (Ollama → Anthropic on
    disagreement). Retrieval / verifier infra is identical to
    thrifty.

Heavy backends are lazy: the bundle calls their constructors only
inside the profile branch that needs them, so heuristic mode never
imports `ollama`, `sqlite-vec`, `transformers`, `fastcoref`, or
`gliner`. The classes themselves defer model loading until first
use (matching the lazy-`_ensure_*` pattern established in
S-036b / S-043b / S-051b / S-052b / S-022b / S-034b / S-061 /
S-110), so constructing a bundle is cheap even on the heavy paths.

SPEC-REF: §4.5 (orchestrator — tiered routing), §4.7 (configuration)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

from ctrldoc.config import Config
from ctrldoc.ingest.coref import CorefResolver, IdentityCorefResolver
from ctrldoc.ingest.embedder import Embedder, HashEmbedder
from ctrldoc.ingest.ner import NERTagger, StubNERTagger
from ctrldoc.ingest.summarizer import HeuristicSummarizer, Summarizer
from ctrldoc.orch.routing import TaskClientRouter
from ctrldoc.retrieval.planner import HeuristicPlanner, Planner
from ctrldoc.retrieval.reranker import IdentityReranker, Reranker
from ctrldoc.store import Store
from ctrldoc.store.bm25 import BM25Index, TantivyBM25Index
from ctrldoc.store.memory import InMemoryStore
from ctrldoc.store.vectors import InMemoryVectorIndex, VectorIndex
from ctrldoc.verify.claim_decomposer import ClaimDecomposer, HeuristicClaimDecomposer
from ctrldoc.verify.judge import HeuristicLLMJudge, LLMJudge
from ctrldoc.verify.nli import HeuristicNLIChecker, NLIChecker

Profile = Literal["heuristic", "thrifty", "production"]
PROFILES: tuple[Profile, ...] = get_args(Profile)

_HEURISTIC_EMBED_DIM = 32
_BGE_M3_EMBED_DIM = 1024


@dataclass(frozen=True)
class BackendBundle:
    """Frozen container — every backend a playbook needs to run.

    `task_client_router` is `None` in the heuristic profile: that
    mode never calls an LLM. Other fields are always populated so
    callers don't pepper their code with `is None` checks.
    """

    profile: Profile
    embedder: Embedder
    vector_index: VectorIndex
    bm25_index: BM25Index
    store: Store
    coref: CorefResolver
    ner: NERTagger
    reranker: Reranker
    planner: Planner
    task_client_router: TaskClientRouter | None
    claim_decomposer: ClaimDecomposer
    nli_checker: NLIChecker
    llm_judge: LLMJudge
    summarizer: Summarizer


def build_bundle(*, config: Config, profile: Profile) -> BackendBundle:
    """Build a `BackendBundle` for the given profile.

    `config` is consumed for filesystem paths (sqlite-vec, SQLite
    store, Tantivy BM25 index) — model identifiers come from the
    backend defaults, since each backend's constructor pins the
    `id` SPEC-REF row already verified in the per-backend slice.
    """
    if profile == "heuristic":
        return _build_heuristic(config)
    if profile == "thrifty":
        return _build_thrifty(config)
    if profile == "production":
        return _build_production(config)
    raise ValueError(f"unknown profile: {profile!r}; expected one of {PROFILES}")


def build_bundle_from_toml(path: str | Path, *, profile: Profile) -> BackendBundle:
    """Convenience: load `ctrldoc.toml` from disk then call `build_bundle`."""
    config = Config.load(path)
    return build_bundle(config=config, profile=profile)


# --- profile builders ---


def _ensure_index_dir(config: Config) -> Path:
    index_dir = Path(config.paths.index_path)
    index_dir.mkdir(parents=True, exist_ok=True)
    return index_dir


def _build_heuristic(config: Config) -> BackendBundle:
    # Heuristic mode never opens sqlite-vec / SQLite for persistence
    # — keep it pure-memory so unit tests don't leave files behind.
    index_dir = _ensure_index_dir(config)
    return BackendBundle(
        profile="heuristic",
        embedder=HashEmbedder(dimension=_HEURISTIC_EMBED_DIM),
        vector_index=InMemoryVectorIndex(dimension=_HEURISTIC_EMBED_DIM),
        bm25_index=TantivyBM25Index(path=index_dir / "bm25"),
        store=InMemoryStore(),
        coref=IdentityCorefResolver(),
        ner=StubNERTagger({}),
        reranker=IdentityReranker(),
        planner=HeuristicPlanner(),
        task_client_router=None,
        claim_decomposer=HeuristicClaimDecomposer(),
        nli_checker=HeuristicNLIChecker(),
        llm_judge=HeuristicLLMJudge(),
        summarizer=HeuristicSummarizer(),
    )


def _resolve_optional_ner() -> NERTagger:
    """Return GLiNERTagger when gliner is importable; else StubNERTagger.

    Lets thrifty / production bundles construct without gliner installed —
    entity-based retrieval just degrades to zero entities (the bundle's
    glossary will be empty and entity-view retrieval steps return nothing).
    Coverage / QA / review / scan / map still work; map gets fewer
    concepts to walk and the audit/QA prefix carries an empty glossary.
    """
    try:
        # Touch `gliner` here so we fall through to the stub when the
        # package itself is missing (GLiNERTagger lazy-imports it).
        import gliner  # type: ignore[import-untyped,import-not-found,unused-ignore] # noqa: F401

        from ctrldoc.ingest.ner_gliner import GLiNERTagger

        return GLiNERTagger()
    except ImportError:
        return StubNERTagger({})


def _build_thrifty(config: Config) -> BackendBundle:
    from ctrldoc.ingest.coref_fastcoref import FastCorefResolver
    from ctrldoc.ingest.embedder_ollama import OllamaEmbedder
    from ctrldoc.orch.task_anthropic import AnthropicTaskClient
    from ctrldoc.orch.task_ollama import OllamaTaskClient
    from ctrldoc.retrieval.reranker_bge import BGEReranker
    from ctrldoc.store.sqlite import SQLiteStore
    from ctrldoc.store.vectors_sqlite_vec import SqliteVecVectorIndex
    from ctrldoc.verify.judge_ollama import OllamaLLMJudge
    from ctrldoc.verify.nli_deberta import DeBERTaNLIChecker

    index_dir = _ensure_index_dir(config)
    return BackendBundle(
        profile="thrifty",
        embedder=OllamaEmbedder(),
        vector_index=SqliteVecVectorIndex(
            dimension=_BGE_M3_EMBED_DIM,
            path=str(index_dir / "vec.db"),
        ),
        bm25_index=TantivyBM25Index(path=index_dir / "bm25"),
        store=SQLiteStore(index_dir / "store.db"),
        coref=FastCorefResolver(),
        ner=_resolve_optional_ner(),
        reranker=BGEReranker(),
        planner=HeuristicPlanner(),
        task_client_router=TaskClientRouter(
            local=OllamaTaskClient(),
            opus=AnthropicTaskClient(),
        ),
        claim_decomposer=HeuristicClaimDecomposer(),
        nli_checker=DeBERTaNLIChecker(),
        llm_judge=OllamaLLMJudge(),
        summarizer=HeuristicSummarizer(),
    )


def _build_production(config: Config) -> BackendBundle:
    from ctrldoc.ingest.coref_fastcoref import FastCorefResolver
    from ctrldoc.ingest.embedder_ollama import OllamaEmbedder
    from ctrldoc.ingest.summarizer_anthropic import AnthropicSummarizer
    from ctrldoc.orch.task_anthropic import AnthropicTaskClient
    from ctrldoc.orch.task_ollama import OllamaTaskClient
    from ctrldoc.retrieval.planner_anthropic import AnthropicPlanner
    from ctrldoc.retrieval.reranker_bge import BGEReranker
    from ctrldoc.store.sqlite import SQLiteStore
    from ctrldoc.store.vectors_sqlite_vec import SqliteVecVectorIndex
    from ctrldoc.verify.claim_decomposer_anthropic import AnthropicClaimDecomposer
    from ctrldoc.verify.judge_anthropic import AnthropicLLMJudge
    from ctrldoc.verify.judge_escalating import EscalatingLLMJudge
    from ctrldoc.verify.judge_ollama import OllamaLLMJudge
    from ctrldoc.verify.nli_deberta import DeBERTaNLIChecker

    index_dir = _ensure_index_dir(config)
    return BackendBundle(
        profile="production",
        embedder=OllamaEmbedder(),
        vector_index=SqliteVecVectorIndex(
            dimension=_BGE_M3_EMBED_DIM,
            path=str(index_dir / "vec.db"),
        ),
        bm25_index=TantivyBM25Index(path=index_dir / "bm25"),
        store=SQLiteStore(index_dir / "store.db"),
        coref=FastCorefResolver(),
        ner=_resolve_optional_ner(),
        reranker=BGEReranker(),
        planner=AnthropicPlanner(),
        task_client_router=TaskClientRouter(
            local=OllamaTaskClient(),
            opus=AnthropicTaskClient(),
        ),
        claim_decomposer=AnthropicClaimDecomposer(),
        nli_checker=DeBERTaNLIChecker(),
        llm_judge=EscalatingLLMJudge(
            tier1=OllamaLLMJudge(),
            tier2=AnthropicLLMJudge(),
        ),
        summarizer=AnthropicSummarizer(),
    )


__all__ = [
    "PROFILES",
    "BackendBundle",
    "Profile",
    "build_bundle",
    "build_bundle_from_toml",
]
