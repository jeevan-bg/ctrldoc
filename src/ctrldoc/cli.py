# ruff: noqa: B008 — typer's CLI declaration idiom requires calling
# `typer.Argument(...)` / `typer.Option(...)` in parameter defaults.
"""ctrldoc CLI — typer app wiring the six playbooks.

Subcommands (one per use case from §5):

  - ``ingest`` — run the L0 pipeline against a source doc and persist
    the per-doc index (SQLiteStore + sqlite-vec + Tantivy BM25).
  - ``qa`` — UC1 trustworthy QA (skeleton; wired in S-114).
  - ``audit`` — UC2 coverage audit / UC3 quality audit (skeleton).
  - ``review`` — UC4 analytical review (skeleton).
  - ``scan`` — UC5 anomaly scan (deterministic detectors today).
  - ``map`` — UC6 concept relation map (skeleton).

Every subcommand sees a `CliState` populated by the global
``@app.callback``: ``--config`` path, ``--profile``, ``--format``
(markdown / json / both), ``--max-cost-usd``. ``.env`` is loaded
once at callback entry; ``ANTHROPIC_API_KEY`` is never echoed.

SPEC-REF: §4.5, §4.7, §5, §6
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

from ctrldoc.backends import PROFILES, Profile, build_bundle
from ctrldoc.canary import CanaryBaseline, save_baseline
from ctrldoc.config import (
    BudgetsConfig,
    ConcurrencyConfig,
    Config,
    ModelsConfig,
    PathsConfig,
)
from ctrldoc.ingest.parser import MarkdownParser
from ctrldoc.ingest.pipeline import IngestStats, ingest_document
from ctrldoc.playbooks.anomaly import (
    AnomalyScanPlaybook,
    EmptySummaryDetector,
    HedgeWordDetector,
)
from ctrldoc.provenance import new_run_id
from ctrldoc.store import Store
from ctrldoc.store.bm25 import BM25Index, TantivyBM25Index
from ctrldoc.store.memory import InMemoryStore
from ctrldoc.store.vectors import InMemoryVectorIndex, VectorIndex

OutputFormat = Literal["markdown", "json", "both"]
_OUTPUT_FORMATS: tuple[OutputFormat, ...] = ("markdown", "json", "both")

_HEURISTIC_EMBED_DIM = 32
_BGE_M3_EMBED_DIM = 1024
_DEFAULT_NER_LABELS = ["person", "system", "concept"]

_DEFAULT_RUNS_PATH = Path("./runs")
_DEFAULT_INDEX_PATH = Path("./ctrldoc-index")
_DEFAULT_TRACES_PATH = Path("./traces")
_DEFAULT_MAX_COST_USD = 5.0


@dataclass(frozen=True)
class CliState:
    """Per-invocation state populated by the global ``@app.callback``."""

    config_path: Path
    profile: Profile
    output_format: OutputFormat
    max_cost_usd: float


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "ctrldoc — local-first document analysis substrate.\n\n"
        "Run `ctrldoc <command> --help` for per-command help."
    ),
)


# --- shared helpers ---


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Manual KEY=VALUE parser; sets entries into `os.environ` if not already set.

    Never echoes a value back to the user. Lines that do not parse as
    `KEY=VALUE` (comments, blanks, malformed) are silently skipped.
    """
    if not path.exists() or not path.is_file():
        return
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip()
        if value and value[0] in ("'", '"') and value[-1] == value[0]:
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def _default_config() -> Config:
    """A built-in default `Config` used when no `ctrldoc.toml` is present.

    Lets the CLI work out of the box from any working directory; the
    user can override any of these by writing a project-local
    ``ctrldoc.toml`` and passing ``--config``.
    """
    return Config(
        models=ModelsConfig(
            planner="claude-opus-4-7",
            judge_tier1="qwen2.5:7b-instruct-q4_K_M",
            judge_tier2="claude-opus-4-7",
            verifier_nli="deberta-v3-large-mnli",
            embedder="bge-m3",
        ),
        budgets=BudgetsConfig(
            max_cost_usd=_DEFAULT_MAX_COST_USD,
            max_tokens_per_call=16000,
            max_wall_clock_min=30,
        ),
        concurrency=ConcurrencyConfig(anthropic_concurrent=8, ollama_concurrent=2),
        paths=PathsConfig(
            index_path=_DEFAULT_INDEX_PATH,
            runs_path=_DEFAULT_RUNS_PATH,
            traces_path=_DEFAULT_TRACES_PATH,
        ),
    )


def _load_config(path: Path) -> Config:
    if path.exists():
        return Config.load(path)
    return _default_config()


def _require_input_path(path: Path) -> None:
    if not path.exists():
        typer.echo(f"error: input file {path} does not exist", err=True)
        raise typer.Exit(code=2)
    if not path.is_file():
        typer.echo(f"error: input {path} is not a regular file", err=True)
        raise typer.Exit(code=2)


def _doc_hash_for(text: str) -> str:
    """16-char hex prefix of sha256 over the document text.

    Short enough to embed in filenames, long enough that two distinct
    docs in a per-user run dir won't collide.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _per_doc_index_dir(config: Config, doc_hash: str) -> Path:
    base = Path(config.paths.runs_path) / "indexes"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _build_per_doc_backends(
    *,
    config: Config,
    profile: Profile,
    doc_hash: str,
) -> tuple[Store, VectorIndex, BM25Index]:
    """Per-doc store / vector_index / bm25_index sized to the profile.

    Heuristic profile stays in-memory for the store + vector index so
    unit tests don't depend on sqlite-vec; thrifty and production
    profiles persist the SQLite + sqlite-vec files at
    ``<runs_path>/indexes/<doc_hash>.db`` and ``.vec.db``. BM25 is
    always Tantivy (file-backed) — it's the same regardless of
    profile.
    """
    index_dir = _per_doc_index_dir(config, doc_hash)
    bm25_path = index_dir / f"{doc_hash}.bm25"
    if profile == "heuristic":
        return (
            InMemoryStore(),
            InMemoryVectorIndex(dimension=_HEURISTIC_EMBED_DIM),
            TantivyBM25Index(path=bm25_path),
        )
    # thrifty / production share persistence; the LLM seams above are
    # what split between them.
    from ctrldoc.store.sqlite import SQLiteStore
    from ctrldoc.store.vectors_sqlite_vec import SqliteVecVectorIndex

    return (
        SQLiteStore(index_dir / f"{doc_hash}.db"),
        SqliteVecVectorIndex(
            dimension=_BGE_M3_EMBED_DIM,
            path=str(index_dir / f"{doc_hash}.vec.db"),
        ),
        TantivyBM25Index(path=bm25_path),
    )


def _emit_output(state: CliState, *, markdown: str, payload: dict[str, object]) -> None:
    """Render the CLI output per the active ``--format`` mode.

    - ``markdown`` (default): print the Markdown report to stdout.
    - ``json``: print the JSON payload to stdout (no Markdown).
    - ``both``: print Markdown then a ``\\n--- JSON ---\\n`` separator
      followed by the JSON payload.
    """
    if state.output_format == "json":
        typer.echo(json.dumps(payload, indent=2))
        return
    if state.output_format == "both":
        typer.echo(markdown)
        typer.echo("\n--- JSON ---")
        typer.echo(json.dumps(payload, indent=2))
        return
    typer.echo(markdown)


# --- global callback ---


@app.callback()
def _global_callback(
    ctx: typer.Context,
    config: Path = typer.Option(
        Path("ctrldoc.toml"),
        "--config",
        help="Path to ctrldoc.toml; falls back to built-in defaults when absent.",
    ),
    profile: str = typer.Option(
        "thrifty",
        "--profile",
        "-p",
        help=f"Backend profile; one of {', '.join(PROFILES)}.",
    ),
    output_format: str = typer.Option(
        "markdown",
        "--format",
        "-f",
        help="Output format: markdown | json | both.",
    ),
    max_cost_usd: float = typer.Option(
        _DEFAULT_MAX_COST_USD,
        "--max-cost-usd",
        help="Hard kill switch — abort the run if estimated cost exceeds this.",
    ),
) -> None:
    """Populate `ctx.obj` with `CliState`. Runs before every subcommand."""
    _load_dotenv()
    if profile not in PROFILES:
        typer.echo(
            f"error: --profile must be one of {PROFILES}, got {profile!r}",
            err=True,
        )
        raise typer.Exit(code=2)
    if output_format not in _OUTPUT_FORMATS:
        typer.echo(
            f"error: --format must be one of {_OUTPUT_FORMATS}, got {output_format!r}",
            err=True,
        )
        raise typer.Exit(code=2)
    if max_cost_usd <= 0:
        typer.echo("error: --max-cost-usd must be positive", err=True)
        raise typer.Exit(code=2)
    ctx.obj = CliState(
        config_path=config,
        profile=profile,
        output_format=output_format,
        max_cost_usd=max_cost_usd,
    )


def _anthropic_key_present() -> bool:
    """Presence check only — never echoes the value."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _emit_stub(
    *,
    command: str,
    inputs: dict[str, object],
    next_step: str,
) -> None:
    """Structured JSON envelope describing a stub command outcome."""
    payload = {
        "command": command,
        "status": "stub",
        "inputs": inputs,
        "next_step": next_step,
        "anthropic_key_present": _anthropic_key_present(),
    }
    typer.echo(json.dumps(payload, indent=2))


# --- ingest ---


def _render_ingest_markdown(
    *,
    input_path: Path,
    profile: Profile,
    run_id: str,
    doc_id: str,
    doc_hash: str,
    stats: IngestStats,
    persisted_paths: dict[str, str],
) -> str:
    paths_rendered = "\n".join(f"- `{label}` → `{p}`" for label, p in persisted_paths.items())
    return (
        "# ctrldoc — ingest report\n"
        "\n"
        f"- **Source**: `{input_path}`\n"
        f"- **Document ID**: `{doc_id}`\n"
        f"- **Document hash**: `{doc_hash}`\n"
        f"- **Profile**: `{profile}`\n"
        f"- **Run ID**: `{run_id}`\n"
        "\n"
        "## Summary\n"
        "\n"
        "| Metric | Count |\n"
        "|---|---:|\n"
        f"| Sections parsed | {stats.sections_parsed} |\n"
        f"| Chunks indexed | {stats.chunks_indexed} |\n"
        f"| Entities indexed | {stats.entities_indexed} |\n"
        f"| Embedded tokens | {stats.embedded_tokens} |\n"
        "\n"
        "## Persisted artifacts\n"
        "\n"
        f"{paths_rendered}\n"
    )


@app.command()
def ingest(
    ctx: typer.Context,
    input_path: Path = typer.Argument(
        ...,
        exists=False,
        help="Path to the source document (Markdown today).",
    ),
    doc_id: str = typer.Option(
        "",
        "--doc-id",
        "-d",
        help="Logical id; defaults to the file stem.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Override `paths.runs_path` from the config for this run.",
    ),
) -> None:
    """Ingest a document end-to-end through the bundle's L0 pipeline.

    Heuristic profile uses `HashEmbedder` + `InMemoryStore` /
    `InMemoryVectorIndex` (no LLM, no Ollama) — the test-friendly
    path. Thrifty and production profiles use `OllamaEmbedder`
    (bge-m3) + `SQLiteStore` + `sqlite-vec`. BM25 is `TantivyBM25Index`
    on every profile. The per-doc index lives under
    ``<runs_path>/indexes/<doc_hash>.{db,vec.db,bm25/}``; the
    per-run report + result.json land in ``<runs_path>/<run_id>/``.
    """
    _require_input_path(input_path)
    state: CliState = ctx.obj
    config = _load_config(state.config_path)
    if output_dir is not None:
        config = config.model_copy(
            update={"paths": config.paths.model_copy(update={"runs_path": output_dir})}
        )

    text = input_path.read_text(encoding="utf-8")
    doc_hash = _doc_hash_for(text)
    effective_doc_id = doc_id or input_path.stem

    bundle = build_bundle(config=config, profile=state.profile)
    store, vector_index, bm25_index = _build_per_doc_backends(
        config=config,
        profile=state.profile,
        doc_hash=doc_hash,
    )

    parser = MarkdownParser()  # PDF / code routing will plug in later.
    stats = ingest_document(
        source=input_path,
        parser=parser,
        coref=bundle.coref,
        ner=bundle.ner,
        ner_labels=_DEFAULT_NER_LABELS,
        embedder=bundle.embedder,
        summarizer=bundle.summarizer,
        store=store,
        vector_index=vector_index,
        bm25_index=bm25_index,
    )

    run_id = new_run_id()
    runs_path = Path(config.paths.runs_path)
    run_dir = runs_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    index_dir = _per_doc_index_dir(config, doc_hash)

    persisted: dict[str, str] = {
        "report": str(run_dir / "report.md"),
        "result": str(run_dir / "result.json"),
        "bm25": str(index_dir / f"{doc_hash}.bm25"),
    }
    if state.profile != "heuristic":
        persisted["store"] = str(index_dir / f"{doc_hash}.db")
        persisted["vector_index"] = str(index_dir / f"{doc_hash}.vec.db")

    signature = {
        "chunk_ids": sorted(c.id for c in store.iter_chunks()),
        "section_ids": sorted(s.id for s in store.iter_sections()),
        "entity_ids": sorted(e.id for e in store.iter_entities()),
    }
    baseline = CanaryBaseline.from_signature(
        doc_id=effective_doc_id,
        playbook="ingest",
        signature=signature,
    )

    payload: dict[str, object] = {
        "command": "ingest",
        "status": "ok",
        "run_id": run_id,
        "doc_id": effective_doc_id,
        "doc_hash": doc_hash,
        "profile": state.profile,
        "input_path": str(input_path),
        "sections_parsed": stats.sections_parsed,
        "chunks_indexed": stats.chunks_indexed,
        "entities_indexed": stats.entities_indexed,
        "embedded_tokens": stats.embedded_tokens,
        "persisted": persisted,
        "signature": signature,
        "signature_hash": baseline.signature_hash,
    }

    markdown = _render_ingest_markdown(
        input_path=input_path,
        profile=state.profile,
        run_id=run_id,
        doc_id=effective_doc_id,
        doc_hash=doc_hash,
        stats=stats,
        persisted_paths=persisted,
    )

    (run_dir / "report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    # Keep the legacy canary signature file the way S-100 placed it
    # so the committed S-090 baseline check still has something to
    # compare against without going through the new result.json shape.
    legacy_sig_path = runs_path / f"{effective_doc_id}__ingest_signature.json"
    save_baseline(legacy_sig_path, baseline)
    legacy_stats_path = runs_path / f"{effective_doc_id}__ingest_stats.json"
    legacy_stats_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "doc_id": effective_doc_id,
                "input_path": str(input_path),
                "chunks_indexed": stats.chunks_indexed,
                "sections_indexed": stats.sections_parsed,
                "entities_indexed": stats.entities_indexed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    _emit_output(state, markdown=markdown, payload=payload)


# --- LLM-backed stubs (wired in S-113 .. S-117) ---


@app.command()
def qa(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="The question to ask the indexed corpus."),
    target_path: Path = typer.Option(
        ...,
        "--target",
        "-t",
        help="Markdown target document to answer against.",
    ),
    doc_id: str = typer.Option(
        "",
        "--doc-id",
        "-d",
        help="Logical id for the target doc; defaults to its file stem.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Override `paths.runs_path` from the config for this run.",
    ),
) -> None:
    """UC1 trustworthy QA — citation-grounded, verifier-bounded.

    Ingests the target doc inline, builds the bundle's retriever +
    cacheable prefix, runs `QAPlaybook` (retrieve → generate →
    decompose → verify each claim), then renders a Markdown answer
    plus a per-claim verification table. Heuristic profile rejected
    here: the playbook needs an LLM seam.
    """
    from ctrldoc.assembler import (
        CacheablePrefix,
        assemble_glossary,
        assemble_skeleton,
    )
    from ctrldoc.cli_audit import BundleRetriever
    from ctrldoc.cli_qa import VerifierRetriever, render_qa_markdown
    from ctrldoc.orch.task import StatelessTaskRunner
    from ctrldoc.playbooks.qa import QAPlaybook
    from ctrldoc.verify.claim_verifier import ClaimVerifier

    state: CliState = ctx.obj
    if not query.strip():
        typer.echo("error: query must not be blank", err=True)
        raise typer.Exit(code=2)
    _require_input_path(target_path)
    if state.profile == "heuristic":
        typer.echo(
            "error: qa requires --profile thrifty|production (heuristic has no LLM seam)",
            err=True,
        )
        raise typer.Exit(code=2)

    config = _load_config(state.config_path)
    if output_dir is not None:
        config = config.model_copy(
            update={"paths": config.paths.model_copy(update={"runs_path": output_dir})}
        )

    text = target_path.read_text(encoding="utf-8")
    doc_hash = _doc_hash_for(text)
    effective_doc_id = doc_id or target_path.stem

    bundle = build_bundle(config=config, profile=state.profile)
    store, vector_index, bm25_index = _build_per_doc_backends(
        config=config,
        profile=state.profile,
        doc_hash=doc_hash,
    )
    ingest_document(
        source=target_path,
        parser=MarkdownParser(),
        coref=bundle.coref,
        ner=bundle.ner,
        ner_labels=_DEFAULT_NER_LABELS,
        embedder=bundle.embedder,
        summarizer=bundle.summarizer,
        store=store,
        vector_index=vector_index,
        bm25_index=bm25_index,
    )

    skeleton = assemble_skeleton(store)
    glossary = assemble_glossary(store)
    bundle_retriever = BundleRetriever(
        bundle=bundle,
        store=store,
        vector_index=vector_index,
        bm25_index=bm25_index,
        prefix_skeleton=skeleton,
        prefix_glossary=glossary,
    )

    prefix = CacheablePrefix(
        system_prompt=_QA_SYSTEM_PROMPT,
        doc_skeleton=skeleton,
        entity_glossary=glossary,
    )

    assert bundle.task_client_router is not None  # guarded by profile check above
    task_client = bundle.task_client_router.for_tier("local")
    task_runner = StatelessTaskRunner(client=task_client)

    verifier = ClaimVerifier(
        nli=bundle.nli_checker,
        judge=bundle.llm_judge,
        retriever=VerifierRetriever(bundle_retriever=bundle_retriever),
    )
    playbook = QAPlaybook(
        prefix=prefix,
        retriever=bundle_retriever,
        task_runner=task_runner,
        decomposer=bundle.claim_decomposer,
        verifier=verifier,
    )

    report = playbook.run(query)

    run_id = new_run_id()
    runs_path = Path(config.paths.runs_path)
    run_dir = runs_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    markdown = render_qa_markdown(
        report=report,
        target_path=target_path,
        profile=state.profile,
        run_id=run_id,
    )
    payload: dict[str, object] = {
        "command": "qa",
        "status": "ok",
        "run_id": run_id,
        "query": query,
        "target_path": str(target_path),
        "target_doc_id": effective_doc_id,
        "target_doc_hash": doc_hash,
        "profile": state.profile,
        "max_cost_usd": state.max_cost_usd,
        "answer": report.answer,
        "claims": [c.model_dump(mode="json") for c in report.claims],
    }

    (run_dir / "report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    _emit_output(state, markdown=markdown, payload=payload)


_QA_SYSTEM_PROMPT = (
    "You are a citation-grounded QA system. Answer the user's QUERY "
    "using only the EVIDENCE spans (each is labelled `[chunk_id] text`). "
    "If the evidence does not contain the answer, say so plainly — do "
    "not speculate.\n\n"
    "Return one JSON object of shape:\n"
    '  {"answer": <plain-text answer, may quote spans by [chunk_id]>}\n'
    "No prose outside the JSON."
)


@app.command()
def audit(
    ctx: typer.Context,
    checklist_path: Path = typer.Option(
        ...,
        "--checklist",
        "-c",
        help="Markdown checklist file (one H2/H3 per item).",
    ),
    target_path: Path = typer.Option(
        ...,
        "--target",
        "-t",
        help="Markdown target document to audit against the checklist.",
    ),
    doc_id: str = typer.Option(
        "",
        "--doc-id",
        "-d",
        help="Logical id for the target doc; defaults to its file stem.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Override `paths.runs_path` from the config for this run.",
    ),
) -> None:
    """UC2 coverage audit — checklist (Markdown) vs target (Markdown).

    Ingests the target inline through the bundle, parses the checklist
    into `ChecklistItem`s via a deterministic Markdown-section parser,
    and runs `CoverageAuditPlaybook`. Per-item LLM calls route through
    the bundle's `task_client_router` (`local` tier in thrifty,
    `opus` in production). Emits a Markdown report grouped by verdict
    plus a JSON payload under `<runs_path>/<run_id>/`.

    The heuristic profile is rejected here: the playbook needs an LLM
    to judge coverage, and heuristic mode has no `task_client_router`.
    """
    from ctrldoc.assembler import (
        assemble_glossary,
        assemble_skeleton,
    )
    from ctrldoc.cli_audit import (
        BundleRetriever,
        parse_checklist_markdown,
        render_coverage_markdown,
    )
    from ctrldoc.ingest.pipeline import ingest_document
    from ctrldoc.orch.batch import BatchedTaskRunner
    from ctrldoc.playbooks.coverage import CoverageAuditPlaybook

    state: CliState = ctx.obj
    _require_input_path(checklist_path)
    _require_input_path(target_path)
    if state.profile == "heuristic":
        typer.echo(
            "error: audit requires --profile thrifty|production (heuristic has no LLM seam)",
            err=True,
        )
        raise typer.Exit(code=2)

    config = _load_config(state.config_path)
    if output_dir is not None:
        config = config.model_copy(
            update={"paths": config.paths.model_copy(update={"runs_path": output_dir})}
        )

    target_text = target_path.read_text(encoding="utf-8")
    doc_hash = _doc_hash_for(target_text)
    effective_doc_id = doc_id or target_path.stem

    bundle = build_bundle(config=config, profile=state.profile)
    store, vector_index, bm25_index = _build_per_doc_backends(
        config=config,
        profile=state.profile,
        doc_hash=doc_hash,
    )
    ingest_document(
        source=target_path,
        parser=MarkdownParser(),
        coref=bundle.coref,
        ner=bundle.ner,
        ner_labels=_DEFAULT_NER_LABELS,
        embedder=bundle.embedder,
        summarizer=bundle.summarizer,
        store=store,
        vector_index=vector_index,
        bm25_index=bm25_index,
    )

    items = parse_checklist_markdown(checklist_path.read_text(encoding="utf-8"))
    if not items:
        typer.echo(
            f"error: no checklist items extracted from {checklist_path} "
            "(expected `##` or `###` Markdown headings)",
            err=True,
        )
        raise typer.Exit(code=2)

    skeleton = assemble_skeleton(store)
    glossary = assemble_glossary(store)
    retriever = BundleRetriever(
        bundle=bundle,
        store=store,
        vector_index=vector_index,
        bm25_index=bm25_index,
        prefix_skeleton=skeleton,
        prefix_glossary=glossary,
    )

    from ctrldoc.assembler import CacheablePrefix

    prefix = CacheablePrefix(
        system_prompt=_COVERAGE_AUDIT_SYSTEM_PROMPT,
        doc_skeleton=skeleton,
        entity_glossary=glossary,
    )

    assert bundle.task_client_router is not None  # guarded by profile check above
    task_client = bundle.task_client_router.for_tier("local")
    batched_runner = BatchedTaskRunner(client=task_client)
    playbook = CoverageAuditPlaybook(
        prefix=prefix,
        retriever=retriever,
        batched_runner=batched_runner,
    )

    report = playbook.run(items)

    run_id = new_run_id()
    runs_path = Path(config.paths.runs_path)
    run_dir = runs_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    markdown = render_coverage_markdown(
        report=report,
        items=items,
        checklist_path=checklist_path,
        target_path=target_path,
        profile=state.profile,
        run_id=run_id,
    )
    payload: dict[str, object] = {
        "command": "audit",
        "status": "ok",
        "run_id": run_id,
        "checklist_path": str(checklist_path),
        "target_path": str(target_path),
        "target_doc_id": effective_doc_id,
        "target_doc_hash": doc_hash,
        "profile": state.profile,
        "max_cost_usd": state.max_cost_usd,
        "items_total": len(items),
        "verdicts": [v.model_dump(mode="json") for v in report.verdicts],
        "summary": {
            label: sum(1 for v in report.verdicts if v.verdict == label)
            for label in ("Covered", "Partial", "NotCovered", "Ambiguous")
        },
    }

    (run_dir / "report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    _emit_output(state, markdown=markdown, payload=payload)


_COVERAGE_AUDIT_SYSTEM_PROMPT = (
    "You are a strict coverage auditor. For each checklist item, "
    "decide whether the EVIDENCE supports it.\n\n"
    "Return a JSON object mapping `id` → verdict object with this exact "
    "shape:\n"
    '  {"verdict": "Covered"|"Partial"|"NotCovered"|"Ambiguous",\n'
    '   "confidence": <0.0-1.0>,\n'
    '   "citation_chunk_ids": [<chunk-id strings copied from EVIDENCE>]}\n\n'
    "Only cite chunk_ids that appear in the EVIDENCE. Be conservative — "
    "if the evidence does not directly support the item, mark NotCovered "
    "or Ambiguous. No prose outside the JSON object."
)


@app.command()
def review(
    ctx: typer.Context,
    doc_type: str = typer.Argument(..., help="Document type, e.g. `Aurora L0 kernel spec`."),
    target_path: Path = typer.Option(
        ...,
        "--target",
        "-t",
        help="Markdown target document to review.",
    ),
    doc_id: str = typer.Option(
        "",
        "--doc-id",
        "-d",
        help="Logical id for the target doc; defaults to its file stem.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Override `paths.runs_path` from the config for this run.",
    ),
) -> None:
    """UC4 analytical review — lens fan-out then one synthesis call.

    Enumerates the canonical 5-lens set for the `doc_type`
    (`HeuristicLensGenerator`), runs one LLM sweep per lens via the
    bundle's `local` tier, then a single synthesis call routed
    through the `opus` tier (the only Opus call per playbook run in
    thrifty mode). Emits a Markdown narrative + per-lens findings
    table.
    """
    from ctrldoc.assembler import (
        CacheablePrefix,
        assemble_glossary,
        assemble_skeleton,
    )
    from ctrldoc.cli_audit import BundleRetriever
    from ctrldoc.cli_review import LLMLensSweeper, render_review_markdown
    from ctrldoc.orch.synthesis import SynthesisRunner
    from ctrldoc.orch.task import StatelessTaskRunner
    from ctrldoc.playbooks.review import (
        AnalyticalReviewPlaybook,
        HeuristicLensGenerator,
    )

    state: CliState = ctx.obj
    if not doc_type.strip():
        typer.echo("error: doc_type must not be blank", err=True)
        raise typer.Exit(code=2)
    _require_input_path(target_path)
    if state.profile == "heuristic":
        typer.echo(
            "error: review requires --profile thrifty|production (heuristic has no LLM seam)",
            err=True,
        )
        raise typer.Exit(code=2)

    config = _load_config(state.config_path)
    if output_dir is not None:
        config = config.model_copy(
            update={"paths": config.paths.model_copy(update={"runs_path": output_dir})}
        )

    text = target_path.read_text(encoding="utf-8")
    doc_hash = _doc_hash_for(text)
    effective_doc_id = doc_id or target_path.stem

    bundle = build_bundle(config=config, profile=state.profile)
    store, vector_index, bm25_index = _build_per_doc_backends(
        config=config,
        profile=state.profile,
        doc_hash=doc_hash,
    )
    ingest_document(
        source=target_path,
        parser=MarkdownParser(),
        coref=bundle.coref,
        ner=bundle.ner,
        ner_labels=_DEFAULT_NER_LABELS,
        embedder=bundle.embedder,
        summarizer=bundle.summarizer,
        store=store,
        vector_index=vector_index,
        bm25_index=bm25_index,
    )

    skeleton = assemble_skeleton(store)
    glossary = assemble_glossary(store)
    bundle_retriever = BundleRetriever(
        bundle=bundle,
        store=store,
        vector_index=vector_index,
        bm25_index=bm25_index,
        prefix_skeleton=skeleton,
        prefix_glossary=glossary,
    )
    sweep_prefix = CacheablePrefix(
        system_prompt=(
            "You are a strict analytical reviewer. Return one JSON object "
            'of shape {"findings": [{"claim": str, "severity": '
            '"info"|"warn"|"critical", "citation_chunk_id": str}]}. '
            "Cite only chunk_ids that appear in the EVIDENCE. No prose "
            "outside the JSON."
        ),
        doc_skeleton=skeleton,
        entity_glossary=glossary,
    )
    synth_prefix = CacheablePrefix(
        system_prompt=(
            "You are an analytical-review synthesiser. Read the structured "
            "findings JSON and emit one JSON object of shape "
            '{"headline": str, "sections": [str], "summary": str}. '
            "Do not invent findings; group the ones you see by theme."
        ),
        doc_skeleton=skeleton,
        entity_glossary=glossary,
    )

    assert bundle.task_client_router is not None  # guarded by profile check above
    local_client = bundle.task_client_router.for_tier("local")
    opus_client = bundle.task_client_router.for_tier("opus")
    sweep_runner = StatelessTaskRunner(client=local_client)
    synthesis_runner = SynthesisRunner(client=opus_client)
    sweeper = LLMLensSweeper(
        prefix=sweep_prefix,
        retriever=bundle_retriever,
        task_runner=sweep_runner,
    )
    playbook = AnalyticalReviewPlaybook(
        prefix=synth_prefix,
        lens_generator=HeuristicLensGenerator(),
        sweeper=sweeper,
        synthesis_runner=synthesis_runner,
    )

    report = playbook.run(doc_type)

    run_id = new_run_id()
    runs_path = Path(config.paths.runs_path)
    run_dir = runs_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    markdown = render_review_markdown(
        report=report,
        target_path=target_path,
        profile=state.profile,
        run_id=run_id,
    )
    payload: dict[str, object] = {
        "command": "review",
        "status": "ok",
        "run_id": run_id,
        "doc_type": doc_type,
        "target_path": str(target_path),
        "target_doc_id": effective_doc_id,
        "target_doc_hash": doc_hash,
        "profile": state.profile,
        "max_cost_usd": state.max_cost_usd,
        "findings": [f.model_dump(mode="json") for f in report.findings],
        "narrative": report.narrative.model_dump(mode="json"),
    }

    (run_dir / "report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    _emit_output(state, markdown=markdown, payload=payload)


@app.command()
def scan(
    ctx: typer.Context,
    target_path: Path = typer.Option(
        ...,
        "--target",
        "-t",
        help="Markdown target document to scan.",
    ),
    doc_id: str = typer.Option(
        "",
        "--doc-id",
        "-d",
        help="Logical id for the target doc; defaults to its file stem.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Override `paths.runs_path` from the config for this run.",
    ),
) -> None:
    """UC5 anomaly scan — deterministic detector battery over the target.

    No LLM dependency: runs `HedgeWordDetector` + `EmptySummaryDetector`
    (§5.5 baseline) against the ingested target. Works in every
    profile including `heuristic`. Emits a Markdown triage queue
    grouped by detector + a JSON payload of every finding.
    """
    from ctrldoc.cli_scan import render_scan_markdown

    state: CliState = ctx.obj
    _require_input_path(target_path)
    config = _load_config(state.config_path)
    if output_dir is not None:
        config = config.model_copy(
            update={"paths": config.paths.model_copy(update={"runs_path": output_dir})}
        )

    text = target_path.read_text(encoding="utf-8")
    doc_hash = _doc_hash_for(text)
    effective_doc_id = doc_id or target_path.stem

    bundle = build_bundle(config=config, profile=state.profile)
    store, vector_index, bm25_index = _build_per_doc_backends(
        config=config,
        profile=state.profile,
        doc_hash=doc_hash,
    )
    ingest_document(
        source=target_path,
        parser=MarkdownParser(),
        coref=bundle.coref,
        ner=bundle.ner,
        ner_labels=_DEFAULT_NER_LABELS,
        embedder=bundle.embedder,
        summarizer=bundle.summarizer,
        store=store,
        vector_index=vector_index,
        bm25_index=bm25_index,
    )

    playbook = AnomalyScanPlaybook(
        detectors=[HedgeWordDetector(), EmptySummaryDetector()],
    )
    queue = playbook.run(store=store)

    run_id = new_run_id()
    runs_path = Path(config.paths.runs_path)
    run_dir = runs_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    markdown = render_scan_markdown(
        queue=queue,
        target_path=target_path,
        profile=state.profile,
        run_id=run_id,
    )
    payload: dict[str, object] = {
        "command": "scan",
        "status": "ok",
        "run_id": run_id,
        "target_path": str(target_path),
        "target_doc_id": effective_doc_id,
        "target_doc_hash": doc_hash,
        "profile": state.profile,
        "findings_total": len(queue.findings),
        "findings": [
            {
                "detector": finding.ctrldoc,
                "severity": finding.severity,
                "claim": finding.claim,
                "chunk_id": finding.location.chunk_id,
                "text": finding.location.text,
            }
            for finding in queue.findings
        ],
    }

    (run_dir / "report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    _emit_output(state, markdown=markdown, payload=payload)


@app.command(name="map")
def map_(
    ctx: typer.Context,
    target_path: Path = typer.Option(
        ...,
        "--target",
        "-t",
        help="Markdown target document to map.",
    ),
    doc_id: str = typer.Option(
        "",
        "--doc-id",
        "-d",
        help="Logical id for the target doc; defaults to its file stem.",
    ),
    max_concepts: int = typer.Option(
        10,
        "--max-concepts",
        help="Cap on the number of concepts to map; bounds the O(M²) pair fan-out.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Override `paths.runs_path` from the config for this run.",
    ),
) -> None:
    """UC6 concept relation map — typed edges between document concepts.

    Pulls top-`--max-concepts` entities from the ingested target as
    nodes; iterates pair (c_i, c_j) over the upper triangle, fetches
    co-occurrence evidence via the bundle retriever, and asks the
    bundle's `local` tier (Ollama Qwen in thrifty) to classify the
    relation. Pairs with no co-occurrence evidence are dropped
    before the classifier call; pairs the model marks `unrelated`
    are dropped from the final graph. Emits a Markdown adjacency
    table + Mermaid graph.
    """
    from ctrldoc.assembler import (
        CacheablePrefix,
        assemble_glossary,
        assemble_skeleton,
    )
    from ctrldoc.cli_audit import BundleRetriever
    from ctrldoc.cli_map import (
        BundleCoOccurrenceRetriever,
        LLMRelationClassifier,
        StoreEntityConceptExtractor,
        render_map_markdown,
    )
    from ctrldoc.orch.task import StatelessTaskRunner
    from ctrldoc.playbooks.relations import RelationMapPlaybook

    state: CliState = ctx.obj
    _require_input_path(target_path)
    if state.profile == "heuristic":
        typer.echo(
            "error: map requires --profile thrifty|production (heuristic has no LLM seam)",
            err=True,
        )
        raise typer.Exit(code=2)
    if max_concepts <= 0:
        typer.echo("error: --max-concepts must be positive", err=True)
        raise typer.Exit(code=2)

    config = _load_config(state.config_path)
    if output_dir is not None:
        config = config.model_copy(
            update={"paths": config.paths.model_copy(update={"runs_path": output_dir})}
        )

    text = target_path.read_text(encoding="utf-8")
    doc_hash = _doc_hash_for(text)
    effective_doc_id = doc_id or target_path.stem

    bundle = build_bundle(config=config, profile=state.profile)
    store, vector_index, bm25_index = _build_per_doc_backends(
        config=config,
        profile=state.profile,
        doc_hash=doc_hash,
    )
    ingest_document(
        source=target_path,
        parser=MarkdownParser(),
        coref=bundle.coref,
        ner=bundle.ner,
        ner_labels=_DEFAULT_NER_LABELS,
        embedder=bundle.embedder,
        summarizer=bundle.summarizer,
        store=store,
        vector_index=vector_index,
        bm25_index=bm25_index,
    )

    skeleton = assemble_skeleton(store)
    glossary = assemble_glossary(store)
    bundle_retriever = BundleRetriever(
        bundle=bundle,
        store=store,
        vector_index=vector_index,
        bm25_index=bm25_index,
        prefix_skeleton=skeleton,
        prefix_glossary=glossary,
    )

    prefix = CacheablePrefix(
        system_prompt=_RELATION_CLASSIFIER_SYSTEM_PROMPT,
        doc_skeleton=skeleton,
        entity_glossary=glossary,
    )

    assert bundle.task_client_router is not None  # guarded by profile check above
    task_client = bundle.task_client_router.for_tier("local")
    task_runner = StatelessTaskRunner(client=task_client)

    playbook = RelationMapPlaybook(
        extractor=StoreEntityConceptExtractor(store=store, max_concepts=max_concepts),
        retriever=BundleCoOccurrenceRetriever(bundle_retriever=bundle_retriever),
        classifier=LLMRelationClassifier(prefix=prefix, task_runner=task_runner),
    )

    graph = playbook.run()

    run_id = new_run_id()
    runs_path = Path(config.paths.runs_path)
    run_dir = runs_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    markdown = render_map_markdown(
        graph=graph,
        target_path=target_path,
        profile=state.profile,
        run_id=run_id,
    )
    payload: dict[str, object] = {
        "command": "map",
        "status": "ok",
        "run_id": run_id,
        "target_path": str(target_path),
        "target_doc_id": effective_doc_id,
        "target_doc_hash": doc_hash,
        "profile": state.profile,
        "max_concepts": max_concepts,
        "nodes": [c.model_dump(mode="json") for c in graph.nodes],
        "edges": [e.model_dump(mode="json") for e in graph.edges],
    }

    (run_dir / "report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    _emit_output(state, markdown=markdown, payload=payload)


_RELATION_CLASSIFIER_SYSTEM_PROMPT = (
    "You classify the relation between two concepts using only the "
    "EVIDENCE spans. Return one JSON object of shape:\n"
    '  {"type": <one of: depends_on, contradicts, refines, instantiates, '
    "conflicts_with, prerequisite_of, alternative_to, unrelated>,\n"
    '   "confidence": <0.0-1.0>,\n'
    '   "citation_chunk_ids": [<chunk_id copied from EVIDENCE>]}\n\n'
    "Pick `unrelated` if the evidence does not establish a relation "
    "between the two concepts. Cite only chunk_ids that appear in "
    "EVIDENCE. No prose outside the JSON."
)


def main() -> None:
    """Entry point for ``python -m ctrldoc``."""
    app()


if __name__ == "__main__":
    main()


__all__ = [
    "CliState",
    "OutputFormat",
    "app",
    "main",
]
