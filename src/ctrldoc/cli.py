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
from typing import Literal, cast

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
from ctrldoc.ingest.parser_dispatch import get_parser
from ctrldoc.ingest.pipeline import IngestStats, ingest_document
from ctrldoc.ops.scan import (
    AnomalyScanPlaybook,
    EmptySummaryDetector,
    HedgeWordDetector,
)
from ctrldoc.provenance import new_run_id
from ctrldoc.store import Store
from ctrldoc.store.bm25 import BM25Index, TantivyBM25Index
from ctrldoc.store.memory import InMemoryStore
from ctrldoc.store.sqlite import SQLiteStore
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

workspace_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "Workspace primitives: create / add / list / info. A workspace "
        "is a typed collection of doc-graphs sharing one concept lattice."
    ),
)
app.add_typer(workspace_app, name="workspace")

ledger_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "Verdict ledger: list / show / replay. The ledger is the "
        "append-only audit trail of every L4 verdict; replay re-runs a "
        "recorded operation and checks the ±0.02 determinism gate."
    ),
)
app.add_typer(ledger_app, name="ledger")

mcp_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "Model Context Protocol server: expose the §6.10 tool surface "
        "over stdio so any MCP-compatible host (Claude Desktop, Claude "
        "CLI, third-party) can drive the substrate."
    ),
)
app.add_typer(mcp_app, name="mcp")

_WORKSPACE_DB_FILENAME = "workspaces.db"
_LEDGER_DB_FILENAME = "ledger.db"


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


def _doc_hash_for_path(path: Path) -> str:
    """16-char hex prefix of sha256 over the source's raw bytes.

    Hashing bytes (not decoded text) lets a single helper cover both
    text sources (`.md`/`.markdown`/`.txt`) and binary sources
    (`.pdf`) without forcing every caller to know the parser
    routing. Content-derived ids stay byte-deterministic across
    re-ingest of the same source path.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


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

    doc_hash = _doc_hash_for_path(input_path)
    effective_doc_id = doc_id or input_path.stem

    bundle = build_bundle(config=config, profile=state.profile)
    store, vector_index, bm25_index = _build_per_doc_backends(
        config=config,
        profile=state.profile,
        doc_hash=doc_hash,
    )

    parser = get_parser(input_path)
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
    from ctrldoc.ops.qa import QAPlaybook
    from ctrldoc.orch.task import StatelessTaskRunner
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

    doc_hash = _doc_hash_for_path(target_path)
    effective_doc_id = doc_id or target_path.stem

    bundle = build_bundle(config=config, profile=state.profile)
    store, vector_index, bm25_index = _build_per_doc_backends(
        config=config,
        profile=state.profile,
        doc_hash=doc_hash,
    )
    ingest_document(
        source=target_path,
        parser=get_parser(target_path),
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
        SequentialBatchedRunner,
        parse_checklist_markdown,
        render_coverage_markdown,
    )
    from ctrldoc.ingest.pipeline import ingest_document
    from ctrldoc.ops.audit import CoverageAuditPlaybook
    from ctrldoc.orch.task import StatelessTaskRunner

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

    doc_hash = _doc_hash_for_path(target_path)
    effective_doc_id = doc_id or target_path.stem

    bundle = build_bundle(config=config, profile=state.profile)
    store, vector_index, bm25_index = _build_per_doc_backends(
        config=config,
        profile=state.profile,
        doc_hash=doc_hash,
    )
    ingest_document(
        source=target_path,
        parser=get_parser(target_path),
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
    stateless_runner = StatelessTaskRunner(client=task_client)
    # The audit playbook hard-codes its private `_BatchedVerdict` shape;
    # import it so the fallback can emit a typed instance when a single
    # Ollama call fails to parse. Without this fallback the whole audit
    # aborts on the first bad model response.
    from ctrldoc.ops.audit import _BatchedVerdict
    from ctrldoc.orch.batch import BatchItem as _BatchItem

    def _audit_fallback(item: _BatchItem, exc: Exception) -> _BatchedVerdict:
        del item, exc  # documented; surfaced in result.json via `Ambiguous`
        return _BatchedVerdict(verdict="Ambiguous", confidence=0.0, citation_chunk_ids=[])

    sequential_runner = SequentialBatchedRunner(
        stateless=stateless_runner, on_error=_audit_fallback
    )
    playbook = CoverageAuditPlaybook(
        prefix=prefix,
        retriever=retriever,
        batched_runner=sequential_runner,  # type: ignore[arg-type]
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
    "You are a strict coverage auditor. You will receive EVIDENCE "
    "spans from a target document plus one checklist item per call. "
    "Decide whether the evidence supports the item, and return one "
    "JSON object with this exact shape:\n"
    '  {"verdict": "Covered"|"Partial"|"NotCovered"|"Ambiguous",\n'
    '   "confidence": <number 0.0-1.0>,\n'
    '   "citation_chunk_ids": [<chunk_id strings copied from EVIDENCE>]}\n\n'
    "Only cite chunk_ids that literally appear in the EVIDENCE. Be "
    "conservative — if the evidence does not directly support the "
    "item, mark NotCovered or Ambiguous. Emit one JSON object only, "
    "no prose, no code fences."
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
    from ctrldoc.ops.review import (
        AnalyticalReviewPlaybook,
        HeuristicLensGenerator,
    )
    from ctrldoc.orch.synthesis import SynthesisRunner
    from ctrldoc.orch.task import StatelessTaskRunner

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

    doc_hash = _doc_hash_for_path(target_path)
    effective_doc_id = doc_id or target_path.stem

    bundle = build_bundle(config=config, profile=state.profile)
    store, vector_index, bm25_index = _build_per_doc_backends(
        config=config,
        profile=state.profile,
        doc_hash=doc_hash,
    )
    ingest_document(
        source=target_path,
        parser=get_parser(target_path),
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

    doc_hash = _doc_hash_for_path(target_path)
    effective_doc_id = doc_id or target_path.stem

    bundle = build_bundle(config=config, profile=state.profile)
    store, vector_index, bm25_index = _build_per_doc_backends(
        config=config,
        profile=state.profile,
        doc_hash=doc_hash,
    )
    ingest_document(
        source=target_path,
        parser=get_parser(target_path),
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
    from ctrldoc.ops.map import RelationMapPlaybook
    from ctrldoc.orch.task import StatelessTaskRunner

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

    doc_hash = _doc_hash_for_path(target_path)
    effective_doc_id = doc_id or target_path.stem

    bundle = build_bundle(config=config, profile=state.profile)
    store, vector_index, bm25_index = _build_per_doc_backends(
        config=config,
        profile=state.profile,
        doc_hash=doc_hash,
    )
    ingest_document(
        source=target_path,
        parser=get_parser(target_path),
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


# --- workspace subcommands (§6.7, §9) ---


def _workspace_store_path(config: Config) -> Path:
    """Resolve the per-installation workspaces DB.

    The workspaces DB is a separate SQLite file from the per-doc
    indexes (which live at ``<runs_path>/indexes/<doc_hash>.db``) so
    workspace CRUD has zero coupling to whichever docs have been
    ingested. The parent dir is created on demand — the user's first
    ``workspace create`` call bootstraps the file.
    """
    runs_path = Path(config.paths.runs_path)
    runs_path.mkdir(parents=True, exist_ok=True)
    return runs_path / _WORKSPACE_DB_FILENAME


def _open_workspace_store(config: Config) -> SQLiteStore:
    return SQLiteStore(_workspace_store_path(config))


def _render_workspace_create_markdown(*, name: str, workspace_id: str) -> str:
    return f"# ctrldoc — workspace create\n\n- **Name**: `{name}`\n- **ID**: `{workspace_id}`\n"


def _render_workspace_add_markdown(*, name: str, doc_ids: list[str]) -> str:
    docs_rendered = "\n".join(f"- `{d}`" for d in doc_ids) or "_(empty)_"
    return (
        "# ctrldoc — workspace add\n"
        "\n"
        f"- **Name**: `{name}`\n"
        f"- **Doc count**: {len(doc_ids)}\n"
        "\n"
        "## Documents\n"
        "\n"
        f"{docs_rendered}\n"
    )


def _render_workspace_list_markdown(*, rows: list[dict[str, object]]) -> str:
    if not rows:
        return "# ctrldoc — workspace list\n\n_(no workspaces yet)_\n"
    body = "\n".join(f"| `{r['name']}` | `{r['id']}` | {r['doc_count']} |" for r in rows)
    return f"# ctrldoc — workspace list\n\n| Name | ID | Doc count |\n|---|---|---:|\n{body}\n"


def _render_workspace_info_markdown(
    *,
    name: str,
    workspace_id: str,
    doc_ids: list[str],
    concept_count: int,
) -> str:
    docs_rendered = "\n".join(f"- `{d}`" for d in doc_ids) or "_(empty)_"
    return (
        "# ctrldoc — workspace info\n"
        "\n"
        f"- **Name**: `{name}`\n"
        f"- **ID**: `{workspace_id}`\n"
        f"- **Doc count**: {len(doc_ids)}\n"
        f"- **Concept count**: {concept_count}\n"
        "\n"
        "## Documents\n"
        "\n"
        f"{docs_rendered}\n"
    )


@workspace_app.command("create")
def workspace_create(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="The workspace's human-readable name."),
) -> None:
    """Create a new workspace; name must be unique."""
    from ctrldoc.ops.workspace import (
        WorkspaceAlreadyExistsError,
        WorkspaceManager,
    )

    state: CliState = ctx.obj
    if not name.strip():
        typer.echo("error: workspace name must not be blank", err=True)
        raise typer.Exit(code=2)
    config = _load_config(state.config_path)
    with _open_workspace_store(config) as store:
        manager = WorkspaceManager(store=store)
        try:
            workspace = manager.create(name)
        except WorkspaceAlreadyExistsError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=2) from exc
    payload: dict[str, object] = {
        "command": "workspace create",
        "status": "ok",
        "name": workspace.name,
        "id": workspace.id,
        "doc_ids": list(workspace.doc_ids),
    }
    markdown = _render_workspace_create_markdown(name=workspace.name, workspace_id=workspace.id)
    _emit_output(state, markdown=markdown, payload=payload)


@workspace_app.command("add")
def workspace_add(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="The workspace name."),
    doc_id: str = typer.Argument(..., help="Logical id of the doc to attach."),
) -> None:
    """Attach a document to an existing workspace; idempotent."""
    from ctrldoc.ops.workspace import WorkspaceManager, WorkspaceNotFoundError

    state: CliState = ctx.obj
    if not doc_id.strip():
        typer.echo("error: doc_id must not be blank", err=True)
        raise typer.Exit(code=2)
    config = _load_config(state.config_path)
    with _open_workspace_store(config) as store:
        manager = WorkspaceManager(store=store)
        try:
            workspace = manager.add(name, doc_id)
        except WorkspaceNotFoundError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=2) from exc
    payload: dict[str, object] = {
        "command": "workspace add",
        "status": "ok",
        "name": workspace.name,
        "id": workspace.id,
        "doc_ids": list(workspace.doc_ids),
    }
    markdown = _render_workspace_add_markdown(name=workspace.name, doc_ids=list(workspace.doc_ids))
    _emit_output(state, markdown=markdown, payload=payload)


@workspace_app.command("list")
def workspace_list(ctx: typer.Context) -> None:
    """List every workspace in creation order."""
    from ctrldoc.ops.workspace import WorkspaceManager

    state: CliState = ctx.obj
    config = _load_config(state.config_path)
    with _open_workspace_store(config) as store:
        manager = WorkspaceManager(store=store)
        workspaces = manager.list()
    rows: list[dict[str, object]] = [
        {
            "name": w.name,
            "id": w.id,
            "doc_count": len(w.doc_ids),
            "doc_ids": list(w.doc_ids),
        }
        for w in workspaces
    ]
    payload: dict[str, object] = {
        "command": "workspace list",
        "status": "ok",
        "workspaces": rows,
    }
    markdown = _render_workspace_list_markdown(rows=rows)
    _emit_output(state, markdown=markdown, payload=payload)


@workspace_app.command("info")
def workspace_info(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="The workspace name."),
) -> None:
    """Show the workspace's docs and shared-concept-lattice rollup."""
    from ctrldoc.ops.workspace import WorkspaceManager, WorkspaceNotFoundError

    state: CliState = ctx.obj
    config = _load_config(state.config_path)
    with _open_workspace_store(config) as store:
        manager = WorkspaceManager(store=store)
        try:
            info = manager.info(name)
        except WorkspaceNotFoundError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=2) from exc
    payload: dict[str, object] = {
        "command": "workspace info",
        "status": "ok",
        "name": info.workspace.name,
        "id": info.workspace.id,
        "doc_count": info.doc_count,
        "doc_ids": list(info.workspace.doc_ids),
        "concept_count": info.concept_count,
        "shared_concept_ids": list(info.shared_concept_ids),
    }
    markdown = _render_workspace_info_markdown(
        name=info.workspace.name,
        workspace_id=info.workspace.id,
        doc_ids=list(info.workspace.doc_ids),
        concept_count=info.concept_count,
    )
    _emit_output(state, markdown=markdown, payload=payload)


# --- ledger subcommands (§6.5, §11) ---


def _ledger_store_path(config: Config) -> Path:
    """Resolve the per-installation ledger DB.

    The verdict ledger lives in its own SQLite file (separate from the
    per-doc indexes and the workspace DB) so an auditor can ship a
    single ``ledger.db`` snapshot without dragging unrelated artefacts.
    The parent dir is created on demand — the first
    ``ledger {list,show,replay}`` call bootstraps the file.
    """
    runs_path = Path(config.paths.runs_path)
    runs_path.mkdir(parents=True, exist_ok=True)
    return runs_path / _LEDGER_DB_FILENAME


def _open_ledger_store(config: Config) -> SQLiteStore:
    return SQLiteStore(_ledger_store_path(config))


def _ledger_entry_payload(entry: object) -> dict[str, object]:
    """Render one `LedgerEntry` into a JSON-safe dict for the CLI envelope."""
    from ctrldoc.orch.ledger import LedgerEntry

    assert isinstance(entry, LedgerEntry)
    return {
        "id": entry.id,
        "workspace_id": entry.workspace_id,
        "operation": entry.operation,
        "inputs": dict(entry.inputs),
        "output": dict(entry.output),
        "calibrated_confidence": entry.calibrated_confidence,
        "model_versions": dict(entry.model_versions),
        "paraphrase_votes": (
            dict(entry.paraphrase_votes) if entry.paraphrase_votes is not None else None
        ),
        "timestamp": entry.timestamp,
    }


def _render_ledger_list_markdown(*, rows: list[dict[str, object]]) -> str:
    if not rows:
        return "# ctrldoc — ledger list\n\n_(no entries yet)_\n"
    header = "| ID | Workspace | Operation | Confidence | Timestamp |\n|---:|---|---|---:|---|\n"
    body = "\n".join(
        f"| {r['id']} | `{r['workspace_id']}` | `{r['operation']}` | "
        f"{cast(float, r['calibrated_confidence']):.3f} | `{r['timestamp']}` |"
        for r in rows
    )
    return f"# ctrldoc — ledger list\n\n{header}{body}\n"


def _render_ledger_show_markdown(*, entry: dict[str, object]) -> str:
    return (
        "# ctrldoc — ledger show\n"
        "\n"
        f"- **ID**: {entry['id']}\n"
        f"- **Workspace**: `{entry['workspace_id']}`\n"
        f"- **Operation**: `{entry['operation']}`\n"
        f"- **Calibrated confidence**: {cast(float, entry['calibrated_confidence']):.3f}\n"
        f"- **Timestamp**: `{entry['timestamp']}`\n"
    )


def _render_ledger_replay_markdown(*, payload: dict[str, object]) -> str:
    flag = "PASS" if payload["is_deterministic"] else "FAIL"
    return (
        "# ctrldoc — ledger replay\n"
        "\n"
        f"- **Entry ID**: {payload['entry_id']}\n"
        f"- **Operation**: `{payload['operation']}`\n"
        f"- **Persisted confidence**: {cast(float, payload['persisted_confidence']):.3f}\n"
        f"- **Replayed confidence**: {cast(float, payload['replayed_confidence']):.3f}\n"
        f"- **Delta**: {cast(float, payload['delta']):.4f} "
        f"(tolerance {cast(float, payload['tolerance']):.2f})\n"
        f"- **Determinism gate**: {flag}\n"
    )


@ledger_app.command("list")
def ledger_list(
    ctx: typer.Context,
    workspace_id: str = typer.Option(
        "",
        "--workspace-id",
        help="Narrow the result set to one workspace id; empty == all workspaces.",
    ),
) -> None:
    """List ledger entries in append order, optionally filtered by workspace."""
    from ctrldoc.orch.ledger import VerdictLedger

    state: CliState = ctx.obj
    config = _load_config(state.config_path)
    filter_id = workspace_id.strip() or None
    with _open_ledger_store(config) as store:
        ledger = VerdictLedger(store=store)
        entries = ledger.list_entries(workspace_id=filter_id)
    rows = [_ledger_entry_payload(entry) for entry in entries]
    payload: dict[str, object] = {
        "command": "ledger list",
        "status": "ok",
        "workspace_id": filter_id,
        "entries": rows,
    }
    markdown = _render_ledger_list_markdown(rows=rows)
    _emit_output(state, markdown=markdown, payload=payload)


@ledger_app.command("show")
def ledger_show(
    ctx: typer.Context,
    entry_id: int = typer.Argument(..., help="The ledger entry id to fetch."),
) -> None:
    """Show one ledger entry by id."""
    from ctrldoc.orch.ledger import LedgerEntryNotFoundError, VerdictLedger

    state: CliState = ctx.obj
    config = _load_config(state.config_path)
    with _open_ledger_store(config) as store:
        ledger = VerdictLedger(store=store)
        try:
            entry = ledger.get(entry_id)
        except LedgerEntryNotFoundError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=2) from exc
    entry_payload = _ledger_entry_payload(entry)
    payload: dict[str, object] = {
        "command": "ledger show",
        "status": "ok",
        "entry": entry_payload,
    }
    markdown = _render_ledger_show_markdown(entry=entry_payload)
    _emit_output(state, markdown=markdown, payload=payload)


@ledger_app.command("replay")
def ledger_replay(
    ctx: typer.Context,
    entry_id: int = typer.Argument(..., help="The ledger entry id to replay."),
) -> None:
    """Replay one ledger entry; report the ±0.02 determinism gate verdict.

    The built-in replayer is the identity function over the persisted
    `calibrated_confidence` — sufficient to round-trip the recorded
    value and prove the gate plumbing end-to-end. Real per-op replayers
    will plug in via the L4 tool dispatcher when the MCP server lands.
    """
    from ctrldoc.orch.ledger import LedgerEntryNotFoundError, VerdictLedger

    state: CliState = ctx.obj
    config = _load_config(state.config_path)
    with _open_ledger_store(config) as store:
        ledger = VerdictLedger(store=store)
        try:
            entry = ledger.get(entry_id)
        except LedgerEntryNotFoundError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=2) from exc
        persisted = float(entry.calibrated_confidence)
        outcome = ledger.replay(entry_id, replayer=lambda _inputs: persisted)
    payload: dict[str, object] = {
        "command": "ledger replay",
        "status": "ok",
        "entry_id": outcome.entry_id,
        "operation": outcome.operation,
        "persisted_confidence": outcome.persisted_confidence,
        "replayed_confidence": outcome.replayed_confidence,
        "delta": outcome.delta,
        "tolerance": outcome.tolerance,
        "is_deterministic": outcome.is_deterministic,
    }
    markdown = _render_ledger_replay_markdown(payload=payload)
    _emit_output(state, markdown=markdown, payload=payload)


@mcp_app.command("serve")
def mcp_serve(
    _ctx: typer.Context,
) -> None:
    """Run the MCP server over stdio (JSON-RPC 2.0, line-framed envelopes).

    Reads requests from stdin and writes responses to stdout. The L4
    tool surface (§6.10) is exposed verbatim — no handlers are wired
    here, so every `tools/call` returns an `isError=true` envelope
    until downstream engines plug their handlers into the dispatcher.
    Hosts can still drive `initialize` and `tools/list` to discover
    the catalogue.
    """
    from ctrldoc.mcp.server import serve_stdio

    # Default `MCPServer()` constructs its own `default_dispatcher()`;
    # no per-call state is needed for the protocol surface.
    serve_stdio()


# --- v1 ops passthroughs (§9 CLI surface over §6.10 tool dispatcher) ---


graph_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Graph operations: show / query / traverse over the typed claim graph.",
)
app.add_typer(graph_app, name="graph")

schema_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Induced-schema operations: show one doc's schema; pin a workspace's schema from a doc.",
)
app.add_typer(schema_app, name="schema")


def _build_dispatcher(config: Config) -> object:
    """Fresh `ToolDispatcher` wired with the per-installation handler floor.

    Mirrors `mcp.server.serve_stdio`'s wiring: `optimal_transport` and
    `calibration` always wire; `subsumes` / `get_claim` /
    `lookup_concept` / `traverse` wire if the per-doc indexes under
    `<runs_path>/indexes/` are openable; `coverage` / `compare` /
    `merge` / `list_check` / `entails` / `qa` / `map` stay unwired
    because their NLI / LLM / per-doc-edges deps aren't constructable
    from the bare CLI (those wire up in later op-routing slices).
    """
    from ctrldoc.mcp.handlers import build_store_backed_deps, register_default_handlers
    from ctrldoc.orch.tools import ToolDispatcher

    dispatcher = ToolDispatcher()
    deps = build_store_backed_deps(runs_path=Path(config.paths.runs_path))
    register_default_handlers(dispatcher=dispatcher, deps=deps)
    return dispatcher


def _dispatch_or_stub(
    *,
    state: CliState,
    command: str,
    tool_name: str,
    raw_input: dict[str, object],
    render_ok: object,
) -> None:
    """Run the dispatcher; on `ToolNotImplementedError` emit a structured stub.

    `render_ok(result_model)` returns a `(markdown, payload_dict)` tuple
    for the success path. Failure path emits `status: "not_implemented"`
    with the inputs echoed so an operator can see which deps were
    missing; markdown carries a `# ctrldoc — <command>` heading plus a
    short "not implemented" sentence so the user-facing rendering is
    explicit (and never silently no-ops, per §13 non-negotiable 3).
    """
    from ctrldoc.orch.tools import ToolNotImplementedError

    dispatcher = _build_dispatcher(_load_config(state.config_path))
    try:
        result = dispatcher.dispatch(tool_name=tool_name, raw_input=raw_input)  # type: ignore[attr-defined]
    except ToolNotImplementedError as exc:
        payload: dict[str, object] = {
            "command": command,
            "status": "not_implemented",
            "tool_name": tool_name,
            "inputs": raw_input,
            "reason": str(exc),
        }
        markdown = (
            f"# ctrldoc — {command}\n\n"
            f"_Tool `{tool_name}` is not implemented in this profile._\n\n"
            f"- **Reason**: {exc}\n"
        )
        _emit_output(state, markdown=markdown, payload=payload)
        return
    markdown, ok_payload = render_ok(result)  # type: ignore[operator]
    _emit_output(state, markdown=markdown, payload=ok_payload)


# --- compare ---


@app.command("compare")
def compare(
    ctx: typer.Context,
    workspace_id: str = typer.Argument(..., help="The workspace id whose docs to compare."),
    doc_ids: list[str] = typer.Argument(
        ..., help="Two or more doc ids inside the workspace to compare."
    ),
) -> None:
    """Symmetric N-doc compare: strengths / weaknesses / gaps over the workspace."""
    if len(doc_ids) < 2:
        typer.echo("error: compare requires at least two doc ids", err=True)
        raise typer.Exit(code=2)
    state: CliState = ctx.obj
    raw_input: dict[str, object] = {"workspace_id": workspace_id, "doc_ids": list(doc_ids)}

    def _render_ok(result: object) -> tuple[str, dict[str, object]]:
        report = result.report  # type: ignore[attr-defined]
        payload: dict[str, object] = {
            "command": "compare",
            "status": "ok",
            "tool_name": "compare",
            "workspace_id": report.workspace_id,
            "doc_ids": list(report.doc_ids),
            "rows": [dict(r) for r in report.rows],
        }
        md = (
            f"# ctrldoc — compare\n\n"
            f"- **Workspace**: `{report.workspace_id}`\n"
            f"- **Docs**: {', '.join(f'`{d}`' for d in report.doc_ids)}\n"
            f"- **Verdict rows**: {len(report.rows)}\n"
        )
        return md, payload

    _dispatch_or_stub(
        state=state,
        command="compare",
        tool_name="compare",
        raw_input=raw_input,
        render_ok=_render_ok,
    )


# --- coverage ---


@app.command("coverage")
def coverage(
    ctx: typer.Context,
    workspace_id: str = typer.Option(..., "--workspace", help="Workspace id."),
    target_doc_id: str = typer.Option(..., "--target", help="Target doc id."),
    source_doc_id: str = typer.Option(..., "--source", help="Source doc id."),
) -> None:
    """Per-claim coverage verdicts for `target` against `source` inside the workspace."""
    state: CliState = ctx.obj
    raw_input: dict[str, object] = {
        "workspace_id": workspace_id,
        "target_doc_id": target_doc_id,
        "source_doc_id": source_doc_id,
    }

    def _render_ok(result: object) -> tuple[str, dict[str, object]]:
        report = result.report  # type: ignore[attr-defined]
        payload: dict[str, object] = {
            "command": "coverage",
            "status": "ok",
            "tool_name": "coverage",
            "report": json.loads(report.model_dump_json()),
        }
        md = (
            f"# ctrldoc — coverage\n\n"
            f"- **Target**: `{target_doc_id}`\n"
            f"- **Source**: `{source_doc_id}`\n"
            f"- **Workspace**: `{workspace_id}`\n"
        )
        return md, payload

    _dispatch_or_stub(
        state=state,
        command="coverage",
        tool_name="coverage",
        raw_input=raw_input,
        render_ok=_render_ok,
    )


# --- merge ---


@app.command("merge")
def merge(
    ctx: typer.Context,
    doc_ids: list[str] = typer.Argument(..., help="Doc ids inside the workspace to merge."),
    workspace_id: str = typer.Option(..., "--workspace", help="Workspace id."),
    output_path: Path = typer.Option(..., "--output", help="Where to write the merged doc."),
) -> None:
    """Lossless N-doc synthesis under the workspace; every input claim → one output cluster."""
    state: CliState = ctx.obj
    raw_input: dict[str, object] = {
        "workspace_id": workspace_id,
        "doc_ids": list(doc_ids),
        # `output_path` is a CLI concern (where to write the merged file)
        # — not part of the tool surface schema, surfaced in the
        # envelope so the operator can see where the file would land.
        "output_path": str(output_path),
    }
    # Drop CLI-only field before dispatcher validation (MergeInput rejects extras).
    dispatch_input = {k: v for k, v in raw_input.items() if k != "output_path"}

    def _render_ok(result: object) -> tuple[str, dict[str, object]]:
        merged = result.merged  # type: ignore[attr-defined]
        payload: dict[str, object] = {
            "command": "merge",
            "status": "ok",
            "tool_name": "merge",
            "workspace_id": merged.workspace_id,
            "cluster_ids": list(merged.cluster_ids),
            "representative_claim_ids": list(merged.representative_claim_ids),
            "output_path": str(output_path),
        }
        md = (
            f"# ctrldoc — merge\n\n"
            f"- **Workspace**: `{merged.workspace_id}`\n"
            f"- **Clusters**: {len(merged.cluster_ids)}\n"
            f"- **Output**: `{output_path}`\n"
        )
        return md, payload

    # _dispatch_or_stub stamps `inputs` from `raw_input`; route the
    # full CLI-shaped dict in so the stub envelope shows `output_path`.
    from ctrldoc.orch.tools import ToolNotImplementedError

    dispatcher = _build_dispatcher(_load_config(state.config_path))
    try:
        result = dispatcher.dispatch(tool_name="merge", raw_input=dispatch_input)  # type: ignore[attr-defined]
    except ToolNotImplementedError as exc:
        stub_payload: dict[str, object] = {
            "command": "merge",
            "status": "not_implemented",
            "tool_name": "merge",
            "inputs": raw_input,
            "reason": str(exc),
        }
        markdown = (
            "# ctrldoc — merge\n\n"
            f"_Tool `merge` is not implemented in this profile._\n\n"
            f"- **Reason**: {exc}\n"
        )
        _emit_output(state, markdown=markdown, payload=stub_payload)
        return
    markdown, ok_payload = _render_ok(result)
    _emit_output(state, markdown=markdown, payload=ok_payload)


# --- list-check ---


@app.command("list-check")
def list_check(
    ctx: typer.Context,
    list_path: Path = typer.Argument(..., help="Markdown list file: one bullet per item."),
    doc_id: str = typer.Argument(..., help="Doc id to check the list against."),
) -> None:
    """Per-item verdict of a Markdown bullet list against one doc."""
    _require_input_path(list_path)
    state: CliState = ctx.obj
    items: list[dict[str, str]] = []
    for idx, raw_line in enumerate(list_path.read_text(encoding="utf-8").splitlines()):
        stripped = raw_line.lstrip()
        if not stripped or stripped[0] not in "-*":
            continue
        text = stripped[1:].strip()
        if text:
            items.append({"item_id": f"item-{idx:04d}", "text": text})
    if not items:
        typer.echo("error: no bullet items parsed from list file", err=True)
        raise typer.Exit(code=2)
    raw_input: dict[str, object] = {"items": items, "doc_id": doc_id}

    def _render_ok(result: object) -> tuple[str, dict[str, object]]:
        verdicts = [json.loads(v.model_dump_json()) for v in result.verdicts]  # type: ignore[attr-defined]
        payload: dict[str, object] = {
            "command": "list-check",
            "status": "ok",
            "tool_name": "list_check",
            "doc_id": doc_id,
            "verdicts": verdicts,
        }
        md = f"# ctrldoc — list-check\n\n- **Doc**: `{doc_id}`\n- **Items**: {len(verdicts)}\n"
        return md, payload

    _dispatch_or_stub(
        state=state,
        command="list-check",
        tool_name="list_check",
        raw_input=raw_input,
        render_ok=_render_ok,
    )


# --- graph show / query / traverse ---


@graph_app.command("show")
def graph_show(
    ctx: typer.Context,
    doc_id: str = typer.Argument(..., help="Doc id to render."),
) -> None:
    """Render the doc's typed-edge graph as Mermaid plus node/edge bookkeeping."""
    state: CliState = ctx.obj
    raw_input: dict[str, object] = {"doc_id": doc_id, "filters": {}}

    def _render_ok(result: object) -> tuple[str, dict[str, object]]:
        payload: dict[str, object] = {
            "command": "graph show",
            "status": "ok",
            "tool_name": "map",
            "doc_id": doc_id,
            "mermaid": result.mermaid,  # type: ignore[attr-defined]
            "node_ids": list(result.node_ids),  # type: ignore[attr-defined]
            "edge_count": result.edge_count,  # type: ignore[attr-defined]
        }
        md = (
            f"# ctrldoc — graph show\n\n"
            f"- **Doc**: `{doc_id}`\n"
            f"- **Nodes**: {len(result.node_ids)}\n"  # type: ignore[attr-defined]
            f"- **Edges**: {result.edge_count}\n\n"  # type: ignore[attr-defined]
            "```mermaid\n"
            f"{result.mermaid}\n"  # type: ignore[attr-defined]
            "```\n"
        )
        return md, payload

    _dispatch_or_stub(
        state=state,
        command="graph show",
        tool_name="map",
        raw_input=raw_input,
        render_ok=_render_ok,
    )


@graph_app.command("query")
def graph_query(
    ctx: typer.Context,
    doc_id: str = typer.Argument(..., help="Doc id whose concept lattice to query."),
    concept: str = typer.Option(..., "--concept", help="Canonical concept name to look up."),
) -> None:
    """Look up a canonical concept name in the doc's concept lattice."""
    state: CliState = ctx.obj
    # The `lookup_concept` tool is doc-agnostic at the dispatcher
    # layer — `doc_id` is kept on the envelope for traceability but
    # never routed into the strict Pydantic input.
    raw_input: dict[str, object] = {"name": concept}

    def _render_ok(result: object) -> tuple[str, dict[str, object]]:
        payload: dict[str, object] = {
            "command": "graph query",
            "status": "ok",
            "tool_name": "lookup_concept",
            "doc_id": doc_id,
            "concept": concept,
            "concept_id": result.concept_id,  # type: ignore[attr-defined]
        }
        md = (
            f"# ctrldoc — graph query\n\n"
            f"- **Doc**: `{doc_id}`\n"
            f"- **Concept**: `{concept}`\n"
            f"- **Resolved id**: `{result.concept_id}`\n"  # type: ignore[attr-defined]
        )
        return md, payload

    _dispatch_or_stub(
        state=state,
        command="graph query",
        tool_name="lookup_concept",
        raw_input=raw_input,
        render_ok=_render_ok,
    )


@graph_app.command("traverse")
def graph_traverse(
    ctx: typer.Context,
    node_id: str = typer.Argument(..., help="Starting node id (claim or concept)."),
    edge_type: str = typer.Option(..., "--edge-type", help="Typed edge label to follow."),
    direction: str = typer.Option("forward", "--direction", help="`forward` / `reverse` / `both`."),
    hops: int = typer.Option(1, "--hops", min=1, max=10, help="Number of hops to walk (1..10)."),
) -> None:
    """Walk the typed-edge graph from `node_id` along one edge type."""
    state: CliState = ctx.obj
    raw_input: dict[str, object] = {
        "node_id": node_id,
        "edge_type": edge_type,
        "direction": direction,
        "hops": hops,
    }

    def _render_ok(result: object) -> tuple[str, dict[str, object]]:
        payload: dict[str, object] = {
            "command": "graph traverse",
            "status": "ok",
            "tool_name": "traverse",
            "node_id": node_id,
            "edge_type": edge_type,
            "direction": direction,
            "hops": hops,
            "node_ids": list(result.node_ids),  # type: ignore[attr-defined]
        }
        md = (
            f"# ctrldoc — graph traverse\n\n"
            f"- **From**: `{node_id}`\n"
            f"- **Edge type**: `{edge_type}` ({direction}, hops={hops})\n"
            f"- **Reached**: {len(result.node_ids)} node(s)\n"  # type: ignore[attr-defined]
        )
        return md, payload

    from ctrldoc.orch.tools import ToolNotImplementedError, ToolValidationError

    dispatcher = _build_dispatcher(_load_config(state.config_path))
    try:
        result = dispatcher.dispatch(tool_name="traverse", raw_input=raw_input)  # type: ignore[attr-defined]
    except ToolValidationError as exc:
        typer.echo(f"error: invalid traverse input — {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except ToolNotImplementedError as exc:
        stub_payload: dict[str, object] = {
            "command": "graph traverse",
            "status": "not_implemented",
            "tool_name": "traverse",
            "inputs": raw_input,
            "reason": str(exc),
        }
        markdown = (
            "# ctrldoc — graph traverse\n\n"
            f"_Tool `traverse` is not implemented in this profile._\n\n"
            f"- **Reason**: {exc}\n"
        )
        _emit_output(state, markdown=markdown, payload=stub_payload)
        return
    markdown, ok_payload = _render_ok(result)
    _emit_output(state, markdown=markdown, payload=ok_payload)


# --- schema show / pin ---


def _doc_schema_path(config: Config, doc_id: str) -> Path:
    """Where the induced-schema YAML lives for one doc id."""
    return _per_doc_index_dir(config, doc_id) / f"{doc_id}.schema.yaml"


def _workspace_schema_path(config: Config, workspace_id: str) -> Path:
    """Where the pinned-schema YAML lives for one workspace id."""
    runs_path = Path(config.paths.runs_path)
    return runs_path / "workspaces" / workspace_id / "schema.yaml"


@schema_app.command("show")
def schema_show(
    ctx: typer.Context,
    doc_id: str = typer.Argument(..., help="Doc id whose induced schema to dump."),
) -> None:
    """Print the induced-schema YAML for one doc."""
    state: CliState = ctx.obj
    config = _load_config(state.config_path)
    schema_path = _doc_schema_path(config, doc_id)
    if not schema_path.exists():
        typer.echo(f"error: no induced schema for doc {doc_id!r} at {schema_path}", err=True)
        raise typer.Exit(code=2)
    yaml_text = schema_path.read_text(encoding="utf-8")
    payload: dict[str, object] = {
        "command": "schema show",
        "status": "ok",
        "doc_id": doc_id,
        "schema_path": str(schema_path),
        "yaml": yaml_text,
    }
    md = (
        f"# ctrldoc — schema show\n\n"
        f"- **Doc**: `{doc_id}`\n"
        f"- **Path**: `{schema_path}`\n\n"
        "```yaml\n"
        f"{yaml_text}\n"
        "```\n"
    )
    _emit_output(state, markdown=md, payload=payload)


@schema_app.command("pin")
def schema_pin(
    ctx: typer.Context,
    workspace_id: str = typer.Option(..., "--workspace", help="Workspace id to pin into."),
    from_doc: str = typer.Option(..., "--from", help="Source doc id to copy the schema from."),
) -> None:
    """Pin one doc's induced schema as the workspace's authoritative schema."""
    state: CliState = ctx.obj
    config = _load_config(state.config_path)
    src = _doc_schema_path(config, from_doc)
    if not src.exists():
        typer.echo(f"error: no induced schema for doc {from_doc!r} at {src}", err=True)
        raise typer.Exit(code=2)
    dst = _workspace_schema_path(config, workspace_id)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    payload: dict[str, object] = {
        "command": "schema pin",
        "status": "ok",
        "workspace_id": workspace_id,
        "from_doc": from_doc,
        "source_path": str(src),
        "pinned_path": str(dst),
    }
    md = (
        f"# ctrldoc — schema pin\n\n"
        f"- **Workspace**: `{workspace_id}`\n"
        f"- **From**: `{from_doc}`\n"
        f"- **Pinned at**: `{dst}`\n"
    )
    _emit_output(state, markdown=md, payload=payload)


# --- calibration ---


@app.command("calibration")
def calibration(ctx: typer.Context) -> None:
    """Surface shipped ECE-per-backend (§6.5 release gate)."""
    state: CliState = ctx.obj

    def _render_ok(result: object) -> tuple[str, dict[str, object]]:
        payload: dict[str, object] = {
            "command": "calibration",
            "status": "ok",
            "tool_name": "calibration",
            "ece_per_backend": dict(result.ece_per_backend),  # type: ignore[attr-defined]
            "sample_sizes": dict(result.sample_sizes),  # type: ignore[attr-defined]
        }
        if not result.ece_per_backend:  # type: ignore[attr-defined]
            body = "_(no backends fit yet)_\n"
        else:
            rows = "\n".join(
                f"| `{b}` | {ece:.4f} | {result.sample_sizes.get(b, 0)} |"  # type: ignore[attr-defined]
                for b, ece in sorted(result.ece_per_backend.items())  # type: ignore[attr-defined]
            )
            body = f"| Backend | ECE | Samples |\n|---|---:|---:|\n{rows}\n"
        md = f"# ctrldoc — calibration\n\n{body}"
        return md, payload

    _dispatch_or_stub(
        state=state,
        command="calibration",
        tool_name="calibration",
        raw_input={},
        render_ok=_render_ok,
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
