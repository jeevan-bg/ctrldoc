# ruff: noqa: B008 — typer's CLI declaration idiom requires calling
# `typer.Argument(...)` / `typer.Option(...)` in parameter defaults.
"""ctrldoc CLI — typer skeleton wiring the six playbooks.

Subcommands (one per use case from §5):

  - ``ingest`` — run the L0 pipeline against a source doc and persist
    the index. End-to-end deterministic; no LLM needed.
  - ``qa`` — UC1 trustworthy QA.
  - ``audit`` — UC2 coverage audit / UC3 quality audit.
  - ``review`` — UC4 analytical review.
  - ``scan`` — UC5 anomaly scan (deterministic detectors today;
    LLM-backed detectors land later).
  - ``map`` — UC6 concept relation map.

The five LLM-backed subcommands are intentionally skeletal in this
slice: they validate their arguments and report which production
wiring is required. The full per-playbook driver wiring lands in
later slices (examples in S-101, README quickstart in S-102).

SPEC-REF: §6, §12
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer

from ctrldoc.canary import CanaryBaseline, save_baseline
from ctrldoc.ingest.coref import IdentityCorefResolver
from ctrldoc.ingest.embedder import HashEmbedder
from ctrldoc.ingest.ner import StubNERTagger
from ctrldoc.ingest.parser import MarkdownParser
from ctrldoc.ingest.pipeline import ingest_document
from ctrldoc.ingest.summarizer import HeuristicSummarizer
from ctrldoc.playbooks.anomaly import (
    AnomalyScanPlaybook,
    EmptySummaryDetector,
    HedgeWordDetector,
)
from ctrldoc.provenance import new_run_id
from ctrldoc.store.bm25 import TantivyBM25Index
from ctrldoc.store.memory import InMemoryStore
from ctrldoc.store.vectors import InMemoryVectorIndex

DEFAULT_EMBEDDING_DIM = 32

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "ctrldoc — local-first document analysis substrate.\n\n"
        "Run `ctrldoc <command> --help` for per-command help."
    ),
)


def _require_input_path(path: Path) -> None:
    if not path.exists():
        typer.echo(f"error: input file {path} does not exist", err=True)
        raise typer.Exit(code=2)
    if not path.is_file():
        typer.echo(f"error: input {path} is not a regular file", err=True)
        raise typer.Exit(code=2)


def _require_anthropic_key() -> bool:
    """Return True if `ANTHROPIC_API_KEY` is set (.env is loaded by the host).

    Emits a structured message to stderr when missing — the message
    does *not* echo the value. The CLI checks presence only.
    """
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _emit_stub(
    *,
    command: str,
    inputs: dict[str, object],
    next_step: str,
) -> None:
    """Print a structured JSON envelope describing the stub outcome."""
    payload = {
        "command": command,
        "status": "stub",
        "inputs": inputs,
        "next_step": next_step,
        "anthropic_key_present": _require_anthropic_key(),
    }
    typer.echo(json.dumps(payload, indent=2))


# --- ingest ---


@app.command()
def ingest(
    input_path: Path = typer.Argument(
        ...,
        exists=False,  # checked manually for a clearer error message
        help="Path to the source document (Markdown today).",
    ),
    output_dir: Path = typer.Option(
        Path("./runs"),
        "--output-dir",
        "-o",
        help="Directory where stats + index baseline will be written.",
    ),
    doc_id: str = typer.Option(
        "doc",
        "--doc-id",
        "-d",
        help="Logical id for this document (used in the run artefacts).",
    ),
    embedding_dim: int = typer.Option(
        DEFAULT_EMBEDDING_DIM,
        "--embedding-dim",
        help="HashEmbedder dimension. 32 is the default test wiring.",
    ),
) -> None:
    """Ingest a document through the deterministic L0 pipeline.

    Writes two artefacts to `output_dir`:

      * ``{doc_id}__ingest_stats.json``  — chunk / section / entity counts.
      * ``{doc_id}__ingest_signature.json`` — pinnable canary signature.
    """
    _require_input_path(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = new_run_id()
    store = InMemoryStore()
    vector_index = InMemoryVectorIndex(dimension=embedding_dim)
    bm25_index = TantivyBM25Index(path=output_dir / f"{doc_id}__bm25")

    stats = ingest_document(
        source=input_path,
        parser=MarkdownParser(),
        coref=IdentityCorefResolver(),
        ner=StubNERTagger({}),
        ner_labels=["person", "system"],
        embedder=HashEmbedder(dimension=embedding_dim),
        summarizer=HeuristicSummarizer(),
        store=store,
        vector_index=vector_index,
        bm25_index=bm25_index,
    )

    stats_path = output_dir / f"{doc_id}__ingest_stats.json"
    # Section count derives from the store (IngestStats only tracks
    # the indexed primitives; the assembled section tree lives on
    # the store itself).
    sections_indexed = sum(1 for _ in store.iter_sections())
    stats_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "doc_id": doc_id,
                "input_path": str(input_path),
                "chunks_indexed": stats.chunks_indexed,
                "sections_indexed": sections_indexed,
                "entities_indexed": stats.entities_indexed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    signature = {
        "chunk_ids": sorted(c.id for c in store.iter_chunks()),
        "section_ids": sorted(s.id for s in store.iter_sections()),
        "entity_ids": sorted(e.id for e in store.iter_entities()),
    }
    baseline = CanaryBaseline.from_signature(
        doc_id=doc_id,
        playbook="ingest",
        signature=signature,
    )
    sig_path = output_dir / f"{doc_id}__ingest_signature.json"
    save_baseline(sig_path, baseline)

    typer.echo(
        json.dumps(
            {
                "command": "ingest",
                "status": "ok",
                "run_id": run_id,
                "stats_path": str(stats_path),
                "signature_path": str(sig_path),
                "chunks_indexed": stats.chunks_indexed,
            },
            indent=2,
        )
    )


# --- LLM-backed stubs ---


@app.command()
def qa(
    query: str = typer.Argument(..., help="The question to ask the indexed corpus."),
    index_path: Path = typer.Option(
        Path("./runs"),
        "--index",
        "-i",
        help="Directory holding the ingested index (output of `ctrldoc ingest`).",
    ),
) -> None:
    """UC1 trustworthy QA over an indexed document (skeleton).

    The production wiring for a generator + verifier pair is not yet
    bound to the CLI; this subcommand validates its arguments and
    emits a structured "next-step" message.
    """
    if not query.strip():
        typer.echo("error: query must not be blank", err=True)
        raise typer.Exit(code=2)
    _emit_stub(
        command="qa",
        inputs={"query": query, "index_path": str(index_path)},
        next_step=(
            "Wire QAPlaybook(prefix, retriever, task_runner, decomposer, verifier) "
            "with production deps and call playbook.run(query)."
        ),
    )


@app.command()
def audit(
    checklist_path: Path = typer.Argument(..., help="Checklist file (JSONL of items)."),
    index_path: Path = typer.Option(
        Path("./runs"),
        "--index",
        "-i",
        help="Directory holding the ingested target-doc index.",
    ),
    kind: str = typer.Option(
        "coverage",
        "--kind",
        "-k",
        help="`coverage` (UC2) or `quality` (UC3, generates criteria first).",
    ),
) -> None:
    """UC2 coverage audit / UC3 quality audit (skeleton)."""
    if kind not in {"coverage", "quality"}:
        typer.echo(f"error: --kind must be 'coverage' or 'quality', got {kind!r}", err=True)
        raise typer.Exit(code=2)
    _require_input_path(checklist_path)
    _emit_stub(
        command="audit",
        inputs={
            "checklist_path": str(checklist_path),
            "index_path": str(index_path),
            "kind": kind,
        },
        next_step=(
            "Wire CoverageAuditPlaybook / QualityAuditPlaybook with a production "
            "BatchedTaskRunner + retriever and call playbook.run(items)."
        ),
    )


@app.command()
def review(
    doc_type: str = typer.Argument(..., help="Document type, e.g. `Aurora L0 kernel spec`."),
    index_path: Path = typer.Option(
        Path("./runs"),
        "--index",
        "-i",
        help="Directory holding the ingested target-doc index.",
    ),
) -> None:
    """UC4 analytical review (skeleton)."""
    if not doc_type.strip():
        typer.echo("error: doc_type must not be blank", err=True)
        raise typer.Exit(code=2)
    _emit_stub(
        command="review",
        inputs={"doc_type": doc_type, "index_path": str(index_path)},
        next_step=(
            "Wire AnalyticalReviewPlaybook with a production lens_generator + "
            "sweeper + synthesis_runner and call playbook.run(doc_type)."
        ),
    )


@app.command()
def scan(
    index_path: Path = typer.Option(
        Path("./runs"),
        "--index",
        "-i",
        help="Directory holding the ingested target-doc index.",
    ),
) -> None:
    """UC5 anomaly scan — deterministic detectors over the indexed store.

    Today's CLI wires the two deterministic reference detectors
    (`hedge_word`, `empty_summary`); the four LLM-backed detectors
    from §5.5 (asymmetry, justification gap, undefined terms,
    boundary silence) plug in once their backends land.
    """
    # The skeleton scans an empty in-memory store so the CLI path
    # exercises the AnomalyScanPlaybook composition end-to-end. A
    # follow-up slice will load the persisted store from `index_path`.
    store = InMemoryStore()
    playbook = AnomalyScanPlaybook(
        detectors=[HedgeWordDetector(), EmptySummaryDetector()],
    )
    queue = playbook.run(store=store)
    typer.echo(
        json.dumps(
            {
                "command": "scan",
                "status": "ok",
                "index_path": str(index_path),
                "findings": [
                    {
                        "detector": finding.ctrldoc,
                        "severity": finding.severity,
                        "claim": finding.claim,
                        "chunk_id": finding.location.chunk_id,
                    }
                    for finding in queue.findings
                ],
            },
            indent=2,
        )
    )


@app.command(name="map")
def map_(
    concepts: list[str] = typer.Argument(
        None,
        help="Optional explicit concept ids to map; auto-extracted when omitted.",
    ),
    index_path: Path = typer.Option(
        Path("./runs"),
        "--index",
        "-i",
        help="Directory holding the ingested target-doc index.",
    ),
) -> None:
    """UC6 concept relation map (skeleton)."""
    _emit_stub(
        command="map",
        inputs={"concepts": list(concepts or []), "index_path": str(index_path)},
        next_step=(
            "Wire RelationMapPlaybook with a production extractor + retriever + "
            "classifier and call playbook.run()."
        ),
    )


def main() -> None:
    """Entry point for ``python -m ctrldoc``."""
    app()


if __name__ == "__main__":
    main()


__all__ = ["app", "main"]
