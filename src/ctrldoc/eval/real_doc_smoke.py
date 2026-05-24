"""Real-doc shakedown driver.

Drives every entry in the corpus manifest
``tests/fixtures/real_docs/MANIFEST.yaml`` through the v1 substrate
on the heuristic profile (no LLM, no network), collects the per-doc
outcomes, builds a workspace from the spec-vs-impl pair, and writes
a single summary JSON the smoke script and tests can both consume.

The driver is intentionally importable and CLI-invokable. The shell
wrapper ``scripts/real_doc_smoke.sh`` calls it via
``python -m ctrldoc.eval.real_doc_smoke``; tests call it
indirectly through the same shell script.

SPEC-REF: §16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from ctrldoc.canary import CanaryBaseline
from ctrldoc.config import (
    BudgetsConfig,
    ConcurrencyConfig,
    Config,
    ModelsConfig,
    PathsConfig,
)
from ctrldoc.ingest.parser import MarkdownParser
from ctrldoc.ingest.pipeline import ingest_document
from ctrldoc.ops.workspace import WorkspaceManager
from ctrldoc.playbooks.anomaly import (
    AnomalyScanPlaybook,
    EmptySummaryDetector,
    HedgeWordDetector,
)
from ctrldoc.store import Store
from ctrldoc.store.bm25 import TantivyBM25Index
from ctrldoc.store.memory import InMemoryStore
from ctrldoc.store.sqlite import SQLiteStore
from ctrldoc.store.vectors import InMemoryVectorIndex

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CORPUS_DIR = REPO_ROOT / "tests" / "fixtures" / "real_docs"
DEFAULT_MANIFEST = DEFAULT_CORPUS_DIR / "MANIFEST.yaml"

_HEURISTIC_NER_LABELS: list[str] = ["person", "system", "concept"]
_HEURISTIC_EMBED_DIM: int = 32


def _doc_hash_for(text: str) -> str:
    """16-char hex prefix of sha256 over the document text — matches the CLI's hashing."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _hermetic_config(output_root: Path) -> Config:
    """Build a minimal Config that routes every artifact under ``output_root``.

    Models / budgets / concurrency are set to plausible defaults — the
    heuristic profile ignores model identifiers, so the values are
    placeholders chosen to satisfy Config's positivity constraints.
    """
    return Config(
        models=ModelsConfig(
            planner="placeholder",
            judge_tier1="placeholder",
            judge_tier2="placeholder",
            verifier_nli="placeholder",
            embedder="placeholder",
        ),
        budgets=BudgetsConfig(
            max_cost_usd=1.0,
            max_tokens_per_call=16000,
            max_wall_clock_min=30,
        ),
        concurrency=ConcurrencyConfig(anthropic_concurrent=1, ollama_concurrent=1),
        paths=PathsConfig(
            index_path=output_root / "ctrldoc-index",
            runs_path=output_root,
            traces_path=output_root / "traces",
        ),
    )


@dataclass(frozen=True)
class DocEntry:
    """One row of the corpus manifest, normalized."""

    doc_id: str
    type: str
    path: Path
    title: str
    role: str | None = None
    pair_id: str | None = None


@dataclass
class IngestOutcome:
    doc_id: str
    type: str
    sections_parsed: int
    chunks_indexed: int
    entities_indexed: int
    signature_hash: str


@dataclass
class ScanOutcome:
    doc_id: str
    findings_total: int


@dataclass
class WorkspaceOutcome:
    name: str
    doc_count: int
    doc_ids: list[str]


@dataclass
class SmokeSummary:
    ingest_count: int = 0
    scan_count: int = 0
    workspace_doc_count: int = 0
    determinism_ok: bool = True
    exit_code: int = 0
    ingests: list[dict[str, Any]] = field(default_factory=list)
    scans: list[dict[str, Any]] = field(default_factory=list)
    workspace: dict[str, Any] = field(default_factory=dict)


def load_manifest(manifest_path: Path) -> list[DocEntry]:
    """Read the corpus manifest and return normalized entries."""
    with manifest_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or "docs" not in raw:
        msg = f"manifest at {manifest_path} must be a mapping with a `docs:` key"
        raise ValueError(msg)
    corpus_dir = manifest_path.parent
    entries: list[DocEntry] = []
    for row in raw["docs"]:
        entries.append(
            DocEntry(
                doc_id=row["doc_id"],
                type=row["type"],
                path=corpus_dir / row["path"],
                title=row["title"],
                role=row.get("role"),
                pair_id=row.get("pair_id"),
            )
        )
    return entries


def _per_doc_bm25_path(output_root: Path, doc_hash: str) -> Path:
    """Build the per-doc Tantivy directory path under ``<output_root>/indexes``."""
    index_dir = output_root / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)
    return index_dir / f"{doc_hash}.bm25"


def _ingest_one(entry: DocEntry, output_root: Path) -> tuple[IngestOutcome, Store]:
    """Ingest a single doc on the heuristic profile, return the outcome and store.

    Mirrors the CLI's heuristic-profile choice: ``InMemoryStore`` +
    ``InMemoryVectorIndex`` so no LLM, no Ollama, and no sqlite-vec
    extension is required. Tantivy BM25 still persists to disk under
    ``<output_root>/indexes/<doc_hash>.bm25`` so the determinism rerun
    exercises a real file-backed component.
    """
    from ctrldoc.backends import build_bundle

    text = entry.path.read_text(encoding="utf-8")
    doc_hash = _doc_hash_for(text)
    bm25_path = _per_doc_bm25_path(output_root, doc_hash)

    # Hermetic config — every artifact lands under output_root.
    config = _hermetic_config(output_root)
    bundle = build_bundle(config=config, profile="heuristic")

    store: Store = InMemoryStore()
    vector_index = InMemoryVectorIndex(dimension=_HEURISTIC_EMBED_DIM)
    bm25_index = TantivyBM25Index(path=bm25_path)

    try:
        stats = ingest_document(
            source=entry.path,
            parser=MarkdownParser(),
            coref=bundle.coref,
            ner=bundle.ner,
            ner_labels=_HEURISTIC_NER_LABELS,
            embedder=bundle.embedder,
            summarizer=bundle.summarizer,
            store=store,
            vector_index=vector_index,
            bm25_index=bm25_index,
        )
    finally:
        # Flush BM25's writer so reopening the same path on a second pass
        # (e.g. the determinism rerun) does not race the lock file.
        bm25_index.close()

    signature = {
        "chunk_ids": sorted(c.id for c in store.iter_chunks()),
        "section_ids": sorted(s.id for s in store.iter_sections()),
        "entity_ids": sorted(e.id for e in store.iter_entities()),
    }
    baseline = CanaryBaseline.from_signature(
        doc_id=entry.doc_id,
        playbook="ingest",
        signature=signature,
    )
    outcome = IngestOutcome(
        doc_id=entry.doc_id,
        type=entry.type,
        sections_parsed=stats.sections_parsed,
        chunks_indexed=stats.chunks_indexed,
        entities_indexed=stats.entities_indexed,
        signature_hash=baseline.signature_hash,
    )
    return outcome, store


def _scan_one(entry: DocEntry, store: Store) -> ScanOutcome:
    """Run the deterministic anomaly detectors over an ingested doc."""
    playbook = AnomalyScanPlaybook(
        detectors=[HedgeWordDetector(), EmptySummaryDetector()],
    )
    queue = playbook.run(store=store)
    return ScanOutcome(doc_id=entry.doc_id, findings_total=len(queue.findings))


def _check_ingest_determinism(
    entries: list[DocEntry], output_root: Path, first_pass: list[IngestOutcome]
) -> bool:
    """Re-ingest every doc into a sibling directory and compare signatures."""
    rerun_root = output_root / "_determinism_rerun"
    rerun_root.mkdir(parents=True, exist_ok=True)
    rerun_sigs: dict[str, str] = {}
    for entry in entries:
        outcome, _store = _ingest_one(entry, rerun_root)
        # InMemoryStore is garbage-collected at the end of the loop iteration;
        # there is nothing further to close on the heuristic profile.
        rerun_sigs[entry.doc_id] = outcome.signature_hash
    first_sigs = {row.doc_id: row.signature_hash for row in first_pass}
    return first_sigs == rerun_sigs


def _build_pair_workspace(entries: list[DocEntry], output_root: Path) -> WorkspaceOutcome:
    """Build a workspace from the spec-vs-impl pair."""
    pair_entries = [e for e in entries if e.type == "spec_vs_impl"]
    if len(pair_entries) != 2:
        msg = "spec-vs-impl pair must contribute exactly two manifest entries"
        raise ValueError(msg)
    workspace_db = output_root / "workspaces.db"
    workspace_name = "spec-vs-impl-tideline"
    with SQLiteStore(workspace_db) as store:
        manager = WorkspaceManager(store=store)
        manager.create(workspace_name)
        for entry in pair_entries:
            manager.add(workspace_name, entry.doc_id)
        info = manager.info(workspace_name)
    return WorkspaceOutcome(
        name=info.workspace.name,
        doc_count=info.doc_count,
        doc_ids=list(info.workspace.doc_ids),
    )


def run_smoke(
    *,
    manifest_path: Path,
    output_root: Path,
    summary_path: Path,
) -> int:
    """Drive every manifest entry end-to-end and write the summary JSON.

    Returns the intended process exit code (0 on success, non-zero on
    any failure surfaced by the substrate).
    """
    entries = load_manifest(manifest_path)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = SmokeSummary()

    ingest_outcomes: list[IngestOutcome] = []
    scan_outcomes: list[ScanOutcome] = []

    for entry in entries:
        outcome, store = _ingest_one(entry, output_root)
        scan = _scan_one(entry, store)
        ingest_outcomes.append(outcome)
        scan_outcomes.append(scan)
        summary.ingests.append(asdict(outcome))
        summary.scans.append(asdict(scan))

    summary.ingest_count = len(ingest_outcomes)
    summary.scan_count = len(scan_outcomes)

    summary.determinism_ok = _check_ingest_determinism(
        entries=entries, output_root=output_root, first_pass=ingest_outcomes
    )
    if not summary.determinism_ok:
        summary.exit_code = 2

    workspace_outcome = _build_pair_workspace(entries=entries, output_root=output_root)
    summary.workspace = asdict(workspace_outcome)
    summary.workspace_doc_count = workspace_outcome.doc_count

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary.exit_code


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drive the real-doc corpus through the v1 substrate."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to the corpus manifest YAML.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory under which all per-doc artifacts and the workspaces DB will be written.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        required=True,
        help="File path the per-run summary JSON will be written to.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run_smoke(
        manifest_path=args.manifest,
        output_root=args.output_root,
        summary_path=args.summary_path,
    )


if __name__ == "__main__":
    sys.exit(main())
