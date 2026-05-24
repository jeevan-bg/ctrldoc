"""Real-doc shakedown corpus + smoke script.

Validates the six-doc-type corpus committed under
``tests/fixtures/real_docs/`` (spec / legal / academic / educational
/ narrative + a spec-vs-impl pair) and exercises
``scripts/real_doc_smoke.sh`` end-to-end on the heuristic profile
(no LLM, no network) so the v1 substrate stays driveable against
realistically-shaped documents.

SPEC-REF: §16
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "tests" / "fixtures" / "real_docs"
MANIFEST_PATH = CORPUS_DIR / "MANIFEST.yaml"
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "real_doc_smoke.sh"

# Six doc-type axes per the §16 end-state walkthrough. The spec-vs-impl
# pair contributes two entries (spec + impl), so the minimum corpus
# size is seven distinct files.
REQUIRED_DOC_TYPES: frozenset[str] = frozenset(
    {"spec", "legal", "academic", "educational", "narrative", "spec_vs_impl"}
)

# Minimum useful body size — a tiny excerpt does not exercise chunking,
# section boundaries, or any of the L0 invariants the smoke is here to
# defend.
MIN_DOC_BYTES = 400


def _load_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    assert isinstance(loaded, dict), "manifest must be a YAML mapping"
    return loaded


def test_manifest_exists_and_is_well_formed() -> None:
    """The corpus manifest is the oracle for everything else; it must load."""
    assert MANIFEST_PATH.exists(), f"missing corpus manifest at {MANIFEST_PATH}"
    manifest = _load_manifest()
    assert "docs" in manifest, "manifest must declare a `docs:` list"
    docs = manifest["docs"]
    assert (
        isinstance(docs, list) and len(docs) >= 7
    ), "corpus must include at least seven docs to cover all six axes plus the spec-vs-impl pair"


def test_manifest_covers_every_required_doc_type() -> None:
    """Every doc-type axis from §16 must appear at least once."""
    manifest = _load_manifest()
    declared_types = {entry["type"] for entry in manifest["docs"]}
    missing = REQUIRED_DOC_TYPES - declared_types
    assert not missing, f"corpus missing doc types: {sorted(missing)}"


def test_manifest_entries_reference_existing_files() -> None:
    """Every manifest row must point to a file that exists and is non-trivial."""
    manifest = _load_manifest()
    for entry in manifest["docs"]:
        rel_path = entry["path"]
        full = CORPUS_DIR / rel_path
        assert full.exists(), f"manifest references missing file: {rel_path}"
        size = full.stat().st_size
        assert (
            size >= MIN_DOC_BYTES
        ), f"{rel_path} is too small ({size} bytes); real-doc smoke needs nontrivial content"


def test_spec_vs_impl_pair_is_paired() -> None:
    """The spec-vs-impl pair must declare both halves and link them."""
    manifest = _load_manifest()
    pair_entries = [e for e in manifest["docs"] if e["type"] == "spec_vs_impl"]
    assert (
        len(pair_entries) == 2
    ), "the spec-vs-impl axis needs exactly two entries (a spec doc and an impl doc)"
    roles = {e["role"] for e in pair_entries}
    assert roles == {
        "spec",
        "impl",
    }, f"pair entries must declare role=spec or role=impl; got {roles}"
    pair_ids = {e["pair_id"] for e in pair_entries}
    assert len(pair_ids) == 1, "both halves of the spec-vs-impl pair must share a single pair_id"


def test_smoke_script_is_executable() -> None:
    """The smoke script must exist and be marked executable."""
    assert SMOKE_SCRIPT.exists(), f"missing smoke script at {SMOKE_SCRIPT}"
    mode = SMOKE_SCRIPT.stat().st_mode
    assert mode & 0o111, "smoke script must be executable (chmod +x)"


def test_doc_ids_are_unique() -> None:
    """Manifest doc_ids must be unique — they key per-doc indexes and workspace membership."""
    manifest = _load_manifest()
    doc_ids = [entry["doc_id"] for entry in manifest["docs"]]
    assert len(doc_ids) == len(set(doc_ids)), f"duplicate doc_ids in manifest: {doc_ids}"


@pytest.mark.slow
@pytest.mark.family_synthetic_gold
def test_real_doc_smoke_script_runs_green(tmp_path: Path) -> None:
    """End-to-end exercise: ingest every doc, scan every doc, build the pair workspace.

    Runs the smoke script under the heuristic profile so no LLM /
    Ollama / network access is required. Per-run artifacts land in a
    tmp directory so this is hermetic.
    """
    assert SMOKE_SCRIPT.exists(), "smoke script must exist before this test runs"
    if shutil.which("bash") is None:
        pytest.skip("bash not available on this platform")

    output_root = tmp_path / "real_doc_smoke_runs"
    output_root.mkdir()
    summary_path = output_root / "summary.json"

    proc = subprocess.run(
        [
            "bash",
            str(SMOKE_SCRIPT),
            "--output-root",
            str(output_root),
            "--summary-path",
            str(summary_path),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )

    assert (
        proc.returncode == 0
    ), f"real_doc_smoke.sh exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert summary_path.exists(), "smoke script must write the summary JSON file"
    summary: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))

    # Hard contract: one ingest result + one scan result per manifest entry,
    # plus the spec-vs-impl workspace info row.
    manifest = _load_manifest()
    expected_doc_count = len(manifest["docs"])
    assert (
        summary["ingest_count"] == expected_doc_count
    ), f"expected {expected_doc_count} ingest results, got {summary['ingest_count']}"
    assert (
        summary["scan_count"] == expected_doc_count
    ), f"expected {expected_doc_count} scan results, got {summary['scan_count']}"
    assert summary["workspace_doc_count"] == 2, "spec-vs-impl workspace must hold exactly two docs"
    assert summary["exit_code"] == 0
    # Determinism: the same script run on the same corpus must reproduce
    # every signature hash. The script asserts this internally; we
    # surface it here so a regression is caught loudly.
    assert summary["determinism_ok"] is True


@pytest.mark.slow
@pytest.mark.family_synthetic_gold
def test_real_doc_smoke_is_idempotent(tmp_path: Path) -> None:
    """Running the smoke script twice against the same corpus produces matching ingest signatures."""
    if shutil.which("bash") is None:
        pytest.skip("bash not available on this platform")

    first_root = tmp_path / "run1"
    second_root = tmp_path / "run2"
    first_root.mkdir()
    second_root.mkdir()
    first_summary = first_root / "summary.json"
    second_summary = second_root / "summary.json"

    def _run(out_root: Path, summary_path: Path) -> None:
        proc = subprocess.run(
            [
                "bash",
                str(SMOKE_SCRIPT),
                "--output-root",
                str(out_root),
                "--summary-path",
                str(summary_path),
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr

    _run(first_root, first_summary)
    _run(second_root, second_summary)

    first_payload = json.loads(first_summary.read_text(encoding="utf-8"))
    second_payload = json.loads(second_summary.read_text(encoding="utf-8"))

    first_sigs = {row["doc_id"]: row["signature_hash"] for row in first_payload["ingests"]}
    second_sigs = {row["doc_id"]: row["signature_hash"] for row in second_payload["ingests"]}
    assert (
        first_sigs == second_sigs
    ), f"ingest signatures must be deterministic across runs; diff={set(first_sigs.items()) ^ set(second_sigs.items())}"


def test_python_runner_module_present_for_offline_use() -> None:
    """The smoke script delegates to a Python driver module so the same corpus is reachable from tests / CI without bash.

    This guards against a future refactor that removes the driver entry point
    `python -m ctrldoc.eval.real_doc_smoke` without also pruning the smoke
    script's dependency on it.
    """
    # The driver script must be importable.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import ctrldoc.eval.real_doc_smoke as m; print(m.__doc__ is not None)",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"
