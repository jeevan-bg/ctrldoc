"""Playbooks package removed; all surfaces live under `ctrldoc.ops`.

The v1 universal substrate replaces the v0.3 playbook layer (§6 frames
the substrate as "kills the playbook layer"). The contract enforced here:

* No code path inside `src/ctrldoc/`, `tests/`, `scripts/`, or
  `examples/` may import `ctrldoc.playbooks` (the package is gone).
* The classes the CLI / eval / examples relied on are re-exported from
  the new module homes under `ctrldoc.ops.*`. Concretely:

  - `playbooks.anomaly`   → `ops.scan`   (matches `ctrldoc scan` CLI)
  - `playbooks.qa`        → `ops.qa`     (matches `ctrldoc qa` CLI)
  - `playbooks.review`    → `ops.review` (matches `ctrldoc review` CLI)
  - `playbooks.relations` → `ops.map`    (matches `ctrldoc map` CLI)
  - `playbooks.coverage`  → `ops.audit`  (matches `ctrldoc audit` CLI;
                                          avoids collision with the
                                          v1 OT-backed `ops.coverage`)
  - `playbooks.quality`   → `ops.quality` (no CLI surface; consumed by
                                           the eval / tests layer only)

SPEC-REF: §6 (universal substrate kills the playbook layer)
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.family_determinism


REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 1. Package is gone from disk + un-importable
# ---------------------------------------------------------------------------


def test_playbooks_package_directory_does_not_exist() -> None:
    pkg_dir = REPO_ROOT / "src" / "ctrldoc" / "playbooks"
    assert not pkg_dir.exists(), (
        f"`{pkg_dir}` still exists; S-146 must remove the v0.3 playbook "
        "package now that the v1 ops substrate carries every surface."
    )


def test_playbooks_module_is_not_importable() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("ctrldoc.playbooks")


# ---------------------------------------------------------------------------
# 2. Every symbol the v0.3 playbook layer exposed is importable from `ops`
# ---------------------------------------------------------------------------


_EXPECTED_OP_SYMBOLS: dict[str, tuple[str, ...]] = {
    "ctrldoc.ops.scan": (
        "AnomalyQueue",
        "AnomalyScanPlaybook",
        "Detector",
        "EmptySummaryDetector",
        "HedgeWordDetector",
    ),
    "ctrldoc.ops.qa": (
        "AnswerReport",
        "QAPlaybook",
        "QARetriever",
    ),
    "ctrldoc.ops.review": (
        "AnalyticalReviewPlaybook",
        "HeuristicLensGenerator",
        "Lens",
        "LensGenerator",
        "LensSweeper",
        "ReviewNarrative",
        "ReviewReport",
    ),
    "ctrldoc.ops.map": (
        "Concept",
        "ConceptExtractor",
        "CoOccurrenceRetriever",
        "RelationClassification",
        "RelationClassifier",
        "RelationGraph",
        "RelationMapPlaybook",
    ),
    "ctrldoc.ops.audit": (
        "ChecklistItem",
        "CoverageAuditPlaybook",
        "CoverageReport",
        "CoverageRetriever",
    ),
    "ctrldoc.ops.quality": (
        "CriteriaGenerator",
        "HeuristicCriteriaGenerator",
        "QualityAuditPlaybook",
        "QualityReport",
    ),
}


@pytest.mark.parametrize(
    ("module_path", "symbol"),
    [
        (module_path, symbol)
        for module_path, symbols in _EXPECTED_OP_SYMBOLS.items()
        for symbol in symbols
    ],
)
def test_v0_3_playbook_symbols_now_export_from_ops(module_path: str, symbol: str) -> None:
    """The old playbook surface keeps working from its new `ops.*` home."""
    module = importlib.import_module(module_path)
    assert hasattr(module, symbol), (
        f"`{module_path}` is missing `{symbol}`; the rename of "
        f"playbooks → ops must preserve every public symbol the CLI / "
        f"eval / examples relied on."
    )


# ---------------------------------------------------------------------------
# 3. No source file under the repo's tracked source/test/example trees
#    still imports `ctrldoc.playbooks` (the package is gone — any leftover
#    import would be a ModuleNotFoundError at runtime).
# ---------------------------------------------------------------------------


_SCAN_ROOTS = (
    REPO_ROOT / "src",
    REPO_ROOT / "tests",
    REPO_ROOT / "examples",
    REPO_ROOT / "scripts",
)


def _module_path_imports_playbooks(path: Path) -> bool:
    """Return True iff `path` ast-imports `ctrldoc.playbooks[.*]`."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "ctrldoc.playbooks" or module.startswith("ctrldoc.playbooks."):
                return True
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ctrldoc.playbooks" or alias.name.startswith("ctrldoc.playbooks."):
                    return True
    return False


def test_no_source_file_imports_ctrldoc_playbooks() -> None:
    offenders: list[Path] = []
    for root in _SCAN_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            # The test file itself is allowed to mention the string in
            # comments / docstrings; we ast-walk imports so it never
            # self-flags. But skip the file outright as belt-and-braces.
            if path.resolve() == Path(__file__).resolve():
                continue
            if _module_path_imports_playbooks(path):
                offenders.append(path)
    assert not offenders, (
        "Found leftover `from ctrldoc.playbooks...` imports after S-146; "
        "rewrite each to the equivalent `ctrldoc.ops.*` path:\n"
        + "\n".join(f"  - {p.relative_to(REPO_ROOT)}" for p in offenders)
    )


# ---------------------------------------------------------------------------
# 4. Grep belt-and-braces — catches dynamic `importlib.import_module(
#    'ctrldoc.playbooks.qa')` and similar string-form references the AST
#    walker would miss. Skips the test file, .pyc caches, and binary files.
# ---------------------------------------------------------------------------


def test_no_repo_text_references_ctrldoc_playbooks_module_path() -> None:
    """Grep ensures no string-form `ctrldoc.playbooks` survives anywhere."""
    self_path = str(Path(__file__).resolve().relative_to(REPO_ROOT))
    cmd = [
        "git",
        "grep",
        "-l",
        "--",
        "ctrldoc.playbooks",
        "src/",
        "tests/",
        "examples/",
        "scripts/",
        "docs/",
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    # `git grep` exits 1 when nothing matches; either way we read stdout.
    matches = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and line.strip() != self_path
    ]
    assert not matches, (
        "String-form `ctrldoc.playbooks` references survive after S-146:\n"
        + "\n".join(f"  - {m}" for m in matches)
    )


# ---------------------------------------------------------------------------
# 5. The new `ctrldoc.ops` package surfaces both the v1 substrate (workspace,
#    cross_doc_edges, coverage/compare/merge/transport) AND the relocated
#    v0.3 surface (scan/qa/review/map/audit/quality). A single pkgutil walk
#    confirms both sets are present.
# ---------------------------------------------------------------------------


def test_ops_package_contains_relocated_and_v1_modules() -> None:
    ops_pkg = importlib.import_module("ctrldoc.ops")
    discovered = {m.name for m in pkgutil.iter_modules(ops_pkg.__path__)}
    required = {
        # v1 substrate (already shipped):
        "workspace",
        "cross_doc_edges",
        "coverage",
        "compare",
        "merge",
        "transport",
        # v0.3 surface relocated by S-146:
        "scan",
        "qa",
        "review",
        "map",
        "audit",
        "quality",
    }
    missing = required - discovered
    assert not missing, (
        f"`ctrldoc.ops` is missing modules {sorted(missing)} after S-146; "
        "the rename must land every v0.3 playbook under its CLI-aligned name."
    )
