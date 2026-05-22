"""Validate `.pre-commit-config.yaml` wires the required quality gates.

The pre-commit config must enforce — at minimum — the same gates CI runs:
ruff lint, ruff format, mypy, the repository content lint
(`scripts/leak_scan.sh`), and a credential / secret scan.

SPEC-REF: §4.7 (cross-cutting / configuration & secrets)
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / ".pre-commit-config.yaml"


@pytest.fixture(scope="module")
def pre_commit_config() -> dict:
    assert CONFIG_PATH.exists(), ".pre-commit-config.yaml is missing at repo root"
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    assert isinstance(data, dict), ".pre-commit-config.yaml must be a mapping"
    return data


def _hook_ids(config: dict) -> set[str]:
    ids: set[str] = set()
    for repo in config.get("repos", []):
        for hook in repo.get("hooks", []):
            hook_id = hook.get("id")
            if isinstance(hook_id, str):
                ids.add(hook_id)
    return ids


def test_config_top_level_shape(pre_commit_config: dict) -> None:
    assert "repos" in pre_commit_config, "missing `repos:` key"
    assert isinstance(pre_commit_config["repos"], list)
    assert pre_commit_config["repos"], "`repos:` must list at least one repo"


def test_ruff_lint_and_format_hooks_present(pre_commit_config: dict) -> None:
    ids = _hook_ids(pre_commit_config)
    assert "ruff" in ids, "ruff lint hook missing from pre-commit config"
    assert "ruff-format" in ids, "ruff-format hook missing from pre-commit config"


def test_mypy_hook_present(pre_commit_config: dict) -> None:
    ids = _hook_ids(pre_commit_config)
    assert "mypy" in ids, "mypy hook missing from pre-commit config"


def test_leak_scan_hook_present(pre_commit_config: dict) -> None:
    ids = _hook_ids(pre_commit_config)
    assert "leak-scan" in ids, (
        "leak-scan hook must wire `scripts/leak_scan.sh` so content leaks are caught before commit"
    )


def test_secret_scan_hook_present(pre_commit_config: dict) -> None:
    ids = _hook_ids(pre_commit_config)
    # Either of these is acceptable: a dedicated secret scanner, or the
    # pre-commit-hooks private-key detector.
    candidates = {"detect-secrets", "detect-private-key", "gitleaks"}
    assert ids & candidates, (
        "expected a secret/credential scan hook "
        f"(one of {sorted(candidates)}); found ids={sorted(ids)}"
    )


def test_uses_pinned_versions(pre_commit_config: dict) -> None:
    for repo in pre_commit_config["repos"]:
        url = repo.get("repo")
        if url == "local":
            continue
        rev = repo.get("rev")
        assert isinstance(rev, str) and rev, (
            f"repo {url!r} must pin a `rev:` (no floating versions)"
        )
