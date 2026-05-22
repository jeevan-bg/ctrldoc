"""Contract tests for the project configuration loader.

The loader parses `ctrldoc.toml`, validates with Pydantic, and refuses
any field that looks like a secret — secrets belong in env vars only.

SPEC-REF: §4.7 (configuration)
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ctrldoc.config import Config, SecretInConfigError

SPEC_EXAMPLE = """
[models]
planner = "claude-opus-4-7"
judge_tier1 = "qwen2.5:7b-instruct-q4_K_M"
judge_tier2 = "claude-opus-4-7"
verifier_nli = "deberta-v3-large-mnli"
embedder = "bge-m3"

[budgets]
max_cost_usd = 20.0
max_tokens_per_call = 16000
max_wall_clock_min = 30

[concurrency]
anthropic_concurrent = 8
ollama_concurrent = 2

[paths]
index_path = "./ctrldoc.db"
runs_path  = "./runs/"
traces_path = "./traces/"
"""


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "ctrldoc.toml"
    p.write_text(body, encoding="utf-8")
    return p


def test_load_spec_example(tmp_path: Path) -> None:
    cfg = Config.load(_write(tmp_path, SPEC_EXAMPLE))
    assert cfg.models.planner == "claude-opus-4-7"
    assert cfg.models.embedder == "bge-m3"
    assert cfg.budgets.max_cost_usd == 20.0
    assert cfg.budgets.max_tokens_per_call == 16000
    assert cfg.concurrency.anthropic_concurrent == 8
    assert cfg.paths.index_path == Path("./ctrldoc.db")


def test_missing_section_fails(tmp_path: Path) -> None:
    body = SPEC_EXAMPLE.replace("[budgets]", "[budgets_removed]")
    with pytest.raises(ValidationError):
        Config.load(_write(tmp_path, body))


def test_zero_or_negative_budget_rejected(tmp_path: Path) -> None:
    body = SPEC_EXAMPLE.replace("max_cost_usd = 20.0", "max_cost_usd = 0")
    with pytest.raises(ValidationError):
        Config.load(_write(tmp_path, body))


def test_zero_max_tokens_rejected(tmp_path: Path) -> None:
    body = SPEC_EXAMPLE.replace("max_tokens_per_call = 16000", "max_tokens_per_call = 0")
    with pytest.raises(ValidationError):
        Config.load(_write(tmp_path, body))


def test_extra_fields_rejected(tmp_path: Path) -> None:
    body = SPEC_EXAMPLE + '\n[unknown_section]\nfoo = "bar"\n'
    with pytest.raises(ValidationError):
        Config.load(_write(tmp_path, body))


@pytest.mark.parametrize(
    "secret_line",
    [
        'api_key = "sk-xxx"',
        'anthropic_api_key = "sk-xxx"',
        'password = "p"',
        'access_token = "t"',
    ],
)
def test_secrets_in_config_rejected(tmp_path: Path, secret_line: str) -> None:
    body = SPEC_EXAMPLE.replace("[models]", f"[models]\n{secret_line}")
    with pytest.raises(SecretInConfigError):
        Config.load(_write(tmp_path, body))


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Config.load(tmp_path / "does_not_exist.toml")


def test_relative_paths_preserved(tmp_path: Path) -> None:
    cfg = Config.load(_write(tmp_path, SPEC_EXAMPLE))
    # The loader must not resolve paths — callers decide the anchor.
    assert not cfg.paths.runs_path.is_absolute()
    assert not cfg.paths.traces_path.is_absolute()
