"""`ctrldoc.toml` loader.

The project configuration lives in a single TOML file. This module
parses it, validates each section with Pydantic, and refuses any key
that looks like a secret — secrets are read from environment variables.

SPEC-REF: §4.7 (configuration)
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt

_SECRET_SEGMENTS: frozenset[str] = frozenset(
    {
        "key",
        "apikey",
        "secret",
        "secrets",
        "password",
        "passwd",
        "token",
        "credential",
        "credentials",
    }
)


class SecretInConfigError(ValueError):
    """Raised when `ctrldoc.toml` contains a key that looks like a secret."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelsConfig(_Strict):
    planner: str
    judge_tier1: str
    judge_tier2: str
    verifier_nli: str
    embedder: str


class BudgetsConfig(_Strict):
    max_cost_usd: PositiveFloat
    max_tokens_per_call: PositiveInt
    max_wall_clock_min: PositiveInt


class ConcurrencyConfig(_Strict):
    anthropic_concurrent: PositiveInt
    ollama_concurrent: PositiveInt


class PathsConfig(_Strict):
    index_path: Path
    runs_path: Path
    traces_path: Path


class Config(_Strict):
    models: ModelsConfig
    budgets: BudgetsConfig
    concurrency: ConcurrencyConfig
    paths: PathsConfig

    @classmethod
    def load(cls, path: str | Path) -> Config:
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(target)
        with target.open("rb") as fh:
            data = tomllib.load(fh)
        _reject_secret_keys(data)
        return cls.model_validate(data)


def _reject_secret_keys(data: dict[str, Any], prefix: str = "") -> None:
    for key, value in data.items():
        path = f"{prefix}{key}"
        segments = set(key.lower().replace("-", "_").split("_"))
        if segments & _SECRET_SEGMENTS:
            raise SecretInConfigError(
                f"secrets must not appear in ctrldoc.toml; offending key: {path!r}"
            )
        if isinstance(value, dict):
            _reject_secret_keys(value, prefix=f"{path}.")


__all__ = [
    "BudgetsConfig",
    "ConcurrencyConfig",
    "Config",
    "ModelsConfig",
    "PathsConfig",
    "SecretInConfigError",
]
