"""Provenance record and run-id factory.

Every `PlaybookOutput` must carry a `Provenance` record so any result
is reproducible and auditable: which playbook, which models, which
schema, which index, when. The default tokenizer name is sourced from
`ctrldoc.tokenizer` so a tokenizer swap propagates without manual
synchronisation.

SPEC-REF: §4.0 (Provenance), §4.7 (provenance)
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime
from typing import Final

from pydantic import BaseModel, ConfigDict

from ctrldoc.tokenizer import TOKENIZER_NAME

SCHEMA_VERSION: Final[str] = "0.2.0"


def new_run_id() -> str:
    """Return a sortable, unique run id: `YYYYMMDDTHHMMSSZ-{8 hex}`."""
    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}-{secrets.token_hex(4)}"


def now_iso() -> str:
    """Return an ISO-8601 UTC timestamp at one-second resolution."""
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class Provenance(BaseModel):
    """Reproducibility metadata attached to every playbook output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    timestamp: str
    playbook: str
    playbook_version: str
    schema_version: str
    index_hash: str
    models: dict[str, str]
    tokenizer: str

    @classmethod
    def create(
        cls,
        *,
        playbook: str,
        playbook_version: str,
        index_hash: str,
        models: dict[str, str],
        run_id: str | None = None,
        timestamp: str | None = None,
        tokenizer: str = TOKENIZER_NAME,
        schema_version: str = SCHEMA_VERSION,
    ) -> Provenance:
        return cls(
            run_id=run_id or new_run_id(),
            timestamp=timestamp or now_iso(),
            playbook=playbook,
            playbook_version=playbook_version,
            schema_version=schema_version,
            index_hash=index_hash,
            models=dict(models),
            tokenizer=tokenizer,
        )


__all__ = ["SCHEMA_VERSION", "Provenance", "new_run_id", "now_iso"]
