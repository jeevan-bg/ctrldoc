"""Contract tests for the provenance record and run-id factory.

Every playbook output carries a Provenance for reproducibility and
audit. The record is frozen, extra fields are forbidden, and the
default tokenizer name is sourced from `ctrldoc.tokenizer` so the two
modules cannot drift.

SPEC-REF: §4.0 (Provenance), §4.7 (provenance)
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from ctrldoc.provenance import SCHEMA_VERSION, Provenance, new_run_id, now_iso
from ctrldoc.tokenizer import TOKENIZER_NAME

_RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{8}$")
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _provenance(**overrides: object) -> Provenance:
    defaults: dict[str, object] = {
        "playbook": "qa",
        "playbook_version": "0.1.0",
        "index_hash": "sha256:0" * 8,
        "models": {"planner": "claude-opus-4-7"},
    }
    defaults.update(overrides)
    return Provenance.create(**defaults)  # type: ignore[arg-type]


def test_schema_version_is_set() -> None:
    assert isinstance(SCHEMA_VERSION, str) and SCHEMA_VERSION


def test_new_run_id_matches_format() -> None:
    rid = new_run_id()
    assert _RUN_ID_RE.match(rid), f"unexpected run-id shape: {rid!r}"


def test_new_run_id_is_unique_across_many_calls() -> None:
    ids = {new_run_id() for _ in range(200)}
    assert len(ids) == 200


def test_now_iso_is_utc_seconds() -> None:
    ts = now_iso()
    assert _ISO_RE.match(ts), f"unexpected timestamp shape: {ts!r}"
    # Parseable back to datetime
    parsed = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    assert parsed.tzinfo is UTC


def test_create_defaults_tokenizer_to_canonical_name() -> None:
    p = _provenance()
    assert p.tokenizer == TOKENIZER_NAME


def test_create_defaults_schema_version() -> None:
    p = _provenance()
    assert p.schema_version == SCHEMA_VERSION


def test_create_assigns_run_id_and_timestamp() -> None:
    p = _provenance()
    assert _RUN_ID_RE.match(p.run_id)
    assert _ISO_RE.match(p.timestamp)


def test_explicit_run_id_is_preserved() -> None:
    p = _provenance(run_id="20260522T000000Z-deadbeef")
    assert p.run_id == "20260522T000000Z-deadbeef"


def test_models_dict_is_copied() -> None:
    src = {"planner": "claude-opus-4-7"}
    p = _provenance(models=src)
    src["planner"] = "mutated"
    assert p.models == {"planner": "claude-opus-4-7"}


def test_provenance_is_frozen() -> None:
    p = _provenance()
    with pytest.raises(ValidationError):
        p.run_id = "tampered"  # type: ignore[misc]


def test_provenance_rejects_extra_fields() -> None:
    p = _provenance()
    payload = p.model_dump()
    payload["extra"] = "nope"
    with pytest.raises(ValidationError):
        Provenance.model_validate(payload)


def test_provenance_round_trip() -> None:
    original = _provenance()
    payload = original.model_dump()
    restored = Provenance.model_validate(payload)
    assert restored == original


def test_required_fields_missing_raises() -> None:
    with pytest.raises(ValidationError):
        Provenance.model_validate({"run_id": "x", "timestamp": "y"})
