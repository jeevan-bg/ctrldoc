"""Contract tests for index versioning and content-hash helpers.

The three version dimensions (`schema_version`, `index_version`,
`embedding_model_version`) live on every persisted index; opening
an index whose versions disagree with the runtime must raise
`IndexVersionMismatchError` with no silent migration. Content hashes
are sha256 hex and deterministic so a corrupted chunk is detectable.

SPEC-REF: §4.7 (versioning), §4.7 (index integrity)
"""

from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from ctrldoc.models import Chunk
from ctrldoc.provenance import SCHEMA_VERSION
from ctrldoc.versioning import (
    EMBEDDING_MODEL_VERSION,
    INDEX_VERSION,
    IndexVersionMismatchError,
    IndexVersions,
    content_hash,
    hash_chunk,
)

_SHA256_HEX = re.compile(r"^sha256:[0-9a-f]{64}$")


def _chunk(**overrides: object) -> Chunk:
    defaults: dict[str, object] = {
        "id": "c1",
        "section_id": "sec-1",
        "text": "hello world",
        "token_count": 2,
        "char_start": 0,
        "char_end": 11,
        "embedding_id": "e1",
    }
    defaults.update(overrides)
    return Chunk(**defaults)  # type: ignore[arg-type]


# --- content_hash ---


def test_content_hash_format() -> None:
    h = content_hash("any text")
    assert _SHA256_HEX.match(h), f"unexpected hash shape: {h!r}"


def test_content_hash_empty_string_is_stable() -> None:
    assert content_hash("") == content_hash("")
    assert _SHA256_HEX.match(content_hash(""))


def test_content_hash_deterministic() -> None:
    assert content_hash("abc") == content_hash("abc")


def test_content_hash_changes_with_input() -> None:
    assert content_hash("abc") != content_hash("abd")


def test_content_hash_handles_unicode() -> None:
    assert content_hash("héllo 漢字 🚀") == content_hash("héllo 漢字 🚀")


# --- hash_chunk ---


def test_hash_chunk_format_and_determinism() -> None:
    a = _chunk()
    b = _chunk()
    h = hash_chunk(a)
    assert _SHA256_HEX.match(h)
    assert hash_chunk(a) == hash_chunk(b)


def test_hash_chunk_changes_with_text() -> None:
    assert hash_chunk(_chunk(text="hello")) != hash_chunk(_chunk(text="goodbye"))


def test_hash_chunk_changes_with_section() -> None:
    assert hash_chunk(_chunk(section_id="A")) != hash_chunk(_chunk(section_id="B"))


def test_hash_chunk_changes_with_char_range() -> None:
    assert hash_chunk(_chunk(char_start=0, char_end=5)) != hash_chunk(
        _chunk(char_start=2, char_end=7)
    )


def test_hash_chunk_independent_of_id_and_embedding_id() -> None:
    # The identity-defining fields are content, not the bookkeeping ids.
    base = _chunk()
    renamed = _chunk(id="renamed", embedding_id="emb-renamed")
    assert hash_chunk(base) == hash_chunk(renamed)


# --- IndexVersions ---


def test_index_versions_current_uses_module_constants() -> None:
    v = IndexVersions.current()
    assert v.schema_version == SCHEMA_VERSION
    assert v.index_version == INDEX_VERSION
    assert v.embedding_model_version == EMBEDDING_MODEL_VERSION


def test_index_versions_round_trip() -> None:
    v = IndexVersions.current()
    assert IndexVersions.model_validate(v.model_dump()) == v


def test_index_versions_is_frozen() -> None:
    v = IndexVersions.current()
    with pytest.raises(ValidationError):
        v.schema_version = "tampered"  # type: ignore[misc]


def test_index_versions_rejects_extra_fields() -> None:
    payload = IndexVersions.current().model_dump()
    payload["bogus"] = "no"
    with pytest.raises(ValidationError):
        IndexVersions.model_validate(payload)


def test_assert_compatible_no_op_when_equal() -> None:
    v = IndexVersions.current()
    v.assert_compatible_with(v)  # must not raise


@pytest.mark.parametrize(
    "field",
    ["schema_version", "index_version", "embedding_model_version"],
)
def test_assert_compatible_raises_on_mismatch(field: str) -> None:
    runtime = IndexVersions.current()
    stored = runtime.model_copy(update={field: "stale-value"})
    with pytest.raises(IndexVersionMismatchError) as info:
        stored.assert_compatible_with(runtime)
    assert "re-ingest required" in str(info.value).lower()
    assert field in str(info.value) or field.split("_")[0] in str(info.value)


def test_assert_compatible_reports_all_drifts_at_once() -> None:
    runtime = IndexVersions.current()
    stored = runtime.model_copy(
        update={
            "schema_version": "x",
            "index_version": "y",
            "embedding_model_version": "z",
        }
    )
    with pytest.raises(IndexVersionMismatchError) as info:
        stored.assert_compatible_with(runtime)
    message = str(info.value)
    for token in ("schema", "index", "embedding"):
        assert token in message
