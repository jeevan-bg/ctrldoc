"""Contract tests for the L1 leaf models: Chunk, Section, Span.

These three carry every byte of indexed text. They are stable contracts
between layers — any change requires a schema_version bump per SPEC §4.0.

SPEC-REF: §4.0 (data model)
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ctrldoc.models import Chunk, Section, Span


def _chunk(**overrides: object) -> Chunk:
    defaults: dict[str, object] = {
        "id": "chunk-0001",
        "section_id": "sec-1",
        "text": "hello world",
        "token_count": 2,
        "char_start": 0,
        "char_end": 11,
        "embedding_id": "emb-0001",
        "metadata": {},
    }
    defaults.update(overrides)
    return Chunk(**defaults)  # type: ignore[arg-type]


def _section(**overrides: object) -> Section:
    defaults: dict[str, object] = {
        "id": "sec-1",
        "parent_id": None,
        "title": "Intro",
        "summary": "Single-sentence summary.",
        "chunk_ids": ["chunk-0001"],
    }
    defaults.update(overrides)
    return Section(**defaults)  # type: ignore[arg-type]


def _span(**overrides: object) -> Span:
    defaults: dict[str, object] = {
        "chunk_id": "chunk-0001",
        "char_start": 0,
        "char_end": 5,
        "text": "hello",
    }
    defaults.update(overrides)
    return Span(**defaults)  # type: ignore[arg-type]


# --- field set ---


def test_chunk_field_set() -> None:
    assert set(Chunk.model_fields) == {
        "id",
        "section_id",
        "text",
        "token_count",
        "char_start",
        "char_end",
        "embedding_id",
        "metadata",
    }


def test_section_field_set() -> None:
    assert set(Section.model_fields) == {
        "id",
        "parent_id",
        "title",
        "summary",
        "chunk_ids",
    }


def test_span_field_set() -> None:
    assert set(Span.model_fields) == {"chunk_id", "char_start", "char_end", "text"}


# --- validation ---


def test_chunk_negative_token_count_rejected() -> None:
    with pytest.raises(ValidationError):
        _chunk(token_count=-1)


def test_chunk_negative_char_start_rejected() -> None:
    with pytest.raises(ValidationError):
        _chunk(char_start=-1)


def test_chunk_end_before_start_rejected() -> None:
    with pytest.raises(ValidationError):
        _chunk(char_start=10, char_end=5)


def test_chunk_end_equal_start_ok_for_empty_text() -> None:
    c = _chunk(text="", token_count=0, char_start=5, char_end=5)
    assert c.text == ""


def test_span_end_before_start_rejected() -> None:
    with pytest.raises(ValidationError):
        _span(char_start=10, char_end=5)


def test_section_parent_id_can_be_none() -> None:
    s = _section(parent_id=None)
    assert s.parent_id is None


def test_section_chunk_ids_can_be_empty() -> None:
    s = _section(chunk_ids=[])
    assert s.chunk_ids == []


# --- strict shape ---


@pytest.mark.parametrize("factory", [_chunk, _section, _span])
def test_models_reject_extra_fields(factory: object) -> None:
    obj = factory()  # type: ignore[operator]
    payload = obj.model_dump()
    payload["bogus"] = "no"
    with pytest.raises(ValidationError):
        type(obj).model_validate(payload)


@pytest.mark.parametrize("factory", [_chunk, _section, _span])
def test_models_are_frozen(factory: object) -> None:
    obj = factory()  # type: ignore[operator]
    with pytest.raises(ValidationError):
        obj.id = "tampered"  # type: ignore[misc]


@pytest.mark.parametrize("factory", [_chunk, _section, _span])
def test_models_round_trip(factory: object) -> None:
    obj = factory()  # type: ignore[operator]
    assert type(obj).model_validate(obj.model_dump()) == obj


# --- metadata defaults ---


def test_chunk_metadata_default_is_empty_dict() -> None:
    c = Chunk(
        id="x",
        section_id="s",
        text="t",
        token_count=1,
        char_start=0,
        char_end=1,
        embedding_id="e",
    )
    assert c.metadata == {}


def test_two_chunk_instances_have_isolated_metadata_defaults() -> None:
    a = Chunk(
        id="x",
        section_id="s",
        text="t",
        token_count=1,
        char_start=0,
        char_end=1,
        embedding_id="e",
    )
    b = Chunk(
        id="y",
        section_id="s",
        text="t",
        token_count=1,
        char_start=0,
        char_end=1,
        embedding_id="e",
    )
    assert a.metadata is not b.metadata
