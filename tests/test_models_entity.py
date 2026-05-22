"""Contract tests for the Entity model and EntityGlossary alias.

The glossary lives in the cacheable prefix `{system_prompt,
doc_skeleton, entity_glossary}` shared across every sub-task in a
session, so this model must be stable and uniqueness must be
verifiable by the storage layer downstream.

SPEC-REF: §4.0 (data model), §3.1 (cacheable prefix), §4.2 (entity index)
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ctrldoc.models import Entity, EntityGlossary, build_entity_glossary


def _entity(**overrides: object) -> Entity:
    defaults: dict[str, object] = {
        "id": "ent-claude",
        "aliases": ["Claude", "the assistant"],
        "type": "system",
        "mention_chunk_ids": ["chunk-0001", "chunk-0007"],
    }
    defaults.update(overrides)
    return Entity(**defaults)  # type: ignore[arg-type]


def test_entity_field_set() -> None:
    assert set(Entity.model_fields) == {
        "id",
        "aliases",
        "type",
        "mention_chunk_ids",
    }


def test_entity_is_frozen() -> None:
    e = _entity()
    with pytest.raises(ValidationError):
        e.id = "tampered"  # type: ignore[misc]


def test_entity_rejects_extra_fields() -> None:
    payload = _entity().model_dump()
    payload["bogus"] = "no"
    with pytest.raises(ValidationError):
        Entity.model_validate(payload)


def test_entity_round_trip() -> None:
    e = _entity()
    assert Entity.model_validate(e.model_dump()) == e


def test_entity_aliases_can_be_empty() -> None:
    e = _entity(aliases=[])
    assert e.aliases == []


def test_entity_mention_chunk_ids_can_be_empty() -> None:
    e = _entity(mention_chunk_ids=[])
    assert e.mention_chunk_ids == []


# --- glossary ---


def test_entity_glossary_alias_is_dict_of_entity() -> None:
    g: EntityGlossary = {
        "ent-claude": _entity(),
        "ent-opus": _entity(id="ent-opus", aliases=["Opus"], type="model"),
    }
    assert len(g) == 2
    assert g["ent-claude"].type == "system"


def test_build_entity_glossary_keys_by_id() -> None:
    a = _entity(id="a", aliases=["A"])
    b = _entity(id="b", aliases=["B"])
    g = build_entity_glossary([a, b])
    assert set(g.keys()) == {"a", "b"}
    assert g["a"] == a


def test_build_entity_glossary_rejects_duplicate_ids() -> None:
    a = _entity(id="dup", aliases=["A"])
    b = _entity(id="dup", aliases=["B"])
    with pytest.raises(ValueError) as info:
        build_entity_glossary([a, b])
    assert "dup" in str(info.value)


def test_build_entity_glossary_empty_input_ok() -> None:
    assert build_entity_glossary([]) == {}
