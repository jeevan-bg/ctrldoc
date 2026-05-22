"""Named-entity recognition + canonicalisation.

`NERTagger` is the protocol every NER backend satisfies. `canonicalize`
folds raw mentions into `Entity` records, deduplicating by canonical
(lowercase-text, label) so repeated surface forms collapse to one
entity with the distinct forms collected as aliases. The LLM-driven
canonicalisation pass described in SPEC §4.1 step 3 layers on top of
this deterministic dedup later.

A separate `GLiNERTagger` (in `ner_gliner.py`) provides the real
backend; we keep the heavy ML import out of this module so callers
that only need the data shapes don't pull torch.

SPEC-REF: §4.1 (ingest step 3 — NER + canonicalisation)
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, NonNegativeInt, model_validator

from ctrldoc.models import Entity, UnitInterval


class EntityMention(BaseModel):
    """One occurrence of an entity in the source text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    label: str
    start: NonNegativeInt
    end: NonNegativeInt
    score: UnitInterval

    @model_validator(mode="after")
    def _check_range(self) -> EntityMention:
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) must be >= start ({self.start})")
        return self


@runtime_checkable
class NERTagger(Protocol):
    """Text + label list → spans tagged with one of those labels."""

    def tag(self, text: str, *, labels: list[str]) -> list[EntityMention]: ...


class StubNERTagger:
    """Test-only NER tagger backed by a fixed text → mentions map.

    Useful in unit tests of canonicalisation, ingest end-to-end flows,
    and any layer that needs a predictable NER stand-in.
    """

    def __init__(self, by_text: dict[str, list[EntityMention]]) -> None:
        self._by_text = by_text

    def tag(self, text: str, *, labels: list[str]) -> list[EntityMention]:
        return list(self._by_text.get(text, []))


def canonicalize(
    mentions: Iterable[EntityMention],
    *,
    chunk_id: str,
) -> list[Entity]:
    """Fold mentions into `Entity` records.

    Two mentions collapse into the same entity when they share the
    same lowercase text *and* label. The entity's `aliases` collect
    every distinct surface form seen; `mention_chunk_ids` records
    `chunk_id` once. Callers ingest the per-chunk results and then
    merge across chunks.
    """
    groups: dict[tuple[str, str], dict[str, object]] = {}
    for mention in mentions:
        key = (mention.text.lower(), mention.label)
        bucket = groups.setdefault(
            key,
            {"label": mention.label, "aliases": [], "seen": set()},
        )
        seen = bucket["seen"]
        assert isinstance(seen, set)
        if mention.text not in seen:
            seen.add(mention.text)
            aliases = bucket["aliases"]
            assert isinstance(aliases, list)
            aliases.append(mention.text)

    entities: list[Entity] = []
    for (lower_text, label), bucket in groups.items():
        aliases_value = bucket["aliases"]
        assert isinstance(aliases_value, list)
        entity = Entity(
            id=f"ent/{label}/{_slugify(lower_text)}",
            aliases=list(aliases_value),
            type=label,
            mention_chunk_ids=[chunk_id],
        )
        entities.append(entity)

    entities.sort(key=lambda e: e.id)
    return entities


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return slug or "entity"


__all__ = [
    "EntityMention",
    "NERTagger",
    "StubNERTagger",
    "canonicalize",
]
