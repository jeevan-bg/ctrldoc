"""In-memory reference implementation of `Store`.

Backed by plain dicts. Intended for unit tests and as a behavioural
oracle for the persistent SQLite backend that follows.

SPEC-REF: §10, §13, §4.2
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from ctrldoc.models import Chunk, Entity, Section
from ctrldoc.versioning import IndexVersions


class InMemoryStore:
    """Reference `Store` implementation that lives in process memory."""

    def __init__(self, *, versions: IndexVersions | None = None) -> None:
        self._versions = versions or IndexVersions.current()
        self._chunks: dict[str, Chunk] = {}
        self._sections: dict[str, Section] = {}
        self._entities: dict[str, Entity] = {}

    @property
    def versions(self) -> IndexVersions:
        return self._versions

    def add_chunks(self, chunks: Iterable[Chunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.id] = chunk

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def iter_chunks(self) -> Iterator[Chunk]:
        return iter(self._chunks.values())

    def add_sections(self, sections: Iterable[Section]) -> None:
        for section in sections:
            self._sections[section.id] = section

    def get_section(self, section_id: str) -> Section | None:
        return self._sections.get(section_id)

    def iter_sections(self) -> Iterator[Section]:
        return iter(self._sections.values())

    def add_entities(self, entities: Iterable[Entity]) -> None:
        for entity in entities:
            self._entities[entity.id] = entity

    def get_entity(self, entity_id: str) -> Entity | None:
        return self._entities.get(entity_id)

    def iter_entities(self) -> Iterator[Entity]:
        return iter(self._entities.values())

    # --- entity inverted-index lookups ---

    def chunks_for_entity(self, entity_id: str) -> list[str]:
        entity = self._entities.get(entity_id)
        if entity is None:
            return []
        return list(entity.mention_chunk_ids)

    def entities_for_chunk(self, chunk_id: str) -> list[str]:
        return [
            entity.id for entity in self._entities.values() if chunk_id in entity.mention_chunk_ids
        ]

    def entity_neighbors(self, entity_id: str) -> list[str]:
        source = self._entities.get(entity_id)
        if source is None:
            return []
        source_chunks = set(source.mention_chunk_ids)
        neighbors: set[str] = set()
        for other in self._entities.values():
            if other.id == entity_id:
                continue
            if source_chunks.intersection(other.mention_chunk_ids):
                neighbors.add(other.id)
        return sorted(neighbors)

    # --- destructive ops ---

    def delete_chunks_for_section(self, section_id: str) -> list[str]:
        to_remove = [
            chunk_id for chunk_id, chunk in self._chunks.items() if chunk.section_id == section_id
        ]
        for chunk_id in to_remove:
            del self._chunks[chunk_id]
        return to_remove

    def delete_section(self, section_id: str) -> None:
        self._sections.pop(section_id, None)


__all__ = ["InMemoryStore"]
