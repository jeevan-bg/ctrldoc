"""In-memory reference implementation of `Store`.

Backed by plain dicts. Intended for unit tests and as a behavioural
oracle for the persistent SQLite backend that follows.

SPEC-REF: §10, §13, §4.2
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from ctrldoc.models import Chunk, Entity, Section
from ctrldoc.models_v1 import Claim, Concept
from ctrldoc.versioning import IndexVersions


class InMemoryStore:
    """Reference `Store` implementation that lives in process memory."""

    def __init__(self, *, versions: IndexVersions | None = None) -> None:
        self._versions = versions or IndexVersions.current()
        self._chunks: dict[str, Chunk] = {}
        self._sections: dict[str, Section] = {}
        self._entities: dict[str, Entity] = {}
        self._claims: dict[str, Claim] = {}
        self._concepts: dict[str, Concept] = {}

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

    # --- v2 claim CRUD (§6.2, §6.4 universal-tuple persistence) ---

    def append_claim(self, claim: Claim) -> None:
        self._claims[claim.id] = claim

    def get_claim(self, claim_id: str) -> Claim | None:
        return self._claims.get(claim_id)

    def iter_claims(self) -> Iterator[Claim]:
        for claim_id in sorted(self._claims):
            yield self._claims[claim_id]

    def iter_claims_for_doc(self, doc_id: str) -> Iterator[Claim]:
        for claim_id in sorted(self._claims):
            claim = self._claims[claim_id]
            if claim.doc_id == doc_id:
                yield claim

    # --- v2 concept CRUD (§6.7, §6.8) ---

    def add_concepts(self, concepts: Iterable[Concept]) -> None:
        """Insert or replace a batch of `Concept` rows (idempotent by id)."""
        for concept in concepts:
            self._concepts[concept.id] = concept

    def get_concept(self, concept_id: str) -> Concept | None:
        return self._concepts.get(concept_id)

    def iter_concepts(self) -> Iterator[Concept]:
        for concept_id in sorted(self._concepts):
            yield self._concepts[concept_id]

    def concepts_for_workspace_docs(self, doc_ids: Iterable[str]) -> Iterator[Concept]:
        """Yield concepts whose `doc_ids` intersects `doc_ids` (§6.7)."""
        member_docs = set(doc_ids)
        if not member_docs:
            return
        for concept in self.iter_concepts():
            if member_docs.intersection(concept.doc_ids):
                yield concept

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
