"""Storage abstraction.

The `Store` protocol is the seam between L1 storage and the rest of
the stack. Every concrete backend (the SQLite default, or a future
Qdrant / FalkorDB / cloud replacement) satisfies the same interface;
switching is a config flip, not a rewrite.

SPEC-REF: §10 (storage abstraction), §13 (non-negotiable #5), §4.2
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Protocol, runtime_checkable

from ctrldoc.models import Chunk, Entity, Section
from ctrldoc.models_v1 import Claim, Concept
from ctrldoc.versioning import IndexVersions


@runtime_checkable
class Store(Protocol):
    """L1 storage contract — structural CRUD over chunks/sections/entities.

    Dense-vector, BM25, and entity-inverted-index queries are added by
    subsequent slices (S-022..S-024); this protocol covers the
    foundation every backend must provide.

    Implementations must be **idempotent by id**: `add_*` with an id
    that already exists overwrites the previous record (last-write-
    wins). This matches `§4.1` "Idempotent. Cacheable. Runs once per
    document." so a re-ingest is safe.
    """

    @property
    def versions(self) -> IndexVersions: ...

    def add_chunks(self, chunks: Iterable[Chunk]) -> None: ...
    def get_chunk(self, chunk_id: str) -> Chunk | None: ...
    def iter_chunks(self) -> Iterator[Chunk]: ...

    def add_sections(self, sections: Iterable[Section]) -> None: ...
    def get_section(self, section_id: str) -> Section | None: ...
    def iter_sections(self) -> Iterator[Section]: ...

    def add_entities(self, entities: Iterable[Entity]) -> None: ...
    def get_entity(self, entity_id: str) -> Entity | None: ...
    def iter_entities(self) -> Iterator[Entity]: ...

    # --- entity inverted-index lookups (§4.2 entity index, §4.3 `neighbors`) ---

    def chunks_for_entity(self, entity_id: str) -> list[str]: ...
    def entities_for_chunk(self, chunk_id: str) -> list[str]: ...
    def entity_neighbors(self, entity_id: str) -> list[str]: ...

    # --- v2 claim CRUD (§6.2, §6.4 universal-tuple persistence) ---

    def append_claim(self, claim: Claim) -> None: ...
    def get_claim(self, claim_id: str) -> Claim | None: ...
    def iter_claims(self) -> Iterator[Claim]: ...
    def iter_claims_for_doc(self, doc_id: str) -> Iterator[Claim]: ...

    # --- v2 concept CRUD (§6.7 shared lattice, §6.8 entity resolution) ---

    def add_concepts(self, concepts: Iterable[Concept]) -> None: ...
    def get_concept(self, concept_id: str) -> Concept | None: ...
    def iter_concepts(self) -> Iterator[Concept]: ...

    # --- destructive ops (§4.7, §8.6 family 13) ---

    def delete_chunks_for_section(self, section_id: str) -> list[str]: ...
    def delete_section(self, section_id: str) -> None: ...


__all__ = ["Store"]
