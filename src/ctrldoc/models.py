"""L1 data-model contracts shared across every layer.

These Pydantic models are the stable interface between ingest,
storage, retrieval, verification, orchestration and playbooks. Any
field added, removed, or retyped requires a `schema_version` bump
per SPEC §4.7 (versioning).

SPEC-REF: §4.0 (data model)
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, model_validator

UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]
"""A probability or score in `[0.0, 1.0]`."""

EVIDENCE_PACK_TOKEN_CAP = 6000  # SPEC §4.3 evidence pack ≤ 6k tokens


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Chunk(_Strict):
    """A leaf chunk of text — the unit of embedding and retrieval."""

    id: str
    section_id: str
    text: str
    token_count: NonNegativeInt
    char_start: NonNegativeInt
    char_end: NonNegativeInt
    embedding_id: str
    metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_char_range(self) -> Chunk:
        if self.char_end < self.char_start:
            raise ValueError(
                f"char_end ({self.char_end}) must be >= char_start ({self.char_start})"
            )
        return self


class Section(_Strict):
    """A node in the structural tree of a document."""

    id: str
    parent_id: str | None
    title: str
    summary: str
    chunk_ids: list[str]


class Span(_Strict):
    """A pointer into a chunk: the exact substring used as evidence."""

    chunk_id: str
    char_start: NonNegativeInt
    char_end: NonNegativeInt
    text: str

    @model_validator(mode="after")
    def _check_char_range(self) -> Span:
        if self.char_end < self.char_start:
            raise ValueError(
                f"char_end ({self.char_end}) must be >= char_start ({self.char_start})"
            )
        return self


class EvidencePack(_Strict):
    """A bundled set of retrieved spans handed to a judge or generator."""

    query: str
    spans: list[Span]
    token_count: Annotated[int, Field(ge=0, le=EVIDENCE_PACK_TOKEN_CAP)]
    retrieval_plan: list[str]


class Claim(_Strict):
    """An atomic claim produced by claim decomposition + verification."""

    text: str
    citations: list[Span]
    verified: bool
    confidence: UnitInterval
    nli_score: UnitInterval
    judge_score: UnitInterval


VerdictLiteral = Literal["Covered", "Partial", "NotCovered", "Ambiguous"]


class Verdict(_Strict):
    """A per-item verdict in coverage_audit / quality_audit."""

    item_id: str
    verdict: VerdictLiteral
    citations: list[Span]
    confidence: UnitInterval


SeverityLiteral = Literal["info", "warn", "critical"]


class Finding(_Strict):
    """A finding emitted by analytical_review / anomaly_scan."""

    ctrldoc: str
    location: Span
    claim: str
    severity: SeverityLiteral


RelationTypeLiteral = Literal[
    "depends_on",
    "contradicts",
    "refines",
    "instantiates",
    "conflicts_with",
    "prerequisite_of",
    "alternative_to",
]


class RelationEdge(_Strict):
    """An edge in the concept-relation graph emitted by relation_map."""

    src_concept: str
    dst_concept: str
    type: RelationTypeLiteral
    citations: list[Span]
    confidence: UnitInterval


class Entity(_Strict):
    """A canonicalised mention cluster — one row in the entity glossary."""

    id: str
    aliases: list[str]
    type: str
    mention_chunk_ids: list[str]


EntityGlossary: TypeAlias = dict[str, Entity]


def build_entity_glossary(entities: Iterable[Entity]) -> EntityGlossary:
    """Index `entities` by their canonical id. Raises on duplicate ids."""
    glossary: EntityGlossary = {}
    for entity in entities:
        if entity.id in glossary:
            raise ValueError(f"duplicate entity id in glossary: {entity.id!r}")
        glossary[entity.id] = entity
    return glossary


__all__ = [
    "EVIDENCE_PACK_TOKEN_CAP",
    "Chunk",
    "Claim",
    "Entity",
    "EntityGlossary",
    "EvidencePack",
    "Finding",
    "RelationEdge",
    "Section",
    "Span",
    "UnitInterval",
    "Verdict",
    "build_entity_glossary",
]
