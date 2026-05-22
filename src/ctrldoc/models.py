"""L1 data-model contracts shared across every layer.

These Pydantic models are the stable interface between ingest,
storage, retrieval, verification, orchestration and playbooks. Any
field added, removed, or retyped requires a `schema_version` bump
per SPEC §4.7 (versioning).

SPEC-REF: §4.0 (data model)
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, model_validator


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


__all__ = ["Chunk", "Section", "Span"]
