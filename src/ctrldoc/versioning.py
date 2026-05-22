"""Index versioning and content-hash helpers.

Three independent versions travel with every persisted index:
`schema_version`, `index_version`, `embedding_model_version`. On
mismatch at index open the runtime must fail fast with
`IndexVersionMismatchError`; silent migration is forbidden.

`content_hash` and `hash_chunk` provide deterministic, sha256-based
fingerprints so the storage layer can detect corruption and the
ingest layer can keep chunk ids stable across re-ingests.

SPEC-REF: §4.7 (versioning), §4.7 (index integrity)
"""

from __future__ import annotations

import hashlib
from typing import Final

from pydantic import BaseModel, ConfigDict

from ctrldoc.models import Chunk
from ctrldoc.provenance import SCHEMA_VERSION

INDEX_VERSION: Final[str] = "0.1.0"
EMBEDDING_MODEL_VERSION: Final[str] = "bge-m3-2024"


class IndexVersionMismatchError(RuntimeError):
    """Raised when an opened index's stored versions disagree with the runtime."""


class IndexVersions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    index_version: str
    embedding_model_version: str

    @classmethod
    def current(cls) -> IndexVersions:
        return cls(
            schema_version=SCHEMA_VERSION,
            index_version=INDEX_VERSION,
            embedding_model_version=EMBEDDING_MODEL_VERSION,
        )

    def assert_compatible_with(self, runtime: IndexVersions) -> None:
        """Compare to the runtime's expectations. Raises on any drift."""
        diffs: list[str] = []
        if self.schema_version != runtime.schema_version:
            diffs.append(
                f"schema_version: stored={self.schema_version!r} runtime={runtime.schema_version!r}"
            )
        if self.index_version != runtime.index_version:
            diffs.append(
                f"index_version: stored={self.index_version!r} runtime={runtime.index_version!r}"
            )
        if self.embedding_model_version != runtime.embedding_model_version:
            diffs.append(
                "embedding_model_version: "
                f"stored={self.embedding_model_version!r} "
                f"runtime={runtime.embedding_model_version!r}"
            )
        if diffs:
            raise IndexVersionMismatchError(
                "re-ingest required; version mismatch:\n  " + "\n  ".join(diffs)
            )


def content_hash(text: str) -> str:
    """Return a stable `sha256:<hex>` digest of `text` (utf-8 encoded)."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_chunk(chunk: Chunk) -> str:
    """Deterministic hash of a chunk's identity-defining fields.

    Bookkeeping ids (`id`, `embedding_id`) are intentionally excluded
    so the hash survives renumbering of the index.
    """
    payload = f"{chunk.section_id}\0{chunk.char_start}\0{chunk.char_end}\0{chunk.text}"
    return content_hash(payload)


__all__ = [
    "EMBEDDING_MODEL_VERSION",
    "INDEX_VERSION",
    "IndexVersionMismatchError",
    "IndexVersions",
    "content_hash",
    "hash_chunk",
]
