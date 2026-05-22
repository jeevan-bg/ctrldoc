"""BM25 lexical index — protocol and Tantivy-backed implementation.

The protocol is the seam every BM25 backend must satisfy. The Tantivy
implementation persists to a directory, supports idempotent-by-id
re-indexing (delete + insert in one writer transaction), and exposes
top-k search ranked by BM25.

SPEC-REF: §4.2 (BM25 lexical), §4.3 (retrieval)
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Protocol, runtime_checkable

import tantivy

BM25Hit = tuple[str, float]


@runtime_checkable
class BM25Index(Protocol):
    """Lexical index over `(chunk_id, text)` pairs with BM25-ranked search."""

    def add(self, chunk_id: str, text: str) -> None: ...

    def remove(self, chunk_id: str) -> None: ...

    def search(self, query: str, *, k: int) -> list[BM25Hit]: ...


class TantivyBM25Index:
    """Tantivy-backed `BM25Index`. Persists to a directory."""

    _CHUNK_ID_FIELD = "chunk_id"
    _BODY_FIELD = "body"

    def __init__(self, *, path: str | Path, writer_memory_mb: int = 32) -> None:
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._writer_memory = writer_memory_mb * 1024 * 1024
        schema_builder = tantivy.SchemaBuilder()
        # `raw` keeps the chunk id un-tokenised so delete_documents can target
        # it as an exact term.
        schema_builder.add_text_field(self._CHUNK_ID_FIELD, stored=True, tokenizer_name="raw")
        schema_builder.add_text_field(self._BODY_FIELD, stored=True)
        self._schema = schema_builder.build()
        self._index = tantivy.Index(self._schema, path=str(self._path))
        self._writer: tantivy.IndexWriter | None = None

    # --- lifecycle ---

    def _ensure_writer(self) -> tantivy.IndexWriter:
        if self._writer is None:
            self._writer = self._index.writer(self._writer_memory, 1)
        return self._writer

    def close(self) -> None:
        if self._writer is None:
            return
        self._writer.commit()
        self._writer.wait_merging_threads()
        self._writer = None
        self._index.reload()

    def __enter__(self) -> TantivyBM25Index:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # --- protocol ---

    def add(self, chunk_id: str, text: str) -> None:
        writer = self._ensure_writer()
        # Delete any prior copy of this chunk so re-adding is idempotent
        # by id (last-write-wins).
        writer.delete_documents(self._CHUNK_ID_FIELD, chunk_id)
        doc = tantivy.Document()
        doc.add_text(self._CHUNK_ID_FIELD, chunk_id)
        doc.add_text(self._BODY_FIELD, text)
        writer.add_document(doc)
        writer.commit()
        self._index.reload()

    def remove(self, chunk_id: str) -> None:
        writer = self._ensure_writer()
        writer.delete_documents(self._CHUNK_ID_FIELD, chunk_id)
        writer.commit()
        self._index.reload()

    def search(self, query: str, *, k: int) -> list[BM25Hit]:
        if k < 0:
            raise ValueError("k must be non-negative")
        if k == 0 or not query.strip():
            return []
        parsed = self._index.parse_query(query, [self._BODY_FIELD])
        searcher = self._index.searcher()
        result = searcher.search(parsed, limit=k)
        hits: list[BM25Hit] = []
        for score, addr in result.hits:
            doc = searcher.doc(addr)
            chunk_id = str(doc[self._CHUNK_ID_FIELD][0])
            hits.append((chunk_id, float(score)))
        return hits


__all__ = ["BM25Hit", "BM25Index", "TantivyBM25Index"]
