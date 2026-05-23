"""`sqlite-vec` backend for the `VectorIndex` protocol.

Persists `(chunk_id, embedding)` pairs in a single SQLite file
(or `:memory:`). The dense vectors live in a `vec0` virtual table
with cosine distance built in; a sidecar `id_map` table holds
the `chunk_id ↔ rowid` mapping so callers can keep using stable
string IDs.

Cosine similarity is returned to the caller (score in `[-1, 1]`,
higher is better) so this backend is interchangeable with
`InMemoryVectorIndex` from the protocol's point of view.

SPEC-REF: §4.2 (dense vectors), §4.3 (retrieval)
"""

from __future__ import annotations

import sqlite3
import struct
from collections.abc import Iterator, Sequence

from ctrldoc.store.vectors import VectorDimensionMismatchError, VectorHit


class SqliteVecVectorIndex:
    """SQLite-backed dense-vector index using the `vec0` extension.

    Suitable for production corpora. The default `:memory:` path
    keeps tests hermetic; pass a file path to persist across runs.
    """

    def __init__(self, *, dimension: int, path: str = ":memory:") -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension
        self._conn = self._open(path)
        self._next_rowid = 1

    @property
    def dimension(self) -> int:
        return self._dimension

    def _open(self, path: str) -> sqlite3.Connection:
        import sqlite_vec  # type: ignore[import-untyped]

        conn = sqlite3.connect(path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS embeddings "
            f"USING vec0(embedding float[{self._dimension}] distance_metric=cosine)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS id_map ("
            " chunk_id TEXT PRIMARY KEY,"
            " rowid INTEGER UNIQUE NOT NULL"
            ")"
        )
        conn.commit()
        return conn

    def add(self, chunk_id: str, embedding: Sequence[float]) -> None:
        self._check_dim(embedding, "add")
        blob = _pack(embedding)
        existing = self._conn.execute(
            "SELECT rowid FROM id_map WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if existing is not None:
            rowid = int(existing[0])
            self._conn.execute("DELETE FROM embeddings WHERE rowid = ?", (rowid,))
        else:
            rowid = self._next_rowid
            self._next_rowid += 1
            self._conn.execute(
                "INSERT INTO id_map(chunk_id, rowid) VALUES (?, ?)", (chunk_id, rowid)
            )
        self._conn.execute("INSERT INTO embeddings(rowid, embedding) VALUES (?, ?)", (rowid, blob))
        self._conn.commit()

    def remove(self, chunk_id: str) -> None:
        row = self._conn.execute(
            "SELECT rowid FROM id_map WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if row is None:
            return
        rowid = int(row[0])
        self._conn.execute("DELETE FROM embeddings WHERE rowid = ?", (rowid,))
        self._conn.execute("DELETE FROM id_map WHERE chunk_id = ?", (chunk_id,))
        self._conn.commit()

    def search(self, query: Sequence[float], *, k: int) -> list[VectorHit]:
        if k < 0:
            raise ValueError("k must be non-negative")
        if k == 0:
            return []
        self._check_dim(query, "search")
        blob = _pack(query)
        rows = self._conn.execute(
            "SELECT e.rowid, e.distance, m.chunk_id "
            "FROM embeddings e JOIN id_map m ON e.rowid = m.rowid "
            "WHERE e.embedding MATCH ? AND k = ? "
            "ORDER BY e.distance",
            (blob, k),
        ).fetchall()
        # Stable insertion-order tie-break (matches InMemoryVectorIndex semantics).
        rows.sort(key=lambda row: (float(row[1]), int(row[0])))
        return [(str(chunk_id), 1.0 - float(distance)) for _, distance, chunk_id in rows]

    def iter(self) -> Iterator[tuple[str, list[float]]]:
        rows = self._conn.execute(
            "SELECT m.chunk_id, e.embedding "
            "FROM id_map m JOIN embeddings e ON m.rowid = e.rowid "
            "ORDER BY m.rowid"
        ).fetchall()
        for chunk_id, blob in rows:
            yield str(chunk_id), _unpack(blob, self._dimension)

    def _check_dim(self, vec: Sequence[float], op: str) -> None:
        if len(vec) != self._dimension:
            raise VectorDimensionMismatchError(
                f"{op} vector has dimension {len(vec)}; expected {self._dimension}"
            )

    def close(self) -> None:
        self._conn.close()


def _pack(vec: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *(float(x) for x in vec))


def _unpack(blob: bytes, dimension: int) -> list[float]:
    return list(struct.unpack(f"<{dimension}f", blob))


__all__ = ["SqliteVecVectorIndex"]
