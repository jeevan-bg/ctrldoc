"""`fastcoref` backend for the `CorefResolver` protocol.

Rewrites a text by replacing each anaphoric mention with its
canonical mention. The canonical mention is chosen per cluster
as the longest span (typically a proper noun rather than a
pronoun), with ties broken by earliest position so the result
is deterministic.

The fastcoref model is loaded lazily on first `resolve()` call so
that importing this module is cheap.

SPEC-REF: §4.1 (ingest step 2 — coref)
"""

from __future__ import annotations

from typing import Any


class FastCorefResolver:
    """Production coreference resolver using `fastcoref`."""

    def __init__(self, *, device: str = "cpu") -> None:
        self._device = device
        self._model: Any | None = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            from fastcoref import FCoref  # type: ignore[import-untyped]

            self._model = FCoref(device=self._device)
        return self._model

    def resolve(self, text: str) -> str:
        if not text.strip():
            return text
        model = self._ensure_model()
        results = model.predict(texts=[text])
        if not results:
            return text
        clusters: list[list[tuple[int, int]]] = list(results[0].get_clusters(as_strings=False))
        if not clusters:
            return text
        return _apply_replacements(text, clusters)


def _apply_replacements(text: str, clusters: list[list[tuple[int, int]]]) -> str:
    """For each cluster, replace every non-canonical span with the canonical text.

    Replacements are applied right-to-left so earlier char offsets stay
    valid throughout the rewrite.
    """
    edits: list[tuple[int, int, str]] = []
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        canonical_start, canonical_end = _pick_canonical(text, cluster)
        canonical_text = text[canonical_start:canonical_end]
        for start, end in cluster:
            if (start, end) == (canonical_start, canonical_end):
                continue
            edits.append((start, end, canonical_text))
    if not edits:
        return text
    # Right-to-left so we never invalidate offsets of pending edits.
    edits.sort(key=lambda edit: edit[0], reverse=True)
    out = text
    for start, end, replacement in edits:
        out = out[:start] + replacement + out[end:]
    return out


def _pick_canonical(text: str, cluster: list[tuple[int, int]]) -> tuple[int, int]:
    """Longest mention wins; ties broken by earliest position."""
    return max(cluster, key=lambda span: (span[1] - span[0], -span[0]))


__all__ = ["FastCorefResolver"]
