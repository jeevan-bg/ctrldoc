"""Evidence-pack builder.

Turns a ranked list of `chunk_id`s into an `EvidencePack` — the ≤ 6k
token bundle the downstream judge sees. Each emitted chunk becomes a
`Span` carrying its stable id and char range so verifier-side
re-retrieval can hit the same bytes.

SPEC-REF: §4.3 (evidence pack builder), §4.0 (Span / EvidencePack)
"""

from __future__ import annotations

from collections.abc import Iterable

from ctrldoc.models import EVIDENCE_PACK_TOKEN_CAP, EvidencePack, Span
from ctrldoc.store import Store


def build_evidence_pack(
    *,
    query: str,
    ranked_chunk_ids: Iterable[str],
    store: Store,
    retrieval_plan: list[str] | None = None,
    max_tokens: int = EVIDENCE_PACK_TOKEN_CAP,
) -> EvidencePack:
    """Assemble an `EvidencePack` from a ranked chunk-id list.

    The function walks `ranked_chunk_ids` in order. For each id, it
    looks up the chunk via `store.get_chunk`; missing ids are skipped
    silently. If adding the next chunk's tokens would exceed
    `max_tokens`, the build stops — the budget is strict.
    """
    if max_tokens < 0:
        raise ValueError("max_tokens must be non-negative")
    if max_tokens > EVIDENCE_PACK_TOKEN_CAP:
        raise ValueError(
            f"max_tokens {max_tokens} exceeds SPEC §4.3 cap of {EVIDENCE_PACK_TOKEN_CAP}"
        )

    spans: list[Span] = []
    used_tokens = 0
    for chunk_id in ranked_chunk_ids:
        chunk = store.get_chunk(chunk_id)
        if chunk is None:
            continue
        if used_tokens + chunk.token_count > max_tokens:
            continue
        spans.append(
            Span(
                chunk_id=chunk.id,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                text=chunk.text,
            )
        )
        used_tokens += chunk.token_count

    return EvidencePack(
        query=query,
        spans=spans,
        token_count=used_tokens,
        retrieval_plan=list(retrieval_plan or []),
    )


__all__ = ["build_evidence_pack"]
