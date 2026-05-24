"""Adapter: extractor `ClaimTuple` ↔ persisted `Claim`.

The §6.2 universal claim tuple is what the SVO extractor (and the
Tier-3 layers above it) emit. The §7 / §8 data model stores a
superset shape with bookkeeping the substrate needs: a content-hashed
`id`, parent `doc_id` / `section_id` / `chunk_id` binding, `span_refs`
into chunk text, and the §7 modal-force alphabet rather than the
§6.2 surface alphabet.

This module is the canonical seam between the two shapes.

* `claim_id_for_tuple` is a deterministic content hash over the six
  logical fields plus the doc / chunk binding. Re-running the
  pipeline over identical inputs yields identical ids — the
  re-ingest safety property.
* `claim_from_tuple` builds a persisted `Claim` end-to-end: maps
  polarity / modality alphabets, normalises the qualifier into the
  `dict[str, object]` shape, anchors one `Span` covering the source
  chunk, and assigns a deterministic floor `confidence = 1.0`
  (pre-calibration; Tier-3 layers replace this).
* `claim_to_tuple` is the inverse: persisted `Claim` back to the §6.2
  universal tuple. Used by every consumer that needs to feed a
  persisted claim into a tuple-shaped engine (the §6.3 Galois floor,
  the §6.6 optimal-transport reductions in `ctrldoc.ops.coverage`,
  the MCP `coverage` / `list_check` handlers).

SPEC-REF: §6.2, §6.4
"""

from __future__ import annotations

from ctrldoc.eval.claim_extraction import ClaimTuple
from ctrldoc.eval.claim_extraction import ModalityLiteral as SurfaceModality
from ctrldoc.eval.claim_extraction import PolarityLiteral as SurfacePolarity
from ctrldoc.models import Chunk, Span
from ctrldoc.models_v1 import Claim
from ctrldoc.models_v1 import ModalityLiteral as PersistedModality
from ctrldoc.models_v1 import PolarityLiteral as PersistedPolarity
from ctrldoc.versioning import content_hash

# Surface polarity (`affirmative` / `negative` per §6.2 extractor
# alphabet) → persisted polarity (`+` / `-` per §7 storage alphabet).
_POLARITY_MAP: dict[SurfacePolarity, PersistedPolarity] = {
    "affirmative": "+",
    "negative": "-",
}

# Surface modality (six-way §6.2 alphabet) → §7 modal force.
#
# `prohibited` collapses with `neg` — a prohibition is a negated
# obligation, which the storage alphabet represents directly.
# `hypothetical` has no §7 modal-force slot, so it maps to `None`
# (the documented "modality was not bound" sentinel). Hypotheticals
# still preserve their cue word in the `qualifier` slot at the
# extractor layer.
_MODALITY_MAP: dict[SurfaceModality, PersistedModality | None] = {
    "asserted": "assert",
    "obligatory": "must",
    "recommended": "should",
    "permitted": "may",
    "prohibited": "neg",
    "hypothetical": None,
}


# Floor confidence for Tier-2 SVO claims. Tier-3 layers (NLI judge,
# paraphrase voting, isotonic calibration) replace this when they run.
_TIER2_FLOOR_CONFIDENCE: float = 1.0


def claim_id_for_tuple(
    *,
    doc_id: str,
    chunk_id: str,
    tuple_: ClaimTuple,
) -> str:
    """Return a deterministic `sha256:<hex>` id for the universal tuple.

    The payload concatenates doc / chunk binding with every logical
    field. Two identical tuples emitted from the same chunk produce
    the same id (idempotent re-ingest); the same tuple emitted from
    a different chunk produces a distinct id (so multi-mention
    bindings can be tracked individually before the L1.5 concept
    resolver collapses them).
    """
    payload = "\0".join(
        (
            doc_id,
            chunk_id,
            tuple_.subject,
            tuple_.predicate,
            tuple_.object,
            tuple_.polarity,
            tuple_.modality,
            tuple_.qualifier,
        )
    )
    return content_hash(payload)


def claim_from_tuple(
    *,
    doc_id: str,
    chunk: Chunk,
    tuple_: ClaimTuple,
    confidence: float = _TIER2_FLOOR_CONFIDENCE,
) -> Claim:
    """Convert a `ClaimTuple` + its source `Chunk` into a persisted `Claim`.

    Anchors exactly one `Span` covering the chunk's full text — the
    SVO extractor works sentence-level but the chunk is the smallest
    persisted unit on the §7 graph, so chunk-level grounding is the
    conservative floor. Tier-3 layers refine this when they bind a
    claim to a narrower span.

    Raises `ValueError` when the tuple carries a modality or polarity
    label outside the §6.2 alphabet — the storage layer must reject
    unknown labels rather than silently coerce them.
    """
    if tuple_.polarity not in _POLARITY_MAP:
        raise ValueError(f"unknown polarity {tuple_.polarity!r}")
    if tuple_.modality not in _MODALITY_MAP:
        raise ValueError(f"unknown modality {tuple_.modality!r}")

    persisted_polarity = _POLARITY_MAP[tuple_.polarity]
    persisted_modality = _MODALITY_MAP[tuple_.modality]

    qualifier: dict[str, object] = {"text": tuple_.qualifier} if tuple_.qualifier else {}

    span = Span(
        chunk_id=chunk.id,
        char_start=0,
        char_end=len(chunk.text),
        text=chunk.text,
    )

    surface_text = f"{tuple_.subject} {tuple_.predicate} {tuple_.object}".strip()

    return Claim(
        id=claim_id_for_tuple(doc_id=doc_id, chunk_id=chunk.id, tuple_=tuple_),
        doc_id=doc_id,
        text=surface_text,
        subject=tuple_.subject or None,
        predicate=tuple_.predicate,
        object=tuple_.object or None,
        polarity=persisted_polarity,
        modality=persisted_modality,
        qualifier=qualifier,
        span_refs=[span],
        section_id=chunk.section_id,
        concept_ids=[],
        typed_slots={},
        confidence=confidence,
    )


# Inverse mappings — persisted alphabet → §6.2 surface alphabet. Used
# by `claim_to_tuple` below and by consumers that prefer to import the
# tables directly.
_PERSISTED_TO_SURFACE_POLARITY: dict[PersistedPolarity, SurfacePolarity] = {
    "+": "affirmative",
    "-": "negative",
}

_PERSISTED_TO_SURFACE_MODALITY: dict[PersistedModality | None, SurfaceModality] = {
    "assert": "asserted",
    "must": "obligatory",
    # `shall` collapses with `must` on the deontic chain (RFC-2119
    # treats them as synonymous; the storage alphabet keeps both for
    # surface-form fidelity, but §6.3's lattice cares about logical
    # force only).
    "shall": "obligatory",
    "may": "permitted",
    "should": "recommended",
    "neg": "prohibited",
    # `None` is the documented "modality was not bound" sentinel from
    # the §7 storage alphabet — projects back to the universal-tuple
    # neutral descriptive force.
    None: "asserted",
}


def claim_to_tuple(claim: Claim) -> ClaimTuple:
    """Convert a persisted `Claim` back into the §6.2 universal tuple.

    The inverse of `claim_from_tuple`'s alphabet mapping. Subject and
    object slots on a persisted claim may be `None` (the §7 storage
    alphabet permits it) — the universal tuple expects strings, so we
    coerce `None` to the empty string. The Galois floor and the §6.6
    transport reduction both treat empty strings as distinct from any
    non-empty value, so the structural-floor verdict's correctness is
    preserved.

    The qualifier slot on the §7 record is a free-form
    ``dict[str, object]``; the §6.2 tuple carries a plain text string.
    This adapter reads the ``"text"`` key written by `claim_from_tuple`
    and falls back to the empty string for any other shape (heuristic
    or LLM-induced claims that put a different structured payload
    there). Non-textual qualifiers are intentionally dropped at this
    seam — the tuple-shaped engines reason on text only.
    """
    qualifier_text = ""
    raw = claim.qualifier.get("text") if isinstance(claim.qualifier, dict) else None
    if isinstance(raw, str):
        qualifier_text = raw
    return ClaimTuple(
        subject=claim.subject or "",
        predicate=claim.predicate,
        object=claim.object or "",
        polarity=_PERSISTED_TO_SURFACE_POLARITY[claim.polarity],
        modality=_PERSISTED_TO_SURFACE_MODALITY[claim.modality],
        qualifier=qualifier_text,
    )


__all__ = [
    "claim_from_tuple",
    "claim_id_for_tuple",
    "claim_to_tuple",
]
