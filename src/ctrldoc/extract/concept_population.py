"""Concept-population wiring — store mentions + claims → `Concept` rows.

The L1.5 substrate (§6.7, §6.8) needs `Concept` rows persisted in the
store before a workspace can surface a shared-concept-lattice slice.
`extract.entity_resolution.EntityResolver` already implements the
four-step §6.8 recipe (blocking → judge → union-find → persist); this
module is the wiring layer that turns the per-document store contents
into the resolver's `ConceptMention` inputs and persists the result.

Two mention sources feed the resolver:

1. **NER `Entity` rows** — every persisted `Entity` whose
   `mention_chunk_ids` overlaps the document's chunks (in heuristic /
   thrifty / production profiles, these are GLiNER and / or
   claim-augmented mentions). One `ConceptMention` per (entity,
   chunk_id) pair; the chunk id stands in for the parent claim id when
   no claim row references the same surface.
2. **`Claim` subjects and objects** — every persisted `Claim`'s
   non-blank subject and object surface form is promoted to a
   `ConceptMention`. The claim's id is the `claim_id` so the resulting
   `Concept.mention_claim_ids` traces back through the verdict ledger.

The deterministic / heuristic profile pairs the resolver with:

* `HashEmbedder` for blocking-time cosine — identical surface forms
  collapse exactly (cosine 1.0); fuzzy variants fall below the default
  `tau_block` and stay separate clusters. The slice's ROADMAP row spells
  this out as "heuristic falls back to hash distance"; production
  swaps in `OllamaEmbedder` (bge-m3) for true semantic blocking.
* `HeuristicERJudge` — a no-LLM judge that returns `equivalent` when
  the two mention surfaces normalise (case-fold + whitespace-collapse)
  to the same canonical form, else `incomparable`. The §6.8 recipe's
  subsumption verdicts (`subsumes` / `subsumed_by`) require semantic
  reasoning beyond a normalised-string match, so the heuristic judge
  conservatively declines to emit them — production swaps in a real
  LLM judge that can.

`populate_concepts_for_doc(store, doc_id, embedder=None, config=None)`
is the single entry point ingest / `workspace add` callers reach for.
Output is the same `EntityResolution` the resolver returns; the
caller's hook is the `store.add_concepts(...)` side-effect.

SPEC-REF: §6.8 (entity resolution / canonicalization)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ctrldoc.extract.entity_resolution import (
    ConceptMention,
    EntityResolution,
    EntityResolutionConfig,
    EntityResolutionVerdict,
    EntityResolver,
)
from ctrldoc.ingest.embedder import Embedder, HashEmbedder
from ctrldoc.store import Store

_DEFAULT_HEURISTIC_EMBED_DIM: int = 32
"""Embedding dimension used when callers do not supply an embedder."""

_WHITESPACE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """Canonical-name normalisation — case-fold + whitespace-collapse."""
    return _WHITESPACE.sub(" ", text.strip()).casefold()


# ---------------------------------------------------------------------------
# Heuristic judge — no LLM, canonical-name equivalence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeuristicERJudge:
    """Deterministic ER judge keyed on normalised mention text.

    Returns `equivalent` when both mention surfaces collapse to the
    same case-folded whitespace-normalised canonical form; otherwise
    `incomparable`. Never emits `subsumes` / `subsumed_by` — the §6.8
    Galois subsumption verdicts require semantic reasoning beyond a
    string match, and producing them spuriously would inflate the
    `is_a` edge set the §6.8 release gates measure precision on.
    """

    def judge(
        self,
        *,
        left: ConceptMention,
        right: ConceptMention,
    ) -> EntityResolutionVerdict:
        if _normalise(left.mention_text) == _normalise(right.mention_text):
            return "equivalent"
        return "incomparable"


# ---------------------------------------------------------------------------
# Adapter — Entity + Claim → ConceptMention
# ---------------------------------------------------------------------------


def mentions_from_store(store: Store, *, doc_id: str) -> list[ConceptMention]:
    """Pull every concept mention for `doc_id` from the store.

    Two sources, merged in deterministic order: persisted `Entity`
    rows (NER + claim-augmented mentions) first, then persisted
    `Claim` subjects / objects in claim-id order. The mention id is
    derived from the source row so re-running this function across
    re-ingests produces the same id set.

    Entity rows are scoped to `doc_id` by intersecting
    `mention_chunk_ids` with the document's chunks. Claim rows are
    scoped directly via `iter_claims_for_doc`.
    """
    mentions: list[ConceptMention] = []

    # Which chunks belong to this doc?
    doc_chunk_ids = {c.id for c in store.iter_chunks() if _chunk_belongs_to_doc(c, doc_id)}

    # 1. Entity rows. One mention per (entity, chunk_id) pair so the
    #    per-chunk mention bookkeeping survives into the cluster.
    for entity in sorted(store.iter_entities(), key=lambda e: e.id):
        # Honour either name in the alias list when available, otherwise
        # fall back to the id-suffix; the id is the slugified canonical
        # form so even alias-empty rows still surface a surface form.
        surface = (entity.aliases[0] if entity.aliases else entity.id.split("/")[-1]).strip()
        if not surface:
            continue
        for chunk_id in entity.mention_chunk_ids:
            if doc_chunk_ids and chunk_id not in doc_chunk_ids:
                continue
            mentions.append(
                ConceptMention(
                    id=f"mention/entity/{entity.id}/{chunk_id}",
                    mention_text=surface,
                    primitive_type="Entity",
                    doc_id=doc_id,
                    claim_id=chunk_id,
                )
            )

    # 2. Claim subjects / objects. Sorted by (claim_id, endpoint) for
    #    deterministic mention-id assignment across re-runs.
    for claim in sorted(store.iter_claims_for_doc(doc_id), key=lambda c: c.id):
        for endpoint, value in (("subject", claim.subject), ("object", claim.object)):
            if value is None:
                continue
            cleaned = value.strip()
            if not cleaned:
                continue
            mentions.append(
                ConceptMention(
                    id=f"mention/claim/{claim.id}/{endpoint}",
                    mention_text=cleaned,
                    primitive_type="Entity",
                    doc_id=doc_id,
                    claim_id=claim.id,
                )
            )

    return mentions


def _chunk_belongs_to_doc(chunk: object, doc_id: str) -> bool:
    """Best-effort scope filter for chunk-doc membership.

    Chunks do not carry a `doc_id` column today — the per-doc
    SQLite store is the de-facto scoping mechanism. When the
    chunk row exposes a `doc_id` attribute we honour it; otherwise
    we admit the chunk so the per-doc store still surfaces every
    mention it persisted.
    """
    return getattr(chunk, "doc_id", doc_id) == doc_id


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def populate_concepts_for_doc(
    *,
    store: Store,
    doc_id: str,
    embedder: Embedder | None = None,
    judge: object | None = None,
    config: EntityResolutionConfig | None = None,
) -> EntityResolution:
    """Run §6.8 ER over the store's mentions + claims and persist concepts.

    `embedder` defaults to `HashEmbedder(dimension=32)` — the
    heuristic-profile choice. Production callers pass `OllamaEmbedder`
    for true semantic blocking. `judge` defaults to `HeuristicERJudge`;
    production callers swap in their LLM judge. `config` plumbs through
    the §6.8 `tau_block` (default 0.85) and any future knobs.

    Side-effect: every cluster's `Concept` is written via
    `store.add_concepts`. The `Concept.doc_ids` carries `doc_id` so the
    `§6.7` workspace-scoped concept lookup filters correctly.

    Returns the full `EntityResolution` (concepts, subsumption edges,
    cluster partition, judge-call count) so callers can log gate
    metrics without re-querying the store.
    """
    mentions = mentions_from_store(store, doc_id=doc_id)
    if not mentions:
        return EntityResolution(
            concepts=[],
            subsumption_edges=[],
            clusters=[],
            judge_calls=0,
        )

    effective_embedder = embedder or HashEmbedder(dimension=_DEFAULT_HEURISTIC_EMBED_DIM)
    effective_judge = judge or HeuristicERJudge()
    resolver = EntityResolver(
        embedder=effective_embedder,
        judge=effective_judge,  # type: ignore[arg-type]
        config=config,
    )
    outcome = resolver.resolve(mentions)
    if outcome.concepts:
        store.add_concepts(outcome.concepts)
    return outcome


__all__ = [
    "HeuristicERJudge",
    "mentions_from_store",
    "populate_concepts_for_doc",
]
