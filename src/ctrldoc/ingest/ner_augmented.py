"""Doc-type label catalogue + claim-graph-augmented `NERTagger`.

Two helpers land here together because they share one goal: lift the
recall of the §6.8 entity-resolution layer without changing its
interface.

`default_labels_for_doc_type` returns a per-doc-type label set wider
than the conservative pre-S-153 default of ``["person", "system",
"concept"]``. Spec / RFC / academic / runbook bodies surface a richer
catalogue (method, metric, dataset, requirement, ...) so GLiNER's
zero-shot prompt steers toward the concepts actually present in
domain prose.

`ClaimAugmentedNERTagger` wraps a base `NERTagger` (typically
`GLiNERTagger` in thrifty / production profiles) and, after the base
tagger runs, promotes every claim-tuple subject and object emitted by
a `ClaimExtractor` into an `EntityMention`. This is the explicit
fallback the slice's ROADMAP row spells out: when GLiNER returns
empty (or thin), the universal-tuple subjects / objects still seed
concepts so the downstream `EntityResolver` is never empty-handed.
Mentions that duplicate something the base tagger already surfaced
(same lower-cased text) are suppressed so the count is not inflated.

SPEC-REF: §6.8 (entity resolution — wider candidate pool feeds the
blocking + LLM-judge pipeline).
"""

from __future__ import annotations

from ctrldoc.eval.claim_extraction import ClaimExtractor
from ctrldoc.ingest.ner import EntityMention, NERTagger

# Label every augmented mention carries by default. Stays distinct from
# the GLiNER label list so callers can tell augmented vs. tagged mentions
# apart at the canonicalisation step if they want.
DEFAULT_AUGMENTED_LABEL = "concept"


# Pre-S-153 default — kept as the conservative fallback when a doc type
# is unknown so heuristic-profile callers keep their existing recall.
_FALLBACK_LABELS: tuple[str, ...] = ("person", "system", "concept")


# Per-doc-type expansions. Each list is curated from the §6.4 closed
# primitive-type alphabet plus a handful of doc-specific concept labels
# GLiNER's zero-shot prompt benefits from when the prose is narrow.
# Lists are deterministic (declaration order is preserved) so the same
# doc type always sees the same prompt.
_LABELS_BY_DOC_TYPE: dict[str, tuple[str, ...]] = {
    "spec": (
        "system",
        "component",
        "interface",
        "requirement",
        "constraint",
        "concept",
        "person",
    ),
    "runbook": (
        "system",
        "component",
        "command",
        "metric",
        "alert",
        "concept",
        "person",
    ),
    "rfc": (
        "system",
        "protocol",
        "field",
        "requirement",
        "term",
        "concept",
        "person",
    ),
    "legal": (
        "party",
        "obligation",
        "term",
        "definition",
        "right",
        "concept",
        "person",
    ),
    "academic": (
        "method",
        "model",
        "dataset",
        "metric",
        "concept",
        "person",
        "organization",
    ),
    "narrative": (
        "person",
        "place",
        "object",
        "concept",
        "system",
    ),
}


def default_labels_for_doc_type(doc_type: str) -> list[str]:
    """Return the label list to seed GLiNER for ``doc_type``.

    Unknown doc-types fall back to the conservative pre-S-153
    triplet so callers passing a never-before-seen type still get a
    usable prompt rather than an exception.
    """
    if doc_type in _LABELS_BY_DOC_TYPE:
        return list(_LABELS_BY_DOC_TYPE[doc_type])
    return list(_FALLBACK_LABELS)


class ClaimAugmentedNERTagger:
    """`NERTagger` that augments a base tagger with claim-tuple endpoints.

    Runs the base tagger first; then runs the claim extractor over the
    same text and promotes every distinct, non-blank subject / object
    that the base did NOT already surface (case-insensitive match) into
    an `EntityMention` carrying ``DEFAULT_AUGMENTED_LABEL``. When the
    string appears verbatim in the input, char offsets locate the first
    occurrence; otherwise offsets fall back to ``(0, 0)`` rather than
    invent a span — downstream span-citation code is responsible for
    handling the zero range when present.

    Idempotent: calling tag twice on the same text returns the same
    list (order is base mentions in their original order, then
    augmented mentions in extractor / claim / subject-before-object
    order).
    """

    def __init__(
        self,
        *,
        base: NERTagger,
        claim_extractor: ClaimExtractor,
        augmented_label: str = DEFAULT_AUGMENTED_LABEL,
        augmented_score: float = 0.5,
    ) -> None:
        self._base = base
        self._claim_extractor = claim_extractor
        self._augmented_label = augmented_label
        self._augmented_score = augmented_score

    def tag(self, text: str, *, labels: list[str]) -> list[EntityMention]:
        if not text or not text.strip():
            return []

        base_mentions = list(self._base.tag(text, labels=labels))
        seen_lowercase = {m.text.lower() for m in base_mentions}

        augmented: list[EntityMention] = []
        for claim in self._claim_extractor.extract(text):
            for endpoint in (claim.subject, claim.object):
                stripped = endpoint.strip()
                if not stripped:
                    continue
                key = stripped.lower()
                if key in seen_lowercase:
                    continue
                seen_lowercase.add(key)
                start = text.find(stripped)
                if start < 0:
                    start = 0
                    end = 0
                else:
                    end = start + len(stripped)
                augmented.append(
                    EntityMention(
                        text=stripped,
                        label=self._augmented_label,
                        start=start,
                        end=end,
                        score=self._augmented_score,
                    )
                )

        return base_mentions + augmented


__all__ = [
    "DEFAULT_AUGMENTED_LABEL",
    "ClaimAugmentedNERTagger",
    "default_labels_for_doc_type",
]
