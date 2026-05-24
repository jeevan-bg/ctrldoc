"""Tier-2 SVO extractor — pure-Python helpers shared with the spaCy backend.

The Tier-2 extractor is the second layer of the §6.4 schema co-induction
pipeline. Where Tier-1 (`tier1.py`) emits typed edges between concept
clusters from lexico-syntactic patterns, Tier-2 lifts each sentence into
the **universal claim tuple** shape from §6.2:

    Claim = (subject, predicate, object, polarity, modality, qualifier,
             span_refs, confidence)

This module holds the deterministic helpers — modality lexicon, polarity
classification, predicate lemmatisation — that **do not require spaCy**.
The spaCy-backed extractor that produces SVO triples lives in the
sibling `tier2_spacy.py` module so callers that do not need the dependency
parse can import the helpers without pulling spaCy in.

Modality is decided from the auxiliary-verb cue in the dependency tree
(`must` / `shall` → obligatory, `should` → recommended, `may` → permitted,
`if` / `when` / `would` / `could` / `might` → hypothetical) with
prohibited reserved for negated obligations and `never`-style clauses.
Polarity flips negative on `not` / `never` / `n't` / `cannot`. Predicate
lemmatisation maps inflected verbs back to a third-person-singular base
form so the eval `claim_tuple_matches` core-match comparison succeeds —
modal constructions keep the base form because the gold tuples are
phrased the same way.

SPEC-REF: §6.4 (schema co-induction — Tier-2 SVO extraction)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ctrldoc.eval.claim_extraction import ModalityLiteral, PolarityLiteral

# --- modality + polarity lexicons -------------------------------------------


MODAL_LEXICON: dict[str, ModalityLiteral] = {
    # Obligation
    "must": "obligatory",
    "shall": "obligatory",
    "required": "obligatory",
    "have": "obligatory",  # "have to" — weaker, but classes here
    "needs": "obligatory",
    # Recommendation
    "should": "recommended",
    "ought": "recommended",
    "recommended": "recommended",
    "advised": "recommended",
    # Permission
    "may": "permitted",
    "can": "permitted",
    "allowed": "permitted",
    "permitted": "permitted",
    "optional": "permitted",
    # Prohibition — only the explicit obligation-flipping cues. Plain
    # `never sees` reads as a negative *assertion* (polarity=negative,
    # modality=asserted), so `never` does NOT live here — it shows up
    # only as a `NEGATION_TOKENS` member that flips polarity. The
    # prohibited modality is reserved for negated obligation modals
    # (`shall not`, `must not`) and the lexical `forbidden`/`prohibited`
    # surface forms.
    "forbidden": "prohibited",
    "prohibited": "prohibited",
    "cannot": "prohibited",
    # Hypothetical
    "could": "hypothetical",
    "would": "hypothetical",
    "might": "hypothetical",
    "if": "hypothetical",
    "when": "hypothetical",
    "unless": "hypothetical",
    "whenever": "hypothetical",
}
"""Lower-cased modal/aux/conditional cue → universal modality label."""


NEGATION_TOKENS: frozenset[str] = frozenset(
    {"not", "n't", "never", "no", "cannot", "neither", "nor"}
)
"""Lower-cased token surface forms that flip polarity to `negative`."""


# Tokens that, when preceded by an obligation modal (`shall`, `must`),
# upgrade the modality from `obligatory` to `prohibited`. These are the
# negation surfaces only; standalone occurrences are handled by the
# `never` / `cannot` entries above.
_PROHIBITION_BOUNDARY: frozenset[str] = frozenset({"not", "n't", "never"})


# Verbs whose canonical form differs from their `+s` inflection — third-person
# singular forms used by the eval gold tuples.
_IRREGULAR_LEMMA_3SG: dict[str, str] = {
    "be": "is",
    "is": "is",
    "am": "is",
    "are": "are",
    "was": "is",
    "were": "are",
    "been": "is",
    "being": "is",
    "have": "has",
    "has": "has",
    "had": "has",
    "do": "does",
    "does": "does",
    "did": "does",
    "go": "goes",
    "goes": "goes",
    "went": "goes",
    "ran": "runs",
    "run": "runs",
    "runs": "runs",
    "running": "runs",
    "say": "says",
    "said": "says",
    "says": "says",
    "see": "sees",
    "saw": "sees",
    "sees": "sees",
    "seen": "sees",
    "take": "takes",
    "took": "takes",
    "taken": "takes",
    "takes": "takes",
    "give": "gives",
    "gave": "gives",
    "given": "gives",
    "gives": "gives",
    "find": "finds",
    "found": "finds",
    "finds": "finds",
}


# Irregular verb-lemma → plural-subject surface form. Used when the
# input lemma needs an explicit different surface for a plural subject
# (`be` → `are`, `have` → `have`). Most lemmas pass through unchanged;
# this table is the small set of irregular surface forms.
_IRREGULAR_PLURAL: dict[str, str] = {
    "be": "are",
    "is": "are",
    "are": "are",
    "was": "were",
    "were": "were",
    "have": "have",
    "has": "have",
    "had": "had",
    "do": "do",
    "does": "do",
    "did": "did",
}


# --- helpers ----------------------------------------------------------------


def classify_modality(*, tokens: list[str]) -> ModalityLiteral:
    """Pick the strongest modality cue present in `tokens`.

    Priority order (highest first):

    1. Prohibition: any deontic modal (`must` / `shall` / `may` /
       `can`) immediately followed by a negation surface (`not` /
       `n't`) -> `prohibited`. Captures `shall not`, `must not`,
       `may not`, and the `cannot` -> `can` + `not` tokenization.
    2. Lexical `prohibited` / `forbidden` / `cannot` cues.
    3. Obligation (`must` / `shall`) — these win over a co-occurring
       conditional `if` (e.g. `If X, the implementation MUST close Y`
       reads as `obligatory`, not `hypothetical`).
    4. Recommendation (`should`).
    5. Permission (`may` / `can`) — but only when no conditional cue
       co-occurs; a conditional `may` (e.g. `the customer may request`
       under an `if` clause) reads as `hypothetical`.
    6. Hypothetical (`if` / `when` / `unless` / `whenever` / `could` /
       `would` / `might`).
    7. `asserted` floor.
    """
    lower = [t.lower() for t in tokens]

    # 1. Modal + negation -> prohibited.
    _flippable: frozenset[ModalityLiteral] = frozenset({"obligatory", "permitted"})
    for i, tok in enumerate(lower):
        if MODAL_LEXICON.get(tok) in _flippable:
            tail = lower[i + 1 : i + 4]
            if any(t in _PROHIBITION_BOUNDARY for t in tail):
                return "prohibited"

    has_conditional = any(MODAL_LEXICON.get(tok) == "hypothetical" for tok in lower)

    # 2 + 3. Lexical prohibited / obligation. Obligation beats
    # conditional.
    for tok in lower:
        mod = MODAL_LEXICON.get(tok)
        if mod in ("prohibited", "obligatory"):
            return mod

    # 4. Recommendation.
    for tok in lower:
        if MODAL_LEXICON.get(tok) == "recommended":
            return "recommended"

    # 5. Permission — demoted to hypothetical when a conditional cue
    # co-occurs.
    for tok in lower:
        if MODAL_LEXICON.get(tok) == "permitted":
            return "hypothetical" if has_conditional else "permitted"

    # 6. Hypothetical from any remaining hypothetical-class cue.
    if has_conditional:
        return "hypothetical"

    return "asserted"


def classify_polarity(*, tokens: list[str]) -> PolarityLiteral:
    """Return `negative` if any token in `tokens` is a known negator."""
    for tok in tokens:
        if tok.lower() in NEGATION_TOKENS:
            return "negative"
    return "affirmative"


def merge_modality_with_polarity(
    *, polarity: PolarityLiteral, modality: ModalityLiteral
) -> tuple[PolarityLiteral, ModalityLiteral]:
    """Reconcile polarity with modality.

    A `prohibited` modality always carries `negative` polarity — the
    eval gold tuples encode "shall not exceed" as `(polarity=negative,
    modality=prohibited)`. Other combinations are returned unchanged.
    """
    if modality == "prohibited":
        return "negative", "prohibited"
    return polarity, modality


def lemmatize_predicate(
    verb: str, *, modality: ModalityLiteral, subject_is_plural: bool = False
) -> str:
    """Normalise a raw verb to the gold-tuple shape.

    The eval gold tuples follow **subject-verb agreement** on the
    canonical subject: a singular subject takes the third-person-
    singular form (`every chunk carries`, `the system uses`), a plural
    subject takes the bare infinitive (`claims carry`, `clients
    retry`). Modality is independent of agreement.

    `verb` should be the lexical lemma (spaCy's `Token.lemma_`). With
    `subject_is_plural=False` the function returns the third-person-
    singular form; with `subject_is_plural=True` it returns the lemma
    itself (caller-provided base form), with a small irregular table
    handling `to be` / `to have` / `to do`.
    """
    del modality  # currently unused; reserved for future routing
    raw = verb.strip().lower()
    if not raw:
        return raw

    if subject_is_plural:
        return _IRREGULAR_PLURAL.get(raw, raw)

    base = _IRREGULAR_LEMMA_3SG.get(raw)
    if base is None:
        base = _to_third_person_singular(raw)
    return base


def _to_third_person_singular(verb: str) -> str:
    """Best-effort regular `+s` / `+es` / `+ies` inflection.

    We do NOT strip `-ed` / `-ing` here — spaCy's `lemma_` already
    returns the lexeme's base form, so callers should pass the lemma
    (e.g. `run`, `carry`, `validate`) rather than an inflected surface
    form. The branches below cover the base→3SG transformation only.
    """
    if verb.endswith("ies"):
        # Already 3SG form (`carries`, `applies`).
        return verb
    if verb.endswith("y") and len(verb) >= 2 and verb[-2] not in "aeiou":
        return verb[:-1] + "ies"
    if verb.endswith(("ch", "sh", "x", "z", "ss")):
        return verb + "es"
    if verb.endswith("s"):
        # Already in `-s` form (`uses`, `gives`, `goes` — irregulars
        # caught upstream, regulars fall through here).
        return verb
    return verb + "s"


# --- config -----------------------------------------------------------------


@dataclass(frozen=True)
class Tier2Config:
    """Tunable knobs for the spaCy-backed Tier-2 SVO extractor."""

    spacy_model: str = "en_core_web_sm"
    """spaCy pipeline to load (must include parser + NER components)."""

    entity_labels: tuple[str, ...] = field(
        default=(
            "PERSON",
            "ORG",
            "GPE",
            "LOC",
            "PRODUCT",
            "DATE",
            "TIME",
            "QUANTITY",
            "CARDINAL",
            "MONEY",
            "PERCENT",
            "EVENT",
        )
    )
    """spaCy NER labels that count as concept-bearing entities for the
    Tier-2 SVO extractor's subject / object normalisation."""

    max_object_tokens: int = 12
    """Cap on the size of the extracted object noun-phrase to keep
    runaway clausal complements out of the gold-comparison surface."""


__all__ = [
    "MODAL_LEXICON",
    "NEGATION_TOKENS",
    "Tier2Config",
    "classify_modality",
    "classify_polarity",
    "lemmatize_predicate",
    "merge_modality_with_polarity",
]
