"""spaCy-backed Tier-2 SVO extractor — the §6.4 dependency-parser layer.

Lifts each sentence into a list of universal `ClaimTuple` rows from
§6.2 by walking the spaCy dependency parse:

* every root-style verb (or its coordinated peers) becomes one tuple;
* the verb's `nsubj` / `nsubjpass` subtree is the subject;
* its `dobj` / `attr` / `acomp` / `pcomp` / `xcomp` subtree is the object,
  falling back to a `prep`-phrase head when no direct object exists;
* a copular `be` / `have` ROOT with an `acomp` child realises a compound
  predicate (`are normalised`), with the acomp's children becoming the
  object surface form;
* modality + polarity come from the auxiliary / negation children, with
  `tier2.classify_modality` and `tier2.classify_polarity` arbitrating;
* qualifier comes from the verb's prepositional modifier (`prep`) or
  `advcl` subtree when present, with the verb's primary object's prep
  carve-out preserved as part of the object.

The spaCy pipeline (parser + NER) is loaded lazily on first call so
that simply importing this module is cheap. The pipeline name is
`Tier2Config.spacy_model` (default `en_core_web_sm`).

The headline contract is that this class satisfies the
`ctrldoc.eval.claim_extraction.ClaimExtractor` Protocol so it can be
dropped straight into `ClaimExtractionEvalRunner`. The §6.4 release
gate for this tier is `F1 >= TIER2_F1_THRESHOLD = 0.75` on the
SVO-amenable subset of the eval set.

SPEC-REF: §6.4 (schema co-induction — Tier-2 SVO extraction)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ctrldoc.eval.claim_extraction import ClaimTuple, ModalityLiteral, PolarityLiteral
from ctrldoc.extract.tier2 import (
    Tier2Config,
    classify_modality,
    classify_polarity,
    lemmatize_predicate,
    merge_modality_with_polarity,
)

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from spacy.tokens import (  # type: ignore[import-untyped,import-not-found,unused-ignore]
        Span,
        Token,
    )


TIER2_F1_THRESHOLD: float = 0.75
"""Per-§6.4 release gate for the Tier-2 SVO extractor."""


# Object-candidate dependency labels, in priority order. The first hit
# off the verb wins.
_OBJECT_DEPS: tuple[str, ...] = (
    "dobj",
    "attr",
    "acomp",
    "oprd",
    "ccomp",
    "xcomp",
    "npadvmod",  # `wait [thirty seconds]`, `commented [yesterday]` — duration / time NP
)


# Dependency labels that signal a subtree we should *exclude* from the
# object's surface form — these belong to the qualifier slot when they
# attach below the object head.
_OBJECT_TRAILING_DROP: frozenset[str] = frozenset(
    {
        "acl",  # `consistent hashing to route keys` → drop `to route ...`
        "relcl",  # restrictive relative clauses
        "advcl",  # adverbial clauses
    }
)


# Prepositions whose subtrees stay inside the object surface — these
# are argument-style PPs (instrument / agent / source / about-ness) that
# the eval gold treats as part of the noun phrase, not the qualifier:
# `rerank candidates with BGE` -> obj=`candidates with BGE`;
# `support re-indexing by chunk id` -> obj=`re-indexing by chunk id`;
# `refuse connections from peers` -> obj=`connections from peers`.
_OBJECT_KEEP_PREPS: frozenset[str] = frozenset({"with", "by", "from", "about"})


class SpacyTier2SVOExtractor:
    """Tier-2 SVO extractor backed by a spaCy dependency parser + NER.

    Implements the `ClaimExtractor` Protocol:

        def extract(self, sentence: str) -> list[ClaimTuple]: ...

    The spaCy pipeline is loaded on the first `extract` call so that
    construction is cheap and importing this module does not pull spaCy.
    """

    def __init__(self, *, config: Tier2Config | None = None) -> None:
        self._config = config or Tier2Config()
        self._nlp: Any | None = None

    def _ensure_nlp(self) -> Any:
        if self._nlp is None:
            import spacy  # type: ignore[import-untyped,import-not-found,unused-ignore]

            self._nlp = spacy.load(self._config.spacy_model)
        return self._nlp

    def extract(self, sentence: str) -> list[ClaimTuple]:
        if not sentence or not sentence.strip():
            return []
        nlp = self._ensure_nlp()
        doc = nlp(sentence)
        tuples: list[ClaimTuple] = []
        seen: set[tuple[str, str, str, PolarityLiteral, ModalityLiteral, str]] = set()
        for sent in doc.sents:
            for claim in self._claims_from_sentence(sent):
                key = (
                    claim.subject,
                    claim.predicate,
                    claim.object,
                    claim.polarity,
                    claim.modality,
                    claim.qualifier,
                )
                if key in seen:
                    continue
                seen.add(key)
                tuples.append(claim)
        return tuples

    # --- claim construction ------------------------------------------------

    def _claims_from_sentence(self, sent: Span) -> list[ClaimTuple]:
        out: list[ClaimTuple] = []
        for verb, copula in _iter_verb_heads(sent):
            for subj in _subject_tokens_for(verb, copula):
                obj_token = _select_object_token(verb)
                claim = self._build_claim(
                    sent=sent,
                    verb=verb,
                    copula=copula,
                    subj=subj,
                    obj_token=obj_token,
                )
                if claim is not None:
                    out.append(claim)
        return out

    def _build_claim(
        self,
        *,
        sent: Span,
        verb: Token,
        copula: Token | None,
        subj: Token,
        obj_token: Token | None,
    ) -> ClaimTuple | None:
        subject = _subject_surface(subj)
        if not subject:
            return None

        # Modal token window: aux/neg of the verb + aux/neg of the copula
        # + any leading `if` / `when` / `unless` mark from the sentence
        # prefix or an ancestor's children.
        modal_tokens = _modal_token_window(verb, sent, copula=copula)
        modality = classify_modality(tokens=modal_tokens)
        polarity = classify_polarity(tokens=modal_tokens)
        polarity, modality = merge_modality_with_polarity(polarity=polarity, modality=modality)

        # Predicate.
        predicate = self._predicate_surface(verb=verb, copula=copula, modality=modality, subj=subj)

        # Object selection differs by copula sub-case. When the predicate
        # is just the copula alone (`is` / `are` for attr or non-participle
        # acomp), the object is the verb-token's own subtree (the
        # adjective phrase / attribute NP). When the predicate combines
        # copula + participle ("are normalised"), the object remains the
        # downstream NP via the standard path. Plain verb roots use the
        # standard `_select_object_token` path.
        is_pred_copula_only = (
            copula is not None
            and predicate == copula.text.lower()
            and verb.dep_ in ("acomp", "attr")
        )
        if is_pred_copula_only:
            obj_text = _adjective_phrase_surface(verb)
            obj_prep_tokens: set[int] = set()
        else:
            obj_text, obj_prep_tokens = _object_surface(
                obj_token=obj_token,
                verb=verb,
                max_tokens=self._config.max_object_tokens,
            )

        qualifier = _qualifier_surface(
            verb=verb,
            copula=copula,
            obj_token=obj_token,
            obj_prep_tokens=obj_prep_tokens,
        )

        return ClaimTuple(
            subject=subject,
            predicate=predicate,
            object=obj_text,
            polarity=polarity,
            modality=modality,
            qualifier=qualifier,
        )

    def _predicate_surface(
        self,
        *,
        verb: Token,
        copula: Token | None,
        modality: ModalityLiteral,
        subj: Token,
    ) -> str:
        if copula is not None and verb.dep_ in ("acomp", "attr"):
            # Copula construction: `are normalised`, `is append-only`,
            # `writes are linearizable`. We distinguish three sub-cases:
            #   * attr child -> predicate is the copula alone (`is`).
            #   * acomp ADJ that is NOT a -ed/-en participle -> the
            #     adjective is the object; predicate is the copula.
            #   * acomp VERB or -ed/-en participle -> predicate joins
            #     copula + participle ("are normalised").
            cop_text: str = str(copula.text).lower()
            if verb.dep_ == "attr":
                return cop_text
            surface: str = str(verb.text).lower()
            is_participle = (
                verb.pos_ == "VERB"
                or surface.endswith(("ed", "en"))
                or (surface.endswith("d") and str(verb.lemma_).endswith("e"))
            )
            if not is_participle:
                # `are linearizable`, `is eventually consistent` -> pred "are" / "is".
                return cop_text
            head_text: str = str(verb.lemma_) or surface
            head_norm = _passive_participle(head_text, raw=str(verb.text))
            combined = f"{cop_text} {head_norm}".strip()
            return combined.lower()

        # Detect a passive construction on a plain verb root:
        # `is exchanged`, `is owned`, `is marked`, `are deprecated`,
        # `shall be resolved` (modal+passive periphrasis).
        # spaCy tags these with an `auxpass` child whose lemma is `be`.
        auxpass: Token | None = None
        for child in verb.children:
            if child.dep_ == "auxpass":
                auxpass = child
                break
        if auxpass is not None:
            surface = verb.text.lower()
            participle = (
                surface
                if surface.endswith(("ed", "en"))
                else _passive_participle(verb.lemma_ or surface, raw=verb.text)
            )
            # Modal-passive periphrasis: `shall be resolved` ->
            # gold uses `are resolved` / `is resolved` (modal absorbed
            # by the modality slot, copula in subject-agreement form).
            # Pure passive: `is owned` -> keep `auxpass.text` ("is" / "are").
            plural = _is_plural_subject(subj)
            cop_form = "are" if plural else "is"
            cop_surface = (
                cop_form
                if auxpass.lemma_ == "be" and auxpass.text.lower() == "be"
                else auxpass.text.lower()
            )
            return f"{cop_surface} {participle}".strip()

        # Plain verb root.
        # Preserve original tense when the verb is morphologically past:
        # narrative sentences use `announced` / `complained` / `had completed`
        # which the gold mirrors. Otherwise apply subject-verb agreement
        # on the present tense.
        tense = verb.morph.get("Tense", []) if hasattr(verb, "morph") else []
        if "Past" in tense:
            return _past_tense_predicate(verb)
        plural = _is_plural_subject(subj)
        return lemmatize_predicate(
            verb.lemma_ or verb.text,
            modality=modality,
            subject_is_plural=plural,
        )


# ---------------------------------------------------------------------------
# Verb-head selection
# ---------------------------------------------------------------------------


def _iter_verb_heads(sent: Span) -> list[tuple[Token, Token | None]]:
    """Iterate `(predicate_token, copula_or_none)` pairs for one sentence.

    The predicate token carries the verb meaning; the copula token, when
    present, supplies the auxiliary surface for the combined predicate.

    Patterns supported:
        * Plain root verb -> (root, None)
        * Copular root (`is`/`are`/`was`/`were`) with `acomp`/`attr`
          child -> (acomp_or_attr, copula_root)
        * Coordinated `conj` peers of the root, recursively decomposed
          so a conj'd copula yields its own (acomp, copula) pair
        * Modal-as-ROOT misparse: `Proxies SHALL forward ...` — spaCy
          sometimes places the modal at ROOT with the lexical verb as
          an advmod / amod child. We pick the most-verb-like child as
          the predicate-bearing token in that case.
    """
    pairs: list[tuple[Token, Token | None]] = []
    seen_indices: set[int] = set()

    def _add(predicate: Token, copula: Token | None) -> None:
        if predicate.i in seen_indices:
            return
        seen_indices.add(predicate.i)
        pairs.append((predicate, copula))

    def _decompose(tok: Token) -> None:
        # Modal-as-ROOT misparse: the AUX (`SHALL` / `MUST`) carries
        # the lexical verb as an advmod / amod / xcomp child.
        if tok.pos_ == "AUX" and tok.lemma_ not in {"be", "have", "do"}:
            for child in tok.children:
                if child.dep_ in ("advmod", "amod", "xcomp", "ccomp") and child.pos_ in (
                    "VERB",
                    "ADV",
                ):
                    _add(child, None)
                    return
            _add(tok, None)
            return
        if tok.lemma_ in {"be", "have"} and tok.pos_ in ("AUX", "VERB"):
            acomp_or_attr: Token | None = None
            for child in tok.children:
                if child.dep_ in ("acomp", "attr"):
                    acomp_or_attr = child
                    break
            if acomp_or_attr is not None:
                _add(acomp_or_attr, tok)
            else:
                _add(tok, None)
            return
        _add(tok, None)

    for tok in sent:
        if tok.dep_ != "ROOT":
            continue
        _decompose(tok)
        for child in tok.children:
            if child.dep_ == "conj" and child.pos_ in ("VERB", "AUX"):
                _decompose(child)
    return pairs


def _subject_tokens_for(verb: Token, copula: Token | None) -> list[Token]:
    """Find the subject tokens (`nsubj` / `nsubjpass`) for a predicate.

    For copula predicates the subject is attached to the copula root;
    for plain verbs it's attached to the verb itself. Coordinated
    subjects via `conj` are flattened so each subject yields its own
    claim.
    """
    out: list[Token] = []
    target = copula if copula is not None else verb
    for child in target.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            out.append(child)
            # Flatten coordinated subjects (`writes and reads`).
            for grandchild in child.children:
                if grandchild.dep_ == "conj":
                    out.append(grandchild)
    if not out and verb.i != target.i:
        # Sometimes a coordinated predicate's nsubj sits on the verb.
        for child in verb.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                out.append(child)
    return out


def _select_object_token(verb: Token) -> Token | None:
    for dep in _OBJECT_DEPS:
        for child in verb.children:
            if child.dep_ == dep:
                return child
    # Passive agent: `is resolved [by arbitration]` -> obj="by arbitration".
    for child in verb.children:
        if child.dep_ == "agent":
            return child
    # Last-resort: an xcomp/acl one step deeper.
    for child in verb.children:
        if child.dep_ in ("xcomp", "acl"):
            for grandchild in child.children:
                if grandchild.dep_ in _OBJECT_DEPS:
                    return grandchild
    # No direct object: use the first prep child as a stand-in. The
    # eval gold treats `retry [on 503 responses]` as obj="on 503
    # responses" — the PP IS the object when nothing else fills that
    # role.
    for child in verb.children:
        if child.dep_ == "prep":
            return child
    return None


# ---------------------------------------------------------------------------
# Surface-form helpers
# ---------------------------------------------------------------------------


def _subject_surface(subj: Token) -> str:
    """Build the canonical subject surface form from a subject token.

    The subject phrase is the noun-phrase span rooted at `subj`,
    bounded so we drop trailing relative clauses and adverbial clauses
    while keeping hyphenated compounds (`cross-doc edges`) intact.
    """
    indices = sorted({t.i for t in subj.subtree})
    if not indices:
        return ""
    # Drop trailing relcl / acl / advcl subtrees so the noun phrase is
    # tight: "the user that shares" -> "the user".
    keep: list[int] = []
    for i in indices:
        tok = subj.doc[i]
        if tok.dep_ in ("relcl", "acl", "advcl") and tok.i > subj.i:
            # If we hit a trailing modifier subtree, stop including
            # tokens at or past it.
            break
        keep.append(i)
    if not keep:
        return ""
    doc = subj.doc
    start = min(keep)
    end = max(keep) + 1
    text = doc[start:end].text
    return _normalise_phrase(text)


def _object_surface(
    *, obj_token: Token | None, verb: Token, max_tokens: int
) -> tuple[str, set[int]]:
    """Build the object surface form. Returns `(text, prep_token_indices)`.

    `prep_token_indices` is the set of token indices that belong to a
    `prep` subtree we excluded from the object — the qualifier extractor
    consumes those to compose the qualifier surface.
    """
    if obj_token is None:
        return "", set()

    doc = obj_token.doc
    keep_indices: set[int] = set()
    prep_indices: set[int] = set()

    for tok in obj_token.subtree:
        # Drop any token under a trailing acl / relcl / advcl subtree.
        if _ancestor_dep_below(tok, obj_token, _OBJECT_TRAILING_DROP):
            continue
        # Drop conj-coordinated objects — they get their own tuple via
        # the conj loop below, not via this subtree.
        if _ancestor_dep_below(tok, obj_token, frozenset({"conj"})):
            continue
        # Track every prep subtree under the object. Argument-style
        # preps (`with`, `by`, `from`, `of`) stay inside the object
        # surface when their head is a *direct* child of the object
        # head — instrument / agent / source PPs that the eval gold
        # treats as part of the NP (`candidates [with BGE]`,
        # `re-indexing [by chunk id]`). Locative / temporal preps
        # (`in`, `to`, `on`, `under`, `across`, ...) and any deeply
        # nested PP (chains of `on X with Y`) move into the qualifier
        # slot.
        if _ancestor_dep_below(tok, obj_token, frozenset({"prep"})):
            prep_head = _find_prep_head(tok, obj_token)
            keep_inside = (
                prep_head is not None
                and prep_head.text.lower() in _OBJECT_KEEP_PREPS
                and prep_head.head.i == obj_token.i
            )
            if keep_inside:
                keep_indices.add(tok.i)
                continue
            prep_indices.add(tok.i)
            continue
        keep_indices.add(tok.i)

    if not keep_indices:
        return "", prep_indices

    indices = sorted(keep_indices)
    if len(indices) > max_tokens:
        indices = indices[:max_tokens]

    # Take the contiguous prefix starting at the earliest kept token —
    # this preserves hyphenation by relying on `Doc.text` for whitespace.
    start = indices[0]
    end = indices[-1] + 1
    contiguous: list[int] = []
    for i in range(start, end):
        if i in keep_indices:
            contiguous.append(i)
        else:
            break
    if not contiguous:
        return "", prep_indices
    text = doc[contiguous[0] : contiguous[-1] + 1].text
    return _normalise_phrase(text), prep_indices


def _qualifier_surface(
    *,
    verb: Token,
    copula: Token | None,
    obj_token: Token | None,
    obj_prep_tokens: set[int],
) -> str:
    """Compose the qualifier surface form.

    Preference order:
        1. `advcl` clause hanging off the verb or copula (`if ...` /
           `when ...` conditionals + temporal clauses).
        2. Prep / nmod subtrees that were dropped from the object (these
           are the trailing PPs attached to the dobj NP — gold treats
           them as qualifier).
        3. Prep children of the verb that are NOT in the object subtree.
        4. `npadvmod` / `advmod` adverbials.
    """
    # 1. advcl clause
    for owner in (verb, copula):
        if owner is None:
            continue
        for child in owner.children:
            if child.dep_ == "advcl":
                txt = _subtree_text(child)
                if txt:
                    return _normalise_qualifier(txt)

    # 2. prep subtrees dropped from the object — recover them by
    # finding the *topmost* prep heads whose subtree was dropped. We
    # emit each prep at the shallowest level only so nested chains
    # (`in the State [of Delaware]`) are not double-counted.
    if obj_token is not None and obj_prep_tokens:
        prep_pieces: list[str] = []
        for tok in obj_token.subtree:
            if tok.dep_ != "prep" or tok.i not in obj_prep_tokens:
                continue
            # Skip if any ancestor prep (between tok and obj_token) is
            # also a dropped prep — the ancestor's subtree text already
            # covers this one.
            ancestor = tok.head
            is_nested = False
            while ancestor.i != obj_token.i and ancestor.head.i != ancestor.i:
                if ancestor.dep_ == "prep" and ancestor.i in obj_prep_tokens:
                    is_nested = True
                    break
                ancestor = ancestor.head
            if is_nested:
                continue
            prep_pieces.append(_subtree_text(tok))
        if prep_pieces:
            return _normalise_qualifier(" ".join(prep_pieces))

    # 3. prep children of the verb (not the object).
    prep_pieces = []
    for owner in (verb, copula):
        if owner is None:
            continue
        for child in owner.children:
            if child.dep_ != "prep":
                continue
            if obj_token is not None and child in obj_token.subtree:
                continue
            prep_pieces.append(_subtree_text(child))
    if prep_pieces:
        return _normalise_qualifier(" ".join(prep_pieces))

    # 4. adverbials
    for child in verb.children:
        if child.dep_ in ("npadvmod", "advmod"):
            text = _subtree_text(child)
            low = text.lower().strip()
            if low and low not in {"never", "not", "n't", "no"}:
                return _normalise_qualifier(text)

    # 5. xcomp purpose clause: `use forced tool calls [to drop the
    # playbook layer]` -> qualifier="to drop the playbook layer".
    for child in verb.children:
        if child.dep_ == "xcomp":
            # The xcomp aux is a `to` particle; include it so the
            # qualifier reads as "to ...".
            xcomp_indices = sorted({t.i for t in child.subtree if not t.is_punct})
            if not xcomp_indices:
                continue
            doc = child.doc
            start = xcomp_indices[0]
            end = xcomp_indices[-1] + 1
            text = doc[start:end].text
            return _normalise_qualifier(text)

    return ""


# ---------------------------------------------------------------------------
# Auxiliary scanning
# ---------------------------------------------------------------------------


def _modal_token_window(verb: Token, sent: Span, *, copula: Token | None) -> list[str]:
    """Collect tokens that participate in modality / polarity detection.

    Order matters because `classify_modality` returns the first matching
    cue. Sentence-prefix conditionals (`if` / `when` / `unless`) come
    first so they dominate downstream `may` / `can` cues — a conditional
    construction reads as `hypothetical` even when the main clause has
    a permission modal.
    """
    tokens: list[str] = []
    # Conditional prefix scan first — these win over deontic modals.
    for tok in sent:
        if tok.i >= verb.i:
            break
        low = tok.text.lower()
        if low in {"if", "when", "unless", "whenever"}:
            tokens.append(tok.text)
    # Then the verb / copula's own aux / neg / mark children.
    for owner in (verb, copula):
        if owner is None:
            continue
        for child in owner.children:
            if child.dep_ in ("aux", "auxpass", "neg", "mark"):
                tokens.append(child.text)
    return tokens


# ---------------------------------------------------------------------------
# Dependency-tree utilities
# ---------------------------------------------------------------------------


def _ancestor_dep_below(token: Token, root: Token, deps: frozenset[str]) -> bool:
    """True iff some ancestor of `token` (strictly below `root`) has
    a `dep_` in `deps`. Used to detect tokens that live inside a
    subtree we want to exclude from the object's surface form.

    Identity comparison via `Token.i` — spaCy creates fresh Token
    wrappers on every attribute access so `is` comparisons are unsafe.
    """
    if token.i == root.i:
        return False
    cur = token
    while cur.i != root.i:
        if cur.dep_ in deps:
            return True
        head = cur.head
        if head.i == cur.i:
            # Reached the sentence root before reaching `root`; the
            # walk leaves `root`'s subtree.
            return False
        cur = head
    return False


def _find_prep_head(token: Token, root: Token) -> Token | None:
    """Walk up from `token` until we find an ancestor with `dep_ == 'prep'`
    (strictly below `root`). Returns the prep-head token or None."""
    if token.i == root.i:
        return None
    cur = token
    while cur.i != root.i:
        if cur.dep_ == "prep":
            return cur
        head = cur.head
        if head.i == cur.i:
            return None
        cur = head
    return None


def _prep_attaches_to(prep_tok: Token, target: Token) -> bool:
    """True iff `prep_tok` (head of a prep phrase) attaches directly
    to `target`. Compares by token index for safety against spaCy's
    non-identity token wrappers."""
    return bool(prep_tok.head.i == target.i)


def _subtree_text(token: Token) -> str:
    """Span-style text covering `token` and every descendant, with
    punctuation stripped for cleaner gold-comparison surfaces."""
    indices = sorted({t.i for t in token.subtree if not t.is_punct})
    if not indices:
        return ""
    doc = token.doc
    # Use the doc's text between the min/max indices to preserve
    # hyphenation and spacing.
    start = indices[0]
    end = indices[-1] + 1
    return str(doc[start:end].text)


# ---------------------------------------------------------------------------
# Predicate normalisation
# ---------------------------------------------------------------------------


def _adjective_phrase_surface(verb: Token) -> str:
    """Return the surface form of a copular complement adjective / NP.

    Used when the predicate collapses to the copula alone (`writes are
    linearizable`, `the verdict ledger is append-only`). We take the
    `verb` token's full subtree minus any prep / advcl children — those
    belong in the qualifier slot.
    """
    indices: list[int] = []
    for tok in verb.subtree:
        if _ancestor_dep_below(tok, verb, frozenset({"prep", "advcl", "relcl", "acl"})):
            continue
        if tok.is_punct and tok.text in ".,;:!?":
            continue
        indices.append(tok.i)
    if not indices:
        return ""
    indices.sort()
    doc = verb.doc
    start = indices[0]
    end = indices[-1] + 1
    return _normalise_phrase(doc[start:end].text)


def _is_plural_subject(subj: Token) -> bool:
    """True iff `subj` is morphologically plural.

    Falls back to inspecting the head noun (skipping the determiner) so
    `[the] systems` reads off the plural marker on `systems`. spaCy's
    POS tagger labels plural nouns `NNS` / `NNPS` and singular nouns
    `NN` / `NNP` — we match either.
    """
    # Direct morphological feature first.
    number = subj.morph.get("Number", [])
    if number:
        return "Plur" in number
    # Tag-based fallback for the small English pipeline (its morph
    # features sometimes come back empty for compound subjects).
    if subj.tag_ in {"NNS", "NNPS"}:
        return True
    if subj.tag_ in {"NN", "NNP"}:
        return False
    # Pronoun-driven plurals (`we`, `they`).
    if subj.lemma_ in {"we", "they"}:
        return True
    if subj.lemma_ in {"i", "he", "she", "it"}:
        return False
    return False


def _past_tense_predicate(verb: Token) -> str:
    """Render a past-tense verb plus any leading `had` / `has` auxiliary.

    Narrative sentences such as `Anthropic announced its release` or
    `The team had completed every milestone` keep the past tense in the
    eval gold (`announced`, `had completed`). We surface them verbatim.

    When the verb carries a `neg` child (`has not commented`), the gold
    drops the perfect aux and emits the past-tense surface only —
    polarity already captures the negation.
    """
    has_neg = any(child.dep_ == "neg" for child in verb.children)
    parts: list[str] = []
    if not has_neg:
        for child in verb.children:
            if child.dep_ == "aux" and child.lemma_ == "have":
                parts.append(child.text.lower())
    parts.append(verb.text.lower())
    return " ".join(parts)


def _passive_participle(lemma: str, *, raw: str) -> str:
    """Map a verb's base lemma + its surface form to the past-participle
    used by the eval gold tuples for copular constructions.

    `lemma="normalise" raw="normalised"` → `"normalised"`. We prefer
    the surface form if it ends in `-ed` / `-en` / `-d` (already a
    participle); otherwise fall back to a regular `+d`/`+ed` form.
    """
    surface = raw.lower()
    if surface.endswith(("ed", "en", "d")):
        return surface
    if lemma.endswith("e"):
        return lemma + "d"
    return lemma + "ed"


# ---------------------------------------------------------------------------
# Phrase normalisation
# ---------------------------------------------------------------------------


def _normalise_phrase(text: str) -> str:
    """Lowercase, collapse internal whitespace, strip trailing sentence
    punctuation. Hyphens in the original are preserved by relying on the
    spaCy `Span.text` (which uses the original whitespace mask)."""
    out = " ".join(text.split()).strip().lower()
    while out and out[-1] in ".,;:!?":
        out = out[:-1].rstrip()
    return out


def _normalise_qualifier(text: str) -> str:
    """Like `_normalise_phrase` but with the conditional `if` rewritten
    to `when` — the eval gold normalises both surface forms to `when`."""
    out = _normalise_phrase(text)
    if out.startswith("if "):
        out = "when " + out[3:]
    return out


__all__ = [
    "TIER2_F1_THRESHOLD",
    "SpacyTier2SVOExtractor",
]
