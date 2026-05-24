"""Pure-Python unit tests for the Tier-2 SVO extractor helpers.

These tests exercise the deterministic modality / polarity / predicate-
normalisation helpers in `ctrldoc.extract.tier2` that do **not** require
spaCy. The spaCy-backed extractor itself is covered by a separate
integration test in `test_extract_tier2_spacy.py`.

SPEC-REF: §6.4 (schema co-induction — Tier-2 SVO extraction)
"""

from __future__ import annotations

import pytest

from ctrldoc.eval.claim_extraction import ClaimTuple
from ctrldoc.extract.tier2 import (
    MODAL_LEXICON,
    NEGATION_TOKENS,
    Tier2Config,
    classify_modality,
    classify_polarity,
    lemmatize_predicate,
    merge_modality_with_polarity,
)

# --- MODAL_LEXICON shape -----------------------------------------------------


def test_modal_lexicon_covers_every_modal_keyword() -> None:
    # `never` is intentionally absent — it flips polarity (via
    # `NEGATION_TOKENS`) but does NOT promote the modality to
    # `prohibited` on its own. Prohibited is reserved for negated
    # obligation modals (`shall not` / `must not`) and the lexical
    # `forbidden` / `prohibited` / `cannot` surface forms.
    expected_keys = {
        "must",
        "shall",
        "required",
        "should",
        "ought",
        "recommended",
        "may",
        "can",
        "allowed",
        "permitted",
        "optional",
        "forbidden",
        "prohibited",
        "cannot",
        "could",
        "would",
        "might",
        "if",
        "when",
        "unless",
    }
    assert expected_keys <= set(MODAL_LEXICON)


def test_modal_lexicon_values_are_canonical_modalities() -> None:
    canonical = {
        "asserted",
        "obligatory",
        "recommended",
        "permitted",
        "prohibited",
        "hypothetical",
    }
    for modality in MODAL_LEXICON.values():
        assert modality in canonical


def test_negation_tokens_include_common_negators() -> None:
    assert {"not", "n't", "never", "no"} <= NEGATION_TOKENS


# --- classify_modality ------------------------------------------------------


def test_classify_modality_default_is_asserted() -> None:
    assert classify_modality(tokens=["the", "system", "uses", "hashing"]) == "asserted"


@pytest.mark.parametrize(
    "tokens,expected",
    [
        (["chunks", "must", "carry", "ids"], "obligatory"),
        (["impls", "shall", "validate", "headers"], "obligatory"),
        (["clients", "should", "retry"], "recommended"),
        (["servers", "may", "include", "header"], "permitted"),
        (["claims", "may", "include", "spans"], "permitted"),
        (["packs", "shall", "not", "exceed", "tokens"], "prohibited"),
        (["operator", "must", "not", "skip"], "prohibited"),
        (["servers", "cannot", "reach"], "prohibited"),
        (["if", "the", "residual", "rate", "exceeds"], "hypothetical"),
        (["could", "expand", "into", "europe"], "hypothetical"),
        (["the", "model", "would", "outperform"], "hypothetical"),
    ],
)
def test_classify_modality_handles_canonical_cues(tokens: list[str], expected: str) -> None:
    assert classify_modality(tokens=tokens) == expected


def test_classify_modality_prefers_prohibited_when_negation_follows_obligatory() -> None:
    # `shall not` is prohibited, not obligatory.
    assert classify_modality(tokens=["the", "user", "shall", "not", "share"]) == "prohibited"


# --- classify_polarity ------------------------------------------------------


def test_classify_polarity_default_is_affirmative() -> None:
    assert classify_polarity(tokens=["a", "b", "c"]) == "affirmative"


@pytest.mark.parametrize(
    "tokens,expected",
    [
        (["the", "system", "does", "not", "support"], "negative"),
        (["it", "never", "restarts"], "negative"),
        (["nodes", "cannot", "reach"], "negative"),
        (["servers", "include", "header"], "affirmative"),
    ],
)
def test_classify_polarity_detects_negators(tokens: list[str], expected: str) -> None:
    assert classify_polarity(tokens=tokens) == expected


# --- merge_modality_with_polarity -------------------------------------------


def test_merge_modality_with_polarity_prohibited_forces_negative() -> None:
    pol, mod = merge_modality_with_polarity(polarity="affirmative", modality="prohibited")
    assert pol == "negative"
    assert mod == "prohibited"


def test_merge_modality_with_polarity_keeps_affirmative_assertion() -> None:
    pol, mod = merge_modality_with_polarity(polarity="affirmative", modality="asserted")
    assert pol == "affirmative"
    assert mod == "asserted"


def test_merge_modality_with_polarity_keeps_negative_assertion() -> None:
    pol, mod = merge_modality_with_polarity(polarity="negative", modality="asserted")
    assert pol == "negative"
    assert mod == "asserted"


# --- lemmatize_predicate ----------------------------------------------------


@pytest.mark.parametrize(
    "lemma,expected",
    [
        ("use", "uses"),
        ("carry", "carries"),
        ("run", "runs"),
        ("be", "is"),
        ("have", "has"),
        ("validate", "validates"),
        ("watch", "watches"),
        ("push", "pushes"),
        ("fix", "fixes"),
        ("cache", "caches"),
    ],
)
def test_lemmatize_predicate_normalises_to_third_person_singular(lemma: str, expected: str) -> None:
    # Input is always the verb lemma (spaCy's `Token.lemma_`); output
    # is the third-person-singular form a singular subject expects.
    assert lemmatize_predicate(lemma, modality="asserted") == expected


def test_lemmatize_predicate_uses_bare_form_for_plural_subjects() -> None:
    # Subject-verb agreement: plural subject -> bare form (lemma).
    # "Implementations MUST validate" -> subj is plural -> "validate".
    assert (
        lemmatize_predicate("validate", modality="obligatory", subject_is_plural=True) == "validate"
    )
    # Singular subject keeps the 3SG form.
    assert (
        lemmatize_predicate("validate", modality="obligatory", subject_is_plural=False)
        == "validates"
    )
    # The copula `be` swaps to `are` for plural subjects.
    assert lemmatize_predicate("be", modality="asserted", subject_is_plural=True) == "are"
    assert lemmatize_predicate("be", modality="asserted", subject_is_plural=False) == "is"


# --- Tier2Config ------------------------------------------------------------


def test_tier2_config_defaults_are_safe() -> None:
    cfg = Tier2Config()
    # Default model name is the small English pipeline shipped by spaCy.
    assert cfg.spacy_model == "en_core_web_sm"
    assert cfg.entity_labels  # non-empty default label set


# --- Protocol compliance (no spaCy import) ----------------------------------


def test_tier2_extractor_implements_claim_extractor_protocol() -> None:
    # Sanity check: the spaCy backend module imports without forcing spaCy
    # at module level. The Protocol check goes via duck-typing through the
    # eval module's ClaimExtractor.
    from ctrldoc.extract.tier2_spacy import SpacyTier2SVOExtractor

    # Class object should expose an `extract` method that accepts a string.
    assert hasattr(SpacyTier2SVOExtractor, "extract")


# --- Sentinel claim shape ---------------------------------------------------


def test_claim_tuple_round_trip_through_helpers() -> None:
    # Smoke check that the helper outputs are valid ClaimTuple field values.
    polarity = classify_polarity(tokens=["clients", "should", "retry"])
    modality = classify_modality(tokens=["clients", "should", "retry"])
    polarity, modality = merge_modality_with_polarity(polarity=polarity, modality=modality)
    predicate = lemmatize_predicate("retry", modality=modality, subject_is_plural=True)
    tuple_ = ClaimTuple(
        subject="clients",
        predicate=predicate,
        object="on 503 responses",
        polarity=polarity,
        modality=modality,
    )
    assert tuple_.modality == "recommended"
    assert tuple_.polarity == "affirmative"
    assert tuple_.predicate == "retry"  # plural subject -> bare form
