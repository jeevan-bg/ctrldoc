"""Galois subsumption lattice over the universal claim tuple.

The lattice ranks two claims by logical entailment: `C1 subsumes C2`
iff `C1` is strictly stronger (implies `C2`), `equivalent` iff each
implies the other, `subsumed_by` is the mirror of `subsumes`, and
`incomparable` is a first-class verdict when neither side entails the
other. The deterministic floor reasons on the six universal-tuple
slots (subject / predicate / object / polarity / modality / qualifier)
only — semantic NLI / LLM paths layer on top in later slices and reuse
this floor as their fallback.

`claim_join` is the lattice join (weakest claim both imply, i.e. the
least upper bound in the strength order) and `claim_meet` is the
lattice meet (strongest claim that implies both, i.e. the greatest
lower bound). Both return `None` for incomparable pairs that share no
common weakening / strengthening within the structural floor.

SPEC-REF: §6.3 (Galois lattice for "stronger than")
"""

from __future__ import annotations

import pytest

from ctrldoc.eval.claim_extraction import ClaimTuple
from ctrldoc.extract.galois import (
    SUBSUMPTION_LABELS,
    Subsumption,
    claim_join,
    claim_meet,
    claim_subsumption,
)


def _claim(
    subject: str = "data",
    predicate: str = "is encrypted",
    obj: str = "in transit",
    polarity: str = "affirmative",
    modality: str = "asserted",
    qualifier: str = "",
) -> ClaimTuple:
    return ClaimTuple(
        subject=subject,
        predicate=predicate,
        object=obj,
        polarity=polarity,  # type: ignore[arg-type]
        modality=modality,  # type: ignore[arg-type]
        qualifier=qualifier,
    )


# ---------------------------------------------------------------------------
# Subsumption alphabet & equivalence
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_subsumption_label_alphabet_is_the_four_named_verdicts() -> None:
    """The substrate exposes exactly the four §6.3 verdicts — no more, no less."""
    assert set(SUBSUMPTION_LABELS) == {
        "equivalent",
        "subsumes",
        "subsumed_by",
        "incomparable",
    }
    # Every alphabet member is assignable to the public `Subsumption` literal.
    label: Subsumption
    for label in SUBSUMPTION_LABELS:
        assert label in SUBSUMPTION_LABELS


@pytest.mark.family_determinism
def test_identical_claims_are_equivalent() -> None:
    """Pure reflexivity: a claim is equivalent to itself across every slot."""
    c = _claim(qualifier="aes-256")
    assert claim_subsumption(c, c) == "equivalent"


@pytest.mark.family_determinism
def test_qualifier_text_normalized_before_equivalence_check() -> None:
    """Equivalence is decided after the §6.2 `normalize_text` pipeline."""
    left = _claim(subject="Data", predicate="IS encrypted ", obj="in transit.")
    right = _claim(subject="data", predicate="is  encrypted", obj="in transit")
    assert claim_subsumption(left, right) == "equivalent"


# ---------------------------------------------------------------------------
# Polarity / SVO mismatches → incomparable
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_polarity_flip_is_incomparable_not_subsumes() -> None:
    """A contradiction lives in a different lattice component (§6.3)."""
    affirm = _claim()
    negate = _claim(polarity="negative")
    assert claim_subsumption(affirm, negate) == "incomparable"
    assert claim_subsumption(negate, affirm) == "incomparable"


@pytest.mark.family_determinism
def test_different_subject_is_incomparable_at_the_structural_floor() -> None:
    """Cross-subject entailment is a Tier-2 NLI path, not the floor."""
    left = _claim(subject="data")
    right = _claim(subject="logs")
    assert claim_subsumption(left, right) == "incomparable"


@pytest.mark.family_determinism
def test_different_predicate_is_incomparable() -> None:
    left = _claim(predicate="is encrypted")
    right = _claim(predicate="is signed")
    assert claim_subsumption(left, right) == "incomparable"


@pytest.mark.family_determinism
def test_different_object_is_incomparable() -> None:
    left = _claim(obj="in transit")
    right = _claim(obj="at rest")
    assert claim_subsumption(left, right) == "incomparable"


# ---------------------------------------------------------------------------
# Modality chain — same SVO + polarity
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_obligatory_subsumes_recommended() -> None:
    """MUST implies SHOULD on the deontic chain (§6.3 modality ordering)."""
    must = _claim(modality="obligatory")
    should = _claim(modality="recommended")
    assert claim_subsumption(must, should) == "subsumes"
    assert claim_subsumption(should, must) == "subsumed_by"


@pytest.mark.family_determinism
def test_recommended_subsumes_permitted() -> None:
    """SHOULD implies MAY (a recommendation entails the option)."""
    should = _claim(modality="recommended")
    may = _claim(modality="permitted")
    assert claim_subsumption(should, may) == "subsumes"


@pytest.mark.family_determinism
def test_obligatory_subsumes_permitted_transitively() -> None:
    must = _claim(modality="obligatory")
    may = _claim(modality="permitted")
    assert claim_subsumption(must, may) == "subsumes"


@pytest.mark.family_determinism
def test_asserted_is_incomparable_with_deontic_modalities() -> None:
    """Descriptive (asserted) lives on its own axis; deontic is normative."""
    asserted = _claim(modality="asserted")
    must = _claim(modality="obligatory")
    assert claim_subsumption(asserted, must) == "incomparable"
    assert claim_subsumption(must, asserted) == "incomparable"


@pytest.mark.family_determinism
def test_hypothetical_is_incomparable_with_deontic_chain() -> None:
    """Conditional/counterfactual is its own axis, not a deontic weakening."""
    hyp = _claim(modality="hypothetical")
    should = _claim(modality="recommended")
    assert claim_subsumption(hyp, should) == "incomparable"


@pytest.mark.family_determinism
def test_prohibited_modality_only_compares_within_negative_polarity() -> None:
    """`prohibited` is the obligatory of the negative-polarity branch."""
    must_not = _claim(polarity="negative", modality="prohibited")
    should_not = _claim(polarity="negative", modality="recommended")
    assert claim_subsumption(must_not, should_not) == "subsumes"


# ---------------------------------------------------------------------------
# Qualifier-narrowing partial order
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_unqualified_is_stronger_than_qualified_within_same_modality() -> None:
    """An unqualified universal claim entails its qualified instance.

    "Data is encrypted" (unconditional) is strictly stronger than
    "Data is encrypted at rest" (a scope-narrowing qualifier), because
    the universal claim implies every scoped instance of it.
    """
    universal = _claim(qualifier="")
    scoped = _claim(qualifier="at rest")
    assert claim_subsumption(universal, scoped) == "subsumes"
    assert claim_subsumption(scoped, universal) == "subsumed_by"


@pytest.mark.family_determinism
def test_two_distinct_non_empty_qualifiers_are_incomparable_at_the_floor() -> None:
    """Without semantic reasoning, two scoped qualifiers do not order."""
    left = _claim(qualifier="at rest")
    right = _claim(qualifier="in transit")
    assert claim_subsumption(left, right) == "incomparable"


@pytest.mark.family_determinism
def test_qualifier_and_modality_compose_when_aligned() -> None:
    """Universal + obligatory subsumes scoped + recommended (both axes weaken)."""
    universal_must = _claim(modality="obligatory", qualifier="")
    scoped_should = _claim(modality="recommended", qualifier="for new tenants")
    assert claim_subsumption(universal_must, scoped_should) == "subsumes"


@pytest.mark.family_determinism
def test_modality_strengthens_but_qualifier_narrows_is_incomparable() -> None:
    """Lattice meets axes — cross-axis disagreement collapses to incomparable."""
    must_scoped = _claim(modality="obligatory", qualifier="at rest")
    should_universal = _claim(modality="recommended", qualifier="")
    assert claim_subsumption(must_scoped, should_universal) == "incomparable"


# ---------------------------------------------------------------------------
# Algebraic laws
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_subsumption_is_antisymmetric() -> None:
    """If both `subsumes` and `subsumed_by` hold, the claims are equivalent."""
    left = _claim(modality="obligatory")
    right = _claim(modality="obligatory")
    assert claim_subsumption(left, right) == "equivalent"


@pytest.mark.family_determinism
def test_subsumption_is_transitive_on_modality_chain() -> None:
    must = _claim(modality="obligatory")
    should = _claim(modality="recommended")
    may = _claim(modality="permitted")
    assert claim_subsumption(must, should) == "subsumes"
    assert claim_subsumption(should, may) == "subsumes"
    assert claim_subsumption(must, may) == "subsumes"


# ---------------------------------------------------------------------------
# Join (LUB) — the weakest claim both imply
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_join_of_equivalent_claims_returns_either() -> None:
    c = _claim()
    out = claim_join(c, c)
    assert out is not None
    assert out == c


@pytest.mark.family_determinism
def test_join_of_comparable_pair_returns_the_weaker() -> None:
    must = _claim(modality="obligatory")
    should = _claim(modality="recommended")
    out = claim_join(must, should)
    assert out == should


@pytest.mark.family_determinism
def test_join_universal_and_scoped_returns_scoped() -> None:
    """`universal ⊔ scoped = scoped` — the scoped claim is the weaker one."""
    universal = _claim(qualifier="")
    scoped = _claim(qualifier="at rest")
    assert claim_join(universal, scoped) == scoped


@pytest.mark.family_determinism
def test_join_of_incomparable_pair_is_none() -> None:
    """No common weakening exists at the structural floor."""
    asserted = _claim(modality="asserted")
    must = _claim(modality="obligatory")
    assert claim_join(asserted, must) is None


@pytest.mark.family_determinism
def test_join_of_polarity_flip_is_none() -> None:
    """Contradictions have no lattice join — they live in different components."""
    affirm = _claim()
    negate = _claim(polarity="negative")
    assert claim_join(affirm, negate) is None


# ---------------------------------------------------------------------------
# Meet (GLB) — the strongest claim that implies both
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_meet_of_equivalent_claims_returns_either() -> None:
    c = _claim()
    assert claim_meet(c, c) == c


@pytest.mark.family_determinism
def test_meet_of_comparable_pair_returns_the_stronger() -> None:
    must = _claim(modality="obligatory")
    should = _claim(modality="recommended")
    assert claim_meet(must, should) == must


@pytest.mark.family_determinism
def test_meet_universal_and_scoped_returns_universal() -> None:
    universal = _claim(qualifier="")
    scoped = _claim(qualifier="at rest")
    assert claim_meet(universal, scoped) == universal


@pytest.mark.family_determinism
def test_meet_of_incomparable_pair_is_none() -> None:
    asserted = _claim(modality="asserted")
    must = _claim(modality="obligatory")
    assert claim_meet(asserted, must) is None


@pytest.mark.family_determinism
def test_join_and_meet_round_trip_on_a_comparable_pair() -> None:
    """join(meet) and meet(join) collapse comparable pairs to the original bounds."""
    must = _claim(modality="obligatory")
    may = _claim(modality="permitted")
    j = claim_join(must, may)
    m = claim_meet(must, may)
    assert j == may
    assert m == must
    assert claim_join(j, m) == j
    assert claim_meet(j, m) == m
