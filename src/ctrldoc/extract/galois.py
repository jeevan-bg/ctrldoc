"""Galois subsumption lattice over the universal claim tuple.

A claim `C1` is *stronger than* `C2` (`C1 ⊑ C2`) when `C1` logically
entails `C2`. The four-verdict alphabet from SPEC §6.3 — `subsumes`,
`subsumed_by`, `equivalent`, `incomparable` — partitions every claim
pair into the lattice positions that the optimal-transport core
(§6.6) needs to weight cross-doc edges.

This module ships the **structural floor**: a deterministic
pure-function judge that reasons only on the six universal-tuple
slots (subject / predicate / object / polarity / modality /
qualifier) from §6.2. Semantic equivalence beyond surface form —
paraphrase, lexical entailment, predicate alignment — is the Tier-2
NLI/LLM path; those layers consult this floor first and only escalate
when it returns `incomparable`.

Three axes drive the ordering inside a same-SVO, same-polarity pair:

1. **Modality** lives on three orthogonal axes:

   - the deontic chain `obligatory ⊐ recommended ⊐ permitted` (RFC-2119
     ordering — `MUST` implies `SHOULD` implies `MAY`),
   - the negative-deontic chain `prohibited ⊐ recommended ⊐ permitted`
     (negative-polarity branch where `MUST NOT` is the strongest),
   - the singleton axes `asserted` (descriptive) and `hypothetical`
     (conditional). Cross-axis pairs are `incomparable` — descriptive
     and normative force never order at the structural floor.

2. **Qualifier** is set-style. An empty qualifier denotes the
   universal claim; any non-empty qualifier denotes a scope-narrowed
   instance. The universal strictly subsumes the scoped (because the
   universal entails every scoped variant). Two distinct non-empty
   qualifiers do not order — that is a semantic question for the
   NLI/LLM path.

3. **Polarity** flips contradict; contradicting claims live in
   different lattice components and are always `incomparable` at this
   layer (§6.3's first-class incomparable verdict).

The lattice operations `claim_join` (least upper bound: weakest claim
both imply) and `claim_meet` (greatest lower bound: strongest claim
that implies both) return `None` when the pair has no common
weakening / strengthening at the structural floor.

SPEC-REF: §6.3 (Galois lattice for "stronger than"), §6.2 (universal claim tuple)
"""

from __future__ import annotations

from typing import Literal, TypeAlias

from ctrldoc.eval.claim_extraction import (
    ClaimTuple,
    ModalityLiteral,
    normalize_text,
)

Subsumption: TypeAlias = Literal["equivalent", "subsumes", "subsumed_by", "incomparable"]
"""The four §6.3 lattice verdicts. `subsumes` means *left* is strictly stronger."""

SUBSUMPTION_LABELS: tuple[Subsumption, ...] = (
    "equivalent",
    "subsumes",
    "subsumed_by",
    "incomparable",
)
"""Public alphabet — consumers iterate this rather than re-listing the literal."""


# A modality's axis-and-rank: claims on different axes never order;
# within an axis the higher rank is the stronger claim. `asserted` and
# `hypothetical` sit on singleton axes (rank zero) — same-axis pairs
# collapse to equivalence, cross-axis pairs to incomparable.
_DEONTIC: str = "deontic"
_PROHIBITIVE: str = "prohibitive"
_DESCRIPTIVE: str = "descriptive"
_CONDITIONAL: str = "conditional"

_MODALITY_AXIS: dict[ModalityLiteral, tuple[str, int]] = {
    # Deontic chain — affirmative-polarity normative force.
    "obligatory": (_DEONTIC, 3),
    "recommended": (_DEONTIC, 2),
    "permitted": (_DEONTIC, 1),
    # Prohibitive chain — negative-polarity normative force. `recommended`
    # and `permitted` are reused with the polarity discriminator chosen
    # by the caller; the chain shape is the mirror of the deontic one
    # with `prohibited` as the strongest rung.
    "prohibited": (_PROHIBITIVE, 3),
    # Singleton axes — descriptive and conditional never order with the
    # normative chains; they collapse to equivalence with themselves.
    "asserted": (_DESCRIPTIVE, 0),
    "hypothetical": (_CONDITIONAL, 0),
}


def _modality_axis(modality: ModalityLiteral, polarity: str) -> tuple[str, int]:
    """Return the (axis, rank) tuple a modality lives on for the given polarity.

    `recommended` and `permitted` are deontic under affirmative polarity
    and prohibitive under negative polarity — the chain shape is
    identical, only the axis label differs so that cross-polarity pairs
    refuse to order even when the modality literals coincide.
    """
    axis, rank = _MODALITY_AXIS[modality]
    if polarity == "negative" and axis == _DEONTIC:
        return (_PROHIBITIVE, rank)
    return (axis, rank)


def _modality_cmp(left: ClaimTuple, right: ClaimTuple) -> int | None:
    """Compare modalities only — returns +1 / 0 / -1, or None if incomparable.

    Polarity is bundled into the axis decision so that two claims with
    different polarities never order at this layer (the caller has
    already short-circuited polarity mismatches, but the axis-mapping
    keeps the function correct in isolation).
    """
    la, lr = _modality_axis(left.modality, left.polarity)
    ra, rr = _modality_axis(right.modality, right.polarity)
    if la != ra:
        return None
    if lr == rr:
        return 0
    return 1 if lr > rr else -1


def _qualifier_cmp(left: ClaimTuple, right: ClaimTuple) -> int | None:
    """Empty qualifier is strictly stronger; equal non-empty qualifiers tie.

    Two distinct non-empty qualifiers are `incomparable` — the floor
    refuses to guess which scope narrows the other without semantic
    reasoning.
    """
    lq = normalize_text(left.qualifier)
    rq = normalize_text(right.qualifier)
    if lq == rq:
        return 0
    if lq == "":
        return 1  # universal (left) is stronger than scoped (right)
    if rq == "":
        return -1
    return None


def _svo_equal(left: ClaimTuple, right: ClaimTuple) -> bool:
    """Surface-form equality on subject / predicate / object after normalization."""
    return (
        normalize_text(left.subject) == normalize_text(right.subject)
        and normalize_text(left.predicate) == normalize_text(right.predicate)
        and normalize_text(left.object) == normalize_text(right.object)
    )


def claim_subsumption(left: ClaimTuple, right: ClaimTuple) -> Subsumption:
    """Decide the §6.3 partial-order relation between two universal tuples.

    Returns one of `equivalent`, `subsumes` (left strictly stronger),
    `subsumed_by` (left strictly weaker), or `incomparable`. Pure
    function — no I/O, no state, deterministic on equal inputs.
    """
    if not _svo_equal(left, right):
        return "incomparable"
    if left.polarity != right.polarity:
        return "incomparable"

    mod = _modality_cmp(left, right)
    if mod is None:
        return "incomparable"
    qual = _qualifier_cmp(left, right)
    if qual is None:
        return "incomparable"

    if mod == 0 and qual == 0:
        return "equivalent"
    # A consistent direction across both axes is needed for ordering.
    if mod >= 0 and qual >= 0 and (mod > 0 or qual > 0):
        return "subsumes"
    if mod <= 0 and qual <= 0 and (mod < 0 or qual < 0):
        return "subsumed_by"
    return "incomparable"


def claim_join(left: ClaimTuple, right: ClaimTuple) -> ClaimTuple | None:
    """Least upper bound: the weakest claim that both `left` and `right` imply.

    For comparable pairs the join is the weaker of the two operands.
    For equivalent pairs either operand is returned (`left` chosen for
    determinism). Returns `None` for `incomparable` pairs — at the
    structural floor there is no common weakening, and the caller
    should escalate to the Tier-2 NLI/LLM path.
    """
    verdict = claim_subsumption(left, right)
    if verdict == "equivalent":
        return left
    if verdict == "subsumes":
        return right
    if verdict == "subsumed_by":
        return left
    return None


def claim_meet(left: ClaimTuple, right: ClaimTuple) -> ClaimTuple | None:
    """Greatest lower bound: the strongest claim that implies both operands.

    Mirror of `claim_join` — returns the stronger of two comparable
    operands, either operand for equivalent ones, and `None` for
    incomparable pairs.
    """
    verdict = claim_subsumption(left, right)
    if verdict == "equivalent":
        return left
    if verdict == "subsumes":
        return left
    if verdict == "subsumed_by":
        return right
    return None


__all__ = [
    "SUBSUMPTION_LABELS",
    "Subsumption",
    "claim_join",
    "claim_meet",
    "claim_subsumption",
]
