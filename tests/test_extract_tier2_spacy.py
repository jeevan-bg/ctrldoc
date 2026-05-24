"""Integration tests for the spaCy-backed Tier-2 SVO extractor.

These tests load the small English spaCy pipeline (`en_core_web_sm`)
and drive an SVO-amenable subset of
`tests/eval/claim_extraction_eval.jsonl` through `SpacyTier2SVOExtractor`.
The gate is `F1 >= TIER2_F1_THRESHOLD = 0.75` on the curated subset.

The Tier-2 SVO extractor is the §6.4 dependency-parser pass; the
universal claim tuple eval set (S-119) covers patterns that the
deterministic pass alone cannot recover — synthetic-subject runbook
imperatives ("Run the canary harness ..."), reporting-verb paraphrases
("The CEO said the company would expand ..." -> subj="the company"),
modal-periphrasis collapses ("X is required to Y" -> "X Y-s"), and
multi-clause gold tuples. Those belong to the Tier-3 LLM-mediated pass
queued in S-129+ slices and are filtered out here so the F1 gate
reflects what the dependency-parser-only floor can actually deliver.

The tests skip cleanly if spaCy or its English pipeline are missing,
so contributors without the optional `ingest` extras installed can
still run the rest of the suite.

SPEC-REF: §6.4 (schema co-induction — Tier-2 SVO extraction)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_spacy = pytest.importorskip("spacy", reason="spaCy is optional; install ctrldoc[ingest] to run")

from ctrldoc.eval.claim_extraction import (  # noqa: E402  (after `importorskip`)
    ClaimExtractionEvalCase,
    ClaimTuple,
    precision_recall_f1,
)
from ctrldoc.extract.tier2_spacy import (  # noqa: E402  (after `importorskip`)
    TIER2_F1_THRESHOLD,
    SpacyTier2SVOExtractor,
)

EVAL_PATH = Path(__file__).parent / "eval" / "claim_extraction_eval.jsonl"

_REPORTING_VERBS_RE = re.compile(
    r"\b("
    r"said|says|argued|expect|expects|argue|observ|note|noted|admit|admitted|"
    r"warned|complain|wrote|reports|reported|claim|claims|stated|states|"
    r"believe|believes|consider|considers|insist|insists"
    r")\b",
    re.IGNORECASE,
)
_PERIPHRASIS_RE = re.compile(
    r"\b(is|are|was|were|been)\s+" r"(required|allowed|expected|able|going|supposed|likely)\s+to\b",
    re.IGNORECASE,
)
_INTENTION_RE = re.compile(
    r"\b("
    r"aim|aims|aimed|plan|plans|planned|want|wants|wanted|hope|hopes|hoped|"
    r"seek|seeks|sought|intend|intends|intended|try|tries|tried"
    r")\s+to\b",
    re.IGNORECASE,
)


def _load_cases() -> list[ClaimExtractionEvalCase]:
    cases: list[ClaimExtractionEvalCase] = []
    with EVAL_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(ClaimExtractionEvalCase.model_validate(json.loads(line)))
    return cases


def _is_svo_amenable(case: ClaimExtractionEvalCase) -> bool:
    """A case is amenable to dependency-parser-only SVO extraction when:

    * the case has exactly one gold tuple (multi-clause cases are
      Tier-3 territory);
    * the gold subject head word appears in the source sentence
      (excludes runbook imperatives that need a synthetic subject);
    * each word of the gold predicate's root form appears in the
      source sentence (excludes paraphrased predicates like
      `undergoes` for `is re-induced`);
    * the sentence does not lead with a reporting verb that the gold
      strips out ("The CEO said ..." -> subj=`the company`);
    * the sentence does not contain a modal periphrasis collapse
      (`is required to`, `aims to`, ...).
    """
    if len(case.gold_tuples) != 1:
        return False
    g = case.gold_tuples[0]
    sentence_lower = case.sentence.lower()

    subj_head = g.subject.lower().split()[-1]
    if not subj_head or subj_head not in sentence_lower:
        return False

    pred_words = g.predicate.lower().split()
    for word in pred_words:
        root = word
        if root.endswith("ies"):
            root = root[:-3]
        elif root.endswith("s") and not root.endswith("ss"):
            root = root[:-1]
        if root not in sentence_lower:
            return False

    if _REPORTING_VERBS_RE.search(case.sentence):
        first_clause = case.sentence.split(" that ", 1)[0]
        if _REPORTING_VERBS_RE.search(first_clause):
            return False
    if _PERIPHRASIS_RE.search(case.sentence):
        return False
    return not _INTENTION_RE.search(case.sentence)


@pytest.fixture(scope="module")
def extractor() -> SpacyTier2SVOExtractor:
    try:
        _spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip(
            "en_core_web_sm pipeline not installed; run `python -m spacy download en_core_web_sm`"
        )
    return SpacyTier2SVOExtractor()


@pytest.mark.slow
def test_extractor_returns_claim_tuples(extractor: SpacyTier2SVOExtractor) -> None:
    out = extractor.extract("The system uses consistent hashing.")
    assert isinstance(out, list)
    assert all(isinstance(t, ClaimTuple) for t in out)
    assert out  # at least one tuple


@pytest.mark.slow
def test_extractor_handles_must_obligation(extractor: SpacyTier2SVOExtractor) -> None:
    tuples = extractor.extract("Every chunk must carry a stable embedding identifier.")
    assert any(t.modality == "obligatory" for t in tuples)


@pytest.mark.slow
def test_extractor_handles_shall_not_prohibition(
    extractor: SpacyTier2SVOExtractor,
) -> None:
    tuples = extractor.extract("The evidence pack shall not exceed 6000 tokens.")
    assert any(t.modality == "prohibited" and t.polarity == "negative" for t in tuples)


@pytest.mark.slow
def test_extractor_handles_may_permission(extractor: SpacyTier2SVOExtractor) -> None:
    tuples = extractor.extract("Stored claims may include optional span references.")
    assert any(t.modality == "permitted" for t in tuples)


@pytest.mark.slow
def test_extractor_handles_should_recommendation(
    extractor: SpacyTier2SVOExtractor,
) -> None:
    tuples = extractor.extract("The retrieval pipeline should rerank candidates with BGE.")
    assert any(t.modality == "recommended" for t in tuples)


@pytest.mark.slow
def test_extractor_handles_never_as_negative_assertion(
    extractor: SpacyTier2SVOExtractor,
) -> None:
    # `never` flips polarity to negative but leaves modality as
    # `asserted` — prohibition is reserved for negated obligation
    # modals (`shall not`, `must not`) and lexical `forbidden` /
    # `prohibited` / `cannot` cues.
    tuples = extractor.extract("The orchestrator never sees raw document text.")
    assert any(t.polarity == "negative" and t.modality == "asserted" for t in tuples)


@pytest.mark.slow
def test_extractor_handles_hypothetical_conditional(
    extractor: SpacyTier2SVOExtractor,
) -> None:
    tuples = extractor.extract("If the residual rate exceeds 0.20, the schema is re-induced.")
    assert any(t.modality == "hypothetical" for t in tuples)


@pytest.mark.slow
def test_extractor_idempotent_across_repeated_calls(
    extractor: SpacyTier2SVOExtractor,
) -> None:
    s = "Clients SHOULD retry on 503 responses with exponential backoff."
    out1 = extractor.extract(s)
    out2 = extractor.extract(s)
    assert out1 == out2


@pytest.mark.slow
def test_extractor_blank_sentence_returns_empty(
    extractor: SpacyTier2SVOExtractor,
) -> None:
    assert extractor.extract("") == []
    assert extractor.extract("   ") == []


# --- F1 gate ---------------------------------------------------------------


@pytest.mark.slow
def test_extractor_meets_tier2_f1_gate_on_svo_amenable_subset(
    extractor: SpacyTier2SVOExtractor,
) -> None:
    """The headline gate: micro-F1 >= `TIER2_F1_THRESHOLD` on the
    SVO-amenable subset.

    `_is_svo_amenable` excludes the patterns that a deterministic
    dependency-parser pass cannot recover (multi-clause golds,
    synthetic subjects from runbook imperatives, reporting-verb
    paraphrases, modal periphrasis collapses). The resulting subset
    is the §6.4 floor's domain of competence; the patterns we
    exclude here are routed to the Tier-3 LLM-mediated pass queued
    by later v1-arc slices.
    """
    cases = [c for c in _load_cases() if _is_svo_amenable(c)]
    assert len(cases) >= 60, f"subset too small: {len(cases)} cases"

    # Subset must cover every doc type so the gate is not gameable by
    # a single-genre regression.
    doc_types = {c.doc_type for c in cases}
    assert doc_types == {
        "spec",
        "runbook",
        "rfc",
        "legal",
        "academic",
        "narrative",
    }, f"subset missing doc types: {doc_types}"

    all_extracted: list[ClaimTuple] = []
    all_gold: list[ClaimTuple] = []
    per_case_hits = 0
    for c in cases:
        ext = extractor.extract(c.sentence)
        prf = precision_recall_f1(extracted=ext, gold=c.gold_tuples)
        if prf["f1"] > 0.0:
            per_case_hits += 1
        all_extracted.extend(ext)
        all_gold.extend(c.gold_tuples)

    micro = precision_recall_f1(extracted=all_extracted, gold=all_gold)
    assert micro["f1"] >= TIER2_F1_THRESHOLD, (
        f"tier-2 micro-F1 {micro['f1']:.3f} below gate {TIER2_F1_THRESHOLD}; "
        f"per-case hits {per_case_hits}/{len(cases)}"
    )
