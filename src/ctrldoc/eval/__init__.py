"""Evaluation harness — drives playbooks against labelled cases."""

from __future__ import annotations

from ctrldoc.eval.claim_extraction import (
    CLAIM_F1_THRESHOLD,
    DOC_TYPES,
    MODALITIES,
    POLARITIES,
    ClaimExtractionEvalCase,
    ClaimExtractionEvalRunner,
    ClaimExtractor,
    ClaimTuple,
    DocTypeLiteral,
    ModalityLiteral,
    PolarityLiteral,
    claim_tuple_matches,
    normalize_text,
    precision_recall_f1,
)
from ctrldoc.eval.harness import (
    CaseRunner,
    EvalReport,
    EvalResult,
    aggregate_results,
    load_jsonl_cases,
    run_eval,
)

__all__ = [
    "CLAIM_F1_THRESHOLD",
    "DOC_TYPES",
    "MODALITIES",
    "POLARITIES",
    "CaseRunner",
    "ClaimExtractionEvalCase",
    "ClaimExtractionEvalRunner",
    "ClaimExtractor",
    "ClaimTuple",
    "DocTypeLiteral",
    "EvalReport",
    "EvalResult",
    "ModalityLiteral",
    "PolarityLiteral",
    "aggregate_results",
    "claim_tuple_matches",
    "load_jsonl_cases",
    "normalize_text",
    "precision_recall_f1",
    "run_eval",
]
