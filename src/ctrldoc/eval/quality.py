"""quality_eval — score `CriteriaGenerator` output against an expert checklist.

Each case carries a `doc_type` to feed the generator and a list of
`gold_criteria_texts` the expert produced for that doc type. The
runner asks the generator for criteria, pairs each gold criterion
to its best-match generated criterion via Jaccard token overlap,
and reports the fraction that clear a similarity threshold. Per
§8.2 the threshold is ≥0.85.

SPEC-REF: §8.1 (quality_eval), §8.2 (quality_audit metrics)
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from ctrldoc.eval.harness import EvalResult
from ctrldoc.ops.quality import CriteriaGenerator

CRITERIA_COVERAGE_THRESHOLD = 0.85
DEFAULT_MATCH_THRESHOLD = 0.5


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    intersection = a & b
    union = a | b
    return len(intersection) / len(union)


def criteria_coverage(
    *,
    gold_texts: Iterable[str],
    generated_texts: Iterable[str],
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> float:
    """Fraction of gold criteria with at least one generated match.

    Each gold criterion is paired to its best-Jaccard generated
    criterion; the pair counts as "matched" iff that best similarity
    is ≥ `match_threshold`. Returns 0.0 when there are no gold criteria
    to grade against.
    """
    gold_list = list(gold_texts)
    if not gold_list:
        return 0.0
    generated_tokens = [_tokenize(text) for text in generated_texts]
    matched = 0
    for gold in gold_list:
        gold_tokens = _tokenize(gold)
        best = max((_jaccard(gold_tokens, gen) for gen in generated_tokens), default=0.0)
        if best >= match_threshold:
            matched += 1
    return matched / len(gold_list)


class QualityEvalCase(BaseModel):
    """One row in quality_eval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    tags: list[str] = []
    doc_type: str
    gold_criteria_texts: list[str] = Field(min_length=1)


class QualityEvalRunner:
    """Adapts a `CriteriaGenerator` into a `CaseRunner`."""

    def __init__(
        self,
        *,
        generator: CriteriaGenerator,
        match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    ) -> None:
        self._generator = generator
        self._match_threshold = match_threshold

    def run_case(self, case: QualityEvalCase) -> EvalResult:
        generated = self._generator.generate(case.doc_type)
        coverage = criteria_coverage(
            gold_texts=case.gold_criteria_texts,
            generated_texts=(c.text for c in generated),
            match_threshold=self._match_threshold,
        )
        return EvalResult(
            case_id=case.id,
            passed=coverage >= CRITERIA_COVERAGE_THRESHOLD,
            score=coverage,
            metrics={"criteria_coverage": coverage},
            notes=(
                f"gold={len(case.gold_criteria_texts)}, generated={len(generated)}, "
                f"coverage={coverage:.3f}"
            ),
        )


__all__ = [
    "CRITERIA_COVERAGE_THRESHOLD",
    "DEFAULT_MATCH_THRESHOLD",
    "QualityEvalCase",
    "QualityEvalRunner",
    "criteria_coverage",
]
