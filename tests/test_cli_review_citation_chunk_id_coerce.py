"""Tolerant citation_chunk_id parsing for the analytical-review sweep payload.

Qwen2.5-Instruct under Ollama `format="json"` sometimes emits the
citation as a single-element JSON array (`["chunk-id"]`) rather than a
bare string (`"chunk-id"`); the Anthropic backend emits the bare
string. The schema must accept both shapes without losing the field's
single-citation contract — exactly one chunk_id per finding, not many.

Contract: `_SweptFinding.citation_chunk_id` accepts `str | list[str]`
via a Pydantic validator that coerces a single-element list to its
element. A list of zero or two-or-more elements is rejected loudly so
multi-citation drift does not silently collapse to the first id.

SPEC-REF: §6.5
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ctrldoc.cli_review import _SweptFinding, _SweptFindings

pytestmark = pytest.mark.family_robustness


# --- _SweptFinding direct shape ---


def test_anthropic_style_bare_string_citation_parses() -> None:
    finding = _SweptFinding.model_validate(
        {
            "claim": "Assumption X is not stated.",
            "severity": "warn",
            "citation_chunk_id": "chunk-001",
        }
    )
    assert finding.citation_chunk_id == "chunk-001"


def test_qwen_style_single_element_list_citation_coerces_to_string() -> None:
    finding = _SweptFinding.model_validate(
        {
            "claim": "Assumption X is not stated.",
            "severity": "warn",
            "citation_chunk_id": ["chunk-001"],
        }
    )
    assert finding.citation_chunk_id == "chunk-001"
    assert isinstance(finding.citation_chunk_id, str)


def test_multi_element_list_citation_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _SweptFinding.model_validate(
            {
                "claim": "Assumption X is not stated.",
                "severity": "warn",
                "citation_chunk_id": ["chunk-001", "chunk-002"],
            }
        )


def test_empty_list_citation_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _SweptFinding.model_validate(
            {
                "claim": "Assumption X is not stated.",
                "severity": "warn",
                "citation_chunk_id": [],
            }
        )


def test_list_with_non_string_element_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _SweptFinding.model_validate(
            {
                "claim": "Assumption X is not stated.",
                "severity": "warn",
                "citation_chunk_id": [42],
            }
        )


def test_list_of_lists_citation_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _SweptFinding.model_validate(
            {
                "claim": "Assumption X is not stated.",
                "severity": "warn",
                "citation_chunk_id": [["chunk-001"]],
            }
        )


# --- _SweptFindings round-trip via the JSON shape the sweeper consumes ---


def test_swept_findings_mixed_payload_round_trip() -> None:
    payload = {
        "findings": [
            {
                "claim": "missing requirement",
                "severity": "critical",
                "citation_chunk_id": "chunk-a",
            },
            {
                "claim": "ambiguous wording",
                "severity": "warn",
                "citation_chunk_id": ["chunk-b"],
            },
        ]
    }
    parsed = _SweptFindings.model_validate(payload)
    assert [f.citation_chunk_id for f in parsed.findings] == ["chunk-a", "chunk-b"]


def test_swept_findings_rejects_multi_citation_inside_list() -> None:
    payload = {
        "findings": [
            {
                "claim": "missing requirement",
                "severity": "critical",
                "citation_chunk_id": ["chunk-a", "chunk-b"],
            }
        ]
    }
    with pytest.raises(ValidationError):
        _SweptFindings.model_validate(payload)
