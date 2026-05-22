"""Contract tests for `PlaybookOutput[T]`.

Every playbook returns one of these and only one of these. Provenance
is mandatory and the wrapper is frozen so the audit metadata cannot
be detached from the result downstream.

SPEC-REF: §4.0 (PlaybookOutput), §4.7 (provenance on every output)
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ctrldoc.models import Finding, PlaybookOutput, Span
from ctrldoc.provenance import Provenance


def _provenance() -> Provenance:
    return Provenance.create(
        playbook="qa",
        playbook_version="0.1.0",
        index_hash="sha256:" + "0" * 64,
        models={"planner": "claude-opus-4-7"},
    )


def _span() -> Span:
    return Span(chunk_id="c1", char_start=0, char_end=5, text="hello")


def test_field_set() -> None:
    assert set(PlaybookOutput.model_fields) == {"provenance", "result"}


def test_provenance_required() -> None:
    with pytest.raises(ValidationError):
        PlaybookOutput.model_validate({"result": "ok"})


def test_str_result() -> None:
    out: PlaybookOutput[str] = PlaybookOutput(provenance=_provenance(), result="ok")
    assert out.result == "ok"


def test_dict_result_round_trip() -> None:
    payload = {"answer": "X", "claims_verified": 4}
    out: PlaybookOutput[dict] = PlaybookOutput(provenance=_provenance(), result=payload)
    restored = PlaybookOutput[dict].model_validate(out.model_dump())
    assert restored.result == payload
    assert restored.provenance == out.provenance


def test_list_of_findings_result() -> None:
    findings = [
        Finding(ctrldoc="assumptions", location=_span(), claim="x", severity="info"),
        Finding(ctrldoc="boundary_cases", location=_span(), claim="y", severity="warn"),
    ]
    out: PlaybookOutput[list[Finding]] = PlaybookOutput(provenance=_provenance(), result=findings)
    restored = PlaybookOutput[list[Finding]].model_validate(out.model_dump())
    assert restored.result == findings


def test_frozen_mutation_rejected() -> None:
    out: PlaybookOutput[str] = PlaybookOutput(provenance=_provenance(), result="ok")
    with pytest.raises(ValidationError):
        out.result = "tampered"  # type: ignore[misc]


def test_extra_fields_rejected() -> None:
    out: PlaybookOutput[str] = PlaybookOutput(provenance=_provenance(), result="ok")
    payload = out.model_dump()
    payload["bogus"] = "no"
    with pytest.raises(ValidationError):
        PlaybookOutput[str].model_validate(payload)


def test_typed_result_validates_payload() -> None:
    # PlaybookOutput[int] should reject a non-int result.
    with pytest.raises(ValidationError):
        PlaybookOutput[int].model_validate(
            {"provenance": _provenance().model_dump(), "result": "not-an-int"}
        )
