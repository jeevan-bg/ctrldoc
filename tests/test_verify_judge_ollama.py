"""Integration tests for the Qwen2.5-7B LLM-judge via Ollama.

These tests hit a real `http://127.0.0.1:11434` Ollama instance
with the `qwen2.5:7b-instruct-q4_K_M` model already pulled. They
skip cleanly when the `ollama` SDK is not installed or no Ollama
service is reachable on the loopback port.

SPEC-REF: §4.4 (verifier step 3 — LLM-as-judge, tier-1)
"""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest

pytest.importorskip("ollama", reason="ollama optional; install ctrldoc[models] to run")

from ctrldoc.verify.judge import JudgeResult, LLMJudge
from ctrldoc.verify.judge_ollama import OllamaLLMJudge


def _ollama_reachable() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


pytestmark = pytest.mark.skipif(not _ollama_reachable(), reason="no local Ollama service reachable")


@pytest.fixture(scope="module")
def judge() -> OllamaLLMJudge:
    return OllamaLLMJudge()


@pytest.mark.slow
@pytest.mark.requires_ollama
def test_satisfies_protocol(judge: OllamaLLMJudge) -> None:
    assert isinstance(judge, LLMJudge)


@pytest.mark.slow
@pytest.mark.requires_ollama
def test_supported_claim_passes(judge: OllamaLLMJudge) -> None:
    result = judge.judge(
        claim="Paris is the capital of France.",
        evidence="Paris has been the capital of France since the 10th century.",
    )
    assert isinstance(result, JudgeResult)
    assert result.passed is True
    assert 0.0 <= result.confidence <= 1.0
    assert result.reasoning.strip() != ""


@pytest.mark.slow
@pytest.mark.requires_ollama
def test_unsupported_claim_fails(judge: OllamaLLMJudge) -> None:
    result = judge.judge(
        claim="Berlin is the capital of France.",
        evidence="Paris has been the capital of France since the 10th century.",
    )
    assert result.passed is False
    assert 0.0 <= result.confidence <= 1.0


@pytest.mark.slow
@pytest.mark.requires_ollama
def test_empty_claim_short_circuits(judge: OllamaLLMJudge) -> None:
    result = judge.judge(claim="", evidence="some evidence")
    assert result.passed is False
    assert result.confidence == 0.0


@pytest.mark.slow
@pytest.mark.requires_ollama
def test_empty_evidence_short_circuits(judge: OllamaLLMJudge) -> None:
    result = judge.judge(claim="some claim", evidence="")
    assert result.passed is False
    assert result.confidence == 0.0


@pytest.mark.slow
@pytest.mark.requires_ollama
def test_confidence_in_unit_interval(judge: OllamaLLMJudge) -> None:
    result = judge.judge(
        claim="Water boils at 100 degrees Celsius at sea level.",
        evidence="At standard atmospheric pressure, water boils at 100 °C.",
    )
    assert 0.0 <= result.confidence <= 1.0
