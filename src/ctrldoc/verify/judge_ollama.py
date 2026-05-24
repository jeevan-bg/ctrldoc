"""Qwen2.5-7B-Instruct via Ollama — tier-1 LLM-judge backend.

The model is instructed to return one JSON object
`{"passed": bool, "confidence": float, "reasoning": str}`. The
response is robust to common drift modes (markdown fences,
out-of-range confidence) — fences are stripped before parsing,
confidence is clamped to `[0, 1]`. Kept in its own module so
importing `ctrldoc.verify.judge` does not require the `ollama`
SDK unless the caller wants the production backend.

SPEC-REF: §4.4 (verifier step 3 — LLM-as-judge, tier-1)
"""

from __future__ import annotations

import json
from typing import Any

from ctrldoc.verify.json_utils import strip_code_fence
from ctrldoc.verify.judge import JudgeResult

_SYSTEM_PROMPT = (
    "You are a strict verifier. Given a CLAIM and a piece of EVIDENCE, "
    "decide whether the evidence supports the claim. Be conservative: if "
    "the evidence does not directly support the claim, mark it as failed. "
    "Return one JSON object only, of the form:\n"
    '  {"passed": <bool>, "confidence": <0.0-1.0>, "reasoning": <short str>}\n'
    "No prose, no code fences, no commentary outside the JSON."
)


class OllamaLLMJudge:
    """Qwen2.5-7B-Instruct LLM-judge via a local Ollama service."""

    def __init__(
        self,
        *,
        model: str = "qwen2.5:7b-instruct-q4_K_M",
        host: str = "http://127.0.0.1:11434",
        max_output_tokens: int = 256,
        temperature: float = 0.0,
    ) -> None:
        self._model = model
        self._host = host
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            import ollama

            self._client = ollama.Client(host=self._host)
        return self._client

    def judge(self, claim: str, evidence: str) -> JudgeResult:
        claim_body = claim.strip()
        evidence_body = evidence.strip()
        if not claim_body:
            return JudgeResult(
                passed=False, confidence=0.0, reasoning="empty claim — nothing to judge"
            )
        if not evidence_body:
            return JudgeResult(
                passed=False,
                confidence=0.0,
                reasoning="empty evidence — cannot support the claim",
            )
        client = self._ensure_client()
        user_payload = f"CLAIM:\n{claim_body}\n\nEVIDENCE:\n{evidence_body}"
        response = client.chat(
            model=self._model,
            options={
                "temperature": self._temperature,
                "num_predict": self._max_output_tokens,
            },
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_payload},
            ],
        )
        return _parse_result(_extract_text(response))


def _extract_text(response: Any) -> str:
    message = (
        response["message"] if isinstance(response, dict) else getattr(response, "message", {})
    )
    content = message["content"] if isinstance(message, dict) else getattr(message, "content", "")
    return str(content).strip()


def _parse_result(text: str) -> JudgeResult:
    payload_text = strip_code_fence(text)
    try:
        payload: Any = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM-judge returned non-JSON: {payload_text[:80]!r}") from exc
    if not isinstance(payload, dict):
        raise ValueError("LLM-judge response must be a JSON object")
    for key in ("passed", "confidence", "reasoning"):
        if key not in payload:
            raise ValueError(f"LLM-judge response missing key: {key!r}")
    confidence_raw = payload["confidence"]
    if not isinstance(confidence_raw, int | float):
        raise ValueError("LLM-judge confidence must be a number")
    confidence = max(0.0, min(1.0, float(confidence_raw)))
    return JudgeResult(
        passed=bool(payload["passed"]),
        confidence=confidence,
        reasoning=str(payload["reasoning"]),
    )


__all__ = ["OllamaLLMJudge"]
