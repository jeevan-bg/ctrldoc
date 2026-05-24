"""Anthropic-backed LLM-judge.

The model is instructed to return a single JSON object
`{"passed": bool, "confidence": float, "reasoning": str}`. Confidence
values that exceed `[0, 1]` (a real failure mode under high
temperature) are clamped rather than rejected so the judge never
crashes the verifier on a malformed score.

SPEC-REF: §4.4 (tier-2 LLM-judge)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ctrldoc.verify.json_utils import strip_code_fence
from ctrldoc.verify.judge import JudgeResult

if TYPE_CHECKING:
    from anthropic import Anthropic


_SYSTEM_PROMPT = (
    "You are a strict verifier. Given a CLAIM and a piece of EVIDENCE, "
    "decide whether the evidence supports the claim. Be conservative: if "
    "the evidence does not directly support the claim, mark it as failed. "
    "Return one JSON object only, of the form:\n"
    '  {"passed": <bool>, "confidence": <0.0-1.0>, "reasoning": <short str>}\n'
    "No prose, no code fences, no commentary outside the JSON."
)


class AnthropicLLMJudge:
    """Anthropic Messages-backed LLM-judge."""

    def __init__(
        self,
        *,
        model: str = "claude-opus-4-7",
        max_output_tokens: int = 256,
        client: Anthropic | None = None,
    ) -> None:
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._client = client

    def _ensure_client(self) -> Anthropic:
        if self._client is None:
            from anthropic import Anthropic

            self._client = Anthropic()
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
        message = client.messages.create(
            model=self._model,
            max_tokens=self._max_output_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_payload}],
        )
        return _parse_result(_extract_text(message))


def _extract_text(message: object) -> str:
    parts: list[str] = []
    for block in getattr(message, "content", []) or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts).strip()


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


__all__ = ["AnthropicLLMJudge"]
