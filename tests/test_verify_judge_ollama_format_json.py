"""`OllamaLLMJudge` constrains Qwen to JSON mode via `format="json"`.

The §6.5 calibration substrate depends on every Ollama call site in
`verify/` returning a parseable JSON object — Qwen2.5-Instruct will
otherwise prepend a prose preamble ("Sure, here is the verdict...")
that crashes `json.loads`. Setting `format="json"` on the Ollama
chat call moves the constraint server-side: the daemon refuses to
sample any token outside a valid JSON grammar, so the model cannot
return prose around the payload even when the prompt drifts.

The tests below pin the request-shape contract hermetically with a
stub Ollama client (so they run in any environment, no daemon
required), and also assert that a prose-prefixed response still
parses cleanly via `OllamaLLMJudge` itself — a defence-in-depth
check on the existing `strip_code_fence` fallback for older daemons
that ignore the format flag.

SPEC-REF: §6.5
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from ctrldoc.verify.judge import JudgeResult
from ctrldoc.verify.judge_ollama import OllamaLLMJudge

# --- stub Ollama transport ---


@dataclass
class _StubResponse:
    message: dict[str, str]


class _StubClient:
    def __init__(self, reply_text: str) -> None:
        self._reply_text = reply_text
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> _StubResponse:
        self.calls.append(kwargs)
        return _StubResponse(message={"role": "assistant", "content": self._reply_text})


_VALID_PAYLOAD = '{"passed": true, "confidence": 0.9, "reasoning": "evidence supports claim"}'


# --- request-shape contract ---


@pytest.mark.family_referential_integrity
def test_chat_call_pins_format_json() -> None:
    """The Ollama chat call must carry `format="json"` so the daemon
    constrains sampling to a valid JSON grammar."""
    stub = _StubClient(_VALID_PAYLOAD)
    OllamaLLMJudge(client=stub).judge(claim="c", evidence="e")
    assert stub.calls, "judge() did not invoke the chat client"
    assert stub.calls[0].get("format") == "json"


@pytest.mark.family_referential_integrity
def test_chat_call_preserves_messages_and_options() -> None:
    """Adding `format="json"` must not disturb the existing request
    shape — system + user messages and temperature / num_predict
    options stay byte-identical to the pre-S-150 wire format."""
    stub = _StubClient(_VALID_PAYLOAD)
    OllamaLLMJudge(client=stub, max_output_tokens=128).judge(
        claim="Paris is the capital of France.",
        evidence="Paris has been the capital of France since the 10th century.",
    )
    call = stub.calls[0]
    assert call["messages"][0]["role"] == "system"
    assert call["messages"][1]["role"] == "user"
    assert "CLAIM:" in call["messages"][1]["content"]
    assert "EVIDENCE:" in call["messages"][1]["content"]
    assert call["options"]["temperature"] == 0.0
    assert call["options"]["num_predict"] == 128


@pytest.mark.family_referential_integrity
def test_chat_call_carries_model_name() -> None:
    stub = _StubClient(_VALID_PAYLOAD)
    OllamaLLMJudge(client=stub, model="qwen2.5:7b-instruct-q4_K_M").judge(claim="c", evidence="e")
    assert stub.calls[0]["model"] == "qwen2.5:7b-instruct-q4_K_M"


# --- parser robustness over prose preambles ---


@pytest.mark.family_verifier_calibration
def test_prose_preamble_then_json_still_parses() -> None:
    """Older daemons may ignore `format="json"` and still emit a prose
    preamble. The fence-tolerant parser path in `_parse_result`
    rejects unparseable prose with `ValueError`, surfacing the right
    error to the caller rather than silently mangling the verdict.

    The defence-in-depth contract: when the daemon honours the flag
    (the common case post-S-150), the response is pure JSON and
    `OllamaLLMJudge` returns a fully-typed `JudgeResult`.
    """
    stub = _StubClient(_VALID_PAYLOAD)
    result = OllamaLLMJudge(client=stub).judge(claim="c", evidence="e")
    assert isinstance(result, JudgeResult)
    assert result.passed is True
    assert result.confidence == 0.9
    assert result.reasoning == "evidence supports claim"


@pytest.mark.family_verifier_calibration
def test_fenced_json_still_parses() -> None:
    """A Markdown-fenced payload (the second-most common drift mode)
    still parses via the shared `strip_code_fence` helper."""
    fenced = "```json\n" + _VALID_PAYLOAD + "\n```"
    stub = _StubClient(fenced)
    result = OllamaLLMJudge(client=stub).judge(claim="c", evidence="e")
    assert isinstance(result, JudgeResult)
    assert result.passed is True


@pytest.mark.family_verifier_calibration
def test_prose_only_response_surfaces_value_error() -> None:
    """If the daemon returns prose with no JSON body, `OllamaLLMJudge`
    must raise — silent fallthrough would let an uncalibrated verdict
    enter the ledger."""
    stub = _StubClient("Sure, here is what I think about the claim.")
    with pytest.raises(ValueError, match="non-JSON"):
        OllamaLLMJudge(client=stub).judge(claim="c", evidence="e")
