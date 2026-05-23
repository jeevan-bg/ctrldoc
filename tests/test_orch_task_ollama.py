"""Qwen2.5-7B-backed `TaskClient` via Ollama.

Mirrors the AnthropicTaskClient design from S-061 so the wrapper
becomes the tier-1 sub-client behind `TaskClientRouter`. Stub-client
tests pin the request shape (chat API messages + temperature=0 +
num_predict cap) hermetically; the real-Ollama integration tests
run against a local `http://127.0.0.1:11434` with
`qwen2.5:7b-instruct-q4_K_M` already pulled and skip cleanly when
that endpoint is not reachable.

SPEC-REF: §4.5 (orchestrator — tiered routing local 7B), §4.7
"""

from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from ctrldoc.assembler import CacheablePrefix
from ctrldoc.orch.task import (
    StatelessTaskRunner,
    TaskClient,
    TaskInput,
)
from ctrldoc.orch.task_ollama import OllamaTaskClient

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


def _prefix() -> CacheablePrefix:
    return CacheablePrefix(
        system_prompt="You are a strict structured-output engine.",
        doc_skeleton="# §1\n\nbody",
        entity_glossary="- **e/1** [concept]",
    )


# --- protocol conformance ---


def test_satisfies_task_client_protocol() -> None:
    client = OllamaTaskClient(client=_StubClient("x"))
    assert isinstance(client, TaskClient)


# --- request shape ---


def test_chat_messages_carry_system_and_user_verbatim() -> None:
    stub = _StubClient("ok")
    OllamaTaskClient(client=stub).call(system="SYS-VERBATIM", user="USER-PAYLOAD")
    call = stub.calls[0]
    assert call["messages"] == [
        {"role": "system", "content": "SYS-VERBATIM"},
        {"role": "user", "content": "USER-PAYLOAD"},
    ]


def test_options_set_temperature_zero_and_num_predict_cap() -> None:
    stub = _StubClient("ok")
    OllamaTaskClient(client=stub, max_output_tokens=512).call(system="s", user="u")
    options = stub.calls[0]["options"]
    assert options["temperature"] == 0.0
    assert options["num_predict"] == 512


def test_options_carry_default_num_ctx_for_long_prefix_prompts() -> None:
    """Ollama defaults `num_ctx` to 2048; we override so the cacheable
    prefix + evidence pack fits without silent truncation."""
    stub = _StubClient("ok")
    OllamaTaskClient(client=stub).call(system="s", user="u")
    assert stub.calls[0]["options"]["num_ctx"] == 16384


def test_custom_num_ctx_forwarded() -> None:
    stub = _StubClient("ok")
    OllamaTaskClient(client=stub, num_ctx=8192).call(system="s", user="u")
    assert stub.calls[0]["options"]["num_ctx"] == 8192


def test_custom_temperature_forwarded() -> None:
    stub = _StubClient("ok")
    OllamaTaskClient(client=stub, temperature=0.2).call(system="s", user="u")
    assert stub.calls[0]["options"]["temperature"] == pytest.approx(0.2)


def test_custom_model_forwarded() -> None:
    stub = _StubClient("ok")
    OllamaTaskClient(client=stub, model="qwen2.5:7b-instruct-q4_K_M").call(system="s", user="u")
    assert stub.calls[0]["model"] == "qwen2.5:7b-instruct-q4_K_M"


def test_identical_systems_produce_byte_identical_message_blocks() -> None:
    """Byte-stable request shape across N calls — same property the
    Anthropic prompt cache relies on. Ollama itself does not cache
    prompts on the server side, but keeping the bytes stable means
    the upstream cache layer (any) still keys identically."""
    stub = _StubClient("ok")
    backend = OllamaTaskClient(client=stub)
    backend.call(system="prefix", user="a")
    backend.call(system="prefix", user="b")
    assert stub.calls[0]["messages"][0] == stub.calls[1]["messages"][0]


# --- response surface ---


def test_call_returns_message_content_string() -> None:
    stub = _StubClient("hello-world")
    out = OllamaTaskClient(client=stub).call(system="s", user="u")
    assert out == "hello-world"


def test_call_returns_object_style_response_content() -> None:
    """ollama python SDK can return either a dict or a typed object."""

    @dataclass
    class _ObjMessage:
        content: str = "object-style-payload"

    @dataclass
    class _ObjResponse:
        message: _ObjMessage

    class _ObjClient:
        def chat(self, **kwargs: Any) -> _ObjResponse:
            return _ObjResponse(message=_ObjMessage())

    out = OllamaTaskClient(client=_ObjClient()).call(system="s", user="u")
    assert out == "object-style-payload"


# --- lazy import ---


def test_lazy_client_construction_does_not_import_when_stub_provided() -> None:
    """If a stub client is supplied the wrapper must not touch the
    `ollama` SDK — keeps the unit-test path hermetic."""
    stub = _StubClient("ok")
    backend = OllamaTaskClient(client=stub)
    backend.call(system="s", user="u")
    assert backend._client is stub  # type: ignore[attr-defined]


# --- end-to-end with StatelessTaskRunner ---


class _Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    label: str
    confidence: float


def test_runner_with_ollama_client_round_trips_a_verdict() -> None:
    stub = _StubClient('{"label": "verified", "confidence": 0.8}')
    backend = OllamaTaskClient(client=stub)
    runner = StatelessTaskRunner(client=backend)
    task = TaskInput(
        prefix=_prefix(),
        evidence_pack="Aurora uses consistent hashing.",
        task_input="Aurora supports consistent hashing.",
    )
    result = runner.run(task, output_model=_Verdict)
    assert result.label == "verified"
    assert result.confidence == pytest.approx(0.8)
    # The system message passed to Ollama is the rendered prefix verbatim.
    sys_block = stub.calls[0]["messages"][0]
    assert sys_block == {"role": "system", "content": task.prefix.render()}


def test_runner_with_ollama_client_unwraps_fenced_json() -> None:
    """Qwen frequently wraps JSON in a ```json``` fence even when
    instructed not to; the runner's `_extract_json` already strips
    fences — verify the integration still produces a clean parse."""
    fenced = '```json\n{"label": "v", "confidence": 0.1}\n```'
    stub = _StubClient(fenced)
    runner = StatelessTaskRunner(client=OllamaTaskClient(client=stub))
    task = TaskInput(
        prefix=_prefix(),
        evidence_pack="x",
        task_input="y",
    )
    result = runner.run(task, output_model=_Verdict)
    assert result.label == "v"
    assert result.confidence == pytest.approx(0.1)


# --- real-Ollama integration tests ---

pytest.importorskip("ollama", reason="ollama optional; install ctrldoc[models] to run")


def _ollama_reachable() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


_OLLAMA_UP = _ollama_reachable()


@pytest.mark.slow
@pytest.mark.requires_ollama
@pytest.mark.skipif(not _OLLAMA_UP, reason="no local Ollama service reachable")
def test_real_ollama_returns_structured_json_via_runner() -> None:
    backend = OllamaTaskClient()
    runner = StatelessTaskRunner(client=backend)
    task = TaskInput(
        prefix=CacheablePrefix(
            system_prompt=(
                "You are a strict JSON-only verifier. Reply with ONE JSON object "
                'of the form {"label": "verified"|"refused", "confidence": <0..1>}. '
                "No prose, no fences."
            ),
            doc_skeleton="",
            entity_glossary="",
        ),
        evidence_pack="Paris has been the capital of France since the 10th century.",
        task_input="Is the claim 'Paris is the capital of France' supported?",
    )
    result = runner.run(task, output_model=_Verdict)
    assert result.label in {"verified", "refused"}
    assert 0.0 <= result.confidence <= 1.0
