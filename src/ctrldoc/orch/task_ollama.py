"""Ollama backend for the stateless task primitive.

The tier-1 sub-client. Mirrors `AnthropicTaskClient` (S-061) so the
`TaskClientRouter` can swap between local and frontier transports
without playbook code knowing the difference. Uses the Ollama chat
API with `temperature=0` + a hard `num_predict` cap so JSON-only
outputs stay byte-stable across identical inputs.

The wrapper is a thin transport. The runner (S-060) keeps all parsing
and validation; this module only ships request shape and returns
the model's raw text. The `system` argument arrives byte-identical
across every sub-task in a session (it is `CacheablePrefix.render()`)
— this wrapper preserves that as the first chat-message block.

SPEC-REF: §4.5 (orchestrator, tiered routing — local 7B tier)
"""

from __future__ import annotations

from typing import Any


class OllamaTaskClient:
    """`TaskClient` backed by Qwen2.5-7B-Instruct via a local Ollama service."""

    def __init__(
        self,
        *,
        model: str = "qwen2.5:7b-instruct-q4_K_M",
        host: str = "http://127.0.0.1:11434",
        max_output_tokens: int = 2048,
        temperature: float = 0.0,
        client: Any | None = None,
    ) -> None:
        self._model = model
        self._host = host
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._client: Any | None = client

    def _ensure_client(self) -> Any:
        if self._client is None:
            import ollama

            self._client = ollama.Client(host=self._host)
        return self._client

    def call(self, *, system: str, user: str) -> str:
        client = self._ensure_client()
        response = client.chat(
            model=self._model,
            options={
                "temperature": self._temperature,
                "num_predict": self._max_output_tokens,
            },
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return _extract_text(response)


def _extract_text(response: Any) -> str:
    message = (
        response["message"] if isinstance(response, dict) else getattr(response, "message", {})
    )
    content = message["content"] if isinstance(message, dict) else getattr(message, "content", "")
    return str(content)


__all__ = ["OllamaTaskClient"]
