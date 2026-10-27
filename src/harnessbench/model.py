from __future__ import annotations

import json
import time
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelResponse:
    text: str
    tokens_in: int
    tokens_out: int
    latency_s: float


class ModelAdapter(ABC):
    name: str

    @abstractmethod
    def call(self, prompt: str) -> ModelResponse: ...


class MockAdapter(ModelAdapter):
    """Deterministic adapter for tests and dry runs — no network calls."""

    def __init__(self, name: str, responder) -> None:
        self.name = name
        self._responder = responder

    def call(self, prompt: str) -> ModelResponse:
        start = time.monotonic()
        text = self._responder(prompt)
        latency_s = time.monotonic() - start
        return ModelResponse(
            text=text,
            tokens_in=len(prompt.split()),
            tokens_out=len(text.split()),
            latency_s=latency_s,
        )


class OpenAICompatibleAdapter(ModelAdapter):
    """Calls any OpenAI-compatible chat completions endpoint.

    Works with providers exposing an OpenAI-shaped `/chat/completions` API
    (OpenRouter, vLLM's own OpenAI-compatible server, Together, etc.) —
    covers most open-weight models (Qwen, DeepSeek, GLM, Kimi, MiniMax,
    GPT-OSS, Devstral) without a dedicated SDK per provider.
    """

    def __init__(self, name: str, *, base_url: str, api_key: str, model: str, timeout_s: float = 60.0) -> None:
        self.name = name
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout_s = timeout_s

    def call(self, prompt: str) -> ModelResponse:
        body = json.dumps(
            {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode()
        request = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        start = time.monotonic()
        with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
            payload = json.loads(response.read())
        latency_s = time.monotonic() - start

        text = payload["choices"][0]["message"]["content"]
        usage = payload.get("usage", {})
        return ModelResponse(
            text=text,
            tokens_in=usage.get("prompt_tokens", len(prompt.split())),
            tokens_out=usage.get("completion_tokens", len(text.split())),
            latency_s=latency_s,
        )
