"""
LLM client interface with a stub and a real Mistral API client.

Switch from stub → real by setting MISTRAL_API_KEY in .env.
"""
from __future__ import annotations

import json
from typing import Iterator, Protocol

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class LLMClient(Protocol):
    """Abstract interface every LLM backend must implement."""

    def generate(self, prompt: str) -> str:
        ...

    def stream(self, prompt: str) -> Iterator[str]:
        ...


# ── Stub (no API key needed) ───────────────────────────────────────────────────

class MistralStub:
    """
    Returns a deterministic mock response.

    Used until a real MISTRAL_API_KEY is configured.
    """

    def generate(self, prompt: str) -> str:
        # Extract the user question from the prompt for a slightly helpful stub
        question_marker = "Question:"
        question = prompt.split(question_marker)[-1].strip() if question_marker in prompt else "your question"
        return (
            f"[STUB RESPONSE — configure MISTRAL_API_KEY to enable real answers]\n\n"
            f"Based on the retrieved context, here is a grounded answer to: \"{question[:80]}\".\n\n"
            "The retrieved documents contain relevant information on this topic. "
            "Please connect to the Mistral API for a real generated response."
        )

    def stream(self, prompt: str) -> Iterator[str]:
        for word in self.generate(prompt).split():
            yield word + " "


# ── Real Mistral API client ────────────────────────────────────────────────────

class MistralAPIClient:
    """
    Calls the Mistral REST API.

    Set MISTRAL_API_KEY, MISTRAL_MODEL, MISTRAL_API_BASE in .env.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key  = settings.mistral_api_key
        self._model    = settings.mistral_model
        self._base_url = settings.mistral_api_base.rstrip("/")
        self._client   = httpx.Client(
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=60,
        )

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 1024,
        }
        response = self._client.post(f"{self._base_url}/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def stream(self, prompt: str) -> Iterator[str]:
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 1024,
            "stream": True,
        }
        with self._client.stream(
            "POST", f"{self._base_url}/chat/completions", json=payload
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except (json.JSONDecodeError, KeyError):
                        continue


# ── Factory ────────────────────────────────────────────────────────────────────

_client_cache: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Return the appropriate LLM client based on configuration."""
    global _client_cache
    if _client_cache is not None:
        return _client_cache

    settings = get_settings()
    if settings.mistral_api_key:
        logger.info("llm_client", backend="mistral_api", model=settings.mistral_model)
        _client_cache = MistralAPIClient()
    else:
        logger.warning("llm_client", backend="stub", reason="no MISTRAL_API_KEY set")
        _client_cache = MistralStub()

    return _client_cache
