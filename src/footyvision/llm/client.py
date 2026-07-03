"""Thin client for a local, OpenAI-compatible LLM endpoint (LM Studio / Ollama).

Kept provider-agnostic on purpose: any server that speaks the OpenAI
`/v1/chat/completions` shape works by pointing LLM_BASE_URL at it.
"""
from __future__ import annotations

import httpx

from footyvision.config import get_settings


class LLMError(RuntimeError):
    """Raised when the LLM endpoint is unreachable or returns an error."""


class LLMClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        s = get_settings()
        self.base_url = (base_url or s.llm_base_url).rstrip("/")
        self.model = model or s.llm_model
        self.embed_model = s.llm_embed_model
        self.api_key = api_key or s.llm_api_key
        self.timeout = timeout

    def health(self) -> bool:
        """True if the endpoint answers a models listing (server is up)."""
        try:
            r = httpx.get(f"{self.base_url}/models", timeout=5.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.4,
        max_tokens: int = 1500,
    ) -> str:
        """Send a system+user prompt, return the assistant's text.

        Note: reasoning models (e.g. Gemma 3n) spend tokens on hidden `reasoning_content`
        before emitting `content`, so `max_tokens` must be generous or `content` comes back
        empty (truncated mid-thought). Callers that need long answers raise it further.
        """
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(
                f"LLM request to {url} failed ({exc}). Is LM Studio / Ollama running?"
            ) from exc

        try:
            return resp.json()["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"Unexpected LLM response shape: {exc}") from exc

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return an embedding vector for each input text (OpenAI /embeddings shape)."""
        url = f"{self.base_url}/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            resp = httpx.post(
                url, json={"model": self.embed_model, "input": texts},
                headers=headers, timeout=self.timeout,
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Embedding request to {url} failed ({exc}).") from exc
        try:
            return [item["embedding"] for item in resp.json()["data"]]
        except (KeyError, TypeError) as exc:
            raise LLMError(f"Unexpected embeddings response shape: {exc}") from exc
