"""Unified client for local (LM Studio / Ollama) and cloud (Google Gemini / OpenAI) LLM endpoints.

Primary: local LLM (for zero-cost private inference on developer machines).
Fallback: cloud LLM (automatic fallback when local endpoint is down, e.g. in cloud deployment).
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from footyvision.config import get_settings
from footyvision.llm import local_embedder

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when the LLM endpoint is unreachable or returns an error."""


class LLMClient:
    # Which model actually produced the last embeddings. The local->cloud fallback
    # means this is only knowable after the call, and the index records it so a
    # later query encoded by a different model can be caught instead of trusted.
    last_embed_model: str = "unknown"

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        embed_model: str | None = None,
        cloud_base_url: str | None = None,
        cloud_model: str | None = None,
        cloud_api_key: str | None = None,
        cloud_embed_model: str | None = None,
    ) -> None:
        s = get_settings()
        self.base_url = (base_url or s.llm_base_url).rstrip("/")
        self.model = model or s.llm_model
        self.api_key = api_key or s.llm_api_key
        self.embed_model = embed_model or s.llm_embed_model
        self.embed_document_prefix = s.llm_embed_document_prefix
        self.embed_query_prefix = s.llm_embed_query_prefix
        self.timeout = timeout

        self.cloud_base_url = (cloud_base_url or s.cloud_llm_base_url).rstrip("/")
        self.cloud_model = cloud_model or s.cloud_llm_model
        self.cloud_embed_model = cloud_embed_model or s.cloud_llm_embed_model
        self.cloud_api_key = cloud_api_key or s.active_cloud_api_key

    def is_local_up(self, timeout: float = 1.5) -> bool:
        """Check if local endpoint is reachable quickly."""
        if (
            not self.base_url
            or "localhost" not in self.base_url
            and "127.0.0.1" not in self.base_url
        ):
            return False
        try:
            r = httpx.get(f"{self.base_url}/models", timeout=timeout)
            return r.status_code == 200
        except Exception:
            return False

    def is_cloud_up(self, timeout: float = 3.0) -> bool:
        """Check if cloud provider is configured and reachable."""
        if not self.cloud_api_key:
            return False
        try:
            headers = {"Authorization": f"Bearer {self.cloud_api_key}"}
            r = httpx.get(f"{self.cloud_base_url}/models", headers=headers, timeout=timeout)
            return r.status_code == 200
        except Exception:
            return False

    def health(self) -> bool:
        """True if either the local endpoint or the cloud provider is reachable."""
        return self.is_local_up() or self.is_cloud_up()

    def active_info(self) -> dict[str, Any]:
        """Return information on which provider is currently active."""
        if self.is_local_up():
            return {
                "provider": "local",
                "base_url": self.base_url,
                "model": self.model,
            }
        if self.is_cloud_up():
            return {
                "provider": "cloud",
                "base_url": self.cloud_base_url,
                "model": self.cloud_model,
            }
        return {
            "provider": "unreachable",
            "base_url": self.base_url,
            "model": self.model,
        }

    def _post_chat(
        self,
        base_url: str,
        model: str,
        api_key: str,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        url = f"{base_url}/chat/completions"
        payload = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        resp = httpx.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        choice = resp.json()["choices"][0]
        msg = choice.get("message", {})
        content = msg.get("content") or msg.get("reasoning_content") or ""
        return content.strip()

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.4,
        max_tokens: int = 1500,
    ) -> str:
        """Send a system+user prompt, return the assistant's text.

        Tries local LLM first; if unreachable and cloud API key is configured,
        seamlessly falls back to the cloud LLM.
        """
        local_exc: Exception | None = None
        # Only try local if it is localhost or explicitly configured
        if self.base_url and ("localhost" in self.base_url or "127.0.0.1" in self.base_url):
            try:
                return self._post_chat(
                    base_url=self.base_url,
                    model=self.model,
                    api_key=self.api_key,
                    system=system,
                    user=user,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except (
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.ReadTimeout,
                httpx.HTTPStatusError,
            ) as exc:
                local_exc = exc
                if not self.cloud_api_key:
                    raise LLMError(
                        f"Local LLM request to {self.base_url} failed ({exc}). Is LM Studio / Ollama running?"
                    ) from exc
                logger.info(
                    "Local LLM unreachable (%s). Falling back to Cloud LLM (%s)...",
                    exc,
                    self.cloud_model,
                )

        # If base_url is a direct remote URL (e.g. set in Render environment directly)
        elif self.base_url and self.base_url != "http://localhost:1234/v1":
            try:
                return self._post_chat(
                    base_url=self.base_url,
                    model=self.model,
                    api_key=self.api_key,
                    system=system,
                    user=user,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                local_exc = exc

        if self.cloud_api_key:
            self.last_embed_model = self.cloud_embed_model
            try:
                return self._post_chat(
                    base_url=self.cloud_base_url,
                    model=self.cloud_model,
                    api_key=self.cloud_api_key,
                    system=system,
                    user=user,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                raise LLMError(
                    f"Cloud LLM request to {self.cloud_base_url} failed ({exc})."
                ) from exc

        if local_exc:
            raise LLMError(f"LLM request failed ({local_exc}).") from local_exc
        raise LLMError("No LLM endpoint configured or local LLM is unreachable.")

    def _post_embed(
        self, base_url: str, model: str, api_key: str, texts: list[str]
    ) -> list[list[float]]:
        if "generativelanguage.googleapis.com" in base_url:
            candidates = [model, "gemini-embedding-001", "gemini-embedding-2"]
            last_exc: Exception | None = None
            for cand in candidates:
                model_clean = cand.removeprefix("models/")
                embed_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_clean}:batchEmbedContents"
                headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
                requests = [
                    {
                        "model": f"models/{model_clean}",
                        "content": {"parts": [{"text": t}]},
                        "output_dimensionality": 768,
                    }
                    for t in texts
                ]
                try:
                    resp = httpx.post(
                        embed_url, json={"requests": requests}, headers=headers, timeout=self.timeout
                    )
                    if resp.status_code == 200:
                        return [item["values"] for item in resp.json()["embeddings"]]
                    if resp.status_code == 429:
                        continue
                    resp.raise_for_status()
                except Exception as exc:
                    last_exc = exc
                    continue
            if last_exc:
                raise last_exc

        url = f"{base_url}/embeddings"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        resp = httpx.post(
            url, json={"model": model, "input": texts}, headers=headers, timeout=self.timeout
        )
        resp.raise_for_status()
        return [item["embedding"] for item in resp.json()["data"]]

    def embed(self, texts: list[str], kind: str = "document") -> list[list[float]]:
        """Return an embedding vector for each input text.

        A fine-tuned local model wins when one is configured; otherwise the local
        endpoint is tried first with task prefixes, then the cloud model.
        """
        prefix = self.embed_query_prefix if kind == "query" else self.embed_document_prefix

        finetuned = get_settings().finetuned_embed_path
        if local_embedder.is_configured(finetuned):
            self.last_embed_model = f"finetuned:{finetuned}"
            return local_embedder.embed(finetuned, texts, prefix)

        local_exc: Exception | None = None
        if self.base_url and ("localhost" in self.base_url or "127.0.0.1" in self.base_url):
            try:
                prefixed_texts = [prefix + t for t in texts]
                self.last_embed_model = self.embed_model
                return self._post_embed(
                    base_url=self.base_url,
                    model=self.embed_model,
                    api_key=self.api_key,
                    texts=prefixed_texts,
                )
            except (
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.ReadTimeout,
                httpx.HTTPStatusError,
            ) as exc:
                local_exc = exc
                if not self.cloud_api_key:
                    raise LLMError(
                        f"Local embedding request to {self.base_url} failed ({exc})."
                    ) from exc
                logger.info(
                    "Local embedding unreachable (%s). Falling back to Cloud Embeddings (%s)...",
                    exc,
                    self.cloud_embed_model,
                )

        elif self.base_url and self.base_url != "http://localhost:1234/v1":
            try:
                return self._post_embed(
                    base_url=self.base_url,
                    model=self.embed_model,
                    api_key=self.api_key,
                    texts=texts,
                )
            except Exception as exc:
                local_exc = exc

        if self.cloud_api_key:
            try:
                return self._post_embed(
                    base_url=self.cloud_base_url,
                    model=self.cloud_embed_model,
                    api_key=self.cloud_api_key,
                    texts=texts,
                )
            except Exception as exc:
                raise LLMError(
                    f"Cloud embedding request to {self.cloud_base_url} failed ({exc})."
                ) from exc

        if local_exc:
            raise LLMError(f"Embedding request failed ({local_exc}).") from local_exc
        raise LLMError("No embedding endpoint configured.")
