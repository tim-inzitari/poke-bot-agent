from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class OllamaClient:
    """Thin HTTP client for a LAN Ollama host (Qwen assist / tooling only)."""

    base_url: str
    timeout_sec: float = 30.0

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")

    @property
    def enabled(self) -> bool:
        return bool(self.base_url)

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("OLLAMA_BASE_URL is not set")
        url = f"{self.base_url}{path}"
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if data is not None else {},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama request failed ({url}): {exc}") from exc
        return json.loads(body) if body else {}

    def health(self) -> dict[str, Any]:
        """Return tag list payload from /api/tags (raises if unreachable)."""
        return self._request("GET", "/api/tags")

    def chat(self, model: str, prompt: str, *, stream: bool = False) -> str:
        """One-shot chat; returns assistant message content."""
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
        }
        result = self._request("POST", "/api/chat", payload)
        message = result.get("message") or {}
        return str(message.get("content", ""))


def ollama_from_config(config: dict[str, Any] | None = None) -> OllamaClient | None:
    """Build a client from config/env, or None when unset."""
    url = ""
    if config is not None:
        url = str(config.get("ollama_base_url") or "")
    if not url.strip():
        url = os.environ.get("OLLAMA_BASE_URL", "")
    url = url.strip()
    if not url:
        return None
    return OllamaClient(base_url=url)
