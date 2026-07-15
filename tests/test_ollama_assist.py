from __future__ import annotations

from poke_agent.assist.ollama_client import OllamaClient, ollama_from_config


def test_ollama_from_config_disabled_when_unset(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    assert ollama_from_config({}) is None
    assert ollama_from_config({"ollama_base_url": ""}) is None


def test_ollama_from_config_reads_config():
    client = ollama_from_config({"ollama_base_url": "http://blackwell:11434/"})
    assert client is not None
    assert client.base_url == "http://blackwell:11434"
    assert client.enabled


def test_ollama_from_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    client = ollama_from_config(None)
    assert client is not None
    assert client.base_url == "http://localhost:11434"


def test_ollama_health_and_chat(monkeypatch):
    client = OllamaClient("http://example.invalid:11434")
    calls: list[tuple[str, str]] = []

    def fake_request(method: str, path: str, payload=None):
        calls.append((method, path))
        if path == "/api/tags":
            return {"models": [{"name": "qwen3.6"}]}
        return {"message": {"content": "hello"}}

    monkeypatch.setattr(client, "_request", fake_request)
    assert client.health()["models"][0]["name"] == "qwen3.6"
    assert client.chat("qwen3.6", "hi") == "hello"
    assert ("GET", "/api/tags") in calls
    assert ("POST", "/api/chat") in calls
