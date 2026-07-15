"""Optional developer-assist clients (not used for CABT policy or Kaggle)."""

from poke_agent.assist.ollama_client import OllamaClient, ollama_from_config

__all__ = ["OllamaClient", "ollama_from_config"]
