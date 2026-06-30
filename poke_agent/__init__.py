"""Poke agent training pipeline extracted from notebooks/poke_agent_unified.ipynb."""

from poke_agent.config import build_config, default_user_config
from poke_agent.outputs import describe_layout, ensure_output_layout

__all__ = ["build_config", "default_user_config", "describe_layout", "ensure_output_layout", "main"]


def __getattr__(name: str):
    if name == "main":
        from poke_agent.main import main

        return main
    raise AttributeError(name)
