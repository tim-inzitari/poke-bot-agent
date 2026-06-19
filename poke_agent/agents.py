from __future__ import annotations

from typing import Any

from poke_agent.model_catalog import ACTIVE_MODEL
from poke_agent.model_registry import (
    build_heuristic_agent,
    build_hybrid_agent,
    build_model_config,
    get_model_spec,
    validate_catalog,
)


def resolve_agent(
    model_id: str,
    *,
    to_observation_class,
    neural_agent: Any | None = None,
):
    validate_catalog()
    spec = get_model_spec(model_id)
    kind = spec.get("kind")

    if kind == "heuristic":
        return build_heuristic_agent(model_id, to_observation_class)
    if kind == "hybrid":
        return build_hybrid_agent(
            model_id,
            to_observation_class=to_observation_class,
            neural_agent=neural_agent,
        )
    if kind == "neural":
        if neural_agent is None:
            raise ValueError(
                f"Neural catalog entry {model_id!r} needs a trained policy agent. "
                "Pass neural_agent=... after loading a checkpoint."
            )
        return neural_agent

    raise ValueError(f"Unsupported model kind: {kind!r}")


def resolve_active_agent(*, to_observation_class, neural_agent: Any | None = None):
    return resolve_agent(ACTIVE_MODEL, to_observation_class=to_observation_class, neural_agent=neural_agent)


def active_model_training_config(root, base_config: dict[str, Any] | None = None):
    spec = get_model_spec(ACTIVE_MODEL)
    if spec.get("kind") != "neural":
        raise ValueError(f"ACTIVE_MODEL {ACTIVE_MODEL!r} is not neural; cannot derive training config")
    return build_model_config(root, ACTIVE_MODEL, base_config)
