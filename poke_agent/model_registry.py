from __future__ import annotations

from pathlib import Path
from typing import Any

from poke_agent.outputs import checkpoint_path, report_path, resolve_checkpoint_path
from poke_agent.model_catalog import (
    ACTIVE_MODEL,
    MODEL_CATALOG,
    NEURAL_OVERRIDE_KEYS,
    TRAIN_MODELS,
    ModelKind,
)
from poke_agent.models.heuristics import HEURISTIC_BUILDERS, HeuristicAgent
from poke_agent.models.hybrid import HybridAgent


class ModelRegistryError(ValueError):
    pass


def list_models(*, kind: ModelKind | None = None) -> list[str]:
    if kind is None:
        return list(MODEL_CATALOG)
    return [name for name, spec in MODEL_CATALOG.items() if spec.get("kind") == kind]


def get_model_spec(model_id: str) -> dict[str, Any]:
    if model_id not in MODEL_CATALOG:
        raise ModelRegistryError(f"Unknown model id: {model_id}")
    return MODEL_CATALOG[model_id]


def validate_catalog() -> None:
    for model_id, spec in MODEL_CATALOG.items():
        kind = spec.get("kind")
        if kind not in {"neural", "heuristic", "hybrid"}:
            raise ModelRegistryError(f"{model_id}: invalid kind {kind!r}")

        if kind == "heuristic":
            strategy = spec.get("strategy")
            if strategy not in HEURISTIC_BUILDERS:
                raise ModelRegistryError(f"{model_id}: unknown heuristic strategy {strategy!r}")

        if kind == "neural":
            if spec.get("architecture", "transformer_rl") != "transformer_rl":
                raise ModelRegistryError(f"{model_id}: unsupported architecture")

        if kind == "hybrid":
            for key in ("neural_model", "heuristic_model", "mode"):
                if key not in spec:
                    raise ModelRegistryError(f"{model_id}: hybrid missing {key}")
            if spec["neural_model"] not in MODEL_CATALOG:
                raise ModelRegistryError(f"{model_id}: unknown neural_model {spec['neural_model']!r}")
            if spec["heuristic_model"] not in MODEL_CATALOG:
                raise ModelRegistryError(f"{model_id}: unknown heuristic_model {spec['heuristic_model']!r}")
            if MODEL_CATALOG[spec["neural_model"]].get("kind") != "neural":
                raise ModelRegistryError(f"{model_id}: neural_model must reference a neural entry")
            if MODEL_CATALOG[spec["heuristic_model"]].get("kind") != "heuristic":
                raise ModelRegistryError(f"{model_id}: heuristic_model must reference a heuristic entry")
            if spec["mode"] not in {"neural_first", "heuristic_first", "fallback"}:
                raise ModelRegistryError(f"{model_id}: invalid hybrid mode {spec['mode']!r}")

    if ACTIVE_MODEL not in MODEL_CATALOG:
        raise ModelRegistryError(f"ACTIVE_MODEL {ACTIVE_MODEL!r} is not in MODEL_CATALOG")

    unknown_train = [name for name in TRAIN_MODELS if name not in MODEL_CATALOG]
    if unknown_train:
        raise ModelRegistryError(f"TRAIN_MODELS contains unknown ids: {unknown_train}")


def describe_catalog(root: Path | None = None) -> list[dict[str, Any]]:
    validate_catalog()
    rows = []
    base = root or Path(".")
    for model_id, spec in MODEL_CATALOG.items():
        checkpoint = spec.get("output_path")
        if not checkpoint and spec.get("kind") == "neural":
            checkpoint = str(checkpoint_path(base, model_id))
        rows.append({
            "id": model_id,
            "kind": spec.get("kind"),
            "description": spec.get("description", ""),
            "active": model_id == ACTIVE_MODEL,
            "train": model_id in TRAIN_MODELS,
            "checkpoint": checkpoint,
            "report": str(report_path(base, model_id)) if spec.get("kind") == "neural" else None,
        })
    return rows


def build_model_config(root: Path, model_id: str, base_config: dict[str, Any] | None = None) -> dict[str, Any]:
    from poke_agent.config import build_config

    spec = get_model_spec(model_id)
    if spec.get("kind") != "neural":
        raise ModelRegistryError(f"{model_id} is not a neural model")

    overrides = {
        key: spec[key]
        for key in NEURAL_OVERRIDE_KEYS
        if key in spec and key not in {"output_path"}
    }
    overrides["model_id"] = model_id
    if spec.get("output_path"):
        overrides["model_output_path"] = spec["output_path"]
    config = build_config(root, overrides or None)
    config["model_id"] = model_id
    config["model_description"] = spec.get("description", "")
    config["architecture"] = spec.get("architecture", "transformer_rl")
    config["model_type"] = spec.get("model_type", "temporal_transformer_rl_complex_loss")
    config["output_path"] = resolve_checkpoint_path(
        root,
        model_id=model_id,
        explicit=spec.get("output_path"),
    )
    config["report_path"] = report_path(root, model_id)
    return config


def build_heuristic_agent(model_id: str, to_observation_class) -> HeuristicAgent:
    spec = get_model_spec(model_id)
    if spec.get("kind") != "heuristic":
        raise ModelRegistryError(f"{model_id} is not a heuristic model")
    builder = HEURISTIC_BUILDERS[spec["strategy"]]
    built = builder(to_observation_class)
    return HeuristicAgent(name=model_id, strategy=built.strategy, fn=built.fn)


def build_hybrid_agent(
    model_id: str,
    *,
    to_observation_class,
    neural_agent: Any | None = None,
) -> HybridAgent:
    spec = get_model_spec(model_id)
    if spec.get("kind") != "hybrid":
        raise ModelRegistryError(f"{model_id} is not a hybrid model")

    heuristic = build_heuristic_agent(spec["heuristic_model"], to_observation_class)
    return HybridAgent(
        name=model_id,
        mode=spec["mode"],
        neural=neural_agent,
        heuristic=heuristic,
        confidence_threshold=float(spec.get("confidence_threshold", 0.0)),
    )
