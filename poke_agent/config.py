from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# =============================================================================
# Edit these variables to configure training, data paths, and model settings.
# Used by the notebook, scripts/train_agent.py, and python -m poke_agent.main.
# Environment variables override individual values when set (see _ENV_MAP below).
# =============================================================================

# --- Data paths ---
PRIMARY_ROLLOUT_DATA = "data/mac-rollouts-100k-fullstate.jsonl"
FALLBACK_ROLLOUT_DATA = [
    "data/mac-rollouts-10k.jsonl",
    "data/notebook_rollouts.jsonl",
    "data/kaggle-output/data/cabt_rollouts.jsonl",
    "data/deckpool-smoke.jsonl",
    "data/container-mp-smoke.jsonl",
    "data/container-smoke.jsonl",
]
AGENT_DECK_PATH = (
    "decks/competitive/high_performing/"
    "2026-05_regional-campinas-2026_4th_dragapult-dudunsparce.csv"
)
CABT_GENERATED_PATH = "data/notebook_rollouts.jsonl"
COMPETITION_RESULTS_PATH = "data/competition-results.jsonl"
MODEL_OUTPUT_PATH = "out/value_model.pt"
REQUIRE_CABT_EVAL_DATA = True

# --- Simulation ---
CABT_EPISODES = None  # None = 3 when cg-lib is available, else 0

# --- Features ---
TRANSITION_CLASSES = 8
STATE_HASH_DIM = 256
WINDOW_SIZE = 128

# --- Model ---
MODEL_D_MODEL = 128
MODEL_HEADS = 4
MODEL_LAYERS = 8
MODEL_FF = None  # None = MODEL_D_MODEL * 4
MODEL_DROPOUT = 0.1
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2

# --- Loss weights ---
LOSS_VALUE_WEIGHT = 1.0
LOSS_POLICY_WEIGHT = 0.35
LOSS_DYNAMICS_WEIGHT = 0.15
LOSS_ENTROPY_WEIGHT = 0.01
LOSS_UNCERTAINTY_WEIGHT = 0.02

# --- Training loop ---
TRAIN_EPOCHS = 1000 
EARLY_STOP_PATIENCE = 25
EARLY_STOP_MIN_DELTA = 1e-5
TRAIN_PRINT_EVERY = 100
BATCH_SIZE = 256

# Flat override keys accepted by build_config(..., overrides=...).
OVERRIDE_KEYS = frozenset({
    "primary_rollout_data",
    "fallback_rollout_data",
    "agent_deck_path",
    "cabt_generated_path",
    "competition_results_path",
    "model_output_path",
    "require_cabt_eval_data",
    "cabt_episodes",
    "transition_classes",
    "state_hash_dim",
    "window_size",
    "d_model",
    "model_heads",
    "model_layers",
    "model_ff",
    "model_dropout",
    "learning_rate",
    "weight_decay",
    "loss_value_weight",
    "loss_policy_weight",
    "loss_dynamics_weight",
    "loss_entropy_weight",
    "loss_uncertainty_weight",
    "train_epochs",
    "early_stop_patience",
    "early_stop_min_delta",
    "train_print_every",
    "batch_size",
})


def default_user_config() -> dict[str, Any]:
    """Return editable defaults as a flat dict for notebooks and scripts."""
    return {
        "primary_rollout_data": PRIMARY_ROLLOUT_DATA,
        "fallback_rollout_data": list(FALLBACK_ROLLOUT_DATA),
        "agent_deck_path": AGENT_DECK_PATH,
        "cabt_generated_path": CABT_GENERATED_PATH,
        "competition_results_path": COMPETITION_RESULTS_PATH,
        "model_output_path": MODEL_OUTPUT_PATH,
        "require_cabt_eval_data": REQUIRE_CABT_EVAL_DATA,
        "cabt_episodes": CABT_EPISODES,
        "transition_classes": TRANSITION_CLASSES,
        "state_hash_dim": STATE_HASH_DIM,
        "window_size": WINDOW_SIZE,
        "d_model": MODEL_D_MODEL,
        "model_heads": MODEL_HEADS,
        "model_layers": MODEL_LAYERS,
        "model_ff": MODEL_FF,
        "model_dropout": MODEL_DROPOUT,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "loss_value_weight": LOSS_VALUE_WEIGHT,
        "loss_policy_weight": LOSS_POLICY_WEIGHT,
        "loss_dynamics_weight": LOSS_DYNAMICS_WEIGHT,
        "loss_entropy_weight": LOSS_ENTROPY_WEIGHT,
        "loss_uncertainty_weight": LOSS_UNCERTAINTY_WEIGHT,
        "train_epochs": TRAIN_EPOCHS,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "early_stop_min_delta": EARLY_STOP_MIN_DELTA,
        "train_print_every": TRAIN_PRINT_EVERY,
        "batch_size": BATCH_SIZE,
    }


_ENV_MAP = {
    "primary_rollout_data": "PRIMARY_ROLLOUT_DATA",
    "agent_deck_path": "AGENT_DECK_PATH",
    "cabt_generated_path": "CABT_GENERATED_PATH",
    "competition_results_path": "COMPETITION_RESULTS_PATH",
    "model_output_path": "MODEL_OUTPUT_PATH",
    "cabt_episodes": "CABT_EPISODES",
    "transition_classes": "TRANSITION_CLASSES",
    "state_hash_dim": "STATE_HASH_DIM",
    "window_size": "WINDOW_SIZE",
    "d_model": "MODEL_D_MODEL",
    "model_heads": "MODEL_HEADS",
    "model_layers": "MODEL_LAYERS",
    "model_ff": "MODEL_FF",
    "model_dropout": "MODEL_DROPOUT",
    "learning_rate": "LEARNING_RATE",
    "weight_decay": "WEIGHT_DECAY",
    "loss_value_weight": "LOSS_VALUE_WEIGHT",
    "loss_policy_weight": "LOSS_POLICY_WEIGHT",
    "loss_dynamics_weight": "LOSS_DYNAMICS_WEIGHT",
    "loss_entropy_weight": "LOSS_ENTROPY_WEIGHT",
    "loss_uncertainty_weight": "LOSS_UNCERTAINTY_WEIGHT",
    "train_epochs": "TRAIN_EPOCHS",
    "early_stop_patience": "EARLY_STOP_PATIENCE",
    "early_stop_min_delta": "EARLY_STOP_MIN_DELTA",
    "train_print_every": "TRAIN_PRINT_EVERY",
    "batch_size": "BATCH_SIZE",
}


def _coerce_value(key: str, value: Any) -> Any:
    if value is None:
        return None
    if key == "require_cabt_eval_data":
        if isinstance(value, str):
            return value not in {"0", "false", "False", "no", "No"}
        return bool(value)
    if key in {
        "transition_classes",
        "state_hash_dim",
        "window_size",
        "d_model",
        "model_heads",
        "model_layers",
        "model_ff",
        "train_epochs",
        "early_stop_patience",
        "train_print_every",
        "batch_size",
        "cabt_episodes",
    }:
        return int(value)
    if key in {
        "model_dropout",
        "learning_rate",
        "weight_decay",
        "loss_value_weight",
        "loss_policy_weight",
        "loss_dynamics_weight",
        "loss_entropy_weight",
        "loss_uncertainty_weight",
        "early_stop_min_delta",
    }:
        return float(value)
    if key == "fallback_rollout_data":
        return list(value)
    return value


def _resolve_settings(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = default_user_config()
    for key, env_name in _ENV_MAP.items():
        if env_name in os.environ:
            settings[key] = _coerce_value(key, os.environ[env_name])
    if "REQUIRE_CABT_EVAL_DATA" in os.environ:
        settings["require_cabt_eval_data"] = os.environ["REQUIRE_CABT_EVAL_DATA"] != "0"
    if overrides:
        unknown = set(overrides) - OVERRIDE_KEYS
        if unknown:
            raise ValueError(f"Unknown config overrides: {sorted(unknown)}")
        for key, value in overrides.items():
            settings[key] = _coerce_value(key, value)
    return settings


def build_config(root: Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = _resolve_settings(overrides)
    d_model = settings["d_model"]
    model_ff = settings["model_ff"] if settings["model_ff"] is not None else d_model * 4
    fallback_paths = [root / path for path in settings["fallback_rollout_data"]]

    return {
        "agent_deck_path": root / settings["agent_deck_path"],
        "data_candidates": [root / settings["primary_rollout_data"], *fallback_paths],
        "generated_path": root / settings["cabt_generated_path"],
        "competition_results_path": root / settings["competition_results_path"],
        "output_path": root / settings["model_output_path"],
        "require_cabt_eval_data": settings["require_cabt_eval_data"],
        "cabt_episodes": settings["cabt_episodes"],
        "transition_classes": settings["transition_classes"],
        "state_hash_dim": settings["state_hash_dim"],
        "window_size": settings["window_size"],
        "model": {
            "d_model": d_model,
            "heads": settings["model_heads"],
            "layers": settings["model_layers"],
            "ff": model_ff,
            "dropout": settings["model_dropout"],
            "learning_rate": settings["learning_rate"],
            "weight_decay": settings["weight_decay"],
        },
        "loss": {
            "value": settings["loss_value_weight"],
            "policy": settings["loss_policy_weight"],
            "dynamics": settings["loss_dynamics_weight"],
            "entropy": settings["loss_entropy_weight"],
            "uncertainty": settings["loss_uncertainty_weight"],
        },
        "training": {
            "epochs": settings["train_epochs"],
            "patience": settings["early_stop_patience"],
            "min_delta": settings["early_stop_min_delta"],
            "print_every": settings["train_print_every"],
            "batch_size": settings["batch_size"],
        },
    }


def resolve_cabt_episodes(config: dict[str, Any], simulator_available: bool) -> int:
    configured = config.get("cabt_episodes")
    if configured is not None:
        return max(0, int(configured))
    return 3 if simulator_available else 0
