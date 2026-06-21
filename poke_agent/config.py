from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from poke_agent.features import COARSE_FEATURE_DIM
from poke_agent.outputs import describe_layout, ensure_output_layout, report_path, resolve_checkpoint_path

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
    "2026-05_regional-melbourne-2026_10th_mega-lucario.csv"
)
CABT_GENERATED_PATH = "outputs/rollouts/notebook_rollouts.jsonl"
COMPETITION_RESULTS_PATH = "data/competition-results.jsonl"
MODEL_ID = "temporal_current"
MODEL_OUTPUT_PATH = "outputs/checkpoints/temporal_current.pt"
REQUIRE_CABT_EVAL_DATA = True

# --- Dataset size (games) ---
DATASET_GAMES = 100  # None = train on all games in file, skip inline generation
                      # 5000 or 100000 = that many CABT games to generate and/or cap training

# --- Features ---
TRANSITION_CLASSES = 8
STATE_HASH_DIM = 256
WINDOW_SIZE = 256
TENSOR_BUILD_WORKERS = None  # None = cpu_count - 2 (e.g. 30 on a 32-thread CPU)

# --- Model ---
MODEL_D_MODEL = 16
MODEL_HEADS = 4
MODEL_LAYERS = 5
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

# --- Rewards (training labels) ---
VALUE_WIN = 1.0
VALUE_NOT_WIN = -1.0  # loss, draw, timeout

# --- Beam search (Kaggle inference) ---
BEAM_WIDTH = 8
BEAM_SEARCH_DEPTH = 1
BEAM_TIME_BUDGET_MS = 150
BEAM_MIN_REMAINING_SEC = 120

# --- Self-play (AlphaGo-style loop) ---
SELF_PLAY_GAMES = 20
SELF_PLAY_ITERATIONS = 3
SELF_PLAY_EVAL_GAMES = 10
SELF_PLAY_OPPONENT_POOL_SIZE = 5
SELF_PLAY_USE_BEAM = True
SELF_PLAY_OUTPUT_PATH = "outputs/rollouts/self_play_rollouts.jsonl"
SELF_PLAY_CHECKPOINT_DIR = "outputs/checkpoints/self_play"
SELF_PLAY_TRAIN_AFTER_COLLECT = True
SELF_PLAY_FIELD_DECK_DIR = "decks/competitive/high_performing"
SELF_PLAY_MATCHUP_MODE = "sample"  # sample | round-robin
SELF_PLAY_TARGET_RANK = 1000  # eval/bar: win vs opponent decks with placement <= this
SELF_PLAY_TARGET_WIN_RATE = 0.55
SELF_PLAY_PLATEAU_PATIENCE = 3

# --- Training loop ---
TRAIN_EPOCHS = 1000
EARLY_STOP_PATIENCE = 25
EARLY_STOP_MIN_DELTA = 1e-5
TRAIN_PRINT_EVERY = 100
BATCH_GAMES = 4  # games per training batch (each game is one temporal sequence)

# Flat override keys accepted by build_config(..., overrides=...).
OVERRIDE_KEYS = frozenset({
    "primary_rollout_data",
    "fallback_rollout_data",
    "agent_deck_path",
    "cabt_generated_path",
    "competition_results_path",
    "model_output_path",
    "model_id",
    "require_cabt_eval_data",
    "dataset_games",
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
    "batch_games",
    "tensor_build_workers",
    "value_win",
    "value_not_win",
    "beam_width",
    "beam_search_depth",
    "beam_time_budget_ms",
    "beam_min_remaining_sec",
    "self_play_games",
    "self_play_iterations",
    "self_play_eval_games",
    "self_play_opponent_pool_size",
    "self_play_use_beam",
    "self_play_output_path",
    "self_play_checkpoint_dir",
    "self_play_train_after_collect",
    "self_play_field_deck_dir",
    "self_play_matchup_mode",
    "self_play_target_rank",
    "self_play_target_win_rate",
    "self_play_plateau_patience",
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
        "model_id": MODEL_ID,
        "require_cabt_eval_data": REQUIRE_CABT_EVAL_DATA,
        "dataset_games": DATASET_GAMES,
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
        "batch_games": BATCH_GAMES,
        "tensor_build_workers": TENSOR_BUILD_WORKERS,
        "value_win": VALUE_WIN,
        "value_not_win": VALUE_NOT_WIN,
        "beam_width": BEAM_WIDTH,
        "beam_search_depth": BEAM_SEARCH_DEPTH,
        "beam_time_budget_ms": BEAM_TIME_BUDGET_MS,
        "beam_min_remaining_sec": BEAM_MIN_REMAINING_SEC,
        "self_play_games": SELF_PLAY_GAMES,
        "self_play_iterations": SELF_PLAY_ITERATIONS,
        "self_play_eval_games": SELF_PLAY_EVAL_GAMES,
        "self_play_opponent_pool_size": SELF_PLAY_OPPONENT_POOL_SIZE,
        "self_play_use_beam": SELF_PLAY_USE_BEAM,
        "self_play_output_path": SELF_PLAY_OUTPUT_PATH,
        "self_play_checkpoint_dir": SELF_PLAY_CHECKPOINT_DIR,
        "self_play_train_after_collect": SELF_PLAY_TRAIN_AFTER_COLLECT,
        "self_play_field_deck_dir": SELF_PLAY_FIELD_DECK_DIR,
        "self_play_matchup_mode": SELF_PLAY_MATCHUP_MODE,
        "self_play_target_rank": SELF_PLAY_TARGET_RANK,
        "self_play_target_win_rate": SELF_PLAY_TARGET_WIN_RATE,
        "self_play_plateau_patience": SELF_PLAY_PLATEAU_PATIENCE,
    }


_ENV_MAP = {
    "primary_rollout_data": "PRIMARY_ROLLOUT_DATA",
    "agent_deck_path": "AGENT_DECK_PATH",
    "cabt_generated_path": "CABT_GENERATED_PATH",
    "competition_results_path": "COMPETITION_RESULTS_PATH",
    "model_output_path": "MODEL_OUTPUT_PATH",
    "model_id": "MODEL_ID",
    "dataset_games": "DATASET_GAMES",
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
    "dataset_games": "DATASET_GAMES",
    "train_epochs": "TRAIN_EPOCHS",
    "early_stop_patience": "EARLY_STOP_PATIENCE",
    "early_stop_min_delta": "EARLY_STOP_MIN_DELTA",
    "train_print_every": "TRAIN_PRINT_EVERY",
    "batch_games": "BATCH_GAMES",
    "tensor_build_workers": "TENSOR_BUILD_WORKERS",
    "value_win": "VALUE_WIN",
    "value_not_win": "VALUE_NOT_WIN",
    "beam_width": "BEAM_WIDTH",
    "beam_search_depth": "BEAM_SEARCH_DEPTH",
    "beam_time_budget_ms": "BEAM_TIME_BUDGET_MS",
    "beam_min_remaining_sec": "BEAM_MIN_REMAINING_SEC",
    "self_play_games": "SELF_PLAY_GAMES",
    "self_play_iterations": "SELF_PLAY_ITERATIONS",
    "self_play_eval_games": "SELF_PLAY_EVAL_GAMES",
    "self_play_opponent_pool_size": "SELF_PLAY_OPPONENT_POOL_SIZE",
    "self_play_use_beam": "SELF_PLAY_USE_BEAM",
    "self_play_output_path": "SELF_PLAY_OUTPUT_PATH",
    "self_play_checkpoint_dir": "SELF_PLAY_CHECKPOINT_DIR",
    "self_play_train_after_collect": "SELF_PLAY_TRAIN_AFTER_COLLECT",
    "self_play_field_deck_dir": "SELF_PLAY_FIELD_DECK_DIR",
    "self_play_matchup_mode": "SELF_PLAY_MATCHUP_MODE",
    "self_play_target_rank": "SELF_PLAY_TARGET_RANK",
    "self_play_target_win_rate": "SELF_PLAY_TARGET_WIN_RATE",
    "self_play_plateau_patience": "SELF_PLAY_PLATEAU_PATIENCE",
}


def _coerce_value(key: str, value: Any) -> Any:
    if value is None:
        return None
    if key == "require_cabt_eval_data":
        if isinstance(value, str):
            return value not in {"0", "false", "False", "no", "No"}
        return bool(value)
    if key == "dataset_games":
        if value is None or value == "":
            return None
        parsed = int(value)
        return None if parsed <= 0 else parsed
    if key in {
        "self_play_use_beam",
        "self_play_train_after_collect",
    }:
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
        "batch_games",
        "tensor_build_workers",
        "beam_width",
        "beam_search_depth",
        "beam_time_budget_ms",
        "beam_min_remaining_sec",
        "self_play_games",
        "self_play_iterations",
        "self_play_eval_games",
        "self_play_opponent_pool_size",
        "self_play_target_rank",
        "self_play_plateau_patience",
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
        "value_win",
        "value_not_win",
        "self_play_target_win_rate",
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
    if "BATCH_SIZE" in os.environ and "BATCH_GAMES" not in os.environ:
        settings["batch_games"] = _coerce_value("batch_games", os.environ["BATCH_SIZE"])
    if overrides:
        unknown = set(overrides) - OVERRIDE_KEYS
        if unknown:
            raise ValueError(f"Unknown config overrides: {sorted(unknown)}")
        for key, value in overrides.items():
            settings[key] = _coerce_value(key, value)
    return settings


def build_config(root: Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = _resolve_settings(overrides)
    ensure_output_layout(root)
    d_model = settings["d_model"]
    model_ff = settings["model_ff"] if settings["model_ff"] is not None else d_model * 4
    fallback_paths = [root / path for path in settings["fallback_rollout_data"]]
    model_id = str(settings["model_id"])
    checkpoint = resolve_checkpoint_path(
        root,
        model_id=model_id,
        explicit=settings["model_output_path"],
    )

    return {
        "agent_deck_path": root / settings["agent_deck_path"],
        "data_candidates": [root / settings["primary_rollout_data"], *fallback_paths],
        "generated_path": root / settings["cabt_generated_path"],
        "competition_results_path": root / settings["competition_results_path"],
        "output_path": checkpoint,
        "report_path": report_path(root, model_id),
        "model_id": model_id,
        "output_layout": describe_layout(root),
        "coarse_feature_dim": COARSE_FEATURE_DIM,
        "require_cabt_eval_data": settings["require_cabt_eval_data"],
        "dataset_games": settings["dataset_games"],
        "transition_classes": settings["transition_classes"],
        "state_hash_dim": settings["state_hash_dim"],
        "window_size": settings["window_size"],
        "tensor_build_workers": settings["tensor_build_workers"],
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
            "games": settings["dataset_games"],
            "epochs": settings["train_epochs"],
            "patience": settings["early_stop_patience"],
            "min_delta": settings["early_stop_min_delta"],
            "print_every": settings["train_print_every"],
            "batch_games": settings["batch_games"],
        },
        "rewards": {
            "value_win": settings["value_win"],
            "value_not_win": settings["value_not_win"],
        },
        "beam_search": {
            "width": settings["beam_width"],
            "depth": settings["beam_search_depth"],
            "time_budget_ms": settings["beam_time_budget_ms"],
            "min_remaining_sec": settings["beam_min_remaining_sec"],
        },
        "self_play": {
            "games_per_iteration": settings["self_play_games"],
            "iterations": settings["self_play_iterations"],
            "eval_games": settings["self_play_eval_games"],
            "opponent_pool_size": settings["self_play_opponent_pool_size"],
            "use_beam": settings["self_play_use_beam"],
            "output_path": settings["self_play_output_path"],
            "checkpoint_dir": settings["self_play_checkpoint_dir"],
            "train_after_collect": settings["self_play_train_after_collect"],
            "field_deck_dir": settings["self_play_field_deck_dir"],
            "matchup_mode": settings["self_play_matchup_mode"],
            "target_rank": settings["self_play_target_rank"],
            "target_win_rate": settings["self_play_target_win_rate"],
            "plateau_patience": settings["self_play_plateau_patience"],
        },
    }


def resolve_generate_games(config: dict[str, Any]) -> int:
    """Games to simulate when generating rollouts (0 = skip)."""
    games = config.get("dataset_games")
    if games is not None:
        return max(0, int(games))
    return 0


def resolve_cabt_episodes(config: dict[str, Any], simulator_available: bool) -> int:
    """Backward-compatible alias for inline rollout generation."""
    del simulator_available
    return resolve_generate_games(config)
