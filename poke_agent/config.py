from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def build_config(root: Path) -> dict[str, Any]:
    d_model = env_int("MODEL_D_MODEL", 512)
    return {
        "agent_deck_path": root / os.environ.get(
            "AGENT_DECK_PATH",
            "decks/competitive/high_performing/2026-05_regional-campinas-2026_4th_dragapult-dudunsparce.csv",
        ),
        "data_candidates": [
            root / os.environ.get("PRIMARY_ROLLOUT_DATA", "data/mac-rollouts-100k-fullstate.jsonl"),
            root / "data/mac-rollouts-10k.jsonl",
            root / "data/notebook_rollouts.jsonl",
            root / "data/kaggle-output/data/cabt_rollouts.jsonl",
            root / "data/deckpool-smoke.jsonl",
            root / "data/container-mp-smoke.jsonl",
            root / "data/container-smoke.jsonl",
        ],
        "generated_path": root / os.environ.get("CABT_GENERATED_PATH", "data/notebook_rollouts.jsonl"),
        "competition_results_path": root / os.environ.get(
            "COMPETITION_RESULTS_PATH",
            "data/competition-results.jsonl",
        ),
        "output_path": root / os.environ.get("MODEL_OUTPUT_PATH", "out/value_model.pt"),
        "require_cabt_eval_data": os.environ.get("REQUIRE_CABT_EVAL_DATA", "1") != "0",
        "transition_classes": env_int("TRANSITION_CLASSES", 8),
        "state_hash_dim": env_int("STATE_HASH_DIM", 256),
        "window_size": env_int("WINDOW_SIZE", 128),
        "model": {
            "d_model": d_model,
            "heads": env_int("MODEL_HEADS", 8),
            "layers": env_int("MODEL_LAYERS", 8),
            "ff": env_int("MODEL_FF", d_model * 4),
            "dropout": env_float("MODEL_DROPOUT", 0.1),
            "learning_rate": env_float("LEARNING_RATE", 3e-4),
            "weight_decay": env_float("WEIGHT_DECAY", 1e-2),
        },
        "loss": {
            "value": env_float("LOSS_VALUE_WEIGHT", 1.0),
            "policy": env_float("LOSS_POLICY_WEIGHT", 0.35),
            "dynamics": env_float("LOSS_DYNAMICS_WEIGHT", 0.15),
            "entropy": env_float("LOSS_ENTROPY_WEIGHT", 0.01),
            "uncertainty": env_float("LOSS_UNCERTAINTY_WEIGHT", 0.02),
        },
        "training": {
            "epochs": env_int("TRAIN_EPOCHS", 500),
            "patience": env_int("EARLY_STOP_PATIENCE", 10),
            "min_delta": env_float("EARLY_STOP_MIN_DELTA", 1e-5),
            "print_every": env_int("TRAIN_PRINT_EVERY", 100),
            "batch_size": env_int("BATCH_SIZE", 256),
        },
    }
