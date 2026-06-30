from __future__ import annotations

from typing import Any, Literal

# =============================================================================
# Multi-model catalog — edit here to add/compare neural, heuristic, and hybrid
# agents. Does not affect the legacy single-model training path in config.py.
# =============================================================================

# Agent used for multi-model rollouts/evaluation (heuristic or hybrid recommended).
ACTIVE_MODEL = "random"

# Neural catalog entries to train when running scripts/train_models.py.
# Heuristic and hybrid entries are skipped automatically.
TRAIN_MODELS = [
    "temporal_current",
    "transformer_small",
    "transformer_default",
]

ModelKind = Literal["neural", "heuristic", "hybrid"]

# Each entry needs:
#   kind: neural | heuristic | hybrid
#   description: short label for notebooks/logs
#
# Neural entries may override training/model keys from poke_agent/config.py.
# Checkpoints/reports are written under outputs/checkpoints and outputs/reports
# using the catalog entry id unless output_path is explicitly overridden.
#
# Heuristic entries need:
#   strategy: random | first_legal | max_option
#
# Hybrid entries need:
#   neural_model: catalog id of a neural model
#   heuristic_model: catalog id of a heuristic model
#   mode: neural_first | heuristic_first | fallback
#   confidence_threshold: reserved for future neural-confidence gating

MODEL_CATALOG: dict[str, dict[str, Any]] = {
    "random": {
        "kind": "heuristic",
        "description": "Uniform random legal actions",
        "strategy": "random",
    },
    "first_legal": {
        "kind": "heuristic",
        "description": "Always pick the first legal option indices",
        "strategy": "first_legal",
    },
    "temporal_current": {
        "kind": "neural",
        "description": "Current temporal KAN RL system (mirrors poke_agent/config.py)",
        "architecture": "temporal_kan_rl",
        "model_type": "temporal_kan_rl_complex_loss",
        "transition_classes": 8,
        "state_hash_dim": 256,
        "window_size": 256,
        "d_model": 128,
        "model_heads": 8,
        "model_layers": 8,
        "model_dropout": 0.1,
        "model_use_kan": True,
        "kan_grid_size": 8,
        "learning_rate": 3e-4,
        "weight_decay": 1e-2,
        "loss_value_weight": 1.0,
        "loss_policy_weight": 0.35,
        "loss_dynamics_weight": 0.15,
        "loss_entropy_weight": 0.01,
        "loss_uncertainty_weight": 0.02,
        "train_epochs": 1000,
        "early_stop_patience": 25,
        "early_stop_min_delta": 1e-5,
        "train_print_every": 100,
        "batch_games": 2,
    },
    "transformer_small": {
        "kind": "neural",
        "description": "Compact transformer for fast iteration",
        "architecture": "transformer_rl",
        "model_use_kan": False,
        "d_model": 128,
        "model_heads": 4,
        "model_layers": 2,
        "model_dropout": 0.1,
        "train_epochs": 100,
        "batch_games": 16,
        "early_stop_patience": 10,
    },
    "transformer_default": {
        "kind": "neural",
        "description": "Default transformer — uses config.py values unless overridden below",
        "architecture": "transformer_rl",
        "model_use_kan": False,
    },
    "transformer_large": {
        "kind": "neural",
        "description": "Larger transformer experiment",
        "architecture": "transformer_rl",
        "model_use_kan": False,
        "d_model": 512,
        "model_heads": 8,
        "model_layers": 8,
        "train_epochs": 500,
        "batch_games": 8,
        "early_stop_patience": 25,
    },
    "hybrid_default_random": {
        "kind": "hybrid",
        "description": "Current temporal model with random fallback on failure",
        "neural_model": "temporal_current",
        "heuristic_model": "random",
        "mode": "fallback",
        "confidence_threshold": 0.0,
    },
    "hybrid_small_first_legal": {
        "kind": "hybrid",
        "description": "Small neural model, heuristic takes priority",
        "neural_model": "transformer_small",
        "heuristic_model": "first_legal",
        "mode": "heuristic_first",
        "confidence_threshold": 0.0,
    },
}

NEURAL_OVERRIDE_KEYS = frozenset({
    "output_path",
    "d_model",
    "model_heads",
    "model_layers",
    "model_ff",
    "model_dropout",
    "model_use_kan",
    "kan_grid_size",
    "learning_rate",
    "weight_decay",
    "loss_value_weight",
    "loss_policy_weight",
    "loss_dynamics_weight",
    "loss_entropy_weight",
    "loss_uncertainty_weight",
    "dataset_games",
    "train_epochs",
    "early_stop_patience",
    "early_stop_min_delta",
    "train_print_every",
    "batch_games",
    "transition_classes",
    "state_hash_dim",
    "window_size",
})
