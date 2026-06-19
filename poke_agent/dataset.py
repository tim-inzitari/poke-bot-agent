from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from poke_agent.cabt_validation import (
    CabtEvaluationDataError,
    assert_cabt_evaluation_rows,
    resolve_cabt_eval_data_path,
)
from poke_agent.features import build_training_arrays


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@dataclass
class TrainingTensors:
    data_path: Path | None
    x: torch.Tensor
    x_padded: torch.Tensor
    y: torch.Tensor
    transition_target: torch.Tensor
    next_x: torch.Tensor
    terminal: torch.Tensor
    history_index: torch.Tensor
    history_mask: torch.Tensor
    feature_mean: np.ndarray
    feature_std: np.ndarray
    transition_classes: int
    window_size: int


def _synthetic_smoke_arrays(
    transition_classes: int,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    x_np = rng.normal(size=(128, 10)).astype(np.float32)
    y_np = np.tanh(x_np[:, 0] * 0.1 + x_np[:, 2] * 0.03 - x_np[:, 5] * 0.03).astype(np.float32)
    transition_np = rng.integers(0, transition_classes, size=(128,), dtype=np.int64)
    next_x_np = (x_np + rng.normal(scale=0.1, size=x_np.shape)).astype(np.float32)
    terminal_np = np.zeros((128,), dtype=np.float32)
    pad_index = len(x_np)
    history_index_np = np.full((len(x_np), window_size), pad_index, dtype=np.int64)
    history_mask_np = np.zeros((len(x_np), window_size), dtype=np.float32)
    for i in range(len(x_np)):
        context = list(range(max(0, i - window_size + 1), i + 1))
        history_index_np[i, -len(context):] = context
        history_mask_np[i, -len(context):] = 1.0
    return x_np, y_np, transition_np, next_x_np, terminal_np, history_index_np, history_mask_np


def prepare_training_tensors(config: dict[str, Any], device: torch.device) -> TrainingTensors:
    transition_classes = config["transition_classes"]
    state_hash_dim = config["state_hash_dim"]
    window_size = config["window_size"]
    data_candidates = config["data_candidates"]
    require_cabt_eval = config.get("require_cabt_eval_data", True)

    data_path = resolve_cabt_eval_data_path(data_candidates) if require_cabt_eval else next(
        (path for path in data_candidates if path.exists()),
        None,
    )
    if data_path is None:
        if require_cabt_eval:
            searched = ", ".join(str(path) for path in data_candidates)
            raise CabtEvaluationDataError(
                "No CABT evaluation rollout JSONL found. "
                f"Searched: {searched}. "
                "Generate with scripts/generate_cabt_data.py or set PRIMARY_ROLLOUT_DATA. "
                "Set REQUIRE_CABT_EVAL_DATA=0 only for smoke tests."
            )
        print("No rollout data found. Using synthetic smoke data so Run All still completes.")
        x_np, y_np, transition_np, next_x_np, terminal_np, history_index_np, history_mask_np = _synthetic_smoke_arrays(
            transition_classes,
            window_size,
        )
    else:
        rows = load_jsonl(data_path)
        assert_cabt_evaluation_rows(rows, path=data_path)
        x_np, y_np, transition_np, next_x_np, terminal_np, history_index_np, history_mask_np = build_training_arrays(
            rows,
            transition_classes=transition_classes,
            state_hash_dim=state_hash_dim,
            window_size=window_size,
        )
        print(f"loaded {len(rows)} CABT evaluation rows from {data_path}")

    feature_mean = x_np.mean(axis=0, keepdims=True)
    feature_std = x_np.std(axis=0, keepdims=True) + 1e-6
    x_norm = (x_np - feature_mean) / feature_std
    next_x_norm = (next_x_np - feature_mean) / feature_std

    x = torch.tensor(x_norm, device=device)
    x_padded = torch.cat([x, torch.zeros((1, x.shape[1]), device=device, dtype=x.dtype)], dim=0)
    y = torch.tensor(y_np, device=device)
    transition_target = torch.tensor(transition_np, device=device)
    next_x = torch.tensor(next_x_norm, device=device)
    terminal = torch.tensor(terminal_np, device=device)
    history_index = torch.tensor(history_index_np, device=device)
    history_mask = torch.tensor(history_mask_np, device=device)

    print("x", tuple(x.shape), "value", tuple(y.shape), "transition", tuple(transition_target.shape))
    print("history", tuple(history_index.shape), "window", window_size)

    return TrainingTensors(
        data_path=data_path,
        x=x,
        x_padded=x_padded,
        y=y,
        transition_target=transition_target,
        next_x=next_x,
        terminal=terminal,
        history_index=history_index,
        history_mask=history_mask,
        feature_mean=feature_mean,
        feature_std=feature_std,
        transition_classes=transition_classes,
        window_size=window_size,
    )
