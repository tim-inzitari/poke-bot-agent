from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def stable_hash_index(text: str, size: int) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % size


def iter_state_items(value: Any, prefix: str = ""):
    if isinstance(value, dict):
        for key in sorted(value):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_state_items(value[key], child_prefix)
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            child_prefix = f"{prefix}[{idx}]"
            yield from iter_state_items(item, child_prefix)
    else:
        yield prefix, value


def hashed_state_vector(observation: Any, action: Any = None, *, state_hash_dim: int) -> np.ndarray:
    vec = np.zeros(state_hash_dim, dtype=np.float32)
    payloads = [("obs", observation, 1.0), ("action", action, 0.5)]
    for label, payload, weight in payloads:
        if payload is None:
            continue
        for key, value in iter_state_items(payload):
            if isinstance(value, bool):
                token = f"{label}.{key}:bool"
                amount = 1.0 if value else -1.0
            elif isinstance(value, (int, float)):
                token = f"{label}.{key}:num"
                amount = float(np.tanh(float(value) / 100.0))
            elif value is None:
                token = f"{label}.{key}:none"
                amount = 1.0
            else:
                token = f"{label}.{key}={value}"
                amount = 1.0
            vec[stable_hash_index(token, state_hash_dim)] += weight * amount
    return vec


def combine_features(
    coarse: list[float],
    observation: Any = None,
    action: Any = None,
    *,
    state_hash_dim: int,
) -> np.ndarray:
    compact = np.array(coarse, dtype=np.float32)
    if observation is None:
        return compact
    return np.concatenate([compact, hashed_state_vector(observation, action, state_hash_dim=state_hash_dim)]).astype(np.float32)


def row_feature_vector(row: dict, *, state_hash_dim: int) -> np.ndarray:
    return combine_features(row["features"], row.get("observation"), row.get("action"), state_hash_dim=state_hash_dim)


def row_next_feature_vector(row: dict, *, state_hash_dim: int) -> np.ndarray:
    if "next_observation" in row and "next_features" in row:
        return combine_features(row["next_features"], row.get("next_observation"), None, state_hash_dim=state_hash_dim)
    return row_feature_vector(row, state_hash_dim=state_hash_dim)


def build_training_arrays(
    rows: list[dict],
    *,
    transition_classes: int,
    state_hash_dim: int,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    by_episode: dict[int, list[dict]] = {}
    for row in rows:
        by_episode.setdefault(int(row["episode"]), []).append(row)
    for episode_rows in by_episode.values():
        episode_rows.sort(key=lambda row: int(row["step"]))

    xs: list[np.ndarray] = []
    values: list[float] = []
    transition_targets: list[int] = []
    next_features: list[np.ndarray] = []
    terminal_mask: list[float] = []
    history_indices: list[list[int]] = []
    history_mask: list[list[float]] = []

    for episode_rows in by_episode.values():
        episode_indices: list[int] = []
        for idx, row in enumerate(episode_rows):
            current_index = len(xs)
            features = row_feature_vector(row, state_hash_dim=state_hash_dim)
            value = float(row["value"])
            if "next_observation" in row and "next_features" in row:
                next_feature = row_next_feature_vector(row, state_hash_dim=state_hash_dim)
                is_terminal = float(row.get("terminal", False))
            elif idx + 1 < len(episode_rows):
                next_row = episode_rows[idx + 1]
                next_feature = row_feature_vector(next_row, state_hash_dim=state_hash_dim)
                is_terminal = 0.0
            else:
                next_feature = features.copy()
                is_terminal = 1.0

            if "action" in row:
                action_key = json.dumps(row["action"], sort_keys=True, separators=(",", ":"))
                transition_class = stable_hash_index(action_key, transition_classes)
            elif is_terminal:
                transition_class = transition_classes - 1
            else:
                delta = next_feature[: len(row["features"])] - features[: len(row["features"])]
                transition_class = int(abs(delta).argmax()) % transition_classes

            context = (episode_indices + [current_index])[-window_size:]
            pad_count = window_size - len(context)
            history_indices.append(([-1] * pad_count) + context)
            history_mask.append(([0.0] * pad_count) + ([1.0] * len(context)))

            xs.append(features)
            values.append(value)
            transition_targets.append(transition_class)
            next_features.append(next_feature)
            terminal_mask.append(is_terminal)
            episode_indices.append(current_index)

    pad_index = len(xs)
    history_indices_np = np.array(history_indices, dtype=np.int64)
    history_indices_np[history_indices_np < 0] = pad_index

    return (
        np.stack(xs).astype(np.float32),
        np.array(values, dtype=np.float32),
        np.array(transition_targets, dtype=np.int64),
        np.stack(next_features).astype(np.float32),
        np.array(terminal_mask, dtype=np.float32),
        history_indices_np,
        np.array(history_mask, dtype=np.float32),
    )
