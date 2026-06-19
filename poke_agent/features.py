from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
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


def default_tensor_build_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, cpu_count - 2 if cpu_count > 4 else cpu_count))


def _group_rows_by_episode(rows: list[dict]) -> list[list[dict]]:
    by_episode: dict[int, list[dict]] = {}
    for row in rows:
        by_episode.setdefault(int(row["episode"]), []).append(row)
    episodes = [by_episode[key] for key in sorted(by_episode)]
    for episode_rows in episodes:
        episode_rows.sort(key=lambda row: int(row["step"]))
    return episodes


def _build_episode_arrays(
    episode_rows: list[dict],
    *,
    transition_classes: int,
    state_hash_dim: int,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    values: list[float] = []
    transition_targets: list[int] = []
    next_features: list[np.ndarray] = []
    terminal_mask: list[float] = []
    history_indices: list[list[int]] = []
    history_mask: list[list[float]] = []
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

    return (
        np.stack(xs).astype(np.float32),
        np.array(values, dtype=np.float32),
        np.array(transition_targets, dtype=np.int64),
        np.stack(next_features).astype(np.float32),
        np.array(terminal_mask, dtype=np.float32),
        np.array(history_indices, dtype=np.int64),
        np.array(history_mask, dtype=np.float32),
    )


def _episode_worker(args: tuple[list[dict], int, int, int]):
    episode_rows, transition_classes, state_hash_dim, window_size = args
    return _build_episode_arrays(
        episode_rows,
        transition_classes=transition_classes,
        state_hash_dim=state_hash_dim,
        window_size=window_size,
    )


def _merge_episode_arrays(
    episode_arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs_parts: list[np.ndarray] = []
    values_parts: list[np.ndarray] = []
    transition_parts: list[np.ndarray] = []
    next_x_parts: list[np.ndarray] = []
    terminal_parts: list[np.ndarray] = []
    history_index_parts: list[np.ndarray] = []
    history_mask_parts: list[np.ndarray] = []

    row_offset = 0
    for x_ep, y_ep, t_ep, next_ep, terminal_ep, history_ep, mask_ep in episode_arrays:
        if len(x_ep) == 0:
            continue
        history_ep = history_ep.copy()
        valid = history_ep >= 0
        history_ep[valid] += row_offset

        xs_parts.append(x_ep)
        values_parts.append(y_ep)
        transition_parts.append(t_ep)
        next_x_parts.append(next_ep)
        terminal_parts.append(terminal_ep)
        history_index_parts.append(history_ep)
        history_mask_parts.append(mask_ep)
        row_offset += len(x_ep)

    pad_index = row_offset
    history_indices_np = np.concatenate(history_index_parts, axis=0)
    history_indices_np[history_indices_np < 0] = pad_index

    return (
        np.concatenate(xs_parts, axis=0),
        np.concatenate(values_parts, axis=0),
        np.concatenate(transition_parts, axis=0),
        np.concatenate(next_x_parts, axis=0),
        np.concatenate(terminal_parts, axis=0),
        history_indices_np,
        np.concatenate(history_mask_parts, axis=0),
    )


def build_training_arrays(
    rows: list[dict],
    *,
    transition_classes: int,
    state_hash_dim: int,
    window_size: int,
    workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    episodes = _group_rows_by_episode(rows)
    worker_count = default_tensor_build_workers() if workers is None else max(1, int(workers))

    if worker_count <= 1 or len(episodes) < worker_count * 2:
        episode_arrays = [
            _build_episode_arrays(
                episode_rows,
                transition_classes=transition_classes,
                state_hash_dim=state_hash_dim,
                window_size=window_size,
            )
            for episode_rows in episodes
        ]
    else:
        tasks = [
            (episode_rows, transition_classes, state_hash_dim, window_size)
            for episode_rows in episodes
        ]
        chunksize = max(1, len(tasks) // (worker_count * 4))
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            episode_arrays = list(executor.map(_episode_worker, tasks, chunksize=chunksize))
        print(f"tensor build: {len(rows)} rows across {len(episodes)} episodes using {worker_count} workers")

    return _merge_episode_arrays(episode_arrays)
