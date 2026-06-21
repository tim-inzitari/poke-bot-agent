from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

try:
    from game_tracker import DERIVED_INFERENCE_DIM, GameEventTracker
except ImportError:
    from poke_agent.game_tracker import DERIVED_INFERENCE_DIM, GameEventTracker

COARSE_BASE_DIM = 11
COARSE_FEATURE_DIM = COARSE_BASE_DIM + DERIVED_INFERENCE_DIM
BASE_FEATURE_NAMES = (
    "turn",
    "your_index",
    "self_deck_count",
    "self_hand_count",
    "self_bench_count",
    "opp_deck_count",
    "opp_hand_count",
    "opp_bench_count",
    "legal_option_count",
    "select_max_count",
    "going_first",
)


def _is_card_id_key(key: str) -> bool:
    lowered = key.lower()
    return lowered.endswith(".id") or lowered.endswith("id") or lowered.endswith("cardid")


def hashed_deck_composition(deck: list[int] | tuple[int, ...], *, state_hash_dim: int) -> np.ndarray:
    """Hash **your** full 60-card deck list (private info you know; not opponent deck)."""
    vec = np.zeros(state_hash_dim, dtype=np.float32)
    if not deck:
        return vec
    counts: dict[int, int] = {}
    for card_id in deck:
        counts[int(card_id)] = counts.get(int(card_id), 0) + 1
    for card_id, count in sorted(counts.items()):
        token = f"self_deck.card={card_id}"
        vec[stable_hash_index(token, state_hash_dim)] += count / 60.0
    return vec


def seat_deck_from_row(row: dict[str, Any]) -> list[int] | None:
    """Return the seated player's own deck list (never the opponent's hidden deck)."""
    obs = row.get("observation") or {}
    your_index = int((obs.get("current") or {}).get("yourIndex", row.get("player", 0)))
    key = "deck0_cards" if your_index == 0 else "deck1_cards"
    deck = row.get(key)
    if isinstance(deck, list) and deck:
        return [int(card_id) for card_id in deck]
    return None


def going_first_feature(obs: dict[str, Any]) -> float:
    """1.0 if this seat goes first, 0.0 if second, 0.5 if undetermined."""
    current = obs.get("current") or {}
    first_player = int(current.get("firstPlayer", -1))
    your_index = int(current.get("yourIndex", 0))
    if first_player < 0:
        return 0.5
    return 1.0 if your_index == first_player else 0.0


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
            elif isinstance(value, (int, float)) and _is_card_id_key(key):
                token = f"{label}.{key}={int(value)}"
                amount = 1.0
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


def base_features_from_observation(obs: dict[str, Any]) -> list[float]:
    current = obs.get("current") or {}
    players = current.get("players") or [{}, {}]
    p0 = players[0] if len(players) > 0 else {}
    p1 = players[1] if len(players) > 1 else {}
    select = obs.get("select") or {}
    return [
        float(current.get("turn", 0)),
        float(current.get("yourIndex", 0)),
        float(p0.get("deckCount", 0)),
        float(p0.get("handCount", 0)),
        float(len(p0.get("bench", []))),
        float(p1.get("deckCount", 0)),
        float(p1.get("handCount", 0)),
        float(len(p1.get("bench", []))),
        float(len(select.get("option", []))),
        float(select.get("maxCount", 0)),
        going_first_feature(obs),
    ]


def pad_coarse_features(stored: list[float] | np.ndarray) -> list[float]:
    values = [float(v) for v in stored[:COARSE_BASE_DIM]]
    if len(stored) == COARSE_BASE_DIM - 1:
        values.append(0.5)
    if len(values) < COARSE_BASE_DIM:
        values.extend([0.0] * (COARSE_BASE_DIM - len(values)))
    if len(stored) >= COARSE_FEATURE_DIM:
        values.extend(float(v) for v in stored[COARSE_BASE_DIM:COARSE_FEATURE_DIM])
    else:
        values.extend([0.0] * DERIVED_INFERENCE_DIM)
    return values


def features_from_observation(
    obs: dict[str, Any],
    tracker: GameEventTracker | None = None,
) -> list[float]:
    coarse = base_features_from_observation(obs)
    if tracker is None:
        tracker = GameEventTracker()
    derived = tracker.observe(obs)
    return coarse + derived


def encode_observation_step(
    observation: dict[str, Any],
    tracker: GameEventTracker,
    *,
    state_hash_dim: int,
    action: Any = None,
    our_deck: list[int] | None = None,
) -> np.ndarray:
    coarse = features_from_observation(observation, tracker)
    return combine_features(
        coarse,
        observation,
        action,
        state_hash_dim=state_hash_dim,
        our_deck=our_deck,
    )


def combine_features(
    coarse: list[float],
    observation: Any = None,
    action: Any = None,
    *,
    state_hash_dim: int,
    our_deck: list[int] | None = None,
) -> np.ndarray:
    compact = np.array(coarse, dtype=np.float32)
    if observation is None and our_deck is None:
        return compact
    state_hash = (
        hashed_state_vector(observation, action, state_hash_dim=state_hash_dim)
        if observation is not None
        else np.zeros(state_hash_dim, dtype=np.float32)
    )
    if our_deck is not None:
        state_hash = state_hash + hashed_deck_composition(our_deck, state_hash_dim=state_hash_dim)
    return np.concatenate([compact, state_hash]).astype(np.float32)


def row_feature_vector(row: dict, *, state_hash_dim: int, tracker: GameEventTracker | None = None) -> np.ndarray:
    our_deck = seat_deck_from_row(row)
    observation = row.get("observation")
    if observation is not None and tracker is not None:
        return encode_observation_step(
            observation,
            tracker,
            state_hash_dim=state_hash_dim,
            action=row.get("action"),
            our_deck=our_deck,
        )
    if observation is not None:
        return combine_features(
            features_from_observation(observation),
            observation,
            row.get("action"),
            state_hash_dim=state_hash_dim,
            our_deck=our_deck,
        )
    return combine_features(
        pad_coarse_features(row["features"]),
        None,
        row.get("action"),
        state_hash_dim=state_hash_dim,
    )


def row_next_feature_vector(row: dict, *, state_hash_dim: int, tracker: GameEventTracker | None = None) -> np.ndarray:
    if "next_observation" in row and tracker is not None:
        return encode_observation_step(row["next_observation"], tracker, state_hash_dim=state_hash_dim)
    if "next_observation" in row and "next_features" in row:
        return combine_features(
            pad_coarse_features(row["next_features"]),
            row.get("next_observation"),
            None,
            state_hash_dim=state_hash_dim,
        )
    return row_feature_vector(row, state_hash_dim=state_hash_dim, tracker=tracker)


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

    tracker = GameEventTracker()
    encoded_steps: list[np.ndarray | None] = []

    for row in episode_rows:
        our_deck = seat_deck_from_row(row)
        if row.get("observation") is not None:
            encoded_steps.append(
                encode_observation_step(
                    row["observation"],
                    tracker,
                    state_hash_dim=state_hash_dim,
                    action=row.get("action"),
                    our_deck=our_deck,
                )
            )
        else:
            encoded_steps.append(
                combine_features(
                    pad_coarse_features(row["features"]),
                    None,
                    row.get("action"),
                    state_hash_dim=state_hash_dim,
                )
            )

    for idx, row in enumerate(episode_rows):
        current_index = len(xs)
        features = encoded_steps[idx]
        assert features is not None
        value = float(row["value"])
        if idx + 1 < len(encoded_steps):
            next_feature = encoded_steps[idx + 1]
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
            delta = next_feature[:COARSE_FEATURE_DIM] - features[:COARSE_FEATURE_DIM]
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    game_lengths = np.array([len(x_ep) for x_ep, *_ in episode_arrays if len(x_ep) > 0], dtype=np.int64)

    return (
        np.concatenate(xs_parts, axis=0),
        np.concatenate(values_parts, axis=0),
        np.concatenate(transition_parts, axis=0),
        np.concatenate(next_x_parts, axis=0),
        np.concatenate(terminal_parts, axis=0),
        history_indices_np,
        np.concatenate(history_mask_parts, axis=0),
        game_lengths,
    )


def build_training_arrays(
    rows: list[dict],
    *,
    transition_classes: int,
    state_hash_dim: int,
    window_size: int,
    workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
