from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from poke_agent.cabt_validation import (
    CabtEvaluationDataError,
    assert_cabt_evaluation_rows,
    assert_training_rollout_rows,
    resolve_training_data_path,
    uses_generated_training_data,
)
from poke_agent.features import COARSE_FEATURE_DIM, build_training_arrays, default_tensor_build_workers


def limit_dataset_games(rows: list[dict], max_games: int | None) -> tuple[list[dict], int, int]:
    """Keep the first max_games complete episodes (by episode id)."""
    source_games = len({int(row["episode"]) for row in rows})
    if max_games is None or max_games <= 0 or source_games <= max_games:
        return rows, len(rows), source_games

    keep_ids = sorted({int(row["episode"]) for row in rows})[:max_games]
    keep_set = set(keep_ids)
    limited = [row for row in rows if int(row["episode"]) in keep_set]
    return limited, len(rows), source_games


def limit_training_rows(rows: list[dict], max_rows: int | None) -> tuple[list[dict], int]:
    """Keep leading complete episodes until row budget is reached."""
    if max_rows is None or max_rows <= 0 or len(rows) <= max_rows:
        return rows, len(rows)

    by_episode: dict[int, list[dict]] = {}
    for row in rows:
        by_episode.setdefault(int(row["episode"]), []).append(row)

    limited: list[dict] = []
    for episode_id in sorted(by_episode):
        episode_rows = sorted(by_episode[episode_id], key=lambda row: int(row["step"]))
        if len(limited) + len(episode_rows) > max_rows:
            break
        limited.extend(episode_rows)

    if not limited:
        limited = rows[:max_rows]
    return limited, len(rows)


def _parse_jsonl_chunk(lines: list[str]) -> list[dict]:
    return [json.loads(line) for line in lines]


def load_jsonl(path: Path, *, workers: int | None = None) -> list[dict]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    worker_count = default_tensor_build_workers() if workers is None else max(1, int(workers))
    if worker_count <= 1 or len(lines) < 5000:
        return _parse_jsonl_chunk(lines)

    chunk_size = max(1000, len(lines) // (worker_count * 4))
    chunks = [lines[index:index + chunk_size] for index in range(0, len(lines), chunk_size)]
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        rows = [row for chunk_rows in executor.map(_parse_jsonl_chunk, chunks) for row in chunk_rows]
    print(f"jsonl load: {len(rows)} rows using {worker_count} workers")
    return rows


def _to_device_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    contiguous = np.ascontiguousarray(array)
    tensor = torch.from_numpy(contiguous)
    return tensor.to(device=device)


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
    game_lengths: torch.Tensor
    feature_mean: np.ndarray
    feature_std: np.ndarray
    transition_classes: int
    window_size: int

    @property
    def num_games(self) -> int:
        return int(self.game_lengths.shape[0])

    @property
    def game_starts(self) -> torch.Tensor:
        device = self.game_lengths.device
        starts = torch.zeros(self.num_games + 1, dtype=torch.long, device=device)
        starts[1:] = torch.cumsum(self.game_lengths, dim=0)
        return starts

    def row_indices_for_games(self, game_ids: torch.Tensor) -> torch.Tensor:
        starts = self.game_starts
        parts = [
            torch.arange(int(starts[game_id]), int(starts[game_id + 1]), device=game_ids.device)
            for game_id in game_ids.tolist()
        ]
        return torch.cat(parts) if parts else torch.empty(0, dtype=torch.long, device=game_ids.device)


def _synthetic_smoke_arrays(
    transition_classes: int,
    state_hash_dim: int,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    games = 8
    steps_per_game = 16
    total = games * steps_per_game
    feature_dim = COARSE_FEATURE_DIM + state_hash_dim
    x_np = rng.normal(size=(total, feature_dim)).astype(np.float32)
    y_np = np.tanh(x_np[:, 0] * 0.1 + x_np[:, 2] * 0.03 - x_np[:, 5] * 0.03).astype(np.float32)
    transition_np = rng.integers(0, transition_classes, size=(total,), dtype=np.int64)
    next_x_np = (x_np + rng.normal(scale=0.1, size=x_np.shape)).astype(np.float32)
    terminal_np = np.zeros((total,), dtype=np.float32)
    for game in range(games):
        terminal_np[(game + 1) * steps_per_game - 1] = 1.0
    pad_index = total
    history_index_np = np.full((total, window_size), pad_index, dtype=np.int64)
    history_mask_np = np.zeros((total, window_size), dtype=np.float32)
    for game in range(games):
        base = game * steps_per_game
        for step in range(steps_per_game):
            i = base + step
            context = list(range(base, i + 1))[-window_size:]
            history_index_np[i, -len(context):] = context
            history_mask_np[i, -len(context):] = 1.0
    game_lengths = np.full((games,), steps_per_game, dtype=np.int64)
    return x_np, y_np, transition_np, next_x_np, terminal_np, history_index_np, history_mask_np, game_lengths


def prepare_training_tensors(config: dict[str, Any], device: torch.device) -> TrainingTensors:
    transition_classes = config["transition_classes"]
    state_hash_dim = config["state_hash_dim"]
    window_size = config["window_size"]
    require_cabt_eval = config.get("require_cabt_eval_data", True)
    workers = config.get("tensor_build_workers")
    if workers is None:
        workers = default_tensor_build_workers()
    else:
        workers = max(1, int(workers))

    data_path = resolve_training_data_path(config)
    if data_path is None:
        if require_cabt_eval:
            searched = ", ".join(str(path) for path in config.get("data_candidates", []))
            if config.get("generated_path") is not None:
                searched = f"{config['generated_path']}, {searched}"
            raise CabtEvaluationDataError(
                "No CABT evaluation rollout JSONL found. "
                f"Searched: {searched}. "
                "Generate with scripts/generate_cabt_data.py or set PRIMARY_ROLLOUT_DATA. "
                "Set REQUIRE_CABT_EVAL_DATA=0 only for smoke tests."
            )
        print("No rollout data found. Using synthetic smoke data so Run All still completes.")
        x_np, y_np, transition_np, next_x_np, terminal_np, history_index_np, history_mask_np, game_lengths_np = _synthetic_smoke_arrays(
            transition_classes,
            state_hash_dim,
            window_size,
        )
    else:
        started = time.perf_counter()
        rows = load_jsonl(data_path, workers=workers)
        if require_cabt_eval and not uses_generated_training_data(config, data_path):
            assert_cabt_evaluation_rows(rows, path=data_path)
        else:
            assert_training_rollout_rows(rows, path=data_path)
        if config.get("require_complete_games", True):
            from poke_agent.rewards import filter_complete_episode_rows

            before = len({int(row["episode"]) for row in rows})
            rows = filter_complete_episode_rows(rows)
            after = len({int(row["episode"]) for row in rows})
            if before != after:
                print(f"complete games only: {after:,} / {before:,} episodes kept")
        train_games = config.get("training", {}).get("games")
        rows, source_rows, source_games = limit_dataset_games(rows, train_games)
        used_games = len({int(row["episode"]) for row in rows})
        if train_games is not None:
            print(
                f"training size: {used_games:,} / {source_games:,} games "
                f"({len(rows):,} / {source_rows:,} rows, DATASET_GAMES={train_games})"
            )
        x_np, y_np, transition_np, next_x_np, terminal_np, history_index_np, history_mask_np, game_lengths_np = build_training_arrays(
            rows,
            transition_classes=transition_classes,
            state_hash_dim=state_hash_dim,
            window_size=window_size,
            workers=workers,
        )
        elapsed = time.perf_counter() - started
        print(f"loaded {len(rows)} rollout rows from {data_path} in {elapsed:.1f}s ({workers} workers)")

        if config.get("require_training_matchup_diversity", True):
            from poke_agent.training_diversity import (
                assert_deck_metadata_not_in_features,
                assert_submission_deck_separate_from_training,
                assert_training_matchup_diversity,
                training_matchup_stats,
            )

            assert_submission_deck_separate_from_training(config, rows, data_path=data_path)
            stats = assert_training_matchup_diversity(
                rows,
                min_matchups=int(config.get("min_training_matchups", 2)),
                min_deck_slugs=int(config.get("min_training_deck_slugs", 2)),
                allow_single_matchup=False,
            )
            assert_deck_metadata_not_in_features(rows, state_hash_dim=state_hash_dim)
            print(
                "training data diversity:"
                f" {stats['games']} games,"
                f" {stats['unique_matchups']} matchups,"
                f" {stats['unique_deck_slugs']} deck slugs"
            )

    feature_mean = x_np.mean(axis=0, keepdims=True)
    feature_std = x_np.std(axis=0, keepdims=True) + 1e-6
    x_norm = (x_np - feature_mean) / feature_std
    next_x_norm = (next_x_np - feature_mean) / feature_std

    x = _to_device_tensor(x_norm, device)
    x_padded = torch.cat([x, torch.zeros((1, x.shape[1]), device=device, dtype=x.dtype)], dim=0)
    y = _to_device_tensor(y_np, device)
    transition_target = _to_device_tensor(transition_np, device)
    next_x = _to_device_tensor(next_x_norm, device)
    terminal = _to_device_tensor(terminal_np, device)
    history_index = _to_device_tensor(history_index_np, device)
    history_mask = _to_device_tensor(history_mask_np, device)
    game_lengths = _to_device_tensor(game_lengths_np, device)

    print(
        "x", tuple(x.shape), "value", tuple(y.shape), "transition", tuple(transition_target.shape),
        "games", int(game_lengths.shape[0]),
    )
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
        game_lengths=game_lengths,
        feature_mean=feature_mean,
        feature_std=feature_std,
        transition_classes=transition_classes,
        window_size=window_size,
    )
