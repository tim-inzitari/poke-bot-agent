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
from poke_agent.features import CARD_ID_SLOT_COUNT, build_training_arrays, default_tensor_build_workers


def limit_dataset_games(rows: list[dict], max_games: int | None) -> tuple[list[dict], int, int]:
    """Keep the first max_games complete episodes (by episode id)."""
    source_games = len({int(row["episode"]) for row in rows})
    if max_games is None or max_games <= 0 or source_games <= max_games:
        return rows, len(rows), source_games

    keep_ids = sorted({int(row["episode"]) for row in rows})[:max_games]
    keep_set = set(keep_ids)
    limited = [row for row in rows if int(row["episode"]) in keep_set]
    return limited, len(rows), source_games


def limit_recent_dataset_games(rows: list[dict], max_games: int | None) -> tuple[list[dict], int, int]:
    """Keep the most recent max_games complete episodes (by episode id)."""
    source_games = len({int(row["episode"]) for row in rows})
    if max_games is None or max_games <= 0 or source_games <= max_games:
        return rows, len(rows), source_games

    keep_ids = sorted({int(row["episode"]) for row in rows})[-max_games:]
    keep_set = set(keep_ids)
    limited = [row for row in rows if int(row["episode"]) in keep_set]
    return limited, len(rows), source_games


def _parse_jsonl_chunk(lines: list[str]) -> list[dict]:
    return [json.loads(line) for line in lines]


def trim_rollout_jsonl(path: Path, max_games: int | None) -> tuple[int, int]:
    """Rewrite JSONL to only the most recent max_games episodes."""
    if max_games is None or max_games <= 0 or not path.exists():
        return 0, 0

    rows = load_jsonl(path, workers=1)
    before = len({int(row["episode"]) for row in rows})
    limited, _, _ = limit_recent_dataset_games(rows, max_games)
    after = len({int(row["episode"]) for row in limited})
    if after >= before:
        return before, after

    with path.open("w", encoding="utf-8") as handle:
        for row in limited:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(
        f"trimmed rollout file: {before:,} -> {after:,} games "
        f"({len(limited):,} rows) at {path}"
    )
    return before, after


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


def resolve_data_device(compute_device: torch.device, mode: str | None) -> torch.device:
    """Where to hold the full dataset tensors.

    "cpu" streams batches to the GPU (dataset bounded by RAM, not VRAM);
    "cuda"/"device" keeps the whole dataset resident on the compute device;
    "auto" (default) keeps data on CPU whenever training on CUDA so large
    datasets fit in system RAM.
    """
    normalized = (mode or "auto").strip().lower()
    if normalized in {"cpu"}:
        return torch.device("cpu")
    if normalized in {"cuda", "gpu", "device"}:
        return compute_device
    # auto
    if compute_device.type == "cuda":
        return torch.device("cpu")
    return compute_device


@dataclass
class TrainingTensors:
    """Sequence-major training tensors: one row per (episode, seat) sequence.

    All tensors are padded to ``window_size`` on the time axis with left-padding;
    ``seq_mask`` marks valid (non-padded) timesteps. There is no per-row window
    gather — the causal transformer attends within each padded sequence directly.
    """

    data_path: Path | None
    x_seq: torch.Tensor
    seq_lengths: torch.Tensor
    seq_mask: torch.Tensor
    y: torch.Tensor
    returns: torch.Tensor
    transition_target: torch.Tensor
    next_x: torch.Tensor
    terminal: torch.Tensor
    card_ids: torch.Tensor
    feature_mean: np.ndarray
    feature_std: np.ndarray
    transition_classes: int
    window_size: int

    @property
    def num_games(self) -> int:
        return int(self.x_seq.shape[0])

    @property
    def num_seqs(self) -> int:
        return int(self.x_seq.shape[0])

    @property
    def dataset_rows(self) -> int:
        return int(self.seq_lengths.sum().item())


def _synthetic_smoke_arrays(
    transition_classes: int,
    window_size: int,
    *,
    feat_dim: int = 32,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rng = np.random.default_rng(7)
    seqs = 8
    steps_per_seq = min(16, window_size)
    x_seq = rng.normal(size=(seqs, window_size, feat_dim)).astype(np.float32)
    y_seq = np.tanh(x_seq[:, :, 0] * 0.1).astype(np.float32)
    returns_seq = y_seq.copy()
    transition_seq = rng.integers(0, transition_classes, size=(seqs, window_size), dtype=np.int64)
    next_x_seq = (x_seq + rng.normal(scale=0.1, size=x_seq.shape)).astype(np.float32)
    terminal_seq = np.zeros((seqs, window_size), dtype=np.float32)
    for seq in range(seqs):
        terminal_seq[seq, steps_per_seq - 1] = 1.0
    pad_count = window_size - steps_per_seq
    seq_mask = np.zeros((seqs, window_size), dtype=np.float32)
    seq_mask[:, pad_count:] = 1.0
    card_ids = rng.integers(0, 100, size=(seqs, window_size, CARD_ID_SLOT_COUNT), dtype=np.int64)
    seq_lengths = np.full((seqs,), steps_per_seq, dtype=np.int64)
    seq_ids = np.arange(seqs, dtype=np.int64)
    return (
        x_seq,
        y_seq,
        returns_seq,
        transition_seq,
        next_x_seq,
        terminal_seq,
        seq_mask,
        card_ids,
        seq_lengths,
        seq_ids,
    )


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

    reward_cfg = config.get("rewards", {})
    objective_cfg = config.get("objective", {})
    card_vocab_size = int(config.get("card_vocab_size", 2000))

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
        arrays = _synthetic_smoke_arrays(transition_classes, window_size)
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
        use_recent = bool(config.get("training", {}).get("recent_games", False))
        limit_fn = limit_recent_dataset_games if use_recent else limit_dataset_games
        rows, source_rows, source_games = limit_fn(rows, train_games)
        used_games = len({int(row["episode"]) for row in rows})
        if train_games is not None:
            which = "recent" if use_recent else "first"
            print(
                f"training size: {used_games:,} / {source_games:,} games "
                f"({len(rows):,} / {source_rows:,} rows, window={train_games} {which})"
            )
        arrays = build_training_arrays(
            rows,
            transition_classes=transition_classes,
            state_hash_dim=state_hash_dim,
            window_size=window_size,
            workers=workers,
            card_vocab_size=card_vocab_size,
            value_gamma=float(objective_cfg.get("value_return_gamma", 0.997)),
            value_shaping_alpha=float(objective_cfg.get("value_shaping_alpha", 0.15)),
            value_win=float(reward_cfg.get("value_win", 1.0)),
            value_not_win=float(reward_cfg.get("value_not_win", -1.0)),
            value_timeout=float(reward_cfg.get("value_timeout", -2.0)),
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

        if config.get("require_top_of_ladder_data", False):
            from poke_agent.training_diversity import assert_top_of_ladder_data

            ladder_stats = assert_top_of_ladder_data(
                rows,
                min_fraction=float(config.get("min_top_of_ladder_fraction", 0.0)),
            )
            print(
                "top-of-ladder data:"
                f" {ladder_stats['ladder_games']}/{ladder_stats['games']} games"
                f" ({ladder_stats['ladder_fraction']:.1%})"
                f" sources={ladder_stats['source_counts']}"
            )

    (
        x_seq_np,
        y_seq_np,
        returns_seq_np,
        transition_seq_np,
        next_x_seq_np,
        terminal_seq_np,
        seq_mask_np,
        card_ids_np,
        seq_lengths_np,
        _seq_ids_np,
    ) = arrays

    feature_mean = x_seq_np.reshape(-1, x_seq_np.shape[-1]).mean(axis=0, keepdims=True)
    feature_std = x_seq_np.reshape(-1, x_seq_np.shape[-1]).std(axis=0, keepdims=True) + 1e-6
    x_norm = (x_seq_np - feature_mean) / feature_std
    next_x_norm = (next_x_seq_np - feature_mean) / feature_std

    data_device = resolve_data_device(device, config.get("train_data_device", "auto"))
    if data_device != device:
        print(
            f"dataset on {data_device} (compute on {device}); "
            "batches stream to the GPU so dataset size is bounded by RAM, not VRAM"
        )

    x_seq = _to_device_tensor(x_norm, data_device)
    y = _to_device_tensor(y_seq_np, data_device)
    returns = _to_device_tensor(returns_seq_np, data_device)
    transition_target = _to_device_tensor(transition_seq_np, data_device)
    next_x = _to_device_tensor(next_x_norm, data_device)
    terminal = _to_device_tensor(terminal_seq_np, data_device)
    seq_mask = _to_device_tensor(seq_mask_np, data_device)
    card_ids = _to_device_tensor(card_ids_np, data_device)
    seq_lengths = _to_device_tensor(seq_lengths_np, data_device)

    print(
        "x_seq",
        tuple(x_seq.shape),
        "value",
        tuple(y.shape),
        "transition",
        tuple(transition_target.shape),
        "seqs",
        int(x_seq.shape[0]),
    )
    print("seq_mask", tuple(seq_mask.shape), "window", window_size)

    return TrainingTensors(
        data_path=data_path,
        x_seq=x_seq,
        seq_lengths=seq_lengths,
        seq_mask=seq_mask,
        y=y,
        returns=returns,
        transition_target=transition_target,
        next_x=next_x,
        terminal=terminal,
        card_ids=card_ids,
        feature_mean=feature_mean,
        feature_std=feature_std,
        transition_classes=transition_classes,
        window_size=window_size,
    )
