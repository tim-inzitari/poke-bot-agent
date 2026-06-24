from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from poke_agent.tensor_cache import (
    build_tensor_cache_fingerprint,
    cache_is_valid,
    describe_training_tensor_cache,
    has_valid_training_tensor_cache,
    load_tensor_cache,
    save_tensor_cache,
    tensor_cache_path,
)


def _tiny_arrays() -> dict[str, np.ndarray]:
    x = np.random.randn(4, 8, 16).astype(np.float32)
    next_x = np.random.randn(4, 8, 16).astype(np.float32)
    mean = x.reshape(-1, x.shape[-1]).mean(axis=0, keepdims=True)
    std = x.reshape(-1, x.shape[-1]).std(axis=0, keepdims=True) + 1e-6
    x = (x - mean) / std
    next_x = (next_x - mean) / std
    return {
        "x_seq": x,
        "y": np.random.randn(4, 8).astype(np.float32),
        "returns": np.random.randn(4, 8).astype(np.float32),
        "transition_target": np.zeros((4, 8), dtype=np.int64),
        "next_x": next_x,
        "terminal": np.zeros((4, 8), dtype=np.float32),
        "seq_mask": np.ones((4, 8), dtype=np.float32),
        "card_ids": np.zeros((4, 8, 30), dtype=np.int64),
        "seq_lengths": np.full((4,), 8, dtype=np.int64),
        "feature_mean": mean,
        "feature_std": std,
    }


def _config_for(path: Path) -> dict:
    return {
        "root": path,
        "feature_schema_version": 2,
        "transition_classes": 64,
        "state_hash_dim": 32,
        "window_size": 8,
        "card_vocab_size": 2000,
        "training": {"games": None, "recent_games": False},
        "require_complete_games": True,
        "objective": {"value_return_gamma": 0.997, "value_shaping_alpha": 0.15},
        "rewards": {"value_win": 1.0, "value_not_win": -1.0, "value_timeout": -2.0},
        "train_tensor_cache_dir": "outputs/cache/training_tensors",
        "train_data_device": "cpu",
    }


def test_tensor_cache_roundtrip(tmp_path: Path) -> None:
    data_path = tmp_path / "rollouts.jsonl"
    data_path.write_text('{"episode": 0, "step": 0}\n', encoding="utf-8")
    config = _config_for(tmp_path)
    fingerprint = build_tensor_cache_fingerprint(data_path, config)
    cache_path = tensor_cache_path(tmp_path, config, fingerprint)
    arrays = _tiny_arrays()

    save_tensor_cache(
        cache_path,
        fingerprint=fingerprint,
        data_path=data_path,
        arrays=arrays,
        transition_classes=64,
        window_size=8,
    )
    assert cache_is_valid(cache_path, fingerprint)

    loaded = load_tensor_cache(
        cache_path,
        data_path=data_path,
        fingerprint=fingerprint,
        transition_classes=64,
        window_size=8,
        device=torch.device("cpu"),
        train_data_device="cpu",
    )
    assert loaded is not None
    assert loaded.num_seqs == 4
    assert torch.allclose(loaded.x_seq, torch.from_numpy(arrays["x_seq"]))


def test_tensor_cache_invalidates_on_data_change(tmp_path: Path) -> None:
    data_path = tmp_path / "rollouts.jsonl"
    data_path.write_text('{"episode": 0}\n', encoding="utf-8")
    config = _config_for(tmp_path)
    fingerprint = build_tensor_cache_fingerprint(data_path, config)
    cache_path = tensor_cache_path(tmp_path, config, fingerprint)
    save_tensor_cache(
        cache_path,
        fingerprint=fingerprint,
        data_path=data_path,
        arrays=_tiny_arrays(),
        transition_classes=64,
        window_size=8,
    )

    data_path.write_text('{"episode": 0}\n{"episode": 1}\n', encoding="utf-8")
    new_fingerprint = build_tensor_cache_fingerprint(data_path, config)
    assert not cache_is_valid(cache_path, new_fingerprint)
    assert (
        load_tensor_cache(
            cache_path,
            data_path=data_path,
            fingerprint=new_fingerprint,
            transition_classes=64,
            window_size=8,
            device=torch.device("cpu"),
            train_data_device="cpu",
        )
        is None
    )


def test_describe_and_has_valid_cache(tmp_path: Path) -> None:
    data_path = tmp_path / "rollouts.jsonl"
    data_path.write_text('{"episode": 0}\n', encoding="utf-8")
    config = _config_for(tmp_path)
    config["training_data_path"] = str(data_path)
    assert has_valid_training_tensor_cache(config) is False
    assert "no tensor cache yet" in (describe_training_tensor_cache(config) or "")

    fingerprint = build_tensor_cache_fingerprint(data_path, config)
    cache_path = tensor_cache_path(tmp_path, config, fingerprint)
    save_tensor_cache(
        cache_path,
        fingerprint=fingerprint,
        data_path=data_path,
        arrays=_tiny_arrays(),
        transition_classes=64,
        window_size=8,
    )
    assert has_valid_training_tensor_cache(config) is True
    assert "tensor cache ready" in (describe_training_tensor_cache(config) or "")
