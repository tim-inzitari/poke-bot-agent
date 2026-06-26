from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from poke_agent.dataset import TrainingTensors, _to_device_tensor, resolve_data_device
from poke_agent.features import FEATURE_SCHEMA_VERSION

TENSOR_CACHE_FORMAT_VERSION = 2

_ARRAY_FILES = (
    "x_seq",
    "y",
    "returns",
    "transition_target",
    "next_x",
    "terminal",
    "seq_mask",
    "card_ids",
    "policy_step_weight",
    "transition_soft_idx",
    "transition_soft_prob",
    "seq_lengths",
    "feature_mean",
    "feature_std",
)


def resolve_tensor_cache_mode(mode: str | None) -> str:
    normalized = (mode or "auto").strip().lower()
    if normalized in {"0", "false", "no", "off", "none", "disable", "disabled"}:
        return "off"
    if normalized in {"1", "true", "yes", "on", "require", "force"}:
        return "require"
    if normalized in {"rebuild", "refresh", "invalidate"}:
        return "rebuild"
    return "auto"


def build_tensor_cache_fingerprint(data_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    stat = data_path.stat()
    reward_cfg = config.get("rewards", {})
    objective_cfg = config.get("objective", {})
    train_cfg = config.get("training", {})
    return {
        "cache_format_version": TENSOR_CACHE_FORMAT_VERSION,
        "feature_schema_version": int(config.get("feature_schema_version", FEATURE_SCHEMA_VERSION)),
        "data_path": str(data_path.resolve()),
        "data_size": int(stat.st_size),
        "data_mtime_ns": int(stat.st_mtime_ns),
        "transition_classes": int(config["transition_classes"]),
        "state_hash_dim": int(config["state_hash_dim"]),
        "window_size": int(config["window_size"]),
        "card_vocab_size": int(config.get("card_vocab_size", 2000)),
        "train_games": train_cfg.get("games"),
        "recent_games": bool(train_cfg.get("recent_games", False)),
        "require_complete_games": bool(config.get("require_complete_games", True)),
        "value_gamma": float(objective_cfg.get("value_return_gamma", 0.997)),
        "value_shaping_alpha": float(objective_cfg.get("value_shaping_alpha", 0.15)),
        "value_win": float(reward_cfg.get("value_win", 1.0)),
        "value_not_win": float(reward_cfg.get("value_not_win", -1.0)),
        "value_timeout": float(reward_cfg.get("value_timeout", -2.0)),
        "value_lambda": float(objective_cfg.get("value_lambda", 0.9)),
        "value_mc_blend": float(objective_cfg.get("value_mc_blend", 0.6)),
        "policy_soft_topk": int(objective_cfg.get("policy_soft_topk", 8)),
        "value_archetype_shaping_weight_lucario": float(
            objective_cfg.get("value_archetype_shaping_weight_lucario", 0.0)
        ),
        "value_archetype_shaping_weight_dragapult": float(
            objective_cfg.get("value_archetype_shaping_weight_dragapult", 0.0)
        ),
        "value_archetype_shaping_weight_abomasnow": float(
            objective_cfg.get("value_archetype_shaping_weight_abomasnow", 0.0)
        ),
        "value_archetype_shaping_weight_iono": float(
            objective_cfg.get("value_archetype_shaping_weight_iono", 0.0)
        ),
        "value_archetype_shaping_weight_starmie": float(
            objective_cfg.get("value_archetype_shaping_weight_starmie", 0.0)
        ),
        "value_archetype_shaping_weight_crustle": float(
            objective_cfg.get("value_archetype_shaping_weight_crustle", 0.0)
        ),
    }


def tensor_cache_dir(root: Path, config: dict[str, Any]) -> Path:
    explicit = config.get("train_tensor_cache_dir")
    if explicit:
        path = Path(explicit)
        return path if path.is_absolute() else root / path
    return root / "outputs" / "cache" / "training_tensors"


def tensor_cache_path(root: Path, config: dict[str, Any], fingerprint: dict[str, Any]) -> Path:
    digest = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return tensor_cache_dir(root, config) / digest


def _manifest_matches(manifest: dict[str, Any], fingerprint: dict[str, Any]) -> bool:
    stored = manifest.get("fingerprint")
    return isinstance(stored, dict) and stored == fingerprint


def _load_manifest(cache_path: Path) -> dict[str, Any] | None:
    manifest_path = cache_path / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _array_path(cache_path: Path, name: str) -> Path:
    return cache_path / f"{name}.npy"


def cache_is_valid(cache_path: Path, fingerprint: dict[str, Any]) -> bool:
    manifest = _load_manifest(cache_path)
    if manifest is None or not _manifest_matches(manifest, fingerprint):
        return False
    return all(_array_path(cache_path, name).is_file() for name in _ARRAY_FILES)


def resolve_training_tensor_cache(config: dict[str, Any]) -> tuple[Path | None, dict[str, Any] | None, Path | None]:
    """Return (data_path, fingerprint, cache_path) when rollout data is configured."""
    from poke_agent.cabt_validation import resolve_training_data_path

    data_path = resolve_training_data_path(config)
    if data_path is None:
        return None, None, None
    root = Path(config["root"])
    fingerprint = build_tensor_cache_fingerprint(data_path, config)
    cache_path = tensor_cache_path(root, config, fingerprint)
    return data_path, fingerprint, cache_path


def has_valid_training_tensor_cache(config: dict[str, Any]) -> bool:
    data_path, fingerprint, cache_path = resolve_training_tensor_cache(config)
    if data_path is None or fingerprint is None or cache_path is None:
        return False
    return cache_is_valid(cache_path, fingerprint)


def describe_training_tensor_cache(config: dict[str, Any]) -> str | None:
    data_path, fingerprint, cache_path = resolve_training_tensor_cache(config)
    if data_path is None or cache_path is None:
        return None
    mode = resolve_tensor_cache_mode(config.get("train_tensor_cache"))
    if mode == "off":
        return f"tensor cache disabled; will build from {data_path.name}"
    if cache_is_valid(cache_path, fingerprint):
        manifest = _load_manifest(cache_path) or {}
        num_seqs = manifest.get("num_seqs", "?")
        return (
            f"tensor cache ready: {cache_path} ({num_seqs} seqs from {data_path.name}); "
            "training will skip JSONL parse + feature build"
        )
    if mode == "require":
        return f"tensor cache required but missing at {cache_path}"
    return f"no tensor cache yet; will build from {data_path.name} and save to {cache_path}"


def try_load_training_tensors_from_cache(
    config: dict[str, Any],
    device: torch.device,
) -> TrainingTensors | None:
    """Load cached tensors when valid; None if cache is off, stale, or rebuild requested."""
    data_path, fingerprint, cache_path = resolve_training_tensor_cache(config)
    if data_path is None or fingerprint is None or cache_path is None:
        return None
    mode = resolve_tensor_cache_mode(config.get("train_tensor_cache"))
    if mode in {"off", "rebuild"}:
        return None
    return load_tensor_cache(
        cache_path,
        data_path=data_path,
        fingerprint=fingerprint,
        transition_classes=int(config["transition_classes"]),
        window_size=int(config["window_size"]),
        device=device,
        train_data_device=config.get("train_data_device"),
    )


def save_tensor_cache(
    cache_path: Path,
    *,
    fingerprint: dict[str, Any],
    data_path: Path,
    arrays: dict[str, np.ndarray],
    transition_classes: int,
    window_size: int,
) -> None:
    """Write normalized training arrays to disk (build once, reuse on resume)."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(cache_path.name + ".tmp")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True)

    started = time.perf_counter()
    for name in _ARRAY_FILES:
        array = np.ascontiguousarray(arrays[name])
        np.save(_array_path(tmp_path, name), array, allow_pickle=False)

    manifest = {
        "fingerprint": fingerprint,
        "data_path": str(data_path),
        "transition_classes": int(transition_classes),
        "window_size": int(window_size),
        "num_seqs": int(arrays["x_seq"].shape[0]),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if cache_path.exists():
        shutil.rmtree(cache_path)
    os.replace(tmp_path, cache_path)
    elapsed = time.perf_counter() - started
    bytes_written = sum(_array_path(cache_path, name).stat().st_size for name in _ARRAY_FILES)
    print(
        f"saved training tensor cache: {cache_path} "
        f"({bytes_written / (1024 ** 3):.2f} GiB, {elapsed:.1f}s)"
    )


def load_tensor_cache(
    cache_path: Path,
    *,
    data_path: Path,
    fingerprint: dict[str, Any],
    transition_classes: int,
    window_size: int,
    device: torch.device,
    train_data_device: str | None,
) -> TrainingTensors | None:
    """Load cached tensors into RAM for training (skips JSONL parse + feature build)."""
    if not cache_is_valid(cache_path, fingerprint):
        return None

    started = time.perf_counter()
    arrays: dict[str, np.ndarray] = {}
    for name in _ARRAY_FILES:
        mmap = np.load(_array_path(cache_path, name), mmap_mode="r")
        arrays[name] = np.array(mmap, copy=True)

    data_device = resolve_data_device(device, train_data_device)
    if data_device != device:
        print(
            f"dataset on {data_device} (compute on {device}); "
            "batches stream to the GPU so dataset size is bounded by RAM, not VRAM"
        )

    tensors = TrainingTensors(
        data_path=data_path,
        x_seq=_to_device_tensor(arrays["x_seq"], data_device),
        seq_lengths=_to_device_tensor(arrays["seq_lengths"], data_device),
        seq_mask=_to_device_tensor(arrays["seq_mask"], data_device),
        y=_to_device_tensor(arrays["y"], data_device),
        returns=_to_device_tensor(arrays["returns"], data_device),
        transition_target=_to_device_tensor(arrays["transition_target"], data_device),
        next_x=_to_device_tensor(arrays["next_x"], data_device),
        terminal=_to_device_tensor(arrays["terminal"], data_device),
        card_ids=_to_device_tensor(arrays["card_ids"], data_device),
        policy_step_weight=_to_device_tensor(arrays["policy_step_weight"], data_device),
        transition_soft_idx=_to_device_tensor(arrays["transition_soft_idx"], data_device),
        transition_soft_prob=_to_device_tensor(arrays["transition_soft_prob"], data_device),
        feature_mean=arrays["feature_mean"],
        feature_std=arrays["feature_std"],
        transition_classes=transition_classes,
        window_size=window_size,
    )
    elapsed = time.perf_counter() - started
    print(
        f"training tensor cache hit: {cache_path} "
        f"({tensors.num_seqs:,} seqs loaded in {elapsed:.1f}s)"
    )
    print(
        "x_seq",
        tuple(tensors.x_seq.shape),
        "value",
        tuple(tensors.y.shape),
        "transition",
        tuple(tensors.transition_target.shape),
        "seqs",
        int(tensors.x_seq.shape[0]),
    )
    print("seq_mask", tuple(tensors.seq_mask.shape), "window", window_size)
    return tensors
