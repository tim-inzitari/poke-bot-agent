from __future__ import annotations

import os

import torch

_AUTO_TOKENS = frozenset({"", "auto", "default", "none"})


def torch_device() -> torch.device:
    """Pick the default compute device (CUDA → MPS → CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_device(spec: str | None) -> torch.device | None:
    """Parse a device string. Empty / auto / None → None (caller uses default)."""
    if spec is None:
        return None
    text = str(spec).strip()
    if text.lower() in _AUTO_TOKENS:
        return None
    return torch.device(text)


def device_spec_is_explicit(spec: str | None) -> bool:
    """True when the caller/env pinned a concrete device (not auto)."""
    return parse_device(spec) is not None


def resolve_train_device(spec: str | None = None) -> torch.device:
    """Resolve training device from explicit spec, TRAIN_DEVICE env, or auto."""
    if spec is None:
        spec = os.environ.get("TRAIN_DEVICE")
    parsed = parse_device(spec)
    return parsed if parsed is not None else torch_device()


def resolve_infer_device(
    spec: str | None = None,
    *,
    train_device: torch.device | None = None,
) -> torch.device:
    """Resolve inference device from INFER_DEVICE, else fall back to train device."""
    if spec is None:
        spec = os.environ.get("INFER_DEVICE")
    parsed = parse_device(spec)
    if parsed is not None:
        return parsed
    if train_device is not None:
        return train_device
    return resolve_train_device()


def resolve_self_play_inference_device(
    *,
    workers: int,
    train_device: torch.device,
    infer_device: torch.device | None = None,
    infer_device_explicit: bool = False,
) -> torch.device:
    """
    Device used for PolicyRuntime during self-play collection.

    When INFER_DEVICE is explicitly set, honor it even with multiple workers.
    Otherwise keep the legacy safe default: multiprocess + CUDA → CPU.
    """
    device = infer_device if infer_device is not None else train_device
    if workers > 1 and device.type == "cuda" and not infer_device_explicit:
        return torch.device("cpu")
    return device
