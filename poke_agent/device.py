from __future__ import annotations

import os

import torch


def _env_device() -> torch.device | None:
    """Optional explicit device: TORCH_DEVICE or CUDA_DEVICE (e.g. cuda:1, cuda)."""
    raw = os.environ.get("TORCH_DEVICE") or os.environ.get("CUDA_DEVICE")
    if not raw or not str(raw).strip():
        return None
    return torch.device(str(raw).strip())


def pick_largest_cuda_device() -> torch.device:
    """Use the GPU with the most VRAM when multiple CUDA devices are visible."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    count = torch.cuda.device_count()
    if count <= 1:
        return torch.device("cuda")
    best_index = max(
        range(count),
        key=lambda index: int(torch.cuda.get_device_properties(index).total_memory),
    )
    return torch.device(f"cuda:{best_index}")


def torch_device() -> torch.device:
    explicit = _env_device()
    if explicit is not None:
        return explicit
    if torch.cuda.is_available():
        return pick_largest_cuda_device()
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_torch_device(device: torch.device) -> str:
    if device.type != "cuda" or not torch.cuda.is_available():
        return str(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    total_gb = props.total_memory / (1024**3)
    return f"cuda:{index} ({props.name}, {total_gb:.0f}GB)"
