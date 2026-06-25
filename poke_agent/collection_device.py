from __future__ import annotations

import torch


def resolve_collection_inference_device(
    configured: str | None,
    *,
    train_device: torch.device | None = None,
) -> torch.device:
    """Pick device for CABT collection/eval inference (transformer forward pass).

    Features are encoded in RAM (numpy); only the forward pass uses this device —
    the same pattern as training with ``TRAIN_DATA_DEVICE=cpu``.

    auto: use ``train_device`` when it is CUDA, else CUDA if available, else CPU.
    cpu: leave VRAM free during collection (all workers on CPU).
    cuda: GPU inference in each worker process.
    """
    if configured is not None:
        normalized = configured.strip().lower()
        if normalized in {"cpu"}:
            return torch.device("cpu")
        if normalized in {"cuda", "gpu", "device"}:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if normalized not in {"auto", ""}:
            return torch.device(configured)

    if train_device is not None and train_device.type == "cuda" and torch.cuda.is_available():
        return train_device
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def warn_if_many_cuda_collection_workers(
    *,
    workers: int,
    inference_device: torch.device,
    max_workers_before_warn: int = 8,
) -> str | None:
    """Return a warning string when many GPU workers may contend for VRAM."""
    if inference_device.type != "cuda" or not torch.cuda.is_available():
        return None
    if workers <= max_workers_before_warn:
        return None
    props = torch.cuda.get_device_properties(inference_device)
    total_gb = props.total_memory / (1024**3)
    if total_gb >= 32:
        return None
    return (
        f"collection: {workers} GPU workers on {total_gb:.0f}GB VRAM — "
        "if you hit OOM, set COLLECTION_INFERENCE_DEVICE=cpu or reduce --workers"
    )
