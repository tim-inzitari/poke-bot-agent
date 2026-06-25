from __future__ import annotations

import torch


def resolve_collection_inference_device(
    configured: str | None,
    *,
    train_device: torch.device | None = None,
) -> torch.device:
    """Pick device for parallel CABT collection workers.

    Default: CPU when GPU VRAM is small (<=16GB) to leave room for training.
    """
    if configured is not None:
        normalized = configured.strip().lower()
        if normalized in {"cpu"}:
            return torch.device("cpu")
        if normalized in {"cuda", "gpu", "device"}:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if normalized not in {"auto", ""}:
            return torch.device(configured)

    if not torch.cuda.is_available():
        return torch.device("cpu")

    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / (1024**3)
    if total_gb < 32:
        return torch.device("cpu")
    return train_device or torch.device("cuda")
