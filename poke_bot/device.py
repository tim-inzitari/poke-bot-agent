"""GPU/device selection.

Hardware on this box (``nvidia-smi -L``):
  - GPU 0: RTX 3080 Ti          -> parallel / batched MCTS leaf eval
  - GPU 1: RTX PRO 5000 Blackwell -> primary network training + batched inference

Policy:
  - Training prefers the Blackwell (matched by name).
  - Leaf/eval batching prefers the 3080 Ti when a second GPU exists.
  - We NEVER silently fall back to CPU when CUDA is available: if a caller asks
    for a CUDA device and CUDA is present, a CUDA device is returned. CPU is
    only returned when CUDA is genuinely unavailable (or explicitly requested).

``torch`` is imported lazily so the rest of the foundation (features, cg_env,
archetypes) stays importable in a torch-less environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str


def _torch():
    import torch  # local import; heavy dependency

    return torch


def cuda_available() -> bool:
    """True if torch is importable and reports at least one CUDA device."""
    try:
        return _torch().cuda.is_available()
    except Exception:
        return False


def mps_available() -> bool:
    """True if Apple Metal Performance Shaders (MPS) is usable."""
    try:
        torch = _torch()
        mps = getattr(torch.backends, "mps", None)
        return bool(mps is not None and mps.is_available())
    except Exception:
        return False


def list_gpus() -> list[GpuInfo]:
    """Return info for every visible CUDA device (empty if none/torch missing)."""
    if not cuda_available():
        return []
    torch = _torch()
    return [
        GpuInfo(index=i, name=torch.cuda.get_device_name(i))
        for i in range(torch.cuda.device_count())
    ]


def find_gpu_by_name(substr: str) -> Optional[int]:
    """Return the index of the first GPU whose name contains ``substr`` (case-insensitive)."""
    substr = substr.lower()
    for g in list_gpus():
        if substr in g.name.lower():
            return g.index
    return None


def _device(index: int):
    return _torch().device(f"cuda:{index}")


def training_device(prefer_name: str = "Blackwell", *, allow_cpu: bool = False):
    """Device for training the network. Prefers the Blackwell GPU.

    Raises ``RuntimeError`` if CUDA is unavailable and ``allow_cpu`` is False.
    """
    if cuda_available():
        idx = find_gpu_by_name(prefer_name)
        if idx is None:
            idx = 0  # any CUDA device beats CPU for training
        return _device(idx)
    # Mac (bert): native MPS when CUDA is absent — never silently train on CPU
    # if Metal is available.
    if mps_available():
        return _torch().device("mps")
    if not allow_cpu:
        raise RuntimeError(
            "CUDA/MPS is not available but training was requested. Pass allow_cpu=True "
            "to override (strongly discouraged for training)."
        )
    return _torch().device("cpu")


def leaf_eval_device(prefer_name: str = "3080", *, allow_cpu: bool = True):
    """Device for batched MCTS leaf evaluation.

    Prefers the 3080 Ti when present so it can run in parallel with training on
    the Blackwell; falls back to any CUDA device, then Apple MPS (bert), then CPU
    (leaf eval on Kaggle may legitimately be CPU-only).
    """
    if cuda_available():
        idx = find_gpu_by_name(prefer_name)
        if idx is None:
            # If only one GPU, share it with training rather than crash.
            idx = 0
        return _device(idx)
    if mps_available():
        return _torch().device("mps")
    if not allow_cpu:
        raise RuntimeError("CUDA/MPS unavailable for leaf eval and allow_cpu=False.")
    return _torch().device("cpu")


def inference_device(*, allow_cpu: bool = True):
    """Device for submission/inference. Prefers Blackwell, then any CUDA, then MPS, then CPU.

    Kaggle may provide no GPU, so CPU is allowed by default here. On Mac bert,
    prefer native MPS over CPU when CUDA is absent.
    """
    if cuda_available():
        idx = find_gpu_by_name("Blackwell")
        if idx is None:
            idx = 0
        return _device(idx)
    if mps_available():
        return _torch().device("mps")
    if not allow_cpu:
        raise RuntimeError("CUDA/MPS unavailable for inference and allow_cpu=False.")
    return _torch().device("cpu")


def describe() -> str:
    """Human-readable summary of the CUDA/MPS situation (for logging)."""
    if cuda_available():
        gpus = list_gpus()
        return "CUDA available: " + ", ".join(f"[{g.index}] {g.name}" for g in gpus)
    if mps_available():
        return "CUDA unavailable; Apple MPS available"
    return "CUDA/MPS unavailable (CPU only)"
