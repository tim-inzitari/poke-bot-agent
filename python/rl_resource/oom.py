"""CUDA OOM detection and batch-scale guard."""

from __future__ import annotations

from typing import Any, Optional


def is_cuda_oom(exc: BaseException) -> bool:
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


class OomGuard:
    """Catch CUDA OOM → empty_cache → halve scale → retry."""

    def __init__(self, *, min_scale: float = 0.125) -> None:
        self.scale = 1.0
        self.min_scale = float(min_scale)
        self.oom_events = 0

    def scaled(self, n: int) -> int:
        return max(1, int(round(n * self.scale)))

    def handle_oom(self, exc: BaseException, torch_module: Any = None) -> bool:
        if not is_cuda_oom(exc):
            return False
        self.oom_events += 1
        try:
            torch = torch_module or __import__("torch")
            if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        if self.scale <= self.min_scale:
            return False
        self.scale = max(self.min_scale, self.scale * 0.5)
        return True
