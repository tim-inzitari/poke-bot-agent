from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class HybridAgent:
    name: str
    mode: str
    neural: Callable[[dict[str, Any]], list[int]] | None
    heuristic: Callable[[dict[str, Any]], list[int]]
    confidence_threshold: float = 0.0

    def __call__(self, obs_dict: dict[str, Any]) -> list[int]:
        if self.neural is None:
            return self.heuristic(obs_dict)

        if self.mode == "heuristic_first":
            return self.heuristic(obs_dict)

        if self.mode == "neural_first":
            return self.neural(obs_dict)

        if self.mode == "fallback":
            try:
                return self.neural(obs_dict)
            except Exception:
                return self.heuristic(obs_dict)

        raise ValueError(f"Unknown hybrid mode: {self.mode}")
