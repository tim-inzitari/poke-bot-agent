from __future__ import annotations

from pathlib import Path

import torch

from poke_agent.policy_agent import PolicyRuntime


class LocalTorchBackend(PolicyRuntime):
    """In-process TemporalTransformer checkpoint scoring."""

    def __init__(self, checkpoint_path: Path, *, device: torch.device | str | None = None):
        super().__init__(checkpoint_path, device=device)
