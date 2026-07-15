from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import torch

from poke_agent.policy_agent import PolicySession


class InferenceBackend(Protocol):
    """Score legal CABT actions from a checkpoint (local Torch today)."""

    checkpoint_path: Path
    device: torch.device

    def new_session(self) -> PolicySession: ...

    def choose_action(
        self,
        obs_dict: dict[str, Any],
        session: PolicySession,
        *,
        our_deck: list[int] | None = None,
        use_beam: bool = False,
        beam_config: Any | None = None,
    ) -> list[int]: ...


def create_inference_backend(
    checkpoint_path: Path | str,
    *,
    device: torch.device | str | None = None,
    infer_device_spec: str | None = None,
) -> InferenceBackend:
    """Build the local Torch backend (default / only implementation for now)."""
    from poke_agent.device import resolve_infer_device
    from poke_agent.inference.local import LocalTorchBackend

    resolved = device if device is not None else resolve_infer_device(infer_device_spec)
    return LocalTorchBackend(Path(checkpoint_path), device=resolved)
