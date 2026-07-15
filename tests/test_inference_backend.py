from __future__ import annotations

from pathlib import Path

import torch

from poke_agent.inference import LocalTorchBackend, create_inference_backend
from poke_agent.policy_agent import PolicyRuntime


def test_create_inference_backend_is_local_torch(tmp_path: Path):
    checkpoint = tmp_path / "missing.pt"
    backend = create_inference_backend(checkpoint, device=torch.device("cpu"))
    assert isinstance(backend, LocalTorchBackend)
    assert isinstance(backend, PolicyRuntime)
    assert backend.device == torch.device("cpu")
    assert backend.checkpoint_path == checkpoint


def test_local_backend_new_session():
    backend = LocalTorchBackend(Path("unused.pt"), device="cpu")
    session = backend.new_session()
    assert session.history == []
