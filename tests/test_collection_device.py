from __future__ import annotations

import torch

from poke_agent.collection_device import resolve_collection_inference_device


def test_collection_device_forces_cpu():
    assert resolve_collection_inference_device("cpu").type == "cpu"


def test_collection_device_forces_cuda_when_available():
    device = resolve_collection_inference_device("cuda")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert device.type == expected
