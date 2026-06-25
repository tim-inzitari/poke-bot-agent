from __future__ import annotations

import torch
import pytest

from poke_agent.collection_device import (
    resolve_collection_inference_device,
    warn_if_many_cuda_collection_workers,
)


def test_collection_device_forces_cpu():
    assert resolve_collection_inference_device("cpu").type == "cpu"


def test_collection_device_forces_cuda_when_available():
    device = resolve_collection_inference_device("cuda")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert device.type == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_collection_device_auto_follows_train_cuda():
    device = resolve_collection_inference_device(
        "auto",
        train_device=torch.device("cuda"),
    )
    assert device.type == "cuda"


def test_collection_device_auto_cpu_when_no_cuda():
    device = resolve_collection_inference_device("auto", train_device=torch.device("cpu"))
    assert device.type == "cpu"


def test_vram_warning_only_for_many_cuda_workers():
    assert warn_if_many_cuda_collection_workers(
        workers=4,
        inference_device=torch.device("cpu"),
    ) is None
