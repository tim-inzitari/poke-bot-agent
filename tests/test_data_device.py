from __future__ import annotations

import torch

from poke_agent.dataset import resolve_data_device

CPU = torch.device("cpu")
CUDA = torch.device("cuda", 0)


def test_auto_streams_from_cpu_when_training_on_cuda():
    assert resolve_data_device(CUDA, "auto") == CPU


def test_auto_keeps_on_device_when_training_on_cpu():
    assert resolve_data_device(CPU, "auto") == CPU


def test_cuda_mode_keeps_dataset_resident_on_gpu():
    assert resolve_data_device(CUDA, "cuda") == CUDA
    assert resolve_data_device(CUDA, "device") == CUDA


def test_cpu_mode_forces_cpu_even_on_cuda():
    assert resolve_data_device(CUDA, "cpu") == CPU


def test_none_defaults_to_auto():
    assert resolve_data_device(CUDA, None) == CPU
