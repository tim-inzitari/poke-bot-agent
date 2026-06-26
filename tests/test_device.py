import pytest
import torch

from poke_agent.device import pick_largest_cuda_device, torch_device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pick_largest_cuda_device_prefers_most_vram():
    device = pick_largest_cuda_device()
    assert device.type == "cuda"
    index = device.index if device.index is not None else 0
    count = torch.cuda.device_count()
    if count <= 1:
        return
    memories = [torch.cuda.get_device_properties(i).total_memory for i in range(count)]
    assert memories[index] == max(memories)


def test_torch_device_env_override(monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE", "cpu")
    assert torch_device().type == "cpu"
