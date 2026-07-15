from __future__ import annotations

import torch

from poke_agent.device import (
    device_spec_is_explicit,
    parse_device,
    resolve_infer_device,
    resolve_self_play_inference_device,
    resolve_train_device,
    torch_device,
)


def test_parse_device_auto_tokens():
    assert parse_device(None) is None
    assert parse_device("") is None
    assert parse_device("auto") is None
    assert parse_device("DEFAULT") is None


def test_parse_device_concrete():
    assert parse_device("cpu") == torch.device("cpu")
    assert parse_device("cuda:0") == torch.device("cuda:0")


def test_device_spec_is_explicit():
    assert device_spec_is_explicit("cuda:1")
    assert not device_spec_is_explicit("")
    assert not device_spec_is_explicit("auto")
    assert not device_spec_is_explicit(None)


def test_resolve_train_device_honors_spec(monkeypatch):
    monkeypatch.delenv("TRAIN_DEVICE", raising=False)
    assert resolve_train_device("cpu") == torch.device("cpu")


def test_resolve_train_device_reads_env(monkeypatch):
    monkeypatch.setenv("TRAIN_DEVICE", "cpu")
    assert resolve_train_device() == torch.device("cpu")


def test_resolve_infer_device_falls_back_to_train(monkeypatch):
    monkeypatch.delenv("INFER_DEVICE", raising=False)
    train = torch.device("cpu")
    assert resolve_infer_device(train_device=train) == train


def test_resolve_infer_device_honors_spec(monkeypatch):
    monkeypatch.delenv("INFER_DEVICE", raising=False)
    assert resolve_infer_device("cpu", train_device=torch.device("cuda")) == torch.device("cpu")


def test_self_play_inference_falls_back_to_cpu_when_multiprocess_cuda():
    train = torch.device("cuda:0")
    got = resolve_self_play_inference_device(
        workers=4,
        train_device=train,
        infer_device=train,
        infer_device_explicit=False,
    )
    assert got == torch.device("cpu")


def test_self_play_inference_honors_explicit_infer_device_with_workers():
    infer = torch.device("cuda:1")
    got = resolve_self_play_inference_device(
        workers=4,
        train_device=torch.device("cuda:0"),
        infer_device=infer,
        infer_device_explicit=True,
    )
    assert got == infer


def test_self_play_inference_single_worker_keeps_device():
    train = torch.device("cuda:0")
    got = resolve_self_play_inference_device(
        workers=1,
        train_device=train,
        infer_device=train,
        infer_device_explicit=False,
    )
    assert got == train


def test_torch_device_returns_a_device():
    assert isinstance(torch_device(), torch.device)
