from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import torch

from poke_agent.dataset import TrainingTensors
from poke_agent.features import CARD_ID_SLOT_COUNT
from poke_agent.models.temporal_transformer import TemporalTransformer
from poke_agent.training import train_model


def _tiny_tensors(*, seqs: int = 2, steps: int = 4, input_dim: int = 16, window: int = 4) -> TrainingTensors:
    x_seq = torch.randn(seqs, window, input_dim)
    seq_mask = torch.zeros(seqs, window)
    for seq in range(seqs):
        seq_mask[seq, window - steps :] = 1.0
    y = torch.randn(seqs, window)
    returns = y.clone()
    transition = torch.zeros(seqs, window, dtype=torch.long)
    next_x = torch.randn(seqs, window, input_dim)
    terminal = torch.zeros(seqs, window)
    card_ids = torch.zeros(seqs, window, CARD_ID_SLOT_COUNT, dtype=torch.long)
    seq_lengths = torch.full((seqs,), steps, dtype=torch.long)
    feature_mean = np.zeros((1, input_dim), dtype=np.float32)
    feature_std = np.ones((1, input_dim), dtype=np.float32)
    return TrainingTensors(
        data_path=None,
        x_seq=x_seq,
        seq_lengths=seq_lengths,
        seq_mask=seq_mask,
        y=y,
        returns=returns,
        transition_target=transition,
        next_x=next_x,
        terminal=terminal,
        card_ids=card_ids,
        policy_step_weight=torch.ones(seqs, window),
        transition_soft_idx=torch.full((seqs, window, 8), -1, dtype=torch.long),
        transition_soft_prob=torch.zeros(seqs, window, 8),
        feature_mean=feature_mean,
        feature_std=feature_std,
        transition_classes=4,
        window_size=window,
    )


def test_early_stop_triggers_on_value_loss_plateau(monkeypatch):
    """Stops once value_loss fails to improve by min_delta for patience epochs."""
    monkeypatch.setattr("poke_agent.training.assert_generic_model_inputs", lambda *args, **kwargs: None)

    tensors = _tiny_tensors()
    model = TemporalTransformer(
        tensors.x_seq.shape[-1],
        4,
        d_model=16,
        nhead=2,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        window_size=tensors.window_size,
        card_vocab_size=128,
        card_embed_dim=8,
        card_slot_count=CARD_ID_SLOT_COUNT,
    )
    config = {
        "model": {
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "d_model": 16,
            "heads": 2,
            "layers": 1,
            "ff": 32,
            "dropout": 0.0,
        },
        "loss": {
            "value": 1.0,
            "policy": 0.0,
            "dynamics": 0.0,
            "entropy": 0.0,
            "uncertainty": 0.0,
        },
        "training": {
            "epochs": 100,
            "patience": 3,
            "min_delta": 0.01,
            "early_stop_metric": "value_loss",
            "batch_games": 2,
        },
        "objective": {"policy_weighting": "none"},
        "rewards": {},
        "beam_search": {},
        "state_hash_dim": 32,
        "card_vocab_size": 128,
        "train_use_amp": False,
    }

    value_losses = iter([0.5, 0.4, 0.39, 0.385, 0.384, 0.383])

    def fake_mse_forward(pred, target):
        value = next(value_losses)
        return torch.full(pred.shape, value, dtype=pred.dtype, device=pred.device, requires_grad=True)

    mock_mse = MagicMock(side_effect=fake_mse_forward)
    monkeypatch.setattr("poke_agent.training.torch.nn.MSELoss", lambda reduction="mean": mock_mse)

    report = train_model(model, tensors, config, torch.device("cpu"))
    assert report["stopped_early"] is True
    assert report["early_stop_metric"] == "value_loss"
    assert report["completed_epochs"] < 100
