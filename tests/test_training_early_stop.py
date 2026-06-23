from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import torch

from poke_agent.dataset import TrainingTensors
from poke_agent.models.temporal_transformer import TemporalTransformer
from poke_agent.training import train_model


def _tiny_tensors(*, steps: int = 4, input_dim: int = 8, window: int = 2) -> TrainingTensors:
    x = torch.randn(steps, input_dim)
    y = torch.randn(steps, 1)
    transition = torch.zeros(steps, dtype=torch.long)
    next_x = torch.randn(steps, input_dim)
    terminal = torch.zeros(steps, 1)
    history_index = torch.arange(steps).unsqueeze(1).expand(-1, window)
    history_mask = torch.ones(steps, window)
    game_lengths = torch.tensor([2, 2], dtype=torch.long)
    feature_mean = np.zeros((1, input_dim), dtype=np.float32)
    feature_std = np.ones((1, input_dim), dtype=np.float32)
    return TrainingTensors(
        data_path=None,
        x=x,
        x_padded=torch.zeros(steps + 1, input_dim),
        y=y,
        transition_target=transition,
        next_x=next_x,
        terminal=terminal,
        history_index=history_index,
        history_mask=history_mask,
        game_lengths=game_lengths,
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
        tensors.x.shape[1],
        4,
        d_model=16,
        nhead=2,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        window_size=tensors.window_size,
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
        "rewards": {},
        "beam_search": {},
    }

    value_losses = iter([0.5, 0.4, 0.39, 0.385, 0.384, 0.383])

    def fake_mse_forward(pred, target):
        value = next(value_losses)
        return torch.tensor(value, dtype=pred.dtype, device=pred.device, requires_grad=True)

    mock_mse = MagicMock(side_effect=fake_mse_forward)
    monkeypatch.setattr("poke_agent.training.torch.nn.MSELoss", lambda: mock_mse)

    report = train_model(model, tensors, config, torch.device("cpu"))
    assert report["stopped_early"] is True
    assert report["early_stop_metric"] == "value_loss"
    assert report["completed_epochs"] < 100
