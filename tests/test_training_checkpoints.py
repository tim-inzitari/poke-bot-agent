from __future__ import annotations

import numpy as np
import torch

from poke_agent.checkpoint import (
    load_training_checkpoint,
    resolve_resume_path,
    training_checkpoint_paths,
)
from poke_agent.dataset import TrainingTensors
from poke_agent.features import CARD_ID_SLOT_COUNT
from poke_agent.models.temporal_transformer import TemporalTransformer
from poke_agent.training import train_model

INPUT_DIM = 16
WINDOW = 4
STEPS = 4
SEQS = 2


def _tiny_tensors() -> TrainingTensors:
    torch.manual_seed(0)
    x_seq = torch.randn(SEQS, WINDOW, INPUT_DIM)
    seq_mask = torch.zeros(SEQS, WINDOW)
    seq_mask[:, WINDOW - STEPS :] = 1.0
    y = torch.randn(SEQS, WINDOW)
    return TrainingTensors(
        data_path=None,
        x_seq=x_seq,
        seq_lengths=torch.full((SEQS,), STEPS, dtype=torch.long),
        seq_mask=seq_mask,
        y=y,
        search_value=y.clone(),
        returns=y.clone(),
        transition_target=torch.zeros(SEQS, WINDOW, dtype=torch.long),
        next_x=torch.randn(SEQS, WINDOW, INPUT_DIM),
        terminal=torch.zeros(SEQS, WINDOW),
        card_ids=torch.zeros(SEQS, WINDOW, CARD_ID_SLOT_COUNT, dtype=torch.long),
        policy_step_weight=torch.ones(SEQS, WINDOW),
        transition_soft_idx=torch.full((SEQS, WINDOW, 8), -1, dtype=torch.long),
        transition_soft_prob=torch.zeros(SEQS, WINDOW, 8),
        feature_mean=np.zeros((1, INPUT_DIM), dtype=np.float32),
        feature_std=np.ones((1, INPUT_DIM), dtype=np.float32),
        transition_classes=4,
        window_size=WINDOW,
    )


def _make_model() -> TemporalTransformer:
    return TemporalTransformer(
        INPUT_DIM,
        4,
        d_model=16,
        nhead=2,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        window_size=WINDOW,
        card_vocab_size=128,
        card_embed_dim=8,
        card_slot_count=CARD_ID_SLOT_COUNT,
    )


def _config(tmp_path, *, epochs: int, checkpoint_every: int) -> dict:
    return {
        "model": {
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "d_model": 16,
            "heads": 2,
            "layers": 1,
            "ff": 32,
            "dropout": 0.0,
        },
        "loss": {"value": 1.0, "policy": 0.0, "dynamics": 0.0, "entropy": 0.0, "uncertainty": 0.0},
        "training": {
            "epochs": epochs,
            "patience": 10_000,  # never early-stop in these tests
            "min_delta": 0.0,
            "early_stop_metric": "value_loss",
            "batch_games": 2,
            "checkpoint_every": checkpoint_every,
        },
        "objective": {"policy_weighting": "none"},
        "rewards": {},
        "beam_search": {},
        "state_hash_dim": 32,
        "card_vocab_size": 128,
        "card_embed_dim": 8,
        "model_id": "ckpt_test",
        "competition_results_path": tmp_path / "no_results.jsonl",
        "train_use_amp": False,
    }


def test_periodic_and_best_checkpoints_written(monkeypatch, tmp_path):
    monkeypatch.setattr("poke_agent.training.assert_generic_model_inputs", lambda *a, **k: None)
    tensors = _tiny_tensors()
    model = _make_model()
    config = _config(tmp_path, epochs=6, checkpoint_every=3)
    out = tmp_path / "model.pt"

    train_model(
        model,
        tensors,
        config,
        torch.device("cpu"),
        checkpoint_path=out,
        checkpoint_every_epochs=3,
    )

    paths = training_checkpoint_paths(out)
    assert paths["latest"].exists(), "expected periodic .latest.pt checkpoint"
    assert paths["best"].exists(), "expected best .best.pt checkpoint"
    # No leftover temp files from the atomic write.
    assert not list(tmp_path.glob("*.tmp"))

    _, training_state = load_training_checkpoint(paths["latest"])
    assert training_state is not None
    assert training_state["completed_epochs"] == 6
    assert "optimizer_state_dict" in training_state


def test_resume_continues_from_latest(monkeypatch, tmp_path):
    monkeypatch.setattr("poke_agent.training.assert_generic_model_inputs", lambda *a, **k: None)
    tensors = _tiny_tensors()
    out = tmp_path / "model.pt"

    first = train_model(
        _make_model(),
        tensors,
        _config(tmp_path, epochs=6, checkpoint_every=2),
        torch.device("cpu"),
        checkpoint_path=out,
        checkpoint_every_epochs=2,
    )
    assert first["completed_epochs"] == 6

    latest = training_checkpoint_paths(out)["latest"]
    resumed = train_model(
        _make_model(),
        tensors,
        _config(tmp_path, epochs=12, checkpoint_every=2),
        torch.device("cpu"),
        checkpoint_path=out,
        checkpoint_every_epochs=2,
        resume_path=latest,
    )
    assert resumed["resumed_from_epoch"] == 6
    assert resumed["completed_epochs"] == 12


def test_resolve_resume_path_modes(tmp_path):
    out = tmp_path / "model.pt"
    latest = training_checkpoint_paths(out)["latest"]

    assert resolve_resume_path(out, "auto") is None  # nothing on disk yet
    assert resolve_resume_path(out, "0") is None

    latest.write_bytes(b"placeholder")
    assert resolve_resume_path(out, "auto") == latest
    assert resolve_resume_path(out, "1") == latest
    assert resolve_resume_path(out, "0") is None
