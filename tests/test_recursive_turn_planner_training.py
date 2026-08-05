"""Unit tests for Recursive Turn Planner / PokeRLM shadow training."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from poke_bot.poke_rlm.model_core import PokeRLMModelCore
from poke_bot.poke_rlm.training.losses import compute_poke_rlm_losses
from poke_bot.poke_rlm.training.labels import PlanSupervisionLabels
from poke_bot.poke_rlm.training.shadow_train import (
    PokeRLMTrainConfig,
    load_poke_rlm_core,
    train_poke_rlm_shadow,
)
from poke_bot.poke_rlm.config import PokeRLMConfig
from poke_bot.recursive_turn_planner.training import (
    load_rtp_checkpoint,
    make_synthetic_batches,
    train_rtp_shadow,
)
from poke_bot.recursive_turn_planner.training.shadow_train import RTPTrainConfig


@pytest.mark.unit
def test_rtp_synthetic_train_and_reload(tmp_path: Path) -> None:
    batches = make_synthetic_batches(n_decisions=24, d_model=96, seed=7)
    result = train_rtp_shadow(
        batches,
        output_dir=tmp_path / "rtp",
        config=RTPTrainConfig(d_model=96, epochs=1, lr=1e-2, seed=7),
    )
    assert Path(result.checkpoint_path).is_file()
    assert Path(result.receipt_path).is_file()
    assert result.metrics["mean_loss"] >= 0.0
    planner = load_rtp_checkpoint(result.checkpoint_path)
    assert int(planner.config.d_model) == 96
    # Forward scores remain finite after load.
    state = batches[0].state
    scores, _ = __import__(
        "poke_bot.recursive_turn_planner.training.shadow_train",
        fromlist=["_action_scores_with_grad"],
    )._action_scores_with_grad(
        planner, state, batches[0].option_hidden, batches[0].legal_actions
    )
    assert torch.isfinite(scores).all()


@pytest.mark.unit
def test_poke_rlm_shadow_train(tmp_path: Path) -> None:
    batches = make_synthetic_batches(n_decisions=16, d_model=96, seed=3)
    result = train_poke_rlm_shadow(
        batches,
        output_dir=tmp_path / "poke",
        config=PokeRLMTrainConfig(d_model=96, epochs=1, seed=3),
    )
    assert Path(result.checkpoint_path).is_file()
    core = load_poke_rlm_core(result.checkpoint_path)
    assert isinstance(core, PokeRLMModelCore)
    state = batches[0].state.unsqueeze(0)
    opts = batches[0].option_hidden.unsqueeze(0)
    heads = core.score_actions(state, opts)
    route = core.route_logits(state)
    assert heads.policy_logits.shape[-1] == opts.size(1)
    assert route.shape[-1] == 3


@pytest.mark.unit
def test_route_label_root_alias() -> None:
    labels = PlanSupervisionLabels(
        chosen_action_index=0,
        route_target="root",
        should_recurse=False,
        stop_reason="ok",
    )
    bundle = compute_poke_rlm_losses(
        action_logits=torch.randn(4),
        route_logits=torch.randn(3),
        recurse_logits=torch.randn(1),
        labels=labels,
    )
    assert torch.isfinite(bundle.total)


@pytest.mark.unit
def test_train_cli_synthetic(tmp_path: Path) -> None:
    import subprocess
    import sys

    out = tmp_path / "run"
    proc = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "train_recursive_turn_planner.py"),
            "--out-dir",
            str(out),
            "--synthetic",
            "--n-synthetic",
            "12",
            "--epochs",
            "1",
            "--also-poke-rlm",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["source"] == "synthetic"
    assert Path(payload["rtp_checkpoint"]).is_file()
    assert Path(payload["poke_rlm_checkpoint"]).is_file()
    assert (out / "experimental" / "pipeline_summary.json").is_file()
