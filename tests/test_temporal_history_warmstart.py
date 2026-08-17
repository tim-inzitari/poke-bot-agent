from __future__ import annotations

from dataclasses import replace

import torch

from poke_bot.dataset import GameSequence
from poke_bot.model import build_model
from poke_bot.pure_rl.model_profile import pure_rl_model_config
from poke_bot.train import cap_game_sequence_context
from scripts.calibrate_temporal_history import _non_temporal_copy_proof
from scripts.warm_start_temporal_history import (
    _single_timestep_parity,
    _zero_temporal_residual_outputs,
)


def test_identity_initialized_temporal_layer_preserves_first_state() -> None:
    state_cfg = pure_rl_model_config(
        d_model=16,
        spatial_layers=1,
        temporal_layers=0,
        option_decoder_layers=1,
        n_heads=4,
        ff_dim=32,
        max_context=8,
        decision_context="stateless",
        kv_cache=False,
        card_embed_dim=8,
        attack_embed_dim=8,
        dropout=0.0,
    )
    teacher = build_model(state_cfg, device=torch.device("cpu"))
    history_cfg = replace(
        state_cfg,
        temporal_layers=1,
        max_context=320,
        decision_context="history",
        kv_cache=True,
    )
    student = build_model(history_cfg, device=torch.device("cpu"))

    teacher_state = teacher.state_dict()
    student_state = student.state_dict()
    for key, target in student_state.items():
        source = teacher_state.get(key)
        if source is not None and tuple(source.shape) == tuple(target.shape):
            student_state[key] = source.detach().clone()
    student.load_state_dict(student_state, strict=True)
    zeroed = _zero_temporal_residual_outputs(student)
    teacher.eval()
    student.eval()

    assert zeroed
    assert _single_timestep_parity(teacher, student) <= 1e-6


def test_temporal_context_cap_keeps_one_game_causal_prefix() -> None:
    decisions = [object() for _ in range(325)]
    sequence = GameSequence(
        episode_id="episode-a",
        seat=1,
        archetype="alakazam",
        opp_archetype="public",
        deck=[],
        value=1.0,
        decisions=decisions,  # type: ignore[arg-type]
        policy_targets=[[float(index)] for index in range(325)],
        factorized_policy_targets=[[{"selected_index": index}] for index in range(325)],
    )

    capped, changed = cap_game_sequence_context(sequence, 320)

    assert changed is True
    assert capped.episode_id == sequence.episode_id
    assert capped.seat == sequence.seat
    assert capped.decisions == decisions[:320]
    assert capped.policy_targets == sequence.policy_targets[:320]
    assert capped.factorized_policy_targets == sequence.factorized_policy_targets[:320]


def test_temporal_calibration_copy_proof_rejects_head_drift() -> None:
    seed = {
        "model_state_dict": {
            "temporal_blocks.0.weight": torch.zeros(2),
            "policy_head.weight": torch.ones(2),
        }
    }
    candidate = {
        "model_state_dict": {
            "temporal_blocks.0.weight": torch.ones(2),
            "policy_head.weight": torch.ones(2),
        }
    }
    proof = _non_temporal_copy_proof(seed, candidate)
    assert proof["changed_frozen_tensors"] == 0

    candidate["model_state_dict"]["policy_head.weight"] = torch.zeros(2)
    try:
        _non_temporal_copy_proof(seed, candidate)
    except RuntimeError as exc:
        assert "policy_head.weight" in str(exc)
    else:
        raise AssertionError("frozen head drift was accepted")
