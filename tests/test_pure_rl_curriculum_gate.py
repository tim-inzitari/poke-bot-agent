"""Curriculum, held-out gate, aborts, hardware profile."""

from __future__ import annotations

from dataclasses import replace

import pytest

from poke_bot.pure_rl.aborts import evaluate_aborts
from poke_bot.pure_rl.curriculum import CurriculumStage, stage_for_iteration
from poke_bot.pure_rl.eval_public import aggregate_heldout_wr
from poke_bot.pure_rl.hardware import full_hardware_profile
from poke_bot.pure_rl.shards import (
    CompactDecision,
    CompactGame,
    CompactShardWriter,
    compact_decision_from_step,
    iter_shard_games,
)


def test_stage_core_until_gate() -> None:
    cfg = stage_for_iteration(core_gate_passed=False)
    assert cfg.stage == CurriculumStage.CORE
    assert cfg.our_deck_mode == "multi_archetype"


def test_stage_specialist_after_core_gate() -> None:
    cfg = stage_for_iteration(core_gate_passed=True)
    assert cfg.stage == CurriculumStage.SPECIALIST
    assert cfg.our_deck_mode == "hammer-pult"


def test_heldout_excludes_forfeits_and_gates() -> None:
    rows = []
    for oid in ("iono", "dragapult-ex", "mega-abomasnow-ex", "mega-lucario-ex"):
        for i in range(50):
            seat = i % 2
            rows.append(
                {
                    "opponent_id": oid,
                    "our_seat": seat,
                    "winner": seat,
                    "baseline_failed": False,
                }
            )
    rows.append(
        {
            "opponent_id": "iono",
            "our_seat": 0,
            "winner": 1,
            "baseline_failed": True,
        }
    )
    gate = aggregate_heldout_wr(rows, target_wr=0.70, min_games=200)
    assert gate.games == 200
    assert gate.forfeits_excluded == 1
    assert gate.win_rate == 1.0
    assert gate.passed


def test_abort_self_distill() -> None:
    decision = evaluate_aborts(
        mean_advantages=[0.0, 0.0, 0.0],
        policy_prev_agreements=[0.99, 0.98, 0.97],
        k=3,
    )
    assert decision.abort
    assert decision.self_distill_flag
    assert decision.advantage_signal == 0.0


def test_abort_uses_raw_advantage_magnitude_not_whitened_mean() -> None:
    decision = evaluate_aborts(
        # A whitened signed mean is zero even when useful advantages exist.
        mean_advantages=[0.0, 0.0, 0.0],
        advantage_mean_abs=[0.45, 0.40, 0.35],
        policy_prev_agreements=[0.99, 0.99, 0.99],
        k=3,
    )
    assert not decision.abort
    assert decision.reason == "ok"
    assert decision.advantage_signal == pytest.approx(0.40)


def test_hardware_requires_both_gpus() -> None:
    hw = full_hardware_profile()
    # Defaults already place leaves on both GPUs.
    hw.validate_or_raise(visible_gpu_count=2)
    bad = replace(hw, leaf_gpu0_replicas=0, allow_single_gpu=False)
    with pytest.raises(ValueError, match="GPU0 and GPU1"):
        bad.validate_or_raise(visible_gpu_count=2)
    ok = replace(hw, allow_single_gpu=True, leaf_gpu0_replicas=0)
    ok.validate_or_raise(visible_gpu_count=1)


def test_compact_shard_roundtrip(tmp_path) -> None:
    path = tmp_path / "shard.jsonl"
    writer = CompactShardWriter(path)
    game = CompactGame(
        episode_id="e1",
        seat=0,
        archetype="core",
        opp_archetype="iono",
        deck=[1] * 60,
        value=1.0,
        decisions=[
            compact_decision_from_step(
                {
                    "action": [0],
                    "selected_index": 0,
                    "n_options": 2,
                    "observation": {"x": 1},
                    "aux_labels": {
                        "opp_hand": [7, 8],
                        "privileged_label_source": "training_fork_exact_same_state",
                    },
                }
            )
        ],
    )
    writer.write_game(game)
    games = list(iter_shard_games(path))
    assert len(games) == 1
    assert games[0].decisions[0].selected_index == 0
    assert games[0].decisions[0].aux_labels["opp_hand"] == [7, 8]
    assert games[0].target_provenance.get("soft_policy_targets") is False
