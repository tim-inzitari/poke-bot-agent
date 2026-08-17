from __future__ import annotations

from pathlib import Path

import pytest

from scripts.handoff_baseline50_at_boundary import (
    inherited_official_heldout,
    render_unit,
)


ROOT = Path(__file__).resolve().parents[1]


def test_render_unit_starts_new_lineage_without_weakening_guards() -> None:
    old = "pure_rl_core_continuous_rehearsal_v6_20260719"
    new = "pure_rl_core_baseline50_v7_20260720"
    source = (
        ROOT / "deploy/systemd/pokebot-pure-rl-continuous-rehearsal.service"
    ).read_text()
    rendered = render_unit(
        source,
        old_run=old,
        new_run=new,
        checkpoint=Path(f"/proof/{old}/heldout.pt"),
        replay_shard=Path(f"/proof/{old}/iter_00027.jsonl"),
    )
    assert f"--run-name {old}" not in rendered
    assert rendered.count(new) >= 2
    assert f"--base-checkpoint /proof/{old}/heldout.pt" in rendered
    assert f"--initial-learner-checkpoint /proof/{old}/heldout.pt" in rendered
    assert f"--initial-replay-shard /proof/{old}/iter_00027.jsonl" in rendered
    assert "Environment=PURE_RL_SELF_PLAY_FRAC=0.50" in rendered
    assert "--train-max-decisions-per-batch 12288" in rendered
    assert "MemoryMax=112G" in rendered


def _state() -> dict:
    digest = "sha256:" + "a" * 64
    audit = {
        "passed": True,
        "checkpoint_digest": digest,
        "valid_games": 1000,
        "exact_distribution": True,
        "exact_weights": True,
        "greedy_required": True,
        "per_opponent": {},
    }
    return {
        "heldout_champion": {"path": "/proof/heldout.pt", "digest": digest},
        "heldout_champion_evidence": {
            "checkpoint_digest": digest,
            "iteration": 26,
            "games": 1000,
            "win_rate": 0.299,
            "audit": audit,
        },
        "history": [
            {
                "iteration": 26,
                "raw_heldout_gate": {
                    "games": 1000,
                    "win_rate": 0.299,
                    "confidence_lower": 0.2714,
                    "confidence_upper": 0.3281,
                    "passed": False,
                    "reason": "per_opponent_floor",
                    "per_opponent": {},
                },
            }
        ],
    }


def test_inherited_official_holdout_requires_three_way_reconciliation() -> None:
    result = inherited_official_heldout(_state())
    assert result["games"] == 1000
    assert result["wr"] == 0.299
    assert result["audit_passed"] is True

    broken = _state()
    broken["heldout_champion_evidence"]["games"] = 999
    with pytest.raises(RuntimeError, match="do not reconcile"):
        inherited_official_heldout(broken)
