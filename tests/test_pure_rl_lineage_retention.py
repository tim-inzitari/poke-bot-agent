from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from poke_bot.pure_rl.artifact_retention import apply_artifact_retention
from poke_bot.pure_rl.eval_public import (
    OFFICIAL_BASELINE_IDS,
    heldout_exploration_decision,
    heldout_goal_rank,
)
from poke_bot.pure_rl.expert_rehearsal import (
    carry_learner_candidate,
    continuous_learner_carry_decision,
    rehearsal_due,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_train_pure_rl():
    path = ROOT / "scripts" / "train_pure_rl.py"
    spec = importlib.util.spec_from_file_location("train_pure_rl_lineage_test", path)
    assert spec is not None and spec.loader is not None
    sys.modules.pop("train_pure_rl_lineage_test", None)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rehearsal_cadence_is_before_each_fifth_iteration() -> None:
    assert rehearsal_due(0, 5) is False
    assert rehearsal_due(4, 5) is False
    assert rehearsal_due(5, 5) is True
    assert rehearsal_due(10, 5) is True
    assert rehearsal_due(10, 0) is False


def test_continuous_learner_carries_safe_rejections_and_rolls_back_collapse() -> None:
    assert carry_learner_candidate({"valid": True, "passed": False, "wr": 0.44}, abort=False) == (
        True,
        "continuous_learner",
    )
    assert carry_learner_candidate({"valid": True, "passed": True, "wr": 0.55}, abort=False) == (
        True,
        "promoted",
    )
    assert carry_learner_candidate({"valid": True, "passed": False, "wr": 0.34}, abort=False)[0] is False
    assert carry_learner_candidate({"valid": True, "passed": True, "wr": 0.80}, abort=True)[0] is False


def test_continuous_learner_is_independent_from_heldout_record_selection() -> None:
    assert continuous_learner_carry_decision(
        candidate_safety_ok=True,
        candidate_safety_reason="continuous_learner",
        heldout_audit_ok=True,
        promoted=False,
    ) == (True, "continuous_learner_safety_carry")
    assert continuous_learner_carry_decision(
        candidate_safety_ok=True,
        candidate_safety_reason="promoted",
        heldout_audit_ok=True,
        promoted=True,
    ) == (True, "promoted_safety_carry")
    assert continuous_learner_carry_decision(
        candidate_safety_ok=False,
        candidate_safety_reason="head_to_head_below_floor",
        heldout_audit_ok=True,
        promoted=False,
    ) == (False, "head_to_head_below_floor")
    assert continuous_learner_carry_decision(
        candidate_safety_ok=True,
        candidate_safety_reason="continuous_learner",
        heldout_audit_ok=False,
        promoted=False,
    ) == (False, "heldout_contract_audit_failed")


def _heldout_evidence(win_rates: dict[str, float]) -> dict:
    per_opponent = {}
    total_wins = 0
    for opponent_id in OFFICIAL_BASELINE_IDS:
        games = 250
        wins = int(round(float(win_rates[opponent_id]) * games))
        total_wins += wins
        per_opponent[opponent_id] = {
            "games": float(games),
            "wins": float(wins),
            "win_rate": wins / games,
        }
    return {
        "games": 1000,
        "win_rate": total_wins / 1000,
        "confidence_lower": max(0.0, total_wins / 1000 - 0.03),
        "per_opponent": per_opponent,
    }


def test_heldout_goal_rank_prioritizes_the_weakest_official_matchup() -> None:
    balanced_progress = _heldout_evidence(
        {
            "iono": 0.30,
            "dragapult-ex": 0.45,
            "mega-abomasnow-ex": 0.45,
            "mega-lucario-ex": 0.45,
        }
    )
    pooled_but_brittle = _heldout_evidence(
        {
            "iono": 0.10,
            "dragapult-ex": 0.75,
            "mega-abomasnow-ex": 0.75,
            "mega-lucario-ex": 0.75,
        }
    )
    assert pooled_but_brittle["win_rate"] > balanced_progress["win_rate"]
    assert heldout_goal_rank(balanced_progress) > heldout_goal_rank(
        pooled_but_brittle
    )


def test_heldout_goal_rank_rejects_pooled_only_legacy_evidence() -> None:
    exact = _heldout_evidence({opponent_id: 0.20 for opponent_id in OFFICIAL_BASELINE_IDS})
    pooled_only = {"games": 1000, "win_rate": 0.90, "confidence_lower": 0.88}
    assert heldout_goal_rank(exact) > heldout_goal_rank(pooled_only)


def test_exploratory_learner_keeps_broad_gain_with_one_game_weakest_noise() -> None:
    anchor = _heldout_evidence(
        {
            "iono": 0.172,
            "dragapult-ex": 0.360,
            "mega-abomasnow-ex": 0.360,
            "mega-lucario-ex": 0.568,
        }
    )
    candidate = _heldout_evidence(
        {
            "iono": 0.168,
            "dragapult-ex": 0.416,
            "mega-abomasnow-ex": 0.480,
            "mega-lucario-ex": 0.568,
        }
    )
    decision = heldout_exploration_decision(candidate, anchor)
    assert decision["eligible"] is True
    assert decision["reason"] == "pareto_noninferior_progress"
    assert decision["per_opponent_delta"]["iono"] == pytest.approx(-0.004)


def test_exploratory_learner_rejects_material_weak_matchup_regression() -> None:
    anchor = _heldout_evidence(
        {
            "iono": 0.172,
            "dragapult-ex": 0.360,
            "mega-abomasnow-ex": 0.360,
            "mega-lucario-ex": 0.568,
        }
    )
    candidate = _heldout_evidence(
        {
            "iono": 0.112,
            "dragapult-ex": 0.412,
            "mega-abomasnow-ex": 0.372,
            "mega-lucario-ex": 0.604,
        }
    )
    decision = heldout_exploration_decision(candidate, anchor)
    assert decision["eligible"] is False
    assert decision["reason"] == "per_opponent_regression_exceeds_margin"
    assert decision["violating_opponents"] == ["iono"]


def test_exploratory_learner_rejects_hidden_single_matchup_collapse() -> None:
    anchor = _heldout_evidence(
        {opponent_id: 0.40 for opponent_id in OFFICIAL_BASELINE_IDS}
    )
    candidate = _heldout_evidence(
        {
            "iono": 0.50,
            "dragapult-ex": 0.50,
            "mega-abomasnow-ex": 0.50,
            "mega-lucario-ex": 0.20,
        }
    )
    decision = heldout_exploration_decision(candidate, anchor)
    assert decision["eligible"] is False
    assert decision["violating_opponents"] == ["mega-lucario-ex"]


def test_exploratory_learner_requires_complete_exact_evidence() -> None:
    anchor = _heldout_evidence(
        {opponent_id: 0.20 for opponent_id in OFFICIAL_BASELINE_IDS}
    )
    candidate = _heldout_evidence(
        {opponent_id: 0.30 for opponent_id in OFFICIAL_BASELINE_IDS}
    )
    del candidate["per_opponent"]["iono"]
    decision = heldout_exploration_decision(candidate, anchor)
    assert decision == {
        "eligible": False,
        "reason": "missing_or_invalid_exact_per_opponent_evidence",
    }


def _external_seed_report(digest: str) -> dict:
    rates = {
        "iono": 0.39,
        "dragapult-ex": 0.55,
        "mega-abomasnow-ex": 0.68,
        "mega-lucario-ex": 0.61,
    }
    matchups = []
    for opponent_id in OFFICIAL_BASELINE_IDS:
        wins = int(250 * rates[opponent_id])
        matchups.append(
            {
                "opponent_id": opponent_id,
                "games": 250,
                "wins": float(wins),
                "losses": float(250 - wins),
                "draws": 0.0,
            }
        )
    return {
        "valid": True,
        "trusted_formal": True,
        "formal_mode": "policy",
        "failures": [],
        "checkpoint": {"digest": digest},
        "expected_opponents": list(OFFICIAL_BASELINE_IDS),
        "scheduled_jobs": 1000,
        "completed_jobs": 1000,
        "min_games_per_opponent": 250,
        "pooled_formal": {
            "games": 1000,
            "wr": sum(rates.values()) / 4,
            "interval_lower": 0.53,
            "interval_upper": 0.59,
        },
        "deck_agnostic_gate": {"exact_deck_seat_balance": True},
        "matchups": matchups,
    }


def test_exact_external_seed_evidence_becomes_nonterminal_retention_anchor(
    tmp_path: Path,
) -> None:
    module = _load_train_pure_rl()
    digest = "sha256:" + "d" * 64
    path = tmp_path / "exact.json"
    path.write_text(json.dumps(_external_seed_report(digest)))
    evidence = module._load_initial_heldout_evidence(
        path,
        checkpoint=SimpleNamespace(digest=digest),
        heldout_games=1000,
    )
    assert evidence["iteration"] == -1
    assert evidence["checkpoint_digest"] == digest
    assert evidence["per_opponent"]["iono"]["win_rate"] == pytest.approx(0.388)
    assert evidence["audit"]["passed"] is True
    assert evidence["audit"]["terminal_gate_eligible"] is False


def test_external_seed_evidence_fails_closed_on_checkpoint_mismatch(
    tmp_path: Path,
) -> None:
    module = _load_train_pure_rl()
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps(_external_seed_report("sha256:" + "d" * 64)))
    with pytest.raises(RuntimeError, match="checkpoint_digest_mismatch"):
        module._load_initial_heldout_evidence(
            path,
            checkpoint=SimpleNamespace(digest="sha256:" + "e" * 64),
            heldout_games=1000,
        )


def test_legacy_heldout_evidence_reconciles_from_exact_committed_history() -> None:
    module = _load_train_pure_rl()
    digest = "sha256:" + "c" * 64
    exact = _heldout_evidence(
        {opponent_id: 0.25 for opponent_id in OFFICIAL_BASELINE_IDS}
    )
    state = {
        "heldout_champion_evidence": {
            "iteration": 3,
            "checkpoint_digest": digest,
            "games": exact["games"],
            "win_rate": exact["win_rate"],
            "confidence_lower": exact["confidence_lower"],
        },
        "history": [
            {
                "iteration": 3,
                "heldout_champion": {"path": "/checkpoints/iter_00003.pt", "digest": digest},
                "heldout_audit": {"passed": True},
                "stage_gate": {
                    "games": exact["games"],
                    "win_rate": exact["win_rate"],
                    "per_opponent": exact["per_opponent"],
                },
            }
        ],
    }
    reconciled = module._reconciled_heldout_champion_evidence(state)
    assert reconciled["evidence_schema"] == 2
    assert reconciled["per_opponent"] == exact["per_opponent"]
    assert reconciled["reconciled_from"] == "committed_history.stage_gate"

    state["history"][0]["heldout_champion"]["digest"] = "sha256:wrong"
    assert "per_opponent" not in module._reconciled_heldout_champion_evidence(state)


def test_retention_keeps_latest_five_and_all_protected_checkpoints(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    for part in ("shards", "checkpoints", "metrics", "artifact_receipts"):
        (run_dir / part).mkdir(parents=True, exist_ok=True)
    for iteration in range(8):
        (run_dir / "shards" / f"iter_{iteration:05d}.jsonl").write_bytes(
            f"shard-{iteration}".encode()
        )
        (run_dir / "checkpoints" / f"iter_{iteration:05d}.pt").write_bytes(
            f"checkpoint-{iteration}".encode()
        )
    expert = run_dir / "checkpoints" / "expert_before_iter_00000.pt"
    expert.write_bytes(b"expert")
    seed = run_dir / "checkpoints" / "seed.pt"
    seed.write_bytes(b"seed")

    def identity(path: Path) -> dict[str, str]:
        return {"path": str(path), "digest": "sha256:test"}

    state = {
        "champion": identity(run_dir / "checkpoints" / "iter_00001.pt"),
        "heldout_champion": identity(run_dir / "checkpoints" / "iter_00002.pt"),
        "learner": identity(run_dir / "checkpoints" / "iter_00004.pt"),
        "lineage_base": identity(seed),
        "opponent_pool": [
            identity(run_dir / "checkpoints" / "iter_00005.pt")
        ],
    }
    report = apply_artifact_retention(
        run_dir,
        state,
        completed_iteration=7,
        replay_window_shards=2,
        history_iterations=5,
    )

    assert report["retired_shards"] == [0, 1, 2]
    assert not (run_dir / "shards" / "iter_00000.jsonl").exists()
    assert (run_dir / "shards" / "iter_00003.jsonl").is_file()
    assert not (run_dir / "checkpoints" / "iter_00000.pt").exists()
    assert (run_dir / "checkpoints" / "iter_00001.pt").is_file()
    assert (run_dir / "checkpoints" / "iter_00002.pt").is_file()
    assert (run_dir / "checkpoints" / "iter_00003.pt").is_file()
    assert report["retired_expert_checkpoints"] == [0]
    assert not expert.exists()
    assert (run_dir / "artifact_receipts" / "shards" / "iter_00000.json").is_file()
    assert (
        run_dir
        / "artifact_receipts"
        / "checkpoints"
        / "expert_before_iter_00000.json"
    ).is_file()


def test_retention_removes_quarantine_only_after_clean_commit(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    attempt = run_dir / "quarantine/iter_00002/attempt_0001"
    attempt.mkdir(parents=True)
    failure = {
        "schema": 1,
        "iteration": 2,
        "reason": "interrupted_before_append_only_commit",
        "artifacts": [{"digest": "sha256:test", "size": 7}],
    }
    (attempt / "failure.json").write_text(json.dumps(failure))
    (attempt / "shards").mkdir()
    (attempt / "shards/iter_00002.jsonl").write_bytes(b"partial")

    before = apply_artifact_retention(
        run_dir,
        {},
        completed_iteration=2,
        replay_window_shards=2,
        history_iterations=5,
    )
    assert before["retired_quarantine_iterations"] == []
    assert attempt.is_dir()

    commit = run_dir / "commits/iter_00002.json"
    commit.parent.mkdir()
    commit.write_text("{}")
    after = apply_artifact_retention(
        run_dir,
        {},
        completed_iteration=2,
        replay_window_shards=2,
        history_iterations=5,
    )
    assert after["retired_quarantine_iterations"] == [2]
    assert after["reclaimed_quarantine_bytes"] > 0
    assert not attempt.exists()
    receipt = run_dir / "artifact_receipts/quarantine/iter_00002.json"
    assert json.loads(receipt.read_text())["failures"][0]["payload"] == failure


def _heldout_rows(digest: str) -> list[dict]:
    rows: list[dict] = []
    for job_index in range(1000):
        pair = job_index // 2
        rows.append(
            {
                "job_index": job_index,
                "opponent_id": OFFICIAL_BASELINE_IDS[pair % 4],
                "our_seat": job_index % 2,
                "winner": job_index % 2,
                "checkpoint_digest": digest,
                "action_selection": "greedy",
                "invalid": False,
            }
        )
    return rows


def test_heldout_audit_requires_all_1000_exact_digest_greedy_balanced_games() -> None:
    module = _load_train_pure_rl()
    digest = "sha256:" + "a" * 64
    rows = _heldout_rows(digest)
    audit = module._audit_heldout_rows(
        rows,
        n_games=1000,
        checkpoint_digest=digest,
    )
    assert audit["passed"] is True
    assert all(cell["games"] == 250 for cell in audit["per_opponent"].values())
    assert all(cell["seat0"] == 125 for cell in audit["per_opponent"].values())
    assert all(cell["seat1"] == 125 for cell in audit["per_opponent"].values())

    rows[700]["checkpoint_digest"] = "sha256:" + "b" * 64
    assert module._audit_heldout_rows(
        rows,
        n_games=1000,
        checkpoint_digest=digest,
    )["passed"] is False
