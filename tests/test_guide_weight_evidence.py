from __future__ import annotations

import hashlib

import pytest

from poke_bot.pure_rl.guide_weight_evidence import compile_schedule


def _digest(path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(tmp_path, *, on_score: float, off_score: float) -> dict:
    on = tmp_path / "guide-on.pt"
    off = tmp_path / "guide-off.pt"
    on.write_bytes(b"on")
    off.write_bytes(b"off")
    rows = []
    for index in range(1000):
        opponent = f"opponent-{index // 100}"
        seat = index % 2
        for variant, checkpoint, score in (
            ("guide_on", on, on_score),
            ("guide_off", off, off_score),
        ):
            rows.append(
                {
                    "variant": variant,
                    "schedule_id": f"pair-{index:04d}",
                    "opponent_id": opponent,
                    "candidate_seat": seat,
                    "requested_seed": 900_000 + index,
                    "checkpoint_sha256": _digest(checkpoint),
                    "score": score,
                    "training_eligible": False,
                    "replay_eligible": False,
                    "formal_gate": False,
                    "invalid": False,
                    "error": None,
                }
            )
    return {
        "schema": "poke_bot.current_deck_guide_paired_evaluation/v1",
        "specialist_id": "archaludon-ex",
        "completed_iteration": 5,
        "current_weight": 0.05,
        "consecutive_nonpositive_evaluations": 0,
        "guide_on_checkpoint": {"path": str(on), "sha256": _digest(on)},
        "guide_off_checkpoint": {"path": str(off), "sha256": _digest(off)},
        "training_eligible": False,
        "replay_eligible": False,
        "formal_gate": False,
        "serving_allowed": False,
        "promotion_allowed": False,
        "rows": rows,
    }


def test_positive_paired_realized_win_evidence_ramps_weight(tmp_path) -> None:
    schedule = compile_schedule(
        _evidence(tmp_path, on_score=1.0, off_score=0.0)
    )
    assert schedule["status"] == "ready_for_clean_boundary"
    assert schedule["previous_state"]["weight"] == 0.05
    assert schedule["next_state"]["weight"] == 0.15
    assert schedule["earliest_activation_boundary_next_iteration"] == 6
    assert schedule["application_boundary"] == (
        "first_available_future_five_iteration_hard_pause"
    )
    assert "activation_boundary_next_iteration" not in schedule
    assert schedule["overall"]["pairs"] == 1000
    assert schedule["overall"]["first_second_balanced"] is True
    assert len(schedule["per_matchup"]) == 10
    assert schedule["guide_off_checkpoint"]["shadow_only"] is True
    assert schedule["serving_allowed"] is False
    assert schedule["promotion_allowed"] is False


def test_first_nonpositive_review_holds_and_records_streak(tmp_path) -> None:
    schedule = compile_schedule(
        _evidence(tmp_path, on_score=0.0, off_score=1.0)
    )
    assert schedule["status"] == "hold"
    assert schedule["next_state"] == {
        "weight": 0.05,
        "consecutive_nonpositive_evaluations": 1,
    }


def test_second_nonpositive_review_decays(tmp_path) -> None:
    evidence = _evidence(tmp_path, on_score=0.0, off_score=1.0)
    evidence["current_weight"] = 0.25
    evidence["consecutive_nonpositive_evaluations"] = 1
    schedule = compile_schedule(evidence)
    assert schedule["status"] == "ready_for_clean_boundary"
    assert schedule["next_state"]["weight"] == 0.15


def test_incomplete_or_training_eligible_pairs_fail_closed(tmp_path) -> None:
    evidence = _evidence(tmp_path, on_score=1.0, off_score=0.0)
    evidence["rows"].pop()
    with pytest.raises(ValueError, match="incomplete"):
        compile_schedule(evidence)
    evidence = _evidence(tmp_path, on_score=1.0, off_score=0.0)
    evidence["rows"][0]["training_eligible"] = True
    with pytest.raises(ValueError, match="violates isolation"):
        compile_schedule(evidence)
