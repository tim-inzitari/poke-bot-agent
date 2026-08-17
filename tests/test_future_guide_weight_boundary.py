from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import apply_future_guide_weight_at_boundary as boundary


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(
    tmp_path: Path,
    *,
    specialist_id: str = "archaludon-ex",
) -> tuple[dict, str, dict]:
    on = tmp_path / "guide-on.pt"
    off = tmp_path / "guide-off.pt"
    on.write_bytes(b"guide-on")
    off.write_bytes(b"guide-off")
    registry = {
        "specialists": {
            specialist_id: {
                "guide_loss_weight": 0.05,
                "guide_weight_policy": {
                    "schema": "poke_bot.current_deck_guide_weight_policy/v1",
                    "prospective_scope_revision": 44,
                    "scope": "future_specialist_training_runs_only",
                    "retroactive_application_to_completed_frozen_or_started_runs": False,
                    "historical_weight_or_receipt_rewrite_allowed": False,
                    "consecutive_nonpositive_evaluations": 0,
                },
            }
        }
    }
    selector = (
        f"POKEBOT_ACTIVE_SPECIALIST={specialist_id}\n"
        "PURE_RL_ALLOW_CLEAN_BOUNDARY_DESIGN_MIGRATION=0\n"
        "PURE_RL_BOUNDARY_MIGRATION_REASON_OVERRIDE=default\n"
    )
    schedule = {
        "schema": "poke_bot.current_deck_guide_weight_schedule/v1",
        "status": "ready_for_clean_boundary",
        "specialist_id": specialist_id,
        "completed_iteration": 5,
        "earliest_activation_boundary_next_iteration": 6,
        "application_boundary": (
            "first_available_future_five_iteration_hard_pause"
        ),
        "evidence_sha256": "sha256:" + "a" * 64,
        "guide_on_checkpoint": {"path": str(on), "sha256": _sha(on)},
        "guide_off_checkpoint": {
            "path": str(off),
            "sha256": _sha(off),
            "shadow_only": True,
            "serving_allowed": False,
            "promotion_allowed": False,
        },
        "overall": {
            "realized_win_rate_delta_lower_confidence_bound": 0.01,
        },
        "previous_state": {
            "weight": 0.05,
            "consecutive_nonpositive_evaluations": 0,
        },
        "next_state": {
            "weight": 0.15,
            "consecutive_nonpositive_evaluations": 0,
        },
        "changed": True,
        "training_eligible": False,
        "replay_eligible": False,
        "formal_gate": False,
        "serving_allowed": False,
        "promotion_allowed": False,
    }
    return registry, selector, schedule


def test_future_schedule_changes_only_active_future_weight(
    tmp_path: Path,
) -> None:
    registry, selector, schedule = _inputs(tmp_path)
    registry["specialists"]["frozen-old"] = {
        "guide_loss_weight": 0.05,
        "status": "passed_frozen",
    }

    specialist, old, new, staged = boundary.validate_and_stage_registry(
        registry,
        selector,
        schedule,
    )

    assert specialist == "archaludon-ex"
    assert (old, new) == (0.05, 0.15)
    assert staged["specialists"]["archaludon-ex"]["guide_loss_weight"] == 0.15
    assert staged["specialists"]["frozen-old"] == registry["specialists"][
        "frozen-old"
    ]
    original = json.loads(json.dumps(staged))
    original["specialists"]["archaludon-ex"]["guide_loss_weight"] = old
    assert original == registry


def test_future_schedule_refuses_teal_and_fabricated_jump(
    tmp_path: Path,
) -> None:
    registry, selector, schedule = _inputs(
        tmp_path,
        specialist_id="teal-mask-ogerpon-ex",
    )
    with pytest.raises(RuntimeError, match="not authorized"):
        boundary.validate_and_stage_registry(registry, selector, schedule)


def test_future_hold_persists_nonpositive_review_state(
    tmp_path: Path,
) -> None:
    registry, selector, schedule = _inputs(tmp_path)
    schedule["status"] = "hold"
    schedule["changed"] = False
    schedule["overall"][
        "realized_win_rate_delta_lower_confidence_bound"
    ] = -0.01
    schedule["next_state"] = {
        "weight": 0.05,
        "consecutive_nonpositive_evaluations": 1,
    }
    specialist, old, new, staged = boundary.validate_and_stage_registry(
        registry,
        selector,
        schedule,
    )
    assert specialist == "archaludon-ex"
    assert old == new == 0.05
    assert staged["specialists"][specialist]["guide_weight_policy"][
        "consecutive_nonpositive_evaluations"
    ] == 1

    registry, selector, schedule = _inputs(tmp_path)
    schedule["next_state"]["weight"] = 0.50
    with pytest.raises(RuntimeError, match="not authorized"):
        boundary.validate_and_stage_registry(registry, selector, schedule)


def test_hard_pause_boundary_requires_open_pause_marker(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "commits").mkdir(parents=True)
    checkpoint = run_dir / "iter5.pt"
    checkpoint.write_bytes(b"model")
    loop = {
        "last_completed_iteration": 5,
        "next_iteration": 6,
        "learner": {"path": str(checkpoint), "digest": _sha(checkpoint)},
    }
    (run_dir / "loop_state.json").write_text(
        json.dumps(loop),
        encoding="utf-8",
    )
    (run_dir / "commits/iter_00005.json").write_text(
        json.dumps(loop),
        encoding="utf-8",
    )
    log = run_dir / "run.log"
    log.write_text(
        "[pure_rl] GATE_BOUNDARY_HARD_PAUSE iteration=5 seconds=30.0 "
        "stage_gate_passed=false next_collection_blocked=true\n",
        encoding="utf-8",
    )

    proof = boundary.hard_pause_boundary(
        run_dir,
        log,
        earliest_completed_iteration=5,
    )
    assert proof is not None
    assert proof[0] == 5

    with log.open("a", encoding="utf-8") as stream:
        stream.write(
            "[pure_rl] GATE_BOUNDARY_HARD_PAUSE_COMPLETE iteration=5 "
            "next_collection_blocked=false\n"
        )
    assert (
        boundary.hard_pause_boundary(
            run_dir,
            log,
            earliest_completed_iteration=5,
        )
        is None
    )
