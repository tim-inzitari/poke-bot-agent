from __future__ import annotations

import json
from pathlib import Path

from poke_bot.pure_rl.core_transition import transition_decision
from poke_bot.pure_rl.model_registry import sha256


def _anchor(tmp_path: Path, wr: float = 0.299) -> dict:
    checkpoint = tmp_path / "anchor.pt"
    checkpoint.write_bytes(b"anchor")
    digest = sha256(checkpoint)
    return {
        "iteration": 26,
        "checkpoint": str(checkpoint),
        "checkpoint_digest": digest,
        "games": 1000,
        "win_rate": wr,
        "confidence_lower": wr - 0.03,
        "confidence_upper": wr + 0.03,
        "audit": {
            "passed": True,
            "valid_games": 1000,
            "exact_distribution": True,
            "exact_weights": True,
            "greedy_required": True,
            "checkpoint_digest": digest,
        },
    }


def _commit(run: Path, iteration: int, wr: float, *, exact: bool = True) -> None:
    checkpoint = run / "checkpoints" / f"iter_{iteration:05d}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(f"candidate-{iteration}".encode())
    digest = sha256(checkpoint)
    audit = {
        "passed": exact,
        "valid_games": 1000 if exact else 999,
        "exact_distribution": exact,
        "exact_weights": exact,
        "greedy_required": True,
        "checkpoint_digest": digest,
    }
    payload = {
        "iteration": iteration,
        "heldout_candidate": {"path": str(checkpoint), "digest": digest},
        "raw_heldout_gate": {
            "games": 1000,
            "win_rate": wr,
            "confidence_lower": wr - 0.03,
            "confidence_upper": wr + 0.03,
            "per_opponent": {},
        },
        "heldout_audit": audit,
    }
    (run / "eval").mkdir(parents=True, exist_ok=True)
    (run / "commits").mkdir(parents=True, exist_ok=True)
    (run / "eval" / f"iter_{iteration:05d}.json").write_text(json.dumps(payload))
    (run / "commits" / f"iter_{iteration:05d}.json").write_text(
        json.dumps({"last_completed_iteration": iteration})
    )


def test_threshold_uses_only_committed_exact_1000_game_heldout(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _commit(run, 0, 0.45, exact=False)
    decision = transition_decision(
        run,
        anchor=_anchor(tmp_path),
        start_iteration=0,
        verify_best_bytes=True,
    )
    assert not decision["triggered"]
    assert decision["exact_iterations_observed"] == 0

    _commit(run, 1, 0.40, exact=True)
    decision = transition_decision(
        run,
        anchor=_anchor(tmp_path),
        start_iteration=0,
        verify_best_bytes=True,
    )
    assert decision["triggered"]
    assert decision["reason"] == "target_win_rate_reached"
    assert decision["best"]["win_rate"] == 0.40


def test_plateau_is_ten_exact_iterations_since_last_strict_improvement(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    anchor = _anchor(tmp_path)
    _commit(run, 0, 0.31)
    for iteration in range(1, 10):
        _commit(run, iteration, 0.30)
    decision = transition_decision(run, anchor=anchor, start_iteration=0)
    assert not decision["triggered"]
    assert decision["non_improving_streak"] == 9

    _commit(run, 10, 0.309)
    decision = transition_decision(run, anchor=anchor, start_iteration=0)
    assert decision["triggered"]
    assert decision["reason"] == "no_new_best_for_patience"
    assert decision["best"]["win_rate"] == 0.31
    assert decision["best"]["iteration"] == 0


def test_uncommitted_eval_never_advances_patience(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _commit(run, 0, 0.20)
    (run / "commits" / "iter_00000.json").unlink()
    decision = transition_decision(run, anchor=_anchor(tmp_path), start_iteration=0)
    assert decision["exact_iterations_observed"] == 0
    assert decision["non_improving_streak"] == 0
