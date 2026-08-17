from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import agent as agent_module
from poke_bot.pure_rl.no_progress import (
    NoProgressTracker,
    STALL_TRAINING_RETURN,
    configured_max_stagnant_turns,
    win_progress_signature,
)


def _obs(turn: int, *, prize_count: int = 6, result: int = -1) -> dict:
    return {
        "current": {
            "turn": turn,
            "yourIndex": turn % 2,
            "result": result,
            "players": [
                {
                    "prize": [None] * prize_count,
                    "active": [
                        {
                            "id": 743,
                            "hp": 140,
                            "maxHp": 140,
                            "energies": [],
                            "energyCards": [],
                        }
                    ],
                    "bench": [],
                    # These churn forever in the observed failure and must not
                    # reset the factual win-progress clock.
                    "deckCount": turn % 3,
                    "hand": [{"id": turn}],
                    "discard": [{"id": 1000 + turn}],
                },
                {
                    "prize": [None] * 6,
                    "active": [
                        {
                            "id": 741,
                            "hp": 70,
                            "maxHp": 70,
                            "energies": [],
                            "energyCards": [],
                        }
                    ],
                    "bench": [],
                },
            ],
        },
        "select": {
            "minCount": 1,
            "maxCount": 1,
            "option": [{"type": 14}],
        },
    }


def test_zone_recycling_does_not_reset_stall_clock() -> None:
    tracker = NoProgressTracker(64)
    assert tracker.observe(_obs(0)) is False
    for turn in range(1, 64):
        assert tracker.observe(_obs(turn)) is False
    assert tracker.observe(_obs(64)) is True
    assert tracker.stagnant_turns == 64


def test_prize_progress_resets_stall_clock() -> None:
    tracker = NoProgressTracker(4)
    assert tracker.observe(_obs(0)) is False
    assert tracker.observe(_obs(3)) is False
    assert tracker.observe(_obs(4, prize_count=5)) is False
    assert tracker.observe(_obs(7, prize_count=5)) is False
    assert tracker.observe(_obs(8, prize_count=5)) is True


def test_signature_ignores_hand_deck_discard_churn() -> None:
    assert win_progress_signature(_obs(1)) == win_progress_signature(_obs(99))


def test_stall_limit_is_disabled_until_boundary_gate(monkeypatch) -> None:
    monkeypatch.delenv("PURE_RL_NO_PROGRESS_MAX_TURNS", raising=False)
    assert configured_max_stagnant_turns({}) is None
    monkeypatch.setenv("PURE_RL_NO_PROGRESS_MAX_TURNS", "64")
    assert configured_max_stagnant_turns({}) == 64
    assert configured_max_stagnant_turns({"max_no_progress_turns": 32}) == 32


def test_play_game_marks_stall_distinct_from_engine_draw(monkeypatch) -> None:
    state = {"turn": 0}

    monkeypatch.setattr(agent_module.cg_env, "battle_start", lambda *_: (_obs(0), None))
    monkeypatch.setattr(
        agent_module.cg_env,
        "battle_select",
        lambda _choice: _obs(state.__setitem__("turn", state["turn"] + 1) or state["turn"]),
    )
    monkeypatch.setattr(
        agent_module.cg_env,
        "is_finished",
        lambda observation: int((observation.get("current") or {}).get("result", -1)) != -1,
    )
    monkeypatch.setattr(agent_module.cg_env, "battle_finish", lambda: None)
    monkeypatch.setattr(agent_module.cg_env, "result_winner", lambda _obs: None)

    result = agent_module.play_game(
        lambda _observation: [0],
        lambda _observation: [0],
        [1] * 60,
        [1] * 60,
        max_steps=4000,
        max_no_progress_turns=64,
    )
    assert result["winner"] == 2
    assert result["incomplete"] is True
    assert result["termination"] == "no_progress_stall"
    assert result["stall_terminated"] is True
    assert result["stall_turns"] == 64
    assert result["steps"] == 64
    assert STALL_TRAINING_RETURN == -1.0
