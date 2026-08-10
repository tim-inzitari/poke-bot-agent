from __future__ import annotations

import copy

import pytest

from replay_inspector.timeline import (
    ReplayTimelineError,
    causal_observation,
    decision_timeline,
)


def _observation(*, turn: int, actor: object, option_count: int = 2) -> dict:
    return {
        "current": {
            "turn": turn,
            "yourIndex": actor,
            "players": [{"hand": []}, {"hand": None}],
        },
        "select": {
            "context": "Hand",
            "option": [{} for _ in range(option_count)],
            "minCount": 1,
            "maxCount": 1,
        },
    }


def _replay(*, seat: int = 0) -> dict:
    obs = _observation(turn=3, actor=seat)
    steps = [
        [
            (
                {"status": "ACTIVE", "observation": copy.deepcopy(obs)}
                if seat == 0
                else {"status": "INACTIVE", "observation": {}}
            ),
            (
                {"status": "ACTIVE", "observation": copy.deepcopy(obs)}
                if seat == 1
                else {"status": "INACTIVE", "observation": {}}
            ),
        ],
        [
            (
                {
                    "status": "ACTIVE",
                    "observation": copy.deepcopy(obs),
                    "action": [1],
                }
                if seat == 0
                else {"status": "INACTIVE", "observation": {}, "action": []}
            ),
            (
                {
                    "status": "ACTIVE",
                    "observation": copy.deepcopy(obs),
                    "action": [1],
                }
                if seat == 1
                else {"status": "INACTIVE", "observation": {}, "action": []}
            ),
        ],
    ]
    return {"steps": steps}


@pytest.mark.parametrize("seat", [0, 1])
def test_timeline_uses_next_kaggle_step_action_and_factorized_address(
    seat: int,
) -> None:
    replay = _replay(seat=seat)
    rows = decision_timeline(replay, seat)
    assert len(rows) == 1
    assert rows[0].step_index == 0
    assert rows[0].factorized_stage == 0
    assert rows[0].recorded_action == (1,)
    assert rows[0].candidates == ((0,), (1,))
    assert rows[0].target_index == 1
    assert rows[0].to_dict(include_candidates=True)["candidates"] == [[0], [1]]


def test_causal_observation_never_reads_visualize_current() -> None:
    replay = _replay()
    replay["steps"][0][0]["visualize"] = [
        {"current": {"players": [{}, {"hand": [9001]}]}}
    ]
    observation = causal_observation(replay, seat=0, step_index=0)
    assert observation["current"]["players"][1]["hand"] is None
    assert "9001" not in str(observation)


def test_illegal_recorded_action_fails_closed() -> None:
    replay = _replay()
    replay["steps"][1][0]["action"] = [99]
    with pytest.raises(ReplayTimelineError, match="illegal recorded action"):
        decision_timeline(replay, 0)


@pytest.mark.parametrize(
    ("seat", "actor", "missing"),
    [
        (0, None, True),
        (0, 1, False),
        (1, 0, False),
        (0, True, False),
        (1, "1", False),
    ],
)
def test_active_select_requires_exact_matching_json_integer_actor(
    seat: int, actor: object, missing: bool
) -> None:
    replay = _replay(seat=seat)
    current = replay["steps"][0][seat]["observation"]["current"]
    if missing:
        del current["yourIndex"]
    else:
        current["yourIndex"] = actor
    with pytest.raises(ReplayTimelineError, match=r"current\.yourIndex"):
        decision_timeline(replay, seat)


def test_inactive_row_with_stale_select_never_becomes_a_decision() -> None:
    replay = _replay(seat=0)
    stale = _observation(turn=3, actor=1)
    replay["steps"][0][1] = {"status": "INACTIVE", "observation": stale}
    replay["steps"][1][1] = {
        "status": "INACTIVE",
        "observation": {},
        "action": [1],
    }

    assert [stage.step_index for stage in decision_timeline(replay, 0)] == [0]
    assert decision_timeline(replay, 1) == []


def test_active_row_with_non_mapping_select_fails_closed() -> None:
    replay = _replay(seat=0)
    replay["steps"][0][0]["observation"]["select"] = [0, 1]
    with pytest.raises(ReplayTimelineError, match="malformed select"):
        decision_timeline(replay, 0)
