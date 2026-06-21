from poke_agent.rewards import (
    VALUE_NOT_WIN,
    VALUE_WIN,
    assign_episode_values,
    is_timeout_observation,
    player_outcome_value,
)


def test_win_value():
    assert player_outcome_value(0, 0) == VALUE_WIN
    assert player_outcome_value(1, 1) == VALUE_WIN


def test_loss_draw_timeout_values():
    assert player_outcome_value(1, 0) == VALUE_NOT_WIN
    assert player_outcome_value(0, 1) == VALUE_NOT_WIN
    assert player_outcome_value(2, 0) == VALUE_NOT_WIN
    assert player_outcome_value(2, 1) == VALUE_NOT_WIN
    assert player_outcome_value(-1, 0, terminal_obs={"remainingOverageTime": 0}) == VALUE_NOT_WIN


def test_is_timeout_observation():
    assert is_timeout_observation({"remainingOverageTime": 0})
    assert is_timeout_observation({"current": {"result": 2}})
    assert not is_timeout_observation({"remainingOverageTime": 30, "current": {"result": -1}})


def test_assign_episode_values():
    rows = [
        {"player": 0, "terminal": False},
        {"player": 1, "terminal": False},
        {"player": 0, "terminal": True},
    ]
    assign_episode_values(rows, 0)
    assert rows[0]["value"] == VALUE_WIN
    assert rows[1]["value"] == VALUE_NOT_WIN
    assert rows[2]["value"] == VALUE_WIN
    assert rows[2]["reward"] == VALUE_WIN
