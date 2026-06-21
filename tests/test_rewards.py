from poke_agent.rewards import (
    VALUE_NOT_WIN,
    VALUE_TIMEOUT,
    VALUE_WIN,
    assign_episode_values,
    filter_complete_episode_rows,
    is_complete_episode,
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
    assert player_outcome_value(-1, 0, terminal_obs={"remainingOverageTime": 0}) == VALUE_TIMEOUT


def test_timeout_harsh_penalty():
    assert player_outcome_value(2, 0, terminal_obs={"remainingOverageTime": 0}) == VALUE_TIMEOUT
    assert VALUE_TIMEOUT < VALUE_NOT_WIN


def test_filter_complete_episode_rows():
    rows = [
        {"episode": 0, "step": 0, "player": 0, "result": 0, "terminal": False, "truncated": False},
        {"episode": 0, "step": 1, "player": 0, "result": 0, "terminal": True, "truncated": False},
        {"episode": 1, "step": 0, "player": 0, "result": 2, "terminal": True, "truncated": False},
    ]
    kept = filter_complete_episode_rows(rows)
    assert len(kept) == 2
    assert all(int(row["episode"]) == 0 for row in kept)


def test_is_complete_episode():
    assert is_complete_episode(0, terminal_obs={"current": {"result": 0}})
    assert not is_complete_episode(2, terminal_obs={"current": {"result": 2}})
    assert not is_complete_episode(0, truncated=True)
    assert is_timeout_observation({"remainingOverageTime": 0})
    assert is_timeout_observation({"current": {"result": 2}})
    assert not is_timeout_observation({"remainingOverageTime": 30, "current": {"result": -1}})


def test_is_timeout_observation():
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
