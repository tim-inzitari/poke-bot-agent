from poke_agent.rewards import (
    DEFAULT_PRIZE_COUNT,
    VALUE_NOT_WIN,
    VALUE_TIMEOUT,
    VALUE_WIN,
    assign_episode_values,
    filter_complete_episode_rows,
    is_complete_episode,
    is_timeout_observation,
    player_outcome_value,
    prize_count,
    prize_progress_value,
    resolve_game_result,
    seat_prize_counts,
    step_prize_reward,
    winner_from_prizes,
)


def _obs(*, p0_prize: int = 6, p1_prize: int = 6, result: int = -1, your_index: int = 0) -> dict:
    return {
        "current": {
            "yourIndex": your_index,
            "result": result,
            "players": [
                {"prize": [None] * p0_prize, "deckCount": 40, "handCount": 5, "bench": []},
                {"prize": [None] * p1_prize, "deckCount": 40, "handCount": 5, "bench": []},
            ],
        }
    }


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


def test_winner_from_prizes_uses_own_prize_zero():
    terminal = _obs(p0_prize=0, p1_prize=6, result=1)
    assert winner_from_prizes(terminal) == 0
    terminal = _obs(p0_prize=6, p1_prize=0, result=0)
    assert winner_from_prizes(terminal) == 1


def test_resolve_game_result_prefers_prizes_over_wrong_result_index():
    terminal = _obs(p0_prize=0, p1_prize=6, result=1)
    assert resolve_game_result(1, terminal) == 0


def test_step_prize_reward_own_prize_taken_on_enemy_ko():
    before = _obs(p0_prize=6, p1_prize=6, your_index=0)
    after = _obs(p0_prize=5, p1_prize=6, your_index=0)
    reward = step_prize_reward(before, after, player=0)
    assert reward > 0
    assert abs(reward - VALUE_WIN / DEFAULT_PRIZE_COUNT) < 1e-6


def test_step_prize_reward_opp_prize_taken_when_we_get_koed():
    before = _obs(p0_prize=6, p1_prize=6, your_index=0)
    after = _obs(p0_prize=6, p1_prize=5, your_index=0)
    reward = step_prize_reward(before, after, player=0)
    assert reward < 0


def test_prize_progress_value_favors_own_prizes_to_zero():
    start = prize_progress_value(_obs(p0_prize=6, p1_prize=6), player=0)
    winning = prize_progress_value(_obs(p0_prize=0, p1_prize=6), player=0)
    losing = prize_progress_value(_obs(p0_prize=6, p1_prize=0), player=0)
    assert winning > start > losing


def test_seat_prize_counts_from_acting_seat():
    obs = _obs(p0_prize=4, p1_prize=6, your_index=1)
    assert seat_prize_counts(obs, seat=1) == (6, 4)
    assert prize_count(obs, 1) == 6


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


def test_assign_episode_values_prize_step_rewards():
    rows = [
        {
            "player": 0,
            "terminal": False,
            "observation": _obs(p0_prize=6, p1_prize=6),
            "next_observation": _obs(p0_prize=5, p1_prize=6),
        },
        {
            "player": 0,
            "terminal": True,
            "observation": _obs(p0_prize=1, p1_prize=6),
            "next_observation": _obs(p0_prize=0, p1_prize=6, result=0),
        },
    ]
    assign_episode_values(rows, 0, terminal_obs=rows[-1]["next_observation"])
    assert rows[0]["reward"] > 0
    assert rows[0]["value"] > rows[1]["value"] or rows[1]["value"] == VALUE_WIN
    assert rows[1]["value"] == VALUE_WIN
    assert rows[1]["self_prize_remaining"] == 1


def test_screenshot_scenario_winner_has_zero_own_prizes():
    """Winner took all their prizes; loser still has full prize count (6)."""
    terminal = _obs(p0_prize=6, p1_prize=0, result=0)
    assert winner_from_prizes(terminal) == 1
    assert player_outcome_value(0, 1, terminal_obs=terminal) == VALUE_WIN
    assert player_outcome_value(0, 0, terminal_obs=terminal) == VALUE_NOT_WIN
    terminal_flip = _obs(p0_prize=0, p1_prize=6, result=1)
    assert winner_from_prizes(terminal_flip) == 0
    assert player_outcome_value(1, 0, terminal_obs=terminal_flip) == VALUE_WIN


def test_inverted_wincon_opponent_prizes_zero_does_not_win_other_player():
    """Wrong rule (opp prizes=0 means you win) would pick the loser — we reject that."""
    terminal = _obs(p0_prize=6, p1_prize=0)
    assert winner_from_prizes(terminal) == 1
    assert winner_from_prizes(terminal) != 0
    terminal = _obs(p0_prize=0, p1_prize=6)
    assert winner_from_prizes(terminal) == 0
    assert winner_from_prizes(terminal) != 1


def test_both_at_zero_prizes_is_ambiguous_no_winner():
    assert winner_from_prizes(_obs(p0_prize=0, p1_prize=0)) is None


def test_neither_at_zero_no_prize_winner():
    assert winner_from_prizes(_obs(p0_prize=3, p1_prize=4)) is None


def test_step_reward_only_own_prize_decrease_is_positive():
    """Taking from opponent's pile (their prizes down, ours unchanged) is a penalty."""
    before = _obs(p0_prize=6, p1_prize=6)
    opp_took_only = _obs(p0_prize=6, p1_prize=5)
    assert step_prize_reward(before, opp_took_only, player=0) < 0
    own_took = _obs(p0_prize=5, p1_prize=6)
    assert step_prize_reward(before, own_took, player=0) > 0


def test_terminal_value_uses_prize_winner_not_misleading_result_index():
    terminal = _obs(p0_prize=0, p1_prize=6, result=1)
    assert player_outcome_value(1, 0, terminal_obs=terminal) == VALUE_WIN
    assert player_outcome_value(1, 1, terminal_obs=terminal) == VALUE_NOT_WIN
