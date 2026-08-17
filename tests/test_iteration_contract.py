import random

import pytest

from poke_bot.dataset import DecisionSample, GameSequence
from poke_bot.iteration_contract import (
    IterationSampleAccounting,
    SearchCollectionContract,
    balanced_seat_coverage,
    cap_game_decisions,
    derive_target_search_decisions,
    game_first_sample,
    history_games_for_fraction,
)


def test_balanced_opponent_seat_strata_at_even_minimum() -> None:
    opponents = ["a", "b", "c"]
    rows = [
        (opponent, game_index % 2)
        for game_index in range(12)
        for opponent in opponents
    ]
    complete, counts = balanced_seat_coverage(
        rows, opponents=opponents, min_games_per_opponent=12
    )
    assert complete
    assert all(seats == {0: 6, 1: 6} for seats in counts.values())


def test_decision_target_stops_only_after_coverage() -> None:
    contract = SearchCollectionContract(
        opponents=26,
        target_search_decisions=1800,
    )
    assert (
        contract.stop_reason(
            games_per_opponent=10,
            trusted_decisions=2000,
            coverage_complete=False,
        )
        is None
    )
    assert (
        contract.stop_reason(
            games_per_opponent=12,
            trusted_decisions=1800,
            coverage_complete=True,
        )
        == "target_search_decisions"
    )


def test_min_nominal_max_safeguards_and_max_stop() -> None:
    with pytest.raises(ValueError, match="even"):
        SearchCollectionContract(opponents=26, min_games_per_opponent=11)
    contract = SearchCollectionContract(
        opponents=26, target_search_decisions=999_999
    )
    assert contract.min_games == 312
    assert contract.nominal_games == 416
    assert contract.max_games == 624
    assert (
        contract.stop_reason(
            games_per_opponent=24,
            trusted_decisions=1,
            coverage_complete=True,
        )
        == "max_games"
    )


def test_pilot_target_is_measured_and_replay_fraction_is_bounded() -> None:
    report = derive_target_search_decisions(
        [1, 1, 1, 1, 1, 1, 1, 9],
        opponents=26,
        nominal_games_per_opponent=16,
    )
    assert report["pilot_games"] == 8
    assert report["target_search_decisions"] == 2 * 416
    assert history_games_for_fraction(100, 0.5) == 100
    assert history_games_for_fraction(100, 0.75) == 300
    with pytest.raises(ValueError, match="\\[0.50, 0.75\\]"):
        history_games_for_fraction(100, 0.49)


def _sequence(name: str, decisions: int) -> GameSequence:
    samples = [
        DecisionSample(
            board={},
            options={},
            action=[0],
            action_combo_index=0,
            action_combos=[[0]],
            env_step=index,
        )
        for index in range(decisions)
    ]
    return GameSequence(
        episode_id=name,
        seat=0,
        archetype="x",
        opp_archetype="y",
        deck=[1] * 60,
        value=0.0,
        decisions=samples,
    )


def test_game_first_sampling_and_decision_windows_bound_long_games() -> None:
    games = [_sequence("short", 4), _sequence("long", 100)]
    selected = game_first_sample(games, count=2, rng=random.Random(1))
    capped = [
        cap_game_decisions(game, max_decisions=16, rng=random.Random(2))
        for game in selected
    ]
    assert {game.episode_id for game in capped} == {"short", "long"}
    assert max(len(game.decisions) for game in capped) == 16
    assert next(
        game for game in capped if game.episode_id == "long"
    ).target_provenance["game_first_decision_window"]["original_decisions"] == 100


def test_promotion_samples_are_accounted_separately() -> None:
    accounting = IterationSampleAccounting(
        fresh_games=416,
        trusted_search_decisions=2400,
        promotion_games=40,
    ).as_dict()
    assert accounting["collection"]["fresh_games"] == 416
    assert accounting["promotion"] == {
        "games": 40,
        "included_in_training": 0,
    }
