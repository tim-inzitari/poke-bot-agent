from __future__ import annotations

import copy

import pytest

from poke_bot.alakazam_public_rule_adapter_r298 import build_public_rule_representation
from poke_bot.alakazam_rule_derivative_model_r298 import semantic_option_feature_rows
from poke_bot.prize_plan_live_features import (
    PrizePlanLiveFeatureError,
    complete_action_features_from_public_replay,
    h3_causal_interval_availability,
)


def _player(*, opponent: bool = False) -> dict:
    return {
        "active": [],
        "bench": [],
        "hand": ([{"id": 999, "serial": 999}] if opponent else []),
        "handCount": 1 if opponent else 0,
        "discard": [],
        "prize": ([{"id": 998, "serial": 998}] + [None] * 5 if opponent else [None] * 6),
        "deck": ([{"id": 997, "serial": 997}] if opponent else []),
        "deckCount": 20,
        "benchMax": 5,
    }


def _observation() -> dict:
    return {
        "current": {
            "yourIndex": 0,
            "turn": 3,
            "turnActionCount": 1,
            "supporterPlayed": False,
            "stadiumPlayed": False,
            "energyAttached": False,
            "retreated": False,
            "players": [_player(), _player(opponent=True)],
            "stadium": [],
            "looking": [],
        },
        "select": {
            "context": "Main",
            "type": "Card",
            "minCount": 1,
            "maxCount": 2,
            "remainDamageCounter": 0,
            "remainEnergyCost": 0,
            "option": [
                {"type": "Skill", "cardId": 77, "serial": 1001},
                {"type": "Number", "number": 4},
                {"type": "Number", "number": 5},
            ],
        },
    }


def _build(observation: dict, action: list[int]):
    return complete_action_features_from_public_replay(
        episode_id="episode-1",
        seat=0,
        env_step=11,
        observation=observation,
        action=action,
    )


def test_live_features_match_public_semantic_rows_and_factorized_program() -> None:
    observation = _observation()
    expected = tuple(
        tuple(row)
        for row in semantic_option_feature_rows(
            build_public_rule_representation(observation).to_dict()
        )
    )
    row = _build(observation, [0, 1])
    assert row.stage_count == 2
    assert row.first_stage_menu == expected
    assert row.selected_stage_features == (expected[0], expected[1])
    assert row.selected_option_indices == (0, 0)
    assert row.selected_legal_counts == (3, 3)
    assert row.selected_action_programs == ((0,), (0, 1))


def test_live_features_encode_stop_without_reusing_last_selected_option() -> None:
    row = _build(_observation(), [0])
    assert row.selected_action_programs == ((0,), (0,))
    assert row.selected_option_indices == (0, 2)
    # Option-kind columns are attack, skill, number, card, pass, ...
    assert row.selected_stage_features[1][4] == 1.0
    assert sum(row.selected_stage_features[1][:8]) == 1.0


def test_live_features_are_invariant_to_hidden_opponent_payloads() -> None:
    first = _observation()
    second = copy.deepcopy(first)
    second["current"]["players"][1]["hand"] = [
        {"id": 1201, "serial": 12001},
        {"id": 1202, "serial": 12002},
    ]
    second["current"]["players"][1]["deck"] = [{"id": 1301, "serial": 13001}]
    second["current"]["players"][1]["prize"] = [{"id": 1401, "serial": 14001}]
    assert _build(first, [1, 2]) == _build(second, [1, 2])


def test_live_features_fail_closed_on_nonlegal_complete_action() -> None:
    with pytest.raises(PrizePlanLiveFeatureError):
        _build(_observation(), [63])


def test_h3_interval_requires_four_exact_monotone_public_frames() -> None:
    decisions = []
    for index, (own, opponent) in enumerate(((6, 6), (6, 5), (5, 5), (4, 5))):
        observation = _observation()
        observation["current"]["players"][0]["prize"] = [None] * own
        observation["current"]["players"][1]["prize"] = [None] * opponent
        decisions.append({"env_step": 10 + index * 3, "observation": observation})
    assert h3_causal_interval_availability(
        decisions, acting_seat=0, start_index=0
    ) == (True, None)
    assert h3_causal_interval_availability(
        decisions, acting_seat=0, start_index=1
    )[0] is False


def test_h3_interval_masks_setup_and_nonmonotone_counts() -> None:
    decisions = []
    for index, own in enumerate((6, 5, 6, 4)):
        observation = _observation()
        observation["current"]["players"][0]["prize"] = [None] * own
        observation["current"]["players"][1]["prize"] = [None] * 6
        decisions.append({"env_step": index + 1, "observation": observation})
    assert h3_causal_interval_availability(
        decisions, acting_seat=0, start_index=0
    ) == (False, "non_monotone_public_prize_count")
    decisions[0]["observation"]["current"]["players"][0]["prize"] = []
    assert h3_causal_interval_availability(
        decisions, acting_seat=0, start_index=0
    ) == (
        False,
        "pre_action_public_prize_count_outside_valid_1_to_6_range",
    )
