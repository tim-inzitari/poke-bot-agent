from __future__ import annotations

from array import array

import torch

from poke_bot import features
from poke_bot.dataset import DecisionSample, GameSequence, PolicyStage
from poke_bot.feature_shards import (
    _target_coverage,
    compact_temporal_expert_sequence,
)
from poke_bot.replay_import import derive_opponent_hidden_remainder
from poke_bot.train import belief_multihots_from_aux_labels


def _sparse(words: int, offset: int = 0) -> features.SparseVector:
    vector = features.SparseVector()
    for word in range(words):
        vector.word_start()
        vector.add(offset + word, 1.0)
    return vector


def test_temporal_expert_compaction_preserves_history_and_aux_targets() -> None:
    stage = PolicyStage(
        options=_sparse(2, 20),
        action_combos=[[0], [1]],
        target_index=1,
        guide_target_index=0,
        guide_confidence=0.75,
    )
    decision = DecisionSample(
        board=_sparse(features.NUM_BOARD_TOKENS),
        options=stage.options,
        action=[1],
        action_combo_index=1,
        action_combos=[[0], [1]],
        env_step=4,
        aux_labels={
            "opp_hidden_remainder": [7, 11],
            "opp_hidden_remainder_source": (
                "official_full_deck_minus_public_evidence"
            ),
            "lethal_threat": 1.0,
            "prize_race": [0.5, 1.0],
        },
        action_token=_sparse(1, 40),
        policy_stages=[stage],
    )
    sequence = GameSequence(
        episode_id="expert",
        seat=0,
        archetype="alakazam",
        opp_archetype="crustle",
        deck=[1] * 60,
        value=1.0,
        decisions=[decision],
        source="pokemon-tcg-ai-battle-episodes-2026-07-20",
    )

    compact_temporal_expert_sequence(sequence)

    assert isinstance(sequence.deck, array)
    assert sequence.source == ""
    assert decision.action == []
    assert decision.action_combos == []
    assert decision.action_token is not None
    assert isinstance(decision.action_token.index, array)
    assert decision.aux_labels["opp_hidden_remainder"] == [7, 11]
    assert stage.action_combos == []
    assert stage.guide_target_index == 0
    assert stage.guide_confidence == 0.75
    assert _target_coverage(sequence) == {
        "temporal_action_rows": 1,
        "opponent_hand_rows": 0,
        "opponent_remainder_rows": 1,
        "opponent_private_prize_rows": 0,
        "lethal_threat_rows": 1,
        "prize_race_rows": 1,
        "guide_rows": 1,
        "combo_state_rows": 0,
    }


def test_official_deck_minus_public_evidence_is_target_only_and_exact() -> None:
    deck = [1, 1, 2, 3, 4] + [9] * 55
    observation = {
        "current": {
            "yourIndex": 0,
            "players": [
                {"hand": [{"id": 8}], "active": [], "bench": [], "discard": []},
                {
                    "hand": None,
                    "active": [
                        {
                            "id": 1,
                            "serial": 10,
                            "playerIndex": 1,
                            "energyCards": [
                                {"id": 2, "serial": 11, "playerIndex": 1}
                            ],
                        }
                    ],
                    "bench": [],
                    "discard": [{"id": 3, "serial": 12, "playerIndex": 1}],
                    "prize": [{"id": 4, "serial": 13, "playerIndex": 1}, None],
                },
            ],
            "stadium": [],
        }
    }

    target = derive_opponent_hidden_remainder(observation, deck)

    assert target is not None
    assert target.count(1) == 1
    assert 2 not in target and 3 not in target and 4 not in target
    assert len(target) == 56
    assert observation["current"]["players"][1]["hand"] is None

    hand, remainder = belief_multihots_from_aux_labels(
        {"opp_hidden_remainder": target},
        32,
        device=torch.device("cpu"),
    )
    assert hand is None
    assert remainder is not None
    assert float(remainder[1]) == 1.0
    assert float(remainder[9]) == 1.0
    assert float(remainder[2]) == 0.0


def test_hidden_remainder_fails_closed_on_impossible_public_card() -> None:
    observation = {
        "current": {
            "yourIndex": 0,
            "players": [
                {"hand": [], "active": [], "bench": [], "discard": []},
                {
                    "hand": None,
                    "active": [],
                    "bench": [],
                    "discard": [{"id": 99, "serial": 1, "playerIndex": 1}],
                    "prize": [],
                },
            ],
        }
    }
    assert derive_opponent_hidden_remainder(observation, [1] * 60) is None
