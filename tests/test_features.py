from poke_agent.deck_pool import parse_deck_placement
from poke_agent.features import (
    GLOBAL_FEATURE_DIM,
    STRUCTURED_FEATURE_DIM,
    encode_observation_step,
    feature_spec,
    going_first_feature,
    hashed_deck_composition,
    seat_deck_from_row,
    total_feature_dim,
)
from poke_agent.game_tracker import GameEventTracker


def test_going_first_feature():
    assert going_first_feature({"current": {"firstPlayer": 0, "yourIndex": 0}}) == 1.0
    assert going_first_feature({"current": {"firstPlayer": 0, "yourIndex": 1}}) == 0.0
    assert going_first_feature({"current": {"firstPlayer": -1, "yourIndex": 0}}) == 0.5


def test_feature_spec_matches_structured_layout():
    spec = feature_spec(state_hash_dim=32)
    assert spec.structured_dim == STRUCTURED_FEATURE_DIM
    assert spec.total_dim == total_feature_dim(state_hash_dim=32)
    assert GLOBAL_FEATURE_DIM == 18


def test_parse_deck_placement():
    assert parse_deck_placement("2026-05_regional-la-2026_282nd_dragapult") == 282
    assert parse_deck_placement("2026-05_regional-melbourne-2026_10th_mega-lucario") == 10
    assert parse_deck_placement("invalid_name") is None


def test_visible_card_ids_change_features():
    import numpy as np

    obs = {
        "current": {
            "turn": 1,
            "yourIndex": 0,
            "firstPlayer": 0,
            "result": -1,
            "players": [
                {
                    "deckCount": 40,
                    "handCount": 1,
                    "bench": [],
                    "hand": [{"id": 111, "serial": 1, "playerIndex": 0}],
                },
                {"deckCount": 40, "handCount": 0, "bench": []},
            ],
        },
        "select": {"option": [0], "minCount": 1, "maxCount": 1},
        "logs": [],
    }
    obs_other = {
        **obs,
        "current": {
            **obs["current"],
            "players": [
                {
                    **obs["current"]["players"][0],
                    "hand": [{"id": 222, "serial": 1, "playerIndex": 0}],
                },
                obs["current"]["players"][1],
            ],
        },
    }
    tracker_a = GameEventTracker()
    tracker_b = GameEventTracker()
    vec_a, cards_a = encode_observation_step(obs, tracker_a, state_hash_dim=32)
    vec_b, cards_b = encode_observation_step(obs_other, tracker_b, state_hash_dim=32)
    assert not np.allclose(vec_a, vec_b)
    assert not np.array_equal(cards_a, cards_b)


def test_full_deck_composition_changes_features():
    import numpy as np

    obs = {
        "current": {"turn": 0, "yourIndex": 0, "firstPlayer": 0, "result": -1, "players": [{}, {}]},
        "select": {"option": [], "minCount": 0, "maxCount": 0},
        "logs": [],
    }
    deck_a = [1] * 60
    deck_b = [2] * 60
    tracker = GameEventTracker()
    vec_a, _ = encode_observation_step(obs, tracker, state_hash_dim=32, our_deck=deck_a)
    vec_b, _ = encode_observation_step(obs, tracker, state_hash_dim=32, our_deck=deck_b)
    assert not np.allclose(vec_a, vec_b)


def test_seat_deck_from_row_uses_your_index():
    row = {
        "player": 1,
        "observation": {"current": {"yourIndex": 1}},
        "deck0_cards": [1] * 60,
        "deck1_cards": [2] * 60,
    }
    assert seat_deck_from_row(row) == [2] * 60


def test_opponent_deck_list_not_in_features():
    import numpy as np

    obs = {
        "current": {"turn": 1, "yourIndex": 0, "firstPlayer": 0, "result": -1, "players": [{}, {}]},
        "select": {"option": [0], "minCount": 1, "maxCount": 1},
        "logs": [],
    }
    row_self = {
        "observation": obs,
        "player": 0,
        "deck0_cards": [10] * 60,
        "deck1_cards": [99] * 60,
    }
    row_opp = {
        "observation": {**obs, "current": {**obs["current"], "yourIndex": 1}},
        "player": 1,
        "deck0_cards": [10] * 60,
        "deck1_cards": [99] * 60,
    }
    tracker_a = GameEventTracker()
    tracker_b = GameEventTracker()
    vec_self, _ = encode_observation_step(obs, tracker_a, state_hash_dim=32, our_deck=seat_deck_from_row(row_self))
    vec_opp, _ = encode_observation_step(
        row_opp["observation"],
        tracker_b,
        state_hash_dim=32,
        our_deck=seat_deck_from_row(row_opp),
    )
    assert not np.allclose(vec_self, vec_opp)
    assert seat_deck_from_row(row_self) == [10] * 60
    assert seat_deck_from_row(row_opp) == [99] * 60


def test_hashed_deck_composition_is_nonzero():
    import numpy as np

    vec = hashed_deck_composition([1, 2, 3], state_hash_dim=32)
    assert vec.shape == (32,)
    assert np.any(vec != 0)
