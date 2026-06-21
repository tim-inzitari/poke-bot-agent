from poke_agent.deck_pool import parse_deck_placement
from poke_agent.features import COARSE_BASE_DIM, going_first_feature


def test_going_first_feature():
    assert going_first_feature({"current": {"firstPlayer": 0, "yourIndex": 0}}) == 1.0
    assert going_first_feature({"current": {"firstPlayer": 0, "yourIndex": 1}}) == 0.0
    assert going_first_feature({"current": {"firstPlayer": -1, "yourIndex": 0}}) == 0.5


def test_going_first_is_base_feature():
    assert COARSE_BASE_DIM == 11
    assert "going_first" in __import__("poke_agent.features", fromlist=["BASE_FEATURE_NAMES"]).BASE_FEATURE_NAMES


def test_parse_deck_placement():
    assert parse_deck_placement("2026-05_regional-la-2026_282nd_dragapult") == 282
    assert parse_deck_placement("2026-05_regional-melbourne-2026_10th_mega-lucario") == 10
    assert parse_deck_placement("invalid_name") is None
