from __future__ import annotations

from scripts.generate_router_calibration_episodes import (
    DEFAULT_TARGETS,
    _public_observation,
    _setup_steps,
)
from poke_bot.replay_import import extract_setup_decks


def test_sparse_router_targets_are_exactly_the_v34_rejections() -> None:
    assert set(DEFAULT_TARGETS) == {
        "dragapult-blaziken",
        "dragapult-dusknoir",
        "dudunsparce",
        "gardevoir",
        "ns-zoroark",
    }


def test_router_calibration_observation_drops_all_private_fields() -> None:
    observation = {
        "current": {
            "yourIndex": 0,
            "players": [
                {
                    "active": [{"id": 1}],
                    "bench": [{"id": 2}],
                    "discard": [{"id": 3}],
                    "hand": [{"id": 4}],
                    "deck": [{"id": 5}],
                    "prizes": [{"id": 6}],
                },
                {
                    "active": [{"id": 7}],
                    "bench": [],
                    "discard": [],
                    "hand": [{"id": 8}],
                    "deck": [{"id": 9}],
                    "prizes": [{"id": 10}],
                },
            ],
        },
        "agent_identity": "forbidden",
        "recorded_archetype": "forbidden",
    }
    public = _public_observation(observation)
    encoded = str(public)
    assert "hand" not in encoded
    assert "deck" not in encoded
    assert "prizes" not in encoded
    assert "agent_identity" not in encoded
    assert "recorded_archetype" not in encoded
    assert public["current"]["players"][1]["active"] == [{"id": 7}]


def test_router_calibration_setup_is_classifiable_as_an_official_replay() -> None:
    decks = (list(range(1, 61)), list(range(101, 161)))
    payload = {"steps": _setup_steps(decks)}
    assert extract_setup_decks(payload) == [decks[0], decks[1]]
