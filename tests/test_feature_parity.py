import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission"


def _load_submission_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(f"submission_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"submission_{name}"] = module
    spec.loader.exec_module(module)
    return module


def test_train_and_submission_feature_parity():
    from poke_agent.features import encode_observation_step, feature_spec, total_feature_dim
    from poke_agent.game_tracker import GameEventTracker

    submission_features = _load_submission_module("features", SUBMISSION / "features.py")
    submission_tracker = _load_submission_module("game_tracker", SUBMISSION / "game_tracker.py")

    obs = {
        "current": {
            "turn": 3,
            "turnActionCount": 2,
            "yourIndex": 0,
            "firstPlayer": 0,
            "result": -1,
            "supporterPlayed": False,
            "stadiumPlayed": False,
            "energyAttached": True,
            "retreated": False,
            "stadium": [{"id": 501, "serial": 90, "playerIndex": 0}],
            "players": [
                {
                    "deckCount": 45,
                    "handCount": 2,
                    "benchMax": 8,
                    "bench": [
                        {
                            "id": 301,
                            "serial": 10,
                            "hp": 90,
                            "maxHp": 120,
                            "appearThisTurn": False,
                            "energies": [1, 1],
                            "tools": [],
                        }
                    ],
                    "hand": [
                        {"id": 111, "serial": 1, "playerIndex": 0},
                        {"id": 222, "serial": 2, "playerIndex": 0},
                    ],
                    "discard": [{"id": 9, "serial": 3, "playerIndex": 0}],
                    "prize": [None] * 5,
                    "active": [
                        {
                            "id": 201,
                            "serial": 5,
                            "hp": 110,
                            "maxHp": 130,
                            "appearThisTurn": False,
                            "energies": [2],
                            "tools": [{"id": 401, "serial": 6, "playerIndex": 0}],
                        }
                    ],
                    "poisoned": False,
                    "burned": False,
                    "asleep": False,
                    "paralyzed": False,
                    "confused": False,
                },
                {
                    "deckCount": 42,
                    "handCount": 3,
                    "benchMax": 8,
                    "bench": [],
                    "prize": [None] * 6,
                    "active": [
                        {
                            "id": 302,
                            "serial": 20,
                            "hp": 70,
                            "maxHp": 100,
                            "appearThisTurn": True,
                            "energies": [],
                            "tools": [],
                        }
                    ],
                },
            ],
        },
        "select": {
            "type": 1,
            "minCount": 1,
            "maxCount": 1,
            "remainDamageCounter": 0,
            "remainEnergyCost": 0,
            "option": [{"type": 0, "index": 0}],
            "effect": {"id": 601, "serial": 7, "playerIndex": 0},
        },
        "logs": [
            {"type": 2, "playerIndex": 0},
            {"type": 4, "playerIndex": 0, "cardId": 111, "serial": 1},
        ],
    }
    deck = [111] * 20 + [222] * 20 + [301] * 20
    state_hash_dim = 32
    card_vocab_size = 2000

    train_tracker = GameEventTracker()
    sub_tracker = submission_tracker.GameEventTracker()
    train_vec, train_cards = encode_observation_step(
        obs,
        train_tracker,
        state_hash_dim=state_hash_dim,
        our_deck=deck,
        card_vocab_size=card_vocab_size,
    )
    sub_vec, sub_cards = submission_features.encode_observation_step(
        obs,
        sub_tracker,
        state_hash_dim=state_hash_dim,
        our_deck=deck,
        card_vocab_size=card_vocab_size,
    )

    assert train_vec.shape == (total_feature_dim(state_hash_dim=state_hash_dim),)
    assert sub_vec.shape == train_vec.shape
    assert np.allclose(train_vec, sub_vec, rtol=0.0, atol=0.0)
    assert np.array_equal(train_cards, sub_cards)

    spec = feature_spec(state_hash_dim=state_hash_dim)
    assert spec.schema_version == submission_features.FEATURE_SCHEMA_VERSION
    assert spec.total_dim == train_vec.shape[0]
