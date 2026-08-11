from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from poke_bot import alakazam_heuristics, archetypes, deck_guides
from poke_bot import alakazam_new_list_heuristics as guide


DECK_PATH = Path("decks/archetype-samples/alakazam-new-list-direct-r241.csv")


def _pokemon(card_id: int, hp: int, *, energy: list[int] | None = None) -> dict:
    return {
        "id": card_id,
        "hp": hp,
        "maxHp": hp,
        "energyCards": [{"id": value} for value in (energy or [])],
        "tools": [],
    }


def _player(
    *,
    hand: list[int] | None = None,
    hand_count: int | None = None,
    active: dict | None = None,
    bench: list[dict] | None = None,
    deck_count: int = 30,
) -> dict:
    hand_cards = [{"id": value} for value in (hand or [])]
    return {
        "active": [] if active is None else [active],
        "bench": list(bench or []),
        "deckCount": deck_count,
        "discard": [],
        "prize": [None] * 6,
        "handCount": len(hand_cards) if hand_count is None else hand_count,
        "hand": hand_cards,
    }


def _obs(me: dict, opp: dict, select: dict, *, stadium: list[dict] | None = None) -> dict:
    return {
        "current": {
            "yourIndex": 0,
            "players": [me, opp],
            "stadium": list(stadium or []),
            "looking": [],
        },
        "select": select,
    }


def test_exact_new_deck_identity_and_additive_archetype_classification() -> None:
    cards = [int(value) for value in DECK_PATH.read_text().splitlines() if value]
    counts = Counter(cards)

    assert len(cards) == 60
    assert tuple(cards) == guide.EXACT_DECK
    assert counts == Counter(
        {
            741: 4, 742: 4, 743: 3, 305: 3, 66: 2, 140: 1,
            1264: 4, 1086: 4, 1231: 4, 1081: 4, 1225: 4,
            1152: 4, 1079: 3, 1097: 2, 1182: 3, 1197: 2,
            1184: 1, 1129: 1, 19: 4, 5: 2, 13: 1,
        }
    )
    assert guide.is_alakazam_new_list_deck(cards)
    assert archetypes.is_alakazam_new_list_direct_r241_representative(cards)
    assert archetypes.classify_deck(cards) == "alakazam"

    legacy = list(archetypes.ALAKAZAM_FINAL_REFRESH_REPRESENTATIVE)
    assert archetypes.classify_deck(legacy) == "alakazam"
    assert not guide.is_alakazam_new_list_deck(legacy)

    mutation = list(cards)
    mutation[-1] = 1
    assert not guide.is_alakazam_new_list_deck(mutation)
    assert archetypes.classify_deck(mutation) != "alakazam"


def test_alakazam_guide_version_dispatch_is_fail_closed(monkeypatch) -> None:
    monkeypatch.setenv("POKEBOT_CURRENT_DECK_GUIDE", "alakazam")
    monkeypatch.setenv("POKEBOT_CURRENT_DECK_GUIDE_TARGETS", "1")

    monkeypatch.delenv("POKEBOT_CURRENT_DECK_GUIDE_VERSION", raising=False)
    assert deck_guides.guide_version() == alakazam_heuristics.GUIDE_VERSION

    monkeypatch.setenv("POKEBOT_CURRENT_DECK_GUIDE_VERSION", guide.GUIDE_VERSION)
    assert deck_guides.guide_version() == guide.GUIDE_VERSION
    assert deck_guides.enabled()

    monkeypatch.setenv("POKEBOT_CURRENT_DECK_GUIDE_VERSION", "unknown-version")
    with pytest.raises(RuntimeError, match="unknown Alakazam guide version"):
        deck_guides.guide_version()


def test_battle_cage_requires_visible_value_and_wrong_deck_masks(monkeypatch) -> None:
    monkeypatch.setenv("POKEBOT_ALAKAZAM_NEW_LIST_GUIDE_TARGETS", "1")
    me = _player(
        hand=[guide.BATTLE_CAGE],
        active=_pokemon(guide.ABRA, 60),
        bench=[_pokemon(guide.ABRA, 60)],
    )
    opp = _player(active=_pokemon(guide.FROSLASS, 90))
    obs = _obs(
        me,
        opp,
        {
            "context": 0,
            "option": [{"type": 7, "index": 0}, {"type": 14}],
            "minCount": 1,
            "maxCount": 1,
        },
    )
    scores = guide.guide_scores(obs, [[0], [1]], deck=guide.EXACT_DECK)
    assert scores is not None
    assert scores[0] > scores[1]

    assert guide.guide_scores(
        obs,
        [[0], [1]],
        deck=archetypes.ALAKAZAM_FINAL_REFRESH_REPRESENTATIVE,
    ) is None


def test_powerful_hand_visible_protection_and_complete_stage_masking(monkeypatch) -> None:
    monkeypatch.setenv("POKEBOT_ALAKAZAM_NEW_LIST_GUIDE_TARGETS", "1")
    me = _player(
        hand_count=14,
        active=_pokemon(guide.ALAKAZAM, 140, energy=[guide.PSYCHIC_ENERGY]),
    )
    opp = _player(
        active=_pokemon(900, 280, energy=[guide.ROCK_FIGHTING_ENERGY])
    )
    obs = _obs(
        me,
        opp,
        {
            "context": 0,
            "option": [
                {"type": 13, "attackId": guide.POWERFUL_HAND_ATTACK},
                {"type": 14},
            ],
            "minCount": 1,
            "maxCount": 1,
        },
    )
    scores = guide.guide_scores(obs, [[0], [1]], deck=guide.EXACT_DECK)
    assert scores is not None
    assert scores[0] < scores[1]

    assert guide.guide_scores(obs, [[0], [99]], deck=guide.EXACT_DECK) is None


def test_enhanced_hammer_prefers_counter_immunity_energy(monkeypatch) -> None:
    monkeypatch.setenv("POKEBOT_ALAKAZAM_NEW_LIST_GUIDE_TARGETS", "1")
    me = _player(active=_pokemon(guide.ALAKAZAM, 140))
    opp = _player(
        active=_pokemon(900, 200, energy=[guide.MIST_ENERGY]),
        bench=[_pokemon(901, 200, energy=[guide.ENRICHING_ENERGY])],
    )
    obs = _obs(
        me,
        opp,
        {
            "context": 30,
            "effect": {"id": guide.ENHANCED_HAMMER},
            "option": [
                {
                    "type": 5,
                    "area": 4,
                    "index": 0,
                    "playerIndex": 1,
                    "energyIndex": 0,
                },
                {
                    "type": 5,
                    "area": 5,
                    "index": 0,
                    "playerIndex": 1,
                    "energyIndex": 0,
                },
            ],
            "minCount": 1,
            "maxCount": 1,
        },
    )
    scores = guide.guide_scores(obs, [[0], [1]], deck=guide.EXACT_DECK)
    assert scores is not None
    assert scores[0] > scores[1]
