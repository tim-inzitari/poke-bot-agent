from __future__ import annotations

from poke_bot import alakazam_heuristics as guide


ALAKAZAM_DECK = [guide.ABRA] * 2 + [guide.KADABRA] * 2 + [guide.ALAKAZAM] * 2


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


def _obs(me: dict, opp: dict, select: dict) -> dict:
    return {
        "current": {
            "yourIndex": 0,
            "players": [me, opp],
            "stadium": [],
            "looking": [],
        },
        "select": select,
    }


def test_targets_are_fail_closed_and_alakazam_only(monkeypatch) -> None:
    me = _player()
    opp = _player()
    obs = _obs(
        me,
        opp,
        {"context": 0, "option": [{"type": 14}], "minCount": 1, "maxCount": 1},
    )
    monkeypatch.delenv("POKEBOT_ALAKAZAM_GUIDE_TARGETS", raising=False)
    assert guide.guide_scores(obs, [[0]], deck=ALAKAZAM_DECK) is None

    monkeypatch.setenv("POKEBOT_ALAKAZAM_GUIDE_TARGETS", "1")
    assert guide.guide_scores(obs, [[0]], deck=[1] * 60) is None


def test_immediate_powerful_hand_ko_beats_ending_turn(monkeypatch) -> None:
    monkeypatch.setenv("POKEBOT_ALAKAZAM_GUIDE_TARGETS", "1")
    me = _player(
        hand_count=14,
        active=_pokemon(guide.ALAKAZAM, 140, energy=[guide.PSYCHIC_ENERGY]),
    )
    opp = _player(active=_pokemon(900, 280))
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
    scores = guide.guide_scores(obs, [[0], [1]], deck=ALAKAZAM_DECK)
    assert scores is not None
    assert scores[0] > scores[1] + 5.0


def test_optional_draw_refuses_a_deckout(monkeypatch) -> None:
    monkeypatch.setenv("POKEBOT_ALAKAZAM_GUIDE_TARGETS", "1")
    me = _player(deck_count=2, active=_pokemon(guide.ALAKAZAM, 140))
    opp = _player(active=_pokemon(900, 200))
    obs = _obs(
        me,
        opp,
        {
            "context": 43,
            "contextCard": {"id": guide.ALAKAZAM},
            "option": [{"type": 1}, {"type": 2}],
            "minCount": 1,
            "maxCount": 1,
        },
    )
    scores = guide.guide_scores(obs, [[0], [1]], deck=ALAKAZAM_DECK)
    assert scores is not None
    assert scores[1] > scores[0]


def test_enriching_energy_prefers_the_recyclable_dudunsparce(monkeypatch) -> None:
    monkeypatch.setenv("POKEBOT_ALAKAZAM_GUIDE_TARGETS", "1")
    me = _player(
        hand=[guide.ENRICHING_ENERGY],
        active=_pokemon(guide.ALAKAZAM, 140),
        bench=[_pokemon(guide.DUDUNSPARCE, 140)],
    )
    opp = _player(active=_pokemon(900, 200))
    obs = _obs(
        me,
        opp,
        {
            "context": 0,
            "option": [
                {
                    "type": 8,
                    "area": 2,
                    "index": 0,
                    "inPlayArea": 4,
                    "inPlayIndex": 0,
                },
                {
                    "type": 8,
                    "area": 2,
                    "index": 0,
                    "inPlayArea": 5,
                    "inPlayIndex": 0,
                },
            ],
            "minCount": 1,
            "maxCount": 1,
        },
    )
    scores = guide.guide_scores(obs, [[0], [1]], deck=ALAKAZAM_DECK)
    assert scores is not None
    assert scores[1] > scores[0]


def test_boss_target_ranking_uses_post_boss_hand_math(monkeypatch) -> None:
    monkeypatch.setenv("POKEBOT_ALAKAZAM_GUIDE_TARGETS", "1")
    me = _player(hand_count=6, active=_pokemon(guide.ALAKAZAM, 140))
    opp = _player(
        active=_pokemon(900, 200),
        bench=[_pokemon(901, 120), _pokemon(902, 300)],
    )
    obs = _obs(
        me,
        opp,
        {
            "context": 3,
            "effect": {"id": guide.BOSS_ORDERS},
            "option": [
                {"type": 3, "area": 5, "index": 0, "playerIndex": 1},
                {"type": 3, "area": 5, "index": 1, "playerIndex": 1},
            ],
            "minCount": 1,
            "maxCount": 1,
        },
    )
    scores = guide.guide_scores(obs, [[0], [1]], deck=ALAKAZAM_DECK)
    assert scores is not None
    assert scores[0] > scores[1]


def test_enhanced_hammer_prioritizes_mist_energy(monkeypatch) -> None:
    monkeypatch.setenv("POKEBOT_ALAKAZAM_GUIDE_TARGETS", "1")
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
    scores = guide.guide_scores(obs, [[0], [1]], deck=ALAKAZAM_DECK)
    assert scores is not None
    assert scores[0] > scores[1]
