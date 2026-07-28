from __future__ import annotations

from poke_bot import thwackey_heuristics as thwackey


DECK = (
    [thwackey.GROOKEY] * 4
    + [thwackey.THWACKEY] * 4
    + [thwackey.APPLIN_TWM] * 4
    + [thwackey.DIPPLIN_TWM] * 4
    + [thwackey.FESTIVAL_GROUNDS] * 4
    + [1] * 40
)


def _player(
    *,
    active: list[int] | None = None,
    bench: list[int] | None = None,
    hand: list[int] | None = None,
) -> dict:
    return {
        "active": [{"id": value} for value in (active or [])],
        "bench": [{"id": value} for value in (bench or [])],
        "hand": [{"id": value} for value in (hand or [])],
        "discard": [],
        "deckCount": 30,
        "prize": [None] * 6,
    }


def _obs(
    me: dict,
    options: list[dict],
    *,
    context: int = thwackey.CTX_MAIN,
    stadium: list[int] | None = None,
) -> dict:
    return {
        "current": {
            "yourIndex": 0,
            "players": [me, _player()],
            "stadium": [{"id": value} for value in (stadium or [])],
            "looking": [],
        },
        "select": {
            "context": context,
            "option": options,
            "minCount": 1,
            "maxCount": 1,
        },
    }


def test_signature_keeps_logical_and_physical_identity_separate() -> None:
    assert thwackey.PHYSICAL_ROUTE_ID == "festival-lead"
    assert thwackey.is_thwackey_deck(DECK)
    assert not thwackey.is_thwackey_deck([1] * 60)


def test_setup_prefers_goldeen_as_exact_festival_lead_active() -> None:
    obs = _obs(
        _player(),
        [
            {"type": thwackey.OPT_CARD, "area": thwackey.AREA_DECK, "index": 0},
            {"type": thwackey.OPT_CARD, "area": thwackey.AREA_DECK, "index": 1},
        ],
        context=thwackey.CTX_SETUP_ACTIVE,
    )
    obs["select"]["deck"] = [
        {"id": thwackey.GOLDEEN_TWM},
        {"id": thwackey.GROOKEY},
    ]
    scores = thwackey.guide_scores(
        obs, [[0], [1]], deck=DECK, force_enabled=True
    )
    assert scores is not None
    assert scores[0] > scores[1] + thwackey.ABSTENTION_MARGIN


def test_main_prefers_visible_core_evolution() -> None:
    me = _player(
        active=[thwackey.GOLDEEN_TWM],
        bench=[thwackey.GROOKEY],
        hand=[thwackey.THWACKEY],
    )
    obs = _obs(
        me,
        [
            {"type": thwackey.OPT_EVOLVE, "area": thwackey.AREA_HAND, "index": 0},
            {"type": 14},
        ],
    )
    scores = thwackey.guide_scores(
        obs, [[0], [1]], deck=DECK, force_enabled=True
    )
    assert scores is not None
    assert scores[0] > scores[1] + thwackey.ABSTENTION_MARGIN


def test_main_prefers_festival_grounds_for_festival_lead_active() -> None:
    me = _player(
        active=[thwackey.DIPPLIN_TWM],
        hand=[thwackey.FESTIVAL_GROUNDS],
    )
    obs = _obs(
        me,
        [
            {"type": thwackey.OPT_PLAY, "index": 0},
            {"type": 14},
        ],
    )
    scores = thwackey.guide_scores(
        obs, [[0], [1]], deck=DECK, force_enabled=True
    )
    assert scores is not None
    assert scores[0] > scores[1] + thwackey.ABSTENTION_MARGIN


def test_boom_prompt_prefers_exact_missing_festival_grounds() -> None:
    me = _player(
        active=[thwackey.DIPPLIN_TWM],
        bench=[thwackey.THWACKEY],
    )
    obs = _obs(
        me,
        [
            {"type": thwackey.OPT_CARD, "area": thwackey.AREA_DECK, "index": 0},
            {"type": thwackey.OPT_CARD, "area": thwackey.AREA_DECK, "index": 1},
        ],
        context=99,
    )
    obs["select"]["effect"] = {"id": thwackey.THWACKEY}
    obs["select"]["deck"] = [
        {"id": thwackey.FESTIVAL_GROUNDS},
        {"id": 1},
    ]
    scores = thwackey.guide_scores(
        obs, [[0], [1]], deck=DECK, force_enabled=True
    )
    assert scores is not None
    assert scores[0] > scores[1] + thwackey.ABSTENTION_MARGIN


def test_unsupported_prompt_masks_complete_stage() -> None:
    obs = _obs(
        _player(),
        [{"type": thwackey.OPT_CARD, "area": thwackey.AREA_DECK, "index": 0}],
        context=99,
    )
    obs["select"]["effect"] = {"id": 999}
    obs["select"]["deck"] = [{"id": thwackey.GROOKEY}]
    assert (
        thwackey.guide_scores(
            obs, [[0]], deck=DECK, force_enabled=True
        )
        is None
    )


def test_malformed_combo_masks_complete_stage() -> None:
    obs = _obs(_player(), [{"type": 14}])
    assert (
        thwackey.guide_scores(
            obs, [[1]], deck=DECK, force_enabled=True
        )
        is None
    )
