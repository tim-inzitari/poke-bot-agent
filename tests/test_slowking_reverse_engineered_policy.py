"""Unit tests for the research-only Slowking reverse-engineered surrogate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from poke_bot.slowking_reverse_engineered_policy import (
    ACADEMY_AT_NIGHT,
    ANNIHILAPE,
    AREA_DISCARD,
    CONKELDURR,
    CTX_SETUP_ACTIVE,
    CTX_SETUP_BENCH,
    CTX_TO_HAND,
    CTX_TOP_DECK,
    FEZANDIPITI_EX,
    KYUREM,
    LATIAS_EX,
    MEGA_KANGASKHAN_EX,
    NIGHT_STRETCHER,
    OPT_CARD,
    OPT_PLAY,
    POLICY_VERSION,
    RESEARCH_ONLY,
    RUNTIME_AUTHORITY,
    SLOWKING,
    SLOWPOKE,
    SMOOCHUM,
    audit_decision,
    choose_action,
    is_slowking_archetype,
    prior_logit_bias,
)

ROOT = Path(__file__).resolve().parents[1]
FINAL_DECK = json.loads(
    (ROOT / "state" / "slowking_top_replay_distillation_2026-08-04.json").read_text(
        encoding="utf-8"
    )
)
# Expand fingerprint card counts into a 60-card list.
_FINAL_COUNTS = {
    row["card_id"]: row["count"] for row in FINAL_DECK["identity"]["decks"][0]["cards"]
}
FINAL_LIST: list[int] = []
for card_id, count in sorted(_FINAL_COUNTS.items()):
    FINAL_LIST.extend([card_id] * count)


def _player(
    *,
    active: list[int] | None = None,
    bench: list[int] | None = None,
    hand: list[int] | None = None,
    discard: list[int] | None = None,
) -> dict:
    return {
        "active": [{"id": value} for value in (active or [])],
        "bench": [{"id": value} for value in (bench or [])],
        "hand": [{"id": value} for value in (hand or [])],
        "discard": [{"id": value} for value in (discard or [])],
    }


def _obs(
    me: dict,
    options: list[dict],
    *,
    context: int,
    effect_id: int | None = None,
    your_index: int = 0,
    first_player: int = 0,
) -> dict:
    select: dict = {
        "context": context,
        "option": options,
        "minCount": 1,
        "maxCount": 1,
    }
    if effect_id is not None:
        select["effect"] = {"id": effect_id}
    return {
        "current": {
            "yourIndex": your_index,
            "firstPlayer": first_player,
            "players": [me, _player()],
            "stadium": [],
            "looking": [],
        },
        "select": select,
    }


@pytest.mark.unit
def test_archetype_gate_and_authority_constants() -> None:
    assert is_slowking_archetype(FINAL_LIST)
    assert not is_slowking_archetype([SLOWKING] * 59)
    assert RESEARCH_ONLY is True
    assert RUNTIME_AUTHORITY == "none"
    assert POLICY_VERSION.startswith("slowking-public-replay-surrogate")


@pytest.mark.unit
def test_serving_logit_bypass_is_exact_zero() -> None:
    bias = prior_logit_bias({}, [[0], [1], [2]], scale=10.0)
    assert bias == [0.0, 0.0, 0.0]


@pytest.mark.unit
def test_opening_active_priority_kangaskhan_over_smoochum() -> None:
    me = _player(hand=[MEGA_KANGASKHAN_EX, SMOOCHUM, SLOWPOKE, KYUREM])
    options = [
        {"type": OPT_PLAY, "index": 0},  # Kangaskhan
        {"type": OPT_PLAY, "index": 1},  # Smoochum
        {"type": OPT_PLAY, "index": 2},  # Slowpoke
        {"type": OPT_PLAY, "index": 3},  # Kyurem
    ]
    obs = _obs(me, options, context=CTX_SETUP_ACTIVE, first_player=0, your_index=0)
    combos = [[0], [1], [2], [3]]
    audit = audit_decision(obs, combos, deck=FINAL_LIST)
    assert audit is not None
    assert audit["stage_class"] == "opening_active"
    assert audit["preferred_combo_index"] == 0
    assert choose_action(obs, combos, deck=FINAL_LIST) == (0,)


@pytest.mark.unit
def test_opening_active_priority_stable_going_second() -> None:
    me = _player(hand=[SMOOCHUM, LATIAS_EX, SLOWPOKE])
    options = [
        {"type": OPT_PLAY, "index": 0},
        {"type": OPT_PLAY, "index": 1},
        {"type": OPT_PLAY, "index": 2},
    ]
    obs = _obs(me, options, context=CTX_SETUP_ACTIVE, first_player=1, your_index=0)
    audit = audit_decision(obs, [[0], [1], [2]], deck=FINAL_LIST)
    assert audit is not None
    assert audit["preferred_combo_index"] == 0  # Smoochum


@pytest.mark.unit
def test_opening_bench_prefers_second_slowpoke() -> None:
    me = _player(active=[MEGA_KANGASKHAN_EX], bench=[SLOWPOKE], hand=[SLOWPOKE, FEZANDIPITI_EX])
    options = [
        {"type": OPT_PLAY, "index": 0},  # Slowpoke
        {"type": OPT_PLAY, "index": 1},  # Fez
    ]
    # hand indices after active/bench construction: hand[0]=Slowpoke, hand[1]=Fez
    me = _player(
        active=[MEGA_KANGASKHAN_EX],
        bench=[SLOWPOKE],
        hand=[SLOWPOKE, FEZANDIPITI_EX],
    )
    obs = _obs(me, options, context=CTX_SETUP_BENCH)
    audit = audit_decision(obs, [[0], [1]], deck=FINAL_LIST)
    assert audit is not None
    assert audit["stage_class"] == "opening_bench"
    assert audit["preferred_combo_index"] == 0


@pytest.mark.unit
def test_night_stretcher_high_confidence_slowking_only() -> None:
    # Two Slowpoke already on board so Slowpoke recovery is not competitive;
    # open evolution line makes Slowking the unique high-margin choice.
    me = _player(
        active=[SLOWPOKE],
        bench=[SLOWPOKE],
        discard=[SLOWKING, KYUREM, ANNIHILAPE],
    )
    options = [
        {"type": OPT_CARD, "area": AREA_DISCARD, "index": 0},  # Slowking
        {"type": OPT_CARD, "area": AREA_DISCARD, "index": 1},  # Kyurem
        {"type": OPT_CARD, "area": AREA_DISCARD, "index": 2},  # Annihilape
    ]
    obs = _obs(me, options, context=CTX_TO_HAND, effect_id=NIGHT_STRETCHER)
    audit = audit_decision(obs, [[0], [1], [2]], deck=FINAL_LIST)
    assert audit is not None
    assert audit["stage_class"] == "night_stretcher_recovery"
    assert choose_action(obs, [[0], [1], [2]], deck=FINAL_LIST) == (0,)


@pytest.mark.unit
def test_night_stretcher_abstains_without_slowking_top() -> None:
    me = _player(active=[SLOWPOKE], discard=[KYUREM, SLOWPOKE])
    options = [
        {"type": OPT_CARD, "area": AREA_DISCARD, "index": 0},
        {"type": OPT_CARD, "area": AREA_DISCARD, "index": 1},
    ]
    obs = _obs(me, options, context=CTX_TO_HAND, effect_id=NIGHT_STRETCHER)
    assert audit_decision(obs, [[0], [1]], deck=FINAL_LIST) is None


@pytest.mark.unit
def test_academy_seek_topdeck_when_slowking_active() -> None:
    me = _player(
        active=[SLOWKING],
        hand=[KYUREM, CONKELDURR, LATIAS_EX],
    )
    options = [
        {"type": OPT_PLAY, "index": 0},  # Kyurem
        {"type": OPT_PLAY, "index": 1},  # Conkeldurr
        {"type": OPT_PLAY, "index": 2},  # Latias
    ]
    obs = _obs(me, options, context=CTX_TOP_DECK, effect_id=ACADEMY_AT_NIGHT)
    audit = audit_decision(obs, [[0], [1], [2]], deck=FINAL_LIST)
    assert audit is not None
    assert audit["stage_class"] == "academy_seek_topdeck"
    assert audit["preferred_combo_index"] == 0


@pytest.mark.unit
def test_academy_abstains_when_slowking_not_active() -> None:
    me = _player(active=[SLOWPOKE], hand=[KYUREM, CONKELDURR])
    options = [
        {"type": OPT_PLAY, "index": 0},
        {"type": OPT_PLAY, "index": 1},
    ]
    obs = _obs(me, options, context=CTX_TOP_DECK, effect_id=ACADEMY_AT_NIGHT)
    assert audit_decision(obs, [[0], [1]], deck=FINAL_LIST) is None


@pytest.mark.unit
def test_unsupported_main_phase_prompt_abstains() -> None:
    me = _player(active=[SLOWKING], hand=[KYUREM, SLOWPOKE])
    options = [
        {"type": OPT_PLAY, "index": 0},
        {"type": OPT_PLAY, "index": 1},
    ]
    obs = _obs(me, options, context=0)  # main / unsupported
    assert audit_decision(obs, [[0], [1]], deck=FINAL_LIST) is None
