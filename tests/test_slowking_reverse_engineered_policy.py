from __future__ import annotations

from poke_bot.slowking_reverse_engineered_policy import (
    ACADEMY_AT_NIGHT,
    AREA_HAND,
    CTX_SETUP_ACTIVE,
    CTX_TOP_DECK,
    KYUREM,
    MEGA_KANGASKHAN_EX,
    OPT_CARD,
    SLOWKING,
    SLOWPOKE,
    SMOOCHUM,
    audit_decision,
    choose_action,
    is_slowking_archetype,
    prior_logit_bias,
)


def deck() -> list[int]:
    return [SLOWKING] * 4 + [SLOWPOKE] * 4 + list(range(10_000, 10_052))


def observation(*, context: int, hand: list[int], active: list[int], first: int = 0, seat: int = 0, effect: int | None = None) -> dict:
    select = {
        "context": context,
        "minCount": 1,
        "maxCount": 1,
        "option": [
            {"type": OPT_CARD, "area": AREA_HAND, "index": index, "playerIndex": seat}
            for index in range(len(hand))
        ],
    }
    if effect is not None:
        select["effect"] = {"id": effect}
    return {
        "current": {
            "yourIndex": seat,
            "firstPlayer": first,
            "players": [
                {
                    "hand": [{"id": card} for card in hand],
                    "active": [{"id": card} for card in active],
                    "bench": [],
                    "discard": [],
                },
                {"hand": [], "active": [], "bench": [], "discard": []},
            ],
            "stadium": [],
        },
        "select": select,
    }


def test_archetype_gate_is_not_exact_list_gate() -> None:
    assert is_slowking_archetype(deck())
    assert not is_slowking_archetype(deck()[:-1])
    assert not is_slowking_archetype(list(range(60)))


def test_opening_policy_uses_recovered_stable_priority_in_both_orders() -> None:
    hand = [MEGA_KANGASKHAN_EX, SMOOCHUM, SLOWPOKE]
    combos = [(0,), (1,), (2,)]
    first = observation(context=CTX_SETUP_ACTIVE, hand=hand, active=[], first=0, seat=0)
    second = observation(context=CTX_SETUP_ACTIVE, hand=hand, active=[], first=1, seat=0)
    assert choose_action(first, combos, deck=deck()) == (0,)
    assert choose_action(second, combos, deck=deck()) == (0,)


def test_academy_payload_rule_requires_active_slowking() -> None:
    hand = [KYUREM, SLOWPOKE]
    combos = [(0,), (1,)]
    attacking = observation(
        context=CTX_TOP_DECK,
        hand=hand,
        active=[SLOWKING],
        effect=ACADEMY_AT_NIGHT,
    )
    waiting = observation(
        context=CTX_TOP_DECK,
        hand=hand,
        active=[SLOWPOKE],
        effect=ACADEMY_AT_NIGHT,
    )
    assert choose_action(attacking, combos, deck=deck()) == (0,)
    assert audit_decision(waiting, combos, deck=deck()) is None


def test_malformed_and_runtime_paths_fail_closed() -> None:
    obs = observation(context=CTX_SETUP_ACTIVE, hand=[SLOWPOKE], active=[])
    assert audit_decision(obs, [(99,)], deck=deck()) is None
    assert prior_logit_bias(obs, [(0,), (1,)]) == [0.0, 0.0]
