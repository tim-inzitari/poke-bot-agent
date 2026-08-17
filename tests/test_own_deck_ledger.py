from __future__ import annotations

import copy
import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from poke_bot.own_deck_ledger import (
    OPTION_FEATURE_DIM,
    OwnDeckLedger,
    OwnDeckLedgerError,
    OwnDeckLedgerSnapshot,
)


def _card(card_id: int | None, serial: int | None = None, **extra):
    result = {"id": card_id, **extra}
    if serial is not None:
        result["serial"] = serial
    return result


def _obs(
    *,
    hand=None,
    active=None,
    bench=None,
    discard=None,
    prize=None,
    deck_count=4,
    looking=None,
    select_deck=None,
    options=None,
    stadium=None,
    opponent=None,
):
    own = {
        "hand": [] if hand is None else hand,
        "active": [] if active is None else active,
        "bench": [] if bench is None else bench,
        "discard": [] if discard is None else discard,
        "prize": [] if prize is None else prize,
        "deckCount": deck_count,
    }
    return {
        "current": {
            "yourIndex": 0,
            "players": [own, opponent or {"hand": [None], "deckCount": 99, "prize": [None]}],
            "looking": [] if looking is None else looking,
            "stadium": [] if stadium is None else stadium,
        },
        "select": {"deck": [] if select_deck is None else select_deck, "option": [] if options is None else options},
    }


def _availability(snapshot, card_id):
    return snapshot.availability_by_card[card_id]


def test_exact_starting_counter_visible_zones_and_prize_uncertainty() -> None:
    # Residual cards are {1: 2, 3: 1}; one is face-down prize and two are in
    # the deck.  This makes card 1 forced-but-not-exact and card 3 uncertain.
    ledger = OwnDeckLedger([1, 1, 1, 2, 3, 3, 4])
    obs = _obs(
        hand=[_card(1, 10)],
        active=[_card(2, 11, energyCards=[_card(3, 12)])],
        discard=[_card(4, 13)],
        prize=[None],
        deck_count=2,
    )
    snapshot = ledger.observe(obs)

    assert ledger.starting_counter == {1: 3, 2: 1, 3: 2, 4: 1}
    assert snapshot.deck_count == 2
    assert snapshot.unknown_prize_slots == 1
    assert snapshot.unknown_non_deck_slots == 1
    assert snapshot.unaccounted_non_deck_slots == 0
    assert _availability(snapshot, 1).lower == 1
    assert _availability(snapshot, 1).upper == 2
    assert _availability(snapshot, 1).expected == pytest.approx(4.0 / 3.0)
    assert _availability(snapshot, 1).probability_at_least_one == 1.0
    assert _availability(snapshot, 1).exact is False
    assert _availability(snapshot, 3).lower == 0
    assert _availability(snapshot, 3).upper == 1
    assert _availability(snapshot, 3).expected == pytest.approx(2.0 / 3.0)
    assert _availability(snapshot, 3).probability_at_least_one == pytest.approx(2.0 / 3.0)
    assert snapshot.integrity_ok is True
    assert snapshot.fail_closed is False


def test_full_select_deck_exposure_makes_deck_multiset_exact() -> None:
    ledger = OwnDeckLedger([1, 1, 1, 2, 3, 3, 4])
    obs = _obs(
        hand=[_card(1, 10)],
        active=[_card(2, 11, energyCards=[_card(3, 12)])],
        discard=[_card(4, 13)],
        prize=[None],
        deck_count=2,
        select_deck=[_card(1, 20), _card(3, 21)],
    )
    snapshot = ledger.observe(obs)
    assert snapshot.prompt_exposure_scope == "select_deck_full_by_count"
    assert _availability(snapshot, 1).lower == _availability(snapshot, 1).upper == 1
    assert _availability(snapshot, 3).lower == _availability(snapshot, 3).upper == 1
    assert _availability(snapshot, 2).lower == _availability(snapshot, 2).upper == 0
    assert all(row.exact for row in snapshot.card_availability)


def test_filtered_select_prompt_tightens_only_current_candidate_bounds() -> None:
    ledger = OwnDeckLedger([1, 1, 1, 2, 3, 3, 4])
    obs = _obs(
        hand=[_card(1, 10)],
        active=[_card(2, 11, energyCards=[_card(3, 12)])],
        discard=[_card(4, 13)],
        prize=[None],
        deck_count=2,
        select_deck=[_card(3, 20)],
    )
    snapshot = ledger.observe(obs)
    assert snapshot.prompt_exposure_scope == "select_deck_candidates"
    assert _availability(snapshot, 3).lower == 1
    assert _availability(snapshot, 3).upper == 1
    assert _availability(snapshot, 3).probability_at_least_one == 1.0
    assert _availability(snapshot, 1).exact is False


def test_revealed_prize_is_retained_only_while_slot_count_is_stable() -> None:
    ledger = OwnDeckLedger([1, 2, 3, 4, 5, 6])
    visible = _obs(prize=[_card(2, 20), None], deck_count=3)
    first = ledger.observe(visible)
    assert dict(first.known_prize_slots) == {0: 2}

    masked_same_count = _obs(prize=[None, None], deck_count=3)
    second = ledger.observe(masked_same_count)
    assert dict(second.known_prize_slots) == {0: 2}
    assert _availability(second, 2).upper == 0

    shrunk = _obs(prize=[None], deck_count=4)
    third = ledger.observe(shrunk)
    assert dict(third.known_prize_slots) == {}
    assert _availability(third, 2).upper == 1


def test_new_direct_prize_slot_invalidates_stale_same_count_history() -> None:
    ledger = OwnDeckLedger([1, 2, 3, 4, 5, 6])
    first = ledger.observe(_obs(prize=[_card(2, 20), None], deck_count=3))
    assert dict(first.known_prize_slots) == {0: 2}

    moved_or_reordered = ledger.observe(
        _obs(prize=[None, _card(3, 21)], deck_count=3)
    )
    assert dict(moved_or_reordered.known_prize_slots) == {1: 3}
    assert _availability(moved_or_reordered, 2).upper == 1
    assert "prize_history_invalidated" in moved_or_reordered.integrity_flags
    assert moved_or_reordered.integrity_ok is True


def test_observe_is_idempotent_and_snapshot_is_immutable_serializable() -> None:
    ledger = OwnDeckLedger([1, 2, 3, 4])
    obs = _obs(hand=[_card(1, 10)], prize=[None], deck_count=2)
    first = ledger.observe(obs)
    again = ledger.observe(copy.deepcopy(obs))
    assert again is first
    assert first.revision == 1
    with pytest.raises(FrozenInstanceError):
        first.deck_count = 7  # type: ignore[misc]

    payload = json.loads(json.dumps(first.to_dict()))
    restored = OwnDeckLedgerSnapshot.from_dict(payload)
    assert restored == first
    assert restored.fingerprint == first.fingerprint
    assert len(restored.scalar_vector) == 10


def test_cg_style_attribute_observation_and_singleton_pre_evolution_are_supported() -> None:
    card = lambda card_id, serial, **fields: SimpleNamespace(
        id=card_id, serial=serial, **fields
    )
    own = SimpleNamespace(
        hand=[card(1, 10)],
        active=[
            card(
                2,
                11,
                energyCards=[card(3, 12)],
                tools=[],
                preEvolution=card(4, 13),
            )
        ],
        bench=[],
        discard=[],
        prize=[None],
        deckCount=1,
    )
    opponent = SimpleNamespace(
        hand=None, active=[], bench=[], discard=[], prize=[None], deckCount=99
    )
    observation = SimpleNamespace(
        current=SimpleNamespace(
            yourIndex=0,
            players=[own, opponent],
            looking=[],
            stadium=[],
        ),
        select=SimpleNamespace(deck=[], option=[]),
    )
    snapshot = OwnDeckLedger([1, 2, 3, 4, 5, 6]).observe(observation)

    assert dict(snapshot.visible_zone_counts)["active"] == ((2, 1), (3, 1), (4, 1))
    assert snapshot.integrity_ok is True
    assert OwnDeckLedgerSnapshot.from_dict(snapshot.to_dict()) == snapshot


def test_attachment_detach_and_evolution_stack_change_refresh_the_fingerprint() -> None:
    ledger = OwnDeckLedger([1, 2, 3, 4, 5, 6, 7, 8])
    first = ledger.observe(
        _obs(
            hand=[_card(1, 10)],
            active=[
                _card(
                    2,
                    11,
                    energyCards=[_card(3, 12)],
                    preEvolution=_card(4, 13),
                )
            ],
            discard=[_card(5, 14)],
            prize=[None],
            deck_count=2,
        )
    )
    changed = ledger.observe(
        _obs(
            hand=[_card(1, 10)],
            # Same active physical root: the energy moved to discard and the
            # evolution stack changed, so stale root-only idempotence is wrong.
            active=[_card(2, 11, energyCards=[], preEvolution=_card(6, 15))],
            discard=[_card(3, 12), _card(5, 14)],
            prize=[None],
            deck_count=2,
        )
    )

    assert changed.revision == first.revision + 1
    assert changed.observation_fingerprint != first.observation_fingerprint
    assert dict(changed.visible_zone_counts)["active"] == ((2, 1), (6, 1))
    assert dict(changed.visible_zone_counts)["discard"] == ((3, 1), (5, 1))
    assert _availability(first, 4).upper == 0
    assert _availability(changed, 4).upper == 1
    assert _availability(first, 6).upper == 1
    assert _availability(changed, 6).upper == 0


def test_reset_and_deepcopy_are_match_scoped() -> None:
    ledger = OwnDeckLedger([1, 2, 3, 4])
    original = ledger.observe(_obs(hand=[_card(1, 10)], prize=[None], deck_count=2))
    branch = copy.deepcopy(ledger)
    changed = branch.observe(_obs(hand=[_card(2, 11)], prize=[None], deck_count=2))
    assert changed.fingerprint != original.fingerprint
    assert ledger.snapshot == original
    ledger.reset()
    assert ledger.snapshot is None
    reset = ledger.observe(_obs(hand=[_card(1, 10)], prize=[None], deck_count=2))
    assert reset.revision == 1
    assert reset.fingerprint == original.fingerprint


def test_option_features_are_current_visible_only_and_stop_is_zero() -> None:
    ledger = OwnDeckLedger([1, 1, 1, 2, 3, 3, 4])
    options = [
        {"type": 10, "area": 1, "index": 0, "playerIndex": 0},
        {"type": 10, "area": 12, "index": 0, "playerIndex": 0},
        {"type": 0},
    ]
    obs = _obs(
        hand=[_card(1, 10)],
        active=[_card(2, 11, energyCards=[_card(3, 12)])],
        discard=[_card(4, 13)],
        prize=[None],
        deck_count=1,
        select_deck=[_card(3, 20)],
        looking=[_card(1, 21)],
        options=options,
    )
    snapshot = ledger.observe(obs)
    rows = snapshot.option_features(obs, [[0], [1], [], [2]])
    assert len(rows) == 4
    assert all(len(row) == OPTION_FEATURE_DIM for row in rows)
    assert rows[0][5] == 1.0  # select.deck occurrence
    assert rows[0][7] == 1.0
    assert rows[1][6] == 1.0  # current.looking occurrence
    assert rows[2] == (0.0,) * OPTION_FEATURE_DIM
    assert rows[3] == (0.0,) * OPTION_FEATURE_DIM


def test_looking_is_physical_transit_and_select_alias_is_not_a_second_deck_copy() -> None:
    ledger = OwnDeckLedger([1, 2, 3, 4])
    transit = _card(2, 20, playerIndex=0)
    obs = _obs(
        hand=[_card(1, 10)],
        prize=[None],
        deck_count=1,
        looking=[transit],
        # This is one physical card serialized in both prompt surfaces.
        select_deck=[dict(transit)],
    )
    snapshot = ledger.observe(obs)

    assert dict(snapshot.visible_zone_counts)["looking"] == ((2, 1),)
    assert snapshot.prompt_exposure_scope == "select_deck_candidates_aliasing_visible"
    assert snapshot.integrity_ok is True
    assert snapshot.fail_closed is False
    assert _availability(snapshot, 2).lower == _availability(snapshot, 2).upper == 0
    assert snapshot.features_for_card(2)[5:8] == (1.0, 1.0, 1.0)


def test_only_explicitly_actor_owned_stadium_enters_the_physical_ledger() -> None:
    owned = OwnDeckLedger([1, 2, 3, 4]).observe(
        _obs(
            hand=[_card(1, 10)],
            prize=[None],
            deck_count=1,
            stadium=[_card(2, 20, playerIndex=0)],
        )
    )
    assert dict(owned.visible_zone_counts)["stadium"] == ((2, 1),)
    assert _availability(owned, 2).upper == 0
    assert owned.integrity_ok is True

    ignored = OwnDeckLedger([1, 2, 3, 4]).observe(
        _obs(
            hand=[_card(1, 10)],
            prize=[None],
            deck_count=2,
            stadium=[_card(2, 20, playerIndex=1)],
        )
    )
    assert dict(ignored.visible_zone_counts)["stadium"] == ()
    assert _availability(ignored, 2).upper == 1
    assert "unowned_stadium_ignored" not in ignored.integrity_flags
    baseline = OwnDeckLedger([1, 2, 3, 4]).observe(
        _obs(hand=[_card(1, 10)], prize=[None], deck_count=2)
    )
    assert ignored.fingerprint == baseline.fingerprint

    ambiguous = OwnDeckLedger([1, 2, 3, 4]).observe(
        _obs(
            hand=[_card(1, 10)],
            prize=[None],
            deck_count=2,
            stadium=[_card(2, 20)],
        )
    )
    assert dict(ambiguous.visible_zone_counts)["stadium"] == ()
    assert "unowned_stadium_ignored" in ambiguous.integrity_flags


def test_option_lookup_does_not_trust_unscoped_direct_card_source() -> None:
    ledger = OwnDeckLedger([1, 2, 3, 4])
    base = _obs(
        hand=[_card(1, 10)],
        prize=[None],
        deck_count=2,
        select_deck=[_card(3, 20)],
        options=[
            {
                "type": 10,
                "area": 1,
                "index": 0,
                "playerIndex": 0,
                "card": _card(999, 999, playerIndex=1),
            }
        ],
    )
    snapshot = ledger.observe(base)
    changed = copy.deepcopy(base)
    changed["select"]["option"][0]["card"] = _card(998, 998, playerIndex=1)

    assert snapshot.option_features(base, [[0]]) == snapshot.option_features(changed, [[0]])
    assert snapshot.option_features(base, [[0]])[0][5] == 1.0


def test_opponent_and_transition_after_fields_are_ignored() -> None:
    ledger = OwnDeckLedger([1, 2, 3, 4])
    base = _obs(hand=[_card(1, 10)], prize=[None], deck_count=2)
    changed = copy.deepcopy(base)
    changed["current"]["players"][1] = {
        "hand": [_card(999, 999)],
        "deck": [_card(998, 998)],
        "prize": [_card(997, 997)],
    }
    changed["transition_after"] = {
        "current": {"players": [{"deck": [_card(996, 996)]}, {}]}
    }
    first = ledger.observe(base)
    ledger.reset()
    second = ledger.observe(changed)
    assert second.fingerprint == first.fingerprint
    assert second.to_dict()["card_availability"] == first.to_dict()["card_availability"]


def test_unkeyed_or_conflicting_visible_cards_fail_closed_without_hidden_guessing() -> None:
    ledger = OwnDeckLedger([1, 2, 3, 4])
    unkeyed = ledger.observe(_obs(hand=[_card(1)], prize=[None], deck_count=2))
    assert unkeyed.fail_closed is True
    assert "unkeyed_visible_card:hand" in unkeyed.integrity_flags

    ledger.reset()
    conflict = ledger.observe(
        _obs(hand=[_card(1, 10)], discard=[_card(2, 10)], prize=[None], deck_count=1)
    )
    assert conflict.fail_closed is True
    assert "serial_card_identity_conflict" in conflict.integrity_flags


def test_invalid_starting_deck_fails_closed_at_construction() -> None:
    with pytest.raises(OwnDeckLedgerError):
        OwnDeckLedger([])
    with pytest.raises(OwnDeckLedgerError):
        OwnDeckLedger([1, -1])


def test_dormant_contract_does_not_activate_runtime() -> None:
    # This core is intentionally pure: constructing/observing it must have no
    # runtime selection authority until a receipt-proven future refresh wires it.
    ledger = OwnDeckLedger([1, 2, 3, 4])
    snapshot = ledger.observe(_obs(hand=[_card(1, 10)], prize=[None], deck_count=2))
    assert snapshot.schema == "poke_bot.own_deck_ledger/v1"
    assert not hasattr(ledger, "select")
