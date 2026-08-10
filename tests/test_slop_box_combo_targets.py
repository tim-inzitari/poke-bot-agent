from __future__ import annotations

import copy

import pytest

from poke_bot.combo_state_contract import (
    COMBO_STATE_KEY,
    SLOP_BOX_COMBO_STATE_TARGET_SCHEMA,
    validate_combo_state_labels,
)
from poke_bot.slop_box_combo_targets import (
    CRISPIN,
    ENERGY_SWITCH,
    GLASS_TRUMPET,
    GRASS_ENERGY,
    LIGHTNING_ENERGY,
    SEEK_CRISPIN,
    SEEK_ENERGY_SWITCH,
    SEEK_TEAL_DANCE,
    TEAL_MASK_OGERPON_EX,
    attach_slop_box_combo_state_labels,
    build_slop_box_combo_state_target,
)
from poke_bot.teal_mask_ogerpon_heuristics import CORE_SIGNATURE_MINIMUMS
from scripts.train_pure_rl import _record_to_compact_game


def _deck() -> list[int]:
    cards: list[int] = []
    for card_id, count in CORE_SIGNATURE_MINIMUMS.items():
        cards.extend([int(card_id)] * int(count))
    # Pad to 60 with Grass Energy while preserving signature minimums.
    while len(cards) < 60:
        cards.append(GRASS_ENERGY)
    return cards[:60]


def _observation(
    *,
    context: int,
    effect: int | dict | None,
    options: list[dict],
    hand: list[dict] | None = None,
    active: list | dict | None = None,
    bench: list[dict] | None = None,
) -> dict:
    if active is None:
        active_zone: list | dict = [{"id": TEAL_MASK_OGERPON_EX}]
    elif isinstance(active, dict):
        active_zone = [active]
    else:
        active_zone = active
    return {
        "current": {
            "yourIndex": 0,
            "players": [
                {
                    "hand": hand
                    or [
                        {"id": CRISPIN},
                        {"id": ENERGY_SWITCH},
                        {"id": GRASS_ENERGY},
                    ],
                    "active": active_zone,
                    "bench": bench or [{"id": 63}],
                    "discard": [{"id": GLASS_TRUMPET}],
                },
                {"hand": None, "active": None, "bench": []},
            ],
        },
        "select": {
            "context": context,
            "effect": effect,
            "option": options,
        },
    }


def test_crispin_energy_class_and_seek_source() -> None:
    target = build_slop_box_combo_state_target(
        _observation(
            context=9,
            effect={"id": CRISPIN},
            options=[{"id": GRASS_ENERGY}, {"id": LIGHTNING_ENERGY}],
        ),
        [1],
    )
    assert target["schema"] == SLOP_BOX_COMBO_STATE_TARGET_SCHEMA
    assert target["top_deck"] == {"target": 2, "mask": True}  # Lightning
    assert target["seek_source"] == {"target": SEEK_CRISPIN, "mask": True}


def test_teal_dance_ability_selection_is_labeled() -> None:
    target = build_slop_box_combo_state_target(
        _observation(
            context=0,
            effect=None,
            options=[
                {"type": 7, "area": 2, "index": 0, "playerIndex": 0},
                {
                    "type": 10,
                    "area": 4,
                    "index": 0,
                    "playerIndex": 0,
                },
            ],
            hand=[{"id": ENERGY_SWITCH}],
            active=[{"id": TEAL_MASK_OGERPON_EX}],
        ),
        [1],
    )
    assert target["seek_source"] == {"target": SEEK_TEAL_DANCE, "mask": True}
    # Engine-legality channel 0 is Teal Dance legal on main.
    assert target["vector"]["mask"][0] is True
    assert target["vector"]["target"][0] == 1.0


def test_energy_switch_seek_and_vector_continuity() -> None:
    target = build_slop_box_combo_state_target(
        _observation(
            context=22,
            effect={"id": ENERGY_SWITCH},
            options=[{"id": GRASS_ENERGY}, {"id": LIGHTNING_ENERGY}],
            active=[{"id": 63}],
            bench=[{"id": TEAL_MASK_OGERPON_EX}],
        ),
        [0],
    )
    assert target["seek_source"] == {"target": SEEK_ENERGY_SWITCH, "mask": True}
    # Continuity: teal active/bench, bolt active/bench → indices 16..19
    assert target["vector"]["target"][16:20] == [0.0, 1.0, 1.0, 0.0]


def test_suffix_cannot_change_target_and_unknown_top_is_masked() -> None:
    observation = _observation(
        context=0,
        effect=None,
        options=[{"type": 12}],
    )
    before = build_slop_box_combo_state_target(observation, [0])
    changed = copy.deepcopy(observation)
    changed["noncausal_future_suffix"] = {"winner": 1, "deck_order": [1, 3, 4]}
    after = build_slop_box_combo_state_target(changed, [0])
    assert before == after
    assert before["top_deck"] == {"target": None, "mask": False}


def test_attach_rejects_non_slop_box_deck() -> None:
    step = {
        "observation": _observation(
            context=9,
            effect={"id": CRISPIN},
            options=[{"id": GRASS_ENERGY}],
        ),
        "action": [0],
        "aux_labels": {},
    }
    coverage = attach_slop_box_combo_state_labels([step], deck=_deck())
    assert coverage["decisions"] == 1
    assert coverage["seek_crispin"] == 1
    validate_combo_state_labels(step["aux_labels"][COMBO_STATE_KEY])

    with pytest.raises(ValueError, match="Slop Box"):
        attach_slop_box_combo_state_labels(
            [copy.deepcopy(step)],
            deck=[9999] * 60,
        )


def test_live_rl_compaction_attaches_slop_box_combo_targets() -> None:
    record = {
        "episode_id": "slop-box-live-rl",
        "seat": 0,
        "archetype": "teal-mask-ogerpon-ex",
        "opp_archetype": "unknown",
        "deck": _deck(),
        "value": 1.0,
        "decisions": [
            {
                "env_step": 1,
                "selected_index": 0,
                "n_options": 2,
                "observation": _observation(
                    context=9,
                    effect={"id": CRISPIN},
                    options=[{"id": GRASS_ENERGY}, {"id": LIGHTNING_ENERGY}],
                ),
                "action": [0],
                "aux_labels": {},
            }
        ],
    }
    compact = _record_to_compact_game(record)
    assert compact is not None
    assert compact.target_provenance["slop_box_combo_state_targets"]["decisions"] == 1
    validate_combo_state_labels(
        compact.decisions[0].aux_labels[COMBO_STATE_KEY]
    )
