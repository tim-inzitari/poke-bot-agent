"""Causal replay targets for Slop Box / Teal Mask Ogerpon ex combo_state.

Maps into the existing generic H10 32-d combo head without width remap:

  top_deck[5]      Crispin / Glass Trumpet selected Basic Energy class
  seek_source[7]   Teal Dance / Crispin / Glass Trumpet / Energy Switch /
                   Area Zero / recovery / other engine source
  vector[20]       engine legality (6) + visible pieces (5) +
                   energy-route readiness (5) + engine continuity (4)

Schema: poke_bot.slop_box_combo_state_targets/v1

Only the acting player's masked observation, current legal options, and
recorded action are read.
"""

from __future__ import annotations

from typing import Any

from . import archetypes
from .combo_observation import (
    all_card_ids,
    card_id,
    chosen_card_ids,
    context,
    effect_card,
    option_card,
    own_player,
    own_public_cards,
)
from .combo_state_contract import (
    COMBO_STATE_KEY,
    SLOP_BOX_COMBO_STATE_TARGET_SCHEMA,
    VECTOR_WIDTH,
    validate_combo_state_labels,
)
from .teal_mask_ogerpon_heuristics import (
    MEGA_KANGASKHAN_EX,
    MEOWTH_EX,
    RAGING_BOLT_EX,
    TEAL_MASK_OGERPON_EX,
    is_teal_mask_ogerpon_ex_deck,
)

GRASS_ENERGY = 1
WATER_ENERGY = 3
LIGHTNING_ENERGY = 4
PSYCHIC_ENERGY = 5
FIGHTING_ENERGY = 6
BASIC_ENERGIES = (
    GRASS_ENERGY,
    WATER_ENERGY,
    LIGHTNING_ENERGY,
    PSYCHIC_ENERGY,
    FIGHTING_ENERGY,
)
ENERGY_CLASS = {
    GRASS_ENERGY: 0,
    WATER_ENERGY: 1,
    LIGHTNING_ENERGY: 2,
    PSYCHIC_ENERGY: 3,
    FIGHTING_ENERGY: 4,
}

CRISPIN = 1198
GLASS_TRUMPET = 1098
ENERGY_SWITCH = archetypes.ENERGY_SWITCH  # 1116
AREA_ZERO = archetypes.AREA_ZERO_UNDERDEPTHS  # 1250
NIGHT_STRETCHER = 1097
ULTRA_BALL = 1121
BOSS_ORDERS = 1182
CYRANO = 1197
XEROSIC = 1205
NEST_BALL = 1088

ENGINE_SUPPORTERS = {CRISPIN, GLASS_TRUMPET, ENERGY_SWITCH}
SEARCH_OR_DISRUPTION = {ULTRA_BALL, NEST_BALL, BOSS_ORDERS, CYRANO, XEROSIC}
RECOVERY = {NIGHT_STRETCHER}

OPT_ABILITY = 10
CTX_MAIN = 0

# seek_source classes
SEEK_TEAL_DANCE = 0
SEEK_CRISPIN = 1
SEEK_GLASS_TRUMPET = 2
SEEK_ENERGY_SWITCH = 3
SEEK_AREA_ZERO = 4
SEEK_RECOVERY = 5
SEEK_OTHER_ENGINE = 6


def is_slop_box_combo_deck(deck: list[int]) -> bool:
    """Family-signature deck check (multi-list Slop Box / teal-mask)."""

    return is_teal_mask_ogerpon_ex_deck(deck)


def _select_card_ids(observation: dict[str, Any]) -> set[int]:
    return {
        card
        for card in (
            card_id(option_card(observation, option))
            for option in (
                (observation.get("select") or {}).get("option") or []
            )
        )
        if card is not None
    }


def _ability_origin_ids(observation: dict[str, Any]) -> set[int]:
    """Card ids whose ability options are currently legal."""

    origins: set[int] = set()
    for option in (observation.get("select") or {}).get("option") or []:
        if not isinstance(option, dict):
            continue
        if option.get("type") != OPT_ABILITY:
            continue
        found = card_id(option_card(observation, option))
        if found is not None:
            origins.add(found)
    return origins


def _hand_ids(observation: dict[str, Any]) -> set[int]:
    player = own_player(observation)
    return set(all_card_ids(player.get("hand")))


def _energy_class_target(chosen: list[int]) -> int | None:
    classes = [ENERGY_CLASS[card] for card in chosen if card in ENERGY_CLASS]
    if not classes:
        return None
    # Crispin / Trumpet expose one or more Basic Energies; use the last
    # demonstrated selection as the resulting typed top/route energy.
    return int(classes[-1])


def _seek_source(
    *,
    effect: int | None,
    ctx: int,
    ability_origins: set[int],
    action: list[int],
    observation: dict[str, Any],
) -> int | None:
    if effect == CRISPIN:
        return SEEK_CRISPIN
    if effect == GLASS_TRUMPET:
        return SEEK_GLASS_TRUMPET
    if effect == ENERGY_SWITCH:
        return SEEK_ENERGY_SWITCH
    if effect == AREA_ZERO:
        return SEEK_AREA_ZERO
    if effect in RECOVERY:
        return SEEK_RECOVERY
    if effect in SEARCH_OR_DISRUPTION:
        return SEEK_OTHER_ENGINE
    if effect == TEAL_MASK_OGERPON_EX:
        return SEEK_TEAL_DANCE
    # Teal Dance: main-phase ability selection whose origin is Teal Mask.
    if ctx == CTX_MAIN and TEAL_MASK_OGERPON_EX in ability_origins and action:
        options = list((observation.get("select") or {}).get("option") or [])
        for raw_index in action:
            index = int(raw_index)
            if not 0 <= index < len(options):
                continue
            option = options[index]
            if (
                isinstance(option, dict)
                and option.get("type") == OPT_ABILITY
                and card_id(option_card(observation, option))
                == TEAL_MASK_OGERPON_EX
            ):
                return SEEK_TEAL_DANCE
    return None


def _visible_vector(
    observation: dict[str, Any],
) -> tuple[list[float | None], list[bool]]:
    cards = set(own_public_cards(observation))
    hand = _hand_ids(observation)
    ability_origins = _ability_origin_ids(observation)
    select_cards = _select_card_ids(observation)
    ctx = context(observation)
    player = own_player(observation)
    active_ids = set(all_card_ids(player.get("active")))
    bench_ids = set(all_card_ids(player.get("bench")))

    teal_dance_legal = (
        float(TEAL_MASK_OGERPON_EX in ability_origins)
        if ctx == CTX_MAIN
        else None
    )
    crispin_legal = float(
        CRISPIN in hand
        or CRISPIN in select_cards
        or effect_card(observation) == CRISPIN
    )
    trumpet_legal = float(
        GLASS_TRUMPET in hand
        or GLASS_TRUMPET in select_cards
        or effect_card(observation) == GLASS_TRUMPET
    )
    switch_legal = float(
        ENERGY_SWITCH in hand
        or ENERGY_SWITCH in select_cards
        or effect_card(observation) == ENERGY_SWITCH
    )
    area_zero_available = float(
        AREA_ZERO in cards or AREA_ZERO in select_cards
    )
    any_engine = float(
        bool(
            (teal_dance_legal or 0.0)
            or crispin_legal
            or trumpet_legal
            or switch_legal
            or area_zero_available
        )
    )
    engine_legality = [
        teal_dance_legal,
        crispin_legal,
        trumpet_legal,
        switch_legal,
        area_zero_available,
        any_engine,
    ]

    pieces = [
        float(TEAL_MASK_OGERPON_EX in cards),
        float(RAGING_BOLT_EX in cards),
        float(bool(cards & {MEGA_KANGASKHAN_EX, MEOWTH_EX})),
        float(bool(cards & ENGINE_SUPPORTERS)),
        float(bool(cards & (RECOVERY | {ULTRA_BALL, NEST_BALL}))),
    ]

    basic_visible = cards & set(BASIC_ENERGIES)
    energy_route = [
        float(GRASS_ENERGY in cards),
        float(bool(basic_visible - {GRASS_ENERGY})),
        float(ENERGY_SWITCH in hand),
        float(len(basic_visible) >= 2),
        float(
            TEAL_MASK_OGERPON_EX in cards
            and (
                GRASS_ENERGY in cards
                or TEAL_MASK_OGERPON_EX in ability_origins
            )
        ),
    ]

    continuity = [
        float(TEAL_MASK_OGERPON_EX in active_ids),
        float(TEAL_MASK_OGERPON_EX in bench_ids),
        float(RAGING_BOLT_EX in active_ids),
        float(RAGING_BOLT_EX in bench_ids),
    ]

    values = engine_legality + pieces + energy_route + continuity
    if len(values) != VECTOR_WIDTH:
        raise AssertionError("Slop Box combo vector width drift")
    return (
        [None if value is None else float(value) for value in values],
        [value is not None for value in values],
    )


def build_slop_box_combo_state_target(
    observation: dict[str, Any],
    action: list[int],
) -> dict[str, Any]:
    """Build one decision target without accepting hidden/future state."""

    ctx = context(observation)
    effect = effect_card(observation)
    chosen = chosen_card_ids(observation, action)
    ability_origins = _ability_origin_ids(observation)

    top_target: int | None = None
    if effect in {CRISPIN, GLASS_TRUMPET}:
        top_target = _energy_class_target(chosen)

    seek_target = _seek_source(
        effect=effect,
        ctx=ctx,
        ability_origins=ability_origins,
        action=action,
        observation=observation,
    )

    vector, vector_mask = _visible_vector(observation)
    target = {
        "schema": SLOP_BOX_COMBO_STATE_TARGET_SCHEMA,
        "top_deck": {
            "target": top_target,
            "mask": top_target is not None,
        },
        "seek_source": {
            "target": seek_target,
            "mask": seek_target is not None,
        },
        "vector": {"target": vector, "mask": vector_mask},
    }
    validate_combo_state_labels(target)
    return target


def attach_slop_box_combo_state_labels(
    steps: list[dict[str, Any]],
    *,
    deck: list[int],
) -> dict[str, int]:
    """Attach strict Slop Box combo labels to an acting-seat record in place."""

    if not is_slop_box_combo_deck(deck):
        raise ValueError("combo-state labels require a Slop Box / teal-mask deck")
    coverage = {
        "decisions": 0,
        "top_deck": 0,
        "seek_source": 0,
        "vector_cells": 0,
        "seek_teal_dance": 0,
        "seek_crispin": 0,
        "seek_glass_trumpet": 0,
        "seek_energy_switch": 0,
        "seek_area_zero": 0,
        "seek_recovery": 0,
        "seek_other_engine": 0,
    }
    for step in steps:
        observation = step.get("observation")
        if not isinstance(observation, dict):
            raise ValueError("Slop Box combo target is missing masked observation")
        target = build_slop_box_combo_state_target(
            observation, [int(value) for value in (step.get("action") or [])]
        )
        clean = validate_combo_state_labels(target)
        aux = dict(step.get("aux_labels") or {})
        aux[COMBO_STATE_KEY] = target
        step["aux_labels"] = aux
        coverage["decisions"] += 1
        coverage["top_deck"] += int(clean["top_deck_mask"])
        coverage["seek_source"] += int(clean["seek_source_mask"])
        coverage["vector_cells"] += sum(clean["vector_mask"])
        if clean["seek_source_mask"]:
            key = {
                SEEK_TEAL_DANCE: "seek_teal_dance",
                SEEK_CRISPIN: "seek_crispin",
                SEEK_GLASS_TRUMPET: "seek_glass_trumpet",
                SEEK_ENERGY_SWITCH: "seek_energy_switch",
                SEEK_AREA_ZERO: "seek_area_zero",
                SEEK_RECOVERY: "seek_recovery",
                SEEK_OTHER_ENGINE: "seek_other_engine",
            }.get(int(clean["seek_source_target"]))
            if key is not None:
                coverage[key] += 1
    return coverage
