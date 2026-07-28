"""Versioned, target-only supervision for the expanded strategic heads.

The values constructed here are deliberately kept under
``DecisionSample.aux_labels``.  They are never copied into board or option
features.  Trajectory-derived labels are attached while the complete
acting-seat history is available, before any model-context truncation.

``transition_after`` is a short-lived collection field containing an explicit
public post-action snapshot.  It is consumed and removed by
:func:`attach_expanded_strategic_labels`; action utility is masked when that
snapshot is absent rather than being inferred from a later state that may
already include an opponent response.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any, Optional

from . import features
from .strategic_schedule import EXPANDED_HEAD_IDS


EXPANDED_STRATEGIC_KEY = "expanded_strategic"
EXPANDED_STRATEGIC_SCHEMA = "poke_bot.expanded_strategic_targets/v2"
EXPANDED_STRATEGIC_SCHEMA_VERSION = 2
PUBLIC_TRANSITION_SCHEMA = "poke_bot.public_transition_snapshot/v1"

ACTION_UTILITY_NAMES: tuple[str, ...] = (
    "damage_dealt",
    "cards_drawn",
    "energy_delta",
    "open_bench_delta",
    "prize_delta",
    "knockout",
)
TACTICAL_OUTCOME_NAMES: tuple[str, ...] = (
    "own_prize",
    "opponent_prize",
    "own_knockout",
    "opponent_knockout",
    "net_damage",
    "net_prize",
)
OPPONENT_RESPONSE_NAMES: tuple[str, ...] = (
    "attack",
    "prize",
    "knockout",
    "active_change",
    "hand_reduction",
    "board_energy_increase",
    "end_without_attack",
)
RESOURCE_FORECAST_NAMES: tuple[str, ...] = (
    "hand_size",
    "deck_size",
    "attached_energy",
    "open_bench_slots",
    "energy_attachment_available",
    "retreat_available",
)
GAME_PHASE_NAMES: tuple[str, ...] = (
    "setup",
    "stabilize",
    "pressure",
    "prize_race",
    "closeout",
)
OUTCOME_CLASS_NAMES: tuple[str, ...] = ("loss", "draw", "win")
TACTICAL_HORIZONS: tuple[int, ...] = (1, 2, 3)
ACTION_FACTOR_NAMES: tuple[str, ...] = (
    "action_type",
    "target",
    "resource",
)

_SCHEMA_DEFINITION = {
    "schema": EXPANDED_STRATEGIC_SCHEMA,
    "version": EXPANDED_STRATEGIC_SCHEMA_VERSION,
    "action_factors": list(ACTION_FACTOR_NAMES),
    "action_utility": list(ACTION_UTILITY_NAMES),
    "tactical_horizons": list(TACTICAL_HORIZONS),
    "tactical_outcomes": list(TACTICAL_OUTCOME_NAMES),
    "opponent_response": list(OPPONENT_RESPONSE_NAMES),
    "resource_forecast": list(RESOURCE_FORECAST_NAMES),
    "game_phase": list(GAME_PHASE_NAMES),
    "outcome_class": list(OUTCOME_CLASS_NAMES),
    "remaining_turns": "log1p complete public turns to terminal",
    "game_phase_rule": (
        "closeout if prizes<=1 or exact lethal>0; prize_race if both "
        "prizes<=3; stabilize if trailing by>=2; setup if turn<=2; "
        "otherwise pressure"
    ),
    "masking": "false/null means unavailable; present malformed values fail",
    "input_contract": "targets only; never policy observations",
}
EXPANDED_STRATEGIC_SCHEMA_DIGEST = "sha256:" + hashlib.sha256(
    json.dumps(
        _SCHEMA_DEFINITION,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
# Stable public name used by shard, manifest, checkpoint, and receipt code.
# Keep the longer compatibility name above because early V6 consumers already
# import it directly.
TARGET_SCHEMA_DIGEST = EXPANDED_STRATEGIC_SCHEMA_DIGEST

_TARGET_KEYS = frozenset(
    {
        "schema",
        "version",
        "digest",
        "action_factors",
        "action_utility",
        "tactical_outcomes",
        "opponent_response",
        "resource_forecast",
        "game_phase",
        "outcome_class",
        "remaining_turns_log1p",
        "provenance",
    }
)
_PROVENANCE_KEYS = frozenset(
    {
        "trajectory",
        "transition_after",
        "terminal_complete",
        "target_only",
    }
)


class StrategicTargetContractError(ValueError):
    """A present expanded-head target violates its versioned contract."""


def strategic_target_contract() -> dict[str, Any]:
    """Return the immutable schema identity and ordered output names."""

    return {
        **_SCHEMA_DEFINITION,
        "digest": EXPANDED_STRATEGIC_SCHEMA_DIGEST,
    }


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _optional_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    try:
        return result if value == result else None
    except Exception:
        return None


def _optional_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    return None


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _card_serial(value: Any) -> Optional[int]:
    return _optional_int(_field(value, "serial"))


def _card_id(value: Any) -> Optional[int]:
    return _optional_int(_field(value, "id"))


def _zone_list(value: Any) -> Optional[list[Any]]:
    return list(value) if isinstance(value, (list, tuple)) else None


def _zone_serials(value: Any) -> Optional[list[int]]:
    rows = _zone_list(value)
    if rows is None:
        return None
    serials: list[int] = []
    for row in rows:
        serial = _card_serial(row)
        if serial is None:
            return None
        serials.append(serial)
    return serials


def _pokemon_summary(value: Any) -> Optional[dict[str, Any]]:
    if not isinstance(value, Mapping) and not hasattr(value, "serial"):
        return None
    serial = _card_serial(value)
    hp = _optional_int(_field(value, "hp"))
    energy = _zone_list(_field(value, "energyCards"))
    energy_count = len(energy) if energy is not None else None
    energy_serials = _zone_serials(energy)
    return {
        "serial": serial,
        "card_id": _card_id(value),
        "hp": hp,
        "energy_count": energy_count,
        "energy_serials": energy_serials,
    }


def _pokemon_zone(value: Any) -> Optional[list[dict[str, Any]]]:
    rows = _zone_list(value)
    if rows is None:
        return None
    result: list[dict[str, Any]] = []
    for row in rows:
        if row is None:
            continue
        summary = _pokemon_summary(row)
        if summary is None:
            return None
        result.append(summary)
    return result


def _count_from_list_or_field(
    player: Mapping[str, Any],
    *,
    zone: str,
    field_names: Sequence[str],
) -> Optional[int]:
    raw = player.get(zone)
    if isinstance(raw, (list, tuple)):
        return len(raw)
    for name in field_names:
        value = _optional_int(player.get(name))
        if value is not None and value >= 0:
            return value
    return None


def _player_snapshot(
    player: Any,
    *,
    include_hand_serials: bool,
) -> dict[str, Any]:
    if not isinstance(player, Mapping):
        return {
            "valid": False,
            "hand_count": None,
            "hand_serials": None,
            "deck_count": None,
            "prize_count": None,
            "bench_max": None,
            "active": None,
            "bench": None,
            "discard_serials": None,
            "energy_count": None,
            "energy_serials": None,
        }
    active = _pokemon_zone(player.get("active"))
    bench = _pokemon_zone(player.get("bench"))
    hand = player.get("hand")
    hand_count = _optional_int(player.get("handCount"))
    if hand_count is None and isinstance(hand, (list, tuple)):
        hand_count = len(hand)
    deck_count = _optional_int(player.get("deckCount"))
    if deck_count is None and isinstance(player.get("deck"), (list, tuple)):
        deck_count = len(player["deck"])
    prize_count = _count_from_list_or_field(
        player,
        zone="prize",
        field_names=("prizeCount", "prize_count", "remainingPrizes"),
    )
    bench_max = _optional_int(player.get("benchMax"))
    if bench_max is None and bench is not None:
        # Five is the competition default, but it is not an exact label when
        # the engine omits benchMax.  Preserve absence instead of guessing.
        bench_max = None
    pokemon = None if active is None or bench is None else active + bench
    energy_count: Optional[int] = None
    energy_serials: Optional[list[int]] = None
    if pokemon is not None:
        counts = [row.get("energy_count") for row in pokemon]
        if all(_is_int(value) and int(value) >= 0 for value in counts):
            energy_count = sum(int(value) for value in counts)
        serial_groups = [row.get("energy_serials") for row in pokemon]
        if all(isinstance(value, list) for value in serial_groups):
            energy_serials = [
                int(serial)
                for group in serial_groups
                for serial in group
            ]
    return {
        "valid": True,
        "hand_count": hand_count,
        "hand_serials": (
            _zone_serials(hand) if include_hand_serials else None
        ),
        "deck_count": deck_count,
        "prize_count": prize_count,
        "bench_max": bench_max,
        "active": active,
        "bench": bench,
        "discard_serials": _zone_serials(player.get("discard")),
        "energy_count": energy_count,
        "energy_serials": energy_serials,
    }


def public_transition_snapshot(
    observation: Any,
    *,
    actor_seat: int,
) -> dict[str, Any]:
    """Return a bounded post-action snapshot safe for target construction.

    The snapshot contains public counts and public board identities only.
    Acting-seat hand serials are retained only when that seat's hand is
    actually present in the supplied observation.  Card identities from a
    visible opponent hand or any deck order are never copied.
    """

    actor = int(actor_seat)
    if actor not in (0, 1):
        raise StrategicTargetContractError(
            f"actor_seat must be 0 or 1, got {actor_seat!r}"
        )
    current = (
        observation.get("current")
        if isinstance(observation, Mapping)
        else None
    )
    if not isinstance(current, Mapping):
        return {
            "schema": PUBLIC_TRANSITION_SCHEMA,
            "version": 1,
            "actor_seat": actor,
            "next_actor_seat": None,
            "turn": None,
            "result": None,
            "energy_attached": None,
            "retreated": None,
            "players": [],
            "valid": False,
        }
    raw_players = current.get("players")
    players = list(raw_players) if isinstance(raw_players, (list, tuple)) else []
    snapshots = [
        _player_snapshot(
            players[seat] if seat < len(players) else None,
            include_hand_serials=(seat == actor),
        )
        for seat in (0, 1)
    ]
    return {
        "schema": PUBLIC_TRANSITION_SCHEMA,
        "version": 1,
        "actor_seat": actor,
        "next_actor_seat": _optional_int(current.get("yourIndex")),
        "turn": _optional_int(current.get("turn")),
        "result": _optional_int(current.get("result")),
        "energy_attached": _optional_bool(current.get("energyAttached")),
        "retreated": _optional_bool(current.get("retreated")),
        "players": snapshots,
        "valid": len(players) == 2 and all(
            bool(row.get("valid")) for row in snapshots
        ),
    }


def _coerce_transition_snapshot(
    value: Any,
    *,
    actor_seat: int,
) -> Optional[dict[str, Any]]:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise StrategicTargetContractError(
            "transition_after must be an observation or public snapshot mapping"
        )
    if value.get("schema") == PUBLIC_TRANSITION_SCHEMA:
        snapshot = dict(value)
        if int(snapshot.get("version", -1)) != 1:
            raise StrategicTargetContractError(
                "unsupported public transition snapshot version"
            )
        if int(snapshot.get("actor_seat", -1)) != int(actor_seat):
            raise StrategicTargetContractError(
                "transition_after actor does not match the decision actor"
            )
        return snapshot
    nested = value.get("observation")
    if isinstance(nested, Mapping):
        return public_transition_snapshot(nested, actor_seat=actor_seat)
    if isinstance(value.get("current"), Mapping):
        return public_transition_snapshot(value, actor_seat=actor_seat)
    raise StrategicTargetContractError(
        "transition_after has no public post-action state"
    )


def _players(snapshot: Optional[Mapping[str, Any]]) -> Optional[list[Any]]:
    if snapshot is None:
        return None
    rows = snapshot.get("players")
    if not isinstance(rows, list) or len(rows) != 2:
        return None
    return rows


def _player(
    snapshot: Optional[Mapping[str, Any]],
    seat: int,
) -> Optional[Mapping[str, Any]]:
    rows = _players(snapshot)
    if rows is None or not isinstance(rows[seat], Mapping):
        return None
    return rows[seat]


def _board_rows(
    snapshot: Optional[Mapping[str, Any]],
    seat: int,
) -> Optional[list[Mapping[str, Any]]]:
    player = _player(snapshot, seat)
    if player is None:
        return None
    active = player.get("active")
    bench = player.get("bench")
    if not isinstance(active, list) or not isinstance(bench, list):
        return None
    rows = active + bench
    if not all(isinstance(row, Mapping) for row in rows):
        return None
    return rows


def _serial_hp_map(
    snapshot: Optional[Mapping[str, Any]],
    seat: int,
) -> Optional[dict[int, int]]:
    rows = _board_rows(snapshot, seat)
    if rows is None:
        return None
    result: dict[int, int] = {}
    for row in rows:
        serial = _optional_int(row.get("serial"))
        hp = _optional_int(row.get("hp"))
        if serial is None or hp is None or hp < 0 or serial in result:
            return None
        result[serial] = hp
    return result


def _serial_set_field(
    snapshot: Optional[Mapping[str, Any]],
    seat: int,
    field: str,
) -> Optional[set[int]]:
    player = _player(snapshot, seat)
    if player is None:
        return None
    raw = player.get(field)
    if not isinstance(raw, list) or not all(_is_int(value) for value in raw):
        return None
    return {int(value) for value in raw}


def _numeric_player_field(
    snapshot: Optional[Mapping[str, Any]],
    seat: int,
    field: str,
) -> Optional[int]:
    player = _player(snapshot, seat)
    if player is None:
        return None
    value = _optional_int(player.get(field))
    return value if value is not None and value >= 0 else None


def _active_serial(
    snapshot: Optional[Mapping[str, Any]],
    seat: int,
) -> tuple[Optional[int], bool]:
    player = _player(snapshot, seat)
    if player is None or not isinstance(player.get("active"), list):
        return None, False
    active = player["active"]
    if not active:
        return None, True
    if not isinstance(active[0], Mapping):
        return None, False
    serial = _optional_int(active[0].get("serial"))
    return serial, serial is not None


def _open_bench_slots(
    snapshot: Optional[Mapping[str, Any]],
    seat: int,
) -> Optional[int]:
    player = _player(snapshot, seat)
    if player is None:
        return None
    bench = player.get("bench")
    bench_max = _optional_int(player.get("bench_max"))
    if not isinstance(bench, list) or bench_max is None:
        return None
    value = bench_max - len(bench)
    return value if value >= 0 else None


def _prize_delta(
    before: Optional[Mapping[str, Any]],
    after: Optional[Mapping[str, Any]],
    seat: int,
) -> Optional[int]:
    first = _numeric_player_field(before, seat, "prize_count")
    second = _numeric_player_field(after, seat, "prize_count")
    if first is None or second is None:
        return None
    return first - second


def _damage_inflicted(
    before: Optional[Mapping[str, Any]],
    after: Optional[Mapping[str, Any]],
    *,
    victim_seat: int,
) -> Optional[float]:
    pre = _serial_hp_map(before, victim_seat)
    post = _serial_hp_map(after, victim_seat)
    discarded = _serial_set_field(after, victim_seat, "discard_serials")
    if pre is None or post is None or discarded is None:
        return None
    damage = 0.0
    for serial, hp in pre.items():
        if serial in post:
            damage += float(max(0, hp - post[serial]))
        elif serial in discarded:
            # HP observations are capped at zero; overkill is intentionally
            # not fabricated.
            damage += float(hp)
        else:
            # A disappeared Pokémon may have been returned to hand/deck rather
            # than damaged.  Without a public proof, mask the field.
            return None
    return damage


def _knockouts(
    before: Optional[Mapping[str, Any]],
    after: Optional[Mapping[str, Any]],
    *,
    victim_seat: int,
) -> Optional[int]:
    pre = _serial_hp_map(before, victim_seat)
    post = _serial_hp_map(after, victim_seat)
    discarded = _serial_set_field(after, victim_seat, "discard_serials")
    if pre is None or post is None or discarded is None:
        return None
    disappeared = set(pre) - set(post)
    if not disappeared.issubset(discarded):
        return None
    return len(disappeared)


def _masked_vector(length: int) -> dict[str, list[Any]]:
    return {
        "values": [0.0] * int(length),
        "mask": [False] * int(length),
    }


def _set_vector_value(
    target: MutableMapping[str, list[Any]],
    index: int,
    value: Optional[float | int | bool],
) -> None:
    if value is None:
        return
    number = float(value)
    if not math.isfinite(number):
        raise StrategicTargetContractError(
            f"derived target at index {index} is not finite"
        )
    target["values"][index] = number
    target["mask"][index] = True


def _action_utility(
    before: Mapping[str, Any],
    after: Optional[Mapping[str, Any]],
    *,
    actor: int,
) -> dict[str, list[Any]]:
    target = _masked_vector(len(ACTION_UTILITY_NAMES))
    if after is None:
        return target
    opponent = 1 - actor
    _set_vector_value(
        target,
        0,
        _damage_inflicted(before, after, victim_seat=opponent),
    )
    pre_hand = _serial_set_field(before, actor, "hand_serials")
    post_hand = _serial_set_field(after, actor, "hand_serials")
    if pre_hand is not None and post_hand is not None:
        _set_vector_value(target, 1, len(post_hand - pre_hand))
    pre_energy = _serial_set_field(before, actor, "energy_serials")
    post_energy = _serial_set_field(after, actor, "energy_serials")
    if pre_energy is not None and post_energy is not None:
        _set_vector_value(target, 2, len(post_energy) - len(pre_energy))
    pre_slots = _open_bench_slots(before, actor)
    post_slots = _open_bench_slots(after, actor)
    if pre_slots is not None and post_slots is not None:
        _set_vector_value(target, 3, post_slots - pre_slots)
    _set_vector_value(target, 4, _prize_delta(before, after, actor))
    knockout_count = _knockouts(before, after, victim_seat=opponent)
    _set_vector_value(
        target,
        5,
        None if knockout_count is None else knockout_count > 0,
    )
    return target


def _opponent_response(
    after_action: Optional[Mapping[str, Any]],
    next_decision: Optional[Mapping[str, Any]],
    *,
    actor: int,
) -> dict[str, list[Any]]:
    target = _masked_vector(len(OPPONENT_RESPONSE_NAMES))
    if after_action is None or next_decision is None:
        return target
    opponent = 1 - actor
    # Attack/no-attack cannot be proven from a state delta alone; an attack can
    # deal zero damage and produce no other visible effect.  Fields 0 and 6
    # therefore remain masked until the engine exports an exact event flag.
    opponent_prizes = _prize_delta(
        after_action, next_decision, opponent
    )
    _set_vector_value(
        target,
        1,
        None if opponent_prizes is None else opponent_prizes > 0,
    )
    knockout_count = _knockouts(
        after_action, next_decision, victim_seat=actor
    )
    _set_vector_value(
        target,
        2,
        None if knockout_count is None else knockout_count > 0,
    )
    first_active, first_valid = _active_serial(after_action, opponent)
    second_active, second_valid = _active_serial(next_decision, opponent)
    if first_valid and second_valid:
        _set_vector_value(
            target,
            3,
            first_active != second_active,
        )
    first_hand = _numeric_player_field(after_action, actor, "hand_count")
    second_hand = _numeric_player_field(next_decision, actor, "hand_count")
    if first_hand is not None and second_hand is not None:
        _set_vector_value(target, 4, second_hand < first_hand)
    first_energy = _numeric_player_field(
        after_action, opponent, "energy_count"
    )
    second_energy = _numeric_player_field(
        next_decision, opponent, "energy_count"
    )
    if first_energy is not None and second_energy is not None:
        _set_vector_value(target, 5, second_energy > first_energy)
    return target


def _normalize_enum(value: Any) -> Any:
    if isinstance(value, str):
        return "".join(ch for ch in value.casefold() if ch.isalnum())
    result = _optional_int(value)
    return result if result is not None else None


def _option_type(option: Any) -> Any:
    return _normalize_enum(_field(option, "type"))


def _target_signature(option: Any) -> Optional[tuple[Any, ...]]:
    values = (
        _normalize_enum(_field(option, "targetPlayerIndex")),
        _normalize_enum(_field(option, "inPlayArea")),
        _optional_int(_field(option, "inPlayIndex")),
    )
    return values if any(value is not None for value in values) else None


def _resource_signature(option: Any) -> Optional[tuple[Any, ...]]:
    values = (
        _normalize_enum(_field(option, "playerIndex")),
        _normalize_enum(_field(option, "area")),
        _optional_int(_field(option, "index")),
        _optional_int(_field(option, "toolIndex")),
        _optional_int(_field(option, "energyIndex")),
        _optional_int(_field(option, "cardId")),
        _optional_int(_field(option, "attackId")),
        _optional_int(_field(option, "number")),
        _normalize_enum(_field(option, "specialConditionType")),
    )
    return values if any(value is not None for value in values) else None


def _factorized_action_masks(
    observation: Mapping[str, Any],
    action: list[int],
) -> list[dict[str, bool]]:
    try:
        stages = features.factorized_teacher_forcing_stages(
            observation, action
        )
    except Exception as exc:
        raise StrategicTargetContractError(
            f"cannot construct canonical factorized stages: {exc}"
        ) from exc
    select = observation.get("select")
    raw_options = (
        list(select.get("option") or [])
        if isinstance(select, Mapping)
        else []
    )
    prefix: list[int] = []
    result: list[dict[str, bool]] = []
    for candidates, selected_index in stages:
        types: set[Any] = set()
        targets: set[Any] = set()
        resources: set[Any] = set()
        for candidate in candidates:
            candidate = [int(value) for value in candidate]
            if candidate == prefix:
                types.add(("stop",))
                continue
            if len(candidate) != len(prefix) + 1 or candidate[:-1] != prefix:
                raise StrategicTargetContractError(
                    "canonical factorized candidate does not extend its prefix"
                )
            option_index = candidate[-1]
            if not 0 <= option_index < len(raw_options):
                raise StrategicTargetContractError(
                    "factorized candidate references an absent raw option"
                )
            option = raw_options[option_index]
            types.add(("option", _option_type(option)))
            target = _target_signature(option)
            if target is not None:
                targets.add(target)
            resource = _resource_signature(option)
            if resource is not None:
                resources.add(resource)
        result.append(
            {
                "action_type": len(types) >= 2,
                "target": len(targets) >= 2,
                "resource": len(resources) >= 2,
            }
        )
        if not 0 <= int(selected_index) < len(candidates):
            raise StrategicTargetContractError(
                "canonical selected stage index is out of range"
            )
        prefix = [int(value) for value in candidates[int(selected_index)]]
    return result


def _resource_forecast(
    next_step: Optional[Mapping[str, Any]],
    next_snapshot: Optional[Mapping[str, Any]],
    *,
    actor: int,
) -> dict[str, list[Any]]:
    target = _masked_vector(len(RESOURCE_FORECAST_NAMES))
    if next_step is None or next_snapshot is None:
        return target
    _set_vector_value(
        target,
        0,
        _numeric_player_field(next_snapshot, actor, "hand_count"),
    )
    _set_vector_value(
        target,
        1,
        _numeric_player_field(next_snapshot, actor, "deck_count"),
    )
    _set_vector_value(
        target,
        2,
        _numeric_player_field(next_snapshot, actor, "energy_count"),
    )
    _set_vector_value(target, 3, _open_bench_slots(next_snapshot, actor))

    observation = next_step.get("observation")
    select = (
        observation.get("select")
        if isinstance(observation, Mapping)
        else None
    )
    context = _normalize_enum(
        select.get("context") if isinstance(select, Mapping) else None
    )
    # MAIN == 0 in the competition enum; JSON traces may serialize "Main".
    if context in (0, "main") and isinstance(select, Mapping):
        option_types = {
            _option_type(option) for option in (select.get("option") or [])
        }
        _set_vector_value(target, 4, bool(option_types & {8, "attach"}))
        _set_vector_value(target, 5, bool(option_types & {12, "retreat"}))
    return target


def _phase_label(
    snapshot: Mapping[str, Any],
    *,
    actor: int,
    lethal: bool,
) -> Optional[int]:
    own = _numeric_player_field(snapshot, actor, "prize_count")
    opponent = _numeric_player_field(snapshot, 1 - actor, "prize_count")
    turn = _optional_int(snapshot.get("turn"))
    if own is None or opponent is None or turn is None:
        return None
    if own <= 1 or lethal:
        return GAME_PHASE_NAMES.index("closeout")
    if own <= 3 and opponent <= 3:
        return GAME_PHASE_NAMES.index("prize_race")
    if own - opponent >= 2:
        return GAME_PHASE_NAMES.index("stabilize")
    if turn <= 2:
        return GAME_PHASE_NAMES.index("setup")
    return GAME_PHASE_NAMES.index("pressure")


def _outcome_class(
    game_value: float,
    *,
    terminal_complete: bool,
) -> Optional[int]:
    if not terminal_complete:
        return None
    value = float(game_value)
    if not math.isfinite(value) or value not in (-1.0, 0.0, 1.0):
        raise StrategicTargetContractError(
            "terminal game_value must be exactly -1, 0, or 1"
        )
    return 0 if value < 0 else 1 if value == 0 else 2


def _terminal_turn(
    posts: Sequence[Optional[Mapping[str, Any]]],
    *,
    terminal_complete: bool,
) -> Optional[int]:
    if not terminal_complete:
        return None
    terminal: Optional[int] = None
    for post in posts:
        if post is None:
            continue
        result = _optional_int(post.get("result"))
        turn = _optional_int(post.get("turn"))
        if result is not None and result != -1 and turn is not None:
            terminal = turn
    return terminal


def _remaining_turns(
    snapshot: Mapping[str, Any],
    terminal_turn: Optional[int],
) -> Optional[float]:
    current = _optional_int(snapshot.get("turn"))
    if current is None or terminal_turn is None or terminal_turn < current:
        return None
    return math.log1p(terminal_turn - current)


def _tactical_outcomes(
    index: int,
    *,
    pre: Sequence[Mapping[str, Any]],
    posts: Sequence[Optional[Mapping[str, Any]]],
    utilities: Sequence[Mapping[str, list[Any]]],
    actor: int,
) -> dict[str, list[list[Any]]]:
    values = [
        [0.0] * len(TACTICAL_OUTCOME_NAMES)
        for _ in TACTICAL_HORIZONS
    ]
    masks = [
        [False] * len(TACTICAL_OUTCOME_NAMES)
        for _ in TACTICAL_HORIZONS
    ]
    for horizon_row, horizon in enumerate(TACTICAL_HORIZONS):
        end = index + horizon
        if end >= len(pre):
            continue
        own_prizes = _prize_delta(pre[index], pre[end], actor)
        opponent_prizes = _prize_delta(pre[index], pre[end], 1 - actor)
        if own_prizes is not None:
            values[horizon_row][0] = float(own_prizes > 0)
            masks[horizon_row][0] = True
        if opponent_prizes is not None:
            values[horizon_row][1] = float(opponent_prizes > 0)
            masks[horizon_row][1] = True
        if own_prizes is not None and opponent_prizes is not None:
            values[horizon_row][5] = float(own_prizes - opponent_prizes)
            masks[horizon_row][5] = True

        window = range(index, end)
        ko_values: list[float] = []
        damage_values: list[float] = []
        opponent_damage_values: list[float] = []
        opponent_ko_values: list[float] = []
        complete = True
        for row in window:
            utility = utilities[row]
            if not bool(utility["mask"][5]):
                complete = False
                break
            ko_values.append(float(utility["values"][5]))
        if complete:
            values[horizon_row][2] = float(sum(ko_values) > 0.0)
            masks[horizon_row][2] = True

        complete = True
        for row in window:
            post = posts[row]
            next_pre = pre[row + 1]
            opponent_ko = _knockouts(
                post,
                next_pre,
                victim_seat=actor,
            )
            if opponent_ko is None:
                complete = False
                break
            opponent_ko_values.append(float(opponent_ko))
        if complete:
            values[horizon_row][3] = float(
                sum(opponent_ko_values) > 0.0
            )
            masks[horizon_row][3] = True

        complete = True
        for row in window:
            utility = utilities[row]
            if not bool(utility["mask"][0]):
                complete = False
                break
            damage_values.append(float(utility["values"][0]))
            response_damage = _damage_inflicted(
                posts[row],
                pre[row + 1],
                victim_seat=actor,
            )
            if response_damage is None:
                complete = False
                break
            opponent_damage_values.append(float(response_damage))
        if complete:
            values[horizon_row][4] = float(
                sum(damage_values) - sum(opponent_damage_values)
            )
            masks[horizon_row][4] = True
    return {"values": values, "mask": masks}


def _exact_lethal_positive(step: Mapping[str, Any]) -> bool:
    aux = step.get("aux_labels")
    if not isinstance(aux, Mapping):
        return False
    raw = aux.get("lethal_threat")
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        value = float(raw)
        return math.isfinite(value) and value > 0.0
    return False


def _validate_masked_vector(
    value: Any,
    *,
    name: str,
    length: int,
) -> None:
    if not isinstance(value, Mapping) or set(value) != {"values", "mask"}:
        raise StrategicTargetContractError(
            f"{name} must contain exactly values and mask"
        )
    values = value.get("values")
    masks = value.get("mask")
    if not isinstance(values, list) or len(values) != length:
        raise StrategicTargetContractError(
            f"{name}.values must contain {length} entries"
        )
    if not isinstance(masks, list) or len(masks) != length:
        raise StrategicTargetContractError(
            f"{name}.mask must contain {length} entries"
        )
    for index, (raw, mask) in enumerate(zip(values, masks)):
        if not isinstance(mask, bool):
            raise StrategicTargetContractError(
                f"{name}.mask[{index}] must be bool"
            )
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise StrategicTargetContractError(
                f"{name}.values[{index}] must be numeric"
            )
        number = float(raw)
        if not math.isfinite(number):
            raise StrategicTargetContractError(
                f"{name}.values[{index}] must be finite"
            )
        if not mask and number != 0.0:
            raise StrategicTargetContractError(
                f"{name}.values[{index}] must be zero when masked"
            )


def validate_expanded_strategic_labels(value: Any) -> dict[str, Any]:
    """Strictly validate one ``expanded_strategic`` decision target."""

    if not isinstance(value, Mapping):
        raise StrategicTargetContractError(
            "expanded strategic target must be a mapping"
        )
    if set(value) != _TARGET_KEYS:
        missing = sorted(_TARGET_KEYS - set(value))
        unknown = sorted(set(value) - _TARGET_KEYS)
        raise StrategicTargetContractError(
            f"expanded strategic target keys mismatch: "
            f"missing={missing} unknown={unknown}"
        )
    if value.get("schema") != EXPANDED_STRATEGIC_SCHEMA:
        raise StrategicTargetContractError("expanded strategic schema mismatch")
    if value.get("version") != EXPANDED_STRATEGIC_SCHEMA_VERSION:
        raise StrategicTargetContractError("expanded strategic version mismatch")
    if value.get("digest") != EXPANDED_STRATEGIC_SCHEMA_DIGEST:
        raise StrategicTargetContractError("expanded strategic digest mismatch")
    factors = value.get("action_factors")
    if not isinstance(factors, list):
        raise StrategicTargetContractError("action_factors must be a list")
    for row_index, row in enumerate(factors):
        if not isinstance(row, Mapping) or set(row) != set(ACTION_FACTOR_NAMES):
            raise StrategicTargetContractError(
                f"action_factors[{row_index}] keys mismatch"
            )
        if not all(isinstance(row[name], bool) for name in ACTION_FACTOR_NAMES):
            raise StrategicTargetContractError(
                f"action_factors[{row_index}] masks must be bool"
            )
    _validate_masked_vector(
        value.get("action_utility"),
        name="action_utility",
        length=len(ACTION_UTILITY_NAMES),
    )
    tactical = value.get("tactical_outcomes")
    if not isinstance(tactical, Mapping) or set(tactical) != {"values", "mask"}:
        raise StrategicTargetContractError(
            "tactical_outcomes must contain exactly values and mask"
        )
    tactical_values = tactical.get("values")
    tactical_masks = tactical.get("mask")
    if (
        not isinstance(tactical_values, list)
        or not isinstance(tactical_masks, list)
        or len(tactical_values) != len(TACTICAL_HORIZONS)
        or len(tactical_masks) != len(TACTICAL_HORIZONS)
    ):
        raise StrategicTargetContractError(
            "tactical_outcomes must contain exactly three horizons"
        )
    for index, (values, masks) in enumerate(
        zip(tactical_values, tactical_masks)
    ):
        _validate_masked_vector(
            {"values": values, "mask": masks},
            name=f"tactical_outcomes[{index}]",
            length=len(TACTICAL_OUTCOME_NAMES),
        )
    _validate_masked_vector(
        value.get("opponent_response"),
        name="opponent_response",
        length=len(OPPONENT_RESPONSE_NAMES),
    )
    _validate_masked_vector(
        value.get("resource_forecast"),
        name="resource_forecast",
        length=len(RESOURCE_FORECAST_NAMES),
    )
    phase = value.get("game_phase")
    if phase is not None and (
        not _is_int(phase) or not 0 <= int(phase) < len(GAME_PHASE_NAMES)
    ):
        raise StrategicTargetContractError("game_phase is outside its class set")
    outcome = value.get("outcome_class")
    if outcome is not None and (
        not _is_int(outcome)
        or not 0 <= int(outcome) < len(OUTCOME_CLASS_NAMES)
    ):
        raise StrategicTargetContractError(
            "outcome_class is outside its class set"
        )
    remaining = value.get("remaining_turns_log1p")
    if remaining is not None and (
        isinstance(remaining, bool)
        or not isinstance(remaining, (int, float))
        or not math.isfinite(float(remaining))
        or float(remaining) < 0.0
    ):
        raise StrategicTargetContractError(
            "remaining_turns_log1p must be finite and nonnegative"
        )
    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != _PROVENANCE_KEYS:
        raise StrategicTargetContractError(
            "expanded strategic provenance keys mismatch"
        )
    if provenance.get("trajectory") != "full_untruncated_same_seat":
        raise StrategicTargetContractError(
            "expanded strategic trajectory provenance mismatch"
        )
    if provenance.get("transition_after") not in (
        "explicit_public_post_action",
        "absent_masked",
    ):
        raise StrategicTargetContractError(
            "expanded strategic transition provenance mismatch"
        )
    if not isinstance(provenance.get("terminal_complete"), bool):
        raise StrategicTargetContractError(
            "terminal_complete provenance must be bool"
        )
    if provenance.get("target_only") is not True:
        raise StrategicTargetContractError(
            "expanded strategic labels must be target-only"
        )
    return dict(value)


def expanded_strategic_decision_head_mask(
    value: Any,
) -> dict[str, bool]:
    """Return per-head label availability for one decision target.

    ``None`` means that the decision has no V6 target and therefore masks all
    heads.  A present target is strictly validated; malformed data is never
    converted into an all-masked row.  Coverage is deliberately decision
    based, including for factorized action heads: a factor head is labeled for
    the decision when at least one canonical policy stage carries that factor.
    """

    if value is None:
        return {head_id: False for head_id in EXPANDED_HEAD_IDS}
    target = validate_expanded_strategic_labels(value)
    factors = target["action_factors"]
    result = {
        "action_q": target["outcome_class"] is not None,
        "action_type": any(row["action_type"] for row in factors),
        "action_target": any(row["target"] for row in factors),
        "action_resource": any(row["resource"] for row in factors),
        "action_utility": any(target["action_utility"]["mask"]),
        "tactical_outcome": any(
            any(row) for row in target["tactical_outcomes"]["mask"]
        ),
        "opponent_response": any(target["opponent_response"]["mask"]),
        "resource_forecast": any(target["resource_forecast"]["mask"]),
        "game_phase": target["game_phase"] is not None,
        "outcome_distribution": target["outcome_class"] is not None,
        "remaining_turns": target["remaining_turns_log1p"] is not None,
    }
    if set(result) != set(EXPANDED_HEAD_IDS):
        raise StrategicTargetContractError(
            "expanded strategic coverage head inventory mismatch"
        )
    return {head_id: bool(result[head_id]) for head_id in EXPANDED_HEAD_IDS}


def _expanded_target_from_decision(value: Any) -> Any:
    """Extract the target from a target, aux mapping, or decision-like row."""

    if value is None:
        return None
    if isinstance(value, Mapping):
        if (
            value.get("schema") == EXPANDED_STRATEGIC_SCHEMA
            or set(value) == _TARGET_KEYS
        ):
            return value
        if EXPANDED_STRATEGIC_KEY in value:
            return value.get(EXPANDED_STRATEGIC_KEY)
        aux = value.get("aux_labels")
    else:
        aux = getattr(value, "aux_labels", None)
    if aux is None:
        return None
    if not isinstance(aux, Mapping):
        raise StrategicTargetContractError(
            "decision aux_labels must be a mapping"
        )
    return aux.get(EXPANDED_STRATEGIC_KEY)


def expanded_strategic_sequence_coverage(
    decisions: Sequence[Any],
) -> dict[str, Any]:
    """Return deterministic per-head labeled/masked decision-row counts."""

    totals = {
        head_id: {
            "labeled_rows": 0,
            "masked_rows": 0,
            "total_rows": 0,
        }
        for head_id in EXPANDED_HEAD_IDS
    }
    decision_count = 0
    for decision in decisions:
        decision_count += 1
        target = _expanded_target_from_decision(decision)
        labeled = expanded_strategic_decision_head_mask(target)
        for head_id in EXPANDED_HEAD_IDS:
            row = totals[head_id]
            row["total_rows"] += 1
            if labeled[head_id]:
                row["labeled_rows"] += 1
            else:
                row["masked_rows"] += 1
    return {
        "schema": EXPANDED_STRATEGIC_SCHEMA,
        "digest": TARGET_SCHEMA_DIGEST,
        "decisions": decision_count,
        "head_coverage": totals,
    }


def masked_expanded_strategic_coverage(decisions: int) -> dict[str, Any]:
    """Represent target absence as masked decision rows, never zero labels."""

    if not _is_int(decisions) or int(decisions) < 0:
        raise StrategicTargetContractError(
            "expanded strategic coverage decisions must be nonnegative"
        )
    count = int(decisions)
    return {
        "schema": EXPANDED_STRATEGIC_SCHEMA,
        "digest": TARGET_SCHEMA_DIGEST,
        "decisions": count,
        "head_coverage": {
            head_id: {
                "labeled_rows": 0,
                "masked_rows": count,
                "total_rows": count,
            }
            for head_id in EXPANDED_HEAD_IDS
        },
    }


def merge_expanded_strategic_coverages(
    coverages: Sequence[Any],
) -> dict[str, Any]:
    """Validate and sum exact expanded-target coverage blocks."""

    result = masked_expanded_strategic_coverage(0)
    for index, raw in enumerate(coverages):
        if not isinstance(raw, Mapping):
            raise StrategicTargetContractError(
                f"expanded strategic coverage {index} must be a mapping"
            )
        if raw.get("schema") != EXPANDED_STRATEGIC_SCHEMA:
            raise StrategicTargetContractError(
                f"expanded strategic coverage {index} schema mismatch"
            )
        if raw.get("digest") != TARGET_SCHEMA_DIGEST:
            raise StrategicTargetContractError(
                f"expanded strategic coverage {index} digest mismatch"
            )
        decisions = raw.get("decisions")
        if not _is_int(decisions) or int(decisions) < 0:
            raise StrategicTargetContractError(
                f"expanded strategic coverage {index} decisions are invalid"
            )
        rows = raw.get("head_coverage")
        if not isinstance(rows, Mapping) or set(rows) != set(
            EXPANDED_HEAD_IDS
        ):
            raise StrategicTargetContractError(
                f"expanded strategic coverage {index} head inventory mismatch"
            )
        for head_id in EXPANDED_HEAD_IDS:
            row = rows[head_id]
            if not isinstance(row, Mapping) or set(row) != {
                "labeled_rows",
                "masked_rows",
                "total_rows",
            }:
                raise StrategicTargetContractError(
                    f"expanded strategic coverage {index}/{head_id} keys mismatch"
                )
            labeled = row.get("labeled_rows")
            masked = row.get("masked_rows")
            total = row.get("total_rows")
            if not all(_is_int(value) for value in (labeled, masked, total)):
                raise StrategicTargetContractError(
                    f"expanded strategic coverage {index}/{head_id} "
                    "counts must be integers"
                )
            if (
                int(labeled) < 0
                or int(masked) < 0
                or int(total) != int(decisions)
                or int(labeled) + int(masked) != int(total)
            ):
                raise StrategicTargetContractError(
                    f"expanded strategic coverage {index}/{head_id} "
                    "counts are inconsistent"
                )
            destination = result["head_coverage"][head_id]
            destination["labeled_rows"] += int(labeled)
            destination["masked_rows"] += int(masked)
            destination["total_rows"] += int(total)
        result["decisions"] += int(decisions)
    return result


def _coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    result = {
        "decisions": len(rows),
        "action_factor_stages": 0,
        "action_utility_rows": 0,
        "tactical_horizon_rows": 0,
        "opponent_response_rows": 0,
        "resource_forecast_rows": 0,
        "game_phase_rows": 0,
        "outcome_class_rows": 0,
        "remaining_turns_rows": 0,
    }
    for row in rows:
        result["action_factor_stages"] += len(row["action_factors"])
        result["action_utility_rows"] += int(
            any(row["action_utility"]["mask"])
        )
        result["tactical_horizon_rows"] += sum(
            int(any(mask)) for mask in row["tactical_outcomes"]["mask"]
        )
        result["opponent_response_rows"] += int(
            any(row["opponent_response"]["mask"])
        )
        result["resource_forecast_rows"] += int(
            any(row["resource_forecast"]["mask"])
        )
        result["game_phase_rows"] += int(row["game_phase"] is not None)
        result["outcome_class_rows"] += int(row["outcome_class"] is not None)
        result["remaining_turns_rows"] += int(
            row["remaining_turns_log1p"] is not None
        )
    return result


def _target_materialization_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result = expanded_strategic_sequence_coverage(rows)
    result["version"] = EXPANDED_STRATEGIC_SCHEMA_VERSION
    result["coverage"] = _coverage(rows)
    return result


def attach_expanded_strategic_labels(
    steps: list[dict[str, Any]],
    *,
    game_value: float,
    terminal_complete: bool = True,
) -> dict[str, Any]:
    """Attach exact/masked expanded targets to a full acting-seat trajectory.

    The function is idempotent for already-materialized records.  A partially
    materialized trajectory fails closed.  Every temporary ``transition_after``
    field is removed before the function returns, including on the idempotent
    path.
    """

    if not isinstance(steps, list):
        raise StrategicTargetContractError("steps must be a list")
    if not steps:
        return _target_materialization_summary([])
    existing: list[Optional[dict[str, Any]]] = []
    for index, step in enumerate(steps):
        if not isinstance(step, MutableMapping):
            raise StrategicTargetContractError(
                f"step {index} must be a mutable mapping"
            )
        aux = step.get("aux_labels")
        if aux is None:
            aux = {}
            step["aux_labels"] = aux
        if not isinstance(aux, MutableMapping):
            raise StrategicTargetContractError(
                f"step {index} aux_labels must be a mapping"
            )
        present = aux.get(EXPANDED_STRATEGIC_KEY)
        existing.append(
            validate_expanded_strategic_labels(present)
            if present is not None
            else None
        )
    if any(row is not None for row in existing):
        if not all(row is not None for row in existing):
            raise StrategicTargetContractError(
                "trajectory has partially materialized expanded targets"
            )
        for step in steps:
            step.pop("transition_after", None)
        materialized = [row for row in existing if row is not None]
        return _target_materialization_summary(materialized)

    actor: Optional[int] = None
    pre: list[dict[str, Any]] = []
    posts: list[Optional[dict[str, Any]]] = []
    for index, step in enumerate(steps):
        observation = step.get("observation")
        if not isinstance(observation, Mapping):
            raise StrategicTargetContractError(
                f"step {index} has no observation mapping"
            )
        current = observation.get("current")
        if not isinstance(current, Mapping):
            raise StrategicTargetContractError(
                f"step {index} has no public current state"
            )
        row_actor = _optional_int(current.get("yourIndex"))
        if row_actor not in (0, 1):
            raise StrategicTargetContractError(
                f"step {index} has invalid acting seat"
            )
        if actor is None:
            actor = row_actor
        elif row_actor != actor:
            raise StrategicTargetContractError(
                "expanded targets require one same-seat trajectory"
            )
        pre.append(public_transition_snapshot(observation, actor_seat=actor))
        transition = step.pop("transition_after", None)
        posts.append(
            _coerce_transition_snapshot(transition, actor_seat=actor)
        )
    assert actor is not None

    utilities = [
        _action_utility(pre[index], posts[index], actor=actor)
        for index in range(len(steps))
    ]
    responses = [
        _opponent_response(
            posts[index],
            pre[index + 1] if index + 1 < len(pre) else None,
            actor=actor,
        )
        for index in range(len(steps))
    ]
    terminal_turn = _terminal_turn(
        posts,
        terminal_complete=bool(terminal_complete),
    )
    outcome = _outcome_class(
        game_value,
        terminal_complete=bool(terminal_complete),
    )
    materialized: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        observation = step["observation"]
        action_raw = step.get("action")
        if not isinstance(action_raw, (list, tuple)):
            raise StrategicTargetContractError(
                f"step {index} action must be a list"
            )
        action = []
        for value in action_raw:
            parsed = _optional_int(value)
            if parsed is None:
                raise StrategicTargetContractError(
                    f"step {index} action contains a non-integer"
                )
            action.append(parsed)
        tactical = _tactical_outcomes(
            index,
            pre=pre,
            posts=posts,
            utilities=utilities,
            actor=actor,
        )
        target = {
            "schema": EXPANDED_STRATEGIC_SCHEMA,
            "version": EXPANDED_STRATEGIC_SCHEMA_VERSION,
            "digest": EXPANDED_STRATEGIC_SCHEMA_DIGEST,
            "action_factors": _factorized_action_masks(
                observation,
                action,
            ),
            "action_utility": utilities[index],
            "tactical_outcomes": tactical,
            "opponent_response": responses[index],
            "resource_forecast": _resource_forecast(
                steps[index + 1] if index + 1 < len(steps) else None,
                pre[index + 1] if index + 1 < len(pre) else None,
                actor=actor,
            ),
            "game_phase": _phase_label(
                pre[index],
                actor=actor,
                lethal=_exact_lethal_positive(step),
            ),
            "outcome_class": outcome,
            "remaining_turns_log1p": _remaining_turns(
                pre[index],
                terminal_turn,
            ),
            "provenance": {
                "trajectory": "full_untruncated_same_seat",
                "transition_after": (
                    "explicit_public_post_action"
                    if posts[index] is not None
                    else "absent_masked"
                ),
                "terminal_complete": bool(terminal_complete),
                "target_only": True,
            },
        }
        validate_expanded_strategic_labels(target)
        step["aux_labels"][EXPANDED_STRATEGIC_KEY] = target
        materialized.append(target)
    return _target_materialization_summary(materialized)
