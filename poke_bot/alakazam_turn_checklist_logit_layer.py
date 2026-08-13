"""Bounded, causal Alakazam turn-checklist residual (r288).

This is deliberately *not* a second policy and it owns no checkpoint
parameters.  It reads only the acting player's current public observation,
the complete legal candidate stage, and (when supplied) the causal own-deck
ledger.  It returns auditable, per-option signed evidence for the eight owner
questions and adds at most ``+/- 0.10`` to post-neural logits.

The module is safe to call from both local and remote-policy paths.  Runtime
arming remains the PolicyAgent's explicit default-off responsibility; this
module merely evaluates a legal stage deterministically once called.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional

from . import alakazam_heuristics as _legacy
from . import alakazam_new_list_heuristics as _guide


CHANNEL_NAMES: tuple[str, ...] = (
    "ko_hand_threshold",
    "safe_spend_above_threshold",
    "replacement_alakazam_line",
    "unavoidable_draws_before_attack",
    "bench_prize_exposure",
    "immediate_disruption_outcome",
    "unknown_prize_robust_line",
    "terminal_before_forced_draw",
)

GUIDE_CHANNEL_NAME = "guide_support"
CONFIG_SCHEMA = "poke_bot.alakazam_turn_checklist_heuristic_logit_layer_config/v1"
_LEGACY_CONFIG_SCHEMA = "poke_bot.alakazam_turn_checklist_heuristic_logit_layer/v1"
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "policy_layers"
    / "alakazam-turn-checklist-r288.json"
)

_GATE_NAMES = (
    "ko_hand_threshold_gate",
    "safe_spend_above_threshold_gate",
    "replacement_alakazam_line_gate",
    "unavoidable_draws_before_attack_gate",
    "bench_prize_exposure_gate",
    "immediate_disruption_outcome_gate",
    "unknown_prize_robust_line_gate",
    "terminal_before_forced_draw_gate",
    "separate_guide_gate",
)
_CHANNEL_GATE = dict(
    zip(CHANNEL_NAMES, _GATE_NAMES[: len(CHANNEL_NAMES)], strict=True)
)
# The historical guide predates the audited card-text corrections in this
# layer.  It remains a named/calibratable channel for offline comparison, but
# it is deliberately inert unless a future, explicitly audited configuration
# says otherwise.  In particular, a staged config must not accidentally make
# legacy guide scores live again.
_DEFAULT_GATES = MappingProxyType(
    {
        **{
            name: 0.01
            for name in _GATE_NAMES
            if name
            not in {
                "bench_prize_exposure_gate",
                "immediate_disruption_outcome_gate",
                "separate_guide_gate",
            }
        },
        "bench_prize_exposure_gate": 0.0,
        "immediate_disruption_outcome_gate": 0.0,
        "separate_guide_gate": 0.0,
    }
)
_EXPECTED_GATE_CHANNEL_MAP = MappingProxyType(
    {
        **{gate: channel for channel, gate in _CHANNEL_GATE.items()},
        "separate_guide_gate": GUIDE_CHANNEL_NAME,
    }
)

# r293 keeps these two broad board/disruption questions observable in traces,
# but deliberately outside the residual until their predicates are separated
# from already-learned routes.  The historic guide is likewise trace-only.
_TRACE_ONLY_GATES = frozenset(
    {
        "bench_prize_exposure_gate",
        "immediate_disruption_outcome_gate",
        "separate_guide_gate",
    }
)
_OVERLAP_GROUPS = MappingProxyType(
    {
        "closure": (
            "ko_hand_threshold",
            "immediate_disruption_outcome",
            "terminal_before_forced_draw",
        ),
        "continuity": (
            "safe_spend_above_threshold",
            "replacement_alakazam_line",
            "unavoidable_draws_before_attack",
            "unknown_prize_robust_line",
        ),
        "board": ("bench_prize_exposure",),
    }
)
_RESIDUAL_GROUP_ORDER: tuple[str, ...] = ("closure", "continuity")
_CHANNEL_OVERLAP_REASON = MappingProxyType(
    {
        "ko_hand_threshold": "closure overlap with the neural policy, learned closure/prize routes, fusion, and matchup adapters; only exact visible hand-to-KO evidence is emitted.",
        "safe_spend_above_threshold": "continuity overlap with learned hand-management and own-deck effects; this layer contributes only post-cost public threshold evidence.",
        "replacement_alakazam_line": "continuity overlap with learned board-development routes; this layer counts only a visible Bench-only ready or legally completing line.",
        "unavoidable_draws_before_attack": "continuity overlap with learned draw/deck-management routes; this layer counts only exact current forced draws and an exact end-turn draw.",
        "bench_prize_exposure": "board overlap with learned prize-race/board policy; r293 fixes this residual gate at zero while retaining the public trace.",
        "immediate_disruption_outcome": "closure overlap with learned disruption and matchup routes; r293 fixes the aggregate residual gate at zero pending predicate-level separation.",
        "unknown_prize_robust_line": "continuity overlap with own-deck ledger effects; this layer uses only an exact deck-bound ledger and direct visible alternate completion routes.",
        "terminal_before_forced_draw": "closure overlap with learned outcome/prize routes; this layer emits only a visible terminal KO before a forced future draw.",
    }
)


class TurnChecklistConfigError(ValueError):
    """A supplied r288 configuration cannot safely define a residual."""


@dataclass(frozen=True)
class TurnChecklistConfig:
    """Frozen numeric controls for the parameter-free residual."""

    schema: str
    total_residual_cap: float
    scalar_gates: tuple[tuple[str, float], ...]
    source: str
    staged_active: bool
    guide_support_runtime_authorized: bool = False

    @property
    def gates(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self.scalar_gates))


@dataclass(frozen=True)
class ChecklistChannelTrace:
    """One exact, aligned channel vector and its audit explanation."""

    name: str
    raw: tuple[float, ...]
    normalized: tuple[float, ...]
    # A channel may have an exact causal answer for only part of a
    # factorised stage.  Values outside this mask are literal neutral zeros;
    # they are not included in centering, so an unresolved prefix never
    # acquires a preference merely because other candidates were answerable.
    option_availability: tuple[bool, ...]
    available: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "raw": [float(value) for value in self.raw],
            "normalized": [float(value) for value in self.normalized],
            "option_availability": [bool(value) for value in self.option_availability],
            "available": bool(self.available),
            "status": "available" if self.available else "unavailable",
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TurnChecklistTrace:
    """Immutable result emitted for a single legal candidate stage."""

    channel_names: tuple[str, ...]
    channels: tuple[ChecklistChannelTrace, ...]
    guide_support: ChecklistChannelTrace
    scalar_gates: tuple[tuple[str, float], ...]
    residuals: tuple[float, ...]
    facts: Mapping[str, Any]
    available: bool
    reason: str
    active: bool

    def to_dict(self) -> dict[str, Any]:
        overlap_audit = self.facts.get("channel_overlap_audit", {})
        channel_payloads: list[dict[str, Any]] = []
        for channel in self.channels:
            payload = channel.to_dict()
            audit = overlap_audit.get(channel.name, {})
            if isinstance(audit, Mapping):
                # r298 Phase 4 consumes these at the per-channel surface,
                # while r293/r289 retain the complete top-level audit below.
                # They are always aligned to the legal candidate order.
                payload["applied_gate"] = float(
                    _finite_float(audit.get("applied_gate"), 0.0)
                )
                payload["post_deduplication_signed_residual"] = _json_safe(
                    audit.get("post_deduplication_signed_residual", ())
                )
                payload["post_cap_residual"] = _json_safe(
                    audit.get("post_total_cap_signed_residual", ())
                )
                payload["group_winner"] = _json_safe(
                    audit.get("group_winner", ())
                )
            channel_payloads.append(payload)
        return {
            "channel_names": list(self.channel_names),
            "channels": channel_payloads,
            "guide_support": self.guide_support.to_dict(),
            "scalar_gates": {
                name: float(value) for name, value in self.scalar_gates
            },
            "residuals": [float(value) for value in self.residuals],
            "facts": _json_safe(self.facts),
            "available": bool(self.available),
            "reason": self.reason,
            "active": bool(self.active),
            # ``active`` above means this explicitly invoked calculation had
            # usable causal evidence.  It deliberately does not arm a staged
            # runtime config; callers own that default-off decision.
            "config_staged_active": bool(
                self.facts.get("config_staged_active", False)
            ),
            "explicit_invocation": True,
            "normalized_channel_vectors": {
                channel.name: [float(value) for value in channel.normalized]
                for channel in self.channels
            },
            "normalized_guide_support_vector": [
                float(value) for value in self.guide_support.normalized
            ],
            "channel_status": {
                channel.name: {
                    "available": bool(channel.available),
                    "status": "available" if channel.available else "unavailable",
                    "reason": channel.reason,
                }
                for channel in (*self.channels, self.guide_support)
            },
            "channel_option_availability": {
                channel.name: [bool(value) for value in channel.option_availability]
                for channel in (*self.channels, self.guide_support)
            },
            # r293/r295 live diagnostics: these are the module's
            # pre-whole-decision-budget values.  PolicyAgent may later clip a
            # selected path's aggregate, and records that separately.
            "channel_overlap_audit": _json_safe(overlap_audit),
            "guide_support_trace_only": True,
            "guide_support_runtime_residual": 0.0,
        }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _finite_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def _default_config(*, source: str = "built_in_defaults") -> TurnChecklistConfig:
    return TurnChecklistConfig(
        schema=CONFIG_SCHEMA,
        total_residual_cap=0.10,
        scalar_gates=tuple((name, float(_DEFAULT_GATES[name])) for name in _GATE_NAMES),
        source=source,
        staged_active=False,
        guide_support_runtime_authorized=False,
    )


def _read_config_mapping(config_path: Optional[str]) -> Mapping[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_file():
        if config_path:
            raise TurnChecklistConfigError("turn-checklist config path is not a file")
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TurnChecklistConfigError("cannot parse turn-checklist config") from exc
    if not isinstance(raw, Mapping):
        raise TurnChecklistConfigError("turn-checklist config must be an object")
    return raw


def _parse_config(value: Mapping[str, Any], *, source: str) -> TurnChecklistConfig:
    if not value:
        return _default_config(source=source)
    schema = value.get("schema")
    if schema not in {CONFIG_SCHEMA, _LEGACY_CONFIG_SCHEMA}:
        raise TurnChecklistConfigError("unexpected turn-checklist config schema")
    runtime = value.get("runtime")
    if runtime is None:
        runtime = value
    if not isinstance(runtime, Mapping):
        raise TurnChecklistConfigError("turn-checklist runtime configuration is invalid")
    # These named identities prevent an incidental config edit from silently
    # rerouting calibrated evidence to a different question.  Older compact
    # configs may omit them; when present they must be exact.
    channel_order = runtime.get("channel_order")
    if channel_order is not None:
        if (
            not isinstance(channel_order, Sequence)
            or isinstance(channel_order, (str, bytes))
            or tuple(channel_order) != CHANNEL_NAMES
        ):
            raise TurnChecklistConfigError("turn-checklist channel order changed")
    guide_channel = runtime.get("guide_support_channel")
    if guide_channel is not None and guide_channel != GUIDE_CHANNEL_NAME:
        raise TurnChecklistConfigError("turn-checklist guide channel changed")
    gate_order = runtime.get("gate_order")
    if gate_order is not None:
        if (
            not isinstance(gate_order, Sequence)
            or isinstance(gate_order, (str, bytes))
            or tuple(gate_order) != _GATE_NAMES
        ):
            raise TurnChecklistConfigError("turn-checklist gate order changed")
    gate_channel_map = runtime.get("gate_channel_map")
    if gate_channel_map is not None:
        if not isinstance(gate_channel_map, Mapping) or dict(gate_channel_map) != dict(_EXPECTED_GATE_CHANNEL_MAP):
            raise TurnChecklistConfigError("turn-checklist gate map changed")
    formula = runtime.get("residual_formula")
    if formula is None:
        formula = runtime
    if not isinstance(formula, Mapping):
        raise TurnChecklistConfigError("turn-checklist residual formula is invalid")
    cap = _finite_float(
        runtime.get("total_residual_cap", formula.get("clip_max", 0.10)),
        float("nan"),
    )
    if not 0.0 < cap <= 0.10:
        raise TurnChecklistConfigError("turn-checklist cap must be in (0, 0.10]")
    clip_min = formula.get("clip_min")
    if clip_min is not None and _finite_float(clip_min, float("nan")) != -cap:
        raise TurnChecklistConfigError("turn-checklist clip minimum disagrees with cap")
    clip_max = formula.get("clip_max")
    if clip_max is not None and _finite_float(clip_max, float("nan")) != cap:
        raise TurnChecklistConfigError("turn-checklist clip maximum disagrees with cap")
    raw_gates = runtime.get("scalar_gates", value.get("scalar_gates", {}))
    if raw_gates is None:
        raw_gates = {}
    if not isinstance(raw_gates, Mapping):
        raise TurnChecklistConfigError("turn-checklist scalar gates must be an object")
    # A future audit may authorize a *corrected* guide explicitly, but a
    # regular r288/r295 config is never such an audit.  Keep this separate
    # from the caller's runtime arm: both are required before historic scores
    # can have any numerical effect.
    guide_authorized = bool(runtime.get("guide_support_runtime_residual_authorized", False))
    guide_audit = runtime.get("guide_support_audit_receipt_sha256")
    if guide_authorized and (
        not isinstance(guide_audit, str) or not guide_audit.startswith("sha256:")
    ):
        raise TurnChecklistConfigError("guide support authorization lacks audit receipt")
    gates: list[tuple[str, float]] = []
    for name in _GATE_NAMES:
        gate = _finite_float(raw_gates.get(name, _DEFAULT_GATES[name]), float("nan"))
        # Calibration is constrained to non-negative tiny gates.  A malformed
        # gate never gets to flip a causal channel or enlarge the cap.
        if not 0.0 <= gate <= 0.10:
            raise TurnChecklistConfigError(f"invalid turn-checklist gate {name}")
        if name in _TRACE_ONLY_GATES:
            gate = 0.0
        gates.append((name, gate))
    return TurnChecklistConfig(
        schema=str(schema),
        total_residual_cap=cap,
        scalar_gates=tuple(gates),
        source=source,
        staged_active=bool(value.get("active", False)),
        # The broad historical scorer is intentionally unavailable in this
        # contract even if a caller supplies a stray gate.  A future module
        # can introduce a separately validated corrected-guide artifact.
        guide_support_runtime_authorized=False,
    )


def load_turn_checklist_config(
    *,
    config_path: Optional[str] = None,
    config: Optional[Mapping[str, Any] | TurnChecklistConfig] = None,
) -> TurnChecklistConfig:
    """Load frozen controls without making config presence a runtime arm."""

    if isinstance(config, TurnChecklistConfig):
        # A dataclass instance is convenient for unit tests and embedding
        # code, but it must not bypass the owner-fixed trace-only gates by
        # avoiding JSON parsing.  Rebuild an immutable normalized copy rather
        # than mutating a caller-owned config object.
        raw_gates = dict(config.scalar_gates)
        gates: list[tuple[str, float]] = []
        for name in _GATE_NAMES:
            gate = _finite_float(raw_gates.get(name, _DEFAULT_GATES[name]), 0.0)
            if name in _TRACE_ONLY_GATES:
                gate = 0.0
            gates.append((name, max(0.0, min(0.10, gate))))
        cap = _finite_float(config.total_residual_cap, 0.10)
        if not 0.0 < cap <= 0.10:
            raise TurnChecklistConfigError("turn-checklist cap must be in (0, 0.10]")
        return TurnChecklistConfig(
            schema=config.schema,
            total_residual_cap=cap,
            scalar_gates=tuple(gates),
            source=config.source,
            staged_active=bool(config.staged_active),
            guide_support_runtime_authorized=False,
        )
    if config is not None:
        if not isinstance(config, Mapping):
            raise TurnChecklistConfigError("turn-checklist config must be a mapping")
        return _parse_config(config, source="explicit_mapping")
    return _parse_config(
        _read_config_mapping(config_path),
        source=str(Path(config_path) if config_path else DEFAULT_CONFIG_PATH),
    )


def _zeros(n: int) -> tuple[float, ...]:
    return tuple(0.0 for _ in range(max(0, n)))


def _unavailable_overlap_audit(n: int, reason: str) -> dict[str, Any]:
    """Return the r293 all-eight trace shape for a fail-closed stage."""

    zeros = [0.0] * max(0, n)
    return {
        name: {
            "existing_route_overlap_or_distinct_reason": _CHANNEL_OVERLAP_REASON[name],
            "attenuation_or_suppression_decision": f"suppressed_unavailable:{reason}",
            "applied_gate": 0.0,
            "overlap_group": next(
                group for group, members in _OVERLAP_GROUPS.items() if name in members
            ),
            "group_winner": [None] * max(0, n),
            "gated_signed_residual": list(zeros),
            "post_deduplication_signed_residual": list(zeros),
            "post_total_cap_signed_residual": list(zeros),
        }
        for name in CHANNEL_NAMES
    }


def _neutral_trace(n: int, reason: str, *, facts: Optional[Mapping[str, Any]] = None) -> TurnChecklistTrace:
    zeros = _zeros(n)
    channels = tuple(
        ChecklistChannelTrace(name, zeros, zeros, (False,) * n, False, reason)
        for name in CHANNEL_NAMES
    )
    return TurnChecklistTrace(
        channel_names=CHANNEL_NAMES,
        channels=channels,
        guide_support=ChecklistChannelTrace(
            GUIDE_CHANNEL_NAME, zeros, zeros, (False,) * n, False, reason
        ),
        scalar_gates=tuple((name, float(_DEFAULT_GATES[name])) for name in _GATE_NAMES),
        residuals=zeros,
        facts=MappingProxyType(
            {
                **dict(facts or {}),
                "channel_overlap_audit": _unavailable_overlap_audit(n, reason),
            }
        ),
        available=False,
        reason=reason,
        active=False,
    )


def _normalise(
    raw: Sequence[float],
    *,
    available: bool,
    option_availability: Optional[Sequence[bool]] = None,
) -> tuple[float, ...]:
    """Center and L-infinity-normalise only causally answerable options.

    This is the r288 calibration contract.  It is permutation equivariant and
    shift invariant.  A flat vector (including a one-option answerable subset)
    is exactly neutral.
    """

    if not available or not raw:
        return _zeros(len(raw))
    values = [_finite_float(value, 0.0) for value in raw]
    mask = (
        [bool(value) for value in option_availability]
        if option_availability is not None
        else [True] * len(values)
    )
    if len(mask) != len(values):
        return _zeros(len(values))
    indices = [index for index, enabled in enumerate(mask) if enabled]
    if len(indices) < 2:
        return _zeros(len(values))
    mean = sum(values[index] for index in indices) / float(len(indices))
    centered = [0.0] * len(values)
    for index in indices:
        centered[index] = values[index] - mean
    scale = max((abs(centered[index]) for index in indices), default=0.0)
    if scale <= 1e-12:
        return _zeros(len(values))
    return tuple(
        max(-1.0, min(1.0, value / scale)) if mask[index] else 0.0
        for index, value in enumerate(centered)
    )


_OPTION_TYPE_NAMES = MappingProxyType(
    {
        "yes": _legacy.OPT_YES,
        "no": _legacy.OPT_NO,
        "card": _legacy.OPT_CARD,
        "toolcard": _legacy.OPT_TOOL_CARD,
        "energycard": _legacy.OPT_ENERGY_CARD,
        "energy": _legacy.OPT_ENERGY,
        "play": _legacy.OPT_PLAY,
        "attach": _legacy.OPT_ATTACH,
        "evolve": _legacy.OPT_EVOLVE,
        "ability": _legacy.OPT_ABILITY,
        "discard": _legacy.OPT_DISCARD,
        "retreat": _legacy.OPT_RETREAT,
        "attack": _legacy.OPT_ATTACK,
        "end": _legacy.OPT_END,
        "endturn": _legacy.OPT_END,
    }
)
_AREA_NAMES = MappingProxyType(
    {
        "deck": _legacy.AREA_DECK,
        "hand": _legacy.AREA_HAND,
        "discard": _legacy.AREA_DISCARD,
        "active": _legacy.AREA_ACTIVE,
        "bench": _legacy.AREA_BENCH,
        "prize": _legacy.AREA_PRIZE,
        "stadium": _legacy.AREA_STADIUM,
        "energy": _legacy.AREA_ENERGY,
        "tool": _legacy.AREA_TOOL,
        "looking": _legacy.AREA_LOOKING,
    }
)


def _enum_token(value: Any) -> str:
    """Normalize replay enum names without accepting arbitrary prose."""

    if hasattr(value, "name"):
        value = getattr(value, "name")
    return "".join(character for character in str(value) if character.isalnum()).casefold()


def _enum_int(value: Any, names: Mapping[str, int]) -> Optional[int]:
    if isinstance(value, str) or hasattr(value, "name"):
        return names.get(_enum_token(value))
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed


def _option_type(option: Mapping[str, Any]) -> int:
    value = _enum_int(option.get("type", -1), _OPTION_TYPE_NAMES)
    return -1 if value is None else value


def _area_value(value: Any) -> Optional[int]:
    return _enum_int(value, _AREA_NAMES)


def _normalised_option(option: Mapping[str, Any]) -> dict[str, Any]:
    """Copy an option only when JSON enum spellings need native values."""

    result = dict(option)
    kind = _option_type(option)
    if kind >= 0:
        result["type"] = kind
    for name in ("area", "inPlayArea"):
        if name in result:
            area = _area_value(result[name])
            if area is not None:
                result[name] = area
    return result


def _card_id(card: Any) -> Optional[int]:
    return _legacy._card_id(card)


def _cards(value: Any) -> list[Any]:
    return _legacy._cards(value)


def _first(value: Any) -> Any:
    return _legacy._first(value)


def _board_cards(player: Mapping[str, Any]) -> list[Any]:
    return _legacy._board_cards(dict(player))


def _board_counts(player: Mapping[str, Any]) -> Counter[int]:
    return _legacy._board_counts(dict(player))


def _hand_count(player: Mapping[str, Any]) -> Optional[int]:
    raw = player.get("handCount")
    if raw is None:
        raw = len(_cards(player.get("hand")))
    try:
        result = int(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if result >= 0 else None


def _deck_count(player: Mapping[str, Any]) -> Optional[int]:
    raw = player.get("deckCount")
    try:
        result = int(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if result >= 0 else None


def _remaining_hp(card: Any) -> Optional[int]:
    if not isinstance(card, Mapping):
        return None
    try:
        value = int(card.get("hp"))
    except (TypeError, ValueError, OverflowError):
        return None
    return value if value > 0 else None


def _energy_ids(card: Any) -> set[int]:
    return set(_legacy._energy_ids(card))


def _has_fighting_type(card: Any) -> Optional[bool]:
    """Return only a *visible* Fighting type result; no card-table guess."""

    if not isinstance(card, Mapping):
        return None
    value = card.get("types", card.get("type"))
    values = value if isinstance(value, (list, tuple, set)) else [value]
    seen = False
    for item in values:
        if item is None:
            continue
        seen = True
        normal = str(getattr(item, "name", item)).strip().lower()
        if normal in {"f", "fighting", "fight"}:
            return True
    return False if seen else None


def _powerful_hand_protection(card: Any) -> tuple[bool, bool]:
    """Return (known protected, unknown conditional-protection state)."""

    # The evaluated card table has one public generic prevention body whose
    # effect stops attack effects, including Powerful Hand's counter placement.
    # Keep this narrow, explicit, and table-independent for native-less replay
    # inspection; unknown card text is never generalized into prevention.
    if _card_id(card) in {203, 835, 1136}:
        return True, False
    energies = _energy_ids(card)
    if _guide.MIST_ENERGY in energies:
        return True, False
    if _guide.ROCK_FIGHTING_ENERGY in energies:
        fighting = _has_fighting_type(card)
        if fighting is True:
            return True, False
        if fighting is None:
            return False, True
    return False, False


def _has_psychic_energy(card: Any) -> bool:
    return bool(
        _energy_ids(card).intersection(
            {_guide.PSYCHIC_ENERGY, _guide.TELEPATH_PSYCHIC_ENERGY}
        )
    )


def _source_card(obs: Mapping[str, Any], option: Mapping[str, Any]) -> Any:
    try:
        return _legacy._option_card(dict(obs), _normalised_option(option))
    except Exception:
        return None


def _candidate_options(
    candidates: Sequence[Sequence[int]], options: Sequence[Any], *, allow_stop: bool
) -> Optional[list[list[Mapping[str, Any]]]]:
    result: list[list[Mapping[str, Any]]] = []
    for candidate in candidates:
        if not isinstance(candidate, Sequence) or isinstance(candidate, (str, bytes)):
            return None
        row: list[Mapping[str, Any]] = []
        for raw_index in candidate:
            try:
                index = int(raw_index)
            except (TypeError, ValueError, OverflowError):
                return None
            if index < 0 or index >= len(options) or not isinstance(options[index], Mapping):
                return None
            row.append(options[index])
        if not row and not allow_stop:
            return None
        result.append(row)
    return result


def _optional_stop_allowed(select: Mapping[str, Any]) -> bool:
    """Accept ``[]`` only when the current public selection permits STOP."""

    try:
        minimum = int(select.get("minCount", 1))
    except (TypeError, ValueError, OverflowError):
        return False
    return minimum <= 0


def _shared_prefix_length(candidates: Sequence[Sequence[int]]) -> int:
    """Length of the public factorised prefix already selected for all rows."""

    if not candidates:
        return 0
    try:
        common = [int(index) for index in candidates[0]]
    except (TypeError, ValueError, OverflowError):
        return 0
    for candidate in candidates[1:]:
        try:
            values = [int(index) for index in candidate]
        except (TypeError, ValueError, OverflowError):
            return 0
        limit = min(len(common), len(values))
        index = 0
        while index < limit and common[index] == values[index]:
            index += 1
        common = common[:index]
        if not common:
            return 0
    return len(common)


def _new_suffix_rows(
    candidates: Sequence[Sequence[int]], rows: Sequence[Sequence[Mapping[str, Any]]]
) -> tuple[list[list[Mapping[str, Any]]], int]:
    """Remove a common factorised prefix so past effects are never re-scored."""

    prefix = _shared_prefix_length(candidates)
    return [list(row[prefix:]) for row in rows], prefix


def _effect_id(select: Mapping[str, Any]) -> Optional[int]:
    try:
        return _legacy._effect_id(dict(select))
    except Exception:
        return None


def _option_energy_id(obs: Mapping[str, Any], option: Mapping[str, Any]) -> Optional[int]:
    host = _source_card(obs, option)
    if not isinstance(host, Mapping):
        return None
    try:
        index = int(option.get("energyIndex", -1))
    except (TypeError, ValueError, OverflowError):
        return None
    energies = host.get("energyCards")
    if not isinstance(energies, Sequence) or isinstance(energies, (str, bytes)):
        return None
    if index < 0 or index >= len(energies):
        return None
    return _card_id(energies[index])


def _candidate_card_ids(obs: Mapping[str, Any], row: Sequence[Mapping[str, Any]]) -> list[int]:
    ids: list[int] = []
    for option in row:
        card_id = _card_id(_source_card(obs, option))
        if card_id is not None:
            ids.append(card_id)
    return ids


def _source_is_own_hand(
    observation: Mapping[str, Any], option: Mapping[str, Any], *, kind: int
) -> Optional[bool]:
    """Resolve whether an option actually spends the acting player's hand.

    In particular, a discard-selection prompt may name the opponent's card.
    Counting that as our -1 hand cost would corrupt KO math.  ``OPT_PLAY`` is
    defined by the engine as playing the acting player's hand card; the other
    hand-spending forms must expose their source area/seat.
    """

    if kind == _legacy.OPT_PLAY:
        return True
    current = observation.get("current")
    if not isinstance(current, Mapping):
        return None
    try:
        your_index = int(current.get("yourIndex", 0))
        area = _area_value(option.get("area"))
        player_index = int(option.get("playerIndex", your_index))
    except (TypeError, ValueError, OverflowError):
        return None
    if area is None:
        return None
    return area == _legacy.AREA_HAND and player_index == your_index


_UNRESOLVED_PLAY_PREFIX_IDS = frozenset(
    {
        _guide.RARE_CANDY,
        _guide.BUDDY_BUDDY_POFFIN,
        _guide.POKE_PAD,
        _guide.HILDA,
        _guide.DAWN,
        _guide.NIGHT_STRETCHER,
        _guide.SACRED_ASH,
        _guide.LANA_AID,
        _guide.BOSS_ORDERS,
        _guide.ENHANCED_HAMMER,
        _guide.XEROSIC,
    }
)


def _candidate_hand_delta(obs: Mapping[str, Any], row: Sequence[Mapping[str, Any]]) -> tuple[Optional[int], bool]:
    """Known immediate net hand movement and whether all relevant costs resolve.

    This function sees only the *new suffix* of a factorised candidate.  A
    trainer play whose compulsory selection is still ahead is intentionally
    unavailable: calling its visible ``-1`` a final post-cost hand would make
    Hilda, Dawn, Poké Pad, Rare Candy, and recovery prefixes look falsely
    unsafe (or falsely safe).
    """

    total = 0
    known = False
    for option in row:
        kind = _option_type(option)
        source_id = _card_id(_source_card(obs, option))
        if kind in {
            _legacy.OPT_PLAY,
            _legacy.OPT_ATTACH,
            _legacy.OPT_EVOLVE,
            _legacy.OPT_DISCARD,
        }:
            if source_id is None:
                return None, False
            is_own_hand = _source_is_own_hand(obs, option, kind=kind)
            if is_own_hand is None:
                return None, False
            if not is_own_hand:
                # The selection is a visible other-player/other-zone action,
                # not a cost paid from our hand.
                continue
            if kind == _legacy.OPT_PLAY and source_id in _UNRESOLVED_PLAY_PREFIX_IDS:
                return None, False
            total -= 1
            known = True
            if kind == _legacy.OPT_ATTACH and source_id == _guide.ENRICHING_ENERGY:
                total += 4
        elif kind == _legacy.OPT_ATTACK or kind == _legacy.OPT_END:
            continue
    return (total, True) if known else (0, True)


def _prize_count(player: Mapping[str, Any]) -> Optional[int]:
    prize = player.get("prize")
    if isinstance(prize, Sequence) and not isinstance(prize, (str, bytes)):
        return len(prize)
    for key in ("prizeCount", "remainingPrizes"):
        try:
            value = int(player.get(key))
        except (TypeError, ValueError, OverflowError):
            continue
        if value >= 0:
            return value
    return None


def _prize_yield(card: Any) -> Optional[int]:
    if not isinstance(card, Mapping):
        return None
    raw = card.get("prizeYield", card.get("prizeCount"))
    if raw is not None:
        try:
            value = int(raw)
        except (TypeError, ValueError, OverflowError):
            value = 0
        if value > 0:
            return value
    if bool(card.get("megaEx", False)):
        # MEGA ex Pokémon concede three prizes.  The observation exposes the
        # exact card class, so do not collapse it into the ordinary ex value.
        return 3
    public_card_id = _card_id(card)
    if public_card_id is not None:
        public_yield = _public_prize_yield_by_id(public_card_id)
        if public_yield is not None:
            return public_yield
    if public_card_id == _guide.FEZANDIPITI_EX:
        # Some replay surfaces omit generic rule-box/ex booleans on this
        # exact known card.  Its identity is public and its two-prize yield is
        # fixed; treating it as a one-prize Bench card distorts both the
        # exposure and terminal-prize questions.
        return 2
    if bool(card.get("ruleBox", False) or card.get("ex", False)):
        return 2
    # A visible ordinary Pokémon is one prize.  This is not a forecast.
    return 1 if public_card_id is not None else None


@lru_cache(maxsize=1)
def _public_prize_yield_table() -> Mapping[int, int]:
    """Read the shipped public card table once for sparse replay identities.

    Native card metadata is not present in every diagnostic environment, so
    failure simply leaves the caller with the explicit observation fields and
    the narrow known-card fallbacks below.  This never reads a game-private
    zone or guesses from HP/name.
    """

    path = Path(__file__).resolve().parents[1] / "cards" / "EN_Card_Data.csv"
    result: dict[int, int] = {}
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                try:
                    card_id = int(row.get("Card ID", ""))
                except (TypeError, ValueError):
                    continue
                rule = str(row.get("Rule", "")).casefold()
                if "mega" in rule and "ex" in rule:
                    result[card_id] = 3
                elif "pokémon ex" in rule or "pokemon ex" in rule:
                    result[card_id] = max(result.get(card_id, 0), 2)
    except OSError:
        return MappingProxyType({})
    return MappingProxyType(result)


def _public_prize_yield_by_id(card_id: int) -> Optional[int]:
    value = _public_prize_yield_table().get(int(card_id))
    if value is not None:
        return int(value)
    # These two raw identities occur in sparse replay test/inspection
    # surfaces even when supplemental CSV loading is unavailable.
    if card_id == _guide.FEZANDIPITI_EX:
        return 2
    if card_id == 652:  # Mega Venusaur ex
        return 3
    return None


def _is_powerful_hand_attack(option: Mapping[str, Any]) -> bool:
    if _option_type(option) != _legacy.OPT_ATTACK:
        return False
    try:
        return int(option.get("attackId", -1)) == _guide.POWERFUL_HAND_ATTACK
    except (TypeError, ValueError, OverflowError):
        return False


def _has_legal_powerful_hand(rows: Sequence[Sequence[Mapping[str, Any]]]) -> bool:
    return any(_is_powerful_hand_attack(option) for row in rows for option in row)


def _active_attack_ready(me: Mapping[str, Any]) -> bool:
    active = _first(me.get("active"))
    return _card_id(active) == _guide.ALAKAZAM and _has_psychic_energy(active)


def _in_play_target(
    observation: Mapping[str, Any], option: Mapping[str, Any]
) -> tuple[Any, Optional[int]]:
    """Resolve an explicitly named own Active/Bench target, if any.

    The action schema supplies a target for evolves/attachments separately
    from the source card.  We deliberately reject a missing player/area/index
    rather than treating an Active target as a Bench replacement.
    """

    current = observation.get("current")
    if not isinstance(current, Mapping):
        return None, None
    try:
        your_index = int(current.get("yourIndex", 0))
        area = _area_value(option.get("inPlayArea"))
        index = int(option.get("inPlayIndex"))
    except (TypeError, ValueError, OverflowError):
        return None, None
    if area not in {_legacy.AREA_ACTIVE, _legacy.AREA_BENCH} or index < 0:
        return None, None
    raw_player = option.get("inPlayPlayerIndex", your_index)
    try:
        player_index = int(raw_player)
    except (TypeError, ValueError, OverflowError):
        return None, None
    if player_index != your_index:
        return None, None
    try:
        target = _legacy._resolve_card(
            dict(observation), area=area, index=index, player_index=your_index
        )
    except Exception:
        return None, None
    return target, area


def _option_explicit_bench_card(
    observation: Mapping[str, Any], option: Mapping[str, Any]
) -> Any:
    """Resolve an option that directly names a Bench card at a sub-prompt."""

    try:
        area = _area_value(option.get("area"))
        current = observation.get("current")
        your_index = int(current.get("yourIndex", 0)) if isinstance(current, Mapping) else -1
        player_index = int(option.get("playerIndex", your_index))
    except (TypeError, ValueError, OverflowError):
        return None
    if area != _legacy.AREA_BENCH or player_index != your_index:
        return None
    return _source_card(observation, option)


def _row_completes_bench_alakazam(
    observation: Mapping[str, Any], row: Sequence[Mapping[str, Any]]
) -> bool:
    """True only for a legal, visible completion of a Bench replacement.

    A bare Abra/Kadabra, cards in hand, a presumed draw, or a Rare Candy
    prefix are not enough.  The legal evolution/attachment option must name a
    Bench target and that resulting Bench Alakazam must visibly have a
    Psychic-providing energy.
    """

    select = observation.get("select")
    effect_id = _effect_id(select) if isinstance(select, Mapping) else None
    for option in row:
        # At the resolved Rare Candy target-selection prompt, the legal
        # prompt itself proves the Stage-2 source and timing.  We still demand
        # that its exact target is a *Bench* Abra and that public Psychic
        # energy is already there.  The main-stage Candy play prefix is never
        # treated as this completed route.
        prompt_target = _option_explicit_bench_card(observation, option)
        if (
            effect_id == _guide.RARE_CANDY
            and _card_id(prompt_target) == _guide.ABRA
            and _has_psychic_energy(prompt_target)
        ):
            return True
        source_id = _card_id(_source_card(observation, option))
        target, area = _in_play_target(observation, option)
        if area != _legacy.AREA_BENCH:
            continue
        target_id = _card_id(target)
        if (
            _option_type(option) == _legacy.OPT_EVOLVE
            and source_id == _guide.ALAKAZAM
            and target_id == _guide.KADABRA
            and _has_psychic_energy(target)
        ):
            return True
        if (
            _option_type(option) == _legacy.OPT_ATTACH
            and source_id in {_guide.PSYCHIC_ENERGY, _guide.TELEPATH_PSYCHIC_ENERGY}
            and target_id == _guide.ALAKAZAM
        ):
            return True
    return False


def _replacement_timing_unavailable(
    observation: Mapping[str, Any], rows: Sequence[Sequence[Mapping[str, Any]]]
) -> bool:
    """Spot malformed/missing public target evidence without guessing a line."""

    select = observation.get("select")
    effect_id = _effect_id(select) if isinstance(select, Mapping) else None
    for row in rows:
        for option in row:
            kind = _option_type(option)
            source_id = _card_id(_source_card(observation, option))
            if kind == _legacy.OPT_EVOLVE and source_id == _guide.ALAKAZAM:
                target, area = _in_play_target(observation, option)
                if target is None or area is None:
                    return True
            elif kind == _legacy.OPT_ATTACH and source_id in {
                _guide.PSYCHIC_ENERGY,
                _guide.TELEPATH_PSYCHIC_ENERGY,
            }:
                target, area = _in_play_target(observation, option)
                if target is None or area is None:
                    return True
            elif effect_id == _guide.RARE_CANDY and kind == _legacy.OPT_CARD:
                # Explicitly malformed/unknown target selection is different
                # from a known Bench Abra that merely lacks visible energy.
                if _source_card(observation, option) is None:
                    return True
    return False


def _visible_replacement_line_fact(
    observation: Mapping[str, Any],
    me: Mapping[str, Any],
    rows: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Return a conservative Bench-only replacement answer.

    The current Active is intentionally omitted.  It can be the attacker that
    is knocked out; counting it as its own replacement is a prize-race error.
    """

    bench = _cards(me.get("bench"))
    bench_line = [card for card in bench if _card_id(card) in _guide.ALAKAZAM_LINE]
    ready = any(
        _card_id(card) == _guide.ALAKAZAM and _has_psychic_energy(card)
        for card in bench_line
    )
    completable = (not ready) and any(
        _row_completes_bench_alakazam(observation, row) for row in rows
    )
    hand_ids = {
        card_id for card_id in (_card_id(card) for card in _cards(me.get("hand")))
        if card_id is not None
    }
    visible_but_unresolved = bool(bench_line) or bool(
        hand_ids.intersection(_guide.ALAKAZAM_LINE)
    )
    timing_unavailable = (not ready) and (not completable) and _replacement_timing_unavailable(
        observation, rows
    )
    status = (
        "ready"
        if ready
        else "completable"
        if completable
        else "unavailable"
        if timing_unavailable
        else "not_live"
    )
    visible_required_resources = {
        "benched_alakazam": sum(_card_id(card) == _guide.ALAKAZAM for card in bench),
        "benched_kadabra": sum(_card_id(card) == _guide.KADABRA for card in bench),
        "benched_abra": sum(_card_id(card) == _guide.ABRA for card in bench),
        "benched_psychic_powered_alakazam": sum(
            _card_id(card) == _guide.ALAKAZAM and _has_psychic_energy(card)
            for card in bench
        ),
        "legal_bench_completion_candidates": sum(
            _row_completes_bench_alakazam(observation, row) for row in rows
        ),
        "visible_hand_line_cards": sorted(hand_ids.intersection(_guide.ALAKAZAM_LINE)),
    }
    if ready:
        unavailable_or_not_live_reason = "visible_benched_powered_alakazam"
    elif completable:
        unavailable_or_not_live_reason = "visible_legal_bench_completion"
    elif timing_unavailable:
        unavailable_or_not_live_reason = "public_evolution_or_target_timing_unavailable"
    elif visible_but_unresolved:
        unavailable_or_not_live_reason = "visible_piece_requires_unknown_or_future_timing"
    else:
        unavailable_or_not_live_reason = "no_visible_bench_replacement_resources"
    return {
        "next_alakazam_line_ready": ready,
        "next_alakazam_line_completable": completable,
        "next_alakazam_line_visible_but_timing_unresolved": (
            visible_but_unresolved and not ready and not completable
        ),
        "next_alakazam_line_status": status,
        "replacement_line_requires_bench_not_active": True,
        # r292 stable audit fields.  They deliberately describe only visible
        # Bench resources; the current Active never appears in these counts.
        "bench_only": True,
        "classification": status,
        "visible_required_resources": visible_required_resources,
        "unavailable_or_not_live_reason": unavailable_or_not_live_reason,
    }


def _facts_base(
    observation: Mapping[str, Any],
    me: Mapping[str, Any], opponent: Mapping[str, Any], rows: Sequence[Sequence[Mapping[str, Any]]]
) -> dict[str, Any]:
    active = _first(opponent.get("active"))
    hp = _remaining_hp(active)
    hand = _hand_count(me)
    protected, uncertain_protection = _powerful_hand_protection(active)
    facts: dict[str, Any] = {
        "hand_count": hand,
        "opponent_active_remaining_hp": hp,
        "powerful_hand_legal_option_present": _has_legal_powerful_hand(rows),
        "powerful_hand_active_ready": _active_attack_ready(me),
        "powerful_hand_effect_prevented": protected,
        "powerful_hand_conditional_protection_unknown": uncertain_protection,
        "deck_count": _deck_count(me),
        "own_remaining_prizes": _prize_count(me),
    }
    if hp is not None:
        facts["ko_hand_size_required"] = int(math.ceil(hp / 20.0))
    facts.update(_visible_replacement_line_fact(observation, me, rows))
    return facts


def _channel_ko(
    obs: Mapping[str, Any],
    rows: Sequence[Sequence[Mapping[str, Any]]],
    *,
    me: Mapping[str, Any],
    opponent: Mapping[str, Any],
    facts: Mapping[str, Any],
) -> tuple[list[float], list[bool], bool, str]:
    threshold = facts.get("ko_hand_size_required")
    hand = facts.get("hand_count")
    has_attack = bool(facts.get("powerful_hand_legal_option_present"))
    if not isinstance(threshold, int) or not isinstance(hand, int) or not has_attack:
        return [0.0] * len(rows), [False] * len(rows), False, "missing_powerful_hand_ko_inputs"
    if bool(facts.get("powerful_hand_conditional_protection_unknown")):
        return [0.0] * len(rows), [False] * len(rows), False, "conditional_protection_type_unknown"
    protected = bool(facts.get("powerful_hand_effect_prevented"))
    if protected:
        # Keep the numerical threshold as an audit fact, but Mist/Rock
        # prevention means it is not a current Powerful Hand KO answer.  A
        # channel must not manufacture a negative attack preference from a
        # condition whose removal is a separate legal action.
        return [0.0] * len(rows), [False] * len(rows), False, "powerful_hand_effect_prevented"
    result: list[float] = []
    mask: list[bool] = []
    for row in rows:
        delta, exact = _candidate_hand_delta(obs, row)
        attacks = any(_is_powerful_hand_attack(option) for option in row)
        # A candidate that does not attack is an exact zero for this immediate
        # KO question.  A candidate with an unresolved cost is not scored.
        if not attacks:
            result.append(0.0)
            mask.append(True)
            continue
        if not exact or delta is None:
            result.append(0.0)
            mask.append(False)
            continue
        post_hand = hand + delta
        # The legal attack option itself is the authoritative current-turn
        # legality proof.  Do not second-guess it from incomplete card text.
        result.append(1.0 if post_hand >= threshold else -1.0)
        mask.append(True)
    return result, mask, any(mask), "visible_exact_ko_math"


def _channel_safe_spend(
    obs: Mapping[str, Any],
    rows: Sequence[Sequence[Mapping[str, Any]]],
    *,
    facts: Mapping[str, Any],
) -> tuple[list[float], list[bool], bool, str]:
    threshold = facts.get("ko_hand_size_required")
    hand = facts.get("hand_count")
    if (
        not isinstance(threshold, int)
        or not isinstance(hand, int)
        or not bool(facts.get("powerful_hand_legal_option_present"))
    ):
        return [0.0] * len(rows), [False] * len(rows), False, "ko_threshold_unavailable"
    if bool(facts.get("powerful_hand_effect_prevented")) or bool(
        facts.get("powerful_hand_conditional_protection_unknown")
    ):
        return [0.0] * len(rows), [False] * len(rows), False, "current_ko_not_publicly_actionable"
    raw: list[float] = []
    mask: list[bool] = []
    saw_spend = False
    exact_all = True
    for row in rows:
        delta, exact = _candidate_hand_delta(obs, row)
        if not exact or delta is None:
            raw.append(0.0)
            mask.append(False)
            exact_all = False
            continue
        spends = any(
            _option_type(option)
            in {
                _legacy.OPT_PLAY,
                _legacy.OPT_ATTACH,
                _legacy.OPT_EVOLVE,
                _legacy.OPT_DISCARD,
            }
            and _source_is_own_hand(
                obs, option, kind=_option_type(option)
            )
            is True
            for option in row
        )
        if not spends:
            raw.append(0.0)
            mask.append(True)
            continue
        saw_spend = True
        raw.append(1.0 if hand + delta >= threshold else -1.0)
        mask.append(True)
    if not saw_spend:
        return raw, mask, False, "no_exact_spend_candidate"
    return raw, mask, any(mask), "some_prefixes_unresolved" if not exact_all else "exact_post_cost_hand"


def _channel_replacement(
    obs: Mapping[str, Any],
    rows: Sequence[Sequence[Mapping[str, Any]]],
    *,
    me: Mapping[str, Any],
) -> tuple[list[float], list[bool], bool, str]:
    # Do not count Active Pokémon here.  This channel is specifically the
    # *next* attacker after the current one is answered.
    bench = _cards(me.get("bench"))
    ready_bench = any(
        _card_id(card) == _guide.ALAKAZAM and _has_psychic_energy(card)
        for card in bench
    )
    completable = [_row_completes_bench_alakazam(obs, row) for row in rows]
    visible_bench_piece = any(
        _card_id(card) in _guide.ALAKAZAM_LINE for card in bench
    )
    available = ready_bench or visible_bench_piece or any(completable)
    if not available:
        return (
            [0.0] * len(rows),
            [False] * len(rows),
            False,
            "no_visible_bench_replacement_or_legal_completion",
        )
    # Ready is a global fact and therefore intentionally produces a flat,
    # zero-normalized vector.  A legal completion gets a small preference only
    # relative to an otherwise non-live candidate in the *same* legal stage.
    raw = [1.0 if ready_bench else (0.6 if complete else 0.0) for complete in completable]
    reason = (
        "ready_bench_alakazam"
        if ready_bench
        else "legal_bench_completion_visible"
        if any(completable)
        else "visible_bench_line_timing_unresolved"
    )
    return raw, [True] * len(rows), True, reason


def _forced_draw_for_option(
    obs: Mapping[str, Any], option: Mapping[str, Any], effect_id: Optional[int]
) -> Optional[int]:
    kind = _option_type(option)
    source_id = _card_id(_source_card(obs, option))
    if kind == _legacy.OPT_ATTACH and source_id == _guide.ENRICHING_ENERGY:
        return 4
    if kind == _legacy.OPT_ABILITY and source_id in {
        _guide.DUDUNSPARCE,
        _guide.KADABRA,
        _guide.ALAKAZAM,
        _guide.FEZANDIPITI_EX,
    }:
        # A selected ability is its own resolving action in the current
        # factorised surface.  Some compatibility encodings retain a YES in
        # the same complete combo; count the one underlying trigger exactly
        # once rather than double-counting it as two draws.
        return {
            _guide.DUDUNSPARCE: 3,
            _guide.KADABRA: 2,
            _guide.ALAKAZAM: 3,
            _guide.FEZANDIPITI_EX: 3,
        }[source_id]
    if kind == _legacy.OPT_YES:
        return {
            _guide.KADABRA: 2,
            _guide.ALAKAZAM: 3,
            _guide.DUDUNSPARCE: 3,
            _guide.FEZANDIPITI_EX: 3,
        }.get(effect_id)
    if kind == _legacy.OPT_NO and effect_id in {
        _guide.KADABRA,
        _guide.ALAKAZAM,
        _guide.DUDUNSPARCE,
        _guide.FEZANDIPITI_EX,
    }:
        return 0
    # Telepath is intentionally absent: it searches Basic Psychic Pokémon to
    # the Bench and never draws them into hand.
    return None


def _row_has_unresolved_draw_prefix(
    observation: Mapping[str, Any], row: Sequence[Mapping[str, Any]]
) -> bool:
    """Whether a new action opens a compulsory unresolved search/effect.

    A completed selection prompt is not a prefix.  In contrast, a main-stage
    play of Hilda/Dawn/Poké Pad/Rare Candy/recovery/disruption has not yet
    exposed its exact continuation and cannot be used as a draw comparator.
    """

    select = observation.get("select")
    effect_id = _effect_id(select) if isinstance(select, Mapping) else None
    if effect_id is not None:
        return False
    return any(
        _option_type(option) == _legacy.OPT_PLAY
        and _card_id(_source_card(observation, option)) in _UNRESOLVED_PLAY_PREFIX_IDS
        for option in row
    )


def _row_current_forced_draws(
    obs: Mapping[str, Any], row: Sequence[Mapping[str, Any]], effect_id: Optional[int]
) -> tuple[int, int, bool]:
    """Return ``(draw_count, recycled_cards, recognized)`` exactly once.

    Run Away Draw returns the selected Dudunsparce to the deck before drawing,
    so its one visible recycled card increases the non-deckout capacity by
    one.  A compatibility Yes paired with the ability is one trigger, not two.
    """

    ability_draws: list[int] = []
    yes_draws: list[int] = []
    other_draws: list[int] = []
    dudunsparce_trigger = False
    for option in row:
        draw = _forced_draw_for_option(obs, option, effect_id)
        if draw is None:
            continue
        kind = _option_type(option)
        if kind == _legacy.OPT_ABILITY:
            ability_draws.append(draw)
            dudunsparce_trigger = dudunsparce_trigger or (
                _card_id(_source_card(obs, option)) == _guide.DUDUNSPARCE
            )
        elif kind == _legacy.OPT_YES:
            yes_draws.append(draw)
            dudunsparce_trigger = dudunsparce_trigger or (
                effect_id == _guide.DUDUNSPARCE
            )
        else:
            other_draws.append(draw)
    remaining_yes = list(yes_draws)
    forced = 0
    for draw in ability_draws:
        forced += draw
        try:
            remaining_yes.remove(draw)
        except ValueError:
            pass
    forced += sum(remaining_yes) + sum(other_draws)
    return forced, 1 if dudunsparce_trigger and forced else 0, bool(
        ability_draws or yes_draws or other_draws
    )


def _mandatory_next_turn_draw_before_attack(
    row: Sequence[Mapping[str, Any]]
) -> Optional[int]:
    """Return only a timing result proven by the selected suffix.

    An attack happens now (zero compulsory start-of-turn draws); an exact End
    action means one normal draw must occur before this player's next attack.
    Any other suffix may still attack this turn, so its next-attack timing is
    intentionally unavailable rather than guessed.
    """

    if any(_is_powerful_hand_attack(option) for option in row):
        return 0
    if any(_option_type(option) == _legacy.OPT_END for option in row):
        return 1
    return None


def _channel_draws(
    obs: Mapping[str, Any],
    rows: Sequence[Sequence[Mapping[str, Any]]],
    *,
    me: Mapping[str, Any],
    facts: dict[str, Any],
) -> tuple[list[float], list[bool], bool, str]:
    deck_count = _deck_count(me)
    if deck_count is None:
        return [0.0] * len(rows), [False] * len(rows), False, "deck_count_unavailable"
    effect_id = _effect_id((obs.get("select") or {}) if isinstance(obs.get("select"), Mapping) else {})
    forced_by_candidate: list[int] = []
    recycled_by_candidate: list[int] = []
    next_turn_by_candidate: list[Optional[int]] = []
    before_next_attack: list[Optional[int]] = []
    raw: list[float] = []
    mask: list[bool] = []
    recognized = False
    for row in rows:
        unresolved_prefix = _row_has_unresolved_draw_prefix(obs, row)
        forced, recycled, row_recognized = _row_current_forced_draws(
            obs, row, effect_id
        )
        forced_by_candidate.append(forced)
        recycled_by_candidate.append(recycled)
        mandatory_next_turn = None if unresolved_prefix else _mandatory_next_turn_draw_before_attack(row)
        next_turn_by_candidate.append(mandatory_next_turn)
        before_next_attack.append(
            None if mandatory_next_turn is None else forced + mandatory_next_turn
        )
        recognized = recognized or row_recognized
        if unresolved_prefix:
            raw.append(0.0)
            mask.append(False)
        elif not row_recognized:
            raw.append(0.0)
            # The complete legal option is known not to be one of the exact
            # forced-draw effects above, so it is a causal zero comparator.
            mask.append(True)
        elif forced > deck_count + recycled:
            # Drawing exactly the remaining deck is legal; deck-out happens
            # only when this *current mandatory draw* asks for more cards
            # than the visible deck capacity (including Run Away Draw's
            # returned Dudunsparce).  A later start-of-turn draw is exposed
            # separately below and is never smuggled into this immediate
            # comparison.
            raw.append(-1.0 if forced else 0.0)
            mask.append(True)
        else:
            raw.append(-0.10 if forced else 0.0)
            mask.append(True)
    # Keep the historical key as the exact *current* forced component, then
    # expose the next-turn component separately so callers never confuse a
    # guaranteed current draw with an unproven future attack route.
    facts["unavoidable_draws_before_attack"] = tuple(forced_by_candidate)
    facts["current_exact_unavoidable_draws"] = tuple(forced_by_candidate)
    facts["run_away_draw_recycled_cards"] = tuple(recycled_by_candidate)
    facts["mandatory_next_turn_draw_before_next_attack"] = tuple(next_turn_by_candidate)
    facts["unavoidable_draws_before_next_attack"] = tuple(before_next_attack)
    facts["maximum_exact_forced_draw_count"] = max(forced_by_candidate, default=0)
    if not recognized:
        return raw, [False] * len(rows), False, "no_exact_current_turn_draw"
    return raw, mask, True, "exact_current_turn_forced_draws_only"


def _channel_bench_exposure(
    obs: Mapping[str, Any], rows: Sequence[Sequence[Mapping[str, Any]]]
) -> tuple[list[float], list[bool], bool, str]:
    select = obs.get("select")
    effect_id = _effect_id(select) if isinstance(select, Mapping) else None
    try:
        context = int(select.get("context", -1)) if isinstance(select, Mapping) else -1
    except (TypeError, ValueError, OverflowError):
        context = -1

    def benches(option: Mapping[str, Any]) -> bool:
        kind = _option_type(option)
        if kind == _legacy.OPT_PLAY:
            # A legal main-stage Pokémon play is a placement choice.  We do
            # not apply this channel to an already-benched ability source.
            return True
        return context == _legacy.CTX_SETUP_BENCH or effect_id in {
            _guide.BUDDY_BUDDY_POFFIN,
            _guide.TELEPATH_PSYCHIC_ENERGY,
        }

    raw: list[float] = []
    seen = False
    for row in rows:
        score = 0.0
        for option in row:
            if not benches(option):
                continue
            card_id = _card_id(_source_card(obs, option))
            if card_id == _guide.FEZANDIPITI_EX:
                # Legal Flip the Script proves only that its prior-KO condition
                # occurred; it does not prove its unknown cards repair a turn.
                score -= 1.0
                seen = True
            # A non-rule-box utility Basic may be a correct setup card, but
            # its future draw/attack contribution is not visible yet.  Leave
            # it neutral rather than invent a prize-race improvement.
        raw.append(score)
    return raw, [bool(seen)] * len(rows), seen, "visible_bench_prize_trade" if seen else "no_visible_bench_card"


def _boss_target_score(
    obs: Mapping[str, Any], row: Sequence[Mapping[str, Any]], *, me: Mapping[str, Any], opponent: Mapping[str, Any], facts: Mapping[str, Any]) -> Optional[float]:
    select = obs.get("select")
    if not isinstance(select, Mapping) or _effect_id(select) != _guide.BOSS_ORDERS:
        return None
    hand = facts.get("hand_count")
    if not isinstance(hand, int):
        return None
    current_active = _first(opponent.get("active"))
    if not isinstance(current_active, Mapping):
        return None
    current_hp = _remaining_hp(current_active)
    current_protected, current_uncertain = _powerful_hand_protection(current_active)
    current_yield = _prize_yield(current_active)
    prizes = _prize_count(me)
    if prizes is None or current_hp is None or current_yield is None:
        return None
    for option in row:
        if not _option_is_opponent_bench(obs, option):
            continue
        target = _source_card(obs, option)
        hp = _remaining_hp(target)
        if hp is None:
            continue
        protected, uncertain = _powerful_hand_protection(target)
        if protected or uncertain or not _active_attack_ready(me):
            return 0.0
        # This is a Boss target-selection prompt: Boss's -1 hand cost has
        # already resolved in the current public hand.  Do not subtract it a
        # second time.  The main-stage prefix stays neutral until a target is
        # actually selected.
        if hand * 20 < hp:
            return 0.0
        yield_count = _prize_yield(target)
        if yield_count is None:
            return 0.0
        active_ko_now = bool(
            current_hp is not None
            and not current_protected
            and not current_uncertain
            and hand * 20 >= current_hp
        )
        # A Boss choice is an immediate positive only when its public target
        # shortens the prize map: it closes the game, yields more than the
        # current Active, or bypasses a current Active we cannot presently KO.
        shortens = bool(
            yield_count >= prizes
            or not active_ko_now
            or (current_yield is not None and yield_count > current_yield)
        )
        if not shortens:
            return 0.0
        return 1.0 if yield_count >= prizes else 0.6
    # The current prompt may be a Boss selection but no candidate named a
    # provable opponent Bench target.  Returning ``None`` keeps it neutral
    # rather than treating Active/own/unknown cards as a gust outcome.
    return None


def _option_is_opponent_active(
    observation: Mapping[str, Any], option: Mapping[str, Any]
) -> bool:
    """Check only the explicit target zone; never infer a future gust line."""

    try:
        area = _area_value(option.get("area", -1))
        player_index = int(option.get("playerIndex", -1))
        your_index = int(
            ((observation.get("current") or {}).get("yourIndex", 0))
        )
    except (AttributeError, TypeError, ValueError, OverflowError):
        return False
    return area == _legacy.AREA_ACTIVE and player_index == 1 - your_index


def _option_is_opponent_bench(
    observation: Mapping[str, Any], option: Mapping[str, Any]
) -> bool:
    """Boss may score only a public, legal opponent Bench target."""

    try:
        area = _area_value(option.get("area", -1))
        player_index = int(option.get("playerIndex", -1))
        your_index = int(((observation.get("current") or {}).get("yourIndex", 0)))
        index = int(option.get("index", -1))
    except (AttributeError, TypeError, ValueError, OverflowError):
        return False
    return (
        area == _legacy.AREA_BENCH
        and player_index == 1 - your_index
        and index >= 0
        and isinstance(_source_card(observation, option), Mapping)
    )


def _channel_disruption(
    obs: Mapping[str, Any],
    rows: Sequence[Sequence[Mapping[str, Any]]],
    *,
    me: Mapping[str, Any],
    opponent: Mapping[str, Any],
    facts: Mapping[str, Any],
) -> tuple[list[float], list[bool], bool, str]:
    raw: list[float] = []
    masks: list[bool] = []
    saw = False
    opponent_ids = {
        card_id for card_id in (_card_id(card) for card in _board_cards(opponent)) if card_id is not None
    }
    for row in rows:
        score = 0.0
        row_available = False
        boss_score = _boss_target_score(obs, row, me=me, opponent=opponent, facts=facts)
        if boss_score is not None:
            score += boss_score
            row_available = True
        for option in row:
            kind = _option_type(option)
            source_id = _card_id(_source_card(obs, option))
            effect_id = _effect_id((obs.get("select") or {}) if isinstance(obs.get("select"), Mapping) else {})
            if kind in {_legacy.OPT_ENERGY_CARD, _legacy.OPT_ENERGY} and effect_id == _guide.ENHANCED_HAMMER:
                energy_id = _option_energy_id(obs, option)
                active_target = _option_is_opponent_active(obs, option)
                if energy_id == _guide.MIST_ENERGY and active_target:
                    # Mist blocks Powerful Hand's attack effect on *any*
                    # Pokémon.  Removing it from the defending Active is a
                    # public immediate change, not a matchup forecast.
                    score += 1.0 if _active_attack_ready(me) else 0.0
                    row_available = True
                elif energy_id == _guide.ROCK_FIGHTING_ENERGY:
                    host = _source_card(obs, option)
                    fighting = _has_fighting_type(host)
                    if fighting is True and active_target:
                        score += 0.8 if _active_attack_ready(me) else 0.0
                        row_available = True
                    elif fighting is False:
                        # Rock Fighting on a non-Fighting target is not a
                        # universal immunity; no inferred hammer outcome.
                        row_available = True
                elif energy_id is not None:
                    # An identifiable non-protection energy is an exact
                    # no-op for the Powerful Hand protection question.
                    row_available = True
            elif kind == _legacy.OPT_PLAY and source_id == _guide.XEROSIC:
                try:
                    opponent_hand = int(opponent.get("handCount"))
                except (TypeError, ValueError, OverflowError):
                    opponent_hand = -1
                # The attachment's mirror rule is public and narrow: do not
                # claim an unknown next attack, only reward a visible
                # Alakazam hand above the seven-card mirror discipline.
                if opponent_hand >= 0 and _guide.ALAKAZAM in opponent_ids:
                    score += 0.5 if opponent_hand > 7 else 0.0
                    row_available = True
            elif kind == _legacy.OPT_PLAY and source_id == _guide.BATTLE_CAGE:
                if (
                    _cards(me.get("bench"))
                    # Munkidori ex (139) does not satisfy the narrow Cage
                    # predicate used here; never fold it in via the broader
                    # historical Munkidori family set.
                    and opponent_ids.intersection({_guide.FROSLASS, 112})
                ):
                    score += 0.4
                    # A relevant public counter-placement body is present.
                    # The aggregate Q6 gate is trace-only, but this remains
                    # a faithful immediate board-predicate trace.
                    row_available = True
        raw.append(score)
        masks.append(row_available)
        saw = saw or row_available
    return raw, masks, saw, "visible_immediate_disruption" if saw else "no_immediate_disruption_outcome"


def _ledger_lower(ledger_snapshot: Any, card_id: int) -> Optional[int]:
    availability = getattr(ledger_snapshot, "availability_by_card", None)
    if not isinstance(availability, Mapping):
        return None
    row = availability.get(card_id)
    try:
        lower = int(getattr(row, "lower"))
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    return lower if lower >= 0 else None


_OWN_DECK_LEDGER_SCHEMA = "poke_bot.own_deck_ledger/v1"
_EXACT_ALAKAZAM_LEDGER_DECK_FINGERPRINT = (
    "sha256:44284481e46dd2aac8d92bea417cbcfcda40db221d76fe39185a01d11754fce8"
)


def _ledger_binding(
    observation: Mapping[str, Any], ledger_snapshot: Any
) -> tuple[bool, Optional[int], str]:
    """Bind Q7 only to the exact actor's immutable own-deck ledger."""

    if ledger_snapshot is None:
        return False, None, "own_deck_ledger_unavailable"
    if getattr(ledger_snapshot, "schema", None) != _OWN_DECK_LEDGER_SCHEMA:
        return False, None, "own_deck_ledger_schema_unbound"
    if getattr(ledger_snapshot, "deck_fingerprint", None) != _EXACT_ALAKAZAM_LEDGER_DECK_FINGERPRINT:
        return False, None, "own_deck_ledger_deck_fingerprint_unbound"
    current = observation.get("current")
    try:
        actor = int(current.get("yourIndex", 0)) if isinstance(current, Mapping) else None
    except (TypeError, ValueError, OverflowError):
        actor = None
    if actor is None or getattr(ledger_snapshot, "actor", None) != actor:
        return False, None, "own_deck_ledger_actor_unbound"
    fingerprint = getattr(ledger_snapshot, "fingerprint", None)
    if not isinstance(fingerprint, str) or not fingerprint.startswith("sha256:") or len(fingerprint) <= len("sha256:"):
        return False, None, "own_deck_ledger_snapshot_fingerprint_missing"
    if not bool(getattr(ledger_snapshot, "integrity_ok", False)) or bool(getattr(ledger_snapshot, "fail_closed", True)):
        return False, None, "own_deck_ledger_not_causally_valid"
    slots = getattr(ledger_snapshot, "unknown_prize_slots", None)
    if isinstance(slots, bool):
        return False, None, "unknown_prize_slots_unavailable"
    try:
        unknown_slots = int(slots)
    except (TypeError, ValueError, OverflowError):
        return False, None, "unknown_prize_slots_unavailable"
    if unknown_slots <= 0:
        return False, unknown_slots, "no_unknown_prize_slot"
    return True, unknown_slots, "ledger_bound"


def _visible_nonactive_resource_ids(me: Mapping[str, Any]) -> set[int]:
    """Physical own resources that still exist if the current Active is lost."""

    return {
        card_id
        for card_id in (
            _card_id(card)
            for card in (_cards(me.get("bench")) + _cards(me.get("hand")))
        )
        if card_id is not None
    }


def _resource_is_proven(
    card_id: int, *, visible_ids: set[int], ledger_snapshot: Any
) -> bool:
    return card_id in visible_ids or bool((_ledger_lower(ledger_snapshot, card_id) or 0) > 0)


def _psychic_route_energy(
    *, visible_ids: set[int], ledger_snapshot: Any
) -> Optional[int]:
    for card_id in (_guide.PSYCHIC_ENERGY, _guide.TELEPATH_PSYCHIC_ENERGY):
        if card_id in visible_ids:
            return card_id
    for card_id in (_guide.PSYCHIC_ENERGY, _guide.TELEPATH_PSYCHIC_ENERGY):
        if (_ledger_lower(ledger_snapshot, card_id) or 0) > 0:
            return card_id
    return None


def _route_fact(
    required_without_energy: Sequence[int],
    *,
    visible_ids: set[int],
    ledger_snapshot: Any,
) -> dict[str, Any]:
    energy = _psychic_route_energy(
        visible_ids=visible_ids, ledger_snapshot=ledger_snapshot
    )
    required = list(required_without_energy)
    if energy is not None:
        required.append(energy)
    proven = energy is not None and all(
        _resource_is_proven(
            card_id, visible_ids=visible_ids, ledger_snapshot=ledger_snapshot
        )
        for card_id in required
    )
    return {
        "required_resources": tuple(required),
        "lower_bound_proven": bool(proven),
        "psychic_providing_energy": energy,
    }


def _unknown_prize_facts(
    observation: Mapping[str, Any],
    *,
    me: Mapping[str, Any],
    ledger_snapshot: Any,
) -> dict[str, Any]:
    """Build Q7 facts without inferring a hidden draw, prize, or recovery."""

    bound, slots, binding_reason = _ledger_binding(observation, ledger_snapshot)
    facts: dict[str, Any] = {
        "unknown_prize_slots": slots,
        "unknown_prize_ledger_binding": binding_reason,
        "unknown_prize_route_classification": "unavailable",
        "safe_paths": {
            "natural": {
                "required_resources": (),
                "lower_bound_proven": False,
                "psychic_providing_energy": None,
            },
            "rare_candy": {
                "required_resources": (),
                "lower_bound_proven": False,
                "psychic_providing_energy": None,
            },
        },
        "key_lower_bounds": {},
        "unknown_prize_unavailable_or_brittle_reason": binding_reason,
    }
    if not bound:
        return facts
    visible_ids = _visible_nonactive_resource_ids(me)
    natural = _route_fact(
        (_guide.ABRA, _guide.KADABRA, _guide.ALAKAZAM),
        visible_ids=visible_ids,
        ledger_snapshot=ledger_snapshot,
    )
    candy = _route_fact(
        (_guide.ABRA, _guide.RARE_CANDY, _guide.ALAKAZAM),
        visible_ids=visible_ids,
        ledger_snapshot=ledger_snapshot,
    )
    key_ids = (
        _guide.ABRA,
        _guide.KADABRA,
        _guide.ALAKAZAM,
        _guide.RARE_CANDY,
        _guide.PSYCHIC_ENERGY,
        _guide.TELEPATH_PSYCHIC_ENERGY,
    )
    facts["safe_paths"] = {"natural": natural, "rare_candy": candy}
    facts["key_lower_bounds"] = {
        card_id: (1 if card_id in visible_ids else _ledger_lower(ledger_snapshot, card_id))
        for card_id in key_ids
    }
    if natural["lower_bound_proven"]:
        facts["unknown_prize_route_classification"] = "natural"
        facts["unknown_prize_unavailable_or_brittle_reason"] = None
    elif candy["lower_bound_proven"]:
        facts["unknown_prize_route_classification"] = "rare_candy"
        facts["unknown_prize_unavailable_or_brittle_reason"] = None
    else:
        facts["unknown_prize_route_classification"] = "brittle"
        facts["unknown_prize_unavailable_or_brittle_reason"] = (
            "no_visible_or_conservative_lower_bound_complete_natural_or_rare_candy_route"
        )
    return facts


def _channel_prize_robust(
    obs: Mapping[str, Any],
    rows: Sequence[Sequence[Mapping[str, Any]]],
    *,
    ledger_snapshot: Any,
    facts: dict[str, Any],
) -> tuple[list[float], list[bool], bool, str]:
    try:
        me, _opponent = _legacy._players(dict(obs))
    except Exception:
        me = None
    if not isinstance(me, Mapping):
        facts.update(
            {
                "unknown_prize_route_classification": "unavailable",
                "unknown_prize_unavailable_or_brittle_reason": "acting_player_unavailable",
            }
        )
        return [0.0] * len(rows), [False] * len(rows), False, "acting_player_unavailable"
    route_facts = _unknown_prize_facts(
        obs, me=me, ledger_snapshot=ledger_snapshot
    )
    facts.update(route_facts)
    if route_facts["unknown_prize_ledger_binding"] != "ledger_bound":
        return (
            [0.0] * len(rows),
            [False] * len(rows),
            False,
            str(route_facts["unknown_prize_unavailable_or_brittle_reason"]),
        )
    visible_ids = _visible_nonactive_resource_ids(me)
    natural = route_facts["safe_paths"]["natural"]
    candy = route_facts["safe_paths"]["rare_candy"]
    complete_route = bool(natural["lower_bound_proven"] or candy["lower_bound_proven"])
    # A line-only proof is intentionally weaker than a powered route.  It may
    # distinguish a resolved tutor's public line card from a key card that can
    # still be the unknown prize, but it cannot make a Q3 replacement live.
    line_only_proven = all(
        _resource_is_proven(
            card_id, visible_ids=visible_ids, ledger_snapshot=ledger_snapshot
        )
        for card_id in (_guide.ABRA, _guide.KADABRA, _guide.ALAKAZAM)
    )
    select = obs.get("select")
    effect_id = _effect_id(select) if isinstance(select, Mapping) else None
    raw: list[float] = []
    mask: list[bool] = []
    saw_relevant = False
    for row in rows:
        ids = set(_candidate_card_ids(obs, row))
        # Sacred Ash returns selected cards to the deck and no recovery card
        # gets a generic immediate score.  This is a fail-closed Q7 answer.
        if effect_id in _guide.RECOVERY_CARDS:
            raw.append(0.0)
            mask.append(False)
            continue
        if _row_completes_bench_alakazam(obs, row) and complete_route:
            raw.append(1.0)
            mask.append(True)
            saw_relevant = True
            continue
        selected_line = ids.intersection(_guide.ALAKAZAM_LINE)
        if effect_id is not None and selected_line:
            per_row_visible = set(visible_ids).union(selected_line)
            per_row_line = all(
                _resource_is_proven(
                    card_id,
                    visible_ids=per_row_visible,
                    ledger_snapshot=ledger_snapshot,
                )
                for card_id in (_guide.ABRA, _guide.KADABRA, _guide.ALAKAZAM)
            )
            raw.append(0.5 if per_row_line else 0.0)
            mask.append(True)
            saw_relevant = saw_relevant or per_row_line
            continue
        raw.append(0.0)
        mask.append(bool(complete_route or line_only_proven))
    if not (complete_route or line_only_proven or saw_relevant):
        return raw, [False] * len(rows), False, "no_conservative_alternate_route"
    return raw, mask, True, "visible_or_ledger_bound_natural_or_rare_candy_alternate_route"


def _channel_terminal(
    obs: Mapping[str, Any],
    rows: Sequence[Sequence[Mapping[str, Any]]],
    *,
    me: Mapping[str, Any],
    opponent: Mapping[str, Any],
    facts: Mapping[str, Any],
) -> tuple[list[float], list[bool], bool, str]:
    threshold = facts.get("ko_hand_size_required")
    hand = facts.get("hand_count")
    prizes = _prize_count(me)
    target = _first(opponent.get("active"))
    yield_count = _prize_yield(target)
    deck_count = _deck_count(me)
    has_attack = bool(facts.get("powerful_hand_legal_option_present"))
    protected = bool(facts.get("powerful_hand_effect_prevented"))
    uncertain = bool(facts.get("powerful_hand_conditional_protection_unknown"))
    opponent_bench = opponent.get("bench")
    no_visible_opponent_bench = isinstance(opponent_bench, Sequence) and not isinstance(
        opponent_bench, (str, bytes)
    ) and len(opponent_bench) == 0
    if (
        not all(isinstance(value, int) for value in (threshold, hand, prizes, yield_count))
        or prizes <= 0
        or not has_attack
    ):
        return [0.0] * len(rows), [False] * len(rows), False, "terminal_prize_or_ko_inputs_unavailable"
    if uncertain:
        return [0.0] * len(rows), [False] * len(rows), False, "conditional_protection_type_unknown"
    if protected:
        return [0.0] * len(rows), [False] * len(rows), False, "powerful_hand_effect_prevented"
    select = obs.get("select")
    effect_id = _effect_id(select) if isinstance(select, Mapping) else None
    raw: list[float] = []
    mask: list[bool] = []
    for row in rows:
        delta, exact = _candidate_hand_delta(obs, row)
        if not exact or delta is None:
            raw.append(0.0)
            mask.append(False)
            continue
        forced_draws, recycled, _recognized = _row_current_forced_draws(
            obs, row, effect_id
        )
        if forced_draws:
            if deck_count is None or forced_draws > deck_count + recycled:
                raw.append(0.0)
                mask.append(False)
                continue
        attacks = any(_is_powerful_hand_attack(option) for option in row)
        if (
            attacks
            and hand + delta >= threshold
            and (yield_count >= prizes or no_visible_opponent_bench)
        ):
            raw.append(1.0)
        else:
            raw.append(0.0)
        mask.append(True)
    return raw, mask, any(mask), "visible_terminal_ko_before_future_draw"


def _nonflat_normalized(values: Sequence[float]) -> bool:
    return any(abs(_finite_float(value, 0.0)) > 1e-12 for value in values)


def _channel_group(name: str) -> str:
    for group, members in _OVERLAP_GROUPS.items():
        if name in members:
            return group
    raise ValueError(f"unknown checklist channel {name}")


def _grouped_residuals(
    traces: Sequence[ChecklistChannelTrace],
    *,
    gates: Mapping[str, float],
    cap: float,
) -> tuple[list[float], dict[str, Any]]:
    """Apply r293's non-additive closure/continuity aggregation exactly.

    Each channel first creates its own bounded signed contribution.  For each
    option the strongest *nonzero* contribution in closure and continuity wins
    independently; a tie keeps ``CHANNEL_NAMES`` order.  Q5's board group,
    Q6's aggregate disruption gate, and the historic guide are trace-only in
    this revision, so no broad existing route can be double-counted.
    """

    by_name = {trace.name: trace for trace in traces}
    if set(by_name) != set(CHANNEL_NAMES):
        raise ValueError("checklist channel inventory changed")
    width = len(traces[0].normalized) if traces else 0
    contributions: dict[str, list[float]] = {}
    audit: dict[str, Any] = {}
    for name in CHANNEL_NAMES:
        trace = by_name[name]
        gate_name = _CHANNEL_GATE[name]
        gate = _finite_float(gates.get(gate_name, 0.0), 0.0)
        # Redundant runtime guard in addition to config parsing: a manually
        # constructed immutable config cannot make r293 trace-only branches
        # influence policy scores.
        if gate_name in _TRACE_ONLY_GATES:
            gate = 0.0
        values = [gate * float(value) for value in trace.normalized]
        contributions[name] = values
        audit[name] = {
            "existing_route_overlap_or_distinct_reason": _CHANNEL_OVERLAP_REASON[name],
            "attenuation_or_suppression_decision": "suppressed_zero_gated_contribution",
            "applied_gate": float(gate),
            "overlap_group": _channel_group(name),
            "group_winner": [None] * width,
            "gated_signed_residual": list(values),
            "post_deduplication_signed_residual": [0.0] * width,
            "post_total_cap_signed_residual": [0.0] * width,
        }

    group_contributions: dict[str, list[float]] = {
        group: [0.0] * width for group in _RESIDUAL_GROUP_ORDER
    }
    for group in _RESIDUAL_GROUP_ORDER:
        members = _OVERLAP_GROUPS[group]
        for index in range(width):
            winner: Optional[str] = None
            winner_value = 0.0
            for name in members:  # declared channel order is the tie-break.
                value = contributions[name][index]
                if abs(value) <= 1e-12:
                    continue
                if winner is None or abs(value) > abs(winner_value) + 1e-12:
                    winner, winner_value = name, value
            if winner is None:
                continue
            group_contributions[group][index] = winner_value
            audit[winner]["group_winner"][index] = winner
            audit[winner]["post_deduplication_signed_residual"][index] = winner_value
            for name in members:
                if name == winner:
                    continue
                audit[name]["group_winner"][index] = winner

    pre_cap = [
        sum(group_contributions[group][index] for group in _RESIDUAL_GROUP_ORDER)
        for index in range(width)
    ]
    residuals = [max(-cap, min(cap, value)) for value in pre_cap]
    # Attribute a whole-stage total-cap attenuation proportionally between the
    # two retained group contributions.  If they cancel naturally, no cap was
    # applied and their signed post-dedup values remain the truthful audit.
    for index, pre_value in enumerate(pre_cap):
        scale = 1.0
        if abs(pre_value) > cap and abs(pre_value) > 1e-12:
            scale = residuals[index] / pre_value
        for name in CHANNEL_NAMES:
            selected = audit[name]["post_deduplication_signed_residual"][index]
            audit[name]["post_total_cap_signed_residual"][index] = selected * scale

    for name in CHANNEL_NAMES:
        row = audit[name]
        gate_name = _CHANNEL_GATE[name]
        group = row["overlap_group"]
        winners = row["group_winner"]
        selected = row["post_deduplication_signed_residual"]
        if gate_name in _TRACE_ONLY_GATES:
            decision = "trace_only_fixed_zero"
        elif any(value != 0.0 for value in selected):
            decision = f"retained_{group}_group_winner"
        elif any(abs(value) > 1e-12 for value in row["gated_signed_residual"]):
            decision = f"suppressed_by_{group}_group_winner"
        elif by_name[name].available:
            decision = "suppressed_zero_gated_contribution"
        else:
            decision = "suppressed_unavailable_or_flat"
        row["attenuation_or_suppression_decision"] = decision
        # The group-winner list is meaningful even for suppressed members;
        # preserve it rather than replacing it with a generic explanation.
        row["group_winner_identity"] = list(winners)
        row["group_winner_contribution"] = list(group_contributions.get(group, [0.0] * width))
    return residuals, audit


def _guide_channel(
    observation: Mapping[str, Any],
    candidates: Sequence[Sequence[int]],
    deck: Iterable[int],
    *,
    runtime_authorized: bool,
) -> tuple[list[float], list[bool], bool, str]:
    # The old guide contains broad mechanical assumptions intentionally
    # superseded by r292 (for example around protection and recovery).  Do
    # not even use it as a residual source until an immutable audit receipt
    # explicitly authorizes that future experiment.  This keeps the named
    # channel in traces while making the default fail closed.
    if not runtime_authorized:
        return (
            [0.0] * len(candidates),
            [False] * len(candidates),
            False,
            "historic_guide_trace_only_unaudited",
        )
    try:
        scores = _guide.guide_scores(dict(observation), candidates, deck=deck, force_enabled=True)
    except Exception:
        return [0.0] * len(candidates), [False] * len(candidates), False, "historic_guide_unavailable"
    if scores is None or len(scores) != len(candidates):
        return [0.0] * len(candidates), [False] * len(candidates), False, "historic_guide_has_no_safe_distinction"
    values = [_finite_float(value, 0.0) for value in scores]
    if max(values, default=0.0) - min(values, default=0.0) < 1e-12:
        return values, [False] * len(candidates), False, "historic_guide_flat"
    return values, [True] * len(candidates), True, "separate_historic_guide_support"


def evaluate_turn_checklist(
    observation: Mapping[str, Any],
    candidates: Sequence[Sequence[int]],
    deck: Iterable[int],
    *,
    ledger_snapshot: Any = None,
    config_path: Optional[str] = None,
    config: Optional[Mapping[str, Any] | TurnChecklistConfig] = None,
) -> TurnChecklistTrace:
    """Return all eight causal channel vectors and one bounded residual.

    A wrong list, malformed stage, unknown key fact, or invalid ledger does
    not throw an action preference into the game: its affected channel is zero
    and unavailable.  A malformed root is completely neutral.
    """

    try:
        n = len(candidates)
    except TypeError:
        return _neutral_trace(0, "candidate_stage_not_a_sequence")
    if n <= 0:
        return _neutral_trace(0, "candidate_stage_empty")
    try:
        deck_tuple = tuple(int(card_id) for card_id in deck)
    except (TypeError, ValueError, OverflowError):
        return _neutral_trace(n, "deck_identity_malformed")
    if not _guide.is_alakazam_new_list_deck(deck_tuple):
        return _neutral_trace(n, "exact_new_list_required")
    try:
        resolved_config = load_turn_checklist_config(config_path=config_path, config=config)
    except TurnChecklistConfigError as exc:
        return _neutral_trace(n, "invalid_config", facts={"config_error": str(exc)})
    if not isinstance(observation, Mapping):
        return _neutral_trace(n, "observation_not_a_mapping")
    select = observation.get("select")
    if not isinstance(select, Mapping):
        return _neutral_trace(n, "select_missing")
    options = select.get("option")
    if not isinstance(options, Sequence) or isinstance(options, (str, bytes)):
        return _neutral_trace(n, "legal_options_missing")
    rows = _candidate_options(
        candidates,
        options,
        allow_stop=_optional_stop_allowed(select),
    )
    if rows is None:
        return _neutral_trace(n, "candidate_option_alignment_invalid")
    suffix_rows, shared_prefix_length = _new_suffix_rows(candidates, rows)
    try:
        me, opponent = _legacy._players(dict(observation))
    except Exception:
        me, opponent = None, None
    if not isinstance(me, Mapping) or not isinstance(opponent, Mapping):
        return _neutral_trace(n, "public_players_missing")

    facts = _facts_base(observation, me, opponent, suffix_rows)
    facts["config_source"] = resolved_config.source
    facts["config_staged_active"] = resolved_config.staged_active
    facts["factorized_shared_prefix_length"] = shared_prefix_length
    channel_builders = (
        lambda: _channel_ko(observation, suffix_rows, me=me, opponent=opponent, facts=facts),
        lambda: _channel_safe_spend(observation, suffix_rows, facts=facts),
        lambda: _channel_replacement(observation, suffix_rows, me=me),
        lambda: _channel_draws(observation, suffix_rows, me=me, facts=facts),
        lambda: _channel_bench_exposure(observation, suffix_rows),
        lambda: _channel_disruption(observation, suffix_rows, me=me, opponent=opponent, facts=facts),
        lambda: _channel_prize_robust(observation, suffix_rows, ledger_snapshot=ledger_snapshot, facts=facts),
        lambda: _channel_terminal(observation, suffix_rows, me=me, opponent=opponent, facts=facts),
    )
    traces: list[ChecklistChannelTrace] = []
    for name, build in zip(CHANNEL_NAMES, channel_builders, strict=True):
        try:
            raw, option_availability, available, reason = build()
            if (
                len(raw) != n
                or len(option_availability) != n
                or any(not math.isfinite(float(value)) for value in raw)
            ):
                raise ValueError("channel output is misaligned or non-finite")
            raw_tuple = tuple(float(value) for value in raw)
            mask_tuple = tuple(bool(value) for value in option_availability)
            normalized_tuple = _normalise(
                raw_tuple,
                available=bool(available),
                option_availability=mask_tuple,
            )
            # Config/Elmo contract: ``available`` means a nonflat actionable
            # distinction, not merely that the channel inspected some public
            # facts.  Preserve the per-option mask/reason on exact flat rows.
            actionable = bool(available) and _nonflat_normalized(normalized_tuple)
            trace_reason = str(reason)
            if bool(available) and not actionable:
                trace_reason = f"{trace_reason}:flat_no_action_distinction"
            traces.append(
                ChecklistChannelTrace(
                    name=name,
                    raw=raw_tuple,
                    normalized=normalized_tuple,
                    option_availability=mask_tuple,
                    available=actionable,
                    reason=trace_reason,
                )
            )
        except Exception as exc:
            zeros = _zeros(n)
            traces.append(
                ChecklistChannelTrace(
                    name, zeros, zeros, (False,) * n, False,
                    f"channel_unavailable:{type(exc).__name__}",
                )
            )

    guide_raw, guide_mask, guide_available, guide_reason = _guide_channel(
        observation,
        candidates,
        deck_tuple,
        runtime_authorized=resolved_config.guide_support_runtime_authorized,
    )
    if (
        len(guide_raw) != n
        or len(guide_mask) != n
        or any(not math.isfinite(float(value)) for value in guide_raw)
    ):
        guide_raw = [0.0] * n
        guide_mask = [False] * n
        guide_available = False
        guide_reason = "historic_guide_misaligned"
    guide_normalized = _normalise(
        guide_raw,
        available=bool(guide_available),
        option_availability=guide_mask,
    )
    guide_trace = ChecklistChannelTrace(
        GUIDE_CHANNEL_NAME,
        tuple(float(value) for value in guide_raw),
        guide_normalized,
        tuple(bool(value) for value in guide_mask),
        bool(guide_available) and _nonflat_normalized(guide_normalized),
        (
            str(guide_reason)
            if not bool(guide_available) or _nonflat_normalized(guide_normalized)
            else f"{guide_reason}:flat_no_action_distinction"
        ),
    )

    gates = resolved_config.gates
    cap = resolved_config.total_residual_cap
    residuals, overlap_audit = _grouped_residuals(
        traces, gates=gates, cap=cap
    )
    facts["channel_overlap_audit"] = overlap_audit
    facts["guide_support_trace_only"] = True
    facts["guide_support_runtime_residual"] = 0.0
    facts["residual_group_order"] = _RESIDUAL_GROUP_ORDER
    available = any(trace.available for trace in traces)
    return TurnChecklistTrace(
        channel_names=CHANNEL_NAMES,
        channels=tuple(traces),
        guide_support=guide_trace,
        scalar_gates=resolved_config.scalar_gates,
        residuals=tuple(residuals),
        facts=MappingProxyType(dict(facts)),
        available=available,
        reason="causal_stage_evaluated" if available else "no_safe_channel_distinction",
        active=available,
    )


def apply_turn_checklist_logits(
    observation: Mapping[str, Any],
    candidates: Sequence[Sequence[int]],
    deck: Iterable[int],
    *,
    logits: Any,
    ledger_snapshot: Any = None,
    config_path: Optional[str] = None,
    config: Optional[Mapping[str, Any] | TurnChecklistConfig] = None,
) -> tuple[Any, TurnChecklistTrace]:
    """Apply the bounded residual while preserving tensor dtype/device/shape."""

    trace = evaluate_turn_checklist(
        observation,
        candidates,
        deck,
        ledger_snapshot=ledger_snapshot,
        config_path=config_path,
        config=config,
    )
    try:
        import torch

        if isinstance(logits, torch.Tensor):
            if logits.numel() != len(trace.residuals):
                return logits, _neutral_trace(len(trace.residuals), "logit_width_mismatch")
            residual = torch.as_tensor(
                trace.residuals,
                dtype=logits.dtype if logits.is_floating_point() else torch.float32,
                device=logits.device,
            ).reshape(logits.shape)
            if not logits.is_floating_point():
                return logits, _neutral_trace(len(trace.residuals), "logits_not_floating")
            return logits + residual, trace
    except ImportError:
        pass
    try:
        values = [float(value) for value in logits]
    except (TypeError, ValueError):
        return logits, _neutral_trace(len(trace.residuals), "logits_not_vector")
    if len(values) != len(trace.residuals):
        return logits, _neutral_trace(len(trace.residuals), "logit_width_mismatch")
    return [value + residual for value, residual in zip(values, trace.residuals, strict=True)], trace


def apply_turn_checklist_probabilities(
    observation: Mapping[str, Any],
    candidates: Sequence[Sequence[int]],
    deck: Iterable[int],
    *,
    probabilities: Sequence[float],
    ledger_snapshot: Any = None,
    config_path: Optional[str] = None,
    config: Optional[Mapping[str, Any] | TurnChecklistConfig] = None,
) -> tuple[list[float], TurnChecklistTrace]:
    """Apply the same residual to remote normalized priors and renormalise."""

    trace = evaluate_turn_checklist(
        observation,
        candidates,
        deck,
        ledger_snapshot=ledger_snapshot,
        config_path=config_path,
        config=config,
    )
    # A neutral residual is an exact identity operation.  In particular, do
    # not take logs with an epsilon floor: remote providers may intentionally
    # assign exact zero support to illegal/pruned candidates, and a no-op layer
    # must never revive that support or perturb their original float values.
    if not any(abs(float(value)) > 0.0 for value in trace.residuals):
        try:
            return list(probabilities), trace
        except TypeError:
            return probabilities, trace
    try:
        values = [float(value) for value in probabilities]
    except (TypeError, ValueError):
        return list(probabilities), _neutral_trace(len(trace.residuals), "probabilities_not_vector")
    if len(values) != len(trace.residuals) or any(not math.isfinite(value) or value < 0.0 for value in values):
        return values, _neutral_trace(len(trace.residuals), "probability_width_or_value_invalid")
    total = sum(values)
    if total <= 0.0:
        return values, _neutral_trace(len(trace.residuals), "probability_mass_invalid")
    supported = [index for index, value in enumerate(values) if value > 0.0]
    if not supported:
        return values, _neutral_trace(len(trace.residuals), "probability_mass_invalid")
    log_scores = [float("-inf")] * len(values)
    for index in supported:
        log_scores[index] = math.log(values[index] / total) + trace.residuals[index]
    maximum = max(log_scores[index] for index in supported)
    exp_scores = [0.0] * len(values)
    for index in supported:
        exp_scores[index] = math.exp(log_scores[index] - maximum)
    normalizer = sum(exp_scores)
    if not math.isfinite(normalizer) or normalizer <= 0.0:
        return values, _neutral_trace(len(trace.residuals), "probability_normalization_invalid")
    return [value / normalizer for value in exp_scores], trace


__all__ = [
    "CHANNEL_NAMES",
    "CONFIG_SCHEMA",
    "ChecklistChannelTrace",
    "DEFAULT_CONFIG_PATH",
    "TurnChecklistConfig",
    "TurnChecklistConfigError",
    "TurnChecklistTrace",
    "apply_turn_checklist_logits",
    "apply_turn_checklist_probabilities",
    "evaluate_turn_checklist",
    "load_turn_checklist_config",
]
