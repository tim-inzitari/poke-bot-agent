"""Receipt-gated, trainer-only Prize-plan-v2 H3 advantage provider.

This module is deliberately inert unless a caller supplies the complete frozen
cache identity *and* the later activation receipt.  It imports neither torch
nor policy/runtime code.  The disabled branch returns the caller's legacy
``z - V_existing`` stage mapping without inspecting any critic artifact.

Revision 23 owns the sole enabled formula::

    (z - V_existing) + 0.025 * m3 * c3 * A_plan_3

``A_plan_3`` is the frozen train-split-scaled ``Q_plan_3 - V_plan_3`` value.
H1, H6, and H12 are retained for diagnostics with actor coefficient zero.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Any, Mapping, Sequence


SCHEMA = "poke_bot.frozen_prize_plan_v2_advantage_cache/v1"
ACTIVATION_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_prize_plan_v2_h3_actor_canary_activation_receipt/v1"
)
H3_COEFFICIENT = 0.025
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_HORIZONS = (1, 3, 6, 12)
_IDENTITY_FIELDS = (
    "contract_sha256",
    "source_binding_sha256",
    "target_set_manifest_sha256",
    "critic_checkpoint_sha256",
    "h3_scale_support_sha256",
    "validation_receipt_sha256",
    "coefficient_configuration_sha256",
    "policy_checkpoint_sha256",
    "activation_receipt_sha256",
)


class FrozenPrizePlanValidationError(ValueError):
    """Raised when an H3 actor cache is not safe to consume."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise FrozenPrizePlanValidationError(
            f"{field} must be a lowercase sha256:<64-hex> digest"
        )
    return value


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise FrozenPrizePlanValidationError(f"{field} must be a non-empty string")
    return value


def _number(value: object, field: str, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise FrozenPrizePlanValidationError(f"{field} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise FrozenPrizePlanValidationError(f"{field} must be finite")
    if result < lower or result > upper:
        raise FrozenPrizePlanValidationError(
            f"{field} must be in [{lower}, {upper}], got {result}"
        )
    return result


def _terminal(value: object) -> float:
    result = _number(value, "terminal_return", -1.0, 1.0)
    if result not in (-1.0, 0.0, 1.0):
        raise FrozenPrizePlanValidationError(
            "terminal_return must be one of {-1.0, 0.0, 1.0}"
        )
    return result


def _stages(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise FrozenPrizePlanValidationError("stage_keys must be a non-empty sequence")
    result = tuple(_text(item, f"stage_keys[{index}]") for index, item in enumerate(value))
    if len(set(result)) != len(result):
        raise FrozenPrizePlanValidationError("stage_keys contains duplicates")
    return result


def _four_values(value: object, field: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise FrozenPrizePlanValidationError(f"{field} must have four horizon values")
    return tuple(
        _number(item, f"{field}[{index}]", -1.0, 1.0)
        for index, item in enumerate(value)
    )  # type: ignore[return-value]


def _four_masks(value: object) -> tuple[bool, bool, bool, bool]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise FrozenPrizePlanValidationError("masks must have four horizon booleans")
    if any(type(item) is not bool for item in value):
        raise FrozenPrizePlanValidationError("masks must contain only booleans")
    return tuple(value)  # type: ignore[return-value]


@dataclass(frozen=True)
class PrizePlanCacheIdentity:
    contract_sha256: str
    source_binding_sha256: str
    target_set_manifest_sha256: str
    critic_checkpoint_sha256: str
    h3_scale_support_sha256: str
    validation_receipt_sha256: str
    coefficient_configuration_sha256: str
    policy_checkpoint_sha256: str
    activation_receipt_sha256: str

    def __post_init__(self) -> None:
        for field in _IDENTITY_FIELDS:
            object.__setattr__(self, field, _digest(getattr(self, field), field))

    def as_mapping(self) -> dict[str, str]:
        return {field: getattr(self, field) for field in _IDENTITY_FIELDS}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "PrizePlanCacheIdentity":
        if not isinstance(value, Mapping) or set(value) != set(_IDENTITY_FIELDS):
            raise FrozenPrizePlanValidationError("identity keys must match exactly")
        return cls(**{field: value[field] for field in _IDENTITY_FIELDS})  # type: ignore[arg-type]


@dataclass(frozen=True)
class PrizePlanCompleteAction:
    action_key: str
    stage_keys: tuple[str, ...]
    alignment_sha256: str
    terminal_return: float
    existing_state_value: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_key", _text(self.action_key, "action_key"))
        object.__setattr__(self, "stage_keys", _stages(self.stage_keys))
        object.__setattr__(self, "alignment_sha256", _digest(self.alignment_sha256, "alignment_sha256"))
        object.__setattr__(self, "terminal_return", _terminal(self.terminal_return))
        object.__setattr__(
            self,
            "existing_state_value",
            _number(self.existing_state_value, "existing_state_value", -1.0, 1.0),
        )


@dataclass(frozen=True)
class FrozenPrizePlanPrediction:
    action_key: str
    stage_keys: tuple[str, ...]
    alignment_sha256: str
    v_plan: tuple[float, float, float, float]
    q_plan: tuple[float, float, float, float]
    masks: tuple[bool, bool, bool, bool]
    scaled_h3_advantage: float
    c3: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_key", _text(self.action_key, "action_key"))
        object.__setattr__(self, "stage_keys", _stages(self.stage_keys))
        object.__setattr__(self, "alignment_sha256", _digest(self.alignment_sha256, "alignment_sha256"))
        object.__setattr__(self, "v_plan", _four_values(self.v_plan, "v_plan"))
        object.__setattr__(self, "q_plan", _four_values(self.q_plan, "q_plan"))
        object.__setattr__(self, "masks", _four_masks(self.masks))
        # This is scaled, not clipped. A generous finite bound catches corrupt artifacts
        # without changing valid AWR advantages.
        object.__setattr__(
            self,
            "scaled_h3_advantage",
            _number(self.scaled_h3_advantage, "scaled_h3_advantage", -100.0, 100.0),
        )
        object.__setattr__(self, "c3", _number(self.c3, "c3", 0.0, 1.0))

    @property
    def raw_plan_advantages(self) -> tuple[float, float, float, float]:
        return tuple(q - v for q, v in zip(self.q_plan, self.v_plan, strict=True))  # type: ignore[return-value]


@dataclass(frozen=True)
class PrizePlanActionDiagnostics:
    legacy_terminal_advantage: float
    raw_plan_advantages: tuple[float, float, float, float]
    scaled_h3_advantage: float
    masks: tuple[bool, bool, bool, bool]
    c3: float
    actor_coefficients: tuple[float, float, float, float]


@dataclass(frozen=True)
class MaterializedPrizePlanAdvantages:
    h3_additive_by_stage: Mapping[str, float]
    diagnostics_by_action: Mapping[str, PrizePlanActionDiagnostics]


class FrozenPrizePlanAdvantageCache:
    """Immutable cache whose enabled use requires a later passing receipt."""

    def __init__(
        self,
        *,
        identity: PrizePlanCacheIdentity,
        predictions: Mapping[str, FrozenPrizePlanPrediction],
        payload_sha256: str,
    ) -> None:
        self.identity = identity
        self.predictions = MappingProxyType(dict(predictions))
        self.payload_sha256 = _digest(payload_sha256, "payload_sha256")

    @classmethod
    def from_records(
        cls,
        *,
        identity: PrizePlanCacheIdentity,
        predictions: Sequence[FrozenPrizePlanPrediction],
        activation_receipt: Mapping[str, object],
    ) -> "FrozenPrizePlanAdvantageCache":
        if activation_receipt.get("schema") != ACTIVATION_RECEIPT_SCHEMA:
            raise FrozenPrizePlanValidationError("activation receipt schema mismatch")
        if activation_receipt.get("activation_eligible") is not True:
            raise FrozenPrizePlanValidationError("activation receipt is not eligible")
        if activation_receipt.get("actor_activation") is not True:
            raise FrozenPrizePlanValidationError("activation receipt does not authorize actor use")
        if activation_receipt.get("safe_boundary") is not True:
            raise FrozenPrizePlanValidationError("activation receipt lacks safe boundary proof")
        receipt_identity = activation_receipt.get("cache_identity_without_activation_receipt_sha256")
        expected_receipt_identity = identity.as_mapping()
        expected_receipt_identity.pop("activation_receipt_sha256")
        if receipt_identity != expected_receipt_identity:
            raise FrozenPrizePlanValidationError("activation receipt identity mismatch")
        receipt_without_digest = dict(activation_receipt)
        observed_receipt_sha = receipt_without_digest.pop("artifact_sha256", None)
        expected_receipt_sha = canonical_sha256(receipt_without_digest)
        if observed_receipt_sha != expected_receipt_sha:
            raise FrozenPrizePlanValidationError("activation receipt digest mismatch")
        if expected_receipt_sha != identity.activation_receipt_sha256:
            raise FrozenPrizePlanValidationError("activation receipt is not identity-bound")

        by_key: dict[str, FrozenPrizePlanPrediction] = {}
        rows: list[dict[str, object]] = []
        for prediction in predictions:
            if prediction.action_key in by_key:
                raise FrozenPrizePlanValidationError("duplicate action prediction")
            by_key[prediction.action_key] = prediction
            rows.append(
                {
                    "action_key": prediction.action_key,
                    "stage_keys": list(prediction.stage_keys),
                    "alignment_sha256": prediction.alignment_sha256,
                    "v_plan": list(prediction.v_plan),
                    "q_plan": list(prediction.q_plan),
                    "masks": list(prediction.masks),
                    "scaled_h3_advantage": prediction.scaled_h3_advantage,
                    "c3": prediction.c3,
                }
            )
        payload = {"schema": SCHEMA, "identity": identity.as_mapping(), "predictions": rows}
        return cls(identity=identity, predictions=by_key, payload_sha256=canonical_sha256(payload))

    def materialize_enabled(
        self,
        actions: Sequence[PrizePlanCompleteAction],
        *,
        expected_identity: PrizePlanCacheIdentity,
        expected_payload_sha256: str,
    ) -> MaterializedPrizePlanAdvantages:
        if self.identity != expected_identity:
            raise FrozenPrizePlanValidationError("cache identity mismatch")
        if self.payload_sha256 != _digest(expected_payload_sha256, "expected_payload_sha256"):
            raise FrozenPrizePlanValidationError("cache payload digest mismatch")
        if not actions or len(actions) != len(self.predictions):
            raise FrozenPrizePlanValidationError("full action coverage mismatch")

        addends: dict[str, float] = {}
        diagnostics: dict[str, PrizePlanActionDiagnostics] = {}
        seen: set[str] = set()
        for action in actions:
            if action.action_key in seen:
                raise FrozenPrizePlanValidationError("duplicate complete action")
            seen.add(action.action_key)
            prediction = self.predictions.get(action.action_key)
            if prediction is None:
                raise FrozenPrizePlanValidationError("full action coverage mismatch")
            if prediction.stage_keys != action.stage_keys:
                raise FrozenPrizePlanValidationError("factorized stage alignment mismatch")
            if prediction.alignment_sha256 != action.alignment_sha256:
                raise FrozenPrizePlanValidationError("alignment digest mismatch")
            legacy = action.terminal_return - action.existing_state_value
            h3_enabled = prediction.masks[1]
            h3_term = H3_COEFFICIENT * prediction.c3 * prediction.scaled_h3_advantage if h3_enabled else 0.0
            if not math.isfinite(h3_term):
                raise FrozenPrizePlanValidationError("materialized H3 addend must be finite")
            for stage in action.stage_keys:
                if stage in addends:
                    raise FrozenPrizePlanValidationError("stage key reused across actions")
                addends[stage] = h3_term
            diagnostics[action.action_key] = PrizePlanActionDiagnostics(
                legacy_terminal_advantage=legacy,
                raw_plan_advantages=prediction.raw_plan_advantages,
                scaled_h3_advantage=prediction.scaled_h3_advantage,
                masks=prediction.masks,
                c3=prediction.c3,
                actor_coefficients=(0.0, H3_COEFFICIENT, 0.0, 0.0),
            )
        if seen != set(self.predictions):
            raise FrozenPrizePlanValidationError("surplus cache prediction")
        return MaterializedPrizePlanAdvantages(
            h3_additive_by_stage=MappingProxyType(addends),
            diagnostics_by_action=MappingProxyType(diagnostics),
        )


def resolve_prize_plan_stage_advantages(
    *,
    enabled: bool,
    legacy_advantages_by_stage: Mapping[str, object],
    cache: FrozenPrizePlanAdvantageCache | None,
    actions: Sequence[PrizePlanCompleteAction],
    expected_identity: PrizePlanCacheIdentity | None = None,
    expected_payload_sha256: str | None = None,
) -> Mapping[str, object] | MaterializedPrizePlanAdvantages:
    """Return exact legacy mapping when disabled; otherwise fail closed."""
    if not enabled:
        return legacy_advantages_by_stage
    if cache is None or expected_identity is None or expected_payload_sha256 is None:
        raise FrozenPrizePlanValidationError("enabled H3 provider requires sealed cache identity")
    return cache.materialize_enabled(
        actions,
        expected_identity=expected_identity,
        expected_payload_sha256=expected_payload_sha256,
    )


@dataclass(frozen=True)
class PortableStageAdvantage:
    """Stable replay-row key used before a dataset has process-local IDs."""

    episode_id: str
    seat: int
    env_step: int
    stage_index: int
    advantage: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_id", _text(self.episode_id, "episode_id"))
        for field in ("seat", "env_step", "stage_index"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise FrozenPrizePlanValidationError(f"{field} must be a nonnegative integer")
        if self.seat not in (0, 1):
            raise FrozenPrizePlanValidationError("seat must be 0 or 1")
        object.__setattr__(self, "advantage", _number(self.advantage, "advantage", -100.0, 100.0))

    @property
    def key(self) -> tuple[str, int, int, int]:
        return (self.episode_id, self.seat, self.env_step, self.stage_index)


def bind_portable_stage_advantages(
    sequences: Sequence[object],
    records: Sequence[PortableStageAdvantage],
) -> dict[tuple[int, int, int], float]:
    """Bind stable receipt rows to one in-memory replay window exactly once.

    Full coverage is mandatory and each complete action must carry one identical
    scalar across every selected factorized stage.  No episode, decision, or
    surplus cache row can be silently ignored.
    """

    portable: dict[tuple[str, int, int, int], float] = {}
    by_action: dict[tuple[str, int, int], set[float]] = {}
    for record in records:
        if record.key in portable:
            raise FrozenPrizePlanValidationError("duplicate portable stage advantage")
        portable[record.key] = record.advantage
        by_action.setdefault(record.key[:3], set()).add(record.advantage)
    if any(len(values) != 1 for values in by_action.values()):
        raise FrozenPrizePlanValidationError(
            "complete-action advantage is not identical across factorized stages"
        )

    bound: dict[tuple[int, int, int], float] = {}
    consumed: set[tuple[str, int, int, int]] = set()
    for sequence in sequences:
        episode_id = _text(getattr(sequence, "episode_id", None), "sequence.episode_id")
        seat = getattr(sequence, "seat", None)
        if isinstance(seat, bool) or not isinstance(seat, int) or seat not in (0, 1):
            raise FrozenPrizePlanValidationError("sequence.seat must be 0 or 1")
        decisions = getattr(sequence, "decisions", None)
        if not isinstance(decisions, list):
            raise FrozenPrizePlanValidationError("sequence.decisions must be a list")
        for decision_index, decision in enumerate(decisions):
            env_step = getattr(decision, "env_step", None)
            stages = getattr(decision, "policy_stages", None)
            if isinstance(env_step, bool) or not isinstance(env_step, int) or env_step < 0:
                raise FrozenPrizePlanValidationError("decision.env_step must be nonnegative")
            if not isinstance(stages, list) or not stages:
                raise FrozenPrizePlanValidationError("decision policy stages are absent")
            action_values: set[float] = set()
            for stage_index in range(len(stages)):
                stable_key = (episode_id, seat, env_step, stage_index)
                if stable_key not in portable:
                    raise FrozenPrizePlanValidationError("portable cache lacks full replay coverage")
                value = portable[stable_key]
                bound[(id(sequence), decision_index, stage_index)] = value
                action_values.add(value)
                consumed.add(stable_key)
            if len(action_values) != 1:
                raise FrozenPrizePlanValidationError(
                    "bound complete-action advantage is not stage-identical"
                )
    if consumed != set(portable):
        raise FrozenPrizePlanValidationError("portable cache contains surplus replay rows")
    return bound
