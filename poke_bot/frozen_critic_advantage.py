"""Fail-closed, trainer-only frozen critic advantage materialization.

The sidecar critic is deliberately represented here as plain, immutable Python
data.  This module does not import a policy model, torch, a simulator, RTP, or
any runtime/action-path code.  A caller must precompute the critic outputs,
seal their identity, and then ask this module to validate and broadcast one
complete-action advantage across the selected factorized stages.

Revision 21 owns the only enabled formula::

    (z - V_existing)
        + 0.05 * m1 * (Q_prize^1 - V_prize^1)

The binary ``V_win`` and ``Q_win`` outputs, and prize horizons two and three,
are retained in diagnostics but are intentionally actor-inert.  The disabled
branch returns the supplied legacy mapping before inspecting a cache, so it
can preserve the existing ``z - V_existing`` code path exactly.
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


SCHEMA = "pokebot.frozen-critic-advantage-cache.v2"
CANARY_PRIZE_H1_COEFFICIENT = 0.05
ZERO_ACTOR_COEFFICIENT = 0.0
_LEGACY_TERMINAL_ADVANTAGE_LIMIT = 2.0
_ENABLED_ADVANTAGE_LIMIT = 2.1
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTITY_FIELDS = (
    "contract_sha256",
    "source_sha256",
    "split_sha256",
    "feature_schema_sha256",
    "action_schema_sha256",
    "target_schema_sha256",
    "coefficient_sha256",
    "critic_checkpoint_sha256",
    "policy_checkpoint_sha256",
)


class FrozenCriticValidationError(ValueError):
    """Raised when a frozen critic artifact cannot safely drive an actor unit."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the stable JSON representation used for all cache digests."""
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return a ``sha256:`` digest of a canonical JSON value."""
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_digest(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise FrozenCriticValidationError(
            f"{field} must be a lowercase sha256:<64-hex> digest"
        )
    return value


def _require_nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise FrozenCriticValidationError(f"{field} must be a non-empty string")
    return value


def _finite_float(value: object, field: str, *, lower: float, upper: float) -> float:
    """Convert only a finite real scalar in the stated closed interval."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise FrozenCriticValidationError(f"{field} must be a finite real scalar")
    converted = float(value)
    if not math.isfinite(converted):
        raise FrozenCriticValidationError(f"{field} must be finite")
    if converted < lower or converted > upper:
        raise FrozenCriticValidationError(
            f"{field} must be in [{lower}, {upper}], got {converted}"
        )
    return converted


def _terminal_return(value: object) -> float:
    """Validate the exact observed loss/draw/win return scale for ``z``."""
    converted = _finite_float(
        value,
        "terminal_return",
        lower=-1.0,
        upper=1.0,
    )
    if converted not in (-1.0, 0.0, 1.0):
        raise FrozenCriticValidationError(
            "terminal_return must be one of {-1.0, 0.0, 1.0}, "
            f"got {converted}"
        )
    return converted


def _three_finite_values(
    value: object,
    field: str,
    *,
    lower: float,
    upper: float,
) -> tuple[float, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise FrozenCriticValidationError(f"{field} must contain exactly three values")
    return tuple(
        _finite_float(item, f"{field}[{index}]", lower=lower, upper=upper)
        for index, item in enumerate(value)
    )  # type: ignore[return-value]


def _three_masks(value: object, field: str) -> tuple[bool, bool, bool]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise FrozenCriticValidationError(f"{field} must contain exactly three booleans")
    masks: list[bool] = []
    for index, item in enumerate(value):
        if type(item) is not bool:  # bool subclasses are not valid JSON booleans.
            raise FrozenCriticValidationError(f"{field}[{index}] must be a boolean")
        masks.append(item)
    return tuple(masks)  # type: ignore[return-value]


def _stage_keys(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)) or not value:
        raise FrozenCriticValidationError(f"{field} must be a non-empty sequence")
    result = tuple(_require_nonempty_string(item, f"{field}[{index}]") for index, item in enumerate(value))
    if len(set(result)) != len(result):
        raise FrozenCriticValidationError(f"{field} contains duplicate stage keys")
    return result


@dataclass(frozen=True)
class CriticCacheIdentity:
    """All immutable artifact bindings required before an actor can use a cache."""

    contract_sha256: str
    source_sha256: str
    split_sha256: str
    feature_schema_sha256: str
    action_schema_sha256: str
    target_schema_sha256: str
    coefficient_sha256: str
    critic_checkpoint_sha256: str
    policy_checkpoint_sha256: str

    def __post_init__(self) -> None:
        for field in _IDENTITY_FIELDS:
            object.__setattr__(self, field, _require_digest(getattr(self, field), field))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "CriticCacheIdentity":
        if not isinstance(value, Mapping):
            raise FrozenCriticValidationError("identity must be a mapping")
        observed = set(value)
        required = set(_IDENTITY_FIELDS)
        if observed != required:
            raise FrozenCriticValidationError(
                "identity keys must match exactly; "
                f"missing={sorted(required - observed)!r} extra={sorted(observed - required)!r}"
            )
        return cls(**{field: value[field] for field in _IDENTITY_FIELDS})  # type: ignore[arg-type]

    def as_mapping(self) -> dict[str, str]:
        return {field: getattr(self, field) for field in _IDENTITY_FIELDS}


@dataclass(frozen=True)
class CompleteAction:
    """The actor-unit view of one recorded selected complete action.

    ``alignment_sha256`` must bind the exact public observation, legal order,
    selected indices, action program, and stage order used to form this action.
    It is intentionally supplied by the sealed materializer rather than being
    recomputed from partial runtime data here.

    ``existing_state_value`` is the frozen existing policy value at the
    complete action's public state.  It is deliberately separate from all
    sidecar outputs: Revision 21 retains the draw-safe legacy terminal term.
    """

    action_key: str
    stage_keys: tuple[str, ...]
    alignment_sha256: str
    terminal_return: float
    existing_state_value: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "action_key",
            _require_nonempty_string(self.action_key, "action_key"),
        )
        object.__setattr__(self, "stage_keys", _stage_keys(self.stage_keys, "stage_keys"))
        object.__setattr__(
            self,
            "alignment_sha256",
            _require_digest(self.alignment_sha256, "alignment_sha256"),
        )
        object.__setattr__(
            self,
            "terminal_return",
            _terminal_return(self.terminal_return),
        )
        object.__setattr__(
            self,
            "existing_state_value",
            _finite_float(
                self.existing_state_value,
                "existing_state_value",
                lower=-1.0,
                upper=1.0,
            ),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "CompleteAction":
        if not isinstance(value, Mapping):
            raise FrozenCriticValidationError("complete action must be a mapping")
        required = {
            "action_key",
            "stage_keys",
            "alignment_sha256",
            "terminal_return",
            "existing_state_value",
        }
        observed = set(value)
        if observed != required:
            raise FrozenCriticValidationError(
                "complete action keys must match exactly; "
                f"missing={sorted(required - observed)!r} extra={sorted(observed - required)!r}"
            )
        return cls(
            action_key=value["action_key"],  # type: ignore[arg-type]
            stage_keys=_stage_keys(value["stage_keys"], "stage_keys"),
            alignment_sha256=value["alignment_sha256"],  # type: ignore[arg-type]
            terminal_return=value["terminal_return"],  # type: ignore[arg-type]
            existing_state_value=value["existing_state_value"],  # type: ignore[arg-type]
        )

    def as_mapping(self) -> dict[str, object]:
        return {
            "action_key": self.action_key,
            "stage_keys": list(self.stage_keys),
            "alignment_sha256": self.alignment_sha256,
            "terminal_return": float(self.terminal_return),
            "existing_state_value": float(self.existing_state_value),
        }


@dataclass(frozen=True)
class FrozenCriticPrediction:
    """Precomputed, bounded sidecar outputs for one selected complete action."""

    action_key: str
    stage_keys: tuple[str, ...]
    alignment_sha256: str
    v_win_probability: float
    q_win_probability: float
    v_prize: tuple[float, float, float]
    q_prize: tuple[float, float, float]
    prize_masks: tuple[bool, bool, bool]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "action_key",
            _require_nonempty_string(self.action_key, "action_key"),
        )
        object.__setattr__(self, "stage_keys", _stage_keys(self.stage_keys, "stage_keys"))
        object.__setattr__(
            self,
            "alignment_sha256",
            _require_digest(self.alignment_sha256, "alignment_sha256"),
        )
        object.__setattr__(
            self,
            "v_win_probability",
            _finite_float(
                self.v_win_probability,
                "v_win_probability",
                lower=0.0,
                upper=1.0,
            ),
        )
        object.__setattr__(
            self,
            "q_win_probability",
            _finite_float(
                self.q_win_probability,
                "q_win_probability",
                lower=0.0,
                upper=1.0,
            ),
        )
        object.__setattr__(
            self,
            "v_prize",
            _three_finite_values(self.v_prize, "v_prize", lower=-1.0, upper=1.0),
        )
        object.__setattr__(
            self,
            "q_prize",
            _three_finite_values(self.q_prize, "q_prize", lower=-1.0, upper=1.0),
        )
        object.__setattr__(self, "prize_masks", _three_masks(self.prize_masks, "prize_masks"))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "FrozenCriticPrediction":
        if not isinstance(value, Mapping):
            raise FrozenCriticValidationError("prediction must be a mapping")
        required = {
            "action_key",
            "stage_keys",
            "alignment_sha256",
            "v_win_probability",
            "q_win_probability",
            "v_prize",
            "q_prize",
            "prize_masks",
        }
        observed = set(value)
        if observed != required:
            raise FrozenCriticValidationError(
                "prediction keys must match exactly; "
                f"missing={sorted(required - observed)!r} extra={sorted(observed - required)!r}"
            )
        return cls(
            action_key=value["action_key"],  # type: ignore[arg-type]
            stage_keys=_stage_keys(value["stage_keys"], "stage_keys"),
            alignment_sha256=value["alignment_sha256"],  # type: ignore[arg-type]
            v_win_probability=value["v_win_probability"],  # type: ignore[arg-type]
            q_win_probability=value["q_win_probability"],  # type: ignore[arg-type]
            v_prize=_three_finite_values(value["v_prize"], "v_prize", lower=-1.0, upper=1.0),
            q_prize=_three_finite_values(value["q_prize"], "q_prize", lower=-1.0, upper=1.0),
            prize_masks=_three_masks(value["prize_masks"], "prize_masks"),
        )

    def as_mapping(self) -> dict[str, object]:
        return {
            "action_key": self.action_key,
            "stage_keys": list(self.stage_keys),
            "alignment_sha256": self.alignment_sha256,
            "v_win_probability": float(self.v_win_probability),
            "q_win_probability": float(self.q_win_probability),
            "v_prize": [float(item) for item in self.v_prize],
            "q_prize": [float(item) for item in self.q_prize],
            "prize_masks": list(self.prize_masks),
        }


@dataclass(frozen=True)
class FrozenCriticDiagnostics:
    """Logged values, including intentionally actor-inert outputs."""

    action_key: str
    existing_state_value: float
    legacy_terminal_advantage: float
    v_win_probability: float
    q_win_probability: float
    q_win_minus_v_win_probability: float
    v_prize: tuple[float, float, float]
    q_prize: tuple[float, float, float]
    prize_masks: tuple[bool, bool, bool]
    prize_advantages: tuple[float, float, float]
    actor_coefficients: tuple[float, float, float, float, float]

    def as_mapping(self) -> dict[str, object]:
        """Return JSON-safe diagnostics with zero coefficients made explicit."""
        return {
            "action_key": self.action_key,
            "existing_state_value": self.existing_state_value,
            "legacy_terminal_advantage": self.legacy_terminal_advantage,
            "v_win_probability": self.v_win_probability,
            "q_win_probability": self.q_win_probability,
            "q_win_minus_v_win_probability": self.q_win_minus_v_win_probability,
            "v_prize": list(self.v_prize),
            "q_prize": list(self.q_prize),
            "prize_masks": list(self.prize_masks),
            "prize_advantages": list(self.prize_advantages),
            "actor_coefficients": {
                "v_win": self.actor_coefficients[0],
                "q_win": self.actor_coefficients[1],
                "prize_h1": self.actor_coefficients[2],
                "prize_h2": self.actor_coefficients[3],
                "prize_h3": self.actor_coefficients[4],
            },
        }


@dataclass(frozen=True)
class EnabledAdvantageResult:
    """Validated enabled-canary advantages and their actor-inert diagnostics."""

    advantages_by_stage: Mapping[str, float]
    diagnostics_by_action: Mapping[str, FrozenCriticDiagnostics]


def _artifact_payload(
    identity: CriticCacheIdentity,
    predictions: Sequence[FrozenCriticPrediction],
) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "identity": identity.as_mapping(),
        "predictions": [prediction.as_mapping() for prediction in predictions],
    }


@dataclass(frozen=True)
class FrozenCriticAdvantageCache:
    """Immutable sidecar prediction cache, bound to a sealed actor contract."""

    identity: CriticCacheIdentity
    predictions: tuple[FrozenCriticPrediction, ...]
    payload_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.identity, CriticCacheIdentity):
            raise FrozenCriticValidationError("identity must be CriticCacheIdentity")
        if not isinstance(self.predictions, tuple) or not self.predictions:
            raise FrozenCriticValidationError("predictions must be a non-empty tuple")
        for index, prediction in enumerate(self.predictions):
            if not isinstance(prediction, FrozenCriticPrediction):
                raise FrozenCriticValidationError(
                    f"predictions[{index}] must be FrozenCriticPrediction"
                )
        self._validate_unique_prediction_coverage()
        object.__setattr__(
            self,
            "payload_sha256",
            _require_digest(self.payload_sha256, "payload_sha256"),
        )
        expected_digest = canonical_sha256(_artifact_payload(self.identity, self.predictions))
        if self.payload_sha256 != expected_digest:
            raise FrozenCriticValidationError(
                "critic cache payload digest mismatch: "
                f"expected={expected_digest!r} observed={self.payload_sha256!r}"
            )

    @classmethod
    def from_records(
        cls,
        *,
        identity: CriticCacheIdentity,
        predictions: Sequence[FrozenCriticPrediction],
        expected_payload_sha256: str | None = None,
    ) -> "FrozenCriticAdvantageCache":
        """Build a cache only when its computed digest matches the supplied seal.

        ``expected_payload_sha256`` is optional solely for a freshly generated
        cache before a receipt exists.  Production activation should construct
        via :meth:`from_artifact`, where the digest is mandatory.
        """
        frozen_predictions = tuple(predictions)
        payload = _artifact_payload(identity, frozen_predictions)
        actual = canonical_sha256(payload)
        if expected_payload_sha256 is not None:
            _require_digest(expected_payload_sha256, "expected_payload_sha256")
            if expected_payload_sha256 != actual:
                raise FrozenCriticValidationError(
                    "provided critic cache payload digest does not match records"
                )
        return cls(
            identity=identity,
            predictions=frozen_predictions,
            payload_sha256=actual,
        )

    @classmethod
    def from_artifact(cls, artifact: Mapping[str, object]) -> "FrozenCriticAdvantageCache":
        """Parse an exact, digest-addressed JSON-safe frozen cache artifact."""
        if not isinstance(artifact, Mapping):
            raise FrozenCriticValidationError("critic cache artifact must be a mapping")
        required = {"schema", "identity", "predictions", "payload_sha256"}
        observed = set(artifact)
        if observed != required:
            raise FrozenCriticValidationError(
                "critic cache artifact keys must match exactly; "
                f"missing={sorted(required - observed)!r} extra={sorted(observed - required)!r}"
            )
        if artifact["schema"] != SCHEMA:
            raise FrozenCriticValidationError(
                f"unsupported critic cache schema: {artifact['schema']!r}"
            )
        raw_predictions = artifact["predictions"]
        if not isinstance(raw_predictions, list) or not raw_predictions:
            raise FrozenCriticValidationError("artifact predictions must be a non-empty list")
        identity = CriticCacheIdentity.from_mapping(artifact["identity"])  # type: ignore[arg-type]
        predictions = tuple(
            FrozenCriticPrediction.from_mapping(raw)  # type: ignore[arg-type]
            for raw in raw_predictions
        )
        payload_sha256 = _require_digest(artifact["payload_sha256"], "payload_sha256")
        return cls(
            identity=identity,
            predictions=predictions,
            payload_sha256=payload_sha256,
        )

    def as_artifact(self) -> dict[str, object]:
        """Return the immutable-cache content suitable for a receipt/artifact."""
        payload = _artifact_payload(self.identity, self.predictions)
        return {**payload, "payload_sha256": self.payload_sha256}

    def _validate_unique_prediction_coverage(self) -> None:
        action_keys: set[str] = set()
        stage_keys: set[str] = set()
        for prediction in self.predictions:
            if prediction.action_key in action_keys:
                raise FrozenCriticValidationError(
                    f"duplicate prediction action_key: {prediction.action_key!r}"
                )
            action_keys.add(prediction.action_key)
            duplicate_stage_keys = stage_keys.intersection(prediction.stage_keys)
            if duplicate_stage_keys:
                raise FrozenCriticValidationError(
                    "selected factorized stage key is assigned to more than one "
                    f"complete action: {sorted(duplicate_stage_keys)!r}"
                )
            stage_keys.update(prediction.stage_keys)

    def validate_identity(self, expected_identity: CriticCacheIdentity) -> None:
        """Require every current receipt binding to equal the frozen cache's.

        A digest-shaped identity is not enough: an otherwise valid cache from
        another source window, split, schema, coefficient set, checkpoint, or
        contract must fail closed before it reaches the actor.
        """
        if not isinstance(expected_identity, CriticCacheIdentity):
            raise FrozenCriticValidationError(
                "expected_identity must be CriticCacheIdentity"
            )
        if self.identity != expected_identity:
            expected = expected_identity.as_mapping()
            observed = self.identity.as_mapping()
            mismatches = {
                field: {"expected": expected[field], "observed": observed[field]}
                for field in _IDENTITY_FIELDS
                if expected[field] != observed[field]
            }
            raise FrozenCriticValidationError(
                f"critic cache identity mismatch: {mismatches!r}"
            )

    def validate_payload_digest(self, expected_payload_sha256: str) -> None:
        """Bind this exact prediction cache to the caller's immutable receipt."""
        expected = _require_digest(
            expected_payload_sha256,
            "expected_payload_sha256",
        )
        if self.payload_sha256 != expected:
            raise FrozenCriticValidationError(
                "critic cache payload digest receipt mismatch: "
                f"expected={expected!r} observed={self.payload_sha256!r}"
            )

    def validate_for_actions(
        self, actions: Sequence[CompleteAction]
    ) -> tuple[CompleteAction, ...]:
        """Fail closed unless cache and current actor unit align exactly.

        This simultaneously proves action-level full coverage, no surplus
        prediction rows, exact stage ordering, and action/observation/legal
        alignment-digest equality.
        """
        frozen_actions = tuple(actions)
        if not frozen_actions:
            raise FrozenCriticValidationError("at least one complete action is required")
        expected_by_action: dict[str, CompleteAction] = {}
        all_stage_keys: set[str] = set()
        for index, action in enumerate(frozen_actions):
            if not isinstance(action, CompleteAction):
                raise FrozenCriticValidationError(
                    f"actions[{index}] must be CompleteAction"
                )
            if action.action_key in expected_by_action:
                raise FrozenCriticValidationError(
                    f"duplicate complete action key: {action.action_key!r}"
                )
            duplicate_stage_keys = all_stage_keys.intersection(action.stage_keys)
            if duplicate_stage_keys:
                raise FrozenCriticValidationError(
                    "selected factorized stage key is assigned to more than one "
                    f"current complete action: {sorted(duplicate_stage_keys)!r}"
                )
            expected_by_action[action.action_key] = action
            all_stage_keys.update(action.stage_keys)

        cached_by_action = {prediction.action_key: prediction for prediction in self.predictions}
        expected_keys = set(expected_by_action)
        cached_keys = set(cached_by_action)
        if expected_keys != cached_keys:
            raise FrozenCriticValidationError(
                "critic cache full-coverage mismatch; "
                f"missing={sorted(expected_keys - cached_keys)!r} "
                f"surplus={sorted(cached_keys - expected_keys)!r}"
            )

        for action_key, action in expected_by_action.items():
            prediction = cached_by_action[action_key]
            if prediction.stage_keys != action.stage_keys:
                raise FrozenCriticValidationError(
                    f"factorized stage alignment mismatch for action {action_key!r}: "
                    f"expected={action.stage_keys!r} observed={prediction.stage_keys!r}"
                )
            if prediction.alignment_sha256 != action.alignment_sha256:
                raise FrozenCriticValidationError(
                    f"action/observation/legal alignment digest mismatch for {action_key!r}"
                )
        return frozen_actions

    def materialize_enabled(
        self,
        actions: Sequence[CompleteAction],
        *,
        expected_identity: CriticCacheIdentity,
        expected_payload_sha256: str,
    ) -> EnabledAdvantageResult:
        """Validate and materialize Revision-21 actor advantages.

        The output is immutable.  Each factorized stage of a complete action
        receives the same scalar; diagnostic values are available separately
        so callers can receipt all eight critic outputs without accidentally
        giving actor weight to diagnostic-only terms.
        """
        self.validate_identity(expected_identity)
        self.validate_payload_digest(expected_payload_sha256)
        frozen_actions = self.validate_for_actions(actions)
        cached_by_action = {prediction.action_key: prediction for prediction in self.predictions}
        advantages_by_stage: dict[str, float] = {}
        diagnostics_by_action: dict[str, FrozenCriticDiagnostics] = {}
        for action in frozen_actions:
            prediction = cached_by_action[action.action_key]
            prize_advantages = tuple(
                prediction.q_prize[index] - prediction.v_prize[index]
                for index in range(3)
            )
            for index, prize_advantage in enumerate(prize_advantages):
                _finite_float(
                    prize_advantage,
                    f"prize_advantages[{index}]",
                    lower=-2.0,
                    upper=2.0,
                )
            h1_term = (
                CANARY_PRIZE_H1_COEFFICIENT * prize_advantages[0]
                if prediction.prize_masks[0]
                else 0.0
            )
            legacy_terminal_advantage = (
                action.terminal_return - action.existing_state_value
            )
            _finite_float(
                legacy_terminal_advantage,
                "legacy_terminal_advantage",
                lower=-_LEGACY_TERMINAL_ADVANTAGE_LIMIT,
                upper=_LEGACY_TERMINAL_ADVANTAGE_LIMIT,
            )
            advantage = legacy_terminal_advantage + h1_term
            _finite_float(
                advantage,
                f"enabled advantage for {action.action_key!r}",
                lower=-_ENABLED_ADVANTAGE_LIMIT,
                upper=_ENABLED_ADVANTAGE_LIMIT,
            )
            q_win_minus_v_win_probability = (
                prediction.q_win_probability - prediction.v_win_probability
            )
            _finite_float(
                q_win_minus_v_win_probability,
                "q_win_minus_v_win_probability",
                lower=-1.0,
                upper=1.0,
            )
            diagnostics = FrozenCriticDiagnostics(
                action_key=action.action_key,
                existing_state_value=action.existing_state_value,
                legacy_terminal_advantage=legacy_terminal_advantage,
                v_win_probability=prediction.v_win_probability,
                q_win_probability=prediction.q_win_probability,
                q_win_minus_v_win_probability=q_win_minus_v_win_probability,
                v_prize=prediction.v_prize,
                q_prize=prediction.q_prize,
                prize_masks=prediction.prize_masks,
                prize_advantages=prize_advantages,
                actor_coefficients=(
                    ZERO_ACTOR_COEFFICIENT,
                    ZERO_ACTOR_COEFFICIENT,
                    CANARY_PRIZE_H1_COEFFICIENT,
                    ZERO_ACTOR_COEFFICIENT,
                    ZERO_ACTOR_COEFFICIENT,
                ),
            )
            diagnostics_by_action[action.action_key] = diagnostics
            for stage_key in action.stage_keys:
                advantages_by_stage[stage_key] = advantage
        return EnabledAdvantageResult(
            advantages_by_stage=MappingProxyType(advantages_by_stage),
            diagnostics_by_action=MappingProxyType(diagnostics_by_action),
        )


def resolve_stage_advantages(
    *,
    enabled: bool,
    legacy_advantages_by_stage: Mapping[str, Any],
    cache: FrozenCriticAdvantageCache | None = None,
    actions: Sequence[CompleteAction] = (),
    expected_identity: CriticCacheIdentity | None = None,
    expected_payload_sha256: str | None = None,
) -> Mapping[str, Any] | Mapping[str, float]:
    """Choose disabled legacy or enabled frozen-critic advantages safely.

    The first branch intentionally returns the *supplied object* without
    validating, coercing, copying, indexing, or inspecting the cache.  That is
    the exact legacy ``z - V_existing`` fallback required for a disabled or
    rolled-back canary.  The enabled branch ignores the legacy mapping and
    requires a fully validated frozen cache/action alignment.
    """
    if not enabled:
        return legacy_advantages_by_stage
    if cache is None:
        raise FrozenCriticValidationError("enabled critic advantage requires a cache")
    if expected_identity is None:
        raise FrozenCriticValidationError(
            "enabled critic advantage requires an expected cache identity"
        )
    if expected_payload_sha256 is None:
        raise FrozenCriticValidationError(
            "enabled critic advantage requires an expected cache payload digest"
        )
    return cache.materialize_enabled(
        actions,
        expected_identity=expected_identity,
        expected_payload_sha256=expected_payload_sha256,
    ).advantages_by_stage


__all__ = [
    "CANARY_PRIZE_H1_COEFFICIENT",
    "SCHEMA",
    "ZERO_ACTOR_COEFFICIENT",
    "CompleteAction",
    "CriticCacheIdentity",
    "EnabledAdvantageResult",
    "FrozenCriticAdvantageCache",
    "FrozenCriticDiagnostics",
    "FrozenCriticPrediction",
    "FrozenCriticValidationError",
    "canonical_json_bytes",
    "canonical_sha256",
    "resolve_stage_advantages",
]
