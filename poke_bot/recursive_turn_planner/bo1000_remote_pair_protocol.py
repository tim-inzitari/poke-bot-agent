"""Offline r207 remote boundary for one immutable BO1000 pair per host.

This module defines the filesystem protocol a separately managed dispatcher and
remote worker can use.  It deliberately contains no network client, service
control, engine implementation, model fallback, or coordinator-side game path.
One request embeds one :class:`BO1000PairEnvelope`; the selected host then uses
the frozen host-local pair controller to run both seat-swapped games in fresh
children.

All authoritative records are create-once ``0444`` JSON.  Leases form an
append-only chain.  An expired, never-started lease may be replaced only when
the caller names that exact lease.  Once a worker publishes an execution-start
receipt, no successor lease is allowed: an expired execution remains unresolved
until its original immutable result appears, preventing blind duplicate play.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from poke_bot.rtp_evaluation_immutable_io import (
    ImmutableEvidenceIOError,
    read_immutable_json_object,
)

from .bo1000_evaluation import (
    R207_CANONICAL_PLANNER_CONFIG_SHA256,
    R207_CONTRACT_SHA256,
    R207_EVALUATION_ID,
    R207_FROZEN_BUNDLE_SHA256,
    R207_FROZEN_CHECKPOINT_SHA256,
    R207_MAX_ACTION_SECONDS,
    R207_MAX_TURN_SECONDS,
    BO1000GameReceipt,
    BO1000GameSpec,
)
from .bo1000_pair_runner import (
    R207_FORBIDDEN_GUIDE_ENV_KEYS,
    R207_GUIDE_ISOLATION_ENV,
    BO1000PairEnvelope,
    BO1000PairRunnerError,
    BO1000PairRunResult,
    HostLocalBO1000PairController,
    apply_r207_no_guide_environment,
    game_receipt_from_payload,
)

REMOTE_PAIR_REQUEST_SCHEMA = "poke_bot.alakazam_mcts_bo1000_remote_pair_request_r207/v1"
HOST_CAPABILITY_BINDING_SCHEMA = (
    "poke_bot.alakazam_mcts_bo1000_host_capability_binding_r207/v1"
)
PAIR_LEASE_SCHEMA = "poke_bot.alakazam_mcts_bo1000_remote_pair_lease_r207/v1"
EXECUTION_START_SCHEMA = (
    "poke_bot.alakazam_mcts_bo1000_remote_pair_execution_start_r207/v1"
)
REMOTE_PAIR_RESULT_SCHEMA = "poke_bot.alakazam_mcts_bo1000_remote_pair_result_r207/v1"

# Frozen pair-runner bytes named by the r207 handoff.  A host binding carrying
# any other implementation is a different protocol and is rejected here.
FROZEN_PAIR_RUNNER_SOURCE_SHA256 = (
    "sha256:50c064756befeb741c22861b1745ba3f3c84e33b9b00977e0003f54d339e5ae1"
)
R207_DECK_CARDS_SHA256 = (
    "sha256:660c1274aac19d88c40fd2bb52187f53dc639d944506760e386f2686b91cc247"
)
HOST_HELLO_PREFLIGHT_SCHEMA = (
    "poke_bot.alakazam_mcts_bo1000_host_hello_preflight_r207/v1"
)
REMOTE_EXECUTION_LIMITS_SCHEMA = (
    "poke_bot.alakazam_mcts_bo1000_remote_execution_limits_r207/v1"
)

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class R207RemotePairProtocolError(ValueError):
    """The remote pair boundary cannot preserve exact immutable evidence."""


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _canonical_sha256(payload: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _require_digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise R207RemotePairProtocolError(f"{name} must be a sha256 digest")
    suffix = value.removeprefix("sha256:")
    if len(suffix) != 64 or any(char not in "0123456789abcdef" for char in suffix):
        raise R207RemotePairProtocolError(f"{name} must be a lowercase sha256 digest")
    return value


def _require_identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise R207RemotePairProtocolError(
            f"{name} must be one bounded ASCII identifier"
        )
    return value


def _require_unix_ns(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise R207RemotePairProtocolError(f"{name} must be an integer >= {minimum}")
    return value


def _require_exact_keys(
    raw: object, *, expected: set[str], label: str
) -> Mapping[str, object]:
    if not isinstance(raw, Mapping) or set(raw) != expected:
        raise R207RemotePairProtocolError(f"{label} schema fields are invalid")
    return raw


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("immutable protocol write made no progress")
        offset += written


def _write_immutable_json(
    path: Path, payload: Mapping[str, object], *, label: str
) -> None:
    encoded = _canonical_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        try:
            material, _ = read_immutable_json_object(path, label)
        except ImmutableEvidenceIOError as exc:
            raise R207RemotePairProtocolError(
                f"existing {label} is unavailable or mutable"
            ) from exc
        if material.payload != encoded:
            raise R207RemotePairProtocolError(
                f"existing {label} differs from create-once payload"
            )
        return
    try:
        _write_all(descriptor, encoded)
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)


def _read_immutable_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        _, payload = read_immutable_json_object(path, label)
    except ImmutableEvidenceIOError as exc:
        raise R207RemotePairProtocolError(f"{label} is unavailable or mutable") from exc
    return payload


@contextlib.contextmanager
def _exclusive_protocol_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class R207HostCapabilityBinding:
    """Exact code, model, deck, config, and host-preflight identity."""

    host_id: str
    source_tree_sha256: str
    evaluation_contract_sha256: str
    pair_runner_source_sha256: str
    checkpoint_sha256: str
    bundle_sha256: str
    deck_cards_sha256: str
    planner_config_sha256: str
    host_capability_receipt_sha256: str
    safe_noninterference_receipt_sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.host_id, name="host_id")
        for name in (
            "source_tree_sha256",
            "evaluation_contract_sha256",
            "pair_runner_source_sha256",
            "checkpoint_sha256",
            "bundle_sha256",
            "deck_cards_sha256",
            "planner_config_sha256",
            "host_capability_receipt_sha256",
            "safe_noninterference_receipt_sha256",
        ):
            _require_digest(getattr(self, name), name=name)
        if self.pair_runner_source_sha256 != FROZEN_PAIR_RUNNER_SOURCE_SHA256:
            raise R207RemotePairProtocolError(
                "host capability does not bind the frozen BO1000 pair runner"
            )
        if self.evaluation_contract_sha256 != R207_CONTRACT_SHA256:
            raise R207RemotePairProtocolError(
                "host capability does not bind the exact r207 evaluation contract"
            )
        if self.checkpoint_sha256 != R207_FROZEN_CHECKPOINT_SHA256:
            raise R207RemotePairProtocolError(
                "host capability does not bind the frozen r195 checkpoint"
            )
        if self.bundle_sha256 != R207_FROZEN_BUNDLE_SHA256:
            raise R207RemotePairProtocolError(
                "host capability does not bind the frozen r195 NO-RTP bundle"
            )
        if self.deck_cards_sha256 != R207_DECK_CARDS_SHA256:
            raise R207RemotePairProtocolError(
                "host capability does not bind the exact r207 deck"
            )
        if self.planner_config_sha256 != R207_CANONICAL_PLANNER_CONFIG_SHA256:
            raise R207RemotePairProtocolError(
                "host capability does not bind the canonical r207 planner config"
            )

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema": HOST_CAPABILITY_BINDING_SCHEMA,
            "host_id": self.host_id,
            "source_tree_sha256": self.source_tree_sha256,
            "evaluation_contract_sha256": self.evaluation_contract_sha256,
            "pair_runner_source_sha256": self.pair_runner_source_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "bundle_sha256": self.bundle_sha256,
            "deck_cards_sha256": self.deck_cards_sha256,
            "planner_config_sha256": self.planner_config_sha256,
            "host_capability_receipt_sha256": self.host_capability_receipt_sha256,
            "safe_noninterference_receipt_sha256": (
                self.safe_noninterference_receipt_sha256
            ),
        }

    @property
    def binding_sha256(self) -> str:
        return _canonical_sha256(self.identity_payload())

    def as_payload(self) -> dict[str, object]:
        return {**self.identity_payload(), "binding_sha256": self.binding_sha256}

    @classmethod
    def from_payload(cls, raw: object) -> R207HostCapabilityBinding:
        material = _require_exact_keys(
            raw,
            expected={
                "schema",
                "host_id",
                "source_tree_sha256",
                "evaluation_contract_sha256",
                "pair_runner_source_sha256",
                "checkpoint_sha256",
                "bundle_sha256",
                "deck_cards_sha256",
                "planner_config_sha256",
                "host_capability_receipt_sha256",
                "safe_noninterference_receipt_sha256",
                "binding_sha256",
            },
            label="host capability binding",
        )
        if material.get("schema") != HOST_CAPABILITY_BINDING_SCHEMA:
            raise R207RemotePairProtocolError("host capability schema is invalid")
        try:
            binding = cls(
                host_id=material["host_id"],  # type: ignore[arg-type]
                source_tree_sha256=material["source_tree_sha256"],  # type: ignore[arg-type]
                evaluation_contract_sha256=material["evaluation_contract_sha256"],  # type: ignore[arg-type]
                pair_runner_source_sha256=material["pair_runner_source_sha256"],  # type: ignore[arg-type]
                checkpoint_sha256=material["checkpoint_sha256"],  # type: ignore[arg-type]
                bundle_sha256=material["bundle_sha256"],  # type: ignore[arg-type]
                deck_cards_sha256=material["deck_cards_sha256"],  # type: ignore[arg-type]
                planner_config_sha256=material["planner_config_sha256"],  # type: ignore[arg-type]
                host_capability_receipt_sha256=material[
                    "host_capability_receipt_sha256"
                ],  # type: ignore[arg-type]
                safe_noninterference_receipt_sha256=material[
                    "safe_noninterference_receipt_sha256"
                ],  # type: ignore[arg-type]
            )
        except (KeyError, TypeError) as exc:
            raise R207RemotePairProtocolError(
                "host capability fields are invalid"
            ) from exc
        if material.get("binding_sha256") != binding.binding_sha256:
            raise R207RemotePairProtocolError("host capability digest is invalid")
        return binding


@dataclass(frozen=True, slots=True)
class R207RemoteExecutionLimits:
    """Durable hard-child bound derived from the exact r207 turn clock."""

    max_game_mcts_turns: int
    child_process_grace_seconds: int
    max_planner_wall_seconds_per_actual_turn: float = R207_MAX_TURN_SECONDS
    max_planner_wall_seconds_before_each_atomic_action: float = R207_MAX_ACTION_SECONDS

    def __post_init__(self) -> None:
        if type(self.max_game_mcts_turns) is not int or self.max_game_mcts_turns < 1:
            raise R207RemotePairProtocolError(
                "max_game_mcts_turns must be a positive integer"
            )
        if (
            type(self.child_process_grace_seconds) is not int
            or self.child_process_grace_seconds < 0
        ):
            raise R207RemotePairProtocolError(
                "child_process_grace_seconds must be a nonnegative integer"
            )
        if (
            type(self.max_planner_wall_seconds_per_actual_turn) is not float
            or self.max_planner_wall_seconds_per_actual_turn != R207_MAX_TURN_SECONDS
            or type(self.max_planner_wall_seconds_before_each_atomic_action)
            is not float
            or self.max_planner_wall_seconds_before_each_atomic_action
            != R207_MAX_ACTION_SECONDS
        ):
            raise R207RemotePairProtocolError(
                "remote execution limits must retain exact r207 20s/5s clocks"
            )

    @property
    def hard_child_timeout_seconds(self) -> float:
        return (
            self.max_game_mcts_turns * self.max_planner_wall_seconds_per_actual_turn
            + self.child_process_grace_seconds
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema": REMOTE_EXECUTION_LIMITS_SCHEMA,
            "max_game_mcts_turns": self.max_game_mcts_turns,
            "child_process_grace_seconds": self.child_process_grace_seconds,
            "max_planner_wall_seconds_per_actual_turn": (
                self.max_planner_wall_seconds_per_actual_turn
            ),
            "max_planner_wall_seconds_before_each_atomic_action": (
                self.max_planner_wall_seconds_before_each_atomic_action
            ),
            "hard_child_timeout_seconds": self.hard_child_timeout_seconds,
            "automatic_extension_authorized": False,
        }

    @property
    def limits_sha256(self) -> str:
        return _canonical_sha256(self.identity_payload())

    def as_payload(self) -> dict[str, object]:
        return {**self.identity_payload(), "limits_sha256": self.limits_sha256}

    @classmethod
    def from_payload(cls, raw: object) -> R207RemoteExecutionLimits:
        material = _require_exact_keys(
            raw,
            expected={
                "schema",
                "max_game_mcts_turns",
                "child_process_grace_seconds",
                "max_planner_wall_seconds_per_actual_turn",
                "max_planner_wall_seconds_before_each_atomic_action",
                "hard_child_timeout_seconds",
                "automatic_extension_authorized",
                "limits_sha256",
            },
            label="remote execution limits",
        )
        if (
            material.get("schema") != REMOTE_EXECUTION_LIMITS_SCHEMA
            or material.get("automatic_extension_authorized") is not False
        ):
            raise R207RemotePairProtocolError("remote execution limits are invalid")
        try:
            limits = cls(
                max_game_mcts_turns=material["max_game_mcts_turns"],  # type: ignore[arg-type]
                child_process_grace_seconds=material["child_process_grace_seconds"],  # type: ignore[arg-type]
                max_planner_wall_seconds_per_actual_turn=material[
                    "max_planner_wall_seconds_per_actual_turn"
                ],  # type: ignore[arg-type]
                max_planner_wall_seconds_before_each_atomic_action=material[
                    "max_planner_wall_seconds_before_each_atomic_action"
                ],  # type: ignore[arg-type]
            )
        except (KeyError, TypeError) as exc:
            raise R207RemotePairProtocolError(
                "remote execution limit fields are invalid"
            ) from exc
        if (
            material.get("hard_child_timeout_seconds")
            != limits.hard_child_timeout_seconds
            or material.get("limits_sha256") != limits.limits_sha256
        ):
            raise R207RemotePairProtocolError(
                "remote execution limit identity is invalid"
            )
        return limits


@dataclass(frozen=True, slots=True)
class R207HostHelloPreflight:
    """Exact host hello accepted before an r207 pair may be submitted."""

    hello_nonce_sha256: str
    host_binding: R207HostCapabilityBinding
    execution_limits: R207RemoteExecutionLimits

    def __post_init__(self) -> None:
        _require_digest(self.hello_nonce_sha256, name="hello_nonce_sha256")
        if not isinstance(self.host_binding, R207HostCapabilityBinding):
            raise R207RemotePairProtocolError("hello host binding type is invalid")
        if not isinstance(self.execution_limits, R207RemoteExecutionLimits):
            raise R207RemotePairProtocolError("hello execution limits type is invalid")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema": HOST_HELLO_PREFLIGHT_SCHEMA,
            "evaluation_id": R207_EVALUATION_ID,
            "hello_nonce_sha256": self.hello_nonce_sha256,
            "host_binding": self.host_binding.as_payload(),
            "execution_limits": self.execution_limits.as_payload(),
            "capabilities": {
                "one_pair_envelope_per_dispatch": True,
                "two_fresh_host_local_children": True,
                "host_local_search_and_inference": True,
                "hard_child_timeout_enforced": True,
                "durable_create_once_records": True,
                "explicit_lease_ownership_and_expiry": True,
                "material_mcts_telemetry_required_for_completed_games": True,
                "per_turn_lan_rpc_authorized": False,
                "per_leaf_lan_rpc_authorized": False,
                "coordinator_local_fallback_authorized": False,
            },
        }

    @property
    def preflight_sha256(self) -> str:
        return _canonical_sha256(self.identity_payload())

    def as_payload(self) -> dict[str, object]:
        return {**self.identity_payload(), "preflight_sha256": self.preflight_sha256}

    @classmethod
    def from_payload(cls, raw: object) -> R207HostHelloPreflight:
        material = _require_exact_keys(
            raw,
            expected={
                "schema",
                "evaluation_id",
                "hello_nonce_sha256",
                "host_binding",
                "execution_limits",
                "capabilities",
                "preflight_sha256",
            },
            label="host hello preflight",
        )
        capabilities = material.get("capabilities")
        if (
            material.get("schema") != HOST_HELLO_PREFLIGHT_SCHEMA
            or material.get("evaluation_id") != R207_EVALUATION_ID
            or not isinstance(capabilities, Mapping)
        ):
            raise R207RemotePairProtocolError("host hello preflight is invalid")
        try:
            hello = cls(
                hello_nonce_sha256=material["hello_nonce_sha256"],  # type: ignore[arg-type]
                host_binding=R207HostCapabilityBinding.from_payload(
                    material["host_binding"]
                ),
                execution_limits=R207RemoteExecutionLimits.from_payload(
                    material["execution_limits"]
                ),
            )
        except (KeyError, TypeError) as exc:
            raise R207RemotePairProtocolError(
                "host hello preflight fields are invalid"
            ) from exc
        if (
            capabilities != hello.identity_payload()["capabilities"]
            or material.get("preflight_sha256") != hello.preflight_sha256
        ):
            raise R207RemotePairProtocolError(
                "host hello preflight identity is invalid"
            )
        return hello


def _execution_contract() -> dict[str, object]:
    return {
        "dispatch_unit": "one_immutable_pair_envelope",
        "games_per_pair": 2,
        "seat_swap_required": True,
        "host_process_scope": "two_fresh_host_local_children",
        "search_scope": "host_local_only",
        "inference_scope": "host_local_only",
        "per_turn_lan_rpc_authorized": False,
        "per_leaf_lan_rpc_authorized": False,
        "coordinator_local_fallback_authorized": False,
    }


def _authority_contract() -> dict[str, object]:
    return {
        "training_eligible": False,
        "training_or_gradient_updates_authorized": False,
        "serving_eligible": False,
        "production_action_authority_enabled": False,
        "selector_change_authorized": False,
        "checkpoint_publication_authorized": False,
        "kaggle_submission_authorized": False,
        "promotion_authorized": False,
        "attempt10_mutation_authorized": False,
    }


@dataclass(frozen=True, slots=True)
class R207RemotePairRequest:
    """One pair-sized dispatch request bound to exactly one preflighted host."""

    dispatch_nonce_sha256: str
    pair_envelope: BO1000PairEnvelope
    host_preflight: R207HostHelloPreflight

    def __post_init__(self) -> None:
        _require_digest(self.dispatch_nonce_sha256, name="dispatch_nonce_sha256")
        if not isinstance(self.pair_envelope, BO1000PairEnvelope):
            raise R207RemotePairProtocolError("pair_envelope type is invalid")
        if not isinstance(self.host_preflight, R207HostHelloPreflight):
            raise R207RemotePairProtocolError("host_preflight type is invalid")
        if self.pair_envelope.evaluation_id != R207_EVALUATION_ID:
            raise R207RemotePairProtocolError(
                "request must use the exact r207 evaluation"
            )
        if self.pair_envelope.execution_host != self.host_binding.host_id:
            raise R207RemotePairProtocolError(
                "pair envelope and host capability name different hosts"
            )
        if (
            self.pair_envelope.checkpoint_sha256 != self.host_binding.checkpoint_sha256
            or self.pair_envelope.bundle_sha256 != self.host_binding.bundle_sha256
        ):
            raise R207RemotePairProtocolError(
                "pair envelope violates the host's frozen-model binding"
            )

    @property
    def host_binding(self) -> R207HostCapabilityBinding:
        return self.host_preflight.host_binding

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema": REMOTE_PAIR_REQUEST_SCHEMA,
            "evaluation_id": R207_EVALUATION_ID,
            "dispatch_nonce_sha256": self.dispatch_nonce_sha256,
            "pair_idempotency_key_sha256": (self.pair_envelope.idempotency_key_sha256),
            "pair_envelope": self.pair_envelope.as_payload(),
            "host_preflight": self.host_preflight.as_payload(),
            "execution_contract": _execution_contract(),
            "authority": _authority_contract(),
        }

    @property
    def request_idempotency_key_sha256(self) -> str:
        return _canonical_sha256(self.identity_payload())

    def as_payload(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "request_idempotency_key_sha256": self.request_idempotency_key_sha256,
        }

    @classmethod
    def from_payload(cls, raw: object) -> R207RemotePairRequest:
        material = _require_exact_keys(
            raw,
            expected={
                "schema",
                "evaluation_id",
                "dispatch_nonce_sha256",
                "pair_idempotency_key_sha256",
                "pair_envelope",
                "host_preflight",
                "execution_contract",
                "authority",
                "request_idempotency_key_sha256",
            },
            label="remote pair request",
        )
        if (
            material.get("schema") != REMOTE_PAIR_REQUEST_SCHEMA
            or material.get("evaluation_id") != R207_EVALUATION_ID
            or material.get("execution_contract") != _execution_contract()
            or material.get("authority") != _authority_contract()
        ):
            raise R207RemotePairProtocolError("remote pair request contract is invalid")
        try:
            request = cls(
                dispatch_nonce_sha256=material["dispatch_nonce_sha256"],  # type: ignore[arg-type]
                pair_envelope=BO1000PairEnvelope.from_payload(
                    material["pair_envelope"]
                ),
                host_preflight=R207HostHelloPreflight.from_payload(
                    material["host_preflight"]
                ),
            )
        except (BO1000PairRunnerError, KeyError, TypeError) as exc:
            raise R207RemotePairProtocolError(
                "remote pair request fields are invalid"
            ) from exc
        if (
            material.get("pair_idempotency_key_sha256")
            != request.pair_envelope.idempotency_key_sha256
            or material.get("request_idempotency_key_sha256")
            != request.request_idempotency_key_sha256
        ):
            raise R207RemotePairProtocolError("remote pair request identity is invalid")
        return request


@dataclass(frozen=True, slots=True)
class R207PairLease:
    request_idempotency_key_sha256: str
    envelope_sha256: str
    host_binding_sha256: str
    lease_sequence: int
    previous_lease_id_sha256: str | None
    lease_nonce_sha256: str
    owner_id: str
    issued_at_unix_ns: int
    expires_at_unix_ns: int

    def __post_init__(self) -> None:
        for name in (
            "request_idempotency_key_sha256",
            "envelope_sha256",
            "host_binding_sha256",
            "lease_nonce_sha256",
        ):
            _require_digest(getattr(self, name), name=name)
        if self.previous_lease_id_sha256 is not None:
            _require_digest(
                self.previous_lease_id_sha256, name="previous_lease_id_sha256"
            )
        if type(self.lease_sequence) is not int or self.lease_sequence < 0:
            raise R207RemotePairProtocolError(
                "lease_sequence must be a nonnegative integer"
            )
        _require_identifier(self.owner_id, name="owner_id")
        _require_unix_ns(self.issued_at_unix_ns, name="issued_at_unix_ns")
        _require_unix_ns(self.expires_at_unix_ns, name="expires_at_unix_ns")
        if self.expires_at_unix_ns <= self.issued_at_unix_ns:
            raise R207RemotePairProtocolError("lease expiry must follow issuance")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema": PAIR_LEASE_SCHEMA,
            "request_idempotency_key_sha256": self.request_idempotency_key_sha256,
            "envelope_sha256": self.envelope_sha256,
            "host_binding_sha256": self.host_binding_sha256,
            "lease_sequence": self.lease_sequence,
            "previous_lease_id_sha256": self.previous_lease_id_sha256,
            "lease_nonce_sha256": self.lease_nonce_sha256,
            "owner_id": self.owner_id,
            "issued_at_unix_ns": self.issued_at_unix_ns,
            "expires_at_unix_ns": self.expires_at_unix_ns,
        }

    @property
    def lease_id_sha256(self) -> str:
        return _canonical_sha256(self.identity_payload())

    def as_payload(self) -> dict[str, object]:
        return {**self.identity_payload(), "lease_id_sha256": self.lease_id_sha256}

    @classmethod
    def from_payload(cls, raw: object) -> R207PairLease:
        material = _require_exact_keys(
            raw,
            expected={
                "schema",
                "request_idempotency_key_sha256",
                "envelope_sha256",
                "host_binding_sha256",
                "lease_sequence",
                "previous_lease_id_sha256",
                "lease_nonce_sha256",
                "owner_id",
                "issued_at_unix_ns",
                "expires_at_unix_ns",
                "lease_id_sha256",
            },
            label="remote pair lease",
        )
        if material.get("schema") != PAIR_LEASE_SCHEMA:
            raise R207RemotePairProtocolError("remote pair lease schema is invalid")
        try:
            lease = cls(
                request_idempotency_key_sha256=material[
                    "request_idempotency_key_sha256"
                ],  # type: ignore[arg-type]
                envelope_sha256=material["envelope_sha256"],  # type: ignore[arg-type]
                host_binding_sha256=material["host_binding_sha256"],  # type: ignore[arg-type]
                lease_sequence=material["lease_sequence"],  # type: ignore[arg-type]
                previous_lease_id_sha256=material["previous_lease_id_sha256"],  # type: ignore[arg-type]
                lease_nonce_sha256=material["lease_nonce_sha256"],  # type: ignore[arg-type]
                owner_id=material["owner_id"],  # type: ignore[arg-type]
                issued_at_unix_ns=material["issued_at_unix_ns"],  # type: ignore[arg-type]
                expires_at_unix_ns=material["expires_at_unix_ns"],  # type: ignore[arg-type]
            )
        except (KeyError, TypeError) as exc:
            raise R207RemotePairProtocolError(
                "remote pair lease fields are invalid"
            ) from exc
        if material.get("lease_id_sha256") != lease.lease_id_sha256:
            raise R207RemotePairProtocolError("remote pair lease identity is invalid")
        return lease


@dataclass(frozen=True, slots=True)
class R207ExecutionStart:
    request_idempotency_key_sha256: str
    envelope_sha256: str
    host_binding_sha256: str
    lease_id_sha256: str
    lease_owner_id: str
    started_at_unix_ns: int

    def __post_init__(self) -> None:
        for name in (
            "request_idempotency_key_sha256",
            "envelope_sha256",
            "host_binding_sha256",
            "lease_id_sha256",
        ):
            _require_digest(getattr(self, name), name=name)
        _require_identifier(self.lease_owner_id, name="lease_owner_id")
        _require_unix_ns(self.started_at_unix_ns, name="started_at_unix_ns")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema": EXECUTION_START_SCHEMA,
            "request_idempotency_key_sha256": self.request_idempotency_key_sha256,
            "envelope_sha256": self.envelope_sha256,
            "host_binding_sha256": self.host_binding_sha256,
            "lease_id_sha256": self.lease_id_sha256,
            "lease_owner_id": self.lease_owner_id,
            "started_at_unix_ns": self.started_at_unix_ns,
            "execution": _execution_contract(),
        }

    @property
    def execution_start_sha256(self) -> str:
        return _canonical_sha256(self.identity_payload())

    def as_payload(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "execution_start_sha256": self.execution_start_sha256,
        }

    @classmethod
    def from_payload(cls, raw: object) -> R207ExecutionStart:
        material = _require_exact_keys(
            raw,
            expected={
                "schema",
                "request_idempotency_key_sha256",
                "envelope_sha256",
                "host_binding_sha256",
                "lease_id_sha256",
                "lease_owner_id",
                "started_at_unix_ns",
                "execution",
                "execution_start_sha256",
            },
            label="remote pair execution start",
        )
        if (
            material.get("schema") != EXECUTION_START_SCHEMA
            or material.get("execution") != _execution_contract()
        ):
            raise R207RemotePairProtocolError(
                "remote pair execution-start contract is invalid"
            )
        try:
            start = cls(
                request_idempotency_key_sha256=material[
                    "request_idempotency_key_sha256"
                ],  # type: ignore[arg-type]
                envelope_sha256=material["envelope_sha256"],  # type: ignore[arg-type]
                host_binding_sha256=material["host_binding_sha256"],  # type: ignore[arg-type]
                lease_id_sha256=material["lease_id_sha256"],  # type: ignore[arg-type]
                lease_owner_id=material["lease_owner_id"],  # type: ignore[arg-type]
                started_at_unix_ns=material["started_at_unix_ns"],  # type: ignore[arg-type]
            )
        except (KeyError, TypeError) as exc:
            raise R207RemotePairProtocolError(
                "remote pair execution-start fields are invalid"
            ) from exc
        if material.get("execution_start_sha256") != start.execution_start_sha256:
            raise R207RemotePairProtocolError(
                "remote pair execution-start identity is invalid"
            )
        return start


@dataclass(frozen=True, slots=True)
class R207RemotePairResultReceipt:
    request_idempotency_key_sha256: str
    dispatch_nonce_sha256: str
    envelope_sha256: str
    host_binding_sha256: str
    host_binding: R207HostCapabilityBinding
    lease_id_sha256: str
    lease_owner_id: str
    execution_start_sha256: str
    terminal_status: Literal["complete", "failed_closed"]
    pair_receipt_canonical_sha256: str
    pair_receipt_file_sha256: str
    game_receipt_file_sha256s: tuple[str, str]

    def __post_init__(self) -> None:
        for name in (
            "request_idempotency_key_sha256",
            "dispatch_nonce_sha256",
            "envelope_sha256",
            "host_binding_sha256",
            "lease_id_sha256",
            "execution_start_sha256",
            "pair_receipt_canonical_sha256",
            "pair_receipt_file_sha256",
        ):
            _require_digest(getattr(self, name), name=name)
        _require_identifier(self.lease_owner_id, name="lease_owner_id")
        if not isinstance(self.host_binding, R207HostCapabilityBinding):
            raise R207RemotePairProtocolError("remote result host binding is invalid")
        if self.host_binding_sha256 != self.host_binding.binding_sha256:
            raise R207RemotePairProtocolError(
                "remote result host binding digest does not match its payload"
            )
        if self.terminal_status not in {"complete", "failed_closed"}:
            raise R207RemotePairProtocolError("remote result status is not terminal")
        if (
            not isinstance(self.game_receipt_file_sha256s, tuple)
            or len(self.game_receipt_file_sha256s) != 2
        ):
            raise R207RemotePairProtocolError(
                "remote result must bind exactly two game receipt files"
            )
        for digest in self.game_receipt_file_sha256s:
            _require_digest(digest, name="game_receipt_file_sha256")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema": REMOTE_PAIR_RESULT_SCHEMA,
            "request_idempotency_key_sha256": self.request_idempotency_key_sha256,
            "dispatch_nonce_sha256": self.dispatch_nonce_sha256,
            "envelope_sha256": self.envelope_sha256,
            "host_binding_sha256": self.host_binding_sha256,
            "lease_id_sha256": self.lease_id_sha256,
            "lease_owner_id": self.lease_owner_id,
            "execution_start_sha256": self.execution_start_sha256,
            "terminal_status": self.terminal_status,
            "pair_receipt_canonical_sha256": self.pair_receipt_canonical_sha256,
            "pair_receipt_file_sha256": self.pair_receipt_file_sha256,
            "game_receipt_file_sha256s": list(self.game_receipt_file_sha256s),
            "host_binding": self.host_binding.as_payload(),
            "authority": _authority_contract(),
        }

    @property
    def result_receipt_sha256(self) -> str:
        return _canonical_sha256(self.identity_payload())

    def as_payload(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "result_receipt_sha256": self.result_receipt_sha256,
        }

    @classmethod
    def from_payload(cls, raw: object) -> R207RemotePairResultReceipt:
        material = _require_exact_keys(
            raw,
            expected={
                "schema",
                "request_idempotency_key_sha256",
                "dispatch_nonce_sha256",
                "envelope_sha256",
                "host_binding_sha256",
                "lease_id_sha256",
                "lease_owner_id",
                "execution_start_sha256",
                "terminal_status",
                "pair_receipt_canonical_sha256",
                "pair_receipt_file_sha256",
                "game_receipt_file_sha256s",
                "host_binding",
                "authority",
                "result_receipt_sha256",
            },
            label="remote pair result receipt",
        )
        if (
            material.get("schema") != REMOTE_PAIR_RESULT_SCHEMA
            or material.get("authority") != _authority_contract()
        ):
            raise R207RemotePairProtocolError("remote pair result contract is invalid")
        game_hashes = material.get("game_receipt_file_sha256s")
        if not isinstance(game_hashes, list):
            raise R207RemotePairProtocolError(
                "remote pair result game receipt hashes are invalid"
            )
        try:
            receipt = cls(
                request_idempotency_key_sha256=material[
                    "request_idempotency_key_sha256"
                ],  # type: ignore[arg-type]
                dispatch_nonce_sha256=material["dispatch_nonce_sha256"],  # type: ignore[arg-type]
                envelope_sha256=material["envelope_sha256"],  # type: ignore[arg-type]
                host_binding_sha256=material["host_binding_sha256"],  # type: ignore[arg-type]
                host_binding=R207HostCapabilityBinding.from_payload(
                    material["host_binding"]
                ),
                lease_id_sha256=material["lease_id_sha256"],  # type: ignore[arg-type]
                lease_owner_id=material["lease_owner_id"],  # type: ignore[arg-type]
                execution_start_sha256=material["execution_start_sha256"],  # type: ignore[arg-type]
                terminal_status=material["terminal_status"],  # type: ignore[arg-type]
                pair_receipt_canonical_sha256=material["pair_receipt_canonical_sha256"],  # type: ignore[arg-type]
                pair_receipt_file_sha256=material["pair_receipt_file_sha256"],  # type: ignore[arg-type]
                game_receipt_file_sha256s=tuple(game_hashes),  # type: ignore[arg-type]
            )
        except (KeyError, TypeError) as exc:
            raise R207RemotePairProtocolError(
                "remote pair result fields are invalid"
            ) from exc
        if material.get("result_receipt_sha256") != receipt.result_receipt_sha256:
            raise R207RemotePairProtocolError("remote pair result identity is invalid")
        return receipt


RemotePairState = Literal[
    "unsubmitted",
    "pending",
    "leased",
    "lease_expired",
    "executing",
    "execution_lease_expired_unresolved",
    "complete",
    "failed_closed",
]


@dataclass(frozen=True, slots=True)
class R207RemotePairStatus:
    request_idempotency_key_sha256: str
    pair_id: str
    state: RemotePairState
    lease_id_sha256: str | None
    lease_owner_id: str | None
    lease_expires_at_unix_ns: int | None
    execution_start_sha256: str | None
    result_receipt_sha256: str | None


@dataclass(frozen=True, slots=True)
class R207RetrievedPair:
    request: R207RemotePairRequest
    lease: R207PairLease
    result_receipt: R207RemotePairResultReceipt
    pair_result: BO1000PairRunResult


class R207HostLocalGameProcess(Protocol):
    """A local game process that proves the request's hard timeout identity."""

    @property
    def execution_limits_sha256(self) -> str:
        """Return the exact hello-bound child timeout identity."""

    def run_game(
        self,
        *,
        request_path: Path,
        request: Mapping[str, object],
        envelope: BO1000PairEnvelope,
        spec: BO1000GameSpec,
    ) -> Mapping[str, object]:
        """Run exactly one game in a fresh host-local process."""


R207CommandBuilder = Callable[[Path, BO1000GameSpec], Sequence[str]]


@dataclass(frozen=True, slots=True)
class R207BoundedHostLocalSubprocessGameRunner:
    """Fresh-process runner with the hello-bound hard child timeout."""

    command_builder: R207CommandBuilder
    execution_limits: R207RemoteExecutionLimits
    cwd: Path | None = None
    extra_env: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.execution_limits, R207RemoteExecutionLimits):
            raise R207RemotePairProtocolError(
                "bounded runner execution limits are invalid"
            )
        forbidden = (
            {
                "POKEBOT_REMOTE_ENDPOINT",
                "POKEBOT_REMOTE_ENDPOINTS",
                "POKEBOT_REMOTE_HOST",
                "POKEBOT_REMOTE_PORT",
                "POKEBOT_REMOTE_DISPATCH",
                "BO1000_EXECUTION_SCOPE",
                "BO1000_REMOTE_DISPATCH_AUTHORIZED",
                "BO1000_HARD_CHILD_TIMEOUT_SECONDS",
            }
            | set(R207_FORBIDDEN_GUIDE_ENV_KEYS)
            | set(R207_GUIDE_ISOLATION_ENV)
        )
        if self.extra_env is not None and forbidden.intersection(self.extra_env):
            raise R207RemotePairProtocolError(
                "bounded runner environment may not override transport safety"
            )

    @property
    def execution_limits_sha256(self) -> str:
        return self.execution_limits.limits_sha256

    def run_game(
        self,
        *,
        request_path: Path,
        request: Mapping[str, object],
        envelope: BO1000PairEnvelope,
        spec: BO1000GameSpec,
    ) -> Mapping[str, object]:
        del request
        command = tuple(self.command_builder(request_path, spec))
        if not command or any(
            not isinstance(part, str) or not part for part in command
        ):
            raise BO1000PairRunnerError(
                "bounded host-local game command must be nonempty strings"
            )
        env = dict(os.environ)
        for key in (
            "POKEBOT_REMOTE_ENDPOINT",
            "POKEBOT_REMOTE_ENDPOINTS",
            "POKEBOT_REMOTE_HOST",
            "POKEBOT_REMOTE_PORT",
            "POKEBOT_REMOTE_DISPATCH",
        ):
            env.pop(key, None)
        apply_r207_no_guide_environment(env)
        if self.extra_env is not None:
            env.update({str(key): str(value) for key, value in self.extra_env.items()})
        env.update(
            {
                "BO1000_EXECUTION_SCOPE": "host_local_fresh_process",
                "BO1000_REMOTE_DISPATCH_AUTHORIZED": "0",
                "BO1000_PAIR_ID": spec.pair_id,
                "BO1000_GAME_NONCE_SHA256": spec.game_nonce_sha256,
                "BO1000_HARD_CHILD_TIMEOUT_SECONDS": str(
                    self.execution_limits.hard_child_timeout_seconds
                ),
            }
        )
        try:
            completed = subprocess.run(
                command,
                check=False,
                cwd=os.fspath(self.cwd or request_path.parent),
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=self.execution_limits.hard_child_timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return {
                "game_nonce_sha256": spec.game_nonce_sha256,
                "pair_id": spec.pair_id,
                "game_index": spec.game_index,
                "mcts_seat": spec.mcts_seat,
                "no_rtp_seat": spec.no_rtp_seat,
                "pair_rng_snapshot_sha256": envelope.pair_rng_snapshot_sha256,
                "deck_order_rng_sha256": envelope.deck_order_rng_sha256,
                "checkpoint_sha256": envelope.checkpoint_sha256,
                "bundle_sha256": envelope.bundle_sha256,
                "terminal_status": "failed_closed",
                "winner_seat": None,
                "illegal_action_count": 0,
                "forfeit_count": 0,
                "crash_count": 0,
                "timeout_count": 1,
                "mcts_turns": [],
            }
        if completed.returncode != 0:
            raise BO1000PairRunnerError("bounded host-local game process failed closed")
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise BO1000PairRunnerError(
                "bounded host-local game process did not emit one JSON receipt"
            ) from exc
        if not isinstance(result, Mapping):
            raise BO1000PairRunnerError(
                "bounded host-local game process emitted non-object JSON"
            )
        return result


class R207RemotePairStore:
    """Append-only host-side request, lease, status, and result store."""

    def __init__(self, root: str | Path, *, execution_host: str):
        base = Path(root).expanduser()
        if not base.is_absolute():
            base = Path.cwd() / base
        self._root = base.resolve()
        self._execution_host = _require_identifier(
            execution_host, name="execution_host"
        )

    @property
    def root(self) -> Path:
        return self._root

    @property
    def execution_host(self) -> str:
        return self._execution_host

    def pair_directory(self, request: R207RemotePairRequest) -> Path:
        self._require_host(request)
        envelope = request.pair_envelope
        return (
            self.root
            / "r207-remote-pairs"
            / self.execution_host
            / envelope.pair_id
            / envelope.pair_nonce_sha256.removeprefix("sha256:")
        )

    def request_path(self, request: R207RemotePairRequest) -> Path:
        return self.pair_directory(request) / "request.json"

    def preflight_path(self, preflight: R207HostHelloPreflight) -> Path:
        if preflight.host_binding.host_id != self.execution_host:
            raise R207RemotePairProtocolError(
                "host hello does not match this remote pair store"
            )
        return (
            self.root
            / "r207-host-preflights"
            / self.execution_host
            / f"{preflight.preflight_sha256.removeprefix('sha256:')}.json"
        )

    def result_path(self, request: R207RemotePairRequest) -> Path:
        return self.pair_directory(request) / "result-receipt.json"

    def execution_start_path(self, request: R207RemotePairRequest) -> Path:
        return self.pair_directory(request) / "execution-start.json"

    def _lock_path(self, request: R207RemotePairRequest) -> Path:
        return self.pair_directory(request) / ".protocol.lock"

    def _leases_directory(self, request: R207RemotePairRequest) -> Path:
        return self.pair_directory(request) / "leases"

    def _lease_path(self, request: R207RemotePairRequest, lease: R207PairLease) -> Path:
        return self._leases_directory(request) / (
            f"{lease.lease_sequence:08d}-"
            f"{lease.lease_id_sha256.removeprefix('sha256:')}.json"
        )

    def _require_host(self, request: R207RemotePairRequest) -> None:
        if (
            request.host_binding.host_id != self.execution_host
            or request.pair_envelope.execution_host != self.execution_host
        ):
            raise R207RemotePairProtocolError(
                "request host does not match this remote pair store"
            )

    def submit(self, request: R207RemotePairRequest) -> Path:
        """Create the pair request once; exact retries reuse the same bytes."""

        self._require_host(request)
        self.register_host_preflight(request.host_preflight)
        with _exclusive_protocol_lock(self._lock_path(request)):
            path = self.request_path(request)
            _write_immutable_json(
                path, request.as_payload(), label="remote pair request"
            )
            if self._load_request(request) != request:
                raise R207RemotePairProtocolError(
                    "stored remote pair request differs from submitted request"
                )
            return path

    def register_host_preflight(self, preflight: R207HostHelloPreflight) -> Path:
        """Persist the exact source/model/config hello before dispatch."""

        path = self.preflight_path(preflight)
        _write_immutable_json(
            path, preflight.as_payload(), label="r207 host hello preflight"
        )
        loaded = R207HostHelloPreflight.from_payload(
            _read_immutable_json(path, label="r207 host hello preflight")
        )
        if loaded != preflight:
            raise R207RemotePairProtocolError(
                "stored host hello differs from requested preflight"
            )
        return path

    def _load_host_preflight(
        self, preflight: R207HostHelloPreflight
    ) -> R207HostHelloPreflight:
        path = self.preflight_path(preflight)
        if not path.exists():
            raise R207RemotePairProtocolError("r207 host hello preflight is absent")
        loaded = R207HostHelloPreflight.from_payload(
            _read_immutable_json(path, label="r207 host hello preflight")
        )
        if loaded != preflight:
            raise R207RemotePairProtocolError(
                "stored host hello does not match the pair request"
            )
        return loaded

    def _load_request(self, request: R207RemotePairRequest) -> R207RemotePairRequest:
        self._load_host_preflight(request.host_preflight)
        path = self.request_path(request)
        if not path.exists():
            raise R207RemotePairProtocolError("remote pair request is not submitted")
        loaded = R207RemotePairRequest.from_payload(
            _read_immutable_json(path, label="remote pair request")
        )
        if loaded != request:
            raise R207RemotePairProtocolError(
                "stored request does not match requested pair identity"
            )
        return loaded

    def _load_leases(self, request: R207RemotePairRequest) -> tuple[R207PairLease, ...]:
        directory = self._leases_directory(request)
        if not directory.exists():
            return ()
        paths = sorted(directory.glob("*.json"))
        leases: list[R207PairLease] = []
        previous: R207PairLease | None = None
        for path in paths:
            lease = R207PairLease.from_payload(
                _read_immutable_json(path, label="remote pair lease")
            )
            expected_sequence = len(leases)
            expected_previous = None if previous is None else previous.lease_id_sha256
            if (
                lease.lease_sequence != expected_sequence
                or lease.previous_lease_id_sha256 != expected_previous
                or lease.request_idempotency_key_sha256
                != request.request_idempotency_key_sha256
                or lease.envelope_sha256 != request.pair_envelope.envelope_sha256
                or lease.host_binding_sha256 != request.host_binding.binding_sha256
                or path != self._lease_path(request, lease)
            ):
                raise R207RemotePairProtocolError(
                    "remote pair lease chain is crossed or noncontiguous"
                )
            leases.append(lease)
            previous = lease
        return tuple(leases)

    def _load_execution_start(
        self, request: R207RemotePairRequest
    ) -> R207ExecutionStart | None:
        path = self.execution_start_path(request)
        if not path.exists():
            return None
        start = R207ExecutionStart.from_payload(
            _read_immutable_json(path, label="remote pair execution start")
        )
        if (
            start.request_idempotency_key_sha256
            != request.request_idempotency_key_sha256
            or start.envelope_sha256 != request.pair_envelope.envelope_sha256
            or start.host_binding_sha256 != request.host_binding.binding_sha256
        ):
            raise R207RemotePairProtocolError(
                "execution-start receipt crosses remote pair identity"
            )
        return start

    def _load_result(
        self, request: R207RemotePairRequest
    ) -> R207RemotePairResultReceipt | None:
        path = self.result_path(request)
        if not path.exists():
            return None
        result = R207RemotePairResultReceipt.from_payload(
            _read_immutable_json(path, label="remote pair result receipt")
        )
        if (
            result.request_idempotency_key_sha256
            != request.request_idempotency_key_sha256
            or result.dispatch_nonce_sha256 != request.dispatch_nonce_sha256
            or result.envelope_sha256 != request.pair_envelope.envelope_sha256
            or result.host_binding_sha256 != request.host_binding.binding_sha256
            or result.host_binding != request.host_binding
        ):
            raise R207RemotePairProtocolError(
                "remote result receipt crosses request identity"
            )
        return result

    def lookup(
        self, request: R207RemotePairRequest, *, observed_at_unix_ns: int
    ) -> R207RemotePairStatus:
        """Derive durable state without executing, renewing, or falling back."""

        now = _require_unix_ns(observed_at_unix_ns, name="observed_at_unix_ns")
        self._require_host(request)
        if not self.request_path(request).exists():
            return R207RemotePairStatus(
                request_idempotency_key_sha256=request.request_idempotency_key_sha256,
                pair_id=request.pair_envelope.pair_id,
                state="unsubmitted",
                lease_id_sha256=None,
                lease_owner_id=None,
                lease_expires_at_unix_ns=None,
                execution_start_sha256=None,
                result_receipt_sha256=None,
            )
        self._load_request(request)
        leases = self._load_leases(request)
        latest = leases[-1] if leases else None
        start = self._load_execution_start(request)
        result = self._load_result(request)
        if result is not None:
            if latest is None or start is None:
                raise R207RemotePairProtocolError(
                    "terminal result lacks lease or execution-start evidence"
                )
            self._validate_terminal_chain(latest, start, result)
            state: RemotePairState = result.terminal_status
        elif start is not None:
            if latest is None or start.lease_id_sha256 != latest.lease_id_sha256:
                raise R207RemotePairProtocolError(
                    "execution-start receipt does not bind the latest lease"
                )
            state = (
                "executing"
                if now < latest.expires_at_unix_ns
                else "execution_lease_expired_unresolved"
            )
        elif latest is None:
            state = "pending"
        else:
            state = "leased" if now < latest.expires_at_unix_ns else "lease_expired"
        return R207RemotePairStatus(
            request_idempotency_key_sha256=request.request_idempotency_key_sha256,
            pair_id=request.pair_envelope.pair_id,
            state=state,
            lease_id_sha256=None if latest is None else latest.lease_id_sha256,
            lease_owner_id=None if latest is None else latest.owner_id,
            lease_expires_at_unix_ns=(
                None if latest is None else latest.expires_at_unix_ns
            ),
            execution_start_sha256=(
                None if start is None else start.execution_start_sha256
            ),
            result_receipt_sha256=(
                None if result is None else result.result_receipt_sha256
            ),
        )

    def acquire_lease(
        self,
        request: R207RemotePairRequest,
        *,
        owner_id: str,
        lease_nonce_sha256: str,
        issued_at_unix_ns: int,
        lease_duration_ns: int,
        expected_expired_lease_id_sha256: str | None = None,
    ) -> R207PairLease:
        """Append one explicit lease, never superseding started execution."""

        owner = _require_identifier(owner_id, name="owner_id")
        nonce = _require_digest(lease_nonce_sha256, name="lease_nonce_sha256")
        issued = _require_unix_ns(issued_at_unix_ns, name="issued_at_unix_ns")
        duration = _require_unix_ns(
            lease_duration_ns, name="lease_duration_ns", minimum=1
        )
        if expected_expired_lease_id_sha256 is not None:
            _require_digest(
                expected_expired_lease_id_sha256,
                name="expected_expired_lease_id_sha256",
            )
        with _exclusive_protocol_lock(self._lock_path(request)):
            self._load_request(request)
            if self._load_result(request) is not None:
                raise R207RemotePairProtocolError("remote pair is already terminal")
            leases = self._load_leases(request)
            latest = leases[-1] if leases else None
            start = self._load_execution_start(request)
            if latest is not None:
                retry = R207PairLease(
                    request_idempotency_key_sha256=(
                        request.request_idempotency_key_sha256
                    ),
                    envelope_sha256=request.pair_envelope.envelope_sha256,
                    host_binding_sha256=request.host_binding.binding_sha256,
                    lease_sequence=latest.lease_sequence,
                    previous_lease_id_sha256=latest.previous_lease_id_sha256,
                    lease_nonce_sha256=nonce,
                    owner_id=owner,
                    issued_at_unix_ns=issued,
                    expires_at_unix_ns=issued + duration,
                )
                if retry == latest:
                    return latest
                if start is not None:
                    raise R207RemotePairProtocolError(
                        "execution has started; successor lease is forbidden"
                    )
                if issued < latest.expires_at_unix_ns:
                    raise R207RemotePairProtocolError(
                        "latest remote pair lease is still active"
                    )
                if expected_expired_lease_id_sha256 != latest.lease_id_sha256:
                    raise R207RemotePairProtocolError(
                        "expired lease takeover requires its exact lease identity"
                    )
            elif expected_expired_lease_id_sha256 is not None:
                raise R207RemotePairProtocolError(
                    "first lease cannot acknowledge an expired predecessor"
                )
            lease = R207PairLease(
                request_idempotency_key_sha256=request.request_idempotency_key_sha256,
                envelope_sha256=request.pair_envelope.envelope_sha256,
                host_binding_sha256=request.host_binding.binding_sha256,
                lease_sequence=0 if latest is None else latest.lease_sequence + 1,
                previous_lease_id_sha256=(
                    None if latest is None else latest.lease_id_sha256
                ),
                lease_nonce_sha256=nonce,
                owner_id=owner,
                issued_at_unix_ns=issued,
                expires_at_unix_ns=issued + duration,
            )
            _write_immutable_json(
                self._lease_path(request, lease),
                lease.as_payload(),
                label="remote pair lease",
            )
            if self._load_leases(request)[-1] != lease:
                raise R207RemotePairProtocolError("new lease was not durably appended")
            return lease

    def _begin_execution(
        self,
        request: R207RemotePairRequest,
        lease: R207PairLease,
        *,
        observed_at_unix_ns: int,
    ) -> R207ExecutionStart | None:
        now = _require_unix_ns(observed_at_unix_ns, name="observed_at_unix_ns")
        with _exclusive_protocol_lock(self._lock_path(request)):
            self._load_request(request)
            if self._load_result(request) is not None:
                return None
            leases = self._load_leases(request)
            if not leases or leases[-1] != lease:
                raise R207RemotePairProtocolError(
                    "worker does not own the latest remote pair lease"
                )
            if now < lease.issued_at_unix_ns or now >= lease.expires_at_unix_ns:
                raise R207RemotePairProtocolError(
                    "worker cannot start outside its lease interval"
                )
            existing = self._load_execution_start(request)
            if existing is not None:
                raise R207RemotePairProtocolError(
                    "remote pair execution already started; lookup only"
                )
            start = R207ExecutionStart(
                request_idempotency_key_sha256=request.request_idempotency_key_sha256,
                envelope_sha256=request.pair_envelope.envelope_sha256,
                host_binding_sha256=request.host_binding.binding_sha256,
                lease_id_sha256=lease.lease_id_sha256,
                lease_owner_id=lease.owner_id,
                started_at_unix_ns=now,
            )
            _write_immutable_json(
                self.execution_start_path(request),
                start.as_payload(),
                label="remote pair execution start",
            )
            return start

    @staticmethod
    def _validate_terminal_chain(
        lease: R207PairLease,
        start: R207ExecutionStart,
        result: R207RemotePairResultReceipt,
    ) -> None:
        if (
            start.lease_id_sha256 != lease.lease_id_sha256
            or start.lease_owner_id != lease.owner_id
            or result.lease_id_sha256 != lease.lease_id_sha256
            or result.lease_owner_id != lease.owner_id
            or result.execution_start_sha256 != start.execution_start_sha256
        ):
            raise R207RemotePairProtocolError(
                "remote pair terminal lease ownership chain is invalid"
            )

    def _pair_evidence(
        self,
        request: R207RemotePairRequest,
        pair_controller: HostLocalBO1000PairController,
    ) -> tuple[
        BO1000PairRunResult,
        str,
        str,
        tuple[str, str],
    ]:
        if pair_controller.execution_host != self.execution_host:
            raise R207RemotePairProtocolError(
                "pair controller is not host-local to this remote store"
            )
        envelope = request.pair_envelope
        try:
            status = pair_controller.status(envelope)
        except BO1000PairRunnerError as exc:
            raise R207RemotePairProtocolError(
                "host-local pair evidence is invalid"
            ) from exc
        if (
            status.status not in {"complete", "failed_closed"}
            or status.observed_game_receipts != 2
            or status.terminal_pair_receipt_sha256 is None
        ):
            raise R207RemotePairProtocolError(
                "host-local pair lacks a terminal immutable receipt"
            )
        pair_dir = pair_controller.pair_directory(envelope)
        pair_path = pair_dir / "pair-receipt.json"
        try:
            pair_material, pair_payload = read_immutable_json_object(
                pair_path, "host-local pair receipt"
            )
        except ImmutableEvidenceIOError as exc:
            raise R207RemotePairProtocolError(
                "host-local pair receipt is unavailable or mutable"
            ) from exc
        if pair_payload.get("canonical_sha256") != status.terminal_pair_receipt_sha256:
            raise R207RemotePairProtocolError(
                "host-local pair receipt canonical identity is invalid"
            )
        pair_game_rows = pair_payload.get("game_receipts")
        if not isinstance(pair_game_rows, list) or len(pair_game_rows) != 2:
            raise R207RemotePairProtocolError(
                "host-local pair receipt does not bind exactly two games"
            )
        receipts: list[BO1000GameReceipt] = []
        receipt_hashes: list[str] = []
        for row_index, spec in enumerate(
            sorted(envelope.game_specs, key=lambda item: item.game_index)
        ):
            path = (
                pair_dir
                / "game-receipts"
                / (
                    f"{spec.game_index}-"
                    f"{spec.game_nonce_sha256.removeprefix('sha256:')}.json"
                )
            )
            try:
                material, payload = read_immutable_json_object(
                    path, "host-local game receipt"
                )
                receipt = game_receipt_from_payload(payload)
            except (ImmutableEvidenceIOError, BO1000PairRunnerError) as exc:
                raise R207RemotePairProtocolError(
                    "host-local game receipt is unavailable or invalid"
                ) from exc
            pair_row = pair_game_rows[row_index]
            if (
                not isinstance(pair_row, Mapping)
                or pair_row.get("game_index") != spec.game_index
                or pair_row.get("game_nonce_sha256") != spec.game_nonce_sha256
                or pair_row.get("receipt_payload_sha256") != material.sha256
                or pair_row.get("terminal_status") != receipt.terminal_status
            ):
                raise R207RemotePairProtocolError(
                    "host-local game receipt is not bound by the pair receipt"
                )
            if (
                receipt.game_nonce_sha256 != spec.game_nonce_sha256
                or receipt.pair_id != envelope.pair_id
                or receipt.game_index != spec.game_index
                or receipt.mcts_seat != spec.mcts_seat
                or receipt.no_rtp_seat != spec.no_rtp_seat
                or receipt.pair_rng_snapshot_sha256 != envelope.pair_rng_snapshot_sha256
                or receipt.deck_order_rng_sha256 != envelope.deck_order_rng_sha256
                or receipt.checkpoint_sha256 != request.host_binding.checkpoint_sha256
                or receipt.bundle_sha256 != request.host_binding.bundle_sha256
            ):
                raise R207RemotePairProtocolError(
                    "host-local game receipt crosses an exact request binding"
                )
            if receipt.terminal_status == "completed":
                if not receipt.mcts_turns or not any(
                    turn.actions_dispatched == 1
                    and turn.simulator_transitions_seen > 0
                    and turn.decision_nodes_expanded > 0
                    and turn.frozen_policy_prior_batches > 0
                    and turn.frozen_policy_prior_evaluations > 0
                    for turn in receipt.mcts_turns
                ):
                    raise R207RemotePairProtocolError(
                        "completed game has no material simulator-backed MCTS telemetry"
                    )
                if any(
                    turn.config_sha256 != request.host_binding.planner_config_sha256
                    for turn in receipt.mcts_turns
                ):
                    raise R207RemotePairProtocolError(
                        "game telemetry violates the hello-bound planner config"
                    )
            receipts.append(receipt)
            receipt_hashes.append(material.sha256)
        pair_result = BO1000PairRunResult(
            envelope=envelope,
            status=status,
            game_receipts=tuple(receipts),
        )
        return (
            pair_result,
            status.terminal_pair_receipt_sha256,
            pair_material.sha256,
            (receipt_hashes[0], receipt_hashes[1]),
        )

    def _seal_result(
        self,
        request: R207RemotePairRequest,
        lease: R207PairLease,
        start: R207ExecutionStart,
        pair_controller: HostLocalBO1000PairController,
    ) -> R207RemotePairResultReceipt:
        pair_result, canonical_hash, pair_file_hash, game_hashes = self._pair_evidence(
            request, pair_controller
        )
        result = R207RemotePairResultReceipt(
            request_idempotency_key_sha256=request.request_idempotency_key_sha256,
            dispatch_nonce_sha256=request.dispatch_nonce_sha256,
            envelope_sha256=request.pair_envelope.envelope_sha256,
            host_binding_sha256=request.host_binding.binding_sha256,
            host_binding=request.host_binding,
            lease_id_sha256=lease.lease_id_sha256,
            lease_owner_id=lease.owner_id,
            execution_start_sha256=start.execution_start_sha256,
            terminal_status=pair_result.status.status,  # type: ignore[arg-type]
            pair_receipt_canonical_sha256=canonical_hash,
            pair_receipt_file_sha256=pair_file_hash,
            game_receipt_file_sha256s=game_hashes,
        )
        with _exclusive_protocol_lock(self._lock_path(request)):
            self._load_request(request)
            leases = self._load_leases(request)
            observed_start = self._load_execution_start(request)
            if not leases or leases[-1] != lease or observed_start != start:
                raise R207RemotePairProtocolError(
                    "lease ownership changed before result publication"
                )
            _write_immutable_json(
                self.result_path(request),
                result.as_payload(),
                label="remote pair result receipt",
            )
            stored = self._load_result(request)
            if stored != result:
                raise R207RemotePairProtocolError(
                    "stored remote pair result differs from execution result"
                )
        return result

    def retrieve(
        self,
        request: R207RemotePairRequest,
        pair_controller: HostLocalBO1000PairController,
    ) -> R207RetrievedPair:
        """Return only fully revalidated terminal evidence; never run a game."""

        self._load_request(request)
        leases = self._load_leases(request)
        start = self._load_execution_start(request)
        result = self._load_result(request)
        if not leases or start is None or result is None:
            raise R207RemotePairProtocolError(
                "remote pair result is incomplete; retrieval fails closed"
            )
        lease = leases[-1]
        self._validate_terminal_chain(lease, start, result)
        pair_result, canonical_hash, pair_file_hash, game_hashes = self._pair_evidence(
            request, pair_controller
        )
        if (
            pair_result.status.status != result.terminal_status
            or canonical_hash != result.pair_receipt_canonical_sha256
            or pair_file_hash != result.pair_receipt_file_sha256
            or game_hashes != result.game_receipt_file_sha256s
        ):
            raise R207RemotePairProtocolError(
                "retrieved host-local evidence does not match remote result receipt"
            )
        return R207RetrievedPair(
            request=request,
            lease=lease,
            result_receipt=result,
            pair_result=pair_result,
        )


class R207RemotePairWorkerBoundary:
    """Execute one owned pair lease through the frozen host-local controller."""

    def __init__(
        self,
        store: R207RemotePairStore,
        pair_controller: HostLocalBO1000PairController,
    ):
        if store.execution_host != pair_controller.execution_host:
            raise R207RemotePairProtocolError(
                "worker store and pair controller must name the same host"
            )
        self._store = store
        self._pair_controller = pair_controller

    def execute_lease(
        self,
        request: R207RemotePairRequest,
        lease: R207PairLease,
        runner: R207HostLocalGameProcess,
        *,
        observed_at_unix_ns: int,
    ) -> R207RetrievedPair:
        """Start exactly once, run both local children, seal, and retrieve."""

        if (
            getattr(runner, "execution_limits_sha256", None)
            != request.host_preflight.execution_limits.limits_sha256
        ):
            raise R207RemotePairProtocolError(
                "worker runner does not enforce the hello-bound child timeout"
            )

        start = self._store._begin_execution(
            request, lease, observed_at_unix_ns=observed_at_unix_ns
        )
        if start is None:
            return self._store.retrieve(request, self._pair_controller)
        self._pair_controller.run_pair(request.pair_envelope, runner)
        self._store._seal_result(request, lease, start, self._pair_controller)
        return self._store.retrieve(request, self._pair_controller)


__all__ = [
    "EXECUTION_START_SCHEMA",
    "FROZEN_PAIR_RUNNER_SOURCE_SHA256",
    "HOST_CAPABILITY_BINDING_SCHEMA",
    "HOST_HELLO_PREFLIGHT_SCHEMA",
    "PAIR_LEASE_SCHEMA",
    "R207_DECK_CARDS_SHA256",
    "R207_FROZEN_BUNDLE_SHA256",
    "R207_FROZEN_CHECKPOINT_SHA256",
    "REMOTE_EXECUTION_LIMITS_SCHEMA",
    "REMOTE_PAIR_REQUEST_SCHEMA",
    "REMOTE_PAIR_RESULT_SCHEMA",
    "R207BoundedHostLocalSubprocessGameRunner",
    "R207ExecutionStart",
    "R207HostCapabilityBinding",
    "R207HostHelloPreflight",
    "R207HostLocalGameProcess",
    "R207PairLease",
    "R207RemoteExecutionLimits",
    "R207RemotePairProtocolError",
    "R207RemotePairRequest",
    "R207RemotePairResultReceipt",
    "R207RemotePairStatus",
    "R207RemotePairStore",
    "R207RemotePairWorkerBoundary",
    "R207RetrievedPair",
]
