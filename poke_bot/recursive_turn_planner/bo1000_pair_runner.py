"""Host-local, fresh-process BO1000 pair controller.

This is deliberately only an execution-envelope and receipt boundary.  It
does not load a model, implement simulator/MCTS mechanics, contact a remote
worker, or grant action authority.  A later, separately preflighted local game
command receives one immutable request file per scheduled game and must emit a
``BO1000GameReceipt`` JSON object on stdout.

The boundary is pair-first: both exact seat-swapped games share one immutable
pair envelope, RNG bindings, frozen-model identity, host identity, and
idempotency key.  Each game command is a new local subprocess.  Evidence files
are create-once and ``0444`` so a retry either reuses byte-identical evidence
or fails closed; it never replaces an observed game.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol

from poke_bot.rtp_evaluation_immutable_io import (
    ImmutableEvidenceIOError,
    read_immutable_json_object,
)

from .bo1000_evaluation import (
    BO1000_PAIR_COUNT,
    CONTROL_ARM,
    MCTS_ARM,
    R207_EVALUATION_ID,
    R207_FROZEN_BUNDLE_SHA256,
    R207_FROZEN_CHECKPOINT_SHA256,
    BO1000EvidenceError,
    BO1000GameReceipt,
    BO1000GameSpec,
    MCTSTurnTelemetry,
)

PAIR_ENVELOPE_SCHEMA = "poke_bot.alakazam_mcts_bo1000_pair_envelope/v1"
GAME_REQUEST_SCHEMA = "poke_bot.alakazam_mcts_bo1000_host_local_game_request/v1"
PAIR_RECEIPT_SCHEMA = "poke_bot.alakazam_mcts_bo1000_pair_receipt/v1"
DEFAULT_EVALUATION_ID = R207_EVALUATION_ID
R207_R195_MATCHUP_TREE_SHA256 = (
    "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049"
)

# The r207 mirror compares the unchanged frozen r195 direct policy with that
# same policy plus MCTS.  Guide2Vec and the submission guide-logit rule are
# separate experiments (r212 and r206 respectively) and must never leak into
# either r207 arm through an inherited process environment.
R207_FORBIDDEN_GUIDE_ENV_KEYS = (
    "POKEBOT_CURRENT_DECK_GUIDE",
    "POKEBOT_CURRENT_DECK_GUIDE_TARGETS",
    "POKEBOT_ALAKAZAM_GUIDE_TARGETS",
    "POKEBOT_GUIDE2VEC_CHECKPOINT",
    "POKEBOT_GUIDE2VEC_ENABLED",
    "POKEBOT_GUIDE_LOGIT_CONFIG",
    "POKEBOT_GUIDE_LOGIT_WEIGHT",
    "POKEBOT_GUIDE_LINEAR_TRANSFORM",
)
R207_FORBIDDEN_LEGACY_RTP_ENV_KEYS = (
    "POKEBOT_RTP_CHECKPOINT",
    "POKEBOT_RTP_SIDELOAD",
    "POKEBOT_RTP_ENABLED",
    "POKEBOT_RECURSIVE_TURN_PLANNER",
    "POKEBOT_RECURSIVE_TURN_PLANNER_ENABLED",
)
R207_GUIDE_ISOLATION_ENV = {
    "BO1000_GUIDE2VEC_ENABLED": "0",
    "BO1000_GUIDE_LOGIT_TRANSFORM_ENABLED": "0",
    "BO1000_GUIDE_LINEAR_TRANSFORM_ENABLED": "0",
    "BO1000_BASE_POLICY_TRANSFORM": "frozen_r195_identity",
    "BO1000_MATCHUP_ADAPTER_ENABLED": "1",
    "BO1000_MATCHUP_TREE_SHA256": R207_R195_MATCHUP_TREE_SHA256,
}


def apply_r207_no_guide_environment(env: dict[str, str]) -> None:
    """Remove guide inputs and stamp the exact no-guide r207 process mode."""

    for key in R207_FORBIDDEN_GUIDE_ENV_KEYS + R207_FORBIDDEN_LEGACY_RTP_ENV_KEYS:
        env.pop(key, None)
    env.update(R207_GUIDE_ISOLATION_ENV)


class BO1000PairRunnerError(ValueError):
    """Raised when the local pair boundary cannot preserve immutable evidence."""


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
        raise BO1000PairRunnerError(f"{name} must be a sha256 digest")
    suffix = value[7:]
    if len(suffix) != 64 or any(char not in "0123456789abcdef" for char in suffix):
        raise BO1000PairRunnerError(f"{name} must be a lowercase sha256 digest")
    return value


def _require_nonempty_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BO1000PairRunnerError(f"{name} must be nonempty text")
    return value


def _require_safe_component(value: object, *, name: str) -> str:
    text = _require_nonempty_text(value, name=name)
    if Path(text).name != text or text in {".", ".."}:
        raise BO1000PairRunnerError(f"{name} must be one safe path component")
    return text


def _game_spec_from_payload(raw: object) -> BO1000GameSpec:
    if not isinstance(raw, Mapping):
        raise BO1000PairRunnerError("game spec payload must be an object")
    expected = set(BO1000GameSpec.__dataclass_fields__)
    if set(raw) != expected:
        raise BO1000PairRunnerError("game spec payload has unexpected fields")
    try:
        return BO1000GameSpec(**dict(raw))
    except (BO1000EvidenceError, TypeError) as exc:
        raise BO1000PairRunnerError("game spec payload is invalid") from exc


def _turn_from_payload(raw: object) -> MCTSTurnTelemetry:
    if not isinstance(raw, Mapping):
        raise BO1000PairRunnerError("MCTS turn payload must be an object")
    expected = set(MCTSTurnTelemetry.__dataclass_fields__)
    if set(raw) != expected:
        raise BO1000PairRunnerError("MCTS turn payload has unexpected fields")
    values = dict(raw)
    turn_key = values.get("turn_key")
    if isinstance(turn_key, list):
        values["turn_key"] = tuple(turn_key)
    try:
        return MCTSTurnTelemetry(**values)
    except (BO1000EvidenceError, TypeError) as exc:
        raise BO1000PairRunnerError("MCTS turn payload is invalid") from exc


def game_receipt_from_payload(raw: object) -> BO1000GameReceipt:
    """Parse an engine-independent JSON result into the canonical receipt type."""

    if not isinstance(raw, Mapping):
        raise BO1000PairRunnerError("game receipt output must be a JSON object")
    expected = set(BO1000GameReceipt.__dataclass_fields__)
    if set(raw) != expected:
        raise BO1000PairRunnerError("game receipt output has unexpected fields")
    values = dict(raw)
    turns = values.get("mcts_turns")
    if not isinstance(turns, list):
        raise BO1000PairRunnerError("game receipt mcts_turns must be a list")
    values["mcts_turns"] = tuple(_turn_from_payload(turn) for turn in turns)
    try:
        return BO1000GameReceipt(**values)
    except (BO1000EvidenceError, TypeError) as exc:
        raise BO1000PairRunnerError("game receipt output is invalid") from exc


def _receipt_payload(receipt: BO1000GameReceipt) -> dict[str, object]:
    return asdict(receipt)


@dataclass(frozen=True, slots=True)
class BO1000PairEnvelope:
    """Immutable local execution input for exactly one seat-swapped pair."""

    evaluation_id: str
    pair_index: int
    pair_id: str
    pair_nonce_sha256: str
    checkpoint_sha256: str
    bundle_sha256: str
    pair_rng_snapshot_sha256: str
    deck_order_rng_sha256: str
    execution_host: str
    game_specs: tuple[BO1000GameSpec, BO1000GameSpec]
    experimental_arm: str = MCTS_ARM
    control_arm: str = CONTROL_ARM

    def __post_init__(self) -> None:
        if self.evaluation_id != DEFAULT_EVALUATION_ID:
            raise BO1000PairRunnerError(
                "evaluation_id must be the exact r207 evaluation"
            )
        if type(self.pair_index) is not int or self.pair_index not in range(
            BO1000_PAIR_COUNT
        ):
            raise BO1000PairRunnerError("pair_index must be an exact r207 pair index")
        _require_safe_component(self.pair_id, name="pair_id")
        _require_digest(self.pair_nonce_sha256, name="pair_nonce_sha256")
        expected_pair_id = (
            f"r207-pair-{self.pair_index:06d}-"
            f"{self.pair_nonce_sha256.removeprefix('sha256:')[:12]}"
        )
        if self.pair_id != expected_pair_id:
            raise BO1000PairRunnerError("pair_id must bind the exact r207 pair nonce")
        _require_digest(self.checkpoint_sha256, name="checkpoint_sha256")
        _require_digest(self.bundle_sha256, name="bundle_sha256")
        if self.checkpoint_sha256 != R207_FROZEN_CHECKPOINT_SHA256:
            raise BO1000PairRunnerError(
                "pair must bind the exact original r195 no-RTP checkpoint"
            )
        if self.bundle_sha256 != R207_FROZEN_BUNDLE_SHA256:
            raise BO1000PairRunnerError(
                "pair must bind the exact original r195 no-RTP bundle"
            )
        _require_digest(self.pair_rng_snapshot_sha256, name="pair_rng_snapshot_sha256")
        _require_digest(self.deck_order_rng_sha256, name="deck_order_rng_sha256")
        _require_nonempty_text(self.execution_host, name="execution_host")
        if self.experimental_arm != MCTS_ARM or self.control_arm != CONTROL_ARM:
            raise BO1000PairRunnerError(
                "pair arms must be the exact r207 arm identities"
            )
        if (
            not isinstance(self.game_specs, tuple)
            or len(self.game_specs) != 2
            or any(not isinstance(spec, BO1000GameSpec) for spec in self.game_specs)
        ):
            raise BO1000PairRunnerError(
                "game_specs must contain exactly two game specs"
            )
        if sorted((spec.game_index, spec.mcts_seat) for spec in self.game_specs) != [
            (0, 0),
            (1, 1),
        ]:
            raise BO1000PairRunnerError("pair games must be exact seat-swapped games")
        for spec in self.game_specs:
            if (
                spec.pair_index != self.pair_index
                or spec.pair_id != self.pair_id
                or spec.pair_nonce_sha256 != self.pair_nonce_sha256
            ):
                raise BO1000PairRunnerError("game spec does not belong to this pair")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema": PAIR_ENVELOPE_SCHEMA,
            "evaluation_id": self.evaluation_id,
            "pair_index": self.pair_index,
            "pair_id": self.pair_id,
            "pair_nonce_sha256": self.pair_nonce_sha256,
            "frozen_model": {
                "checkpoint_sha256": self.checkpoint_sha256,
                "bundle_sha256": self.bundle_sha256,
            },
            "rng": {
                "pair_rng_snapshot_sha256": self.pair_rng_snapshot_sha256,
                "deck_order_rng_sha256": self.deck_order_rng_sha256,
            },
            "execution": {
                "scope": "host_local_fresh_process",
                "host": self.execution_host,
                "remote_dispatch_authorized": False,
                "fresh_process_required_per_game": True,
            },
            "arms": {
                "experimental": self.experimental_arm,
                "control": self.control_arm,
            },
            "policy_transforms": {
                "frozen_base_policy": "r195_identity",
                "guide2vec_enabled": False,
                "guide_logit_transform_enabled": False,
                "guide_linear_transform_enabled": False,
            },
            "matchup_adapter": {
                "matchup_tree_sha256": R207_R195_MATCHUP_TREE_SHA256,
                "enabled": True,
                "trained": True,
                "frozen": True,
                "same_exact_runtime_required_for_both_arms": True,
            },
            "game_specs": [spec.as_payload() for spec in self.game_specs],
        }

    @property
    def idempotency_key_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema": "poke_bot.alakazam_mcts_bo1000_pair_idempotency/v1",
                "evaluation_id": self.evaluation_id,
                "pair_nonce_sha256": self.pair_nonce_sha256,
                "pair_index": self.pair_index,
                "pair_id": self.pair_id,
                "checkpoint_sha256": self.checkpoint_sha256,
                "bundle_sha256": self.bundle_sha256,
                "matchup_tree_sha256": R207_R195_MATCHUP_TREE_SHA256,
                "matchup_adapter_enabled": True,
                "matchup_adapter_trained": True,
                "matchup_adapter_frozen": True,
                "pair_rng_snapshot_sha256": self.pair_rng_snapshot_sha256,
                "deck_order_rng_sha256": self.deck_order_rng_sha256,
                "execution_host": self.execution_host,
                "experimental_arm": self.experimental_arm,
                "control_arm": self.control_arm,
                "game_nonces_sha256": [
                    spec.game_nonce_sha256
                    for spec in sorted(
                        self.game_specs, key=lambda item: item.game_index
                    )
                ],
            }
        )

    @property
    def envelope_sha256(self) -> str:
        return _canonical_sha256(self.identity_payload())

    def as_payload(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "idempotency_key_sha256": self.idempotency_key_sha256,
            "envelope_sha256": self.envelope_sha256,
        }

    @classmethod
    def from_payload(cls, raw: object) -> BO1000PairEnvelope:
        if not isinstance(raw, Mapping):
            raise BO1000PairRunnerError("pair envelope must be a JSON object")
        expected = {
            "schema",
            "evaluation_id",
            "pair_index",
            "pair_id",
            "pair_nonce_sha256",
            "frozen_model",
            "rng",
            "execution",
            "arms",
                "policy_transforms",
                "matchup_adapter",
                "game_specs",
            "idempotency_key_sha256",
            "envelope_sha256",
        }
        if set(raw) != expected or raw.get("schema") != PAIR_ENVELOPE_SCHEMA:
            raise BO1000PairRunnerError("pair envelope schema is invalid")
        frozen_model = raw.get("frozen_model")
        rng = raw.get("rng")
        execution = raw.get("execution")
        arms = raw.get("arms")
        policy_transforms = raw.get("policy_transforms")
        matchup_adapter = raw.get("matchup_adapter")
        game_specs = raw.get("game_specs")
        if not all(
            isinstance(value, Mapping)
            for value in (
                frozen_model,
                rng,
                execution,
                arms,
                policy_transforms,
                matchup_adapter,
            )
        ) or not isinstance(game_specs, list):
            raise BO1000PairRunnerError("pair envelope sections are invalid")
        if set(frozen_model) != {"checkpoint_sha256", "bundle_sha256"}:
            raise BO1000PairRunnerError("pair envelope frozen-model binding is invalid")
        if set(rng) != {"pair_rng_snapshot_sha256", "deck_order_rng_sha256"}:
            raise BO1000PairRunnerError("pair envelope RNG binding is invalid")
        if set(execution) != {
            "scope",
            "host",
            "remote_dispatch_authorized",
            "fresh_process_required_per_game",
        } or (
            execution.get("scope") != "host_local_fresh_process"
            or execution.get("remote_dispatch_authorized") is not False
            or execution.get("fresh_process_required_per_game") is not True
        ):
            raise BO1000PairRunnerError("pair envelope execution scope is invalid")
        if set(arms) != {"experimental", "control"}:
            raise BO1000PairRunnerError("pair envelope arms are invalid")
        if dict(policy_transforms) != {
            "frozen_base_policy": "r195_identity",
            "guide2vec_enabled": False,
            "guide_logit_transform_enabled": False,
            "guide_linear_transform_enabled": False,
        }:
            raise BO1000PairRunnerError(
                "pair envelope must bind the exact no-guide policy transform"
            )
        if dict(matchup_adapter) != {
            "matchup_tree_sha256": R207_R195_MATCHUP_TREE_SHA256,
            "enabled": True,
            "trained": True,
            "frozen": True,
            "same_exact_runtime_required_for_both_arms": True,
        }:
            raise BO1000PairRunnerError(
                "pair envelope must bind exact r195 matchup adapter ON for both arms"
            )
        try:
            envelope = cls(
                evaluation_id=raw["evaluation_id"],
                pair_index=raw["pair_index"],
                pair_id=raw["pair_id"],
                pair_nonce_sha256=raw["pair_nonce_sha256"],
                checkpoint_sha256=frozen_model["checkpoint_sha256"],
                bundle_sha256=frozen_model["bundle_sha256"],
                pair_rng_snapshot_sha256=rng["pair_rng_snapshot_sha256"],
                deck_order_rng_sha256=rng["deck_order_rng_sha256"],
                execution_host=execution["host"],
                game_specs=tuple(_game_spec_from_payload(item) for item in game_specs),
                experimental_arm=arms["experimental"],
                control_arm=arms["control"],
            )
        except (KeyError, TypeError, BO1000PairRunnerError) as exc:
            raise BO1000PairRunnerError("pair envelope fields are invalid") from exc
        if (
            raw.get("idempotency_key_sha256") != envelope.idempotency_key_sha256
            or raw.get("envelope_sha256") != envelope.envelope_sha256
        ):
            raise BO1000PairRunnerError(
                "pair envelope identity does not match its bytes"
            )
        return envelope


def build_bo1000_pair_envelope(
    schedule: Sequence[BO1000GameSpec],
    *,
    pair_index: int,
    checkpoint_sha256: str,
    bundle_sha256: str,
    pair_rng_snapshot_sha256: str,
    deck_order_rng_sha256: str,
    execution_host: str,
    evaluation_id: str = DEFAULT_EVALUATION_ID,
    experimental_arm: str = MCTS_ARM,
    control_arm: str = CONTROL_ARM,
) -> BO1000PairEnvelope:
    """Select one immutable schedule pair and bind its host-local inputs."""

    if type(pair_index) is not int or pair_index < 0:
        raise BO1000PairRunnerError("pair_index must be a nonnegative integer")
    selected = tuple(spec for spec in schedule if spec.pair_index == pair_index)
    if len(selected) != 2:
        raise BO1000PairRunnerError(
            "schedule must contain exactly two specs for pair_index"
        )
    first = selected[0]
    return BO1000PairEnvelope(
        evaluation_id=evaluation_id,
        pair_index=pair_index,
        pair_id=first.pair_id,
        pair_nonce_sha256=first.pair_nonce_sha256,
        checkpoint_sha256=checkpoint_sha256,
        bundle_sha256=bundle_sha256,
        pair_rng_snapshot_sha256=pair_rng_snapshot_sha256,
        deck_order_rng_sha256=deck_order_rng_sha256,
        execution_host=execution_host,
        game_specs=selected,  # type: ignore[arg-type]
        experimental_arm=experimental_arm,
        control_arm=control_arm,
    )


@dataclass(frozen=True, slots=True)
class BO1000PairStatus:
    """Derived status; terminal status comes only from an immutable pair receipt."""

    pair_id: str
    pair_nonce_sha256: str
    idempotency_key_sha256: str
    envelope_sha256: str
    status: Literal["pending", "partial", "complete", "failed_closed"]
    observed_game_receipts: int
    terminal_pair_receipt_sha256: str | None


@dataclass(frozen=True, slots=True)
class BO1000PairRunResult:
    envelope: BO1000PairEnvelope
    status: BO1000PairStatus
    game_receipts: tuple[BO1000GameReceipt, ...]


class HostLocalGameProcess(Protocol):
    """A command adapter that runs exactly one game in a local child process."""

    def run_game(
        self,
        *,
        request_path: Path,
        request: Mapping[str, object],
        envelope: BO1000PairEnvelope,
        spec: BO1000GameSpec,
    ) -> Mapping[str, object]:
        """Return one JSON receipt object.  It must not dispatch remotely."""


CommandBuilder = Callable[[Path, BO1000GameSpec], Sequence[str]]


def build_r207_local_game_command(
    request_path: Path,
    spec: BO1000GameSpec,
) -> tuple[str, ...]:
    """Return the isolated child command for one r207 local MCTS game.

    The command has no built-in runtime factory.  It therefore emits a
    canonical failed-closed receipt until a caller has supplied the separate
    real-successor and frozen-calibration interfaces required by r207.  This
    small builder is the only direct wiring from the immutable pair runner to
    the MCTS game-command adapter; it never reaches Guide2Vec, the r206 guide
    transform, an RTP sidecar, or a remote dispatcher.
    """

    if not isinstance(request_path, Path):
        raise BO1000PairRunnerError("r207 local game request path must be a Path")
    if not isinstance(spec, BO1000GameSpec):
        raise BO1000PairRunnerError("r207 local game spec is invalid")
    return (
        sys.executable,
        "-m",
        "poke_bot.recursive_turn_planner.r207_local_game_command",
        os.fspath(request_path),
    )


@dataclass(frozen=True, slots=True)
class HostLocalSubprocessGameRunner:
    """Run a separately supplied game command once per fresh local process.

    The command receives the immutable request path as an argument.  It must
    print exactly one JSON ``BO1000GameReceipt`` object to stdout.  This class
    deliberately has no endpoint, socket, worker, or service-manager API.
    """

    command_builder: CommandBuilder
    cwd: Path | None = None
    extra_env: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        forbidden = (
            set(R207_FORBIDDEN_GUIDE_ENV_KEYS)
            | set(R207_FORBIDDEN_LEGACY_RTP_ENV_KEYS)
            | set(R207_GUIDE_ISOLATION_ENV)
        )
        if self.extra_env is not None and forbidden.intersection(self.extra_env):
            raise BO1000PairRunnerError(
                "host-local game environment may not enable or override guide transforms"
            )

    def run_game(
        self,
        *,
        request_path: Path,
        request: Mapping[str, object],
        envelope: BO1000PairEnvelope,
        spec: BO1000GameSpec,
    ) -> Mapping[str, object]:
        del request, envelope
        command = tuple(self.command_builder(request_path, spec))
        if not command or any(
            not isinstance(part, str) or not part for part in command
        ):
            raise BO1000PairRunnerError(
                "host-local game command must be nonempty strings"
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
        env.update(
            {
                "BO1000_EXECUTION_SCOPE": "host_local_fresh_process",
                "BO1000_REMOTE_DISPATCH_AUTHORIZED": "0",
                "BO1000_PAIR_ID": spec.pair_id,
                "BO1000_GAME_NONCE_SHA256": spec.game_nonce_sha256,
            }
        )
        if self.extra_env is not None:
            env.update({str(key): str(value) for key, value in self.extra_env.items()})
        completed = subprocess.run(
            command,
            check=False,
            cwd=os.fspath(self.cwd or request_path.parent),
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise BO1000PairRunnerError("host-local game process failed closed")
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise BO1000PairRunnerError(
                "host-local game process did not emit one JSON receipt"
            ) from exc
        if not isinstance(result, Mapping):
            raise BO1000PairRunnerError(
                "host-local game process emitted non-object JSON"
            )
        return result


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("immutable receipt write made no progress")
        offset += written


def _write_immutable_json(
    path: Path, payload: Mapping[str, object], *, label: str
) -> None:
    """Create an evidence file once, or require byte-identical reuse."""

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
            raise BO1000PairRunnerError(f"existing {label} is not immutable") from exc
        if material.payload != encoded:
            raise BO1000PairRunnerError(
                f"existing {label} differs from immutable payload"
            )
        return
    try:
        _write_all(descriptor, encoded)
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)


@contextlib.contextmanager
def _exclusive_pair_lock(path: Path):
    """Serialize one local pair without making mutable status authoritative."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _read_immutable_payload(path: Path, *, label: str) -> dict[str, object]:
    try:
        _, payload = read_immutable_json_object(path, label)
    except ImmutableEvidenceIOError as exc:
        raise BO1000PairRunnerError(f"{label} is unavailable or mutable") from exc
    return payload


def _receipt_matches_envelope(
    receipt: BO1000GameReceipt,
    *,
    envelope: BO1000PairEnvelope,
    spec: BO1000GameSpec,
) -> None:
    if (
        receipt.game_nonce_sha256 != spec.game_nonce_sha256
        or receipt.pair_id != spec.pair_id
        or receipt.game_index != spec.game_index
        or receipt.mcts_seat != spec.mcts_seat
        or receipt.no_rtp_seat != spec.no_rtp_seat
    ):
        raise BO1000PairRunnerError("game receipt does not match its immutable spec")
    if (
        receipt.pair_rng_snapshot_sha256 != envelope.pair_rng_snapshot_sha256
        or receipt.deck_order_rng_sha256 != envelope.deck_order_rng_sha256
    ):
        raise BO1000PairRunnerError("game receipt does not match immutable pair RNG")
    if (
        receipt.checkpoint_sha256 != envelope.checkpoint_sha256
        or receipt.bundle_sha256 != envelope.bundle_sha256
    ):
        raise BO1000PairRunnerError("game receipt violates frozen-model identity")


def _failed_closed_receipt(
    envelope: BO1000PairEnvelope,
    spec: BO1000GameSpec,
) -> BO1000GameReceipt:
    """Record a command/protocol failure without inventing game or MCTS data."""

    return BO1000GameReceipt(
        game_nonce_sha256=spec.game_nonce_sha256,
        pair_id=spec.pair_id,
        game_index=spec.game_index,
        mcts_seat=spec.mcts_seat,
        no_rtp_seat=spec.no_rtp_seat,
        pair_rng_snapshot_sha256=envelope.pair_rng_snapshot_sha256,
        deck_order_rng_sha256=envelope.deck_order_rng_sha256,
        checkpoint_sha256=envelope.checkpoint_sha256,
        bundle_sha256=envelope.bundle_sha256,
        terminal_status="failed_closed",
        winner_seat=None,
        illegal_action_count=0,
        forfeit_count=0,
        crash_count=1,
        timeout_count=0,
        mcts_turns=(),
    )


class HostLocalBO1000PairController:
    """Materialize and run one immutable BO1000 pair only on this host.

    The controller intentionally keeps no global queue or remote-dispatch
    behavior.  A caller chooses one preflighted pair, supplies a local command
    adapter, and receives the pair's immutable status.  A full BO1000 report
    remains the responsibility of :func:`compile_bo1000_report` after all 500
    pairs are complete.
    """

    def __init__(self, output_root: str | Path, *, execution_host: str | None = None):
        root = Path(output_root).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        self._output_root = root.resolve()
        self._execution_host = execution_host or platform.node() or "unknown-host"
        _require_nonempty_text(self._execution_host, name="execution_host")

    @property
    def output_root(self) -> Path:
        return self._output_root

    @property
    def execution_host(self) -> str:
        return self._execution_host

    def pair_directory(self, envelope: BO1000PairEnvelope) -> Path:
        self._require_local_host(envelope)
        return (
            self.output_root
            / "bo1000-pairs"
            / envelope.pair_id
            / envelope.pair_nonce_sha256.removeprefix("sha256:")
        )

    def _require_local_host(self, envelope: BO1000PairEnvelope) -> None:
        if envelope.execution_host != self.execution_host:
            raise BO1000PairRunnerError(
                "pair envelope host does not match this host-local controller"
            )

    def envelope_path(self, envelope: BO1000PairEnvelope) -> Path:
        return self.pair_directory(envelope) / "pair-envelope.json"

    def _request_path(self, envelope: BO1000PairEnvelope, spec: BO1000GameSpec) -> Path:
        return (
            self.pair_directory(envelope)
            / "game-requests"
            / f"{spec.game_index}-{spec.game_nonce_sha256.removeprefix('sha256:')}.json"
        )

    def _game_receipt_path(
        self, envelope: BO1000PairEnvelope, spec: BO1000GameSpec
    ) -> Path:
        return (
            self.pair_directory(envelope)
            / "game-receipts"
            / f"{spec.game_index}-{spec.game_nonce_sha256.removeprefix('sha256:')}.json"
        )

    def _pair_receipt_path(self, envelope: BO1000PairEnvelope) -> Path:
        return self.pair_directory(envelope) / "pair-receipt.json"

    def materialize_envelope(self, envelope: BO1000PairEnvelope) -> Path:
        self._require_local_host(envelope)
        path = self.envelope_path(envelope)
        _write_immutable_json(path, envelope.as_payload(), label="pair envelope")
        stored = BO1000PairEnvelope.from_payload(
            _read_immutable_payload(path, label="pair envelope")
        )
        if stored != envelope:
            raise BO1000PairRunnerError(
                "stored pair envelope does not match requested pair"
            )
        return path

    def _game_request_payload(
        self, envelope: BO1000PairEnvelope, spec: BO1000GameSpec
    ) -> dict[str, object]:
        return {
            "schema": GAME_REQUEST_SCHEMA,
            "evaluation_id": envelope.evaluation_id,
            "pair": {
                "pair_index": envelope.pair_index,
                "pair_id": envelope.pair_id,
                "pair_nonce_sha256": envelope.pair_nonce_sha256,
                "idempotency_key_sha256": envelope.idempotency_key_sha256,
                "envelope_sha256": envelope.envelope_sha256,
            },
            "execution": {
                "scope": "host_local_fresh_process",
                "host": envelope.execution_host,
                "remote_dispatch_authorized": False,
                "fresh_process_required": True,
            },
            "frozen_model": {
                "checkpoint_sha256": envelope.checkpoint_sha256,
                "bundle_sha256": envelope.bundle_sha256,
            },
            "rng": {
                "pair_rng_snapshot_sha256": envelope.pair_rng_snapshot_sha256,
                "deck_order_rng_sha256": envelope.deck_order_rng_sha256,
            },
            "arms": {
                "experimental": envelope.experimental_arm,
                "control": envelope.control_arm,
            },
            "policy_transforms": {
                "frozen_base_policy": "r195_identity",
                "guide2vec_enabled": False,
                "guide_logit_transform_enabled": False,
                "guide_linear_transform_enabled": False,
            },
            "matchup_adapter": {
                "matchup_tree_sha256": R207_R195_MATCHUP_TREE_SHA256,
                "enabled": True,
                "trained": True,
                "frozen": True,
                "same_exact_runtime_required_for_both_arms": True,
            },
            "game": spec.as_payload(),
            "authority": {
                "training_eligible": False,
                "serving_eligible": False,
                "remote_dispatch_authorized": False,
            },
        }

    def _materialize_game_request(
        self, envelope: BO1000PairEnvelope, spec: BO1000GameSpec
    ) -> tuple[Path, dict[str, object]]:
        request = self._game_request_payload(envelope, spec)
        path = self._request_path(envelope, spec)
        _write_immutable_json(path, request, label="host-local game request")
        stored = _read_immutable_payload(path, label="host-local game request")
        if stored != request:
            raise BO1000PairRunnerError(
                "stored game request does not match pair envelope"
            )
        return path, request

    def _load_game_receipt(
        self, envelope: BO1000PairEnvelope, spec: BO1000GameSpec
    ) -> BO1000GameReceipt | None:
        path = self._game_receipt_path(envelope, spec)
        if not path.exists():
            return None
        receipt = game_receipt_from_payload(
            _read_immutable_payload(path, label="host-local game receipt")
        )
        _receipt_matches_envelope(receipt, envelope=envelope, spec=spec)
        return receipt

    def _pair_receipt_payload(
        self,
        envelope: BO1000PairEnvelope,
        receipts: Sequence[BO1000GameReceipt],
    ) -> dict[str, object]:
        if len(receipts) != 2:
            raise BO1000PairRunnerError("pair receipt needs exactly two game receipts")
        status = (
            "failed_closed"
            if any(receipt.terminal_status != "completed" for receipt in receipts)
            else "complete"
        )
        games = []
        for spec, receipt in zip(
            sorted(envelope.game_specs, key=lambda item: item.game_index), receipts
        ):
            _receipt_matches_envelope(receipt, envelope=envelope, spec=spec)
            payload = _receipt_payload(receipt)
            games.append(
                {
                    "game_index": spec.game_index,
                    "game_nonce_sha256": spec.game_nonce_sha256,
                    "receipt_payload_sha256": _canonical_sha256(payload),
                    "terminal_status": receipt.terminal_status,
                }
            )
        base: dict[str, object] = {
            "schema": PAIR_RECEIPT_SCHEMA,
            "evaluation_id": envelope.evaluation_id,
            "pair_id": envelope.pair_id,
            "pair_nonce_sha256": envelope.pair_nonce_sha256,
            "idempotency_key_sha256": envelope.idempotency_key_sha256,
            "envelope_sha256": envelope.envelope_sha256,
            "execution": {
                "scope": "host_local_fresh_process",
                "host": envelope.execution_host,
                "remote_dispatch_authorized": False,
                "fresh_process_games_started": 2,
            },
            "status": status,
            "game_receipts": games,
        }
        return {**base, "canonical_sha256": _canonical_sha256(base)}

    def _validate_pair_receipt(
        self,
        raw: Mapping[str, object],
        *,
        envelope: BO1000PairEnvelope,
        receipts: Sequence[BO1000GameReceipt],
    ) -> str:
        expected = {
            "schema",
            "evaluation_id",
            "pair_id",
            "pair_nonce_sha256",
            "idempotency_key_sha256",
            "envelope_sha256",
            "execution",
            "status",
            "game_receipts",
            "canonical_sha256",
        }
        if set(raw) != expected or raw.get("schema") != PAIR_RECEIPT_SCHEMA:
            raise BO1000PairRunnerError("pair receipt schema is invalid")
        expected_payload = self._pair_receipt_payload(envelope, receipts)
        if dict(raw) != expected_payload:
            raise BO1000PairRunnerError(
                "pair receipt does not match immutable game evidence"
            )
        return str(raw["status"])

    def _loaded_receipts(
        self, envelope: BO1000PairEnvelope
    ) -> tuple[BO1000GameReceipt, ...]:
        receipts = []
        for spec in sorted(envelope.game_specs, key=lambda item: item.game_index):
            receipt = self._load_game_receipt(envelope, spec)
            if receipt is not None:
                receipts.append(receipt)
        return tuple(receipts)

    def status(self, envelope: BO1000PairEnvelope) -> BO1000PairStatus:
        """Inspect immutable artifacts without starting a process or changing status."""

        self._require_local_host(envelope)
        envelope_path = self.envelope_path(envelope)
        if not envelope_path.exists():
            return BO1000PairStatus(
                pair_id=envelope.pair_id,
                pair_nonce_sha256=envelope.pair_nonce_sha256,
                idempotency_key_sha256=envelope.idempotency_key_sha256,
                envelope_sha256=envelope.envelope_sha256,
                status="pending",
                observed_game_receipts=0,
                terminal_pair_receipt_sha256=None,
            )
        stored = BO1000PairEnvelope.from_payload(
            _read_immutable_payload(envelope_path, label="pair envelope")
        )
        if stored != envelope:
            raise BO1000PairRunnerError(
                "stored pair envelope does not match requested pair"
            )
        receipts = self._loaded_receipts(envelope)
        pair_receipt_path = self._pair_receipt_path(envelope)
        if pair_receipt_path.exists():
            if len(receipts) != 2:
                raise BO1000PairRunnerError(
                    "terminal pair receipt is missing game evidence"
                )
            raw = _read_immutable_payload(pair_receipt_path, label="pair receipt")
            terminal_status = self._validate_pair_receipt(
                raw, envelope=envelope, receipts=receipts
            )
            return BO1000PairStatus(
                pair_id=envelope.pair_id,
                pair_nonce_sha256=envelope.pair_nonce_sha256,
                idempotency_key_sha256=envelope.idempotency_key_sha256,
                envelope_sha256=envelope.envelope_sha256,
                status=terminal_status,  # type: ignore[arg-type]
                observed_game_receipts=2,
                terminal_pair_receipt_sha256=str(raw["canonical_sha256"]),
            )
        return BO1000PairStatus(
            pair_id=envelope.pair_id,
            pair_nonce_sha256=envelope.pair_nonce_sha256,
            idempotency_key_sha256=envelope.idempotency_key_sha256,
            envelope_sha256=envelope.envelope_sha256,
            status="pending" if not receipts else "partial",
            observed_game_receipts=len(receipts),
            terminal_pair_receipt_sha256=None,
        )

    def run_pair(
        self, envelope: BO1000PairEnvelope, runner: HostLocalGameProcess
    ) -> BO1000PairRunResult:
        """Run missing games locally, then seal a single terminal pair receipt.

        Existing immutable terminal evidence is returned without starting another
        child process.  A malformed or failed local command becomes a binding
        ``failed_closed`` game receipt; the opposite seat is still attempted so
        the pair never silently crosses or drops a scheduled game.
        """

        pair_dir = self.pair_directory(envelope)
        with _exclusive_pair_lock(pair_dir / ".controller.lock"):
            self.materialize_envelope(envelope)
            before = self.status(envelope)
            receipts = list(self._loaded_receipts(envelope))
            if before.status in {"complete", "failed_closed"}:
                return BO1000PairRunResult(
                    envelope=envelope,
                    status=before,
                    game_receipts=tuple(receipts),
                )

            by_nonce = {receipt.game_nonce_sha256: receipt for receipt in receipts}
            for spec in sorted(envelope.game_specs, key=lambda item: item.game_index):
                if spec.game_nonce_sha256 in by_nonce:
                    continue
                request_path, request = self._materialize_game_request(envelope, spec)
                try:
                    raw_receipt = runner.run_game(
                        request_path=request_path,
                        request=request,
                        envelope=envelope,
                        spec=spec,
                    )
                    receipt = game_receipt_from_payload(raw_receipt)
                    _receipt_matches_envelope(receipt, envelope=envelope, spec=spec)
                except (
                    BO1000PairRunnerError,
                    BO1000EvidenceError,
                    OSError,
                    ValueError,
                ):
                    receipt = _failed_closed_receipt(envelope, spec)
                _write_immutable_json(
                    self._game_receipt_path(envelope, spec),
                    _receipt_payload(receipt),
                    label="host-local game receipt",
                )
                by_nonce[spec.game_nonce_sha256] = receipt

            ordered = tuple(
                by_nonce[spec.game_nonce_sha256]
                for spec in sorted(
                    envelope.game_specs, key=lambda item: item.game_index
                )
            )
            _write_immutable_json(
                self._pair_receipt_path(envelope),
                self._pair_receipt_payload(envelope, ordered),
                label="pair receipt",
            )
            terminal_status = self.status(envelope)
            return BO1000PairRunResult(
                envelope=envelope,
                status=terminal_status,
                game_receipts=ordered,
            )


__all__ = [
    "DEFAULT_EVALUATION_ID",
    "GAME_REQUEST_SCHEMA",
    "PAIR_ENVELOPE_SCHEMA",
    "PAIR_RECEIPT_SCHEMA",
    "R207_FORBIDDEN_LEGACY_RTP_ENV_KEYS",
    "R207_R195_MATCHUP_TREE_SHA256",
    "BO1000PairEnvelope",
    "BO1000PairRunResult",
    "BO1000PairRunnerError",
    "BO1000PairStatus",
    "HostLocalBO1000PairController",
    "HostLocalGameProcess",
    "HostLocalSubprocessGameRunner",
    "build_r207_local_game_command",
    "build_bo1000_pair_envelope",
    "game_receipt_from_payload",
]
