"""Fail-closed local game command for the r207 no-RTP MCTS BO1000.

This module is deliberately a narrow glue layer between the immutable
``bo1000_pair_runner`` request/receipt boundary and the separately verified
simulator MCTS implementation.  It does *not* create a successor arena,
calibrate a model, start a service, contact a remote, load an RTP sidecar, or
enable any guide transform.  A caller must inject a live-game factory only
after preflight has provided both:

* a real, opaque native successor interface with its receipts; and
* a source-excluded frozen r195 outcome/value calibration receipt.

Without either, the adapter emits a binding ``failed_closed`` game receipt.
That makes the module safe to wire into the pair runner before the native V3
successor implementation is ready, while preventing a direct-policy fallback
from being misreported as MCTS.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .bo1000_evaluation import (
    CONTROL_ARM,
    MCTS_ARM,
    R207_CANONICAL_PLANNER_CONFIG_SHA256,
    R207_EVALUATION_ID,
    R207_FROZEN_BUNDLE_SHA256,
    R207_FROZEN_CHECKPOINT_SHA256,
    BO1000EvidenceError,
    BO1000GameReceipt,
    BO1000GameSpec,
    MCTSTurnTelemetry,
)
from .chance_aware_tree import ChanceAwareSearchConfig
from .neural_leaf_reranker import (
    BatchedNeuralLeafReranker,
    FrozenPolicyIdentity,
    LeafScoreCalibration,
)
from .r207_frozen_leaf_calibration import (
    FrozenLeafCalibrationError,
    leaf_score_calibration_from_r207_receipt,
)
from .r207_simulator_arena import (
    SuccessorArena,
    TransitionKind,
    TruthfulReplayArena,
)
from poke_bot.rtp_evaluation_immutable_io import (
    ImmutableEvidenceIOError,
    read_immutable_file_bytes,
)
from .simulator_one_turn_expectimax import (
    PolicyDecisionView,
    SimulatorInterTurnMCTSSession,
)


R207_LOCAL_GAME_COMMAND_SCHEMA = "poke_bot.alakazam_mcts_bo1000_r207_local_game_command/v1"
R207_REAL_SUCCESSOR_INTERFACE_KIND = "r207_native_clone_successor/v1"
R207_LOCAL_RUNTIME_FACTORY_ENV = "BO1000_R207_LOCAL_RUNTIME_FACTORY"
R207_R195_MATCHUP_TREE_SHA256 = (
    "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049"
)
_MAX_GAME_ACTIONS = 4_000

# The pair runner already clears these values for its fresh subprocess.  Check
# again here because this command can also be invoked directly during local
# preflight.  A missing transform is fine; a nonzero transform is never fine.
_FORBIDDEN_ENABLED_ENV = (
    "POKEBOT_CURRENT_DECK_GUIDE",
    "POKEBOT_CURRENT_DECK_GUIDE_TARGETS",
    "POKEBOT_ALAKAZAM_GUIDE_TARGETS",
    "POKEBOT_GUIDE2VEC_CHECKPOINT",
    "POKEBOT_GUIDE2VEC_ENABLED",
    "POKEBOT_GUIDE_LOGIT_CONFIG",
    "POKEBOT_GUIDE_LOGIT_WEIGHT",
    "POKEBOT_GUIDE_LINEAR_TRANSFORM",
    "POKEBOT_RTP_CHECKPOINT",
    "POKEBOT_RTP_SIDELOAD",
    "POKEBOT_RTP_ENABLED",
    "POKEBOT_RECURSIVE_TURN_PLANNER",
    "POKEBOT_RECURSIVE_TURN_PLANNER_ENABLED",
)
_REQUIRED_DISABLED_ENV = {
    "BO1000_GUIDE2VEC_ENABLED": "0",
    "BO1000_GUIDE_LOGIT_TRANSFORM_ENABLED": "0",
    "BO1000_GUIDE_LINEAR_TRANSFORM_ENABLED": "0",
    "BO1000_BASE_POLICY_TRANSFORM": "frozen_r195_identity",
    "BO1000_MATCHUP_ADAPTER_ENABLED": "1",
    "BO1000_MATCHUP_TREE_SHA256": R207_R195_MATCHUP_TREE_SHA256,
    "BO1000_REMOTE_DISPATCH_AUTHORIZED": "0",
}


class R207LocalGameCommandError(RuntimeError):
    """The local game command cannot truthfully run the r207 experimental arm."""


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise R207LocalGameCommandError(f"{label} must be a sha256 digest")
    suffix = value.removeprefix("sha256:")
    if len(suffix) != 64 or any(character not in "0123456789abcdef" for character in suffix):
        raise R207LocalGameCommandError(f"{label} must be a lowercase sha256 digest")
    return value


def _nonempty(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise R207LocalGameCommandError(f"{label} must be nonempty text")
    return value


def _mapping(value: object, *, label: str, fields: set[str]) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise R207LocalGameCommandError(f"{label} has an invalid schema")
    return dict(value)


def _action(value: Sequence[int], *, label: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise R207LocalGameCommandError(f"{label} must be an action sequence")
    try:
        action = tuple(value)
    except TypeError as exc:
        raise R207LocalGameCommandError(f"{label} must be an action sequence") from exc
    if any(type(part) is not int or part < 0 for part in action):
        raise R207LocalGameCommandError(f"{label} must contain nonnegative exact integers")
    return action


@dataclass(frozen=True, slots=True)
class R207LocalGameRequest:
    """One immutable r207 local-game request parsed from the pair runner."""

    game_spec: BO1000GameSpec
    pair_rng_snapshot_sha256: str
    deck_order_rng_sha256: str
    execution_host: str
    pair_envelope_sha256: str
    pair_idempotency_key_sha256: str
    matchup_tree_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.game_spec, BO1000GameSpec):
            raise R207LocalGameCommandError("game_spec must be a BO1000GameSpec")
        _digest(self.pair_rng_snapshot_sha256, label="pair_rng_snapshot_sha256")
        _digest(self.deck_order_rng_sha256, label="deck_order_rng_sha256")
        _digest(self.pair_envelope_sha256, label="pair_envelope_sha256")
        _digest(self.pair_idempotency_key_sha256, label="pair_idempotency_key_sha256")
        if self.matchup_tree_sha256 != R207_R195_MATCHUP_TREE_SHA256:
            raise R207LocalGameCommandError("request must bind exact r195 matchup tree")
        _nonempty(self.execution_host, label="execution_host")


def parse_r207_local_game_request(raw: Mapping[str, object]) -> R207LocalGameRequest:
    """Parse exactly the immutable request shape emitted by the pair runner."""

    root = _mapping(
        raw,
        label="r207 local game request",
        fields={
            "schema",
            "evaluation_id",
            "pair",
            "execution",
            "frozen_model",
            "rng",
            "arms",
            "policy_transforms",
            "matchup_adapter",
            "game",
            "authority",
        },
    )
    if root["schema"] != "poke_bot.alakazam_mcts_bo1000_host_local_game_request/v1":
        raise R207LocalGameCommandError("local game request schema is not r207")
    if root["evaluation_id"] != R207_EVALUATION_ID:
        raise R207LocalGameCommandError("local game request evaluation identity is not r207")
    pair = _mapping(
        root["pair"],
        label="request.pair",
        fields={
            "pair_index",
            "pair_id",
            "pair_nonce_sha256",
            "idempotency_key_sha256",
            "envelope_sha256",
        },
    )
    execution = _mapping(
        root["execution"],
        label="request.execution",
        fields={"scope", "host", "remote_dispatch_authorized", "fresh_process_required"},
    )
    frozen_model = _mapping(
        root["frozen_model"],
        label="request.frozen_model",
        fields={"checkpoint_sha256", "bundle_sha256"},
    )
    rng = _mapping(
        root["rng"],
        label="request.rng",
        fields={"pair_rng_snapshot_sha256", "deck_order_rng_sha256"},
    )
    arms = _mapping(root["arms"], label="request.arms", fields={"experimental", "control"})
    transforms = _mapping(
        root["policy_transforms"],
        label="request.policy_transforms",
        fields={
            "frozen_base_policy",
            "guide2vec_enabled",
            "guide_logit_transform_enabled",
            "guide_linear_transform_enabled",
        },
    )
    matchup_adapter = _mapping(
        root["matchup_adapter"],
        label="request.matchup_adapter",
        fields={
            "matchup_tree_sha256",
            "enabled",
            "trained",
            "frozen",
            "same_exact_runtime_required_for_both_arms",
        },
    )
    authority = _mapping(
        root["authority"],
        label="request.authority",
        fields={"training_eligible", "serving_eligible", "remote_dispatch_authorized"},
    )
    if execution != {
        "scope": "host_local_fresh_process",
        "host": execution["host"],
        "remote_dispatch_authorized": False,
        "fresh_process_required": True,
    }:
        raise R207LocalGameCommandError("request is not a host-local fresh-process game")
    _nonempty(execution["host"], label="request.execution.host")
    if frozen_model != {
        "checkpoint_sha256": R207_FROZEN_CHECKPOINT_SHA256,
        "bundle_sha256": R207_FROZEN_BUNDLE_SHA256,
    }:
        raise R207LocalGameCommandError("request does not bind exact r195 NO-RTP model bytes")
    if arms != {"experimental": MCTS_ARM, "control": CONTROL_ARM}:
        raise R207LocalGameCommandError("request arm labels are not the exact r207 arms")
    if transforms != {
        "frozen_base_policy": "r195_identity",
        "guide2vec_enabled": False,
        "guide_logit_transform_enabled": False,
        "guide_linear_transform_enabled": False,
    }:
        raise R207LocalGameCommandError("request permits a guide or policy transform")
    if matchup_adapter != {
        "matchup_tree_sha256": R207_R195_MATCHUP_TREE_SHA256,
        "enabled": True,
        "trained": True,
        "frozen": True,
        "same_exact_runtime_required_for_both_arms": True,
    }:
        raise R207LocalGameCommandError(
            "request does not bind exact r195 matchup adapter ON for both arms"
        )
    if authority != {
        "training_eligible": False,
        "serving_eligible": False,
        "remote_dispatch_authorized": False,
    }:
        raise R207LocalGameCommandError("request has non-evaluation authority")
    game_payload = root["game"]
    if not isinstance(game_payload, Mapping):
        raise R207LocalGameCommandError("request.game must be an object")
    expected_game_fields = set(BO1000GameSpec.__dataclass_fields__)
    if set(game_payload) != expected_game_fields:
        raise R207LocalGameCommandError("request.game has an invalid schema")
    try:
        spec = BO1000GameSpec(**dict(game_payload))
    except (BO1000EvidenceError, TypeError) as exc:
        raise R207LocalGameCommandError("request.game is invalid") from exc
    if (
        pair["pair_index"] != spec.pair_index
        or pair["pair_id"] != spec.pair_id
        or pair["pair_nonce_sha256"] != spec.pair_nonce_sha256
    ):
        raise R207LocalGameCommandError("request.game does not belong to request.pair")
    return R207LocalGameRequest(
        game_spec=spec,
        pair_rng_snapshot_sha256=_digest(
            rng["pair_rng_snapshot_sha256"], label="request.rng.pair_rng_snapshot_sha256"
        ),
        deck_order_rng_sha256=_digest(
            rng["deck_order_rng_sha256"], label="request.rng.deck_order_rng_sha256"
        ),
        execution_host=_nonempty(execution["host"], label="request.execution.host"),
        pair_envelope_sha256=_digest(pair["envelope_sha256"], label="request.pair.envelope_sha256"),
        pair_idempotency_key_sha256=_digest(
            pair["idempotency_key_sha256"], label="request.pair.idempotency_key_sha256"
        ),
        matchup_tree_sha256=matchup_adapter["matchup_tree_sha256"],
    )


@dataclass(frozen=True, slots=True)
class R207RealSuccessorBinding:
    """The narrow receipt-bound bridge to a real native successor arena.

    The current replay arena is intentionally excluded: it is a useful
    preflight simulator but is not an arbitrary live-midgame clone interface.
    ``arena`` is kept opaque and must expose this binding's interface kind and
    ABI receipt identity itself, which makes an accidental unbound replay
    arena fail closed before any MCTS action is selected.
    """

    arena: SuccessorArena = field(repr=False, compare=False)
    interface_kind: str
    successor_abi_receipt_sha256: str
    future_legality_receipt_sha256: str
    exact_chance_receipt_sha256: str
    terminal_result_parity_receipt_sha256: str
    deterministic_receipt_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.arena, SuccessorArena):
            raise R207LocalGameCommandError("successor binding needs a SuccessorArena")
        if self.interface_kind != R207_REAL_SUCCESSOR_INTERFACE_KIND:
            raise R207LocalGameCommandError("successor binding is not a real r207 native interface")
        for field_name in (
            "successor_abi_receipt_sha256",
            "future_legality_receipt_sha256",
            "exact_chance_receipt_sha256",
            "terminal_result_parity_receipt_sha256",
            "deterministic_receipt_sha256",
        ):
            _digest(getattr(self, field_name), label=field_name)
        # v2 can truthfully produce useful boundaries but cannot serve as a
        # real successor branch ABI for the BO1000 experimental arm.
        if isinstance(self.arena, TruthfulReplayArena):
            raise R207LocalGameCommandError("r207 replay arena is not a real successor interface")
        expected_arena_receipts = {
            "r207_successor_interface_kind": self.interface_kind,
            "r207_successor_abi_receipt_sha256": self.successor_abi_receipt_sha256,
            "r207_future_legality_receipt_sha256": self.future_legality_receipt_sha256,
            "r207_exact_chance_receipt_sha256": self.exact_chance_receipt_sha256,
            "r207_terminal_result_parity_receipt_sha256": (
                self.terminal_result_parity_receipt_sha256
            ),
            "r207_deterministic_receipt_sha256": self.deterministic_receipt_sha256,
        }
        for attribute, expected in expected_arena_receipts.items():
            if getattr(self.arena, attribute, None) != expected:
                raise R207LocalGameCommandError(
                    f"successor arena lacks attested r207 prerequisite: {attribute}"
                )


@dataclass(frozen=True, slots=True)
class R207FrozenLeafCalibrationBinding:
    """Open and verify the actual immutable calibration receipt before MCTS.

    ``LeafScoreCalibration`` intentionally carries only the resulting blend;
    by itself it cannot establish that a source-excluded r207 calibration
    receipt was ever compiled.  This binding is the live-game boundary that
    does that independent immutable receipt and input verification.  It is
    deliberately filesystem-only and has no inference, training, service, or
    remote behavior.
    """

    calibration_receipt_path: Path
    heldout_evidence_path: Path
    frozen_predictions_path: Path
    r195_training_source_manifest_path: Path
    leaf_calibration: LeafScoreCalibration = field(init=False)
    receipt_file_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        path_fields = (
            "calibration_receipt_path",
            "heldout_evidence_path",
            "frozen_predictions_path",
            "r195_training_source_manifest_path",
        )
        for field_name in path_fields:
            path = getattr(self, field_name)
            if not isinstance(path, Path) or not path.is_absolute():
                raise R207LocalGameCommandError(
                    f"{field_name} must be an absolute immutable receipt path"
                )
        try:
            material = read_immutable_file_bytes(
                self.calibration_receipt_path,
                "r207 frozen leaf calibration receipt",
                exact_mode=0o444,
            )
            raw = json.loads(material.payload.decode("utf-8"))
            if not isinstance(raw, Mapping):
                raise R207LocalGameCommandError(
                    "r207 frozen leaf calibration receipt must be a JSON object"
                )
            calibration = leaf_score_calibration_from_r207_receipt(
                raw,
                heldout_evidence_path=self.heldout_evidence_path,
                frozen_predictions_path=self.frozen_predictions_path,
                r195_training_source_manifest_path=self.r195_training_source_manifest_path,
            )
        except (
            FrozenLeafCalibrationError,
            ImmutableEvidenceIOError,
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise R207LocalGameCommandError(
                "r207 runtime lacks a valid immutable source-excluded calibration receipt"
            ) from exc
        object.__setattr__(self, "leaf_calibration", calibration)
        object.__setattr__(self, "receipt_file_sha256", material.sha256)


@dataclass(frozen=True, slots=True)
class R207MCTSRuntimeBinding:
    """All non-game prerequisites that must exist before r207 MCTS can run."""

    successor: R207RealSuccessorBinding
    calibration: R207FrozenLeafCalibrationBinding
    frozen_policy_identity: FrozenPolicyIdentity
    no_rtp_runtime_receipt_sha256: str
    matchup_tree_sha256: str
    matchup_adapter_bank_sha256: str
    matchup_adapter_training_receipt_sha256: str
    matchup_adapter_runtime_graph_sha256: str
    direct_runtime_graph_sha256: str
    matchup_adapter_enabled: bool = True
    matchup_adapter_trained: bool = True
    matchup_adapter_frozen: bool = True
    guide2vec_enabled: bool = False
    guide_logit_transform_enabled: bool = False
    guide_linear_transform_enabled: bool = False
    legacy_rtp_sidecar_used: bool = False
    legacy_rtp_executor_used: bool = False
    training_or_gradient_updates_enabled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.successor, R207RealSuccessorBinding):
            raise R207LocalGameCommandError("r207 runtime lacks a real successor binding")
        if not isinstance(self.calibration, R207FrozenLeafCalibrationBinding):
            raise R207LocalGameCommandError("r207 runtime lacks a leaf calibration interface")
        if not isinstance(self.frozen_policy_identity, FrozenPolicyIdentity):
            raise R207LocalGameCommandError("r207 runtime lacks frozen policy identity")
        expected_identity = FrozenPolicyIdentity.r205_no_rtp()
        if self.frozen_policy_identity != expected_identity:
            raise R207LocalGameCommandError("runtime policy is not exact r195 NO-RTP")
        if (
            self.calibration.leaf_calibration.frozen_policy_identity_sha256
            != expected_identity.identity_sha256
        ):
            raise R207LocalGameCommandError("calibration belongs to another frozen policy")
        _digest(self.no_rtp_runtime_receipt_sha256, label="no_rtp_runtime_receipt_sha256")
        if self.matchup_tree_sha256 != R207_R195_MATCHUP_TREE_SHA256:
            raise R207LocalGameCommandError("runtime matchup tree is not exact r195")
        for field_name in (
            "matchup_adapter_bank_sha256",
            "matchup_adapter_training_receipt_sha256",
            "matchup_adapter_runtime_graph_sha256",
            "direct_runtime_graph_sha256",
        ):
            _digest(getattr(self, field_name), label=field_name)
        for field_name in (
            "matchup_adapter_enabled",
            "matchup_adapter_trained",
            "matchup_adapter_frozen",
        ):
            if getattr(self, field_name) is not True:
                raise R207LocalGameCommandError(
                    f"exact r195 matchup adapter must remain ON: {field_name}"
                )
        for field_name in (
            "guide2vec_enabled",
            "guide_logit_transform_enabled",
            "guide_linear_transform_enabled",
            "legacy_rtp_sidecar_used",
            "legacy_rtp_executor_used",
            "training_or_gradient_updates_enabled",
        ):
            if getattr(self, field_name) is not False:
                raise R207LocalGameCommandError(f"r207 runtime enables forbidden {field_name}")


@dataclass(frozen=True, slots=True)
class R207AppliedAction:
    """Observed post-action cache evidence supplied by the real game loop."""

    realized_transition_kind: TransitionKind | None = None
    next_mcts_view: PolicyDecisionView | None = None

    def __post_init__(self) -> None:
        if self.realized_transition_kind is not None and not isinstance(
            self.realized_transition_kind, TransitionKind
        ):
            raise R207LocalGameCommandError("realized transition kind is invalid")
        if self.next_mcts_view is not None and not isinstance(
            self.next_mcts_view, PolicyDecisionView
        ):
            raise R207LocalGameCommandError("next_mcts_view is invalid")


@dataclass(frozen=True, slots=True)
class R207GameTerminal:
    """Exact terminal facts from a completed local battle, never a model score."""

    winner_seat: int | None
    illegal_action_count: int = 0
    forfeit_count: int = 0
    crash_count: int = 0
    timeout_count: int = 0

    def __post_init__(self) -> None:
        if self.winner_seat not in {None, 0, 1}:
            raise R207LocalGameCommandError("winner_seat must be 0, 1, or None")
        for field_name in (
            "illegal_action_count",
            "forfeit_count",
            "crash_count",
            "timeout_count",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise R207LocalGameCommandError(f"{field_name} must be a nonnegative exact int")


@dataclass(frozen=True, slots=True)
class R207ArmRuntimeTelemetry:
    """Per-arm proof that MCTS wraps, rather than replaces, r195 runtime."""

    frozen_policy_identity_sha256: str
    no_rtp_runtime_receipt_sha256: str
    direct_runtime_graph_sha256: str
    matchup_tree_sha256: str
    matchup_adapter_bank_sha256: str
    matchup_adapter_training_receipt_sha256: str
    matchup_adapter_runtime_graph_sha256: str
    matchup_adapter_enabled: bool
    matchup_adapter_trained: bool
    matchup_adapter_frozen: bool
    guide2vec_enabled: bool
    guide_logit_transform_enabled: bool
    guide_linear_transform_enabled: bool
    legacy_rtp_sidecar_used: bool
    legacy_rtp_executor_used: bool

    def __post_init__(self) -> None:
        for field_name in (
            "frozen_policy_identity_sha256",
            "no_rtp_runtime_receipt_sha256",
            "direct_runtime_graph_sha256",
            "matchup_tree_sha256",
            "matchup_adapter_bank_sha256",
            "matchup_adapter_training_receipt_sha256",
            "matchup_adapter_runtime_graph_sha256",
        ):
            _digest(getattr(self, field_name), label=field_name)
        if self.matchup_tree_sha256 != R207_R195_MATCHUP_TREE_SHA256:
            raise R207LocalGameCommandError("arm telemetry does not bind exact r195 matchup tree")
        for field_name in (
            "matchup_adapter_enabled",
            "matchup_adapter_trained",
            "matchup_adapter_frozen",
        ):
            if getattr(self, field_name) is not True:
                raise R207LocalGameCommandError("arm telemetry has matchup adapter OFF")
        for field_name in (
            "guide2vec_enabled",
            "guide_logit_transform_enabled",
            "guide_linear_transform_enabled",
            "legacy_rtp_sidecar_used",
            "legacy_rtp_executor_used",
        ):
            if getattr(self, field_name) is not False:
                raise R207LocalGameCommandError(f"arm telemetry enables forbidden {field_name}")


@dataclass(frozen=True, slots=True)
class R207RuntimeParityTelemetry:
    """Receipt-compatible arm comparison emitted by the live runtime boundary."""

    mcts_arm: R207ArmRuntimeTelemetry
    direct_arm: R207ArmRuntimeTelemetry

    def __post_init__(self) -> None:
        if not isinstance(self.mcts_arm, R207ArmRuntimeTelemetry) or not isinstance(
            self.direct_arm, R207ArmRuntimeTelemetry
        ):
            raise R207LocalGameCommandError("runtime parity telemetry has invalid arms")
        if self.mcts_arm != self.direct_arm:
            raise R207LocalGameCommandError(
                "MCTS must wrap the exact same frozen r195 runtime as direct policy"
            )


@runtime_checkable
class R207LocalLiveGame(Protocol):
    """Minimal actual-game boundary used by the local BO1000 child command."""

    prerequisites: R207MCTSRuntimeBinding
    mcts_session: SimulatorInterTurnMCTSSession

    def next_decision(self) -> PolicyDecisionView | None:
        """Return the current live policy-visible decision, or ``None`` at terminal."""

    def apply_action(self, action: Sequence[int]) -> R207AppliedAction:
        """Apply exactly one selected live action and return cache evidence."""

    def terminal(self) -> R207GameTerminal:
        """Return the exact engine terminal result after ``next_decision`` is ``None``."""

    def runtime_parity_telemetry(self) -> R207RuntimeParityTelemetry:
        """Return the exact shared frozen-r195/adapter runtime identity for both arms."""


R207LocalGameFactory = Callable[[R207LocalGameRequest], R207LocalLiveGame]


def _validate_environment() -> None:
    for key in _FORBIDDEN_ENABLED_ENV:
        value = os.environ.get(key)
        if value is not None and value.strip() not in {"", "0", "false", "False", "off", "OFF"}:
            raise R207LocalGameCommandError(f"forbidden guide/RTP environment is enabled: {key}")
    for key, expected in _REQUIRED_DISABLED_ENV.items():
        observed = os.environ.get(key)
        if observed is not None and observed != expected:
            raise R207LocalGameCommandError(f"r207 environment does not bind {key}={expected}")


def _validate_session(
    session: SimulatorInterTurnMCTSSession,
    binding: R207MCTSRuntimeBinding,
) -> None:
    if not isinstance(session, SimulatorInterTurnMCTSSession):
        raise R207LocalGameCommandError("live game does not expose SimulatorInterTurnMCTSSession")
    expected_config = ChanceAwareSearchConfig()
    if session.config != expected_config or session.config.identity_sha256 != R207_CANONICAL_PLANNER_CONFIG_SHA256:
        raise R207LocalGameCommandError("MCTS session does not use exact r207 20s/5s config")
    if session.deadlines.config_sha256 != R207_CANONICAL_PLANNER_CONFIG_SHA256:
        raise R207LocalGameCommandError("MCTS deadline controller has a different timing identity")
    if session.adapter.arena is not binding.successor.arena:
        raise R207LocalGameCommandError("MCTS session does not use bound real successor arena")
    reranker = session.reranker
    if not isinstance(reranker, BatchedNeuralLeafReranker):
        raise R207LocalGameCommandError("MCTS session lacks batched frozen leaf reranker")
    if reranker.config.frozen_policy_identity != binding.frozen_policy_identity:
        raise R207LocalGameCommandError("leaf reranker frozen policy identity differs from direct policy")
    if reranker.config.calibration != binding.calibration.leaf_calibration:
        raise R207LocalGameCommandError("leaf reranker lacks exact calibrated receipt binding")
    if reranker.config.calibration is None:
        raise R207LocalGameCommandError("leaf reranker has no calibration")
    if reranker.guide_logit_bonus_used is not False:
        raise R207LocalGameCommandError("leaf reranker enables a guide-logit bonus")
    if reranker.production_action_authority_enabled is not False:
        raise R207LocalGameCommandError("leaf reranker grants production authority")
    # The generic reranker identity binds checkpoint/bundle bytes, but r195's
    # serving graph also includes the checksum-verified matchup adapter.  A
    # production backend must expose the exact graph facts; otherwise it could
    # silently score leaves through a nearby no-adapter runtime.
    backend = getattr(reranker, "_backend", None)
    expected_backend_values = {
        "r207_direct_runtime_graph_sha256": binding.direct_runtime_graph_sha256,
        "r207_no_rtp_runtime_receipt_sha256": binding.no_rtp_runtime_receipt_sha256,
        "r207_matchup_tree_sha256": binding.matchup_tree_sha256,
        "r207_matchup_adapter_bank_sha256": binding.matchup_adapter_bank_sha256,
        "r207_matchup_adapter_training_receipt_sha256": (
            binding.matchup_adapter_training_receipt_sha256
        ),
        "r207_matchup_adapter_runtime_graph_sha256": (
            binding.matchup_adapter_runtime_graph_sha256
        ),
        "r207_matchup_adapter_enabled": True,
        "r207_matchup_adapter_trained": True,
        "r207_matchup_adapter_frozen": True,
        "r207_guide2vec_enabled": False,
        "r207_guide_logit_transform_enabled": False,
        "r207_guide_linear_transform_enabled": False,
        "r207_legacy_rtp_sidecar_used": False,
        "r207_legacy_rtp_executor_used": False,
        "r207_leaf_calibration_receipt_sha256": (
            binding.calibration.leaf_calibration.receipt_sha256
        ),
        "r207_leaf_calibration_receipt_file_sha256": (
            binding.calibration.receipt_file_sha256
        ),
    }
    for attribute, expected in expected_backend_values.items():
        if getattr(backend, attribute, None) != expected:
            raise R207LocalGameCommandError(
                f"frozen reranker backend lacks exact r195 runtime assertion: {attribute}"
            )


def _validate_runtime_parity(
    telemetry: R207RuntimeParityTelemetry,
    binding: R207MCTSRuntimeBinding,
    request: R207LocalGameRequest,
) -> None:
    if not isinstance(telemetry, R207RuntimeParityTelemetry):
        raise R207LocalGameCommandError("live game does not expose runtime parity telemetry")
    expected_identity = binding.frozen_policy_identity.identity_sha256
    for arm in (telemetry.mcts_arm, telemetry.direct_arm):
        if arm.frozen_policy_identity_sha256 != expected_identity:
            raise R207LocalGameCommandError("arm telemetry belongs to another frozen policy")
        if arm.no_rtp_runtime_receipt_sha256 != binding.no_rtp_runtime_receipt_sha256:
            raise R207LocalGameCommandError("arm no-RTP runtime receipt differs from binding")
        if arm.direct_runtime_graph_sha256 != binding.direct_runtime_graph_sha256:
            raise R207LocalGameCommandError("arm direct runtime graph differs from binding")
        if arm.matchup_tree_sha256 != request.matchup_tree_sha256:
            raise R207LocalGameCommandError("arm matchup tree differs from immutable request")
        if arm.matchup_adapter_bank_sha256 != binding.matchup_adapter_bank_sha256:
            raise R207LocalGameCommandError("arm matchup adapter bank differs from binding")
        if (
            arm.matchup_adapter_training_receipt_sha256
            != binding.matchup_adapter_training_receipt_sha256
        ):
            raise R207LocalGameCommandError("arm matchup adapter receipt differs from binding")
        if arm.matchup_adapter_runtime_graph_sha256 != binding.matchup_adapter_runtime_graph_sha256:
            raise R207LocalGameCommandError("arm matchup adapter runtime differs from binding")


def _failed_closed_receipt(
    request: R207LocalGameRequest,
    *,
    mcts_turns: Sequence[MCTSTurnTelemetry] = (),
    timeout_count: int = 0,
) -> BO1000GameReceipt:
    spec = request.game_spec
    return BO1000GameReceipt(
        game_nonce_sha256=spec.game_nonce_sha256,
        pair_id=spec.pair_id,
        game_index=spec.game_index,
        mcts_seat=spec.mcts_seat,
        no_rtp_seat=spec.no_rtp_seat,
        pair_rng_snapshot_sha256=request.pair_rng_snapshot_sha256,
        deck_order_rng_sha256=request.deck_order_rng_sha256,
        checkpoint_sha256=R207_FROZEN_CHECKPOINT_SHA256,
        bundle_sha256=R207_FROZEN_BUNDLE_SHA256,
        terminal_status="failed_closed",
        winner_seat=None,
        illegal_action_count=0,
        forfeit_count=0,
        crash_count=1,
        timeout_count=timeout_count,
        mcts_turns=tuple(mcts_turns),
    )


def _material_mcts_turn(turn: MCTSTurnTelemetry) -> bool:
    return (
        turn.actions_dispatched == 1
        and turn.simulator_transitions_seen > 0
        and turn.decision_nodes_expanded > 0
        and turn.frozen_policy_prior_batches > 0
        and turn.frozen_policy_prior_evaluations > 0
        and turn.result_or_leaf_evaluations_seen > 0
    )


class R207LocalGameCommandAdapter:
    """Execute one local r207 game through a receipt-bound live-game factory."""

    def __init__(self, factory: R207LocalGameFactory | None = None) -> None:
        if factory is not None and not callable(factory):
            raise R207LocalGameCommandError("r207 local game factory must be callable")
        self._factory = factory

    def execute(self, request: R207LocalGameRequest) -> BO1000GameReceipt:
        """Return a completed receipt or a binding failed-closed receipt.

        A partially observed MCTS trace remains in a failed-closed receipt so
        timing and simulator work are auditable, but it never counts as a game
        result.  This mirrors the pair runner's fail-closed evidence behavior.
        """

        turns: list[MCTSTurnTelemetry] = []
        try:
            _validate_environment()
            if self._factory is None:
                raise R207LocalGameCommandError("no preflighted r207 local game factory is configured")
            game = self._factory(request)
            if not isinstance(game, R207LocalLiveGame):
                raise R207LocalGameCommandError("factory did not return R207LocalLiveGame")
            binding = game.prerequisites
            if not isinstance(binding, R207MCTSRuntimeBinding):
                raise R207LocalGameCommandError("live game lacks r207 prerequisite binding")
            _validate_session(game.mcts_session, binding)
            _validate_runtime_parity(game.runtime_parity_telemetry(), binding, request)
            seen_turn_ids: set[str] = set()
            seen_turn_keys: set[tuple[int, int]] = set()
            for action_ordinal in range(_MAX_GAME_ACTIONS):
                view = game.next_decision()
                if view is None:
                    terminal = game.terminal()
                    if not isinstance(terminal, R207GameTerminal):
                        raise R207LocalGameCommandError("live game terminal facts are invalid")
                    if not turns or not any(_material_mcts_turn(turn) for turn in turns):
                        raise R207LocalGameCommandError("completed game has no material MCTS telemetry")
                    spec = request.game_spec
                    return BO1000GameReceipt(
                        game_nonce_sha256=spec.game_nonce_sha256,
                        pair_id=spec.pair_id,
                        game_index=spec.game_index,
                        mcts_seat=spec.mcts_seat,
                        no_rtp_seat=spec.no_rtp_seat,
                        pair_rng_snapshot_sha256=request.pair_rng_snapshot_sha256,
                        deck_order_rng_sha256=request.deck_order_rng_sha256,
                        checkpoint_sha256=R207_FROZEN_CHECKPOINT_SHA256,
                        bundle_sha256=R207_FROZEN_BUNDLE_SHA256,
                        terminal_status="completed",
                        winner_seat=terminal.winner_seat,
                        illegal_action_count=terminal.illegal_action_count,
                        forfeit_count=terminal.forfeit_count,
                        crash_count=terminal.crash_count,
                        timeout_count=terminal.timeout_count,
                        mcts_turns=tuple(turns),
                    )
                if not isinstance(view, PolicyDecisionView):
                    raise R207LocalGameCommandError("live game emitted a non-policy-visible decision")
                if view.acting_seat == request.game_spec.mcts_seat:
                    planner_turn_id = (
                        f"{request.game_spec.game_nonce_sha256.removeprefix('sha256:')[:16]}"
                        f":{view.turn_key[0]}:{view.turn_key[1]}:{action_ordinal}"
                    )
                    result = game.mcts_session.plan_next(
                        planner_turn_id=planner_turn_id,
                        seat=request.game_spec.mcts_seat,
                        real_view=view,
                    )
                    selected_action = _action(result.selected_action, label="MCTS selected_action")
                    if selected_action not in view.legal_actions:
                        raise R207LocalGameCommandError("MCTS selected an action outside exact legal set")
                    turn = result.telemetry.to_bo1000_turn_telemetry(
                        game_nonce_sha256=request.game_spec.game_nonce_sha256,
                        pair_id=request.game_spec.pair_id,
                        mcts_seat=request.game_spec.mcts_seat,
                        selected_action=selected_action,
                        legal_actions=view.legal_actions,
                    )
                    if not isinstance(turn, MCTSTurnTelemetry):
                        raise R207LocalGameCommandError("MCTS telemetry did not adapt to BO1000 schema")
                    if turn.planner_turn_id in seen_turn_ids or turn.turn_key in seen_turn_keys:
                        raise R207LocalGameCommandError("live game repeated an MCTS turn identity")
                    seen_turn_ids.add(turn.planner_turn_id)
                    seen_turn_keys.add(turn.turn_key)
                    turns.append(turn)
                    applied = game.apply_action(selected_action)
                    if not isinstance(applied, R207AppliedAction):
                        raise R207LocalGameCommandError("live game action result is invalid")
                    game.mcts_session.observe_real_action(
                        action=selected_action,
                        next_view=applied.next_mcts_view,
                        realized_transition_kind=applied.realized_transition_kind,
                    )
                elif view.acting_seat == request.game_spec.no_rtp_seat:
                    # The control arm is *only* the exact base action already
                    # produced by the same frozen no-RTP model in the view.
                    # It receives no MCTS call, guide transform, or RTP bridge.
                    direct_action = _action(view.direct_action, label="direct policy action")
                    if direct_action not in view.legal_actions:
                        raise R207LocalGameCommandError("direct policy action is not exactly legal")
                    applied = game.apply_action(direct_action)
                    if not isinstance(applied, R207AppliedAction):
                        raise R207LocalGameCommandError("live game action result is invalid")
                else:
                    raise R207LocalGameCommandError("live decision actor is outside both immutable seats")
            raise R207LocalGameCommandError("live game exceeded exact atomic-action ceiling")
        except Exception:  # noqa: BLE001 - every runtime defect becomes immutable failed-closed evidence.
            timeout_count = int(any(turn.deadline_hit for turn in turns))
            return _failed_closed_receipt(
                request,
                mcts_turns=turns,
                timeout_count=timeout_count,
            )

    def execute_payload(self, raw: Mapping[str, object]) -> BO1000GameReceipt:
        return self.execute(parse_r207_local_game_request(raw))


def _load_factory(reference: str | None) -> R207LocalGameFactory | None:
    if reference is None or not reference.strip():
        return None
    module_name, separator, attribute = reference.strip().partition(":")
    if not separator or not module_name or not attribute:
        raise R207LocalGameCommandError("runtime factory must use module:attribute syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute)
    if not callable(factory):
        raise R207LocalGameCommandError("runtime factory attribute is not callable")
    return factory


def command_main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint used by ``HostLocalSubprocessGameRunner``.

    A factory is intentionally optional at parse time.  With no supplied
    factory the command still emits a canonical failed-closed receipt for a
    valid request, which is safer and more useful than pretending an MCTS game
    was run with a direct-policy fallback.
    """

    parser = argparse.ArgumentParser(description="Run one local r207 MCTS BO1000 game")
    parser.add_argument("request_path", type=Path)
    parser.add_argument("--runtime-factory", default=os.environ.get(R207_LOCAL_RUNTIME_FACTORY_ENV))
    arguments = parser.parse_args(argv)
    try:
        raw = json.loads(arguments.request_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise R207LocalGameCommandError("request JSON must be an object")
        request = parse_r207_local_game_request(raw)
        adapter = R207LocalGameCommandAdapter(_load_factory(arguments.runtime_factory))
        receipt = adapter.execute(request)
    except (OSError, json.JSONDecodeError, R207LocalGameCommandError, ImportError, AttributeError) as exc:
        # A malformed request cannot safely be bound to a receipt.  Let the
        # pair runner create its canonical failed-closed game receipt instead.
        print(f"r207 local game command failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(asdict(receipt), sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests.
    raise SystemExit(command_main())


__all__ = [
    "R207_LOCAL_GAME_COMMAND_SCHEMA",
    "R207_LOCAL_RUNTIME_FACTORY_ENV",
    "R207_R195_MATCHUP_TREE_SHA256",
    "R207_REAL_SUCCESSOR_INTERFACE_KIND",
    "R207ArmRuntimeTelemetry",
    "R207AppliedAction",
    "R207GameTerminal",
    "R207FrozenLeafCalibrationBinding",
    "R207LocalGameCommandAdapter",
    "R207LocalGameCommandError",
    "R207LocalGameFactory",
    "R207LocalGameRequest",
    "R207LocalLiveGame",
    "R207MCTSRuntimeBinding",
    "R207RealSuccessorBinding",
    "R207RuntimeParityTelemetry",
    "command_main",
    "parse_r207_local_game_request",
]
