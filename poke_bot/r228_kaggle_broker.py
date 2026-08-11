"""Crash-contained subprocess broker for the r238 two-lane Kaggle runtime.

The Phase-1 runtime intentionally exercises exactly two raw ``libcg`` Search
handles, matching Kaggle's two-vCPU allocation.  A native Search call can block
forever or terminate the process, so it must never run in the competition-agent
controller itself.  This module keeps the controller in one process and starts
a fresh Python interpreter for the native runtime.  The two processes
communicate only through newline JSON on a Unix socket; no Python
multiprocessing, forked CUDA state, model object, or CUDA tensor crosses the
boundary.

The public controller is deliberately small and package-local.  It is intended
to be used by the r228 submission entrypoint, not by the BO1000 fleet.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
import select
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA = "poke_bot.r228_kaggle_subprocess_broker/v1"
COMPLETE_ACTION_CAP = 65_536
SIMULATOR_SEARCH_LANE_COUNT = 2
R238_MINIMUM_BACKUPS_BEFORE_STABILITY = 8
R238_STABLE_ROOT_LEADER_OBSERVATIONS_REQUIRED = 3
R238_MAXIMUM_BACKUPS_PER_DECISION = 32
PROVEN_TERMINAL_WIN_REVISION = 246
PROVEN_TERMINAL_WIN_STOP_REASON = (
    "proven_deterministic_terminal_win_this_turn"
)
PROVEN_TERMINAL_WIN_PROOF_KIND = (
    "exact_deterministic_simulator_terminal_win_this_turn"
)
PHASE1_THREAD_CAPS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
_MAX_MESSAGE_BYTES = 32 * 1024 * 1024
_MAX_PROGRESS_EVENTS = 256
_CHILD_CLEAN_ZERO_MODE = "clean_deadline_zero_backup_frozen_model_fallback"
_PARENT_CLEAN_ZERO_MODE = "zero_backup_precomputed_direct_fallback"
CUDA_RUNTIME_OBSERVATION_SCHEMA = "poke_bot.r238_cuda_runtime_observation/v1"
CUDA_RUNTIME_OBSERVATION_PHASE = "before_search"


class R228BrokerError(RuntimeError):
    """A broker transport, child, or result-integrity failure."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "broker_error",
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.detail = dict(detail or {})


def _positive_seconds(value: object, *, fallback: float, label: str) -> float:
    """Return one finite positive wall-clock limit without silent zero values."""

    try:
        parsed = float(value) if value is not None else float(fallback)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive finite number") from exc
    if not parsed > 0.0 or parsed == float("inf") or parsed != parsed:
        raise ValueError(f"{label} must be a positive finite number")
    return parsed


def _json_copy(value: Any) -> Any:
    """Make the IPC boundary explicit and reject non-JSON observations."""

    try:
        return json.loads(json.dumps(value, allow_nan=False, separators=(",", ":")))
    except (TypeError, ValueError) as exc:
        raise R228BrokerError(
            "broker request is not JSON-native",
            code="non_json_request",
        ) from exc


def capture_cuda_runtime_before_search(model: Any) -> dict[str, Any]:
    """Return non-authoritative CUDA/model telemetry after a model has loaded.

    Kaggle's published Phase-1 CPU/RAM envelope does not prove that CUDA is
    absent.  The frozen r195 entrypoint already chooses ``cuda`` whenever
    PyTorch reports it, so this probe deliberately observes that existing
    choice rather than changing it.  It never reads environment variables,
    device UUIDs, serials, or process state, and it intentionally absorbs
    telemetry failures: a missing diagnostic must not change direct-policy or
    MCTS action authority.

    Callers invoke this only after their frozen model is available and before
    the first MCTS/search request.  In the normal CUDA path the model load has
    already initialized CUDA, making ``mem_get_info`` observational rather
    than a new accelerator-use decision.
    """

    errors: list[str] = []
    model_device = "unavailable"
    try:
        parameters = getattr(model, "parameters", None)
        if not callable(parameters):
            raise TypeError("model_has_no_parameters")
        first_parameter = next(iter(parameters()))
        device = getattr(first_parameter, "device", None)
        if device is None:
            raise TypeError("model_parameter_has_no_device")
        candidate = str(device).strip()
        if not candidate:
            raise ValueError("model_device_empty")
        model_device = candidate
    except Exception as exc:  # telemetry must never alter action authority
        errors.append(f"model_device:{type(exc).__name__}")

    payload: dict[str, Any] = {
        "schema": CUDA_RUNTIME_OBSERVATION_SCHEMA,
        "phase": CUDA_RUNTIME_OBSERVATION_PHASE,
        "torch_imported": False,
        "cuda_available": False,
        "cuda_initialized": False,
        "device_count": 0,
        "devices": [],
        "model_device": model_device,
        "telemetry_complete": False,
        "error_types": errors,
    }
    torch_module = sys.modules.get("torch")
    if torch_module is None:
        errors.append("torch:NotImported")
        return payload
    payload["torch_imported"] = True
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None:
        errors.append("torch_cuda:Unavailable")
        return payload

    try:
        available = bool(cuda.is_available())
        initialized = bool(cuda.is_initialized())
        device_count = int(cuda.device_count())
        if device_count < 0:
            raise ValueError("negative_device_count")
    except Exception as exc:  # telemetry must never alter action authority
        errors.append(f"cuda_status:{type(exc).__name__}")
        return payload

    payload.update(
        {
            "cuda_available": available,
            "cuda_initialized": initialized,
            "device_count": device_count,
        }
    )
    if not available:
        if device_count != 0 or initialized:
            errors.append("cuda_status:UnavailableButInitializedOrEnumerated")
            return payload
        payload["telemetry_complete"] = not errors
        return payload
    # Avoid creating a CUDA context purely to collect diagnostics.  The frozen
    # r195 loader has already selected and loaded the model onto CUDA in the
    # normal visible-GPU path, so an uninitialized CUDA report is an anomalous
    # incomplete observation rather than a reason for this probe to initialize
    # it itself.
    if not initialized:
        errors.append("cuda_status:AvailableButNotInitialized")
        return payload

    devices: list[dict[str, Any]] = []
    for index in range(device_count):
        try:
            name = str(cuda.get_device_name(index)).strip()
            free_bytes, total_bytes = cuda.mem_get_info(index)
            free = int(free_bytes)
            total = int(total_bytes)
            if not name or total <= 0 or free < 0 or free > total:
                raise ValueError("invalid_device_memory")
            devices.append(
                {
                    "device_index": index,
                    "device_name": name,
                    "total_memory_bytes": total,
                    "free_memory_bytes": free,
                }
            )
        except Exception as exc:  # telemetry must never alter action authority
            errors.append(f"cuda_device_{index}:{type(exc).__name__}")
            break
    payload["devices"] = devices
    payload["telemetry_complete"] = (
        not errors and len(devices) == device_count and device_count > 0
    )
    return payload


def _as_action(value: object, *, label: str) -> list[int]:
    if not isinstance(value, list):
        raise R228BrokerError(f"{label} must be a list", code="malformed_action")
    action: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise R228BrokerError(
                f"{label} contains a non-integer index", code="malformed_action"
            )
        action.append(int(item))
    return action


def _syntactically_legal(obs: Mapping[str, Any], action: Sequence[int]) -> bool:
    """Check engine selection shape without materializing an action space."""

    selection = obs.get("select")
    if not isinstance(selection, Mapping):
        return False
    options = selection.get("option")
    if not isinstance(options, list):
        return False
    raw_min = selection.get("minCount", 0)
    raw_max = selection.get("maxCount", 0)
    if isinstance(raw_min, bool) or isinstance(raw_max, bool):
        return False
    if not isinstance(raw_min, int) or not isinstance(raw_max, int):
        return False
    lower = max(0, min(int(raw_min), len(options)))
    upper = max(lower, min(int(raw_max), len(options)))
    normalized = list(action)
    return (
        lower <= len(normalized) <= upper
        and len(set(normalized)) == len(normalized)
        and all(0 <= int(index) < len(options) for index in normalized)
    )


def _complete_legal_order(obs: Mapping[str, Any]) -> tuple[tuple[int, ...], ...]:
    """Materialize the bounded complete ordered action list for validation."""

    try:
        from poke_bot import features

        actions = features.enumerate_action_combos(
            dict(obs), max_combos=COMPLETE_ACTION_CAP
        )
    except Exception as exc:  # ActionSpaceTooLarge is intentionally a fault.
        raise R228BrokerError(
            f"complete legal action enumeration failed: {type(exc).__name__}: {exc}",
            code="complete_action_enumeration_failed",
        ) from exc
    normalized = tuple(tuple(int(item) for item in action) for action in actions)
    if not normalized or len(set(normalized)) != len(normalized):
        raise R228BrokerError(
            "complete legal action enumeration returned an empty or duplicate order",
            code="complete_action_enumeration_failed",
        )
    return normalized


def _complete_legal_actions(obs: Mapping[str, Any]) -> set[tuple[int, ...]]:
    """Backward-compatible set view of the complete ordered legal actions."""

    return set(_complete_legal_order(obs))


def _canonical_observation_fingerprint(obs: Mapping[str, Any]) -> str:
    try:
        canonical = json.dumps(
            dict(obs),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise R228BrokerError(
            "current root observation is not canonical JSON",
            code="terminal_win_root_binding_invalid",
        ) from exc
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _legal_order_fingerprint(legal: Sequence[Sequence[int]]) -> str:
    canonical = json.dumps(
        [[int(item) for item in action] for action in legal],
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _current_actor_seat(obs: Mapping[str, Any]) -> int:
    current = obs.get("current")
    actor = current.get("yourIndex") if isinstance(current, Mapping) else None
    if isinstance(actor, bool) or not isinstance(actor, int) or actor not in (0, 1):
        raise R228BrokerError(
            "current root actor is invalid",
            code="terminal_win_root_binding_invalid",
        )
    return int(actor)


def _exact_int(value: object, *, field: str) -> int:
    """Accept one JSON integer without bool/coercion ambiguity."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise R228BrokerError(
            f"two-lane receipt field {field!r} must be an exact integer",
            code="two_lane_receipt_invalid",
        )
    return int(value)


def _validate_lane_id(value: object, *, field: str) -> int:
    lane = _exact_int(value, field=field)
    if not 0 <= lane < SIMULATOR_SEARCH_LANE_COUNT:
        raise R228BrokerError(
            f"two-lane telemetry {field!r}={lane!r} is outside lane ids 0..1",
            code="two_lane_progress_lane_invalid",
        )
    return lane


def _validate_handle_identity(value: object, *, field: str) -> int | str:
    """Accept one nonempty JSON-native raw AgentStart handle identity."""

    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise R228BrokerError(
            f"two-lane receipt {field!r} must be an integer or string handle identity",
            code="two_lane_receipt_handle_identity_invalid",
        )
    if isinstance(value, str) and not value:
        raise R228BrokerError(
            f"two-lane receipt {field!r} must not be an empty handle identity",
            code="two_lane_receipt_handle_identity_invalid",
        )
    return value


def _validate_handle_scoped_first_search_id_composites(
    payload: Mapping[str, Any],
    *,
    handles: Sequence[int | str],
    first_search_ids: Sequence[int],
) -> None:
    """Bind lane order to official handle-local first SearchIds."""

    expected = SIMULATOR_SEARCH_LANE_COUNT
    states = payload.get("handle_scoped_first_search_id_composite_states")
    if not isinstance(states, (list, tuple)) or len(states) != expected:
        raise R228BrokerError(
            "two-lane receipt lacks both handle-scoped SearchBegin composites",
            code="two_lane_receipt_lane_vector_invalid",
        )
    first_ids = payload.get("per_lane_first_search_ids")
    if not isinstance(first_ids, (list, tuple)) or len(first_ids) != expected:
        raise R228BrokerError(
            "two-lane receipt lacks both lane-first SearchIds",
            code="two_lane_receipt_lane_vector_invalid",
        )
    normalized_first_ids = [
        _exact_int(value, field="per_lane_first_search_ids") for value in first_ids
    ]
    if normalized_first_ids != list(first_search_ids):
        raise R228BrokerError(
            "two-lane receipt lane-first SearchIds disagree with SearchBegin chains",
            code="two_lane_receipt_lane_vector_invalid",
        )
    for lane_id, state in enumerate(states):
        if not isinstance(state, Mapping) or set(state) != {
            "lane_id",
            "handle_identity",
            "first_search_id",
        }:
            raise R228BrokerError(
                "two-lane receipt has a malformed handle-scoped SearchBegin composite",
                code="two_lane_receipt_lane_vector_invalid",
            )
        if _validate_lane_id(state.get("lane_id"), field="lane_id") != lane_id:
            raise R228BrokerError(
                "two-lane receipt handle-scoped composite is not lane ordered",
                code="two_lane_receipt_lane_vector_invalid",
            )
        if (
            _validate_handle_identity(
                state.get("handle_identity"), field="handle_identity"
            )
            != handles[lane_id]
            or _exact_int(state.get("first_search_id"), field="first_search_id")
            != first_search_ids[lane_id]
        ):
            raise R228BrokerError(
                "two-lane receipt handle-scoped composite disagrees with lane vectors",
                code="two_lane_receipt_lane_vector_invalid",
            )


def _validate_progress_lanes(payload: Mapping[str, Any]) -> None:
    """Reject a child telemetry row that claims a non-Phase-1 lane."""

    for field in ("lane_id", "lane"):
        if field in payload:
            _validate_lane_id(payload[field], field=field)
    if "lane_count" in payload:
        observed = _exact_int(payload["lane_count"], field="lane_count")
        if observed != SIMULATOR_SEARCH_LANE_COUNT:
            raise R228BrokerError(
                f"two-lane telemetry lane_count={observed!r}, expected "
                f"{SIMULATOR_SEARCH_LANE_COUNT}",
                code="two_lane_progress_count_invalid",
            )
    for field in ("pending_lanes", "lanes"):
        if field not in payload:
            continue
        lanes = payload[field]
        if not isinstance(lanes, (list, tuple)):
            raise R228BrokerError(
                f"two-lane telemetry {field!r} must be a lane-id list",
                code="two_lane_progress_lane_invalid",
            )
        normalized = [_validate_lane_id(value, field=field) for value in lanes]
        if len(set(normalized)) != len(normalized):
            raise R228BrokerError(
                f"two-lane telemetry {field!r} repeats a lane id",
                code="two_lane_progress_lane_invalid",
            )
        # Ready-leaf microbatches may contain either one or both lanes.  The
        # exact two-lane topology is proved independently by the opened handle
        # vectors; in particular r246 may stop as soon as one lane returns an
        # exact terminal win while the other native call is boundedly drained.
        if field == "lanes" and not normalized:
            raise R228BrokerError(
                "two-lane evaluator telemetry contained no ready lane",
                code="two_lane_progress_batch_invalid",
            )


_TERMINAL_WIN_PROOF_KEYS = frozenset(
    {
        "proof_kind",
        "root_observation_fingerprint",
        "root_legal_order_fingerprint",
        "root_actor_seat",
        "root_action",
        "selected_action",
        "terminal_result",
        "terminal_winner_seat",
        "terminal_leaf_reached",
        "proof_path_action_count",
        "path_actor_seats",
        "path_no_actor_change_boundary",
        "path_no_opponent_boundary_crossing",
        "path_no_chance_boundary",
        "path_no_unresolved_randomness",
        "proof_is_deterministic",
        "discovering_lane_id",
    }
)


def _validate_terminal_win_proof(
    payload: Mapping[str, Any],
    *,
    observation: Mapping[str, Any],
    legal_order: Sequence[Sequence[int]],
    selected_action: Sequence[int],
) -> dict[str, Any]:
    """Independently bind the child's r246 exception to this IPC request."""

    proof = payload.get("terminal_win_proof")
    if not isinstance(proof, Mapping) or set(proof) != set(_TERMINAL_WIN_PROOF_KEYS):
        raise R228BrokerError(
            "terminal-win proof has an invalid exact schema",
            code="terminal_win_proof_invalid",
        )
    if _exact_int(
        payload.get("owner_proven_deterministic_terminal_win_this_turn_revision"),
        field="owner_proven_deterministic_terminal_win_this_turn_revision",
    ) != PROVEN_TERMINAL_WIN_REVISION:
        raise R228BrokerError(
            "terminal-win proof owner revision is invalid",
            code="terminal_win_proof_invalid",
        )
    root_fingerprint = _canonical_observation_fingerprint(observation)
    legal_fingerprint = _legal_order_fingerprint(legal_order)
    actor = _current_actor_seat(observation)
    if (
        payload.get("root_observation_fingerprint") != root_fingerprint
        or proof.get("root_observation_fingerprint") != root_fingerprint
    ):
        raise R228BrokerError(
            "terminal-win proof is stale for the current observation",
            code="terminal_win_proof_stale",
        )
    if (
        payload.get("root_legal_order_fingerprint") != legal_fingerprint
        or proof.get("root_legal_order_fingerprint") != legal_fingerprint
    ):
        raise R228BrokerError(
            "terminal-win proof is stale for the current legal order",
            code="terminal_win_proof_stale",
        )
    if (
        _exact_int(payload.get("root_actor_seat"), field="root_actor_seat")
        != actor
        or _exact_int(payload.get("root_seat"), field="root_seat") != actor
        or _exact_int(proof.get("root_actor_seat"), field="root_actor_seat")
        != actor
    ):
        raise R228BrokerError(
            "terminal-win proof is stale for the current actor",
            code="terminal_win_proof_stale",
        )
    if proof.get("proof_kind") != PROVEN_TERMINAL_WIN_PROOF_KIND:
        raise R228BrokerError(
            "terminal-win proof kind is invalid", code="terminal_win_proof_invalid"
        )
    if proof.get("terminal_result") != "win" or _exact_int(
        proof.get("terminal_winner_seat"), field="terminal_winner_seat"
    ) != actor:
        raise R228BrokerError(
            "terminal-win proof is not a win for the current actor",
            code="terminal_win_proof_invalid",
        )
    for field in (
        "terminal_leaf_reached",
        "path_no_actor_change_boundary",
        "path_no_opponent_boundary_crossing",
        "path_no_chance_boundary",
        "path_no_unresolved_randomness",
        "proof_is_deterministic",
    ):
        if proof.get(field) is not True:
            raise R228BrokerError(
                f"terminal-win proof does not prove {field}",
                code="terminal_win_proof_invalid",
            )
    root_action = _as_action(proof.get("root_action"), label="proof root_action")
    proof_selected = _as_action(
        proof.get("selected_action"), label="proof selected_action"
    )
    normalized_selected = [int(item) for item in selected_action]
    normalized_legal = {
        tuple(int(item) for item in action) for action in legal_order
    }
    if (
        root_action != proof_selected
        or proof_selected != normalized_selected
        or tuple(proof_selected) not in normalized_legal
        or _as_action(payload.get("selected_action"), label="receipt selected_action")
        != normalized_selected
    ):
        raise R228BrokerError(
            "terminal-win proof does not bind its selected legal root action",
            code="terminal_win_proof_invalid",
        )
    action_count = _exact_int(
        proof.get("proof_path_action_count"), field="proof_path_action_count"
    )
    actor_path = proof.get("path_actor_seats")
    if (
        action_count < 1
        or action_count > _exact_int(
            payload.get("completed_backups"), field="completed_backups"
        )
        or not isinstance(actor_path, (list, tuple))
        or len(actor_path) != action_count
        or any(
            isinstance(path_actor, bool)
            or not isinstance(path_actor, int)
            or path_actor != actor
            for path_actor in actor_path
        )
    ):
        raise R228BrokerError(
            "terminal-win proof path is not root-actor-only",
            code="terminal_win_proof_invalid",
        )
    discovering_lane = _validate_lane_id(
        proof.get("discovering_lane_id"), field="discovering_lane_id"
    )
    depths = payload.get("per_lane_depth")
    if not isinstance(depths, (list, tuple)) or _exact_int(
        depths[discovering_lane], field="per_lane_depth"
    ) < action_count:
        raise R228BrokerError(
            "terminal-win proof exceeds its backed lane depth",
            code="terminal_win_proof_invalid",
        )
    if payload.get("principal_variation") not in ([], ()):
        raise R228BrokerError(
            "terminal-win proof retained a continuation plan",
            code="terminal_win_proof_invalid",
        )
    if payload.get("proven_deterministic_terminal_win_this_turn") is not True:
        raise R228BrokerError(
            "terminal-win receipt omitted its public proof classification",
            code="terminal_win_proof_invalid",
        )
    return dict(proof)


def _validate_terminal_win_execution_marker_facts(
    payload: Mapping[str, Any],
) -> None:
    """Reject a terminal IPC receipt missing its literal runtime facts.

    The child runtime writes these only after its exact proof, shared-root
    backup, and lane cleanup validations complete.  The broker validates the
    literal fields before it adds its own IPC/start facts, so a stale child
    cannot be relabelled by the parent or a probe converter.
    """

    for field in (
        "two_lane_topology_initialized_before_terminal_win_override",
        "terminal_win_proof_backed_up_into_shared_root_tree",
        "terminal_leaf_returned_by_exact_stock_simulator",
        "all_owned_lane_resources_reservations_and_child_cleanup_complete",
    ):
        if payload.get(field) is not True:
            raise R228BrokerError(
                f"terminal-win receipt lacks literal {field}",
                code="terminal_win_proof_invalid",
            )
    completed_backups = _exact_int(
        payload.get("completed_backups"), field="completed_backups"
    )
    if _exact_int(
        payload.get("completed_root_backup_count"),
        field="completed_root_backup_count",
    ) != completed_backups:
        raise R228BrokerError(
            "terminal-win root backup count disagrees with the native receipt",
            code="terminal_win_proof_invalid",
        )
    for field in (
        "terminal_win_proof_count",
        "proven_deterministic_terminal_win_this_turn_stop_count",
    ):
        if _exact_int(payload.get(field), field=field) != 1:
            raise R228BrokerError(
                f"terminal-win receipt has invalid {field}",
                code="terminal_win_proof_invalid",
            )
    value = payload.get("child_search_elapsed_seconds")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise R228BrokerError(
            "terminal-win receipt lacks a numeric child search elapsed time",
            code="terminal_win_proof_invalid",
        )
    elapsed_seconds = float(value)
    if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0.0:
        raise R228BrokerError(
            "terminal-win receipt has an invalid child search elapsed time",
            code="terminal_win_proof_invalid",
        )


def _validate_two_lane_receipt(
    receipt: Mapping[str, Any],
    *,
    observation: Mapping[str, Any] | None = None,
    legal_order: Sequence[Sequence[int]] | None = None,
    selected_action: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Require the child search receipt to prove exactly two live lanes."""

    payload = dict(receipt)
    expected = SIMULATOR_SEARCH_LANE_COUNT
    stop_reason = payload.get("stop_reason")
    if stop_reason not in {
        "stable_root_leader",
        "maximum_backups",
        "decision_deadline",
        "tree_exhausted",
        PROVEN_TERMINAL_WIN_STOP_REASON,
    }:
        raise R228BrokerError(
            "two-lane receipt has an unrecognized adaptive stop reason",
            code="adaptive_stop_receipt_invalid",
        )
    terminal_win_stop = stop_reason == PROVEN_TERMINAL_WIN_STOP_REASON
    for field in (
        "requested_simulator_lane_count",
        "active_simulator_lane_count",
        "arena_count",
        "unique_handle_count",
        "search_begin_calls",
        "search_end_calls",
    ):
        observed = _exact_int(payload.get(field), field=field)
        if observed != expected:
            raise R228BrokerError(
                f"two-lane receipt {field}={observed!r}, expected {expected}",
                code="two_lane_receipt_count_invalid",
            )

    releases = _exact_int(
        payload.get("search_release_calls"), field="search_release_calls"
    )
    if releases < expected:
        raise R228BrokerError(
            "two-lane receipt released fewer native searches than its two lanes",
            code="two_lane_receipt_count_invalid",
        )

    depths = payload.get("per_lane_depth")
    if not isinstance(depths, (list, tuple)) or len(depths) != expected:
        raise R228BrokerError(
            "two-lane receipt has an incomplete per_lane_depth vector",
            code="two_lane_receipt_lane_vector_invalid",
        )
    minimum_depth = 0 if terminal_win_stop else 1
    normalized_depths = [
        _exact_int(depth, field="per_lane_depth") for depth in depths
    ]
    if any(depth < minimum_depth for depth in normalized_depths):
        raise R228BrokerError(
            "two-lane receipt has a lane without a completed search step",
            code="two_lane_receipt_lane_vector_invalid",
        )
    if terminal_win_stop and not any(depth >= 1 for depth in normalized_depths):
        raise R228BrokerError(
            "terminal-win receipt has no backed discovering lane",
            code="terminal_win_proof_invalid",
        )

    chains = payload.get("per_lane_search_id_chains")
    if not isinstance(chains, (list, tuple)) or len(chains) != expected:
        raise R228BrokerError(
            "two-lane receipt has an incomplete SearchBegin-id vector",
            code="two_lane_receipt_lane_vector_invalid",
        )
    handles = payload.get("per_lane_handle_identities")
    if not isinstance(handles, (list, tuple)) or len(handles) != expected:
        raise R228BrokerError(
            "two-lane receipt has an incomplete raw-handle identity vector",
            code="two_lane_receipt_handle_identity_invalid",
        )
    normalized_handles = [
        _validate_handle_identity(value, field="per_lane_handle_identities")
        for value in handles
    ]
    if len(set(normalized_handles)) != expected:
        raise R228BrokerError(
            "two-lane receipt did not prove two distinct raw AgentStart handles",
            code="two_lane_receipt_handle_identity_invalid",
        )

    first_search_ids: list[int] = []
    for chain in chains:
        if not isinstance(chain, (list, tuple)) or not chain:
            raise R228BrokerError(
                "two-lane receipt has a lane without a SearchBegin id",
                code="two_lane_receipt_lane_vector_invalid",
            )
        first_search_ids.append(
            _exact_int(chain[0], field="per_lane_search_id_chains")
        )
    # Official libcg SearchIds are handle-local; two distinct arenas may both
    # start at raw SearchId 0.  The topology witness is the composite state,
    # never a globally unique raw SearchId.
    composites = set(zip(normalized_handles, first_search_ids))
    if len(composites) != expected:
        raise R228BrokerError(
            "two-lane receipt did not prove two distinct handle/SearchBegin composites",
            code="two_lane_receipt_lane_vector_invalid",
        )
    if _exact_int(
        payload.get("distinct_search_begin_composite_count"),
        field="distinct_search_begin_composite_count",
    ) != expected:
        raise R228BrokerError(
            "two-lane receipt composite SearchBegin count is invalid",
            code="two_lane_receipt_lane_vector_invalid",
        )
    _validate_handle_scoped_first_search_id_composites(
        payload,
        handles=normalized_handles,
        first_search_ids=first_search_ids,
    )

    microbatches = payload.get("microbatch_sizes")
    if not isinstance(microbatches, (list, tuple)) or not microbatches:
        raise R228BrokerError(
            "two-lane receipt omitted its frozen-evaluator microbatches",
            code="two_lane_receipt_batch_invalid",
        )
    if any(
        not 1 <= _exact_int(size, field="microbatch_sizes") <= expected
        for size in microbatches
    ):
        raise R228BrokerError(
            "two-lane receipt used an invalid frozen-evaluator batch size",
            code="two_lane_receipt_batch_invalid",
        )
    in_flight = _exact_int(
        payload.get("max_simulator_calls_in_flight"),
        field="max_simulator_calls_in_flight",
    )
    if not 1 <= in_flight <= expected:
        raise R228BrokerError(
            "two-lane receipt has an invalid in-flight simulator count",
            code="two_lane_receipt_batch_invalid",
        )
    if _exact_int(
        payload.get("outstanding_virtual_loss"), field="outstanding_virtual_loss"
    ) != 0:
        raise R228BrokerError(
            "two-lane receipt returned with outstanding virtual loss",
            code="two_lane_receipt_cleanup_invalid",
        )

    completed_backups = _exact_int(
        payload.get("completed_backups"), field="completed_backups"
    )
    minimum_backups = 1 if terminal_win_stop else expected
    if (
        completed_backups < minimum_backups
        or completed_backups > R238_MAXIMUM_BACKUPS_PER_DECISION
        or sum(normalized_depths) != completed_backups
    ):
        raise R228BrokerError(
            "two-lane receipt backup count is invalid for its stop",
            code="adaptive_stop_receipt_invalid",
        )
    adaptive_expected = {
        "minimum_backups_before_stability": R238_MINIMUM_BACKUPS_BEFORE_STABILITY,
        "stable_root_leader_observations_required": (
            R238_STABLE_ROOT_LEADER_OBSERVATIONS_REQUIRED
        ),
        "maximum_backups_per_decision": R238_MAXIMUM_BACKUPS_PER_DECISION,
    }
    for field, expected_value in adaptive_expected.items():
        if _exact_int(payload.get(field), field=field) != expected_value:
            raise R228BrokerError(
                f"two-lane receipt {field} does not bind r238 adaptive limits",
                code="adaptive_stop_receipt_invalid",
            )
    observed_stable = _exact_int(
        payload.get("observed_stable_root_leader_observations"),
        field="observed_stable_root_leader_observations",
    )
    if observed_stable < 0 or observed_stable > R238_MAXIMUM_BACKUPS_PER_DECISION:
        raise R228BrokerError(
            "two-lane receipt has an invalid observed leader-stability count",
            code="adaptive_stop_receipt_invalid",
        )
    if (
        stop_reason == "stable_root_leader"
        and observed_stable < R238_STABLE_ROOT_LEADER_OBSERVATIONS_REQUIRED
    ):
        raise R228BrokerError(
            "stable-root stop lacks the required leader observations",
            code="adaptive_stop_receipt_invalid",
        )
    if terminal_win_stop:
        if observation is None or legal_order is None or selected_action is None:
            raise R228BrokerError(
                "terminal-win proof lacks the current broker request binding",
                code="terminal_win_root_binding_invalid",
            )
        payload["terminal_win_proof"] = _validate_terminal_win_proof(
            payload,
            observation=observation,
            legal_order=legal_order,
            selected_action=selected_action,
        )
        _validate_terminal_win_execution_marker_facts(payload)
    else:
        if payload.get("terminal_win_proof") is not None or payload.get(
            "proven_deterministic_terminal_win_this_turn"
        ) not in (None, False):
            raise R228BrokerError(
                "non-terminal stop claimed terminal-win authority",
                code="terminal_win_proof_invalid",
            )

    payload["configured_simulator_lane_count"] = expected
    return payload


def _validate_clean_zero_backup_receipt(
    receipt: Mapping[str, Any], *, direct_action: Sequence[int]
) -> dict[str, Any]:
    """Validate a clean, zero-backup deadline before parent-direct authority.

    This is deliberately narrower than a normal MCTS result: it proves the
    two native lanes were opened and fully cleaned, but gives the child no
    action authority.  The caller returns *only* its already precomputed
    parent action after reaping this exact child.
    """

    payload = dict(receipt)
    if payload.get("mode") != _CHILD_CLEAN_ZERO_MODE:
        raise R228BrokerError(
            "child did not report the clean zero-backup deadline mode",
            code="clean_zero_receipt_invalid",
        )
    if payload.get("mcts_action_authority") is not False:
        raise R228BrokerError(
            "clean zero-backup child receipt claimed MCTS action authority",
            code="clean_zero_receipt_invalid",
        )
    if _as_action(payload.get("selected_action"), label="clean-zero selected_action") != list(
        direct_action
    ):
        raise R228BrokerError(
            "clean zero-backup child action differs from supplied parent direct action",
            code="clean_zero_receipt_invalid",
        )
    if payload.get("stop_reason") != "decision_deadline":
        raise R228BrokerError(
            "clean zero-backup receipt lacks a decision-deadline stop reason",
            code="clean_zero_receipt_invalid",
        )
    if payload.get("clean_deadline_cleanup_complete") is not True:
        raise R228BrokerError(
            "clean zero-backup receipt does not prove completed lane cleanup",
            code="clean_zero_receipt_invalid",
        )
    if _exact_int(payload.get("completed_backups"), field="completed_backups") != 0:
        raise R228BrokerError(
            "clean zero-backup receipt reported completed backups",
            code="clean_zero_receipt_invalid",
        )

    expected = SIMULATOR_SEARCH_LANE_COUNT
    for field in (
        "requested_simulator_lane_count",
        "active_simulator_lane_count",
        "arena_count",
        "unique_handle_count",
        "search_begin_calls",
        "search_end_calls",
    ):
        if _exact_int(payload.get(field), field=field) != expected:
            raise R228BrokerError(
                f"clean zero-backup receipt {field} is not the exact two-lane count",
                code="clean_zero_receipt_invalid",
            )
    if _exact_int(
        payload.get("search_release_calls"), field="search_release_calls"
    ) < expected:
        raise R228BrokerError(
            "clean zero-backup receipt released fewer searches than its lanes",
            code="clean_zero_receipt_invalid",
        )
    if _exact_int(
        payload.get("outstanding_virtual_loss"), field="outstanding_virtual_loss"
    ) != 0:
        raise R228BrokerError(
            "clean zero-backup receipt retained virtual loss",
            code="clean_zero_receipt_invalid",
        )

    depths = payload.get("per_lane_depth")
    if not isinstance(depths, (list, tuple)) or len(depths) != expected:
        raise R228BrokerError(
            "clean zero-backup receipt lacks both lane-depth rows",
            code="clean_zero_receipt_invalid",
        )
    if any(_exact_int(value, field="per_lane_depth") < 0 for value in depths):
        raise R228BrokerError(
            "clean zero-backup receipt has a negative lane depth",
            code="clean_zero_receipt_invalid",
        )
    handles = payload.get("per_lane_handle_identities")
    if not isinstance(handles, (list, tuple)) or len(handles) != expected:
        raise R228BrokerError(
            "clean zero-backup receipt lacks both raw handle identities",
            code="clean_zero_receipt_invalid",
        )
    normalized_handles = [
        _validate_handle_identity(value, field="per_lane_handle_identities")
        for value in handles
    ]
    if len(set(normalized_handles)) != expected:
        raise R228BrokerError(
            "clean zero-backup receipt has duplicate raw handle identities",
            code="clean_zero_receipt_invalid",
        )
    chains = payload.get("per_lane_search_id_chains")
    if not isinstance(chains, (list, tuple)) or len(chains) != expected:
        raise R228BrokerError(
            "clean zero-backup receipt lacks both SearchBegin chains",
            code="clean_zero_receipt_invalid",
        )
    first_search_ids: list[int] = []
    for chain in chains:
        if not isinstance(chain, (list, tuple)) or not chain:
            raise R228BrokerError(
                "clean zero-backup receipt lacks a lane SearchBegin id",
                code="clean_zero_receipt_invalid",
            )
        first_search_ids.append(
            _exact_int(chain[0], field="per_lane_search_id_chains")
        )
    if len(set(zip(normalized_handles, first_search_ids))) != expected:
        raise R228BrokerError(
            "clean zero-backup receipt lacks distinct handle/SearchBegin composites",
            code="clean_zero_receipt_invalid",
        )
    if _exact_int(
        payload.get("distinct_search_begin_composite_count"),
        field="distinct_search_begin_composite_count",
    ) != expected:
        raise R228BrokerError(
            "clean zero-backup receipt has an invalid composite SearchBegin count",
            code="clean_zero_receipt_invalid",
        )
    _validate_handle_scoped_first_search_id_composites(
        payload,
        handles=normalized_handles,
        first_search_ids=first_search_ids,
    )
    return payload


class _JsonSocket:
    """Small newline-JSON stream helper with no reader or feeder thread."""

    def __init__(self, sock: socket.socket, *, nonblocking: bool) -> None:
        self.sock = sock
        self.sock.setblocking(not nonblocking)
        self._buffer = bytearray()
        self._messages: deque[dict[str, Any]] = deque()
        # EOF can follow a final JSON acknowledgement in the same readiness
        # drain (notably the clean-zero ``closed`` acknowledgement).  Retain
        # the already decoded acknowledgement for its caller, then make any
        # later use of this channel fail closed.
        self._peer_closed = False
        # Native worker progress can arrive from a simulator thread while the
        # child main loop is replying to a request.  One lock preserves JSON
        # line framing without creating a feeder thread.
        self._send_lock = threading.Lock()

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass

    @property
    def peer_closed(self) -> bool:
        """Whether the other endpoint sent EOF after its queued messages."""

        return self._peer_closed

    def send(self, payload: Mapping[str, Any], *, deadline: float) -> None:
        try:
            encoded = (
                json.dumps(dict(payload), allow_nan=False, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise R228BrokerError(
                "broker message is not JSON encodable", code="non_json_message"
            ) from exc
        if len(encoded) > _MAX_MESSAGE_BYTES:
            raise R228BrokerError("broker message is too large", code="message_too_large")
        with self._send_lock:
            sent = 0
            while sent < len(encoded):
                remaining = float(deadline) - time.monotonic()
                if remaining <= 0.0:
                    raise R228BrokerError("broker send timed out", code="send_timeout")
                try:
                    _readable, writable, _errors = select.select(
                        [], [self.sock], [self.sock], remaining
                    )
                except (OSError, ValueError) as exc:
                    raise R228BrokerError(
                        f"broker send select failed: {exc}", code="send_select_failed"
                    ) from exc
                if not writable:
                    raise R228BrokerError("broker send timed out", code="send_timeout")
                try:
                    count = self.sock.send(encoded[sent:])
                except BlockingIOError:
                    continue
                except OSError as exc:
                    raise R228BrokerError(
                        f"broker send failed: {exc}", code="send_failed"
                    ) from exc
                if count <= 0:
                    raise R228BrokerError("broker socket closed while sending", code="send_closed")
                sent += count

    def _decode_complete_lines(self) -> None:
        while True:
            try:
                end = self._buffer.index(0x0A)
            except ValueError:
                break
            raw = bytes(self._buffer[:end])
            del self._buffer[: end + 1]
            if not raw:
                continue
            try:
                message = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise R228BrokerError(
                    "broker emitted malformed JSON", code="malformed_child_json"
                ) from exc
            if not isinstance(message, dict):
                raise R228BrokerError(
                    "broker emitted a non-object message", code="malformed_child_message"
                )
            self._messages.append(message)

    def recv_available(self) -> list[dict[str, Any]]:
        """Drain currently readable bytes without waiting for a future message."""

        messages: list[dict[str, Any]] = []
        if self._peer_closed:
            raise R228BrokerError("broker socket closed", code="child_socket_closed")
        while True:
            try:
                data = self.sock.recv(64 * 1024)
            except BlockingIOError:
                break
            except OSError as exc:
                raise R228BrokerError(
                    f"broker receive failed: {exc}", code="receive_failed"
                ) from exc
            if not data:
                self._peer_closed = True
                break
            self._buffer.extend(data)
            if len(self._buffer) > _MAX_MESSAGE_BYTES:
                raise R228BrokerError(
                    "broker response exceeds message limit", code="response_too_large"
                )
            self._decode_complete_lines()
        while self._messages:
            messages.append(self._messages.popleft())
        if not messages and self._peer_closed:
            raise R228BrokerError("broker socket closed", code="child_socket_closed")
        return messages

    def recv_blocking_child(self) -> dict[str, Any] | None:
        """Child-side request read.  Parent owns all hard wait limits."""

        while not self._messages:
            try:
                data = self.sock.recv(64 * 1024)
            except OSError:
                return None
            if not data:
                return None
            self._buffer.extend(data)
            if len(self._buffer) > _MAX_MESSAGE_BYTES:
                raise R228BrokerError(
                    "parent request exceeds message limit", code="request_too_large"
                )
            self._decode_complete_lines()
        return self._messages.popleft()


class IsolatedR228SearchBroker:
    """Parent-side hard-deadline controller for one r228 Kaggle game.

    ``direct_action`` is supplied by the caller and is returned unchanged on
    every broker failure.  This object never selects a random fallback and
    never performs a native Search call in the parent process.
    """

    def __init__(
        self,
        stage: Path,
        action_timeout_seconds: float | None = None,
        startup_timeout_seconds: float | None = None,
        reap_grace_seconds: float | None = None,
        search_seconds: float | None = None,
    ) -> None:
        self.stage = Path(stage).expanduser().resolve()
        if not self.stage.is_dir() or self.stage.is_symlink():
            raise ValueError("broker stage must be a physical directory")
        self.action_timeout_seconds = _positive_seconds(
            action_timeout_seconds
            if action_timeout_seconds is not None
            else os.environ.get("POKEBOT_R238_BROKER_ACTION_TIMEOUT_SECONDS"),
            # This is the parent end-to-end deadline, not the native search
            # budget.  The Phase-1 two-lane path reserves both bounded reaps
            # inside this four-second callback envelope.
            fallback=4.0,
            label="action_timeout_seconds",
        )
        self.search_seconds = _positive_seconds(
            search_seconds
            if search_seconds is not None
            else os.environ.get("POKEBOT_R238_SEARCH_SECONDS"),
            fallback=2.0,
            label="search_seconds",
        )
        self.startup_timeout_seconds = _positive_seconds(
            startup_timeout_seconds
            if startup_timeout_seconds is not None
            else os.environ.get("POKEBOT_R228_BROKER_STARTUP_TIMEOUT_SECONDS"),
            fallback=30.0,
            label="startup_timeout_seconds",
        )
        self.reap_grace_seconds = _positive_seconds(
            reap_grace_seconds
            if reap_grace_seconds is not None
            else os.environ.get("POKEBOT_R228_BROKER_REAP_GRACE_SECONDS"),
            fallback=0.25,
            label="reap_grace_seconds",
        )
        self._child: subprocess.Popen[bytes] | None = None
        self._channel: _JsonSocket | None = None
        self._child_identity: dict[str, Any] | None = None
        self._child_history_count = 0
        self._journal: list[dict[str, Any]] = []
        self._next_request_id = 1
        self._decision_count = 0
        self._degraded = False
        self._closed = False
        self._last_fault: dict[str, Any] | None = None
        self._progress_events: deque[dict[str, Any]] = deque(maxlen=_MAX_PROGRESS_EVENTS)
        self._progress_by_lane: dict[str, dict[str, Any]] = {}

    @property
    def disabled(self) -> bool:
        """True once closed or deliberately degraded for the current game."""

        return self._closed or self._degraded

    @property
    def degraded(self) -> bool:
        return self._degraded

    @property
    def decision_count(self) -> int:
        return self._decision_count

    @property
    def has_live_child(self) -> bool:
        """Whether a previously started child can receive history-only IPC."""

        return self._child_alive()

    @property
    def last_fault(self) -> dict[str, Any] | None:
        return None if self._last_fault is None else _json_copy(self._last_fault)

    def marker_payload(self) -> dict[str, Any]:
        """Return auditable state for the submission entrypoint to print itself."""

        child = self._child
        return {
            "schema": SCHEMA,
            "disabled": self.disabled,
            "degraded": self._degraded,
            "decision_count": self._decision_count,
            "child_pid": int(child.pid) if child is not None and child.poll() is None else None,
            "child_identity": (
                None
                if self._child_identity is None
                else _json_copy(self._child_identity)
            ),
            "last_fault": self.last_fault,
            "progress_by_lane": _json_copy(self._progress_by_lane),
            "progress_event_count": len(self._progress_events),
            "complete_action_cap": COMPLETE_ACTION_CAP,
            "action_timeout_seconds": self.action_timeout_seconds,
            "search_seconds": self.search_seconds,
            "startup_timeout_seconds": self.startup_timeout_seconds,
            "reap_grace_seconds": self.reap_grace_seconds,
            "configured_simulator_lane_count": SIMULATOR_SEARCH_LANE_COUNT,
            "phase1_thread_caps": dict(PHASE1_THREAD_CAPS),
        }

    def _record_progress(self, message: Mapping[str, Any]) -> None:
        payload = message.get("payload")
        normalized = dict(payload) if isinstance(payload, Mapping) else {}
        _validate_progress_lanes(normalized)
        normalized["type"] = str(message.get("type") or "progress")
        request_id = message.get("request_id")
        if isinstance(request_id, int) and not isinstance(request_id, bool):
            normalized["request_id"] = int(request_id)
        normalized["observed_monotonic"] = time.monotonic()
        self._progress_events.append(normalized)
        lane = normalized.get("lane_id", normalized.get("lane"))
        if lane is not None:
            self._progress_by_lane[str(_validate_lane_id(lane, field="lane_id"))] = dict(
                normalized
            )

    def _new_fault(
        self,
        *,
        code: str,
        message: str,
        request_id: int | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        child = self._child
        identity = (
            None if self._child_identity is None else _json_copy(self._child_identity)
        )
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "kind": "r228_broker_fault",
            "code": str(code),
            "message": str(message),
            "decision_count": self._decision_count,
            "observed_monotonic": time.monotonic(),
            "child_pid": int(child.pid) if child is not None else None,
            "child_identity": identity,
            "progress_by_lane": _json_copy(self._progress_by_lane),
            "configured_simulator_lane_count": SIMULATOR_SEARCH_LANE_COUNT,
            "phase1_thread_caps": dict(PHASE1_THREAD_CAPS),
        }
        if request_id is not None:
            payload["request_id"] = int(request_id)
        if detail:
            payload["detail"] = _json_copy(dict(detail))
        self._last_fault = payload
        return payload

    def _dispose_child(
        self, *, reason: str, deadline: float | None = None
    ) -> dict[str, Any]:
        """Boundedly reap this exact child or surface containment failure.

        A native process that survives both signals is not a safe condition in
        which to continue the Kaggle agent.  Keep its Popen identity until a
        successful ``poll``/``wait`` proves it reaped; callers then propagate a
        hard :class:`R228BrokerError` instead of pretending direct fallback
        contains an arbitrary surviving native runtime.
        """

        channel, child = self._channel, self._child
        self._channel = None
        if channel is not None:
            channel.close()
        disposed_at = time.monotonic()
        identity = (
            {} if self._child_identity is None else dict(self._child_identity)
        )
        if child is None:
            self._child_history_count = 0
            self._child_identity = None
            return {
                "reason": str(reason),
                "dispose_started_monotonic": disposed_at,
                "child_present": False,
                "reaped": True,
            }

        identity.setdefault("pid", int(child.pid))
        report: dict[str, Any] = {
            "reason": str(reason),
            "child_present": True,
            "child_identity": _json_copy(identity),
            "dispose_started_monotonic": disposed_at,
            "term_sent": False,
            "kill_sent": False,
        }

        def reap_wait_seconds() -> float:
            """Use at most one grace window and never cross a hard deadline."""

            if deadline is None:
                return self.reap_grace_seconds
            return max(0.0, min(self.reap_grace_seconds, deadline - time.monotonic()))

        returncode = child.poll()
        if returncode is None:
            # ``Popen.terminate`` and ``Popen.kill`` address the exact PID
            # created above.  We deliberately never signal a group/session.
            try:
                child.terminate()
                report["term_sent"] = True
            except ProcessLookupError:
                pass
            term_wait_started = time.monotonic()
            try:
                returncode = child.wait(timeout=reap_wait_seconds())
            except subprocess.TimeoutExpired:
                returncode = child.poll()
            report["term_wait_seconds"] = time.monotonic() - term_wait_started
        if returncode is None:
            try:
                child.kill()
                report["kill_sent"] = True
            except ProcessLookupError:
                pass
            kill_wait_started = time.monotonic()
            try:
                returncode = child.wait(timeout=reap_wait_seconds())
            except subprocess.TimeoutExpired:
                returncode = child.poll()
            report["kill_wait_seconds"] = time.monotonic() - kill_wait_started

        report["dispose_elapsed_seconds"] = time.monotonic() - disposed_at
        if returncode is not None:
            report["returncode"] = int(returncode)
            report["reaped"] = True
            self._child = None
            self._child_identity = None
            self._child_history_count = 0
            return report

        report["reaped"] = False
        # Leave ``_child`` and its identity in place.  It is deliberately
        # impossible to replace this survivor with another native child.
        self._child_history_count = 0
        self._degraded = True
        raise R228BrokerError(
            "broker child survived bounded TERM and KILL reaping",
            code="child_unreaped",
            detail={"reap": report},
        )

    def _child_alive(self) -> bool:
        return (
            self._child is not None
            and self._channel is not None
            and self._child.poll() is None
        )

    def _start_child(
        self, *, deadline: float, reap_deadline: float | None = None
    ) -> None:
        if self._closed:
            raise R228BrokerError("broker is closed", code="broker_closed")
        if self._child_alive():
            return
        self._dispose_child(reason="child_replace", deadline=reap_deadline)
        parent_sock, child_sock = socket.socketpair()
        child_fd = child_sock.detach()
        os.set_inheritable(child_fd, True)
        env = dict(os.environ)
        stage_text = str(self.stage)
        inherited_path = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            stage_text if not inherited_path else stage_text + os.pathsep + inherited_path
        )
        # Phase-1 exposes only two vCPUs.  The child has exactly two native
        # lanes, so inherited BLAS/OpenMP pools must not oversubscribe either
        # lane.  Assign rather than setdefault: a host's larger pool is not a
        # valid cap for this Kaggle execution contract.
        env.update(PHASE1_THREAD_CAPS)
        # The child runtime consumes only this r238-specific search budget.
        # Do not inherit the historical eight-second r228 environment value.
        env["POKEBOT_R238_SEARCH_SECONDS"] = f"{self.search_seconds:.6f}"
        try:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "poke_bot.r228_kaggle_broker",
                    "--child-fd",
                    str(child_fd),
                    "--stage",
                    stage_text,
                ],
                cwd=stage_text,
                env=env,
                stdin=subprocess.DEVNULL,
                # All broker diagnostics travel through the bounded JSON
                # channel.  Inheriting streams would duplicate child receipt
                # markers into the competition controller's output.
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                pass_fds=(child_fd,),
            )
        except OSError as exc:
            parent_sock.close()
            try:
                os.close(child_fd)
            except OSError:
                pass
            raise R228BrokerError(
                f"broker child could not start: {exc}", code="child_start_failed"
            ) from exc
        finally:
            try:
                os.close(child_fd)
            except OSError:
                pass
        self._child = child
        self._child_identity = {
            "pid": int(child.pid),
            "started_monotonic": time.monotonic(),
            "stage": stage_text,
            "configured_simulator_lane_count": SIMULATOR_SEARCH_LANE_COUNT,
            "phase1_thread_caps": dict(PHASE1_THREAD_CAPS),
        }
        self._channel = _JsonSocket(parent_sock, nonblocking=True)
        try:
            ready = self._wait_for(
                request_id=None,
                expected_type="ready",
                deadline=deadline,
            )
            if ready.get("schema") != SCHEMA:
                raise R228BrokerError(
                    "broker child schema changed", code="child_schema_mismatch"
                )
            preload_identity = ready.get("preload_stock_library")
            if not isinstance(preload_identity, Mapping):
                raise R228BrokerError(
                    "broker child omitted its pre-load stock-library receipt",
                    code="missing_child_preload_stock_library_receipt",
                )
            # The child obtained this receipt through the public no-CG helper
            # before importing the frozen direct entrypoint.  Retain it with
            # the exact child identity so both normal and degraded parent
            # receipts prove the native pre-load boundary.
            self._child_identity["preload_stock_library"] = _json_copy(
                dict(preload_identity)
            )
            cuda_runtime_before_search = ready.get("cuda_runtime_before_search")
            if isinstance(cuda_runtime_before_search, Mapping):
                self._child_identity["cuda_runtime_before_search"] = _json_copy(
                    dict(cuda_runtime_before_search)
                )
            else:
                # This intentionally remains a diagnostic observation rather
                # than a new broker-action condition.  A later immutable
                # preflight can reject incomplete telemetry without making a
                # live legal direct fallback unavailable.
                self._child_identity["cuda_runtime_before_search"] = {
                    "schema": CUDA_RUNTIME_OBSERVATION_SCHEMA,
                    "phase": CUDA_RUNTIME_OBSERVATION_PHASE,
                    "torch_imported": False,
                    "cuda_available": False,
                    "cuda_initialized": False,
                    "device_count": 0,
                    "devices": [],
                    "model_device": "unavailable",
                    "telemetry_complete": False,
                    "error_types": ["child_ready:MissingCudaObservation"],
                }
        except Exception:
            # A survivor is more serious than the original startup failure;
            # _dispose_child raises a hard containment error in that case.
            self._dispose_child(
                reason="startup_failure", deadline=reap_deadline
            )
            raise

    def _send(self, payload: Mapping[str, Any], *, deadline: float) -> None:
        if not self._child_alive() or self._channel is None:
            raise R228BrokerError("broker child is not alive", code="child_not_alive")
        self._channel.send(payload, deadline=deadline)

    def _wait_for(
        self,
        *,
        request_id: int | None,
        expected_type: str,
        deadline: float,
    ) -> dict[str, Any]:
        """Wait for one exact reply while continuously retaining progress rows."""

        if self._channel is None or self._child is None:
            raise R228BrokerError("broker child is absent", code="child_absent")
        while True:
            remaining = float(deadline) - time.monotonic()
            if remaining <= 0.0:
                raise R228BrokerError("broker response timed out", code="response_timeout")
            try:
                readable, _writable, errors = select.select(
                    [self._channel.sock], [], [self._channel.sock], remaining
                )
            except (OSError, ValueError) as exc:
                raise R228BrokerError(
                    f"broker receive select failed: {exc}", code="receive_select_failed"
                ) from exc
            if errors:
                raise R228BrokerError("broker socket reported an error", code="socket_error")
            if not readable:
                raise R228BrokerError("broker response timed out", code="response_timeout")
            for message in self._channel.recv_available():
                if message.get("schema") != SCHEMA:
                    raise R228BrokerError(
                        "broker child schema changed", code="child_schema_mismatch"
                    )
                kind = str(message.get("type") or "")
                if kind == "progress":
                    self._record_progress(message)
                    continue
                if kind == "error":
                    message_id = message.get("request_id")
                    if request_id is None or message_id == request_id:
                        raise R228BrokerError(
                            str(message.get("message") or "broker child reported an error"),
                            code=str(message.get("code") or "child_error"),
                            detail=(
                                dict(message.get("detail"))
                                if isinstance(message.get("detail"), Mapping)
                                else {}
                            ),
                        )
                    raise R228BrokerError(
                        "broker child error had the wrong request id",
                        code="wrong_request_id",
                    )
                if kind != expected_type:
                    raise R228BrokerError(
                        f"broker child returned unexpected message type {kind!r}",
                        code="unexpected_message_type",
                    )
                if request_id is not None and message.get("request_id") != request_id:
                    raise R228BrokerError(
                        "broker child reply request id did not match", code="wrong_request_id"
                    )
                return message
            # A child may exit immediately after writing its final ``closed``
            # acknowledgement.  Drain the socket first so that clean exact
            # reaping can observe that acknowledgement rather than turning it
            # into a false child-exited protocol fault.
            if self._child.poll() is not None:
                raise R228BrokerError(
                    f"broker child exited with code {self._child.returncode}",
                    code="child_exited",
                    detail={"returncode": self._child.returncode},
                )

    def _sync_child(self, *, deadline: float) -> None:
        if not self._child_alive():
            raise R228BrokerError("broker child is absent", code="child_absent")
        if self._child_history_count == len(self._journal):
            return
        # ``sync`` resets the child policy before replaying.  Always provide
        # the complete committed history, rather than only the delta, so a
        # replacement child cannot accidentally lose an earlier context row.
        events = self._journal
        request_id = self._next_request_id
        self._next_request_id += 1
        self._send(
            {
                "schema": SCHEMA,
                "type": "sync",
                "request_id": request_id,
                "events": _json_copy(events),
            },
            deadline=deadline,
        )
        reply = self._wait_for(
            request_id=request_id, expected_type="synced", deadline=deadline
        )
        applied = reply.get("applied")
        if isinstance(applied, bool) or not isinstance(applied, int) or applied != len(events):
            raise R228BrokerError(
                "broker child replayed an incomplete journal", code="incomplete_journal_replay"
            )
        self._child_history_count = len(self._journal)

    def _append_journal(self, obs: Mapping[str, Any], action: Sequence[int]) -> None:
        self._journal.append(
            {"observation": _json_copy(dict(obs)), "action": [int(item) for item in action]}
        )

    def _degrade(
        self,
        *,
        exc: Exception | None,
        code: str | None = None,
        request_id: int | None = None,
        direct_action: Sequence[int] | None = None,
        hard_deadline: float | None = None,
    ) -> dict[str, Any]:
        if isinstance(exc, R228BrokerError):
            fault_code = code or exc.code
            detail = exc.detail
            message = str(exc)
        else:
            fault_code = code or "broker_failure"
            detail = {}
            message = str(exc) if exc is not None else "broker is unavailable"
        fault = self._new_fault(
            code=fault_code,
            message=message,
            request_id=request_id,
            detail=detail,
        )
        if direct_action is not None:
            fault["direct_fallback_action"] = [int(item) for item in direct_action]
        self._degraded = True
        try:
            reap = self._dispose_child(
                reason="broker_fault", deadline=hard_deadline
            )
        except R228BrokerError as reap_exc:
            fault["containment_failure"] = True
            fault["child_reap"] = _json_copy(reap_exc.detail.get("reap", {}))
            self._last_fault = fault
            raise R228BrokerError(
                "broker containment failed after a causal fault",
                code=reap_exc.code,
                detail={
                    "causal_fault": _json_copy(fault),
                    "reap": _json_copy(reap_exc.detail.get("reap", {})),
                },
            ) from reap_exc
        fault["child_reap"] = _json_copy(reap)
        self._last_fault = fault
        return fault

    def _close_clean_zero_child(self, *, hard_deadline: float) -> dict[str, Any]:
        """Close then reap the exact child after a proven clean zero result."""

        child_identity = (
            None if self._child_identity is None else _json_copy(self._child_identity)
        )
        if not self._child_alive():
            raise R228BrokerError(
                "clean zero-backup child was not alive for exact cleanup",
                code="clean_zero_child_not_alive",
            )
        close_deadline = hard_deadline - (2.0 * self.reap_grace_seconds)
        if close_deadline <= time.monotonic():
            raise R228BrokerError(
                "clean zero-backup path has no bounded close window",
                code="clean_zero_close_window_exhausted",
            )
        request_id = self._next_request_id
        self._next_request_id += 1
        self._send(
            {"schema": SCHEMA, "type": "close", "request_id": request_id},
            deadline=close_deadline,
        )
        self._wait_for(
            request_id=request_id, expected_type="closed", deadline=close_deadline
        )
        reap = self._dispose_child(
            reason="clean_zero_backup_deadline", deadline=hard_deadline
        )
        if reap.get("reaped") is not True:
            raise R228BrokerError(
                "clean zero-backup child did not reap", code="child_unreaped"
            )
        return {
            "child_identity": child_identity,
            "close_request_id": request_id,
            "reap": _json_copy(reap),
        }

    def begin_game(self, *, start_child: bool = True) -> None:
        """Reset one physical game and optionally defer native child startup.

        The parent can journal trusted direct turns before it needs MCTS.  A
        deferred start keeps r240 high-confidence direct decisions entirely
        out of the native subprocess while retaining an exact journal for the
        first later low-confidence MCTS prompt.
        """

        if self._closed:
            raise R228BrokerError("broker is closed", code="broker_closed")
        hard_deadline = time.monotonic() + self.startup_timeout_seconds
        self._dispose_child(
            reason="begin_game_replace", deadline=hard_deadline
        )
        self._journal.clear()
        self._child_history_count = 0
        self._decision_count = 0
        self._degraded = False
        self._last_fault = None
        self._progress_events.clear()
        self._progress_by_lane.clear()
        if not isinstance(start_child, bool):
            raise ValueError("start_child must be bool")
        if not start_child:
            return
        deadline = hard_deadline - (2.0 * self.reap_grace_seconds)
        try:
            if deadline <= time.monotonic():
                raise R228BrokerError(
                    "startup deadline is too short for bounded containment",
                    code="insufficient_containment_window",
                )
            self._start_child(
                deadline=deadline, reap_deadline=hard_deadline
            )
        except Exception as exc:  # Caller can still play direct policy.
            self._degrade(
                exc=exc, code="startup_failure", hard_deadline=hard_deadline
            )

    def note_direct_action(self, obs: Mapping[str, Any], actual_action: Sequence[int]) -> None:
        """Commit a non-MCTS real action so a live child stays synchronized.

        When the child is unavailable the event remains in the replay journal.
        A future replacement child receives it before attempting MCTS.
        """

        if not isinstance(obs, Mapping):
            raise ValueError("direct-action observation must be a mapping")
        action = _as_action(list(actual_action), label="actual_action")
        if not _syntactically_legal(obs, action):
            raise R228BrokerError("direct action is not legal", code="illegal_direct_action")
        self._append_journal(obs, action)
        if self._closed or self._degraded or not self._child_alive():
            return
        hard_deadline = time.monotonic() + min(
            self.action_timeout_seconds, self.startup_timeout_seconds
        )
        deadline = hard_deadline - (2.0 * self.reap_grace_seconds)
        request_id = self._next_request_id
        self._next_request_id += 1
        try:
            if deadline <= time.monotonic():
                raise R228BrokerError(
                    "direct-note deadline is too short for bounded containment",
                    code="insufficient_containment_window",
                )
            self._send(
                {
                    "schema": SCHEMA,
                    "type": "note",
                    "request_id": request_id,
                    "event": _json_copy(self._journal[-1]),
                },
                deadline=deadline,
            )
            reply = self._wait_for(
                request_id=request_id, expected_type="noted", deadline=deadline
            )
            if reply.get("applied") is not True:
                raise R228BrokerError(
                    "broker child did not commit direct action", code="direct_note_rejected"
                )
            self._child_history_count = len(self._journal)
        except Exception as exc:
            self._degrade(
                exc=exc,
                code="direct_note_failure",
                hard_deadline=hard_deadline,
            )

    def select(
        self, obs: Mapping[str, Any], direct_action: Sequence[int]
    ) -> tuple[list[int], dict[str, Any] | None, dict[str, Any] | None]:
        """Attempt one MCTS decision or return the already supplied direct action.

        The method's deadline includes IPC, child synchronization, search, and
        result validation.  A contained child fault returns ``direct_action``
        and disables MCTS for the remainder of the game.  An invalid caller
        action, legal-space failure, or unreapable native child is deliberately
        a hard error rather than an unsafe fallback.
        """

        if not isinstance(obs, Mapping):
            raise ValueError("selection observation must be a mapping")
        direct = _as_action(list(direct_action), label="direct_action")
        # This validation is outside the contained-child path.  Returning a
        # caller-supplied illegal action would be worse than failing closed.
        if not _syntactically_legal(obs, direct):
            raise R228BrokerError(
                "supplied direct action is outside the complete legal order",
                code="illegal_direct_action",
            )
        if self._closed:
            raise R228BrokerError("broker is closed", code="broker_closed")
        if self._degraded:
            self._append_journal(obs, direct)
            causal = self.last_fault
            if causal is None:
                causal = self._new_fault(
                    code="degraded_without_fault",
                    message="broker is degraded without a causal receipt",
                )
            return direct, None, causal
        legal_order = _complete_legal_order(obs)
        legal = set(legal_order)
        if tuple(direct) not in legal:
            raise R228BrokerError(
                "supplied direct action is outside the complete legal order",
                code="illegal_direct_action",
            )
        # This is observed before a possible lazy start.  A persistent child
        # may correctly serve later decisions, so the receipt distinguishes a
        # live existing child from one started for this exact request.
        child_was_live_before_select = self._child_alive()
        hard_deadline = time.monotonic() + self.action_timeout_seconds
        # Reserve both TERM and KILL grace windows before waiting for a child
        # response.  Thus a hung native call cannot push the parent past its
        # advertised end-to-end action budget.
        deadline = hard_deadline - (2.0 * self.reap_grace_seconds)
        request_id: int | None = None
        try:
            if deadline <= time.monotonic():
                raise R228BrokerError(
                    "hard action deadline is too short for bounded containment",
                    code="insufficient_containment_window",
                )
            if not self._child_alive():
                self._start_child(
                    deadline=deadline, reap_deadline=hard_deadline
                )
            self._sync_child(deadline=deadline)
            request_id = self._next_request_id
            self._next_request_id += 1
            # The child receives its independent native-search budget, while
            # ``deadline`` already leaves both outer reaping windows intact.
            remaining_for_search = deadline - time.monotonic()
            if remaining_for_search < 0.25:
                raise R228BrokerError(
                    "hard deadline left no bounded native-search window",
                    code="insufficient_search_window",
                )
            child_seconds = min(self.search_seconds, remaining_for_search)
            self._send(
                {
                    "schema": SCHEMA,
                    "type": "select",
                    "request_id": request_id,
                    "observation": _json_copy(dict(obs)),
                    "direct_action": list(direct),
                    "timeout_seconds": child_seconds,
                },
                deadline=deadline,
            )
            reply = self._wait_for(
                request_id=request_id, expected_type="result", deadline=deadline
            )
            selected = _as_action(reply.get("action"), label="child action")
            if tuple(selected) not in legal:
                raise R228BrokerError(
                    "broker child returned an action outside complete legal order",
                    code="illegal_child_action",
                )
            receipt = reply.get("receipt")
            if not isinstance(receipt, Mapping):
                raise R228BrokerError(
                    "broker child omitted its decision receipt", code="missing_receipt"
                )
            receipt_payload = _json_copy(dict(receipt))
            receipt_action = _as_action(
                receipt_payload.get("selected_action"), label="receipt selected_action"
            )
            if receipt_action != selected:
                raise R228BrokerError(
                    "broker receipt action differs from returned action",
                    code="receipt_action_mismatch",
                )
            child_identity = self._child_identity
            preload_identity = (
                None
                if child_identity is None
                else child_identity.get("preload_stock_library")
            )
            if not isinstance(preload_identity, Mapping):
                raise R228BrokerError(
                    "broker child lacks a pre-load stock-library receipt",
                    code="missing_child_preload_stock_library_receipt",
                )
            receipt_payload["child_preload_stock_library"] = _json_copy(
                dict(preload_identity)
            )
            if receipt_payload.get("mode") == _CHILD_CLEAN_ZERO_MODE:
                receipt_payload = _validate_clean_zero_backup_receipt(
                    receipt_payload, direct_action=direct
                )
                if selected != direct:
                    raise R228BrokerError(
                        "clean zero-backup child returned a non-parent action",
                        code="clean_zero_receipt_invalid",
                    )
                cleanup = self._close_clean_zero_child(hard_deadline=hard_deadline)
                receipt_payload.update(
                    {
                        "mode": _PARENT_CLEAN_ZERO_MODE,
                        "child_mode": _CHILD_CLEAN_ZERO_MODE,
                        "selected_action": list(direct),
                        "direct_action": list(direct),
                        "mcts_action_authority": False,
                        "zero_backup_precomputed_direct_fallback": True,
                        "exact_child_cleanup_and_reap": cleanup,
                        "configured_simulator_lane_count": (
                            SIMULATOR_SEARCH_LANE_COUNT
                        ),
                    }
                )
                self._append_journal(obs, direct)
                self._child_history_count = 0
                self._decision_count += 1
                return list(direct), receipt_payload, None
            if receipt_payload.get("mode") != "shared_tree_mcts":
                raise R228BrokerError(
                    "broker child returned a non-authoritative search receipt",
                    code="non_authoritative_receipt_mode",
                    detail={"mode": receipt_payload.get("mode")},
                )
            receipt_payload = _validate_two_lane_receipt(
                receipt_payload,
                observation=obs,
                legal_order=legal_order,
                selected_action=selected,
            )
            if not self._child_alive() or not isinstance(self._child_identity, Mapping):
                raise R228BrokerError(
                    "broker child disappeared before authoritative receipt return",
                    code="child_not_alive",
                )
            # These are literal broker/IPC observations, not marker-derived
            # inference.  They are installed only after exactly one select
            # request received and passed full receipt validation.
            receipt_payload.update(
                {
                    "broker_started": True,
                    "mcts_child_started": True,
                    "mcts_child_started_for_this_decision": (
                        not child_was_live_before_select
                    ),
                    "mcts_child_called": True,
                    "mcts_child_call_count": 1,
                    "mcts_select_call_count": 1,
                    "child_search_budget_seconds": child_seconds,
                }
            )
            self._append_journal(obs, selected)
            self._child_history_count = len(self._journal)
            self._decision_count += 1
            return selected, receipt_payload, None
        except Exception as exc:
            fault = self._degrade(
                exc=exc,
                request_id=request_id,
                direct_action=direct,
                hard_deadline=hard_deadline,
            )
            self._append_journal(obs, direct)
            return direct, None, fault

    def close(self) -> None:
        """Dispose only the owned broker child and permanently disable this object."""

        if self._closed:
            return
        self._closed = True
        self._dispose_child(reason="broker_close")


def _child_load_direct(stage: Path) -> Any:
    stage = stage.resolve()
    source = stage / "r195_direct_main.py"
    if not source.is_file() or source.is_symlink():
        raise R228BrokerError(
            "staged frozen direct entrypoint is missing", code="missing_direct_entrypoint"
        )
    stage_text = str(stage)
    os.chdir(stage_text)
    if stage_text not in sys.path:
        sys.path.insert(0, stage_text)
    spec = importlib.util.spec_from_file_location("r228_broker_r195_direct", source)
    if spec is None or spec.loader is None:
        raise R228BrokerError(
            "cannot import staged frozen direct entrypoint", code="direct_import_failed"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _child_progress_sender(channel: _JsonSocket):
    def callback(*args: Any, **kwargs: Any) -> None:
        payload: dict[str, Any] = {}
        if len(args) == 1 and isinstance(args[0], Mapping):
            payload.update(dict(args[0]))
        elif args:
            payload["event"] = str(args[0])
            if len(args) > 1:
                payload["args"] = [str(value) for value in args[1:]]
        payload.update({str(key): value for key, value in kwargs.items()})
        try:
            channel.send(
                {"schema": SCHEMA, "type": "progress", "payload": _json_copy(payload)},
                deadline=time.monotonic() + 0.25,
            )
        except Exception:
            # Diagnostic progress must not deadlock or alter native action work.
            pass

    return callback


def _child_new_runtime(stage: Path, channel: _JsonSocket) -> Any:
    # This public helper has no cg/Torch/model import path.  It must run
    # before ``r195_direct_main`` or ``direct._ensure_runtime`` can map the
    # staged native library in the child process.
    from poke_bot.r228_kaggle_async_runtime import (
        R228AsyncGameplay,
        validate_staged_stock_library_identity,
    )

    preload_stock_library = validate_staged_stock_library_identity(stage)
    direct = _child_load_direct(stage)
    frozen_turn_order_choice = getattr(direct, "_turn_order_choice", None)
    if not callable(frozen_turn_order_choice):
        raise R228BrokerError(
            "staged frozen direct entrypoint lacks its turn-order resolver",
            code="missing_frozen_turn_order_resolver",
        )
    deck, model, policy = direct._ensure_runtime()
    # The model is fully loaded before this point.  Record the actual CUDA
    # visibility and selected model device now, before any native arena or
    # search work can begin.  This is diagnostic only; it does not select a
    # device or change the frozen r195 runtime's existing choice.
    cuda_runtime_before_search = capture_cuda_runtime_before_search(model)

    callback = _child_progress_sender(channel)
    kwargs = {"stage": stage, "model": model, "policy": policy, "deck": deck}
    try:
        signature = inspect.signature(R228AsyncGameplay)
        accepts_progress = "progress_callback" in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
    except (TypeError, ValueError):
        accepts_progress = True
    if accepts_progress:
        try:
            runtime = R228AsyncGameplay(**kwargs, progress_callback=callback)
            setattr(
                runtime,
                "child_preload_stock_library_receipt",
                dict(preload_stock_library),
            )
            setattr(
                runtime,
                "child_cuda_runtime_before_search",
                dict(cuda_runtime_before_search),
            )
            # Journal replay must use the exact frozen entrypoint's setup
            # resolver.  The sealed r195 feature module predates the newer
            # workspace-only ``features.forced_go_first_action`` helper.
            setattr(runtime, "child_frozen_turn_order_choice", frozen_turn_order_choice)
            return runtime
        except TypeError as exc:
            if "progress_callback" not in str(exc):
                raise
    # Compatibility with the current rough runtime.  The follow-up runtime
    # implementation consumes this attribute and emits per-lane progress.
    runtime = R228AsyncGameplay(**kwargs)
    setattr(runtime, "progress_callback", callback)
    setattr(
        runtime,
        "child_preload_stock_library_receipt",
        dict(preload_stock_library),
    )
    setattr(
        runtime,
        "child_cuda_runtime_before_search",
        dict(cuda_runtime_before_search),
    )
    setattr(runtime, "child_frozen_turn_order_choice", frozen_turn_order_choice)
    return runtime


def _child_frozen_turn_order_choice(
    runtime: Any, observation: Mapping[str, Any]
) -> list[int] | None:
    """Resolve setup prompts through the staged frozen r195 entrypoint.

    The MCTS child must replay the parent journal without importing a helper
    that only exists in the workspace checkout.  ``_turn_order_choice`` is
    deliberately part of the frozen r195 submission boundary and returns
    ``[]`` for a recognized prompt that must use its own fail-closed action.
    In that latter case the exact chosen action has already been validated by
    the parent; this helper reports the setup prompt without recomputing it.
    """

    resolver = getattr(runtime, "child_frozen_turn_order_choice", None)
    if not callable(resolver):
        raise R228BrokerError(
            "child runtime lacks the staged frozen turn-order resolver",
            code="missing_frozen_turn_order_resolver",
        )
    try:
        choice = resolver(dict(observation))
    except Exception as exc:
        raise R228BrokerError(
            f"staged frozen turn-order resolver failed: {type(exc).__name__}: {exc}",
            code="frozen_turn_order_resolver_failed",
        ) from exc
    if choice is None:
        return None
    return _as_action(choice, label="staged frozen turn-order action")


def _child_commit_action(runtime: Any, event: Mapping[str, Any]) -> None:
    observation = event.get("observation")
    action = event.get("action")
    if not isinstance(observation, Mapping):
        raise R228BrokerError("journal observation is malformed", code="malformed_journal")
    actual = _as_action(action, label="journal action")
    if not _syntactically_legal(observation, actual):
        raise R228BrokerError("journal action is illegal", code="illegal_journal_action")

    policy = runtime.policy
    frozen_turn_order = _child_frozen_turn_order_choice(runtime, observation)
    if frozen_turn_order is not None:
        # The frozen resolver returns an empty list for a recognized IsFirst
        # prompt whose desired row is unavailable.  Its own direct entrypoint
        # then makes the legal fail-closed response, so replay must preserve
        # that setup boundary without trying to draw a second random action.
        if frozen_turn_order and frozen_turn_order != actual:
            raise R228BrokerError(
                "journal turn-order action differs from forced choice",
                code="turn_order_journal_mismatch",
            )
        # The frozen r195 wrapper resolves IsFirst with `_turn_order_choice`
        # before its PolicyAgent path, so it does not add this prompt to
        # temporal board/action history.  Replay must mirror that baseline.
        return
    # This is intentionally after the frozen setup resolver.  The sealed r195
    # feature module still owns ordinary action-token construction, but it
    # does not expose the newer workspace-only setup helper.
    from poke_bot import features

    router = getattr(policy, "_matchup_adapter_shadow_router", None)
    if router is not None and hasattr(router, "observe"):
        router.observe(
            dict(observation), scope="game_root", depth=len(policy.board_history)
        )
    policy._append_decision_history(dict(observation))
    policy._previous_action_token = features.build_option_tokens(
        dict(observation), [actual]
    )


def _child_send_error(
    channel: _JsonSocket,
    *,
    request_id: object,
    exc: Exception,
) -> None:
    code = exc.code if isinstance(exc, R228BrokerError) else "child_exception"
    detail = exc.detail if isinstance(exc, R228BrokerError) else {}
    try:
        channel.send(
            {
                "schema": SCHEMA,
                "type": "error",
                "request_id": request_id,
                "code": code,
                "message": f"{type(exc).__name__}: {exc}",
                "detail": _json_copy(detail),
            },
            deadline=time.monotonic() + 0.5,
        )
    except Exception:
        pass


def _child_main(*, child_fd: int, stage: Path) -> int:
    """Fresh-interpreter child loop.  It may block; parent owns containment."""

    sock = socket.socket(fileno=int(child_fd))
    channel = _JsonSocket(sock, nonblocking=False)
    runtime: Any | None = None
    try:
        runtime = _child_new_runtime(stage, channel)
        preload_stock_library = getattr(
            runtime, "child_preload_stock_library_receipt", None
        )
        if not isinstance(preload_stock_library, Mapping):
            raise R228BrokerError(
                "child runtime omitted its pre-load stock-library receipt",
                code="missing_child_preload_stock_library_receipt",
            )
        cuda_runtime_before_search = getattr(
            runtime, "child_cuda_runtime_before_search", None
        )
        if not isinstance(cuda_runtime_before_search, Mapping):
            # Do not turn a diagnostic omission into a new action-authority
            # branch.  The parent preserves this explicitly incomplete record
            # for the immutable preflight/audit receipt instead.
            cuda_runtime_before_search = {
                "schema": CUDA_RUNTIME_OBSERVATION_SCHEMA,
                "phase": CUDA_RUNTIME_OBSERVATION_PHASE,
                "torch_imported": False,
                "cuda_available": False,
                "cuda_initialized": False,
                "device_count": 0,
                "devices": [],
                "model_device": "unavailable",
                "telemetry_complete": False,
                "error_types": ["child_runtime:MissingCudaObservation"],
            }
        channel.send(
            {
                "schema": SCHEMA,
                "type": "ready",
                "payload": {"pid": os.getpid(), "stage": str(stage.resolve())},
                "preload_stock_library": _json_copy(
                    dict(preload_stock_library)
                ),
                "cuda_runtime_before_search": _json_copy(
                    dict(cuda_runtime_before_search)
                ),
            },
            deadline=time.monotonic() + 5.0,
        )
        while True:
            request = channel.recv_blocking_child()
            if request is None:
                return 0
            request_id = request.get("request_id")
            try:
                if request.get("schema") != SCHEMA:
                    raise R228BrokerError("parent schema changed", code="parent_schema_mismatch")
                kind = str(request.get("type") or "")
                if kind == "sync":
                    events = request.get("events")
                    if not isinstance(events, list):
                        raise R228BrokerError("sync events must be a list", code="malformed_sync")
                    runtime.reset_game()
                    for event in events:
                        if not isinstance(event, Mapping):
                            raise R228BrokerError(
                                "sync contains a malformed event", code="malformed_sync"
                            )
                        _child_commit_action(runtime, event)
                    channel.send(
                        {
                            "schema": SCHEMA,
                            "type": "synced",
                            "request_id": request_id,
                            "applied": len(events),
                        },
                        deadline=time.monotonic() + 1.0,
                    )
                elif kind == "note":
                    event = request.get("event")
                    if not isinstance(event, Mapping):
                        raise R228BrokerError("note event is malformed", code="malformed_note")
                    _child_commit_action(runtime, event)
                    channel.send(
                        {
                            "schema": SCHEMA,
                            "type": "noted",
                            "request_id": request_id,
                            "applied": True,
                        },
                        deadline=time.monotonic() + 1.0,
                    )
                elif kind == "select":
                    observation = request.get("observation")
                    if not isinstance(observation, Mapping):
                        raise R228BrokerError(
                            "selection observation is malformed", code="malformed_select"
                        )
                    timeout = _positive_seconds(
                        request.get("timeout_seconds"),
                        fallback=0.05,
                        label="child timeout_seconds",
                    )
                    channel.send(
                        {
                            "schema": SCHEMA,
                            "type": "progress",
                            "request_id": request_id,
                            "payload": {"phase": "select_started"},
                        },
                        deadline=time.monotonic() + 0.5,
                    )
                    # Runtime search duration is bound once at child startup
                    # through ``POKEBOT_R238_SEARCH_SECONDS``.  The request
                    # value exists solely as a parent transport deadline and
                    # must not revive the legacy r228 eight-second setting.
                    if timeout <= 0.0:  # defensive, _positive_seconds above
                        raise R228BrokerError(
                            "child timeout must be positive", code="invalid_child_timeout"
                        )
                    # The parent already computed and validated this exact
                    # frozen-r195 fallback against its complete legal order.
                    # Carry it over the bounded IPC explicitly: clean
                    # zero-backup deadline authority must return that parent
                    # action, never a child-side re-computation that might
                    # drift because of temporal-policy state.
                    precomputed_direct = _as_action(
                        request.get("direct_action"),
                        label="precomputed parent direct_action",
                    )
                    if not _syntactically_legal(observation, precomputed_direct):
                        raise R228BrokerError(
                            "parent direct action is not legal in child selection",
                            code="illegal_parent_direct_action",
                        )
                    action = list(
                        runtime.select(
                            dict(observation),
                            precomputed_direct_action=precomputed_direct,
                        )
                    )
                    if not runtime.decision_receipts:
                        raise R228BrokerError(
                            "r228 runtime returned without a receipt", code="missing_runtime_receipt"
                        )
                    receipt = dict(runtime.decision_receipts[-1])
                    channel.send(
                        {
                            "schema": SCHEMA,
                            "type": "result",
                            "request_id": request_id,
                            "action": [int(item) for item in action],
                            "receipt": _json_copy(receipt),
                        },
                        deadline=time.monotonic() + 1.0,
                    )
                elif kind == "close":
                    # This can block in native cleanup; parent will terminate
                    # this exact child after its bounded grace period.
                    runtime.close()
                    channel.send(
                        {
                            "schema": SCHEMA,
                            "type": "closed",
                            "request_id": request_id,
                        },
                        deadline=time.monotonic() + 1.0,
                    )
                    return 0
                else:
                    raise R228BrokerError(
                        f"unsupported broker request {kind!r}", code="unknown_request"
                    )
            except Exception as exc:
                _child_send_error(channel, request_id=request_id, exc=exc)
    except Exception as exc:
        _child_send_error(channel, request_id=None, exc=exc)
        return 2
    finally:
        channel.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child-fd", type=int, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args(argv)
    return _child_main(child_fd=int(args.child_fd), stage=args.stage)


if __name__ == "__main__":  # pragma: no cover - exercised through Popen.
    raise SystemExit(main())


__all__ = [
    "COMPLETE_ACTION_CAP",
    "IsolatedR228SearchBroker",
    "PHASE1_THREAD_CAPS",
    "R238_MAXIMUM_BACKUPS_PER_DECISION",
    "R238_MINIMUM_BACKUPS_BEFORE_STABILITY",
    "R238_STABLE_ROOT_LEADER_OBSERVATIONS_REQUIRED",
    "R228BrokerError",
    "SCHEMA",
    "SIMULATOR_SEARCH_LANE_COUNT",
]
