"""Kaggle parent entrypoint for the bounded r228 native-search diagnostic.

The frozen r195 policy always chooses a legal action in this parent process
first.  The raw ``libcg`` search runtime runs only in an owned subprocess via
``IsolatedR228SearchBroker``.  A contained child fault therefore cannot leave
the Kaggle controller blocked: the already-computed frozen action is played
for the rest of that game.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from poke_bot.r228_kaggle_broker import (
    CUDA_RUNTIME_OBSERVATION_PHASE,
    CUDA_RUNTIME_OBSERVATION_SCHEMA,
    COMPLETE_ACTION_CAP,
    IsolatedR228SearchBroker,
    PHASE1_THREAD_CAPS,
    R228BrokerError,
    SIMULATOR_SEARCH_LANE_COUNT as BROKER_SIMULATOR_SEARCH_LANE_COUNT,
    capture_cuda_runtime_before_search,
)

# Successful searched actions have this authority.  The containment marker
# explicitly identifies the separately precomputed direct fallback authority.
R228_ASYNC_SELECTED_ACTION_AUTHORITY = "receipt.selected_action"
SCHEMA = "poke_bot.r238_two_lane_kaggle_viability/v1"
CONTAINMENT_SCHEMA = "poke_bot.r234_kaggle_native_containment/v1"
DECISION_PREFIX = "R238_TWO_LANE_BOUNDED_MCTS_DECISION"
FULL_GAMEPLAY_SUCCESS_PREFIX = "R238_TWO_LANE_BOUNDED_MCTS_FULL_GAMEPLAY_SUCCESS"
DEGRADED_PREFIX = "R234_KAGGLE_NATIVE_CONTAINMENT_DEGRADED"
HARD_FAILURE_PREFIX = "R238_TWO_LANE_BOUNDED_MCTS_HARD_FAILURE"
SIMULATOR_SEARCH_LANE_COUNT = 2

# Keep the end-to-end parent deadline distinct from the child native-search
# window: Phase-1 reserves both bounded reaps inside one four-second callback.
R234_BROKER_ACTION_TIMEOUT_SECONDS = 4.0
R234_BROKER_SEARCH_SECONDS = 2.0
R234_BROKER_STARTUP_TIMEOUT_SECONDS = 30.0
R234_BROKER_REAP_GRACE_SECONDS = 0.25
R238_MINIMUM_BACKUPS_BEFORE_STABILITY = 8
R238_STABLE_ROOT_LEADER_OBSERVATIONS_REQUIRED = 3
R238_MAXIMUM_BACKUPS_PER_DECISION = 32
R238_HIGH_CONFIDENCE_DIRECT_THRESHOLD = 0.80
R240_MAX_PRINCIPAL_VARIATION_DEPTH = 8
PROVEN_TERMINAL_WIN_REVISION = 246
PROVEN_TERMINAL_WIN_STOP_REASON = (
    "proven_deterministic_terminal_win_this_turn"
)
PROVEN_TERMINAL_WIN_PROOF_KIND = (
    "exact_deterministic_simulator_terminal_win_this_turn"
)
R236_LINUX_X86_64_STOCK_LIBRARY_MEMBER = "cg/libcg.so"
R236_LINUX_X86_64_STOCK_LIBRARY_SHA256 = (
    "sha256:d16244a3157fc55c3314f08dcc7c5179168697d78c105b95c7debd556b764bb7"
)

_DIRECT: Any | None = None
_BROKER: IsolatedR228SearchBroker | None = None
_AGENT_DIRS = (Path.cwd(), Path("/kaggle_simulations/agent"))
_GAME_SUCCESSFUL_MCTS_DECISIONS = 0
_GAME_DEGRADED_FAULTS = 0
_GAME_FAULT_KEYS: set[str] = set()
_EMITTED_FAULT_KEYS: set[str] = set()
# The broker normally latches ``disabled`` itself.  This extra parent latch
# covers a malformed public broker reply: after a protocol fault the parent
# must not ask that child (or a test double representing it) for another MCTS
# action during the same game.
_GAME_MCTS_DISABLED = False
_GAME_PENDING_FAULT: dict[str, Any] | None = None
_GAME_DIRECT_JOURNAL: list[dict[str, Any]] = []
_GAME_PRINCIPAL_VARIATION: list[dict[str, Any]] = []
_GAME_PRINCIPAL_VARIATION_ROOT_SEAT: int | None = None
_PARENT_THREAD_CAPS_APPLIED = False
_PARENT_TORCH_THREAD_CAPS_APPLIED = False
_PARENT_STOCK_LIBRARY_RECEIPT: dict[str, Any] | None = None
_PARENT_CUDA_RUNTIME_BEFORE_SEARCH: dict[str, Any] | None = None


def _agent_dir() -> Path:
    for candidate in _AGENT_DIRS:
        if (candidate / "r195_direct_main.py").is_file():
            return candidate
    return Path.cwd()


def _apply_phase1_parent_thread_caps() -> None:
    """Enforce one-thread pools before the frozen direct runtime can load torch."""

    global _PARENT_THREAD_CAPS_APPLIED, _PARENT_TORCH_THREAD_CAPS_APPLIED
    for name, value in PHASE1_THREAD_CAPS.items():
        # This is intentionally an assignment, never ``setdefault``: a host
        # inherited from a larger machine cannot exceed the Phase-1 envelope.
        os.environ[name] = value
    _PARENT_THREAD_CAPS_APPLIED = True

    # Do not import torch merely for this cap.  When an earlier import exists,
    # apply the runtime setters once but tolerate its documented late-call
    # RuntimeError so a legal direct fallback is never made unavailable.
    torch_module = sys.modules.get("torch")
    if torch_module is None or _PARENT_TORCH_THREAD_CAPS_APPLIED:
        return
    for setter_name in ("set_num_threads", "set_num_interop_threads"):
        setter = getattr(torch_module, setter_name, None)
        if not callable(setter):
            continue
        try:
            setter(1)
        except (RuntimeError, TypeError, ValueError):
            pass
    _PARENT_TORCH_THREAD_CAPS_APPLIED = True


def _parent_thread_caps_receipt() -> dict[str, Any]:
    return {
        "enforced": _PARENT_THREAD_CAPS_APPLIED,
        "effective_environment": {
            name: os.environ.get(name) for name in PHASE1_THREAD_CAPS
        },
        "torch_thread_cap_attempted": _PARENT_TORCH_THREAD_CAPS_APPLIED,
    }


def _parent_stock_library_receipt() -> dict[str, Any]:
    """Return only the parent-side, pre-load r236 identity evidence."""

    if _PARENT_STOCK_LIBRARY_RECEIPT is None:
        return {"validated": False}
    return dict(_PARENT_STOCK_LIBRARY_RECEIPT)


def _parent_cuda_runtime_before_search_receipt() -> dict[str, Any]:
    """Return the parent model's observational pre-search CUDA record.

    The direct runtime is lazy.  A missing record therefore describes an
    as-yet-unloaded model rather than a claim that the Kaggle host has no GPU.
    This field is diagnostic-only and never changes the frozen action path.
    """

    if _PARENT_CUDA_RUNTIME_BEFORE_SEARCH is not None:
        return dict(_PARENT_CUDA_RUNTIME_BEFORE_SEARCH)
    return {
        "schema": CUDA_RUNTIME_OBSERVATION_SCHEMA,
        "phase": CUDA_RUNTIME_OBSERVATION_PHASE,
        "torch_imported": False,
        "cuda_available": False,
        "cuda_initialized": False,
        "device_count": 0,
        "devices": [],
        "model_device": "unavailable",
        "telemetry_complete": False,
        "error_types": ["parent_runtime:ModelNotLoadedBeforeSearch"],
    }


def _capture_parent_cuda_runtime_before_search(model: Any) -> None:
    """Observe the existing r195 model device after lazy load, before search."""

    global _PARENT_CUDA_RUNTIME_BEFORE_SEARCH
    try:
        _PARENT_CUDA_RUNTIME_BEFORE_SEARCH = dict(
            capture_cuda_runtime_before_search(model)
        )
    except Exception as exc:  # diagnostic capture never changes action authority
        _PARENT_CUDA_RUNTIME_BEFORE_SEARCH = {
            "schema": CUDA_RUNTIME_OBSERVATION_SCHEMA,
            "phase": CUDA_RUNTIME_OBSERVATION_PHASE,
            "torch_imported": False,
            "cuda_available": False,
            "cuda_initialized": False,
            "device_count": 0,
            "devices": [],
            "model_device": "unavailable",
            "telemetry_complete": False,
            "error_types": [f"parent_capture:{type(exc).__name__}"],
        }


def _validate_parent_staged_stock_library_identity() -> None:
    """Verify the staged r236 DSO before importing the frozen direct runtime.

    The runtime helper hashes a file only; it deliberately does not load
    ``cg``/``libcg``.  That keeps a missing or tampered staged native member
    from gaining a parent-process mapping before the identity boundary.
    """

    global _PARENT_STOCK_LIBRARY_RECEIPT
    if _PARENT_STOCK_LIBRARY_RECEIPT is not None:
        return
    try:
        from poke_bot.r228_kaggle_async_runtime import (
            validate_staged_stock_library_identity,
        )

        receipt = validate_staged_stock_library_identity(_agent_dir())
    except Exception as exc:
        _hard_failure(code="parent_stock_library_identity_invalid", exc=exc)
    if not isinstance(receipt, Mapping):
        _hard_failure(code="parent_stock_library_identity_invalid")
    normalized = dict(receipt)
    # The Kaggle submission image is Linux x86-64.  Do not accept a generic
    # cross-platform-looking receipt there: this replacement binds the exact
    # official 1.32.6 Linux member, not the historical r195 DSO.
    machine = platform.machine().lower()
    if sys.platform.startswith("linux") and machine in {"x86_64", "amd64"}:
        if (
            normalized.get("member") != R236_LINUX_X86_64_STOCK_LIBRARY_MEMBER
            or normalized.get("sha256") != R236_LINUX_X86_64_STOCK_LIBRARY_SHA256
        ):
            _hard_failure(
                code="parent_stock_library_identity_invalid",
                stock_library=normalized,
                expected_member=R236_LINUX_X86_64_STOCK_LIBRARY_MEMBER,
                expected_sha256=R236_LINUX_X86_64_STOCK_LIBRARY_SHA256,
            )
    _PARENT_STOCK_LIBRARY_RECEIPT = normalized


def _direct() -> Any:
    global _DIRECT
    if _DIRECT is not None:
        return _DIRECT
    source = _agent_dir() / "r195_direct_main.py"
    spec = importlib.util.spec_from_file_location("r228_r195_direct", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen r195 direct entrypoint")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _DIRECT = module
    return module


def _json_marker(prefix: str, payload: Mapping[str, Any]) -> None:
    print(prefix + " " + json.dumps(dict(payload), sort_keys=True), flush=True)


def _hard_failure(*, code: str, exc: BaseException | None = None, **extra: Any) -> None:
    """Emit a hard marker only for parent identity/legality/reap failures."""

    payload: dict[str, Any] = {
        "schema": CONTAINMENT_SCHEMA,
        "code": str(code),
        "parent_stock_library": _parent_stock_library_receipt(),
        "parent_cuda_runtime_before_search": _parent_cuda_runtime_before_search_receipt(),
    }
    if exc is not None:
        payload.update(
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    payload.update(extra)
    _json_marker(HARD_FAILURE_PREFIX, payload)
    if exc is None:
        raise RuntimeError(str(code))
    raise RuntimeError(f"{code}: {exc}") from exc


def _action(value: object, *, label: str) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a list of integer option indices")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{label} contains a non-integer option index")
        result.append(int(item))
    return result


class _ValidatedDirectPolicyReceiptError(RuntimeError):
    """The frozen direct call did not leave auditable, non-random evidence."""


def _direct_policy_and_targets(direct: Any) -> tuple[Any, list[Any], int, object]:
    """Enable one temporary direct-policy target without changing action state."""

    ensure_runtime = getattr(direct, "_ensure_runtime", None)
    if not callable(ensure_runtime):
        raise _ValidatedDirectPolicyReceiptError(
            "frozen direct entrypoint lacks _ensure_runtime"
        )
    try:
        runtime = ensure_runtime()
        _capture_parent_cuda_runtime_before_search(runtime[1])
        policy = runtime[2]
    except Exception as exc:
        raise _ValidatedDirectPolicyReceiptError(
            f"cannot resolve frozen direct policy: {type(exc).__name__}: {exc}"
        ) from exc
    targets = getattr(policy, "targets", None)
    if not isinstance(targets, list) or not hasattr(policy, "collect_targets"):
        raise _ValidatedDirectPolicyReceiptError(
            "frozen direct policy cannot temporarily collect a target receipt"
        )
    return policy, targets, len(targets), getattr(policy, "collect_targets")


def _history_lengths(policy: Any) -> tuple[int, int, int]:
    boards = getattr(policy, "board_history", None)
    previous = getattr(policy, "previous_action_history", None)
    limit_getter = getattr(policy, "_history_context_limit", None)
    if not hasattr(boards, "__len__") or not hasattr(previous, "__len__"):
        raise _ValidatedDirectPolicyReceiptError(
            "frozen direct policy lacks history state"
        )
    if not callable(limit_getter):
        raise _ValidatedDirectPolicyReceiptError(
            "frozen direct policy lacks its history-context limit"
        )
    try:
        limit = int(limit_getter())
    except Exception as exc:
        raise _ValidatedDirectPolicyReceiptError(
            f"frozen direct policy history limit is invalid: {exc}"
        ) from exc
    if limit <= 0:
        raise _ValidatedDirectPolicyReceiptError(
            "frozen direct policy history limit is non-positive"
        )
    return len(boards), len(previous), limit


def _validated_factorized_direct_receipt(
    obs_dict: Mapping[str, Any],
    direct_action: Sequence[int],
    target: Mapping[str, Any],
    *,
    expected_history_length: int,
) -> dict[str, Any]:
    """Validate the one actual frozen-policy target before trusting fallback.

    The stock r195 entrypoint catches inference failures and can fail closed to
    a random legal response.  Only its factorized, trusted direct target is
    allowed to establish r238/r240 fallback authority.
    """

    try:
        target_action = _action(target.get("action"), label="direct target action")
    except Exception as exc:
        raise _ValidatedDirectPolicyReceiptError(
            f"direct target action is malformed: {exc}"
        ) from exc
    if target_action != list(direct_action):
        raise _ValidatedDirectPolicyReceiptError(
            "direct target action does not match the returned direct action"
        )
    diagnostics = target.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise _ValidatedDirectPolicyReceiptError("direct target diagnostics are absent")
    if diagnostics.get("target_source") != "history_policy" or diagnostics.get(
        "trusted"
    ) is not True:
        raise _ValidatedDirectPolicyReceiptError(
            "direct target is not the trusted frozen history policy"
        )
    if diagnostics.get("history_length") != expected_history_length:
        raise _ValidatedDirectPolicyReceiptError(
            "direct target history length does not prove one committed step"
        )

    stages = target.get("factorized_stages")
    if not isinstance(stages, list) or not stages:
        raise _ValidatedDirectPolicyReceiptError(
            "direct target lacks factorized selected-stage telemetry"
        )
    from poke_bot import features

    prefix: list[int] = []
    selected_probabilities: list[float] = []
    for stage_index, stage in enumerate(stages):
        if not isinstance(stage, Mapping):
            raise _ValidatedDirectPolicyReceiptError(
                f"direct factorized stage {stage_index} is malformed"
            )
        raw_candidates = stage.get("action_combos")
        raw_policy = stage.get("policy")
        selected_index = stage.get("selected_index")
        if not isinstance(raw_candidates, list) or not isinstance(raw_policy, list):
            raise _ValidatedDirectPolicyReceiptError(
                f"direct factorized stage {stage_index} lacks candidates or policy"
            )
        if isinstance(selected_index, bool) or not isinstance(selected_index, int):
            raise _ValidatedDirectPolicyReceiptError(
                f"direct factorized stage {stage_index} selected index is invalid"
            )
        try:
            candidates = [
                _action(candidate, label="direct factorized candidate")
                for candidate in raw_candidates
            ]
            expected_candidates = [
                _action(candidate, label="expected factorized candidate")
                for candidate in features.factorized_action_candidates(
                    dict(obs_dict), list(prefix)
                )
            ]
            probabilities = [float(value) for value in raw_policy]
        except Exception as exc:
            raise _ValidatedDirectPolicyReceiptError(
                f"direct factorized stage {stage_index} cannot be decoded: {exc}"
            ) from exc
        if candidates != expected_candidates:
            raise _ValidatedDirectPolicyReceiptError(
                f"direct factorized stage {stage_index} candidate order drifted"
            )
        if len(probabilities) != len(candidates) or not (
            0 <= selected_index < len(candidates)
        ):
            raise _ValidatedDirectPolicyReceiptError(
                f"direct factorized stage {stage_index} has an invalid policy shape"
            )
        if any(not math.isfinite(value) or value < 0.0 for value in probabilities):
            raise _ValidatedDirectPolicyReceiptError(
                f"direct factorized stage {stage_index} has non-finite probabilities"
            )
        if not math.isclose(sum(probabilities), 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise _ValidatedDirectPolicyReceiptError(
                f"direct factorized stage {stage_index} probabilities do not normalize"
            )
        selected = candidates[selected_index]
        selected_probability = probabilities[selected_index]
        if not math.isfinite(selected_probability):
            raise _ValidatedDirectPolicyReceiptError(
                f"direct factorized stage {stage_index} selected probability is invalid"
            )
        selected_probabilities.append(selected_probability)
        if selected == prefix:
            if stage_index != len(stages) - 1:
                raise _ValidatedDirectPolicyReceiptError(
                    "direct factorized telemetry continues after its stop choice"
                )
            break
        if selected[: len(prefix)] != prefix:
            raise _ValidatedDirectPolicyReceiptError(
                f"direct factorized stage {stage_index} does not extend its prefix"
            )
        prefix = selected
    if prefix != list(direct_action):
        raise _ValidatedDirectPolicyReceiptError(
            "direct factorized stages do not terminate at the returned action"
        )
    return {
        "factorized_selected_stage_probabilities": selected_probabilities,
        "minimum_selected_factorized_stage_probability": min(selected_probabilities),
        "factorized_stage_count": len(selected_probabilities),
    }


def _precompute_validated_direct_action(
    direct: Any, obs_dict: Mapping[str, Any]
) -> tuple[list[int], dict[str, Any] | None, bool]:
    """Return one frozen direct action and auditable confidence telemetry.

    The r195 turn-order short circuit is deliberately checked first.  It does
    not append model history and remains separate from normal factorized
    policy actions.
    """

    turn_order_choice = getattr(direct, "_turn_order_choice", None)
    if callable(turn_order_choice):
        try:
            turn_order = turn_order_choice(dict(obs_dict))
        except Exception as exc:
            raise _ValidatedDirectPolicyReceiptError(
                f"frozen turn-order resolver failed: {type(exc).__name__}: {exc}"
            ) from exc
        if turn_order is not None:
            choice = _action(turn_order, label="frozen r195 turn-order action")
            if choice:
                return choice, None, True
            # The stock wrapper uses its own exact fail-closed turn-order
            # response for an exposed prompt without a unique preferred row.
            # It is still a setup prompt and never launches MCTS.
            try:
                return (
                    _action(direct.agent(dict(obs_dict)), label="frozen r195 turn-order response"),
                    None,
                    True,
                )
            except Exception as exc:
                raise _ValidatedDirectPolicyReceiptError(
                    f"frozen turn-order fallback failed: {type(exc).__name__}: {exc}"
                ) from exc

    policy, targets, target_start, original_collect = _direct_policy_and_targets(direct)
    before_boards, before_previous, history_limit = _history_lengths(policy)
    target_rows: list[Any] = []
    try:
        policy.collect_targets = True
        action = _action(direct.agent(dict(obs_dict)), label="frozen r195 direct action")
        target_rows = list(targets[target_start:])
    finally:
        # Temporary telemetry must not become an extra training/evaluation row
        # or alter future target cardinality.  It never changes history.
        del targets[target_start:]
        policy.collect_targets = original_collect

    after_boards, after_previous, _after_limit = _history_lengths(policy)
    expected_boards = min(before_boards + 1, history_limit)
    expected_previous = min(before_previous + 1, history_limit)
    if after_boards != expected_boards or after_previous != expected_previous:
        raise _ValidatedDirectPolicyReceiptError(
            "frozen direct policy did not mutate temporal history exactly once"
        )
    if len(target_rows) != 1 or not isinstance(target_rows[0], Mapping):
        raise _ValidatedDirectPolicyReceiptError(
            "frozen direct policy did not emit exactly one temporary target receipt"
        )
    telemetry = _validated_factorized_direct_receipt(
        obs_dict,
        action,
        target_rows[0],
        expected_history_length=after_boards,
    )
    return action, telemetry, False


def _legal_order(obs_dict: Mapping[str, Any]) -> list[list[int]]:
    """Materialize the exact root order or hard-fail above its finite cap."""

    from poke_bot import features

    try:
        raw = features.enumerate_action_combos(
            dict(obs_dict), max_combos=COMPLETE_ACTION_CAP
        )
    except Exception as exc:
        _hard_failure(
            code="complete_root_legal_order_invalid_or_over_cap",
            exc=exc,
            complete_action_cap=COMPLETE_ACTION_CAP,
        )
    try:
        return [_action(action, label="complete legal action") for action in raw]
    except Exception as exc:
        _hard_failure(
            code="complete_root_legal_order_invalid",
            exc=exc,
            complete_action_cap=COMPLETE_ACTION_CAP,
        )
    raise AssertionError("unreachable")


def _legal_order_fingerprint(legal: Sequence[Sequence[int]]) -> str:
    canonical = json.dumps(
        [[int(item) for item in action] for action in legal],
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _fault_key(fault: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        dict(fault), sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _record_fault(fault: Mapping[str, Any]) -> str:
    global _GAME_DEGRADED_FAULTS
    key = _fault_key(fault)
    if key not in _GAME_FAULT_KEYS:
        _GAME_FAULT_KEYS.add(key)
        _GAME_DEGRADED_FAULTS += 1
    return key


def _append_parent_direct_journal(
    obs_dict: Mapping[str, Any], actual_action: Sequence[int]
) -> None:
    """Retain an exact real-action row before a deferred child exists."""

    _GAME_DIRECT_JOURNAL.append(
        {
            "observation": dict(obs_dict),
            "action": [int(item) for item in actual_action],
        }
    )


def _broker_state(broker: IsolatedR228SearchBroker | None) -> dict[str, Any]:
    if broker is None:
        return {
            "schema": "poke_bot.r228_kaggle_subprocess_broker/v1",
            "disabled": False,
            "degraded": False,
            "not_started": True,
            "child_pid": None,
            "progress_by_lane": {},
            "configured_simulator_lane_count": SIMULATOR_SEARCH_LANE_COUNT,
            "phase1_thread_caps": dict(PHASE1_THREAD_CAPS),
        }
    try:
        state = broker.marker_payload()
    except Exception as exc:
        # The causal child receipt remains authoritative when collecting its
        # optional diagnostic state itself fails.
        return {
            "schema": "poke_bot.r228_kaggle_subprocess_broker/v1",
            "disabled": True,
            "degraded": True,
            "child_pid": None,
            "progress_by_lane": {},
            "configured_simulator_lane_count": SIMULATOR_SEARCH_LANE_COUNT,
            "phase1_thread_caps": dict(PHASE1_THREAD_CAPS),
            "marker_payload_error": f"{type(exc).__name__}: {exc}",
        }
    return dict(state) if isinstance(state, Mapping) else {}


def _synthesized_fault(
    broker: IsolatedR228SearchBroker | None, *, code: str, message: str
) -> dict[str, Any]:
    state = _broker_state(broker)
    return {
        "schema": CONTAINMENT_SCHEMA,
        "kind": "r228_broker_fault",
        "code": code,
        "message": message,
        "child_pid": state.get("child_pid"),
        "child_identity": state.get("child_identity"),
        "progress_by_lane": state.get("progress_by_lane") or {},
        "configured_simulator_lane_count": SIMULATOR_SEARCH_LANE_COUNT,
    }


def _disable_broker_after_contained_fault(
    broker: IsolatedR228SearchBroker,
    *,
    fault: Mapping[str, Any],
) -> None:
    """Latch direct-only play and reap an unexpectedly live child.

    The normal ``select``/``note_direct_action`` fault paths already dispose
    their child.  This is for faults observed by the parent while decoding a
    public broker result, where continuing to use an uncertain child would
    violate the one-game containment boundary.
    """

    global _GAME_MCTS_DISABLED, _GAME_PENDING_FAULT
    _GAME_MCTS_DISABLED = True
    _GAME_PENDING_FAULT = dict(fault)
    try:
        if not broker.disabled:
            broker.close()
    except Exception as exc:
        # A failed exact-child close means the parent cannot establish that
        # native work is gone.  This is the explicitly non-contained case.
        _hard_failure(code="broker_child_not_reaped", exc=exc)


def _contained_broker_fallback(
    *,
    broker: IsolatedR228SearchBroker,
    fault: Mapping[str, Any],
    direct_action: Sequence[int],
    legal: Sequence[Sequence[int]],
) -> list[int]:
    """Make a contained broker/protocol fault direct-only for this game."""

    _clear_principal_variation()
    _disable_broker_after_contained_fault(broker, fault=fault)
    _emit_degraded_fallback(
        broker=broker,
        fault=fault,
        direct_action=direct_action,
        legal=legal,
    )
    return list(direct_action)


def _two_lane_progress_subset(value: object) -> dict[str, Any]:
    """Keep the public degraded progress projection inside lane ids 0 and 1."""

    if not isinstance(value, Mapping):
        return {}
    normalized: dict[str, Any] = {}
    for raw_lane, progress in value.items():
        try:
            if isinstance(raw_lane, bool):
                continue
            lane = int(raw_lane)
        except (TypeError, ValueError):
            continue
        if 0 <= lane < SIMULATOR_SEARCH_LANE_COUNT:
            normalized[str(lane)] = progress
    return normalized


def _emit_degraded_fallback(
    *,
    broker: IsolatedR228SearchBroker | None,
    fault: Mapping[str, Any],
    direct_action: Sequence[int],
    legal: Sequence[Sequence[int]],
) -> None:
    """Emit exactly one containment receipt for each distinct child fault."""

    key = _record_fault(fault)
    if key in _EMITTED_FAULT_KEYS:
        return
    _EMITTED_FAULT_KEYS.add(key)
    state = _broker_state(broker)
    child_fault = dict(fault)
    per_lane = _two_lane_progress_subset(
        child_fault.get("progress_by_lane") or state.get("progress_by_lane") or {}
    )
    _json_marker(
        DEGRADED_PREFIX,
        {
            "schema": CONTAINMENT_SCHEMA,
            "action_authority": "precomputed_frozen_r195_direct_action",
            "mcts_action_authority": False,
            "selected_action": list(direct_action),
            "direct_action": list(direct_action),
            "legal_action_count": len(legal),
            "legal_order_fingerprint": _legal_order_fingerprint(legal),
            "complete_action_cap": COMPLETE_ACTION_CAP,
            "configured_simulator_lane_count": SIMULATOR_SEARCH_LANE_COUNT,
            "child_pid": child_fault.get("child_pid", state.get("child_pid")),
            "child_identity": child_fault.get(
                "child_identity", state.get("child_identity")
            ),
            "child_fault": child_fault,
            "per_lane_progress": per_lane,
            "parent_thread_caps": _parent_thread_caps_receipt(),
            "parent_stock_library": _parent_stock_library_receipt(),
            "parent_cuda_runtime_before_search": _parent_cuda_runtime_before_search_receipt(),
            "broker": state,
        },
    )


def _new_broker() -> IsolatedR228SearchBroker:
    # Keep the parent and child topology identities coupled.  A legacy broker
    # module must never silently launch an eight-lane child under the r238
    # two-vCPU receipt label.
    if BROKER_SIMULATOR_SEARCH_LANE_COUNT != SIMULATOR_SEARCH_LANE_COUNT:
        _hard_failure(
            code="broker_identity_invalid",
            parent_simulator_lane_count=SIMULATOR_SEARCH_LANE_COUNT,
            broker_simulator_lane_count=BROKER_SIMULATOR_SEARCH_LANE_COUNT,
        )
    try:
        return IsolatedR228SearchBroker(
            stage=_agent_dir(),
            action_timeout_seconds=R234_BROKER_ACTION_TIMEOUT_SECONDS,
            search_seconds=R234_BROKER_SEARCH_SECONDS,
            startup_timeout_seconds=R234_BROKER_STARTUP_TIMEOUT_SECONDS,
            reap_grace_seconds=R234_BROKER_REAP_GRACE_SECONDS,
        )
    except Exception as exc:
        _hard_failure(code="broker_identity_invalid", exc=exc)
    raise AssertionError("unreachable")


def _close_broker_at_boundary() -> None:
    global _BROKER
    broker = _BROKER
    if broker is None:
        return
    try:
        broker.close()
        state = _broker_state(broker)
    except Exception as exc:
        _hard_failure(code="broker_child_not_reaped", exc=exc)
    if state.get("child_pid") is not None:
        _hard_failure(
            code="broker_child_not_reaped",
            broker=state,
        )
    _BROKER = None


def _begin_game() -> None:
    global _BROKER, _GAME_SUCCESSFUL_MCTS_DECISIONS, _GAME_DEGRADED_FAULTS
    global _GAME_FAULT_KEYS, _EMITTED_FAULT_KEYS, _GAME_MCTS_DISABLED
    global _GAME_PENDING_FAULT, _GAME_DIRECT_JOURNAL
    global _GAME_PRINCIPAL_VARIATION_ROOT_SEAT
    _close_broker_at_boundary()
    _GAME_SUCCESSFUL_MCTS_DECISIONS = 0
    _GAME_DEGRADED_FAULTS = 0
    _GAME_FAULT_KEYS = set()
    _EMITTED_FAULT_KEYS = set()
    _GAME_MCTS_DISABLED = False
    _GAME_PENDING_FAULT = None
    _GAME_DIRECT_JOURNAL = []
    _clear_principal_variation()
    # Do not launch native work at a deck boundary.  r240 direct/forced turns
    # remain parent-only and are replayed into a child only if a later branch
    # genuinely needs MCTS.
    _BROKER = None


def _initialize_deferred_broker(
    legal: Sequence[Sequence[int]],
) -> IsolatedR228SearchBroker:
    """Create and replay the direct journal without starting native search."""

    global _BROKER, _GAME_MCTS_DISABLED, _GAME_PENDING_FAULT
    _BROKER = _new_broker()
    try:
        _BROKER.begin_game(start_child=False)
    except R228BrokerError as exc:
        if exc.code == "child_unreaped":
            _hard_failure(code="broker_child_not_reaped", exc=exc)
        # The concrete broker contains startup failures internally.  Retain a
        # direct-only fault if an equivalent public implementation reports it
        # instead, then emit the receipt at the first selection prompt.
        fault = _synthesized_fault(
            _BROKER, code=exc.code, message=f"broker begin_game failed: {exc}"
        )
        _record_fault(fault)
        _disable_broker_after_contained_fault(_BROKER, fault=fault)
    except Exception as exc:
        fault = _synthesized_fault(
            _BROKER,
            code="broker_startup_exception",
            message=f"broker begin_game raised {type(exc).__name__}: {exc}",
        )
        _record_fault(fault)
        _disable_broker_after_contained_fault(_BROKER, fault=fault)
    if _BROKER.degraded and _BROKER.last_fault is not None:
        startup_fault = _BROKER.last_fault
        _record_fault(startup_fault)
        _GAME_MCTS_DISABLED = True
        _GAME_PENDING_FAULT = dict(startup_fault)
    if not _GAME_MCTS_DISABLED:
        for event in _GAME_DIRECT_JOURNAL:
            observation = event.get("observation")
            action = event.get("action")
            if not isinstance(observation, Mapping) or not isinstance(action, list):
                _hard_failure(code="parent_direct_journal_identity_invalid")
            try:
                _BROKER.note_direct_action(observation, action)
            except R228BrokerError as exc:
                if exc.code == "child_unreaped":
                    _hard_failure(code="broker_child_not_reaped", exc=exc)
                fault = _synthesized_fault(
                    _BROKER,
                    code=exc.code,
                    message=f"broker journal replay failed: {exc}",
                )
                _disable_broker_after_contained_fault(_BROKER, fault=fault)
                break
            except Exception as exc:
                fault = _synthesized_fault(
                    _BROKER,
                    code="broker_journal_replay_exception",
                    message=(
                        "broker journal replay raised "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
                _disable_broker_after_contained_fault(_BROKER, fault=fault)
                break
            if _BROKER.degraded:
                replay_fault = _BROKER.last_fault or _synthesized_fault(
                    _BROKER,
                    code="broker_journal_replay_degraded",
                    message="broker degraded while replaying parent journal",
                )
                _GAME_MCTS_DISABLED = True
                _GAME_PENDING_FAULT = dict(replay_fault)
                break
    return _BROKER


def _ensure_broker_for_selection(
    legal: Sequence[Sequence[int]],
) -> IsolatedR228SearchBroker:
    global _BROKER
    if _BROKER is None:
        return _initialize_deferred_broker(legal)
    assert _BROKER is not None
    return _BROKER


def _note_direct_action(
    broker: IsolatedR228SearchBroker,
    obs_dict: Mapping[str, Any],
    direct_action: Sequence[int],
    legal: Sequence[Sequence[int]],
) -> None:
    """Journal every setup/forced/direct real action without a second policy call."""

    global _GAME_MCTS_DISABLED, _GAME_PENDING_FAULT
    prior_fault = broker.last_fault
    try:
        broker.note_direct_action(obs_dict, direct_action)
    except R228BrokerError as exc:
        if exc.code == "child_unreaped":
            _hard_failure(code="broker_child_not_reaped", exc=exc)
        if exc.code == "illegal_direct_action":
            _hard_failure(code="broker_direct_action_illegal", exc=exc)
        fault = _synthesized_fault(
            broker, code=exc.code, message=f"broker direct journal failed: {exc}"
        )
        _contained_broker_fallback(
            broker=broker,
            fault=fault,
            direct_action=direct_action,
            legal=legal,
        )
        return
    except Exception as exc:
        fault = _synthesized_fault(
            broker,
            code="broker_direct_journal_exception",
            message=f"broker direct journal raised {type(exc).__name__}: {exc}",
        )
        _contained_broker_fallback(
            broker=broker,
            fault=fault,
            direct_action=direct_action,
            legal=legal,
        )
        return
    fault = broker.last_fault
    if broker.degraded and fault is not None and fault != prior_fault:
        _GAME_MCTS_DISABLED = True
        _GAME_PENDING_FAULT = dict(fault)
        _emit_degraded_fallback(
            broker=broker,
            fault=fault,
            direct_action=direct_action,
            legal=legal,
        )
    elif broker.degraded and fault is not None:
        # Startup degradation has no action to mark until the first prompt.
        _GAME_MCTS_DISABLED = True
        _GAME_PENDING_FAULT = dict(fault)
        _emit_degraded_fallback(
            broker=broker,
            fault=fault,
            direct_action=direct_action,
            legal=legal,
        )


def _journal_real_action(
    obs_dict: Mapping[str, Any],
    actual_action: Sequence[int],
    legal: Sequence[Sequence[int]],
) -> int:
    """Record one parent-authoritative action, with no eager native startup."""

    _append_parent_direct_journal(obs_dict, actual_action)
    broker = _BROKER
    if broker is not None:
        # A clean zero-backup deadline has already reaped its child.  In that
        # state ``note_direct_action`` only extends the broker's deferred
        # replay journal; it is not the permitted IPC to an existing child.
        # Test doubles from earlier entrypoint coverage do not expose this
        # property, so preserve their established live-child interpretation.
        live_child = getattr(broker, "has_live_child", True)
        had_existing_child = bool(live_child)
        _note_direct_action(broker, obs_dict, actual_action, legal)
        # This is a history-only IPC note to a child that already existed.  It
        # never constructs a child or asks it to select/search for this action.
        return 1 if had_existing_child else 0
    return 0


def _overwrite_parent_action_token(
    direct: Any, obs_dict: Mapping[str, Any], selected_action: Sequence[int]
) -> None:
    """Replace only the direct token; direct.agent already appended history once."""

    try:
        _deck, _model, policy = direct._ensure_runtime()
        from poke_bot import features

        policy._previous_action_token = features.build_option_tokens(
            dict(obs_dict), [list(selected_action)]
        )
    except Exception as exc:
        _hard_failure(code="parent_action_history_identity_invalid", exc=exc)


def _receipt_exact_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an exact integer")
    return int(value)


def _receipt_handle_identity(value: object, *, field: str) -> int | str:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{field} must be an integer or string handle identity")
    if isinstance(value, str) and not value:
        raise ValueError(f"{field} must not be an empty handle identity")
    return value


def _validate_handle_scoped_first_search_id_composites(
    receipt: Mapping[str, Any],
    *,
    handles: Sequence[int | str],
    first_search_ids: Sequence[int],
) -> None:
    expected = SIMULATOR_SEARCH_LANE_COUNT
    first_ids = receipt.get("per_lane_first_search_ids")
    if not isinstance(first_ids, (list, tuple)) or len(first_ids) != expected:
        raise ValueError("receipt lacks both per-lane first SearchIds")
    if [
        _receipt_exact_int(value, field="per_lane_first_search_ids")
        for value in first_ids
    ] != list(first_search_ids):
        raise ValueError("per-lane first SearchIds disagree with the SearchBegin chains")
    states = receipt.get("handle_scoped_first_search_id_composite_states")
    if not isinstance(states, (list, tuple)) or len(states) != expected:
        raise ValueError("receipt lacks both handle-scoped SearchBegin composites")
    for lane_id, state in enumerate(states):
        if not isinstance(state, Mapping) or set(state) != {
            "lane_id",
            "handle_identity",
            "first_search_id",
        }:
            raise ValueError("receipt has a malformed handle-scoped composite")
        if _receipt_exact_int(state.get("lane_id"), field="lane_id") != lane_id:
            raise ValueError("receipt handle-scoped composites are not lane ordered")
        if (
            _receipt_handle_identity(
                state.get("handle_identity"), field="handle_identity"
            )
            != handles[lane_id]
            or _receipt_exact_int(state.get("first_search_id"), field="first_search_id")
            != first_search_ids[lane_id]
        ):
            raise ValueError("receipt composite disagrees with the lane vectors")


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
    receipt: Mapping[str, Any],
    *,
    obs_dict: Mapping[str, Any],
    legal: Sequence[Sequence[int]],
    selected_action: Sequence[int],
) -> None:
    """Bind r246 terminal authority to the parent's exact current prompt."""

    proof = receipt.get("terminal_win_proof")
    if not isinstance(proof, Mapping) or set(proof) != set(_TERMINAL_WIN_PROOF_KEYS):
        raise ValueError("terminal-win proof has an invalid exact schema")
    if _receipt_exact_int(
        receipt.get("owner_proven_deterministic_terminal_win_this_turn_revision"),
        field="owner_proven_deterministic_terminal_win_this_turn_revision",
    ) != PROVEN_TERMINAL_WIN_REVISION:
        raise ValueError("terminal-win proof has an invalid owner revision")
    root_fingerprint = _canonical_observation_fingerprint(obs_dict)
    if root_fingerprint is None:
        raise ValueError("current root cannot produce an exact observation fingerprint")
    legal_fingerprint = _legal_order_fingerprint(legal)
    actor = _current_actor_seat(obs_dict)
    if actor not in (0, 1):
        raise ValueError("current root actor is invalid")
    if (
        receipt.get("root_observation_fingerprint") != root_fingerprint
        or proof.get("root_observation_fingerprint") != root_fingerprint
    ):
        raise ValueError("terminal-win proof is stale for the current root")
    if (
        receipt.get("root_legal_order_fingerprint") != legal_fingerprint
        or proof.get("root_legal_order_fingerprint") != legal_fingerprint
    ):
        raise ValueError("terminal-win proof is stale for the current legal order")
    if (
        _receipt_exact_int(receipt.get("root_actor_seat"), field="root_actor_seat")
        != actor
        or _receipt_exact_int(receipt.get("root_seat"), field="root_seat")
        != actor
        or _receipt_exact_int(proof.get("root_actor_seat"), field="root_actor_seat")
        != actor
    ):
        raise ValueError("terminal-win proof is stale for the current actor")
    if proof.get("proof_kind") != PROVEN_TERMINAL_WIN_PROOF_KIND:
        raise ValueError("terminal-win proof kind is invalid")
    if proof.get("terminal_result") != "win" or _receipt_exact_int(
        proof.get("terminal_winner_seat"), field="terminal_winner_seat"
    ) != actor:
        raise ValueError("terminal-win proof is not a win for the current actor")
    for field in (
        "terminal_leaf_reached",
        "path_no_actor_change_boundary",
        "path_no_opponent_boundary_crossing",
        "path_no_chance_boundary",
        "path_no_unresolved_randomness",
        "proof_is_deterministic",
    ):
        if proof.get(field) is not True:
            raise ValueError(f"terminal-win proof does not prove {field}")
    proof_root_action = _action(proof.get("root_action"), label="proof root action")
    proof_selected = _action(
        proof.get("selected_action"), label="proof selected action"
    )
    normalized_selected = [int(item) for item in selected_action]
    if (
        proof_root_action != proof_selected
        or proof_selected != normalized_selected
        or proof_selected not in legal
        or _action(receipt.get("selected_action"), label="receipt selected action")
        != normalized_selected
    ):
        raise ValueError("terminal-win proof does not bind its selected legal action")
    path_count = _receipt_exact_int(
        proof.get("proof_path_action_count"), field="proof_path_action_count"
    )
    path_actors = proof.get("path_actor_seats")
    if (
        path_count < 1
        or path_count
        > _receipt_exact_int(receipt.get("completed_backups"), field="completed_backups")
        or not isinstance(path_actors, (list, tuple))
        or len(path_actors) != path_count
        or any(
            isinstance(path_actor, bool)
            or not isinstance(path_actor, int)
            or path_actor != actor
            for path_actor in path_actors
        )
    ):
        raise ValueError("terminal-win proof path is not root-actor-only")
    lane = _receipt_exact_int(
        proof.get("discovering_lane_id"), field="discovering_lane_id"
    )
    if not 0 <= lane < SIMULATOR_SEARCH_LANE_COUNT:
        raise ValueError("terminal-win proof has an invalid discovering lane")
    depths = receipt.get("per_lane_depth")
    if not isinstance(depths, (list, tuple)) or _receipt_exact_int(
        depths[lane], field="per_lane_depth"
    ) < path_count:
        raise ValueError("terminal-win proof exceeds its backed lane depth")
    if receipt.get("principal_variation") not in ([], ()):
        raise ValueError("terminal-win receipt retained a continuation plan")
    if receipt.get("proven_deterministic_terminal_win_this_turn") is not True:
        raise ValueError("terminal-win receipt omitted its public proof classification")


def _receipt_finite_nonnegative_seconds(value: Any, *, field: str) -> float:
    """Decode an elapsed field without accepting a compatibility default."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be a finite nonnegative number")
    return result


def _validate_terminal_win_execution_marker_facts(
    receipt: Mapping[str, Any],
) -> None:
    """Require literal child/broker facts before the parent emits R246 truth.

    The parent does not reconstruct these facts from a stop reason, ordinary
    counts, or old elapsed spelling.  Missing source evidence is contained as
    a broker protocol fault instead of becoming a terminal-win marker.
    """

    for field in (
        "broker_started",
        "mcts_child_started",
        "mcts_child_called",
        "two_lane_topology_initialized_before_terminal_win_override",
        "terminal_win_proof_backed_up_into_shared_root_tree",
        "terminal_leaf_returned_by_exact_stock_simulator",
        "all_owned_lane_resources_reservations_and_child_cleanup_complete",
    ):
        if receipt.get(field) is not True:
            raise ValueError(f"terminal-win receipt lacks literal {field}")
    completed_backups = _receipt_exact_int(
        receipt.get("completed_backups"), field="completed_backups"
    )
    if _receipt_exact_int(
        receipt.get("completed_root_backup_count"),
        field="completed_root_backup_count",
    ) != completed_backups:
        raise ValueError("terminal-win root backup count disagrees with receipt")
    for field in (
        "terminal_win_proof_count",
        "proven_deterministic_terminal_win_this_turn_stop_count",
    ):
        if _receipt_exact_int(receipt.get(field), field=field) != 1:
            raise ValueError(f"terminal-win receipt has invalid {field}")
    child_elapsed = _receipt_finite_nonnegative_seconds(
        receipt.get("child_search_elapsed_seconds"),
        field="child_search_elapsed_seconds",
    )
    runtime_elapsed = _receipt_finite_nonnegative_seconds(
        receipt.get("elapsed_seconds"), field="elapsed_seconds"
    )
    if child_elapsed != runtime_elapsed:
        raise ValueError("terminal-win child elapsed aliases disagree")


def _validate_phase1_two_lane_receipt(
    receipt: Mapping[str, Any],
    *,
    obs_dict: Mapping[str, Any] | None = None,
    legal: Sequence[Sequence[int]] | None = None,
    selected_action: Sequence[int] | None = None,
) -> None:
    """Independently bind all parent-visible r238 action authority fields."""

    expected = SIMULATOR_SEARCH_LANE_COUNT
    stop_reason = receipt.get("stop_reason")
    if stop_reason not in {
        "stable_root_leader",
        "maximum_backups",
        "decision_deadline",
        "tree_exhausted",
        PROVEN_TERMINAL_WIN_STOP_REASON,
    }:
        raise ValueError("receipt has an unrecognized adaptive stop reason")
    terminal_win_stop = stop_reason == PROVEN_TERMINAL_WIN_STOP_REASON
    for field in (
        "requested_simulator_lane_count",
        "active_simulator_lane_count",
        "arena_count",
        "unique_handle_count",
        "search_begin_calls",
        "search_end_calls",
    ):
        if _receipt_exact_int(receipt.get(field), field=field) != expected:
            raise ValueError(f"{field} must equal the configured two lanes")
    if _receipt_exact_int(
        receipt.get("search_release_calls"), field="search_release_calls"
    ) < expected:
        raise ValueError("search_release_calls is below the lane count")
    depths = receipt.get("per_lane_depth")
    if not isinstance(depths, (list, tuple)) or len(depths) != expected:
        raise ValueError("per_lane_depth does not contain both lanes")
    normalized_depths = [
        _receipt_exact_int(value, field="per_lane_depth") for value in depths
    ]
    minimum_depth = 0 if terminal_win_stop else 1
    if any(depth < minimum_depth for depth in normalized_depths):
        raise ValueError("per_lane_depth has a lane without a completed step")
    if terminal_win_stop and not any(depth >= 1 for depth in normalized_depths):
        raise ValueError("terminal-win receipt has no backed discovering lane")
    chains = receipt.get("per_lane_search_id_chains")
    if not isinstance(chains, (list, tuple)) or len(chains) != expected:
        raise ValueError("per_lane_search_id_chains does not contain both lanes")
    handles = receipt.get("per_lane_handle_identities")
    if not isinstance(handles, (list, tuple)) or len(handles) != expected:
        raise ValueError("per_lane_handle_identities does not contain both lanes")
    normalized_handles = [
        _receipt_handle_identity(value, field="per_lane_handle_identities")
        for value in handles
    ]
    if len(set(normalized_handles)) != expected:
        raise ValueError("the receipt does not prove distinct raw handles")

    first_search_ids: list[int] = []
    for chain in chains:
        if not isinstance(chain, (list, tuple)) or not chain:
            raise ValueError("a simulator lane lacks a SearchBegin-id chain")
        first_search_ids.append(
            _receipt_exact_int(chain[0], field="per_lane_search_id_chains")
        )
    # Stock libcg allocates SearchId values per AgentStart handle.  Raw first
    # IDs may therefore both be zero; only the (handle, SearchId) composite is
    # a global topology witness.
    if len(set(zip(normalized_handles, first_search_ids))) != expected:
        raise ValueError("the receipt lacks distinct handle/SearchBegin composites")
    if _receipt_exact_int(
        receipt.get("distinct_search_begin_composite_count"),
        field="distinct_search_begin_composite_count",
    ) != expected:
        raise ValueError("the receipt composite SearchBegin count is invalid")
    _validate_handle_scoped_first_search_id_composites(
        receipt,
        handles=normalized_handles,
        first_search_ids=first_search_ids,
    )
    microbatches = receipt.get("microbatch_sizes")
    if not isinstance(microbatches, (list, tuple)) or not microbatches:
        raise ValueError("microbatch_sizes is absent")
    if any(
        not 1 <= _receipt_exact_int(value, field="microbatch_sizes") <= expected
        for value in microbatches
    ):
        raise ValueError("microbatch size is outside the two-lane range")
    in_flight = _receipt_exact_int(
        receipt.get("max_simulator_calls_in_flight"),
        field="max_simulator_calls_in_flight",
    )
    if not 1 <= in_flight <= expected:
        raise ValueError("in-flight simulator count is outside the two-lane range")
    if _receipt_exact_int(
        receipt.get("outstanding_virtual_loss"), field="outstanding_virtual_loss"
    ) != 0:
        raise ValueError("receipt retains outstanding virtual loss")

    completed_backups = _receipt_exact_int(
        receipt.get("completed_backups"), field="completed_backups"
    )
    minimum_backups = 1 if terminal_win_stop else expected
    if (
        completed_backups < minimum_backups
        or completed_backups > R238_MAXIMUM_BACKUPS_PER_DECISION
        or sum(normalized_depths) != completed_backups
    ):
        raise ValueError("receipt backup count is invalid for its stop")
    expected_adaptive = {
        "minimum_backups_before_stability": R238_MINIMUM_BACKUPS_BEFORE_STABILITY,
        "stable_root_leader_observations_required": (
            R238_STABLE_ROOT_LEADER_OBSERVATIONS_REQUIRED
        ),
        "maximum_backups_per_decision": R238_MAXIMUM_BACKUPS_PER_DECISION,
    }
    for field, expected_value in expected_adaptive.items():
        if _receipt_exact_int(receipt.get(field), field=field) != expected_value:
            raise ValueError(f"{field} does not bind the r238 adaptive limit")
    observed_stable = _receipt_exact_int(
        receipt.get("observed_stable_root_leader_observations"),
        field="observed_stable_root_leader_observations",
    )
    if not 0 <= observed_stable <= R238_MAXIMUM_BACKUPS_PER_DECISION:
        raise ValueError("observed leader stability is outside the bounded range")
    if (
        stop_reason == "stable_root_leader"
        and observed_stable < R238_STABLE_ROOT_LEADER_OBSERVATIONS_REQUIRED
    ):
        raise ValueError("stable-root stop lacks the required leader observations")
    if terminal_win_stop:
        if obs_dict is None or legal is None or selected_action is None:
            raise ValueError("terminal-win proof lacks the current parent root binding")
        _validate_terminal_win_proof(
            receipt,
            obs_dict=obs_dict,
            legal=legal,
            selected_action=selected_action,
        )
        _validate_terminal_win_execution_marker_facts(receipt)
    elif receipt.get("terminal_win_proof") is not None or receipt.get(
        "proven_deterministic_terminal_win_this_turn"
    ) not in (None, False):
        raise ValueError("non-terminal stop claimed terminal-win authority")


def _validate_phase1_clean_zero_backup_receipt(
    receipt: Mapping[str, Any], *, direct_action: Sequence[int]
) -> None:
    """Bind the clean two-lane deadline proof before direct fallback authority."""

    if receipt.get("mode") != "zero_backup_precomputed_direct_fallback":
        raise ValueError("receipt is not the parent clean-zero fallback mode")
    if receipt.get("child_mode") != "clean_deadline_zero_backup_frozen_model_fallback":
        raise ValueError("receipt lacks the child clean-zero mode")
    if receipt.get("mcts_action_authority") is not False:
        raise ValueError("clean-zero receipt claimed MCTS action authority")
    if receipt.get("zero_backup_precomputed_direct_fallback") is not True:
        raise ValueError("clean-zero receipt does not identify direct fallback")
    if receipt.get("stop_reason") != "decision_deadline":
        raise ValueError("clean-zero receipt lacks a decision-deadline stop")
    if receipt.get("clean_deadline_cleanup_complete") is not True:
        raise ValueError("clean-zero receipt lacks completed native cleanup")
    if _receipt_exact_int(
        receipt.get("completed_backups"), field="completed_backups"
    ) != 0:
        raise ValueError("clean-zero receipt reported completed backups")
    if _action(receipt.get("selected_action"), label="clean-zero selected action") != list(
        direct_action
    ):
        raise ValueError("clean-zero child action differs from parent direct action")

    expected = SIMULATOR_SEARCH_LANE_COUNT
    for field in (
        "requested_simulator_lane_count",
        "active_simulator_lane_count",
        "arena_count",
        "unique_handle_count",
        "search_begin_calls",
        "search_end_calls",
    ):
        if _receipt_exact_int(receipt.get(field), field=field) != expected:
            raise ValueError(f"clean-zero {field} is not the exact two-lane count")
    if _receipt_exact_int(
        receipt.get("search_release_calls"), field="search_release_calls"
    ) < expected:
        raise ValueError("clean-zero receipt released fewer searches than its lanes")
    if _receipt_exact_int(
        receipt.get("outstanding_virtual_loss"), field="outstanding_virtual_loss"
    ) != 0:
        raise ValueError("clean-zero receipt retains virtual loss")

    depths = receipt.get("per_lane_depth")
    if not isinstance(depths, (list, tuple)) or len(depths) != expected:
        raise ValueError("clean-zero receipt lacks both lane depths")
    if any(_receipt_exact_int(value, field="per_lane_depth") < 0 for value in depths):
        raise ValueError("clean-zero receipt has a negative lane depth")
    handles = receipt.get("per_lane_handle_identities")
    if not isinstance(handles, (list, tuple)) or len(handles) != expected:
        raise ValueError("clean-zero receipt lacks both raw handles")
    normalized_handles = [
        _receipt_handle_identity(value, field="per_lane_handle_identities")
        for value in handles
    ]
    if len(set(normalized_handles)) != expected:
        raise ValueError("clean-zero receipt has duplicate raw handles")
    chains = receipt.get("per_lane_search_id_chains")
    if not isinstance(chains, (list, tuple)) or len(chains) != expected:
        raise ValueError("clean-zero receipt lacks both SearchBegin chains")
    first_search_ids: list[int] = []
    for chain in chains:
        if not isinstance(chain, (list, tuple)) or not chain:
            raise ValueError("clean-zero receipt lacks a lane SearchBegin id")
        first_search_ids.append(
            _receipt_exact_int(chain[0], field="per_lane_search_id_chains")
        )
    if len(set(zip(normalized_handles, first_search_ids))) != expected:
        raise ValueError("clean-zero receipt lacks distinct handle/SearchBegin composites")
    if _receipt_exact_int(
        receipt.get("distinct_search_begin_composite_count"),
        field="distinct_search_begin_composite_count",
    ) != expected:
        raise ValueError("clean-zero receipt composite SearchBegin count is invalid")
    _validate_handle_scoped_first_search_id_composites(
        receipt,
        handles=normalized_handles,
        first_search_ids=first_search_ids,
    )
    cleanup = receipt.get("exact_child_cleanup_and_reap")
    if not isinstance(cleanup, Mapping):
        raise ValueError("clean-zero receipt lacks parent exact-child reap evidence")
    reap = cleanup.get("reap")
    if not isinstance(reap, Mapping) or reap.get("reaped") is not True:
        raise ValueError("clean-zero receipt lacks a reaped exact child")


def _clear_principal_variation() -> None:
    """Discard a plan as soon as its deterministic proof no longer applies."""

    global _GAME_PRINCIPAL_VARIATION, _GAME_PRINCIPAL_VARIATION_ROOT_SEAT
    _GAME_PRINCIPAL_VARIATION = []
    _GAME_PRINCIPAL_VARIATION_ROOT_SEAT = None


def _current_actor_seat(obs_dict: Mapping[str, Any]) -> int | None:
    current = obs_dict.get("current")
    if not isinstance(current, Mapping):
        return None
    value = current.get("yourIndex")
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        return None
    return int(value)


def _canonical_observation_fingerprint(obs_dict: Mapping[str, Any]) -> str | None:
    """Use the child runtime's exact, lane-independent public fingerprint."""

    try:
        from poke_bot.r228_kaggle_async_runtime import canonical_observation_fingerprint

        fingerprint = canonical_observation_fingerprint(obs_dict)
    except Exception:
        return None
    return fingerprint if isinstance(fingerprint, str) and fingerprint else None


def _replace_principal_variation(
    receipt: Mapping[str, Any], *, root_seat: int | None
) -> None:
    """Store only a bounded child-proved deterministic continuation plan.

    A malformed optional plan never gains action authority.  It simply leaves
    no plan to consume, so the next prompt re-enters ordinary direct/MCTS
    scheduling instead of treating a partial receipt as a child fault.
    """

    _clear_principal_variation()
    raw_entries = receipt.get("principal_variation")
    if raw_entries is None or raw_entries == []:
        return
    if root_seat not in (0, 1) or not isinstance(raw_entries, list):
        return
    if not 1 <= len(raw_entries) <= R240_MAX_PRINCIPAL_VARIATION_DEPTH:
        return
    try:
        receipt_root_seat = _receipt_exact_int(
            receipt.get("root_seat"), field="root_seat"
        )
    except ValueError:
        return
    if receipt_root_seat != root_seat:
        return

    normalized: list[dict[str, Any]] = []
    for entry in raw_entries:
        if not isinstance(entry, Mapping) or set(entry) != {
            "observation_fingerprint",
            "action",
        }:
            return
        fingerprint = entry.get("observation_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            return
        try:
            action = _action(entry.get("action"), label="principal-variation action")
        except Exception:
            return
        normalized.append(
            {
                "observation_fingerprint": fingerprint,
                "action": action,
            }
        )

    global _GAME_PRINCIPAL_VARIATION, _GAME_PRINCIPAL_VARIATION_ROOT_SEAT
    _GAME_PRINCIPAL_VARIATION = normalized
    _GAME_PRINCIPAL_VARIATION_ROOT_SEAT = receipt_root_seat


def _consume_principal_variation(
    obs_dict: Mapping[str, Any], legal: Sequence[Sequence[int]]
) -> tuple[list[int], str, int, int] | None:
    """Consume one exact continuation entry, or invalidate the entire plan."""

    global _GAME_PRINCIPAL_VARIATION_ROOT_SEAT
    if not _GAME_PRINCIPAL_VARIATION:
        return None
    if _current_actor_seat(obs_dict) != _GAME_PRINCIPAL_VARIATION_ROOT_SEAT:
        _clear_principal_variation()
        return None
    current_fingerprint = _canonical_observation_fingerprint(obs_dict)
    if current_fingerprint is None:
        _clear_principal_variation()
        return None
    entry = _GAME_PRINCIPAL_VARIATION[0]
    root_seat = _GAME_PRINCIPAL_VARIATION_ROOT_SEAT
    depth_before_consume = len(_GAME_PRINCIPAL_VARIATION)
    planned_action = list(entry["action"])
    if (
        entry["observation_fingerprint"] != current_fingerprint
        or planned_action not in legal
    ):
        _clear_principal_variation()
        return None
    del _GAME_PRINCIPAL_VARIATION[0]
    if not _GAME_PRINCIPAL_VARIATION:
        # The root-seat evidence has no meaning after the final plan entry.
        _GAME_PRINCIPAL_VARIATION_ROOT_SEAT = None
    assert root_seat in (0, 1)
    return planned_action, current_fingerprint, root_seat, depth_before_consume


def _is_high_confidence_direct(telemetry: Mapping[str, Any] | None) -> bool:
    if not isinstance(telemetry, Mapping):
        return False
    values = telemetry.get("factorized_selected_stage_probabilities")
    if not isinstance(values, list) or not values:
        return False
    try:
        return all(
            math.isfinite(float(value))
            and float(value) >= R238_HIGH_CONFIDENCE_DIRECT_THRESHOLD
            for value in values
        )
    except (TypeError, ValueError):
        return False


def _parent_action_elapsed_seconds(started_monotonic: float) -> float:
    """Read the callback elapsed time from the live monotonic clock."""

    if isinstance(started_monotonic, bool) or not isinstance(
        started_monotonic, (int, float)
    ):
        raise ValueError("parent action start time is invalid")
    elapsed = time.monotonic() - float(started_monotonic)
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise ValueError("parent action elapsed time is invalid")
    return elapsed


def _emit_parent_direct_or_continuation_decision(
    *,
    mode: str,
    selected_action: Sequence[int],
    direct_action: Sequence[int],
    legal: Sequence[Sequence[int]],
    telemetry: Mapping[str, Any] | None,
    planned_fingerprint: str | None = None,
    continuation_root_seat: int | None = None,
    continuation_depth_before_consume: int | None = None,
    history_only_existing_child_journal_count: int = 0,
    parent_action_started_monotonic: float,
) -> None:
    """Emit one parent-owned non-new-search decision receipt.

    Both valid r240 shortcuts are intentionally explicit.  A high-confidence
    response is direct authority; a continuation remains prior two-lane
    receipt authority, but starts neither a child nor a new search call.
    """

    continuation = mode == "deterministic_mcts_continuation"
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "mode": mode,
        "selected_action": list(selected_action),
        "direct_action": list(direct_action),
        "frozen_r195_direct_action": list(direct_action),
        "legal_action_count": len(legal),
        "legal_order_fingerprint": _legal_order_fingerprint(legal),
        "complete_action_cap": COMPLETE_ACTION_CAP,
        "configured_simulator_lane_count": SIMULATOR_SEARCH_LANE_COUNT,
        # This is authored here only after `agent` has precomputed and
        # checked the exact direct action against the complete legal order.
        "direct_action_precomputed_and_validated": True,
        "parent_action_elapsed_seconds": _parent_action_elapsed_seconds(
            parent_action_started_monotonic
        ),
        "mcts_action_authority": False,
        "confidence_classification": mode,
        "mcts_child_started": False,
        "mcts_child_call_count": 0,
        "mcts_child_started_for_this_decision": False,
        "mcts_select_call_count": 0,
        "history_only_existing_child_journal_count": int(
            history_only_existing_child_journal_count
        ),
        "degraded": False,
        "action_authority": (
            "prior_two_lane_broker_receipt.principal_variation_backed_action"
            if continuation
            else "precomputed_frozen_r195_direct_action"
        ),
        "parent_thread_caps": _parent_thread_caps_receipt(),
        "parent_stock_library": _parent_stock_library_receipt(),
        "parent_cuda_runtime_before_search": _parent_cuda_runtime_before_search_receipt(),
        "broker": _broker_state(_BROKER),
    }
    if telemetry is not None:
        payload.update(
            {
                "high_confidence_threshold": R238_HIGH_CONFIDENCE_DIRECT_THRESHOLD,
                "selected_factorized_stage_probability_threshold": (
                    R238_HIGH_CONFIDENCE_DIRECT_THRESHOLD
                ),
                "factorized_selected_stage_probabilities": list(
                    telemetry["factorized_selected_stage_probabilities"]
                ),
                # r242's journal schema names the selected stages first.
                # Retain the older factorized-first spelling for downstream
                # replay readers, but make the canonical field identical.
                "selected_factorized_stage_probabilities": list(
                    telemetry["factorized_selected_stage_probabilities"]
                ),
                "minimum_selected_factorized_stage_probability": telemetry[
                    "minimum_selected_factorized_stage_probability"
                ],
                "factorized_stage_count": telemetry["factorized_stage_count"],
                "all_selected_factorized_stages_meet_threshold": _is_high_confidence_direct(
                    telemetry
                ),
            }
        )
    if continuation:
        payload.update(
            {
                "planned_action": list(selected_action),
                "direct_action": list(direct_action),
                "planned_observation_fingerprint": planned_fingerprint,
                "continuation_plan_observation_fingerprint": planned_fingerprint,
                # Normalized r242 journal keys are the staged contract.  Keep
                # the earlier ``continuation_plan_*`` aliases above only for
                # replay-reader compatibility; both describe this one
                # parent-validated continuation step.
                "continuation_observation_fingerprint": planned_fingerprint,
                "planned_action_differs_from_direct": list(selected_action)
                != list(direct_action),
                "planned_action_differs_from_direct_action": list(selected_action)
                != list(direct_action),
                "planned_vs_direct_action_changed": list(selected_action)
                != list(direct_action),
                "continuation_plan_depth_limit": R240_MAX_PRINCIPAL_VARIATION_DEPTH,
                "continuation_plan_depth_before_consume": (
                    continuation_depth_before_consume
                ),
                "continuation_plan_root_seat": continuation_root_seat,
                "continuation_plan_actor_seat": continuation_root_seat,
                "continuation_actor_seat": continuation_root_seat,
                "continuation_plan_depth_remaining": len(
                    _GAME_PRINCIPAL_VARIATION
                ),
                "continuation_plan_exact_fingerprint_matched": True,
                "continuation_plan_same_actor_as_root": True,
                "continuation_plan_action_in_complete_legal_order": True,
                "continuation_plan_both_lanes_same_fingerprint": True,
                "continuation_plan_both_lanes_backed_leader_agreement": True,
                "continuation_both_lanes_same_fingerprint": True,
                "continuation_backed_leader_agreement": True,
                "continuation_plan_two_lane_backed_action": True,
                "continuation_plan_no_chance_boundary_or_opponent_transition": True,
                "history_rewritten_to_actual_action": True,
                "journal_count": len(_GAME_DIRECT_JOURNAL),
                "remaining_principal_variation_depth": len(
                    _GAME_PRINCIPAL_VARIATION
                ),
            }
        )
    _json_marker(DECISION_PREFIX, payload)


def _emit_decision(
    *,
    broker: IsolatedR228SearchBroker | None,
    receipt: Mapping[str, Any],
    selected_action: Sequence[int],
    direct_action: Sequence[int],
    legal: Sequence[Sequence[int]],
    direct_telemetry: Mapping[str, Any],
    parent_action_started_monotonic: float,
) -> None:
    payload = dict(receipt)
    terminal_win_authority = (
        receipt.get("stop_reason") == PROVEN_TERMINAL_WIN_STOP_REASON
    )
    depths = receipt.get("per_lane_depth")
    both_lanes_progressed = isinstance(depths, (list, tuple)) and len(depths) == 2 and all(
        isinstance(depth, int) and not isinstance(depth, bool) and depth >= 1
        for depth in depths
    )
    payload.update(
        {
            "selected_action": list(selected_action),
            "direct_action": list(direct_action),
            "frozen_r195_direct_action": list(direct_action),
            "legal_action_count": len(legal),
            "legal_order_fingerprint": _legal_order_fingerprint(legal),
            "complete_action_cap": COMPLETE_ACTION_CAP,
            "configured_simulator_lane_count": SIMULATOR_SEARCH_LANE_COUNT,
            # These are parent facts established before and after the broker
            # request, respectively.  None are reconstructed by a probe.
            "direct_action_precomputed_and_validated": True,
            "parent_action_elapsed_seconds": _parent_action_elapsed_seconds(
                parent_action_started_monotonic
            ),
            "parent_action_deadline_seconds": R234_BROKER_ACTION_TIMEOUT_SECONDS,
            "action_authority": R228_ASYNC_SELECTED_ACTION_AUTHORITY,
            "terminal_win_action_authority": terminal_win_authority,
            "owner_proven_deterministic_terminal_win_this_turn_revision": (
                PROVEN_TERMINAL_WIN_REVISION
            ),
            "confidence_classification": "ambiguous_mcts",
            "selected_factorized_stage_probability_threshold": (
                R238_HIGH_CONFIDENCE_DIRECT_THRESHOLD
            ),
            "selected_factorized_stage_probabilities": list(
                direct_telemetry["factorized_selected_stage_probabilities"]
            ),
            "all_selected_factorized_stages_meet_threshold": False,
            "child_search_hard_seconds": R234_BROKER_SEARCH_SECONDS,
            "parent_action_hard_seconds": R234_BROKER_ACTION_TIMEOUT_SECONDS,
            "deterministic_root_leader_observations": receipt.get(
                "deterministic_root_leader_observations",
                receipt.get("observed_stable_root_leader_observations"),
            ),
            "both_lanes_progressed": both_lanes_progressed,
            "adaptive_early_stop_qualified": receipt.get("stop_reason")
            == "stable_root_leader",
            "hard_completed_backup_stop": receipt.get("stop_reason")
            == "maximum_backups",
            "zero_backup_precomputed_direct_fallback": False,
            "parent_validated_current_root_observation_legal_fingerprint_and_actor": (
                terminal_win_authority
            ),
            "degraded": False,
            "parent_thread_caps": _parent_thread_caps_receipt(),
            "parent_stock_library": _parent_stock_library_receipt(),
            "parent_cuda_runtime_before_search": _parent_cuda_runtime_before_search_receipt(),
            "broker": _broker_state(broker),
        }
    )
    _json_marker(DECISION_PREFIX, payload)


def _emit_clean_zero_backup_fallback(
    *,
    broker: IsolatedR228SearchBroker | None,
    receipt: Mapping[str, Any],
    direct_action: Sequence[int],
    legal: Sequence[Sequence[int]],
    direct_telemetry: Mapping[str, Any],
    parent_action_started_monotonic: float,
) -> None:
    """Emit the clean-deadline parent-direct result without MCTS authority."""

    payload = dict(receipt)
    payload.update(
        {
            "mode": "zero_backup_precomputed_direct_fallback",
            "selected_action": list(direct_action),
            "direct_action": list(direct_action),
            "frozen_r195_direct_action": list(direct_action),
            "legal_action_count": len(legal),
            "legal_order_fingerprint": _legal_order_fingerprint(legal),
            "complete_action_cap": COMPLETE_ACTION_CAP,
            "configured_simulator_lane_count": SIMULATOR_SEARCH_LANE_COUNT,
            "direct_action_precomputed_and_validated": True,
            "parent_action_elapsed_seconds": _parent_action_elapsed_seconds(
                parent_action_started_monotonic
            ),
            "parent_action_deadline_seconds": R234_BROKER_ACTION_TIMEOUT_SECONDS,
            "action_authority": "precomputed_frozen_r195_direct_action",
            "mcts_action_authority": False,
            "confidence_classification": "ambiguous_mcts",
            "selected_factorized_stage_probability_threshold": (
                R238_HIGH_CONFIDENCE_DIRECT_THRESHOLD
            ),
            "selected_factorized_stage_probabilities": list(
                direct_telemetry["factorized_selected_stage_probabilities"]
            ),
            "factorized_selected_stage_probabilities": list(
                direct_telemetry["factorized_selected_stage_probabilities"]
            ),
            "all_selected_factorized_stages_meet_threshold": False,
            "mcts_child_started": True,
            "mcts_child_call_count": 1,
            "mcts_child_started_for_this_decision": True,
            "mcts_select_call_count": 1,
            "child_search_hard_seconds": R234_BROKER_SEARCH_SECONDS,
            "parent_action_hard_seconds": R234_BROKER_ACTION_TIMEOUT_SECONDS,
            "both_lanes_progressed": False,
            "adaptive_early_stop_qualified": False,
            "hard_completed_backup_stop": False,
            "zero_backup_precomputed_direct_fallback": True,
            "degraded": False,
            "parent_thread_caps": _parent_thread_caps_receipt(),
            "parent_stock_library": _parent_stock_library_receipt(),
            "parent_cuda_runtime_before_search": _parent_cuda_runtime_before_search_receipt(),
            "broker": _broker_state(broker),
        }
    )
    _json_marker(DECISION_PREFIX, payload)


def _finish_prior_game() -> None:
    if _GAME_SUCCESSFUL_MCTS_DECISIONS < 1 or _GAME_DEGRADED_FAULTS != 0:
        return
    _json_marker(
        FULL_GAMEPLAY_SUCCESS_PREFIX,
        {
            "schema": SCHEMA,
            "mcts_branching_decisions": _GAME_SUCCESSFUL_MCTS_DECISIONS,
            "degraded_fault_count": _GAME_DEGRADED_FAULTS,
            "complete_action_cap": COMPLETE_ACTION_CAP,
            "configured_simulator_lane_count": SIMULATOR_SEARCH_LANE_COUNT,
            "parent_thread_caps": _parent_thread_caps_receipt(),
            "parent_stock_library": _parent_stock_library_receipt(),
            "parent_cuda_runtime_before_search": _parent_cuda_runtime_before_search_receipt(),
        },
    )


def _contained_fallback_and_journal(
    *,
    broker: IsolatedR228SearchBroker,
    fault: Mapping[str, Any],
    obs_dict: Mapping[str, Any],
    direct_action: Sequence[int],
    legal: Sequence[Sequence[int]],
) -> list[int]:
    """Contain a child fault and record the one parent-authoritative result."""

    selected = _contained_broker_fallback(
        broker=broker,
        fault=fault,
        direct_action=direct_action,
        legal=legal,
    )
    # ``broker.select`` owns its own replay journal.  This separate parent
    # journal is only for a later deferred-child replay, so append exactly one
    # real fallback row without issuing a second broker ``note`` request.
    _append_parent_direct_journal(obs_dict, selected)
    return selected


def _emit_disabled_broker_fallback_if_needed(
    *,
    direct_action: Sequence[int],
    legal: Sequence[Sequence[int]],
) -> bool:
    """Return whether a just-journaled action belongs to a degraded game."""

    broker = _BROKER
    if not _GAME_MCTS_DISABLED and (broker is None or not broker.disabled):
        return False
    if broker is None:
        _hard_failure(code="broker_identity_invalid")
    fault = _GAME_PENDING_FAULT or broker.last_fault
    if isinstance(fault, Mapping):
        _emit_degraded_fallback(
            broker=broker,
            fault=fault,
            direct_action=direct_action,
            legal=legal,
        )
        return True
    synthesized = _synthesized_fault(
        broker,
        code="broker_disabled_without_fault",
        message="broker was disabled before a parent-authoritative action",
    )
    _contained_broker_fallback(
        broker=broker,
        fault=synthesized,
        direct_action=direct_action,
        legal=legal,
    )
    return True


def agent(obs_dict: dict) -> list[int]:
    """Return a parent-authoritative r240 direct, plan, or two-lane action."""

    global _GAME_SUCCESSFUL_MCTS_DECISIONS
    if not isinstance(obs_dict, Mapping):
        _hard_failure(code="selection_identity_invalid")
    parent_action_started_monotonic = time.monotonic()
    _apply_phase1_parent_thread_caps()
    # This must complete before the frozen r195 entrypoint can import/load a
    # native wrapper.  It is file identity only and cannot open a CG arena.
    _validate_parent_staged_stock_library_identity()
    try:
        direct = _direct()
    except Exception as exc:
        _hard_failure(code="direct_entrypoint_identity_invalid", exc=exc)

    selection = obs_dict.get("select")
    if selection is None:
        # A deck request is the only reliable physical-game boundary exposed
        # by Kaggle.  It clears a possible deterministic plan before the next
        # game starts and emits success only for a nondegraded searched game.
        _finish_prior_game()
        _clear_principal_variation()
        try:
            deck = _action(direct.agent(dict(obs_dict)), label="frozen r195 deck response")
        except Exception as exc:
            _hard_failure(code="direct_action_invalid", exc=exc)
        _begin_game()
        return deck

    # The frozen action and target receipt come before complete-root
    # enumeration, broker construction, or any native work.  Turn-order setup
    # remains an r195 short circuit inside this helper.
    try:
        direct_action, direct_telemetry, forced_setup = _precompute_validated_direct_action(
            direct, obs_dict
        )
    except _ValidatedDirectPolicyReceiptError as exc:
        _hard_failure(code="validated_direct_policy_receipt_missing", exc=exc)
    except Exception as exc:
        _hard_failure(code="direct_action_invalid", exc=exc)

    legal = _legal_order(obs_dict)
    if direct_action not in legal:
        _hard_failure(
            code="direct_action_outside_complete_legal_order",
            direct_action=direct_action,
            legal_action_count=len(legal),
            legal_order_fingerprint=_legal_order_fingerprint(legal),
        )

    if forced_setup:
        _clear_principal_variation()
        _journal_real_action(obs_dict, direct_action, legal)
        _emit_disabled_broker_fallback_if_needed(
            direct_action=direct_action, legal=legal
        )
        return direct_action

    if len(legal) <= 1:
        # A one-choice prompt cannot prove a continuation/search branch.  It
        # is still journaled once so a later child sees exact history.
        _clear_principal_variation()
        _journal_real_action(obs_dict, direct_action, legal)
        _emit_disabled_broker_fallback_if_needed(
            direct_action=direct_action, legal=legal
        )
        return direct_action

    continuation = _consume_principal_variation(obs_dict, legal)
    if continuation is not None:
        (
            selected_action,
            planned_fingerprint,
            continuation_root_seat,
            continuation_depth_before_consume,
        ) = continuation
        if selected_action != direct_action:
            _overwrite_parent_action_token(direct, obs_dict, selected_action)
        history_only_existing_child_journal_count = _journal_real_action(
            obs_dict, selected_action, legal
        )
        if not _emit_disabled_broker_fallback_if_needed(
            direct_action=direct_action, legal=legal
        ):
            _emit_parent_direct_or_continuation_decision(
                mode="deterministic_mcts_continuation",
                selected_action=selected_action,
                direct_action=direct_action,
                legal=legal,
                telemetry=direct_telemetry,
                planned_fingerprint=planned_fingerprint,
                continuation_root_seat=continuation_root_seat,
                continuation_depth_before_consume=continuation_depth_before_consume,
                history_only_existing_child_journal_count=(
                    history_only_existing_child_journal_count
                ),
                parent_action_started_monotonic=parent_action_started_monotonic,
            )
        return selected_action

    if _is_high_confidence_direct(direct_telemetry):
        history_only_existing_child_journal_count = _journal_real_action(
            obs_dict, direct_action, legal
        )
        if not _emit_disabled_broker_fallback_if_needed(
            direct_action=direct_action, legal=legal
        ):
            _emit_parent_direct_or_continuation_decision(
                mode="high_confidence_frozen_direct",
                selected_action=direct_action,
                direct_action=direct_action,
                legal=legal,
                telemetry=direct_telemetry,
                history_only_existing_child_journal_count=(
                    history_only_existing_child_journal_count
                ),
                parent_action_started_monotonic=parent_action_started_monotonic,
            )
        return direct_action

    broker = _ensure_broker_for_selection(legal)
    if _GAME_MCTS_DISABLED or broker.disabled:
        _journal_real_action(obs_dict, direct_action, legal)
        _emit_disabled_broker_fallback_if_needed(
            direct_action=direct_action, legal=legal
        )
        return direct_action

    try:
        result = broker.select(obs_dict, direct_action)
    except R228BrokerError as exc:
        if exc.code == "child_unreaped":
            _hard_failure(code="broker_child_not_reaped", exc=exc)
        if exc.code == "illegal_direct_action":
            _hard_failure(code="broker_direct_action_illegal", exc=exc)
        fault = _synthesized_fault(
            broker, code=exc.code, message=f"broker select failed: {exc}"
        )
        return _contained_fallback_and_journal(
            broker=broker,
            fault=fault,
            obs_dict=obs_dict,
            direct_action=direct_action,
            legal=legal,
        )
    except Exception as exc:
        fault = _synthesized_fault(
            broker,
            code="broker_select_exception",
            message=f"broker select raised {type(exc).__name__}: {exc}",
        )
        return _contained_fallback_and_journal(
            broker=broker,
            fault=fault,
            obs_dict=obs_dict,
            direct_action=direct_action,
            legal=legal,
        )

    if not isinstance(result, tuple) or len(result) != 3:
        fault = _synthesized_fault(
            broker,
            code="broker_result_protocol_invalid",
            message="broker select did not return (selected, receipt, fault)",
        )
        return _contained_fallback_and_journal(
            broker=broker,
            fault=fault,
            obs_dict=obs_dict,
            direct_action=direct_action,
            legal=legal,
        )
    selected, receipt, fault = result
    if fault is not None:
        if not isinstance(fault, Mapping):
            fault = _synthesized_fault(
                broker,
                code="broker_fault_protocol_invalid",
                message="broker fault receipt was not a mapping",
            )
        return _contained_fallback_and_journal(
            broker=broker,
            fault=fault,
            obs_dict=obs_dict,
            direct_action=direct_action,
            legal=legal,
        )

    try:
        selected_action = _action(selected, label="broker selected action")
    except Exception as exc:
        fault = _synthesized_fault(
            broker,
            code="broker_selected_action_protocol_invalid",
            message=f"broker selected action was malformed: {exc}",
        )
        return _contained_fallback_and_journal(
            broker=broker,
            fault=fault,
            obs_dict=obs_dict,
            direct_action=direct_action,
            legal=legal,
        )
    if selected_action not in legal:
        fault = _synthesized_fault(
            broker,
            code="broker_action_outside_complete_legal_order",
            message="broker returned an action outside the complete legal order",
        )
        return _contained_fallback_and_journal(
            broker=broker,
            fault=fault,
            obs_dict=obs_dict,
            direct_action=direct_action,
            legal=legal,
        )
    if not isinstance(receipt, Mapping):
        fault = _synthesized_fault(
            broker,
            code="broker_receipt_protocol_invalid",
            message="broker omitted a mapping decision receipt",
        )
        return _contained_fallback_and_journal(
            broker=broker,
            fault=fault,
            obs_dict=obs_dict,
            direct_action=direct_action,
            legal=legal,
        )
    if receipt.get("mode") == "zero_backup_precomputed_direct_fallback":
        try:
            receipt_action = _action(
                receipt.get("selected_action"), label="clean-zero receipt action"
            )
            _validate_phase1_clean_zero_backup_receipt(
                receipt, direct_action=direct_action
            )
        except Exception as exc:
            fault = _synthesized_fault(
                broker,
                code="broker_clean_zero_receipt_protocol_invalid",
                message=(
                    "broker clean-zero receipt was malformed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
            return _contained_fallback_and_journal(
                broker=broker,
                fault=fault,
                obs_dict=obs_dict,
                direct_action=direct_action,
                legal=legal,
            )
        if receipt_action != direct_action or selected_action != direct_action:
            fault = _synthesized_fault(
                broker,
                code="broker_clean_zero_parent_action_mismatch",
                message="clean-zero result differed from the precomputed parent direct action",
            )
            return _contained_fallback_and_journal(
                broker=broker,
                fault=fault,
                obs_dict=obs_dict,
                direct_action=direct_action,
                legal=legal,
            )
        # The broker has already journaled its exact parent-direct fallback;
        # the parent retains a separate deferred-child journal for any later
        # ambiguity in the same physical game.  This is not a searched action
        # and must not earn full-game MCTS-success credit.
        _append_parent_direct_journal(obs_dict, direct_action)
        _clear_principal_variation()
        _emit_clean_zero_backup_fallback(
            broker=broker,
            receipt=receipt,
            direct_action=direct_action,
            legal=legal,
            direct_telemetry=direct_telemetry,
            parent_action_started_monotonic=parent_action_started_monotonic,
        )
        return direct_action
    try:
        receipt_action = _action(
            receipt.get("selected_action"), label="broker receipt selected action"
        )
        _validate_phase1_two_lane_receipt(
            receipt,
            obs_dict=obs_dict,
            legal=legal,
            selected_action=selected_action,
        )
    except Exception as exc:
        fault = _synthesized_fault(
            broker,
            code="broker_receipt_protocol_invalid",
            message=f"broker receipt was malformed: {type(exc).__name__}: {exc}",
        )
        return _contained_fallback_and_journal(
            broker=broker,
            fault=fault,
            obs_dict=obs_dict,
            direct_action=direct_action,
            legal=legal,
        )
    if receipt_action != selected_action or receipt.get("mcts_action_authority") is not True:
        fault = _synthesized_fault(
            broker,
            code="broker_receipt_authority_mismatch",
            message="broker receipt did not authorize its returned MCTS action",
        )
        return _contained_fallback_and_journal(
            broker=broker,
            fault=fault,
            obs_dict=obs_dict,
            direct_action=direct_action,
            legal=legal,
        )

    if selected_action != direct_action:
        _overwrite_parent_action_token(direct, obs_dict, selected_action)
    _append_parent_direct_journal(obs_dict, selected_action)
    _replace_principal_variation(
        receipt, root_seat=_current_actor_seat(obs_dict)
    )
    _GAME_SUCCESSFUL_MCTS_DECISIONS += 1
    _emit_decision(
        broker=broker,
        receipt=receipt,
        selected_action=selected_action,
        direct_action=direct_action,
        legal=legal,
        direct_telemetry=direct_telemetry,
        parent_action_started_monotonic=parent_action_started_monotonic,
    )
    return selected_action
