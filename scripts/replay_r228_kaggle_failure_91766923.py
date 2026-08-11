#!/usr/bin/env python3
"""Replay the exact failed 91766923 seat-0 step-58 callback, read-only.

The harness restores the saved seat-0 ACTIVE history into the sealed package,
then asks its real parent entrypoint for the final two-choice action.  Setup is
unbounded only by a separate, explicit setup deadline; once the child reports
that the saved history is ready, the real package callback has the r242 four
second deadline.  Timeout recovery addresses only that exact Popen child.

This is neither a Kaggle client nor a package builder.  It never writes a stage
member and its receipt records before/after whole-tree snapshots.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, TextIO

from r228_kaggle_r244_harness_common import (
    COMPLETE_ACTION_CAP,
    DECISION_PREFIX,
    DEGRADED_FALLBACK_PREFIX,
    HARD_FAILURE_PREFIX,
    HarnessContractError,
    SAVED_EPISODE_RECEIPT_NAME,
    as_action,
    collect_markers,
    json_digest,
    load_binding_identity,
    load_stage_contract,
    passed_preflight_receipt,
    prepare_exact_stage_import,
    require,
    require_module_from_exact_stage,
    sha256_file,
    stage_snapshot,
    validate_decision_marker,
    validate_degraded_marker,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLAY = ROOT / "evidence/kaggle/55416396/episode-91766923-replay.json"
TARGET_STEP = 58
SCHEMA = "poke_bot.r244_kaggle_episode_91766923_step58_replay/v1"
SCOPE = "exact_r236_r238_r242_r244_package_episode_91766923_seat0_step58"
FINAL_PREFIX = "R244_KAGGLE_REPLAY_91766923 "
CHILD_RESULT_PREFIX = "R244_KAGGLE_REPLAY_CHILD_RESULT "


class ReplayError(RuntimeError):
    """The saved-episode regression did not meet its safety contract."""


class _Tee:
    def __init__(self, *targets: TextIO) -> None:
        self._targets = targets
        self._parts: list[str] = []

    def write(self, value: str) -> int:
        self._parts.append(value)
        for target in self._targets:
            target.write(value)
        return len(value)

    def flush(self) -> None:
        for target in self._targets:
            target.flush()

    @property
    def text(self) -> str:
        return "".join(self._parts)


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ReplayError("replay data is not JSON-native") from exc


def _positive_seconds(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite positive number") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def _seat_row(steps: list[Any], index: int) -> dict[str, Any]:
    require(index < len(steps), "saved replay row is absent")
    row = steps[index]
    require(isinstance(row, list) and bool(row), "saved replay seat-0 row is malformed")
    seat = row[0]
    require(isinstance(seat, dict), "saved replay seat-0 entry is malformed")
    return seat


def _load_replay_target(replay: Path, target_step: int) -> tuple[dict[str, Any], list[Any]]:
    try:
        payload = json.loads(replay.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayError("saved Kaggle replay is unreadable") from exc
    require(isinstance(payload, dict), "saved replay is not an object")
    steps = payload.get("steps")
    require(isinstance(steps, list) and target_step + 1 < len(steps), "saved replay lacks step 58")
    target_row = _seat_row(steps, target_step)
    require(target_row.get("status") == "ACTIVE", "step-58 seat 0 is not ACTIVE")
    observation = target_row.get("observation")
    require(isinstance(observation, dict), "step-58 observation is absent")
    require(observation.get("step") == target_step, "step-58 observation identity drift")
    current = observation.get("current")
    require(
        isinstance(current, dict) and current.get("yourIndex") == 0,
        "saved target is not seat 0",
    )
    selection = observation.get("select")
    require(isinstance(selection, dict), "saved target has no active selection")
    options = selection.get("option")
    require(isinstance(options, list) and len(options) == 2, "saved target is not the exact two-choice prompt")
    require(
        isinstance(observation.get("search_begin_input"), str)
        and bool(observation["search_begin_input"]),
        "saved target lacks native-search input",
    )
    return _json_copy(observation), steps


def _prior_active_events(steps: list[Any], target_step: int) -> list[dict[str, Any]]:
    """Return every prior seat-0 ACTIVE prompt plus its actual next action."""

    events: list[dict[str, Any]] = []
    for step in range(target_step):
        row = _seat_row(steps, step)
        observation = row.get("observation")
        if row.get("status") != "ACTIVE" or not isinstance(observation, dict):
            continue
        if not isinstance(observation.get("select"), dict):
            continue
        next_action = as_action(
            _seat_row(steps, step + 1).get("action"),
            field=f"saved action after step {step}",
        )
        events.append(
            {
                "replay_step_index": step,
                "observation": _json_copy(observation),
                "action": next_action,
            }
        )
    require(bool(events), "saved replay has no prior seat-0 ACTIVE history")
    return events


def _complete_legal_actions(features: Any, observation: Mapping[str, Any]) -> list[list[int]]:
    try:
        raw = features.enumerate_action_combos(
            dict(observation), max_combos=COMPLETE_ACTION_CAP
        )
    except Exception as exc:
        raise ReplayError(
            "complete legal enumeration failed under "
            f"complete_action_cap={COMPLETE_ACTION_CAP}: {type(exc).__name__}: {exc}"
        ) from exc
    legal = [as_action(row, field="complete legal action") for row in raw]
    require(bool(legal), "saved ACTIVE prompt has no complete legal action")
    return legal


def _load_stage(stage: Path) -> tuple[Any, Any, Any]:
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    stage = prepare_exact_stage_import(stage)
    stage_text = str(stage)
    os.environ["CG_LIB_PATH"] = stage_text
    os.chdir(stage_text)
    main = importlib.import_module("main")
    require_module_from_exact_stage(main, module_name="main", stage=stage)
    poke_bot = importlib.import_module("poke_bot")
    require_module_from_exact_stage(poke_bot, module_name="poke_bot", stage=stage)
    direct = main._direct()
    from poke_bot import features

    require_module_from_exact_stage(features, module_name="poke_bot.features", stage=stage)
    return main, direct, features


def _hydrate_exact_parent_history(
    *, main: Any, direct: Any, features: Any, events: list[dict[str, Any]]
) -> dict[str, Any]:
    """Mirror the parent history/journal without creating a parallel broker."""

    begin_game = getattr(main, "_begin_game", None)
    append_parent = getattr(main, "_append_parent_direct_journal", None)
    require(callable(begin_game), "staged main lacks its game-boundary reset")
    require(callable(append_parent), "staged main lacks its parent direct journal")
    begin_game()
    _deck, _model, policy = direct._ensure_runtime()
    reset_game = getattr(policy, "reset_game", None)
    require(callable(reset_game), "frozen direct policy lacks reset_game")
    reset_game()

    committed: list[dict[str, Any]] = []
    for event in events:
        observation = _json_copy(event["observation"])
        action = as_action(event["action"], field="saved replay action")
        legal = _complete_legal_actions(features, observation)
        require(
            action in legal,
            f"saved action after step {event['replay_step_index']} is not legal",
        )
        forced = direct._turn_order_choice(observation)
        forced_prompt = forced is not None
        if forced_prompt:
            require(
                action == list(forced),
                f"saved turn-order action after step {event['replay_step_index']} drifted",
            )
        else:
            router = getattr(policy, "_matchup_adapter_shadow_router", None)
            if router is not None and hasattr(router, "observe"):
                router.observe(
                    observation, scope="game_root", depth=len(policy.board_history)
                )
            policy._append_decision_history(observation)
            policy._previous_action_token = features.build_option_tokens(
                observation, [action]
            )
        # This is the package's own deferred journal.  It is deliberately not
        # an IPC call and does not start a child before the actual target
        # callback decides whether this prompt is ambiguous.
        append_parent(observation, action)
        committed.append(
            {
                "replay_step_index": event["replay_step_index"],
                "observation_sha256": json_digest(observation),
                "action": action,
                "legal_action_count": len(legal),
                "turn_order_prompt": forced_prompt,
            }
        )
    parent_journal = getattr(main, "_GAME_DIRECT_JOURNAL", None)
    require(isinstance(parent_journal, list) and len(parent_journal) == len(committed), "package parent journal did not retain exact saved history")
    return {
        "event_count": len(committed),
        "events": committed,
        "events_sha256": json_digest(committed),
        "policy_board_history_length": len(getattr(policy, "board_history", [])),
        "parent_direct_journal_length": len(parent_journal),
        "broker_started_before_target": getattr(main, "_BROKER", None) is not None,
    }


def _close_exact_broker(main: Any) -> dict[str, Any]:
    broker = getattr(main, "_BROKER", None)
    if broker is None:
        return {"present": False, "close_called": False, "child_pid_after_close": None}
    close = getattr(broker, "close", None)
    require(callable(close), "staged package broker lacks close")
    close()
    state = broker.marker_payload() if hasattr(broker, "marker_payload") else {}
    require(
        not isinstance(state, Mapping) or state.get("child_pid") is None,
        "staged package broker retained a child after harness close",
    )
    return {
        "present": True,
        "close_called": True,
        "child_pid_after_close": None if not isinstance(state, Mapping) else state.get("child_pid"),
    }


def _write_ready_once(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(dict(payload), sort_keys=True).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _child_payload(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    """Prepare exact history, wait for parent GO, then time only agent()."""

    started = time.monotonic()
    stage = args.stage.resolve()
    replay = args.replay.resolve()
    main: Any | None = None
    package_before: dict[str, Any] = {}
    package_after: dict[str, Any] = {}
    close_receipt: dict[str, Any] = {"present": False, "close_called": False}
    fault_handle: Any | None = None
    fault_cleanup: list[dict[str, Any]] = []
    tee = _Tee(sys.stdout)
    payload: dict[str, Any]
    code = 1
    try:
        stage_contract = load_stage_contract(stage)
        package_before = dict(stage_contract["stage_snapshot"])
        target, steps = _load_replay_target(replay, int(args.target_step))
        events = _prior_active_events(steps, int(args.target_step))
        main, direct, features = _load_stage(stage)
        history = _hydrate_exact_parent_history(
            main=main, direct=direct, features=features, events=events
        )
        target_legal = _complete_legal_actions(features, target)
        require(
            len(target_legal) == 2,
            "saved step-58 complete legal order is not exactly two actions",
        )
        if args.fault_class is not None:
            # This is a reversible, package-external Popen hook.  The real
            # staged parent still owns the direct-policy answer, socket wait,
            # exact-PID termination/reap, and containment marker.
            from run_r244_exact_staged_fault_harness import (
                install_fault_injected_broker_child,
            )

            fault_handle = install_fault_injected_broker_child(
                stage, str(args.fault_class)
            )
        if args.ready_file is not None:
            _write_ready_once(
                Path(args.ready_file),
                {
                    "schema": SCHEMA,
                    "pid": os.getpid(),
                    "status": "ready_for_exact_step58_callback",
                    "history_event_count": len(events),
                    "target_legal_action_count": len(target_legal),
                    "broker_started_before_target": history["broker_started_before_target"],
                },
            )
            require(sys.stdin.readline().strip() == "GO", "parent did not authorize callback")
        callback_started = time.monotonic()
        with contextlib.redirect_stdout(tee):
            action = as_action(main.agent(_json_copy(target)), field="target action")
        callback_elapsed = time.monotonic() - callback_started
        require(
            callback_elapsed <= float(args.callback_deadline_seconds),
            "saved step-58 callback exceeded the r242 four-second deadline",
        )
        require(action in target_legal, "saved step-58 callback returned an illegal action")
        markers = collect_markers(tee.text)
        require(not markers["hard_failures"], "target callback emitted hard failure")
        require(
            len(markers["decisions"]) + len(markers["degraded_fallbacks"]) == 1,
            "target callback did not emit exactly one action-authority receipt",
        )
        require(len(markers["decisions"]) <= 1, "target callback emitted multiple decision receipts")
        require(len(markers["degraded_fallbacks"]) <= 1, "target callback emitted multiple containment receipts")
        if markers["decisions"]:
            decision = validate_decision_marker(
                markers["decisions"][0],
                legal_actions=target_legal,
                observation=target,
            )
            require(decision["selected_action"] == action, "decision receipt differs from returned action")
            result_path = str(decision["mode"])
            degraded = False
            containment = None
        else:
            containment = validate_degraded_marker(
                markers["degraded_fallbacks"][0], legal_actions=target_legal
            )
            require(containment["selected_action"] == action, "containment receipt differs from returned action")
            result_path = "contained_precomputed_parent_direct_fallback_after_exact_child_reap"
            degraded = True
        fault_reap_proved = False
        if args.fault_class is not None:
            require(
                degraded and containment is not None,
                "fault-injected saved callback did not emit a degraded containment receipt",
            )
            child_reap = containment.get("child_reap")
            require(
                isinstance(child_reap, Mapping) and child_reap.get("reaped") is True,
                "fault-injected saved callback did not prove exact child reap",
            )
            fault_reap_proved = True
        close_receipt = _close_exact_broker(main)
        package_after = stage_snapshot(stage)
        require(package_after == package_before, "saved replay harness mutated sealed package")
        payload = {
            "schema": SCHEMA,
            "status": "pass",
            "scope": SCOPE,
            "stage": str(stage),
            "replay": str(replay),
            "replay_sha256": sha256_file(replay),
            "target_step": int(args.target_step),
            "target_observation_sha256": json_digest(target),
            "target_selection_context": target["select"].get("context"),
            "target_option_count": len(target["select"]["option"]),
            "target_complete_legal_actions": target_legal,
            "target_complete_legal_actions_sha256": json_digest(target_legal),
            "final_action": action,
            "callback_elapsed_seconds": callback_elapsed,
            "callback_deadline_seconds": float(args.callback_deadline_seconds),
            "permitted_result_path": result_path,
            "degraded": degraded,
            "containment": containment,
            "fault_injection": {
                "requested_fault_class": args.fault_class,
                "installed": fault_handle is not None,
                "actual_degraded_containment_marker": degraded,
            },
            "fault_injected_broker_child_reap_proved": fault_reap_proved,
            "history": history,
            "stage_contract": stage_contract,
            "package_mutation_check": {
                "before": package_before,
                "after": package_after,
                "unchanged": True,
            },
            "markers": markers,
            "broker_close": close_receipt,
            "elapsed_seconds": max(0.0, time.monotonic() - started),
        }
        code = 0
    except Exception as exc:  # noqa: BLE001 - parent must receive sealed failure evidence
        try:
            package_after = stage_snapshot(stage)
        except Exception:
            pass
        payload = {
            "schema": SCHEMA,
            "status": "failed_closed",
            "scope": SCOPE,
            "stage": str(stage),
            "replay": str(replay),
            "target_step": int(args.target_step),
            "elapsed_seconds": max(0.0, time.monotonic() - started),
            "failure": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "package_mutation_check": {
                "before": package_before,
                "after": package_after,
                "unchanged": bool(package_before) and package_before == package_after,
            },
            "markers": collect_markers(tee.text),
            "broker_close": close_receipt,
        }
    finally:
        if main is not None and not close_receipt.get("close_called"):
            try:
                close_receipt = _close_exact_broker(main)
                payload["broker_close"] = close_receipt
            except Exception as close_exc:  # noqa: BLE001
                code = 1
                payload["status"] = "failed_closed"
                payload["cleanup_error"] = f"{type(close_exc).__name__}: {close_exc}"
        if fault_handle is not None:
            try:
                fault_handle.restore()
                fault_cleanup = list(fault_handle.cleanup_owned_children())
                require(
                    bool(fault_cleanup)
                    and all(row.get("reaped") is True for row in fault_cleanup),
                    "fault harness retained an owned fake child after replay",
                )
                payload["fault_injection"] = {
                    **dict(payload.get("fault_injection") or {}),
                    "requested_fault_class": args.fault_class,
                    "installed": True,
                    "hook_restored": bool(getattr(fault_handle, "restored", False)),
                    "owned_child_cleanup": fault_cleanup,
                    "all_owned_children_reaped": True,
                }
            except Exception as fault_cleanup_exc:  # noqa: BLE001
                code = 1
                payload["status"] = "failed_closed"
                payload["fault_cleanup_error"] = (
                    f"{type(fault_cleanup_exc).__name__}: {fault_cleanup_exc}"
                )
        payload["stdout"] = {
            "sha256": "sha256:" + hashlib.sha256(tee.text.encode("utf-8")).hexdigest(),
            "bytes": len(tee.text.encode("utf-8")),
            "tail": tee.text[-8192:],
        }
    return code, payload


def _child_main(args: argparse.Namespace) -> int:
    code, payload = _child_payload(args)
    print(CHILD_RESULT_PREFIX + json.dumps(payload, sort_keys=True), flush=True)
    return code


def _stream_summary(value: str) -> dict[str, Any]:
    return {
        "sha256": "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest(),
        "bytes": len(value.encode("utf-8")),
        "tail": value[-8192:],
    }


def _parse_child_result(stdout: str) -> dict[str, Any] | None:
    rows = [
        row
        for row in stdout.splitlines()
        if row.startswith(CHILD_RESULT_PREFIX)
    ]
    if len(rows) != 1:
        return None
    try:
        value = json.loads(rows[0][len(CHILD_RESULT_PREFIX) :])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _terminate_exact_owned_child(
    child: subprocess.Popen[str], *, grace_seconds: float, phase: str
) -> tuple[str, str, str]:
    """Terminate only the direct Popen child; never signal a process group."""

    action = f"terminate_exact_owned_child_{phase}"
    try:
        child.terminate()
    except ProcessLookupError:
        pass
    try:
        stdout, stderr = child.communicate(timeout=grace_seconds)
        return action, stdout, stderr
    except subprocess.TimeoutExpired:
        action = f"kill_exact_owned_child_after_grace_{phase}"
        try:
            child.kill()
        except ProcessLookupError:
            pass
        stdout, stderr = child.communicate()
        return action, stdout, stderr


def _wait_for_ready_file(
    *, child: subprocess.Popen[str], ready_file: Path, timeout_seconds: float, grace_seconds: float
) -> tuple[dict[str, Any] | None, str | None, str, str]:
    """Wait only for child setup; actual callback timing starts after GO."""

    started = time.monotonic()
    while time.monotonic() - started <= timeout_seconds:
        if ready_file.is_file():
            try:
                value = json.loads(ready_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                value = None
            if isinstance(value, dict):
                return value, None, "", ""
        if child.poll() is not None:
            stdout, stderr = child.communicate()
            return None, "child_exited_before_ready", stdout, stderr
        time.sleep(0.01)
    termination, stdout, stderr = _terminate_exact_owned_child(
        child, grace_seconds=grace_seconds, phase="setup_timeout"
    )
    return None, termination, stdout, stderr


def _run_owned_child(args: argparse.Namespace) -> dict[str, Any]:
    """Bound the exact callback only after child reports saved history ready."""

    with tempfile.TemporaryDirectory(prefix="r244-91766923-") as temporary:
        ready_file = Path(temporary) / "ready.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child-mode",
            "replay",
            "--stage",
            str(args.stage.resolve()),
            "--replay",
            str(args.replay.resolve()),
            "--target-step",
            str(args.target_step),
            "--callback-deadline-seconds",
            str(args.callback_deadline_seconds),
            "--ready-file",
            str(ready_file),
        ]
        if args.fault_class is not None:
            command.extend(["--fault-class", str(args.fault_class)])
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        started = time.monotonic()
        try:
            child = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=environment,
            )
        except OSError as exc:
            return {
                "created": False,
                "reaped": True,
                "pid": None,
                "returncode": None,
                "setup_timeout": False,
                "callback_timeout": False,
                "termination": None,
                "launch_error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": max(0.0, time.monotonic() - started),
                "stdout": _stream_summary(""),
                "stderr": _stream_summary(""),
                "result": None,
            }
        ready, setup_termination, stdout, stderr = _wait_for_ready_file(
            child=child,
            ready_file=ready_file,
            timeout_seconds=float(args.setup_timeout_seconds),
            grace_seconds=float(args.reap_grace_seconds),
        )
        if ready is None:
            return {
                "created": True,
                "reaped": child.poll() is not None,
                "pid": int(child.pid),
                "returncode": child.returncode,
                "setup_timeout": setup_termination is not None and "setup_timeout" in setup_termination,
                "callback_timeout": False,
                "termination": setup_termination,
                "ready": None,
                "elapsed_seconds": max(0.0, time.monotonic() - started),
                "stdout": _stream_summary(stdout),
                "stderr": _stream_summary(stderr),
                "result": _parse_child_result(stdout),
            }
        if ready.get("pid") != child.pid or ready.get("status") != "ready_for_exact_step58_callback":
            termination, stdout, stderr = _terminate_exact_owned_child(
                child, grace_seconds=float(args.reap_grace_seconds), phase="invalid_ready"
            )
            return {
                "created": True,
                "reaped": child.poll() is not None,
                "pid": int(child.pid),
                "returncode": child.returncode,
                "setup_timeout": False,
                "callback_timeout": False,
                "termination": termination,
                "ready": ready,
                "elapsed_seconds": max(0.0, time.monotonic() - started),
                "stdout": _stream_summary(stdout),
                "stderr": _stream_summary(stderr),
                "result": _parse_child_result(stdout),
            }
        require(child.stdin is not None, "owned replay child has no stdin")
        child.stdin.write("GO\n")
        child.stdin.flush()
        # Allow the exact child to perform its bounded close/reap after the
        # four-second action.  The child independently records action-only
        # elapsed time and the parent interrupts an unreturned action.
        callback_window = float(args.callback_deadline_seconds) + float(args.reap_grace_seconds) + 0.5
        callback_timeout = False
        termination: str | None = None
        try:
            stdout, stderr = child.communicate(timeout=callback_window)
        except subprocess.TimeoutExpired:
            callback_timeout = True
            termination, stdout, stderr = _terminate_exact_owned_child(
                child, grace_seconds=float(args.reap_grace_seconds), phase="callback_timeout"
            )
        return {
            "created": True,
            "reaped": child.poll() is not None,
            "pid": int(child.pid),
            "returncode": child.returncode,
            "setup_timeout": False,
            "callback_timeout": callback_timeout,
            "termination": termination,
            "ready": ready,
            "elapsed_seconds": max(0.0, time.monotonic() - started),
            "callback_window_seconds": callback_window,
            "stdout": _stream_summary(stdout),
            "stderr": _stream_summary(stderr),
            "result": _parse_child_result(stdout),
        }


def _write_receipt_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ReplayError(f"receipt already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ReplayError(f"temporary receipt path already exists: {temporary}")
    encoded = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # The hard link below preserves this inode mode.  Seal the temporary
        # before publishing so the final saved-replay gate is never briefly a
        # writable binding candidate.
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ReplayError(f"receipt already exists: {path}") from exc
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _parent_payload(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    started = time.monotonic()
    stage = args.stage.resolve()
    replay = args.replay.resolve()
    binding_identity = load_binding_identity(
        stage=stage,
        candidate_archive=args.candidate_archive,
        member_manifest=args.member_manifest,
        r225_contract=args.r225_contract,
        r236_contract=args.r236_contract,
    )
    parent_contract = dict(binding_identity["stage_contract"])
    parent_before = dict(parent_contract["stage_snapshot"])
    target, steps = _load_replay_target(replay, int(args.target_step))
    child = _run_owned_child(args)
    result = child.get("result")
    legal_action: list[int] | None = None
    child_valid = False
    if isinstance(result, Mapping):
        try:
            legal = result.get("target_complete_legal_actions")
            if isinstance(legal, list):
                normalized_legal = [as_action(item, field="child legal action") for item in legal]
                legal_action = as_action(result.get("final_action"), field="child final action")
                child_valid = (
                    legal_action in normalized_legal
                    and result.get("status") == "pass"
                    and isinstance(result.get("callback_elapsed_seconds"), (int, float))
                    and not isinstance(result.get("callback_elapsed_seconds"), bool)
                    and float(result["callback_elapsed_seconds"])
                    <= float(args.callback_deadline_seconds)
                    and result.get("permitted_result_path")
                    in {
                        "high_confidence_frozen_direct",
                        "deterministic_mcts_continuation",
                        "shared_tree_mcts",
                        "zero_backup_precomputed_direct_fallback",
                        "contained_precomputed_parent_direct_fallback_after_exact_child_reap",
                    }
                )
        except (HarnessContractError, TypeError, ValueError):
            child_valid = False
    parent_after = stage_snapshot(stage)
    unchanged = parent_after == parent_before
    require(unchanged, "saved replay parent observed a package mutation")
    passed = (
        child.get("created") is True
        and child.get("reaped") is True
        and child.get("setup_timeout") is False
        and child.get("callback_timeout") is False
        and child.get("returncode") == 0
        and child_valid
    )
    status = "passed" if passed else "failed_closed"
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": status,
        "scope": SCOPE,
        "stage": str(stage),
        "exact_package_identity": binding_identity["exact_package"],
        "replay": str(replay),
        "replay_sha256": sha256_file(replay),
        "target_step": int(args.target_step),
        "target_observation_sha256": json_digest(target),
        "target_selection_context": target["select"].get("context"),
        "target_option_count": len(target["select"]["option"]),
        "prior_active_event_count": len(_prior_active_events(steps, int(args.target_step))),
        "callback_deadline_seconds": float(args.callback_deadline_seconds),
        "setup_timeout_seconds": float(args.setup_timeout_seconds),
        "complete_action_cap": COMPLETE_ACTION_CAP,
        "final_action": legal_action,
        "stage_contract": parent_contract,
        "package_mutation_check": {
            "before": parent_before,
            "after": parent_after,
            "unchanged": unchanged,
        },
        "exact_owned_replay_child": child,
        "elapsed_seconds": max(0.0, time.monotonic() - started),
    }
    if passed:
        # This exact target run is only binder-eligible when its child marker
        # itself proves a fault-injected exact reap.  The fault harness is
        # attached in the child; a normal MCTS/high-direct result remains
        # useful diagnostic evidence but cannot manufacture this assertion.
        fault_reap_proved = bool(
            isinstance(result, Mapping)
            and result.get("fault_injected_broker_child_reap_proved") is True
        )
        if fault_reap_proved:
            payload = {
                **passed_preflight_receipt(
                    receipt_name=SAVED_EPISODE_RECEIPT_NAME,
                    common_identity=binding_identity["common_identity"],
                    harness_schema=SCHEMA,
                ),
                **payload,
                "schema": "poke_bot.r235_r236_local_preflight_receipt/v1",
                "status": "passed",
                "passed": True,
                "immutable": True,
                "write_once": True,
                "source_submission_id": 55_416_396,
                "source_episode_id": 91_766_923,
                "seat": 0,
                "final_callback_step": TARGET_STEP,
                "final_callback_ordered_legal_action_count": 2,
                "legal_action_before_hard_deadline": True,
                "fault_injected_broker_child_reap_proved": True,
                "result_path": {
                    "high_confidence_frozen_direct": "high_confidence_frozen_direct",
                    "deterministic_mcts_continuation": "validated_deterministic_continuation_plan_action",
                    "shared_tree_mcts": "validated_mcts_action",
                    "zero_backup_precomputed_direct_fallback": "contained_precomputed_parent_direct_fallback_after_exact_child_reap",
                    "contained_precomputed_parent_direct_fallback_after_exact_child_reap": "contained_precomputed_parent_direct_fallback_after_exact_child_reap",
                }[str(result.get("permitted_result_path"))],
            }
        else:
            payload["binder_eligible"] = False
            payload["binder_ineligibility"] = (
                "exact saved replay lacked actual fault-injected broker-child reap evidence"
            )
    return (0 if passed else 1), payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--candidate-archive", type=Path)
    parser.add_argument("--member-manifest", type=Path)
    parser.add_argument(
        "--r225-contract",
        type=Path,
        default=ROOT / "state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json",
    )
    parser.add_argument(
        "--r236-contract",
        type=Path,
        default=ROOT / "state/canonical-libcg-r236.json",
    )
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--target-step", type=int, default=TARGET_STEP)
    parser.add_argument("--callback-deadline-seconds", type=_positive_seconds, default=4.0)
    parser.add_argument("--setup-timeout-seconds", type=_positive_seconds, default=60.0)
    parser.add_argument("--reap-grace-seconds", type=_positive_seconds, default=0.25)
    parser.add_argument(
        "--fault-class",
        choices=("timeout", "crash", "protocol", "evaluator", "native", "cleanup"),
        help=(
            "inject one package-external exact-child fault at the saved callback; "
            "required for a binder-eligible saved-step reap proof"
        ),
    )
    parser.add_argument("--child-mode", choices=("replay",), help=argparse.SUPPRESS)
    parser.add_argument("--ready-file", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(args.target_step == TARGET_STEP, "this regression is pinned to saved step 58")
    if args.child_mode is not None:
        return _child_main(args)
    require(args.receipt is not None, "--receipt is required for parent harness")
    require(args.candidate_archive is not None, "--candidate-archive is required for parent harness")
    require(args.member_manifest is not None, "--member-manifest is required for parent harness")
    receipt = args.receipt.resolve()
    require(not receipt.exists() and not receipt.is_symlink(), f"receipt already exists: {receipt}")
    try:
        exit_code, payload = _parent_payload(args)
    except Exception as exc:  # noqa: BLE001 - record a sealed parent failure
        exit_code = 1
        payload = {
            "schema": SCHEMA,
            "status": "failed_closed",
            "scope": SCOPE,
            "stage": str(args.stage.resolve()),
            "replay": str(args.replay.resolve()),
            "target_step": int(args.target_step),
            "failure": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }
    _write_receipt_once(receipt, payload)
    print(FINAL_PREFIX + json.dumps(payload, sort_keys=True), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
