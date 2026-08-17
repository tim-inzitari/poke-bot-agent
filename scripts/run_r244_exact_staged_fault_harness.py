#!/usr/bin/env python3
"""Exercise r244 Kaggle child containment against one sealed package.

This is intentionally a package-*external* fault harness.  It never alters a
staged member and it never starts a Kaggle client, queue, upload, GPU job, or
BO1000 workload.  For every fault class it imports the supplied sealed
``main.py`` and its sealed broker in a fresh worker process.  The broker is
then given one fresh, real socket child that deliberately behaves like a
native/evaluator fault.  Thus the package's own bounded socket wait, exact
``Popen`` TERM/KILL/reap implementation, parent-direct fallback, and marker
logic execute for real instead of being represented by hand-written receipts.

The fake child is deliberately outside the package and only exists in this
explicit harness process.  It never loads a simulator or model, while the
sealed parent still performs its normal package identity validation before the
fault-injection adapter supplies a minimal deterministic direct-policy target.
That keeps this focused containment gate CPU-only; the separate saved-replay
and full-physical-game harnesses exercise the real frozen model/simulator.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from r228_kaggle_r244_harness_common import (
    PREFLIGHT_RECEIPT_SCHEMA,
    R225_TYPED_CONTRACT_MEMBER,
    R236_NATIVE_MEMBERS,
    collect_markers,
    load_binding_identity,
    stage_snapshot,
    validate_degraded_marker,
)

SCHEMA = "poke_bot.r244_exact_staged_native_child_fault_harness/v1"
RECEIPT_NAME = "focused_native_child_fault_suite_receipt"
WORKER_RESULT_PREFIX = "R244_EXACT_STAGED_FAULT_WORKER_RESULT "
FINAL_PREFIX = "R244_EXACT_STAGED_FAULT_HARNESS "

FAULT_CLASSES = ("timeout", "crash", "protocol", "evaluator", "native", "cleanup")
_ALL_WORKER_CASES = (*FAULT_CLASSES, "invalid_parent_direct", "unreaped_child")
_FAKE_CHILD_READY_SCHEMA = "poke_bot.r228_kaggle_subprocess_broker/v1"
R246_TYPED_CONTRACT_OWNER_DECISION_REVISION = 246
R246_TYPED_CONTRACT_SHA256 = (
    "sha256:3225b07997bc58cc5e89239491533628cae654b48c092dec76ce56a6b8205eb3"
)
_FAKE_CHILD_PRELOAD = {
    "member": "cg/libcg.so",
    "sha256": R236_NATIVE_MEMBERS["cg/libcg.so"]["sha256"],
    "size_bytes": R236_NATIVE_MEMBERS["cg/libcg.so"]["size_bytes"],
}


class FaultHarnessError(RuntimeError):
    """The focused package fault gate cannot truthfully pass."""


def _validate_r246_canonical_binding(
    *, r225_contract: Path, common_identity: Mapping[str, Any]
) -> dict[str, Any]:
    """Pin this gate to the final r246 typed source, not merely any r225 file.

    ``load_binding_identity`` proves that the supplied canonical source and
    staged member are byte-identical.  This extra check fixes *which* canonical
    source may receive a focused-fault pass: the owner-final r246 contract
    containing the deterministic-terminal-win rule.  The fault suite neither
    invokes nor claims that proof; it only verifies that a faulted game cannot
    turn it into viability credit.
    """

    canonical = r225_contract.expanduser().resolve()
    if not canonical.is_file() or canonical.is_symlink():
        raise FaultHarnessError("r246 canonical r225 contract is not a regular file")
    observed_sha256 = _sha256_bytes(canonical.read_bytes())
    if observed_sha256 != R246_TYPED_CONTRACT_SHA256:
        raise FaultHarnessError(
            "focused fault suite is not bound to the final r246 canonical r225 digest"
        )
    if common_identity.get("r225_contract_sha256") != R246_TYPED_CONTRACT_SHA256:
        raise FaultHarnessError("focused fault common identity r225 digest drifted from r246")
    try:
        payload = json.loads(canonical.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FaultHarnessError("r246 canonical r225 contract is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise FaultHarnessError("r246 canonical r225 contract is not an object")
    if payload.get("owner_decision_revision") != R246_TYPED_CONTRACT_OWNER_DECISION_REVISION:
        raise FaultHarnessError("r246 canonical r225 owner decision revision drifted")
    if (
        payload.get("owner_proven_deterministic_terminal_win_this_turn_revision")
        != R246_TYPED_CONTRACT_OWNER_DECISION_REVISION
    ):
        raise FaultHarnessError("r246 canonical r225 terminal-win binding is absent")
    return {
        "owner_decision_revision": R246_TYPED_CONTRACT_OWNER_DECISION_REVISION,
        "r225_contract_sha256": R246_TYPED_CONTRACT_SHA256,
        "terminal_win_proof_exercised_by_fault_suite": False,
        "faulted_game_viability_success_allowed": False,
    }


def _canonical_json(value: object) -> bytes:
    try:
        return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise FaultHarnessError("fault-harness receipt is not canonical JSON") from exc


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False, separators=(",", ":")))
    except (TypeError, ValueError) as exc:
        raise FaultHarnessError("fault-harness value is not JSON-native") from exc


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _stream_summary(value: str) -> dict[str, Any]:
    raw = value.encode("utf-8", errors="replace")
    return {
        "sha256": _sha256_bytes(raw),
        "bytes": len(raw),
        "tail": value[-8192:],
    }


def _positive_seconds(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite positive number") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def _write_once_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish a receipt once; never overwrite a prior gate result."""

    target = path.expanduser()
    if target.is_symlink() or target.exists():
        raise FaultHarnessError(f"fault receipt already exists: {target}")
    parent = target.parent
    if not parent.is_dir() or parent.is_symlink():
        raise FaultHarnessError("fault receipt parent must be an existing physical directory")
    encoded = _canonical_json(dict(payload))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        try:
            written = 0
            while written < len(encoded):
                amount = os.write(descriptor, encoded[written:])
                if amount <= 0:
                    raise OSError("could not write fault receipt")
                written += amount
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise FaultHarnessError(f"fault receipt already exists: {target}") from exc
        directory_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _under_stage(path: str | Path, stage: Path) -> bool:
    try:
        Path(path).resolve().relative_to(stage.resolve())
    except (OSError, ValueError):
        return False
    return True


def _clear_staged_modules() -> None:
    """Avoid a workspace module masquerading as the sealed package."""

    for name in tuple(sys.modules):
        if name == "main" or name == "poke_bot" or name.startswith("poke_bot."):
            sys.modules.pop(name, None)


def _load_exact_stage_main(stage: Path) -> tuple[Any, Any, Any]:
    """Import the parent and broker solely from ``stage`` without bytecode writes."""

    stage = stage.expanduser().resolve()
    if not stage.is_dir() or stage.is_symlink():
        raise FaultHarnessError("stage is not a physical directory")
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["CG_LIB_PATH"] = str(stage)
    os.chdir(stage)
    _clear_staged_modules()
    sys.path.insert(0, str(stage))
    importlib.invalidate_caches()
    main = importlib.import_module("main")
    broker = importlib.import_module("poke_bot.r228_kaggle_broker")
    features = importlib.import_module("poke_bot.features")
    for module in (main, broker, features):
        location = getattr(module, "__file__", None)
        if not isinstance(location, str) or not _under_stage(location, stage):
            raise FaultHarnessError("staged import resolved outside the sealed package")
    return main, broker, features


class _HarnessPolicy:
    """Minimal target-emitting direct-policy surface for the parent only."""

    def __init__(self) -> None:
        self.board_history: list[object] = []
        self.previous_action_history: list[object] = []
        self._previous_action_token: object = None
        self.targets: list[dict[str, Any]] = []
        self.collect_targets = False

    @staticmethod
    def _history_context_limit() -> int:
        return 16


class _HarnessDirect:
    """Deterministic low-confidence direct adapter used only in a worker."""

    def __init__(self, *, action: Sequence[int], stage_candidates: Sequence[Sequence[int]]) -> None:
        self.action = [int(item) for item in action]
        self.stage_candidates = [[int(item) for item in row] for row in stage_candidates]
        self.policy = _HarnessPolicy()

    def _turn_order_choice(self, _observation: Mapping[str, Any]) -> None:
        return None

    def _ensure_runtime(self) -> tuple[list[int], object, _HarnessPolicy]:
        # The entrypoint only samples diagnostic device metadata here.  A
        # plain object truthfully produces an unavailable observation without
        # starting a model/GPU during this focused child-containment gate.
        return [741] * 60, object(), self.policy

    def agent(self, observation: Mapping[str, Any]) -> list[int]:
        if observation.get("select") is None:
            return [741] * 60
        limit = self.policy._history_context_limit()
        self.policy.board_history.append(("fault-harness", len(self.policy.board_history)))
        self.policy.previous_action_history.append(self.policy._previous_action_token)
        self.policy.board_history = self.policy.board_history[-limit:]
        self.policy.previous_action_history = self.policy.previous_action_history[-limit:]
        self.policy._previous_action_token = ("direct", list(self.action))
        if self.policy.collect_targets:
            try:
                selected_index = self.stage_candidates.index(list(self.action))
            except ValueError as exc:  # defensive harness construction failure
                raise FaultHarnessError("direct action absent from harness factorized candidates") from exc
            count = len(self.stage_candidates)
            if count < 1:
                raise FaultHarnessError("harness factorized candidate list is empty")
            # Deliberately ambiguous (<0.80) so the sealed parent exercises
            # its broker path.  This target is never a search/model result.
            selected_probability = 0.50 if count > 1 else 1.0
            remainder = (1.0 - selected_probability) / max(1, count - 1)
            probabilities = [remainder] * count
            probabilities[selected_index] = selected_probability
            self.policy.targets.append(
                {
                    "observation": dict(observation),
                    "action": list(self.action),
                    "factorized_stages": [
                        {
                            "action_combos": _json_copy(self.stage_candidates),
                            "policy": probabilities,
                            "selected_index": selected_index,
                        }
                    ],
                    "diagnostics": {
                        "target_source": "history_policy",
                        "trusted": True,
                        "history_length": len(self.policy.board_history),
                    },
                }
            )
        return list(self.action)


def _branch_observation() -> dict[str, Any]:
    return {
        "current": {"yourIndex": 0},
        "select": {"option": [{}, {}], "minCount": 1, "maxCount": 1},
    }


def _deck_observation() -> dict[str, Any]:
    return {"current": {"yourIndex": 0}, "select": None}


class _ControlledUnreapedProxy:
    """Exercise the broker's unreaped hard-fail branch without leaking a PID.

    The wrapped fake child is a real process.  The proxy reports both bounded
    waits as timeouts after forwarding the exact TERM/KILL calls.  The harness
    then reaps the underlying exact child itself and records that cleanup.  It
    therefore tests the parent hard-fail rule without falsely claiming that a
    real process was intentionally left alive.
    """

    def __init__(self, child: subprocess.Popen[bytes]) -> None:
        self._child = child
        self.pid = int(child.pid)
        self.returncode: int | None = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_timeouts: list[float | None] = []

    def poll(self) -> None:
        return None

    def terminate(self) -> None:
        self.terminate_calls += 1
        try:
            self._child.terminate()
        except ProcessLookupError:
            pass

    def kill(self) -> None:
        self.kill_calls += 1
        try:
            self._child.kill()
        except ProcessLookupError:
            pass

    def wait(self, timeout: float | None = None) -> NoReturn:
        self.wait_timeouts.append(timeout)
        raise subprocess.TimeoutExpired(["controlled-unreaped-child"], timeout)

    def reap_underlying(self) -> dict[str, Any]:
        return _bounded_reap_exact_process(
            self._child,
            reason="controlled_unreaped_proxy_harness_cleanup",
            grace_seconds=0.25,
        )


def _bounded_reap_exact_process(
    child: subprocess.Popen[Any], *, reason: str, grace_seconds: float
) -> dict[str, Any]:
    """Boundedly reap exactly one process created by this harness.

    This intentionally calls only the concrete ``Popen`` instance; it never
    sends a process-group/session signal and never discovers or touches an
    unrelated process.
    """

    started = time.monotonic()
    report: dict[str, Any] = {
        "reason": reason,
        "pid": int(child.pid),
        "term_sent": False,
        "kill_sent": False,
        "reaped": False,
    }
    returncode = child.poll()
    if returncode is None:
        try:
            child.terminate()
            report["term_sent"] = True
        except ProcessLookupError:
            pass
        try:
            returncode = child.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            returncode = child.poll()
    if returncode is None:
        try:
            child.kill()
            report["kill_sent"] = True
        except ProcessLookupError:
            pass
        try:
            returncode = child.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            returncode = child.poll()
    report["elapsed_seconds"] = max(0.0, time.monotonic() - started)
    if returncode is not None:
        report["returncode"] = int(returncode)
        report["reaped"] = True
    return report


@dataclass
class FaultInjectionHandle:
    """Reversible Popen hook for one staged broker module."""

    stage: Path
    fault_class: str
    broker_module: Any
    original_popen: Callable[..., subprocess.Popen[Any]]
    spawned: list[subprocess.Popen[Any]] = field(default_factory=list)
    controlled_unreaped: list[_ControlledUnreapedProxy] = field(default_factory=list)
    restored: bool = False

    def restore(self) -> None:
        if not self.restored:
            self.broker_module.subprocess.Popen = self.original_popen
            self.restored = True

    def cleanup_owned_children(self) -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        for proxy in self.controlled_unreaped:
            reports.append(proxy.reap_underlying())
        proxy_children = {id(proxy._child) for proxy in self.controlled_unreaped}
        for child in self.spawned:
            if id(child) in proxy_children:
                continue
            reports.append(
                _bounded_reap_exact_process(
                    child,
                    reason="fault_harness_owned_child_cleanup",
                    grace_seconds=0.25,
                )
            )
        return reports


def install_fault_injected_broker_child(stage: Path, fault_class: str) -> FaultInjectionHandle:
    """Install a reversible fresh-child hook for the *loaded staged* broker.

    This is deliberately public for the saved-step58 harness.  Call it only
    after that harness has imported the exact staged ``main``/broker, and call
    ``restore()`` plus ``cleanup_owned_children()`` in its ``finally`` block.
    It touches neither stage bytes nor global process groups.
    """

    if fault_class not in {*FAULT_CLASSES, "unreaped_child"}:
        raise FaultHarnessError(f"unsupported injected child fault class: {fault_class!r}")
    stage = stage.expanduser().resolve()
    broker_module = sys.modules.get("poke_bot.r228_kaggle_broker")
    if broker_module is None:
        broker_module = importlib.import_module("poke_bot.r228_kaggle_broker")
    module_file = getattr(broker_module, "__file__", None)
    if not isinstance(module_file, str) or not _under_stage(module_file, stage):
        raise FaultHarnessError("fault injection refuses a broker outside the staged package")
    original_popen = broker_module.subprocess.Popen
    handle = FaultInjectionHandle(
        stage=stage,
        fault_class=fault_class,
        broker_module=broker_module,
        original_popen=original_popen,
    )

    def spawn_exact_fake_child(_argv: Sequence[str], **kwargs: Any) -> Any:
        pass_fds = kwargs.get("pass_fds")
        if not isinstance(pass_fds, (list, tuple)) or len(pass_fds) != 1:
            raise FaultHarnessError("staged broker did not provide one owned child socket fd")
        child_fd = int(pass_fds[0])
        environment = dict(kwargs.get("env") or os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["POKEBOT_R244_FAULT_HARNESS"] = "1"
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--fake-child",
            "--child-fd",
            str(child_fd),
            "--fault-class",
            ("cleanup" if fault_class == "unreaped_child" else fault_class),
        ]
        child = original_popen(
            command,
            cwd=str(stage),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            pass_fds=(child_fd,),
        )
        handle.spawned.append(child)
        if fault_class == "unreaped_child":
            proxy = _ControlledUnreapedProxy(child)
            handle.controlled_unreaped.append(proxy)
            return proxy
        return child

    broker_module.subprocess.Popen = spawn_exact_fake_child
    return handle


def _send_line(sock: socket.socket, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(dict(payload), allow_nan=False, separators=(",", ":")).encode("utf-8")
    sock.sendall(encoded + b"\n")


def _fake_child_main(*, child_fd: int, fault_class: str) -> int:
    """A tiny actual child endpoint for the staged broker's socket protocol."""

    if fault_class not in FAULT_CLASSES:
        raise FaultHarnessError(f"unsupported fake child fault class: {fault_class!r}")
    channel = socket.socket(fileno=int(child_fd))
    channel.settimeout(None)
    try:
        _send_line(
            channel,
            {
                "schema": _FAKE_CHILD_READY_SCHEMA,
                "type": "ready",
                "preload_stock_library": dict(_FAKE_CHILD_PRELOAD),
                "fault_harness": True,
            },
        )
        buffer = bytearray()
        while True:
            incoming = channel.recv(64 * 1024)
            if not incoming:
                return 0
            buffer.extend(incoming)
            while b"\n" in buffer:
                raw, _separator, rest = buffer.partition(b"\n")
                buffer[:] = rest
                if not raw:
                    continue
                request = json.loads(raw.decode("utf-8"))
                if not isinstance(request, Mapping):
                    return 2
                kind = request.get("type")
                request_id = request.get("request_id")
                if kind == "sync":
                    events = request.get("events")
                    _send_line(
                        channel,
                        {
                            "schema": _FAKE_CHILD_READY_SCHEMA,
                            "type": "synced",
                            "request_id": request_id,
                            "applied": len(events) if isinstance(events, list) else -1,
                        },
                    )
                    continue
                if kind == "note":
                    _send_line(
                        channel,
                        {
                            "schema": _FAKE_CHILD_READY_SCHEMA,
                            "type": "noted",
                            "request_id": request_id,
                            "applied": True,
                        },
                    )
                    continue
                if kind == "close":
                    _send_line(
                        channel,
                        {
                            "schema": _FAKE_CHILD_READY_SCHEMA,
                            "type": "closed",
                            "request_id": request_id,
                        },
                    )
                    return 0
                if kind != "select":
                    return 2
                if fault_class == "timeout":
                    _send_line(
                        channel,
                        {
                            "schema": _FAKE_CHILD_READY_SCHEMA,
                            "type": "progress",
                            "request_id": request_id,
                            "payload": {"lane_id": 0, "phase": "fault_harness_timeout"},
                        },
                    )
                    while True:
                        time.sleep(1.0)
                if fault_class == "crash":
                    os._exit(23)
                if fault_class == "protocol":
                    channel.sendall(b"not-json\n")
                    while True:
                        time.sleep(1.0)
                fault_code = {
                    "evaluator": "fault_injected_evaluator",
                    "native": "fault_injected_native",
                    "cleanup": "fault_injected_cleanup",
                }[fault_class]
                _send_line(
                    channel,
                    {
                        "schema": _FAKE_CHILD_READY_SCHEMA,
                        "type": "error",
                        "request_id": request_id,
                        "code": fault_code,
                        "message": f"explicit r244 fault harness {fault_class} fault",
                        "detail": {"fault_harness": True, "fault_class": fault_class},
                    },
                )
                while True:
                    time.sleep(1.0)
    finally:
        try:
            channel.close()
        except OSError:
            pass


def _install_harness_direct(main: Any, features: Any, *, invalid_parent_direct: bool) -> _HarnessDirect:
    direct_action = [9] if invalid_parent_direct else [0]
    factorized = [[9]] if invalid_parent_direct else [[0], [1]]
    direct = _HarnessDirect(action=direct_action, stage_candidates=factorized)
    main._direct = lambda: direct
    features.enumerate_action_combos = lambda _obs, *, max_combos: [[0], [1]]
    features.factorized_action_candidates = lambda _obs, _prefix: _json_copy(factorized)
    return direct


def _assert_exact_reap(marker: Mapping[str, Any], *, expected_pid: int) -> dict[str, Any]:
    child_fault = marker.get("child_fault")
    if not isinstance(child_fault, Mapping):
        raise FaultHarnessError("degraded marker omitted its child fault")
    child_reap = child_fault.get("child_reap")
    if not isinstance(child_reap, Mapping) or child_reap.get("reaped") is not True:
        raise FaultHarnessError("degraded marker did not prove exact child reaping")
    identity = child_fault.get("child_identity")
    if not isinstance(identity, Mapping) or identity.get("pid") != expected_pid:
        raise FaultHarnessError("degraded marker child identity does not match injected child")
    reap_identity = child_reap.get("child_identity")
    if isinstance(reap_identity, Mapping) and reap_identity.get("pid") != expected_pid:
        raise FaultHarnessError("broker reaped a PID other than its injected child")
    return dict(child_reap)


def _run_fault_case_worker(stage: Path, fault_class: str) -> dict[str, Any]:
    """Run one real sealed-parent/fresh-child containment scenario."""

    started = time.monotonic()
    stage_before = stage_snapshot(stage)
    main: Any | None = None
    handle: FaultInjectionHandle | None = None
    captured = io.StringIO()
    result: dict[str, Any]
    try:
        main, _broker, features = _load_exact_stage_main(stage)
        _install_harness_direct(main, features, invalid_parent_direct=False)
        handle = install_fault_injected_broker_child(stage, fault_class)
        with contextlib.redirect_stdout(captured):
            deck_action = main.agent(_deck_observation())
            selected = main.agent(_branch_observation())
            direct_only_after_fault = main.agent(_branch_observation())
            terminal_deck_action = main.agent(_deck_observation())
        markers = collect_markers(captured.getvalue())
        if deck_action != [741] * 60 or terminal_deck_action != [741] * 60:
            raise FaultHarnessError("harness deck-boundary action drifted")
        if selected != [0] or direct_only_after_fault != [0]:
            raise FaultHarnessError("contained path did not return the precomputed direct action")
        if markers["hard_failures"]:
            raise FaultHarnessError("contained child fault emitted a parent hard failure")
        if len(markers["degraded_fallbacks"]) != 1:
            raise FaultHarnessError("faulted game did not emit exactly one degraded marker")
        if markers["full_gameplay_successes"]:
            raise FaultHarnessError("faulted game received viability-success credit")
        if handle is None or len(handle.spawned) != 1:
            raise FaultHarnessError("fault case did not create exactly one fresh owned child")
        child = handle.spawned[0]
        marker = markers["degraded_fallbacks"][0]
        validation = validate_degraded_marker(marker, legal_actions=[[0], [1]])
        reap = _assert_exact_reap(marker, expected_pid=int(child.pid))
        expected_codes = {
            "timeout": {"response_timeout"},
            "crash": {"child_socket_closed", "child_exited"},
            "protocol": {"malformed_child_json"},
            "evaluator": {"fault_injected_evaluator"},
            "native": {"fault_injected_native"},
            "cleanup": {"fault_injected_cleanup"},
        }
        observed_code = validation.get("fault_code")
        if observed_code not in expected_codes[fault_class]:
            raise FaultHarnessError(
                f"fault class {fault_class} produced wrong containment code: {observed_code!r}"
            )
        result = {
            "status": "passed",
            "fault_class": fault_class,
            "selected_action": selected,
            "direct_only_action_after_fault": direct_only_after_fault,
            "fault_code": observed_code,
            "degraded_marker_count": len(markers["degraded_fallbacks"]),
            "full_gameplay_success_marker_count": len(markers["full_gameplay_successes"]),
            "exact_owned_child": {
                "pid": int(child.pid),
                "reap": reap,
                "reaped": True,
            },
            "markers": {
                "degraded": _json_copy(marker),
                "hard_failure_count": len(markers["hard_failures"]),
            },
        }
    except Exception as exc:  # noqa: BLE001 - preserve a sealed failure payload
        result = {
            "status": "failed_closed",
            "fault_class": fault_class,
            "failure": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }
    finally:
        if handle is not None:
            handle.restore()
            cleanup = handle.cleanup_owned_children()
            result["harness_owned_child_cleanup"] = cleanup
            if any(not row.get("reaped") for row in cleanup):
                result["status"] = "failed_closed"
                result["cleanup_failure"] = "harness exact child failed bounded cleanup"
        stage_after = stage_snapshot(stage)
        result["stage_mutation_check"] = {
            "before": stage_before,
            "after": stage_after,
            "unchanged": stage_before == stage_after,
        }
        if stage_before != stage_after:
            result["status"] = "failed_closed"
            result["stage_mutation_failure"] = "fault harness mutated the sealed package"
        result["stdout"] = _stream_summary(captured.getvalue())
        result["elapsed_seconds"] = max(0.0, time.monotonic() - started)
    return result


def _run_invalid_direct_worker(stage: Path) -> dict[str, Any]:
    started = time.monotonic()
    stage_before = stage_snapshot(stage)
    captured = io.StringIO()
    result: dict[str, Any]
    try:
        main, _broker, features = _load_exact_stage_main(stage)
        _install_harness_direct(main, features, invalid_parent_direct=True)
        with contextlib.redirect_stdout(captured):
            try:
                main.agent(_branch_observation())
            except RuntimeError:
                pass
            else:
                raise FaultHarnessError("invalid parent direct action did not hard-fail")
        markers = collect_markers(captured.getvalue())
        if len(markers["hard_failures"]) != 1:
            raise FaultHarnessError("invalid parent direct test did not emit exactly one hard marker")
        marker = markers["hard_failures"][0]
        if marker.get("code") != "direct_action_outside_complete_legal_order":
            raise FaultHarnessError("invalid parent direct test emitted the wrong hard-failure code")
        if markers["degraded_fallbacks"] or markers["full_gameplay_successes"]:
            raise FaultHarnessError("invalid parent direct test fell back or claimed success")
        result = {
            "status": "passed",
            "case": "invalid_parent_direct",
            "hard_failure_code": marker.get("code"),
            "child_started": False,
        }
    except Exception as exc:  # noqa: BLE001 - preserve a sealed failure payload
        result = {
            "status": "failed_closed",
            "case": "invalid_parent_direct",
            "failure": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }
    finally:
        stage_after = stage_snapshot(stage)
        result["stage_mutation_check"] = {
            "before": stage_before,
            "after": stage_after,
            "unchanged": stage_before == stage_after,
        }
        if stage_before != stage_after:
            result["status"] = "failed_closed"
        result["stdout"] = _stream_summary(captured.getvalue())
        result["elapsed_seconds"] = max(0.0, time.monotonic() - started)
    return result


def _run_unreaped_child_worker(stage: Path) -> dict[str, Any]:
    started = time.monotonic()
    stage_before = stage_snapshot(stage)
    captured = io.StringIO()
    handle: FaultInjectionHandle | None = None
    result: dict[str, Any]
    try:
        main, _broker, features = _load_exact_stage_main(stage)
        _install_harness_direct(main, features, invalid_parent_direct=False)
        handle = install_fault_injected_broker_child(stage, "unreaped_child")
        with contextlib.redirect_stdout(captured):
            main.agent(_deck_observation())
            try:
                main.agent(_branch_observation())
            except RuntimeError:
                pass
            else:
                raise FaultHarnessError("unreaped child containment did not hard-fail")
        markers = collect_markers(captured.getvalue())
        if len(markers["hard_failures"]) != 1:
            raise FaultHarnessError("unreaped child test did not emit exactly one hard marker")
        marker = markers["hard_failures"][0]
        if marker.get("code") != "broker_child_not_reaped":
            raise FaultHarnessError("unreaped child test emitted the wrong hard-failure code")
        if markers["degraded_fallbacks"] or markers["full_gameplay_successes"]:
            raise FaultHarnessError("unreaped child test issued fallback or success credit")
        if handle is None or len(handle.controlled_unreaped) != 1:
            raise FaultHarnessError("unreaped child test did not use the controlled exact child")
        proxy = handle.controlled_unreaped[0]
        if proxy.terminate_calls != 1 or proxy.kill_calls != 1:
            raise FaultHarnessError("broker did not attempt bounded TERM then KILL on exact child")
        result = {
            "status": "passed",
            "case": "unreaped_child",
            "hard_failure_code": marker.get("code"),
            "proxy_pid": proxy.pid,
            "term_calls": proxy.terminate_calls,
            "kill_calls": proxy.kill_calls,
            "wait_call_count": len(proxy.wait_timeouts),
            "test_mode": "controlled_unreap_proxy_over_real_exact_child",
        }
    except Exception as exc:  # noqa: BLE001 - preserve a sealed failure payload
        result = {
            "status": "failed_closed",
            "case": "unreaped_child",
            "failure": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }
    finally:
        if handle is not None:
            handle.restore()
            cleanup = handle.cleanup_owned_children()
            result["harness_owned_child_cleanup"] = cleanup
            if not cleanup or any(not row.get("reaped") for row in cleanup):
                result["status"] = "failed_closed"
                result["cleanup_failure"] = "controlled exact child was not reaped"
        stage_after = stage_snapshot(stage)
        result["stage_mutation_check"] = {
            "before": stage_before,
            "after": stage_after,
            "unchanged": stage_before == stage_after,
        }
        if stage_before != stage_after:
            result["status"] = "failed_closed"
        result["stdout"] = _stream_summary(captured.getvalue())
        result["elapsed_seconds"] = max(0.0, time.monotonic() - started)
    return result


def _worker_main(args: argparse.Namespace) -> int:
    stage = Path(args.stage).expanduser().resolve()
    if args.case in FAULT_CLASSES:
        payload = _run_fault_case_worker(stage, args.case)
    elif args.case == "invalid_parent_direct":
        payload = _run_invalid_direct_worker(stage)
    elif args.case == "unreaped_child":
        payload = _run_unreaped_child_worker(stage)
    else:  # pragma: no cover - argparse guards this
        raise FaultHarnessError(f"unsupported worker case: {args.case!r}")
    print(WORKER_RESULT_PREFIX + json.dumps(payload, sort_keys=True), flush=True)
    return 0 if payload.get("status") == "passed" else 1


def _parse_worker_result(stdout: str) -> dict[str, Any]:
    rows = [line for line in stdout.splitlines() if line.startswith(WORKER_RESULT_PREFIX)]
    if len(rows) != 1:
        raise FaultHarnessError("fault worker did not emit exactly one result row")
    try:
        payload = json.loads(rows[0][len(WORKER_RESULT_PREFIX) :])
    except json.JSONDecodeError as exc:
        raise FaultHarnessError("fault worker result is malformed JSON") from exc
    if not isinstance(payload, dict):
        raise FaultHarnessError("fault worker result is not an object")
    return payload


def _run_owned_worker(*, stage: Path, case: str, timeout_seconds: float) -> dict[str, Any]:
    """Run an isolated worker with a bounded exact-PID reaping policy."""

    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        "--stage",
        str(stage),
        "--case",
        case,
    ]
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["POKEBOT_R244_FAULT_HARNESS"] = "1"
    started = time.monotonic()
    child = subprocess.Popen(
        command,
        cwd=str(stage),
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        close_fds=True,
    )
    timed_out = False
    try:
        stdout, stderr = child.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        child.terminate()
        try:
            stdout, stderr = child.communicate(timeout=0.25)
        except subprocess.TimeoutExpired:
            child.kill()
            stdout, stderr = child.communicate(timeout=0.25)
    reap = {
        "pid": int(child.pid),
        "returncode": child.returncode,
        "reaped": child.poll() is not None,
        "timed_out": timed_out,
        "elapsed_seconds": max(0.0, time.monotonic() - started),
        "containment": "exact_owned_worker_pid_only_no_process_group_signal",
    }
    if timed_out:
        raise FaultHarnessError(f"fault worker {case} exceeded hard timeout: {reap}")
    if child.returncode != 0:
        raise FaultHarnessError(
            f"fault worker {case} failed with {child.returncode}: "
            f"stdout={_stream_summary(stdout)!r} stderr={_stream_summary(stderr)!r}"
        )
    payload = _parse_worker_result(stdout)
    if payload.get("status") != "passed":
        raise FaultHarnessError(f"fault worker {case} reported failure: {payload!r}")
    payload["outer_worker_reap"] = reap
    payload["worker_stderr"] = _stream_summary(stderr)
    return payload


def run_fault_harness(
    *,
    stage: Path,
    candidate_archive: Path,
    member_manifest: Path,
    output: Path,
    worker_timeout_seconds: float = 12.0,
    r225_contract: Path = ROOT / R225_TYPED_CONTRACT_MEMBER,
    r236_contract: Path = ROOT / "state/canonical-libcg-r236.json",
) -> dict[str, Any]:
    """Run all actual focused child-fault cases and publish one gate receipt."""

    if not math.isfinite(worker_timeout_seconds) or worker_timeout_seconds <= 0.0:
        raise FaultHarnessError("worker timeout must be a finite positive number")
    started = time.monotonic()
    stage = stage.expanduser().resolve()
    try:
        binding = load_binding_identity(
            stage=stage,
            candidate_archive=candidate_archive,
            member_manifest=member_manifest,
            r225_contract=r225_contract,
            r236_contract=r236_contract,
        )
        r246_binding = _validate_r246_canonical_binding(
            r225_contract=r225_contract,
            common_identity=binding["common_identity"],
        )
        before = stage_snapshot(stage)
        fault_results = {
            fault: _run_owned_worker(
                stage=stage, case=fault, timeout_seconds=worker_timeout_seconds
            )
            for fault in FAULT_CLASSES
        }
        invalid = _run_owned_worker(
            stage=stage, case="invalid_parent_direct", timeout_seconds=worker_timeout_seconds
        )
        unreaped = _run_owned_worker(
            stage=stage, case="unreaped_child", timeout_seconds=worker_timeout_seconds
        )
        after = stage_snapshot(stage)
        if before != after:
            raise FaultHarnessError("focused fault suite mutated the sealed package tree")
        if any(
            result.get("full_gameplay_success_marker_count") != 0
            or result.get("degraded_marker_count") != 1
            for result in fault_results.values()
        ):
            raise FaultHarnessError("fault-injected game path claimed viability success")
        payload: dict[str, Any] = {
            "schema": PREFLIGHT_RECEIPT_SCHEMA,
            "receipt_name": RECEIPT_NAME,
            "status": "passed",
            "passed": True,
            "immutable": True,
            "write_once": True,
            **binding["common_identity"],
            "focused_fault_suite_passed": True,
            "nonreaped_child_hard_fail_test_passed": True,
            "parent_returned_action_legality_hard_fail_test_passed": True,
            "fault_injected_full_game_degraded_marker_and_no_viability_credit_passed": True,
            "fault_classes_covered": list(FAULT_CLASSES),
            "harness": {
                "schema": SCHEMA,
                "scope": "exact_staged_r238_r242_r244_parent_broker_actual_owned_socket_children",
                "network_accessed": False,
                "kaggle_api_called": False,
                "kaggle_upload_used": False,
                "gpu_used": False,
                "bo1000_modified": False,
                "production_stage_mutated": False,
                "test_direct_adapter": "worker_only_minimal_target_adapter_no_model_or_simulator",
                "child_containment": "exact_owned_popen_pid_bounded_TERM_KILL_reap_no_process_group_signal",
                "resource_envelope_does_not_infer_cuda_availability": True,
                "r246_canonical_contract_binding": r246_binding,
            },
            "exact_package": binding["exact_package"],
            "stage_contract": binding["stage_contract"],
            "package_mutation_check": {
                "before": before,
                "after": after,
                "unchanged": True,
            },
            "fault_cases": fault_results,
            "invalid_parent_direct_hard_fail": invalid,
            "unreaped_child_hard_fail": unreaped,
            "elapsed_seconds": max(0.0, time.monotonic() - started),
        }
    except Exception as exc:
        raise FaultHarnessError(
            f"focused exact-package fault suite failed closed: {type(exc).__name__}: {exc}"
        ) from exc
    _write_once_atomic(output, payload)
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fake-child", action="store_true")
    parser.add_argument("--child-fd", type=int)
    parser.add_argument("--fault-class", choices=FAULT_CLASSES)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--case", choices=_ALL_WORKER_CASES)
    parser.add_argument("--candidate-archive", type=Path)
    parser.add_argument("--member-manifest", type=Path)
    parser.add_argument("--r225-contract", type=Path, default=ROOT / R225_TYPED_CONTRACT_MEMBER)
    parser.add_argument("--r236-contract", type=Path, default=ROOT / "state/canonical-libcg-r236.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-timeout-seconds", type=_positive_seconds, default=12.0)
    args = parser.parse_args(argv)
    if args.fake_child:
        if args.child_fd is None or args.fault_class is None:
            parser.error("--fake-child requires --child-fd and --fault-class")
    elif args.worker:
        if args.stage is None or args.case is None:
            parser.error("--worker requires --stage and --case")
    else:
        for name in ("stage", "candidate_archive", "member_manifest", "output"):
            if getattr(args, name) is None:
                parser.error(f"--{name.replace('_', '-')} is required")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.fake_child:
        return _fake_child_main(child_fd=int(args.child_fd), fault_class=str(args.fault_class))
    if args.worker:
        return _worker_main(args)
    try:
        result = run_fault_harness(
            stage=args.stage,
            candidate_archive=args.candidate_archive,
            member_manifest=args.member_manifest,
            output=args.output,
            worker_timeout_seconds=float(args.worker_timeout_seconds),
            r225_contract=args.r225_contract,
            r236_contract=args.r236_contract,
        )
    except Exception as exc:  # noqa: BLE001 - CLI must emit a bounded failure record
        print(
            FINAL_PREFIX
            + json.dumps(
                {
                    "schema": SCHEMA,
                    "status": "failed_closed",
                    "failure": {"type": type(exc).__name__, "message": str(exc)},
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 1
    print(
        FINAL_PREFIX
        + json.dumps(
            {
                "schema": SCHEMA,
                "status": "passed",
                "receipt": str(args.output.expanduser().resolve()),
                "fault_classes_covered": result["fault_classes_covered"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - invoked as an owned child.
    raise SystemExit(main())
