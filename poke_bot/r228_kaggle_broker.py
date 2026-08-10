"""Crash-contained subprocess broker for the r228 Kaggle search runtime.

The r228 eight-lane runtime intentionally exercises raw ``libcg`` Search
handles.  A native Search call can block forever or terminate the process, so
it must never run in the competition-agent controller itself.  This module
keeps the controller in one process and starts a fresh Python interpreter for
the native runtime.  The two processes communicate only through newline JSON
on a Unix socket; no Python multiprocessing, forked CUDA state, model object,
or CUDA tensor crosses the boundary.

The public controller is deliberately small and package-local.  It is intended
to be used by the r228 submission entrypoint, not by the BO1000 fleet.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import select
import socket
import subprocess
import sys
import time
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA = "poke_bot.r228_kaggle_subprocess_broker/v1"
COMPLETE_ACTION_CAP = 65_536
_MAX_MESSAGE_BYTES = 32 * 1024 * 1024
_MAX_PROGRESS_EVENTS = 256


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
    upper = min(int(raw_max), len(options))
    normalized = list(action)
    return (
        lower <= len(normalized) <= upper
        and len(set(normalized)) == len(normalized)
        and all(0 <= int(index) < len(options) for index in normalized)
    )


def _complete_legal_actions(obs: Mapping[str, Any]) -> set[tuple[int, ...]]:
    """Materialize the bounded complete ordered action set for child validation."""

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
    return {tuple(int(item) for item in action) for action in actions}


class _JsonSocket:
    """Small newline-JSON stream helper with no reader or feeder thread."""

    def __init__(self, sock: socket.socket, *, nonblocking: bool) -> None:
        self.sock = sock
        self.sock.setblocking(not nonblocking)
        self._buffer = bytearray()
        self._messages: deque[dict[str, Any]] = deque()

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass

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
                raise R228BrokerError("broker socket closed", code="child_socket_closed")
            self._buffer.extend(data)
            if len(self._buffer) > _MAX_MESSAGE_BYTES:
                raise R228BrokerError(
                    "broker response exceeds message limit", code="response_too_large"
                )
            self._decode_complete_lines()
        while self._messages:
            messages.append(self._messages.popleft())
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
    ) -> None:
        self.stage = Path(stage).expanduser().resolve()
        if not self.stage.is_dir() or self.stage.is_symlink():
            raise ValueError("broker stage must be a physical directory")
        self.action_timeout_seconds = _positive_seconds(
            action_timeout_seconds
            if action_timeout_seconds is not None
            else os.environ.get("POKEBOT_R228_BROKER_ACTION_TIMEOUT_SECONDS"),
            fallback=float(os.environ.get("POKEBOT_R228_DECISION_SECONDS", "8.0")),
            label="action_timeout_seconds",
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
            "last_fault": self.last_fault,
            "progress_by_lane": _json_copy(self._progress_by_lane),
            "progress_event_count": len(self._progress_events),
            "complete_action_cap": COMPLETE_ACTION_CAP,
        }

    def _record_progress(self, message: Mapping[str, Any]) -> None:
        payload = message.get("payload")
        normalized = dict(payload) if isinstance(payload, Mapping) else {}
        normalized["type"] = str(message.get("type") or "progress")
        request_id = message.get("request_id")
        if isinstance(request_id, int) and not isinstance(request_id, bool):
            normalized["request_id"] = int(request_id)
        normalized["observed_monotonic"] = time.monotonic()
        self._progress_events.append(normalized)
        lane = normalized.get("lane_id", normalized.get("lane"))
        if isinstance(lane, int) and not isinstance(lane, bool) and 0 <= lane < 8:
            self._progress_by_lane[str(lane)] = dict(normalized)

    def _new_fault(
        self,
        *,
        code: str,
        message: str,
        request_id: int | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        child = self._child
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "kind": "r228_broker_fault",
            "code": str(code),
            "message": str(message),
            "decision_count": self._decision_count,
            "observed_monotonic": time.monotonic(),
            "child_pid": int(child.pid) if child is not None else None,
            "progress_by_lane": _json_copy(self._progress_by_lane),
        }
        if request_id is not None:
            payload["request_id"] = int(request_id)
        if detail:
            payload["detail"] = _json_copy(dict(detail))
        self._last_fault = payload
        return payload

    def _dispose_child(self) -> None:
        """Reap only this broker's exact Popen child with bounded waits."""

        channel, child = self._channel, self._child
        self._channel = None
        self._child = None
        self._child_history_count = 0
        if channel is not None:
            channel.close()
        if child is None:
            return
        if child.poll() is not None:
            return
        # ``Popen.terminate`` and ``Popen.kill`` address the exact PID created
        # above.  We deliberately do not signal a process group or any session.
        try:
            child.terminate()
        except ProcessLookupError:
            return
        try:
            child.wait(timeout=self.reap_grace_seconds)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            child.kill()
        except ProcessLookupError:
            return
        try:
            child.wait(timeout=self.reap_grace_seconds)
        except subprocess.TimeoutExpired:
            # Do not convert a bounded reap into an unbounded parent wait.  The
            # child remains unreferenced and cannot receive future requests.
            pass

    def _child_alive(self) -> bool:
        return (
            self._child is not None
            and self._channel is not None
            and self._child.poll() is None
        )

    def _start_child(self, *, deadline: float) -> None:
        if self._closed:
            raise R228BrokerError("broker is closed", code="broker_closed")
        if self._child_alive():
            return
        self._dispose_child()
        parent_sock, child_sock = socket.socketpair()
        child_fd = child_sock.detach()
        os.set_inheritable(child_fd, True)
        env = dict(os.environ)
        stage_text = str(self.stage)
        inherited_path = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            stage_text if not inherited_path else stage_text + os.pathsep + inherited_path
        )
        # The child has eight native CPU lanes.  Leaving an inherited BLAS pool
        # at dozens of threads can starve those lanes on Kaggle's 16-vCPU host.
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
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
                stdout=None,
                stderr=None,
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
        except Exception:
            self._dispose_child()
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
            if self._child.poll() is not None:
                raise R228BrokerError(
                    f"broker child exited with code {self._child.returncode}",
                    code="child_exited",
                    detail={"returncode": self._child.returncode},
                )
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

    def _sync_child(self, *, deadline: float) -> None:
        if not self._child_alive():
            raise R228BrokerError("broker child is absent", code="child_absent")
        pending = self._journal[self._child_history_count :]
        if not pending:
            return
        request_id = self._next_request_id
        self._next_request_id += 1
        self._send(
            {
                "schema": SCHEMA,
                "type": "sync",
                "request_id": request_id,
                "events": _json_copy(pending),
            },
            deadline=deadline,
        )
        reply = self._wait_for(
            request_id=request_id, expected_type="synced", deadline=deadline
        )
        if int(reply.get("applied", -1)) != len(pending):
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
        self._dispose_child()
        return fault

    def begin_game(self) -> None:
        """Start a fresh native broker early, before the first branch prompt."""

        if self._closed:
            raise R228BrokerError("broker is closed", code="broker_closed")
        self._dispose_child()
        self._journal.clear()
        self._child_history_count = 0
        self._decision_count = 0
        self._degraded = False
        self._last_fault = None
        self._progress_events.clear()
        self._progress_by_lane.clear()
        deadline = time.monotonic() + self.startup_timeout_seconds
        try:
            self._start_child(deadline=deadline)
        except Exception as exc:  # Caller can still play direct policy.
            self._degrade(exc=exc, code="startup_failure")

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
        deadline = time.monotonic() + min(
            self.action_timeout_seconds, self.startup_timeout_seconds
        )
        request_id = self._next_request_id
        self._next_request_id += 1
        try:
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
            self._degrade(exc=exc, code="direct_note_failure")

    def select(
        self, obs: Mapping[str, Any], direct_action: Sequence[int]
    ) -> tuple[list[int], dict[str, Any] | None, dict[str, Any] | None]:
        """Attempt one MCTS decision or return the already supplied direct action.

        The method's deadline includes IPC, child synchronization, search, and
        result validation.  On every fault it returns ``direct_action`` and
        disables MCTS for the remainder of the game.
        """

        if not isinstance(obs, Mapping):
            raise ValueError("selection observation must be a mapping")
        direct = _as_action(list(direct_action), label="direct_action")
        deadline = time.monotonic() + self.action_timeout_seconds
        request_id: int | None = None
        try:
            legal = _complete_legal_actions(obs)
            if tuple(direct) not in legal:
                raise R228BrokerError(
                    "supplied direct action is outside the complete legal order",
                    code="illegal_direct_action",
                )
            if self._closed:
                raise R228BrokerError("broker is closed", code="broker_closed")
            if self._degraded:
                raise R228BrokerError(
                    "broker was already degraded for this game", code="already_degraded"
                )
            if not self._child_alive():
                self._start_child(deadline=deadline)
            self._sync_child(deadline=deadline)
            request_id = self._next_request_id
            self._next_request_id += 1
            # Leave a tiny parent-side reap window inside the hard action limit.
            child_seconds = max(0.05, deadline - time.monotonic() - self.reap_grace_seconds)
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
            receipt_action = receipt_payload.get("selected_action")
            if receipt_action != selected:
                raise R228BrokerError(
                    "broker receipt action differs from returned action",
                    code="receipt_action_mismatch",
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
            )
            self._append_journal(obs, direct)
            return direct, None, fault

    def close(self) -> None:
        """Dispose only the owned broker child and permanently disable this object."""

        if self._closed:
            return
        self._closed = True
        self._dispose_child()


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
    direct = _child_load_direct(stage)
    deck, model, policy = direct._ensure_runtime()
    from poke_bot.r228_kaggle_async_runtime import R228AsyncGameplay

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
            return R228AsyncGameplay(**kwargs, progress_callback=callback)
        except TypeError as exc:
            if "progress_callback" not in str(exc):
                raise
    # Compatibility with the current rough runtime.  The follow-up runtime
    # implementation consumes this attribute and emits per-lane progress.
    runtime = R228AsyncGameplay(**kwargs)
    setattr(runtime, "progress_callback", callback)
    return runtime


def _child_commit_action(runtime: Any, event: Mapping[str, Any]) -> None:
    observation = event.get("observation")
    action = event.get("action")
    if not isinstance(observation, Mapping):
        raise R228BrokerError("journal observation is malformed", code="malformed_journal")
    actual = _as_action(action, label="journal action")
    if not _syntactically_legal(observation, actual):
        raise R228BrokerError("journal action is illegal", code="illegal_journal_action")
    from poke_bot import features

    policy = runtime.policy
    forced = features.forced_go_first_action(dict(observation))
    if forced is not None:
        if list(forced) != actual:
            raise R228BrokerError(
                "journal turn-order action differs from forced choice",
                code="turn_order_journal_mismatch",
            )
        policy._record_go_first(dict(observation), actual)
        return
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
        channel.send(
            {
                "schema": SCHEMA,
                "type": "ready",
                "payload": {"pid": os.getpid(), "stage": str(stage.resolve())},
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
                    prior_timeout = os.environ.get("POKEBOT_R228_DECISION_SECONDS")
                    os.environ["POKEBOT_R228_DECISION_SECONDS"] = f"{timeout:.6f}"
                    try:
                        action = list(runtime.select(dict(observation)))
                    finally:
                        if prior_timeout is None:
                            os.environ.pop("POKEBOT_R228_DECISION_SECONDS", None)
                        else:
                            os.environ["POKEBOT_R228_DECISION_SECONDS"] = prior_timeout
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
    "R228BrokerError",
    "SCHEMA",
]
