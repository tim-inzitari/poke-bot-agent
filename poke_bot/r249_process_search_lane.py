"""BO-only process boundary for one official-libcg simulator search lane.

The shared MCTS tree and frozen model remain in the evaluator process.  This
module moves only the uncancellable native Search ABI into a small, persistent
Python child.  If a native call wedges, the parent can reap that exact child;
it never attempts to cancel a Python thread trapped inside ``libcg``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import select
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any


SCHEMA = "poke_bot.r249_process_search_lane/v1"
MAX_MESSAGE_BYTES = 256 * 1024 * 1024
CANONICAL_NATIVE_SHA256 = {
    "libcg.so": "d16244a3157fc55c3314f08dcc7c5179168697d78c105b95c7debd556b764bb7",
    "libcg-arm64.so": "1670740b73fab46586fd25c0a1f96608ea75b1f39381d66a0b8d9486bea6d4a2",
    "libcg.dylib": "7a157f045d333f99d1996d49c12bdbdd148072a619af246385c7295518776e30",
    "cg.dll": "eae88634e26dc31d94150a4d8202fc9d32596b8c688ef67e14cb4088cd4d5771",
}
THREAD_CAPS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


class R249ProcessLaneError(RuntimeError):
    """A bounded lane IPC, child, or native operation failed."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "lane_error",
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.detail = dict(detail or {})


def _seconds(value: object, *, fallback: float, label: str) -> float:
    try:
        parsed = float(fallback if value is None else value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be positive and finite") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    return parsed


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False, separators=(",", ":")))
    except (TypeError, ValueError) as exc:
        raise R249ProcessLaneError(
            "lane payload is not JSON-native", code="non_json_payload"
        ) from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _LineChannel:
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
            raise R249ProcessLaneError(
                "lane message is not JSON encodable", code="non_json_message"
            ) from exc
        if len(encoded) > MAX_MESSAGE_BYTES:
            raise R249ProcessLaneError(
                "lane message exceeds the bounded IPC limit", code="message_too_large"
            )
        sent = 0
        while sent < len(encoded):
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise R249ProcessLaneError("lane send timed out", code="send_timeout")
            try:
                _readable, writable, errors = select.select(
                    [], [self.sock], [self.sock], remaining
                )
            except (OSError, ValueError) as exc:
                raise R249ProcessLaneError(
                    f"lane send select failed: {exc}", code="send_select_failed"
                ) from exc
            if errors or not writable:
                raise R249ProcessLaneError("lane send timed out", code="send_timeout")
            try:
                count = self.sock.send(encoded[sent:])
            except (BlockingIOError, InterruptedError):
                continue
            except OSError as exc:
                raise R249ProcessLaneError(
                    f"lane send failed: {exc}", code="send_failed"
                ) from exc
            if count <= 0:
                raise R249ProcessLaneError("lane socket closed", code="send_closed")
            sent += count

    def _decode(self) -> None:
        while True:
            try:
                end = self._buffer.index(0x0A)
            except ValueError:
                return
            raw = bytes(self._buffer[:end])
            del self._buffer[: end + 1]
            if not raw:
                continue
            try:
                message = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise R249ProcessLaneError(
                    "lane emitted malformed JSON", code="malformed_child_json"
                ) from exc
            if not isinstance(message, dict):
                raise R249ProcessLaneError(
                    "lane emitted a non-object", code="malformed_child_message"
                )
            self._messages.append(message)

    def receive_parent(self, *, deadline: float) -> dict[str, Any]:
        while not self._messages:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise R249ProcessLaneError(
                    "native lane response timed out", code="response_timeout"
                )
            try:
                readable, _writable, errors = select.select(
                    [self.sock], [], [self.sock], remaining
                )
            except (OSError, ValueError) as exc:
                raise R249ProcessLaneError(
                    f"lane receive select failed: {exc}", code="receive_select_failed"
                ) from exc
            if errors or not readable:
                raise R249ProcessLaneError(
                    "native lane response timed out", code="response_timeout"
                )
            try:
                data = self.sock.recv(64 * 1024)
            except (BlockingIOError, InterruptedError):
                continue
            except OSError as exc:
                raise R249ProcessLaneError(
                    f"lane receive failed: {exc}", code="receive_failed"
                ) from exc
            if not data:
                raise R249ProcessLaneError(
                    "native lane child closed its socket", code="child_socket_closed"
                )
            self._buffer.extend(data)
            if len(self._buffer) > MAX_MESSAGE_BYTES:
                raise R249ProcessLaneError(
                    "lane response exceeds the bounded IPC limit",
                    code="response_too_large",
                )
            self._decode()
        return self._messages.popleft()

    def receive_child(self) -> dict[str, Any] | None:
        while not self._messages:
            try:
                data = self.sock.recv(64 * 1024)
            except OSError:
                return None
            if not data:
                return None
            self._buffer.extend(data)
            if len(self._buffer) > MAX_MESSAGE_BYTES:
                raise R249ProcessLaneError(
                    "parent request exceeds the bounded IPC limit",
                    code="request_too_large",
                )
            self._decode()
        return self._messages.popleft()


class R249ProcessSearchLane:
    """Thread-affine proxy for one process-owned native Search handle."""

    def __init__(
        self,
        lane_id: int,
        *,
        stage: Path,
        startup_timeout_seconds: float | None = None,
        call_timeout_seconds: float | None = None,
        cleanup_timeout_seconds: float | None = None,
        reap_grace_seconds: float | None = None,
        child_module: str = "poke_bot.r249_process_search_lane",
    ) -> None:
        self.lane_id = int(lane_id)
        if self.lane_id not in (0, 1):
            raise ValueError("r249 lane_id must be 0 or 1")
        self.stage = Path(stage).resolve(strict=True)
        self.startup_timeout_seconds = _seconds(
            startup_timeout_seconds
            if startup_timeout_seconds is not None
            else os.environ.get("POKEBOT_R249_LANE_STARTUP_TIMEOUT_SECONDS"),
            fallback=45.0,
            label="startup_timeout_seconds",
        )
        self.call_timeout_seconds = _seconds(
            call_timeout_seconds
            if call_timeout_seconds is not None
            else os.environ.get("POKEBOT_R249_LANE_CALL_TIMEOUT_SECONDS"),
            fallback=10.0,
            label="call_timeout_seconds",
        )
        self.cleanup_timeout_seconds = _seconds(
            cleanup_timeout_seconds
            if cleanup_timeout_seconds is not None
            else os.environ.get("POKEBOT_R249_LANE_CLEANUP_TIMEOUT_SECONDS"),
            fallback=2.0,
            label="cleanup_timeout_seconds",
        )
        self.reap_grace_seconds = _seconds(
            reap_grace_seconds
            if reap_grace_seconds is not None
            else os.environ.get("POKEBOT_R249_LANE_REAP_GRACE_SECONDS"),
            fallback=0.5,
            label="reap_grace_seconds",
        )
        self.child_module = str(child_module).strip()
        if not self.child_module:
            raise ValueError("child_module must be non-empty")
        self._owner_thread_id = threading.get_ident()
        self._child: subprocess.Popen[bytes] | None = None
        self._channel: _LineChannel | None = None
        self._request_id = 1
        self._live_search_ids: set[int] = set()
        self._handle_identity: str | None = None
        self._child_identity: dict[str, Any] | None = None
        self._faults: list[dict[str, Any]] = []
        self._closed = False
        self._start_child()

    @property
    def owner_thread_id(self) -> int:
        return self._owner_thread_id

    @property
    def handle_identity(self) -> int | str:
        if self._handle_identity is None:
            raise R249ProcessLaneError(
                "lane has no proven AgentStart handle", code="missing_handle_identity"
            )
        return self._handle_identity

    @property
    def live_search_ids(self) -> frozenset[int]:
        return frozenset(self._live_search_ids)

    @property
    def faults(self) -> tuple[dict[str, Any], ...]:
        return tuple(_json_copy(self._faults))

    def telemetry_snapshot(self) -> dict[str, Any]:
        child = self._child
        return {
            "lane_id": self.lane_id,
            "owner_thread_id": self._owner_thread_id,
            "handle_identity": self._handle_identity,
            "child_identity": _json_copy(self._child_identity),
            "child_alive": bool(child is not None and child.poll() is None),
            "live_search_ids": sorted(self._live_search_ids),
            "faults": _json_copy(self._faults),
        }

    def _assert_owner(self) -> None:
        if threading.get_ident() != self._owner_thread_id:
            raise R249ProcessLaneError(
                "process lane was called from a non-owner thread",
                code="thread_affinity_violation",
            )

    def _dispose_child(self, *, reason: str) -> dict[str, Any]:
        channel, child = self._channel, self._child
        self._channel = None
        if channel is not None:
            channel.close()
        report: dict[str, Any] = {
            "reason": str(reason),
            "child_identity": _json_copy(self._child_identity),
            "term_sent": False,
            "kill_sent": False,
        }
        if child is None:
            report.update({"child_present": False, "reaped": True})
            return report
        report["child_present"] = True
        returncode = child.poll()
        if returncode is None:
            try:
                child.terminate()
                report["term_sent"] = True
            except ProcessLookupError:
                pass
            try:
                returncode = child.wait(timeout=self.reap_grace_seconds)
            except subprocess.TimeoutExpired:
                returncode = child.poll()
        if returncode is None:
            try:
                child.kill()
                report["kill_sent"] = True
            except ProcessLookupError:
                pass
            try:
                returncode = child.wait(timeout=self.reap_grace_seconds)
            except subprocess.TimeoutExpired:
                returncode = child.poll()
        report["returncode"] = returncode
        report["reaped"] = returncode is not None
        if returncode is None:
            raise R249ProcessLaneError(
                "owned native lane child survived TERM and KILL",
                code="child_unreaped",
                detail={"reap": report},
            )
        self._child = None
        self._child_identity = None
        self._live_search_ids.clear()
        return report

    def _record_fault(
        self, *, operation: str, exc: Exception, reap: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        payload = {
            "schema": SCHEMA,
            "lane_id": self.lane_id,
            "operation": str(operation),
            "code": (
                exc.code if isinstance(exc, R249ProcessLaneError) else "lane_exception"
            ),
            "error": f"{type(exc).__name__}: {exc}",
            "child_identity": _json_copy(self._child_identity),
            "reap": _json_copy(reap),
            "observed_monotonic": time.monotonic(),
        }
        self._faults.append(payload)
        return payload

    def _start_child(self) -> None:
        self._assert_owner()
        if self._closed:
            raise R249ProcessLaneError("lane is closed", code="lane_closed")
        if (
            self._child is not None
            and self._channel is not None
            and self._child.poll() is None
        ):
            return
        if self._child is not None or self._channel is not None:
            self._dispose_child(reason="replace_dead_child")
        parent_sock, child_sock = socket.socketpair()
        child_fd = child_sock.detach()
        os.set_inheritable(child_fd, True)
        env = dict(os.environ)
        stage_text = str(self.stage)
        existing = env.get("PYTHONPATH", "").strip()
        env["PYTHONPATH"] = stage_text if not existing else stage_text + os.pathsep + existing
        env["CG_LIB_PATH"] = stage_text
        env.update(THREAD_CAPS)
        try:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    self.child_module,
                    "--child-fd",
                    str(child_fd),
                    "--stage",
                    stage_text,
                    "--lane-id",
                    str(self.lane_id),
                ],
                cwd=stage_text,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                pass_fds=(child_fd,),
            )
        except OSError as exc:
            parent_sock.close()
            raise R249ProcessLaneError(
                f"native lane child could not start: {exc}", code="child_start_failed"
            ) from exc
        finally:
            try:
                os.close(child_fd)
            except OSError:
                pass
        self._child = child
        self._channel = _LineChannel(parent_sock, nonblocking=True)
        self._child_identity = {
            "pid": int(child.pid),
            "lane_id": self.lane_id,
            "started_monotonic": time.monotonic(),
        }
        try:
            ready = self._channel.receive_parent(
                deadline=time.monotonic() + self.startup_timeout_seconds
            )
            if ready.get("schema") != SCHEMA or ready.get("type") != "ready":
                raise R249ProcessLaneError(
                    "native lane child returned malformed readiness",
                    code="malformed_ready",
                )
            identity = ready.get("identity")
            if not isinstance(identity, Mapping):
                raise R249ProcessLaneError(
                    "native lane child omitted identity", code="missing_child_identity"
                )
            if identity.get("lane_id") != self.lane_id or identity.get("pid") != child.pid:
                raise R249ProcessLaneError(
                    "native lane child identity drifted", code="child_identity_drift"
                )
            member = str(identity.get("native_member") or "")
            expected = CANONICAL_NATIVE_SHA256.get(member)
            if expected is None or identity.get("native_sha256") != "sha256:" + expected:
                raise R249ProcessLaneError(
                    "native lane child loaded a non-canonical library",
                    code="native_identity_drift",
                )
            handle = identity.get("handle_identity")
            if not isinstance(handle, str) or not handle:
                raise R249ProcessLaneError(
                    "native lane child omitted its handle", code="missing_handle_identity"
                )
            self._handle_identity = handle
            self._child_identity.update(_json_copy(dict(identity)))
        except Exception as exc:
            reap = self._dispose_child(reason="startup_failure")
            self._record_fault(operation="startup", exc=exc, reap=reap)
            raise

    def _request(
        self, operation: str, payload: Mapping[str, Any], *, timeout_seconds: float
    ) -> dict[str, Any]:
        self._assert_owner()
        self._start_child()
        if self._channel is None or self._child is None:
            raise R249ProcessLaneError("lane child is absent", code="child_absent")
        request_id = self._request_id
        self._request_id += 1
        deadline = time.monotonic() + timeout_seconds
        try:
            self._channel.send(
                {
                    "schema": SCHEMA,
                    "type": operation,
                    "request_id": request_id,
                    **_json_copy(dict(payload)),
                },
                deadline=deadline,
            )
            response = self._channel.receive_parent(deadline=deadline)
            if (
                response.get("schema") != SCHEMA
                or response.get("request_id") != request_id
            ):
                raise R249ProcessLaneError(
                    "lane response identity drifted", code="response_identity_drift"
                )
            if response.get("type") == "error":
                raise R249ProcessLaneError(
                    str(response.get("message") or "native lane child failed"),
                    code=str(response.get("code") or "child_error"),
                )
            if response.get("type") != "result":
                raise R249ProcessLaneError(
                    "lane response type drifted", code="response_type_drift"
                )
            return response
        except Exception as exc:
            try:
                reap = self._dispose_child(reason=f"{operation}_failure")
            except Exception as reap_exc:
                self._record_fault(operation=operation, exc=exc, reap=None)
                raise reap_exc from exc
            fault = self._record_fault(operation=operation, exc=exc, reap=reap)
            raise R249ProcessLaneError(
                f"r249 lane {self.lane_id} {operation} failed: {exc}",
                code="contained_native_lane_fault",
                detail={"fault": fault},
            ) from exc

    @staticmethod
    def _state(response: Mapping[str, Any]) -> Any:
        state = response.get("state")
        if not isinstance(state, Mapping):
            raise R249ProcessLaneError(
                "native lane omitted search state", code="missing_search_state"
            )
        search_id = state.get("search_id")
        observation = state.get("observation")
        if isinstance(search_id, bool) or not isinstance(search_id, int):
            raise R249ProcessLaneError(
                "native lane returned an invalid SearchId", code="invalid_search_id"
            )
        if not isinstance(observation, Mapping):
            raise R249ProcessLaneError(
                "native lane returned an invalid observation", code="invalid_observation"
            )
        return SimpleNamespace(searchId=int(search_id), observation=dict(observation))

    def search_begin(
        self,
        obs_dict: Mapping[str, Any] | Any,
        search_inputs: Mapping[str, Sequence[int]],
        *,
        manual_coin: bool = True,
    ) -> Any:
        if not isinstance(obs_dict, Mapping):
            if is_dataclass(obs_dict):
                obs_dict = asdict(obs_dict)
            else:
                raise R249ProcessLaneError(
                    "SearchBegin observation is not serializable",
                    code="invalid_begin_observation",
                )
        response = self._request(
            "search_begin",
            {
                "observation": dict(obs_dict),
                "search_inputs": dict(search_inputs),
                "manual_coin": bool(manual_coin),
            },
            timeout_seconds=self.call_timeout_seconds,
        )
        state = self._state(response)
        self._live_search_ids = {int(state.searchId)}
        return state

    def search_step(self, search_id: int, select_action: Sequence[int]) -> Any:
        response = self._request(
            "search_step",
            {
                "search_id": int(search_id),
                "action": [int(value) for value in select_action],
            },
            timeout_seconds=self.call_timeout_seconds,
        )
        state = self._state(response)
        self._live_search_ids.add(int(state.searchId))
        return state

    def search_release(self, search_id: int) -> None:
        search_id = int(search_id)
        if self._child is None:
            self._live_search_ids.discard(search_id)
            return
        try:
            self._request(
                "search_release",
                {"search_id": search_id},
                timeout_seconds=self.cleanup_timeout_seconds,
            )
        except R249ProcessLaneError:
            # Reaping the process is the cleanup.  Let the coordinator receive
            # a close row instead of recreating the historical infinite wait.
            self._live_search_ids.clear()
            return
        self._live_search_ids.discard(search_id)

    def search_end(self) -> None:
        if self._child is None:
            self._live_search_ids.clear()
            return
        try:
            self._request(
                "search_end", {}, timeout_seconds=self.cleanup_timeout_seconds
            )
        except R249ProcessLaneError:
            self._live_search_ids.clear()
            return
        self._live_search_ids.clear()

    def close(self) -> None:
        self._assert_owner()
        if self._closed:
            return
        if self._child is not None:
            try:
                self._request(
                    "close", {}, timeout_seconds=self.cleanup_timeout_seconds
                )
            except R249ProcessLaneError:
                pass
        self._closed = True
        if self._child is not None or self._channel is not None:
            self._dispose_child(reason="lane_close")


def _native_identity(stage: Path, lane: Any) -> dict[str, Any]:
    import cg.sim as sim

    loaded = Path(str(getattr(sim.lib, "_name", ""))).resolve(strict=True)
    cg_root = (stage / "cg").resolve(strict=True)
    if loaded.parent != cg_root or loaded.name not in CANONICAL_NATIVE_SHA256:
        raise R249ProcessLaneError(
            "child native library is outside the sealed cg directory",
            code="native_path_drift",
        )
    digest = _sha256(loaded)
    if digest != CANONICAL_NATIVE_SHA256[loaded.name]:
        raise R249ProcessLaneError(
            "child native library digest drifted", code="native_digest_drift"
        )
    pid = os.getpid()
    raw_handle = lane.handle_identity
    return {
        "pid": pid,
        "lane_id": int(lane.lane_id),
        "platform": platform.system().lower(),
        "machine": platform.machine().lower(),
        "native_member": loaded.name,
        "native_sha256": "sha256:" + digest,
        "raw_handle_identity": raw_handle,
        # Pointer values may repeat in separate address spaces.  Process plus
        # raw handle is the opaque global identity of this AgentStart arena.
        "handle_identity": f"process:{pid}:handle:{raw_handle}",
    }


def _send_error(
    channel: _LineChannel, *, request_id: object, exc: Exception
) -> None:
    try:
        channel.send(
            {
                "schema": SCHEMA,
                "type": "error",
                "request_id": request_id,
                "code": (
                    exc.code if isinstance(exc, R249ProcessLaneError) else "child_exception"
                ),
                "message": f"{type(exc).__name__}: {exc}",
            },
            deadline=time.monotonic() + 1.0,
        )
    except Exception:
        pass


def _state_payload(state: Any) -> dict[str, Any]:
    observation = state.observation
    if is_dataclass(observation):
        observation = asdict(observation)
    if not isinstance(observation, Mapping):
        raise R249ProcessLaneError(
            "native Search state observation is not serializable",
            code="invalid_native_observation",
        )
    return {"search_id": int(state.searchId), "observation": dict(observation)}


def _child_main(*, child_fd: int, stage: Path, lane_id: int) -> int:
    sock = socket.socket(fileno=int(child_fd))
    channel = _LineChannel(sock, nonblocking=False)
    try:
        stage = stage.resolve(strict=True)
        os.chdir(stage)
        stage_text = str(stage)
        if stage_text not in sys.path:
            sys.path.insert(0, stage_text)
        from poke_bot.r225_stock_native_lane import (
            R225StockNativeSearchLane,
            prewarm_stock_cg,
        )

        api, sim = prewarm_stock_cg()
        lane = R225StockNativeSearchLane(lane_id, lib=sim.lib, api_module=api)
        identity = _native_identity(stage, lane)
        channel.send(
            {"schema": SCHEMA, "type": "ready", "identity": identity},
            deadline=time.monotonic() + 5.0,
        )
        while True:
            request = channel.receive_child()
            if request is None:
                return 0
            request_id = request.get("request_id")
            try:
                if request.get("schema") != SCHEMA:
                    raise R249ProcessLaneError(
                        "parent lane schema drifted", code="parent_schema_drift"
                    )
                kind = str(request.get("type") or "")
                payload: dict[str, Any] = {}
                if kind == "search_begin":
                    observation = request.get("observation")
                    search_inputs = request.get("search_inputs")
                    if not isinstance(observation, Mapping) or not isinstance(
                        search_inputs, Mapping
                    ):
                        raise R249ProcessLaneError(
                            "malformed SearchBegin request", code="malformed_begin"
                        )
                    state = lane.search_begin(
                        observation,
                        search_inputs,
                        manual_coin=bool(request.get("manual_coin", True)),
                    )
                    payload["state"] = _state_payload(state)
                elif kind == "search_step":
                    action = request.get("action")
                    if not isinstance(action, list):
                        raise R249ProcessLaneError(
                            "malformed SearchStep action", code="malformed_step"
                        )
                    state = lane.search_step(
                        int(request.get("search_id")),
                        [int(value) for value in action],
                    )
                    payload["state"] = _state_payload(state)
                elif kind == "search_release":
                    lane.search_release(int(request.get("search_id")))
                    payload["released"] = True
                elif kind == "search_end":
                    lane.search_end()
                    payload["ended"] = True
                elif kind == "close":
                    channel.send(
                        {
                            "schema": SCHEMA,
                            "type": "result",
                            "request_id": request_id,
                            "closed": True,
                        },
                        deadline=time.monotonic() + 1.0,
                    )
                    return 0
                else:
                    raise R249ProcessLaneError(
                        f"unknown lane operation {kind!r}", code="unknown_operation"
                    )
                channel.send(
                    {
                        "schema": SCHEMA,
                        "type": "result",
                        "request_id": request_id,
                        **payload,
                    },
                    deadline=time.monotonic() + 2.0,
                )
            except Exception as exc:
                _send_error(channel, request_id=request_id, exc=exc)
    except Exception as exc:
        _send_error(channel, request_id=None, exc=exc)
        return 2
    finally:
        channel.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child-fd", type=int, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--lane-id", type=int, choices=(0, 1), required=True)
    args = parser.parse_args(argv)
    return _child_main(
        child_fd=args.child_fd, stage=args.stage, lane_id=args.lane_id
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANONICAL_NATIVE_SHA256",
    "R249ProcessLaneError",
    "R249ProcessSearchLane",
    "SCHEMA",
]
