"""Owned bounded child-process transport for tactical sequence transitions."""

from __future__ import annotations

import multiprocessing as mp
import time
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any, Protocol

from .tactical_sequence_planner import (
    Action,
    TacticalSearchState,
    TacticalSequenceError,
    TacticalTransition,
)


class TacticalChildWorker(Protocol):
    def advance(
        self, state: TacticalSearchState, action: Action
    ) -> TacticalTransition: ...

    def close(self) -> None: ...


WorkerFactory = Callable[[], TacticalChildWorker]


def _child_main(connection: Connection, factory: WorkerFactory) -> None:
    worker: TacticalChildWorker | None = None
    try:
        worker = factory()
        connection.send({"kind": "ready"})
        while True:
            message = connection.recv()
            kind = message.get("kind") if isinstance(message, dict) else None
            if kind == "close":
                connection.send({"kind": "closed"})
                return
            if kind != "advance":
                raise TacticalSequenceError("unknown tactical child command")
            request_id = int(message["request_id"])
            try:
                result = worker.advance(message["state"], message["action"])
                connection.send(
                    {"kind": "result", "request_id": request_id, "result": result}
                )
            except BaseException as exc:  # child converts faults to typed rows
                connection.send(
                    {
                        "kind": "fault",
                        "request_id": request_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
    except (EOFError, BrokenPipeError):
        return
    finally:
        if worker is not None:
            try:
                worker.close()
            except BaseException:
                pass
        connection.close()


@dataclass(frozen=True, slots=True)
class TacticalChildReceipt:
    child_pid: int | None
    requests: int
    completed: int
    faults: int
    timeouts: int
    bounded_reap_attempted: bool
    bounded_reap_succeeded: bool
    closed: bool


class OwnedProcessTacticalBackend:
    """Persistent child transport that reaps only the process it created."""

    isolation_mode = "owned_bounded_child"

    def __init__(
        self,
        factory: WorkerFactory,
        *,
        startup_seconds: float = 5.0,
        cleanup_seconds: float = 1.0,
        start_method: str = "spawn",
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if startup_seconds <= 0.0 or cleanup_seconds <= 0.0:
            raise TacticalSequenceError("child startup/cleanup bounds must be positive")
        self._monotonic = monotonic
        self._cleanup_seconds = float(cleanup_seconds)
        self._request_id = 0
        self._requests = 0
        self._completed = 0
        self._faults = 0
        self._timeouts = 0
        self._bounded_reap_attempted = False
        self._bounded_reap_succeeded = False
        self._closed = False
        context = mp.get_context(start_method)
        parent, child = context.Pipe(duplex=True)
        self._connection = parent
        self._process = context.Process(
            target=_child_main,
            args=(child, factory),
            name="poke-tactical-sequence-child",
            daemon=True,
        )
        self._process.start()
        child.close()
        if not parent.poll(float(startup_seconds)):
            self._timeouts += 1
            self._bounded_reap()
            raise TacticalSequenceError("tactical child startup deadline exceeded")
        try:
            ready = parent.recv()
        except (EOFError, BrokenPipeError) as exc:
            self._faults += 1
            self._bounded_reap()
            raise TacticalSequenceError("tactical child exited during startup") from exc
        if ready != {"kind": "ready"}:
            self._faults += 1
            self._bounded_reap()
            raise TacticalSequenceError("tactical child returned malformed readiness")

    @property
    def child_pid(self) -> int | None:
        return self._process.pid

    @property
    def receipt(self) -> TacticalChildReceipt:
        return TacticalChildReceipt(
            child_pid=self.child_pid,
            requests=self._requests,
            completed=self._completed,
            faults=self._faults,
            timeouts=self._timeouts,
            bounded_reap_attempted=self._bounded_reap_attempted,
            bounded_reap_succeeded=self._bounded_reap_succeeded,
            closed=self._closed,
        )

    def _bounded_reap(self) -> None:
        self._bounded_reap_attempted = True
        if self._process.is_alive():
            self._process.terminate()
        self._process.join(timeout=self._cleanup_seconds)
        if self._process.is_alive():
            # kill() still targets only the exact child created above.  It is
            # never a process group, session, managed service, or PID search.
            self._process.kill()
            self._process.join(timeout=self._cleanup_seconds)
        self._bounded_reap_succeeded = not self._process.is_alive()
        self._closed = True
        self._connection.close()

    def advance(
        self,
        state: TacticalSearchState,
        action: Action,
        *,
        deadline_monotonic: float,
    ) -> TacticalTransition:
        if self._closed or not self._process.is_alive():
            raise TacticalSequenceError("tactical child is not available")
        remaining = float(deadline_monotonic) - self._monotonic()
        if remaining <= 0.0:
            self._timeouts += 1
            self._bounded_reap()
            raise TacticalSequenceError("tactical transition deadline already expired")
        self._request_id += 1
        request_id = self._request_id
        self._requests += 1
        try:
            self._connection.send(
                {
                    "kind": "advance",
                    "request_id": request_id,
                    "state": state,
                    "action": action,
                }
            )
        except (BrokenPipeError, EOFError) as exc:
            self._faults += 1
            self._bounded_reap()
            raise TacticalSequenceError("tactical child command transport failed") from exc
        if not self._connection.poll(remaining):
            self._timeouts += 1
            self._bounded_reap()
            raise TacticalSequenceError("tactical transition deadline exceeded")
        try:
            response: Any = self._connection.recv()
        except (EOFError, BrokenPipeError) as exc:
            self._faults += 1
            self._bounded_reap()
            raise TacticalSequenceError("tactical child exited without a response") from exc
        if not isinstance(response, dict) or response.get("request_id") != request_id:
            self._faults += 1
            self._bounded_reap()
            raise TacticalSequenceError("tactical child returned a malformed response")
        if response.get("kind") == "fault":
            self._faults += 1
            self._bounded_reap()
            raise TacticalSequenceError(
                f"tactical child fault {response.get('error_type')}: "
                f"{response.get('error')}"
            )
        result = response.get("result")
        if response.get("kind") != "result" or not isinstance(
            result, TacticalTransition
        ):
            self._faults += 1
            self._bounded_reap()
            raise TacticalSequenceError("tactical child result has the wrong type")
        self._completed += 1
        return result

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._process.is_alive():
                self._connection.send({"kind": "close"})
                if self._connection.poll(self._cleanup_seconds):
                    response = self._connection.recv()
                    if response == {"kind": "closed"}:
                        self._process.join(timeout=self._cleanup_seconds)
        except (EOFError, BrokenPipeError):
            pass
        if self._process.is_alive():
            self._bounded_reap()
        else:
            self._bounded_reap_succeeded = True
            self._closed = True
            self._connection.close()

    def __enter__(self) -> "OwnedProcessTacticalBackend":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = [
    "OwnedProcessTacticalBackend",
    "TacticalChildReceipt",
    "TacticalChildWorker",
]
