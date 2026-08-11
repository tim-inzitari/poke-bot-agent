"""BO-only bounded recovery wrapper for serial process-owned MCTS."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .r228_async_shared_tree_queue import AsyncEightWorkerError
from .r253_restarting_serial_mcts import R253RestartingSerialMCTS


SCHEMA = "poke_bot.r250_serial_process_lane_recovery/v1"
MAX_ATTEMPTS = 2


class R250SerialRecoveryExhausted(AsyncEightWorkerError):
    """Both complete, fresh serial attempts failed at a recoverable boundary."""

    def __init__(self, message: str, *, telemetry: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.telemetry = dict(telemetry)


def _root_fingerprint(
    observation: Mapping[str, Any], actions: Sequence[Sequence[int]]
) -> str:
    encoded = json.dumps(
        {
            "observation": observation,
            "complete_ordered_legal_actions": [
                [int(value) for value in action] for action in actions
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _recoverable(exc: Exception, new_faults: Sequence[Mapping[str, Any]]) -> bool:
    if new_faults:
        return True
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "contained_native_lane_fault",
            "decision deadline expired before one arena opened",
            "searchbegin failed",
            "searchstep failed",
            "native lane",
        )
    )


class R250RecoveringSerialTree:
    """Keep one tree per attempt and replace the serial lane after a fault.

    A failed attempt's tree is discarded because a replacement native handle
    can sample a different hidden/random world.  The same exact root is retried
    once with a fresh process-owned handle.
    """

    def __init__(
        self,
        *,
        arena_factory: Callable[[int], Any],
        make_packet: Callable[[int, Any], Any],
        evaluate_batch: Callable[[Sequence[Any]], Sequence[Any]],
        puct_c: float = 1.25,
        coalesce_seconds: float = 0.001,
        max_rollouts: int | None = None,
    ) -> None:
        self._arena_factory = arena_factory
        self._core_kwargs = {
            "make_packet": make_packet,
            "evaluate_batch": evaluate_batch,
            "puct_c": float(puct_c),
            "coalesce_seconds": float(coalesce_seconds),
            "max_rollouts": int(
                os.environ.get("POKEBOT_R253_MAX_ROLLOUTS", "1000")
                if max_rollouts is None
                else max_rollouts
            ),
        }
        self._core: R253RestartingSerialMCTS | None = None
        self._lanes: list[Any] = []
        self.last_decision_recovery: dict[str, Any] = {
            "schema": SCHEMA,
            "serial_lane_count": 1,
            "attempt_count": 0,
            "recovered_search": False,
            "exhausted_direct_fallback": False,
            "attempts": [],
        }
        self.total_decisions = 0
        self.total_recovered_searches = 0
        self.total_exhausted_fallbacks = 0
        self.total_lane_faults = 0
        self._new_core()

    def _new_core(self) -> None:
        lanes: list[Any] = []

        def tracked_factory(lane_id: int) -> Any:
            lane = self._arena_factory(lane_id)
            lanes.append(lane)
            return lane

        core = R253RestartingSerialMCTS(
            arena_factory=tracked_factory, **self._core_kwargs
        )
        if len(lanes) != 1:
            try:
                core.close()
            finally:
                raise AsyncEightWorkerError(
                    f"r250 serial tree opened {len(lanes)} native lanes instead of one"
                )
        self._core = core
        self._lanes = lanes

    @staticmethod
    def _lane_snapshot(lanes: Sequence[Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for lane in lanes:
            snapshot = getattr(lane, "telemetry_snapshot", None)
            if callable(snapshot):
                try:
                    row = snapshot()
                except Exception as exc:
                    row = {
                        "lane_id": getattr(lane, "lane_id", None),
                        "telemetry_error": f"{type(exc).__name__}: {exc}",
                    }
            else:
                row = {"lane_id": getattr(lane, "lane_id", None)}
            rows.append(dict(row))
        return rows

    def _close_core(self) -> str | None:
        core, self._core = self._core, None
        if core is None:
            return None
        try:
            core.close()
            return None
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"

    def _replace_core(self) -> str | None:
        close_error = self._close_core()
        self._lanes = []
        try:
            self._new_core()
        except Exception as exc:
            create_error = f"{type(exc).__name__}: {exc}"
            return (
                create_error
                if close_error is None
                else f"close={close_error}; create={create_error}"
            )
        return close_error

    def run_decision(self, **kwargs: Any) -> Any:
        if self._core is None:
            raise AsyncEightWorkerError("r250 serial tree is closed")
        observation = kwargs.get("root_observation")
        actions = kwargs.get("root_actions")
        if not isinstance(observation, Mapping) or not isinstance(actions, Sequence):
            raise AsyncEightWorkerError("r250 decision lacks a bindable root")
        initial_deadline = float(kwargs.get("deadline_monotonic"))
        attempt_budget = max(0.25, initial_deadline - time.monotonic())
        started = time.monotonic()
        attempts: list[dict[str, Any]] = []
        root_sha = _root_fingerprint(observation, actions)

        for attempt_number in range(1, MAX_ATTEMPTS + 1):
            core = self._core
            if core is None:
                raise AsyncEightWorkerError("r250 serial tree disappeared")
            lane_fault_offsets = [
                len(getattr(lane, "faults", ())) for lane in self._lanes
            ]
            attempt_started = time.monotonic()
            call_kwargs = dict(kwargs)
            call_kwargs["deadline_monotonic"] = (
                initial_deadline
                if attempt_number == 1
                else time.monotonic() + attempt_budget
            )
            try:
                receipt = core.run_decision(**call_kwargs)
            except Exception as exc:
                snapshots = self._lane_snapshot(self._lanes)
                new_faults = []
                for index, row in enumerate(snapshots):
                    faults = row.get("faults")
                    if isinstance(faults, list):
                        new_faults.extend(faults[lane_fault_offsets[index] :])
                recoverable = _recoverable(exc, new_faults)
                attempts.append(
                    {
                        "attempt": attempt_number,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "recoverable_lane_fault": recoverable,
                        "elapsed_seconds": time.monotonic() - attempt_started,
                        "lane_processes": snapshots,
                        "new_lane_faults": new_faults,
                    }
                )
                self.total_lane_faults += len(new_faults) or int(recoverable)
                if not recoverable:
                    self.last_decision_recovery = {
                        "schema": SCHEMA,
                        "serial_lane_count": 1,
                        "root_fingerprint": root_sha,
                        "attempt_count": attempt_number,
                        "recovered_search": False,
                        "exhausted_direct_fallback": False,
                        "attempts": attempts,
                        "recovery_elapsed_seconds": time.monotonic() - started,
                    }
                    raise
                replacement_error = self._replace_core()
                attempts[-1]["replacement_error"] = replacement_error
                if attempt_number < MAX_ATTEMPTS and replacement_error is None:
                    continue
                telemetry = {
                    "schema": SCHEMA,
                    "serial_lane_count": 1,
                    "root_fingerprint": root_sha,
                    "attempt_count": attempt_number,
                    "recovered_search": False,
                    "exhausted_direct_fallback": True,
                    "attempts": attempts,
                    "recovery_elapsed_seconds": time.monotonic() - started,
                }
                self.last_decision_recovery = telemetry
                self.total_decisions += 1
                self.total_exhausted_fallbacks += 1
                raise R250SerialRecoveryExhausted(
                    "r250 serial process-lane recovery exhausted",
                    telemetry=telemetry,
                ) from exc

            snapshots = self._lane_snapshot(self._lanes)
            new_faults = []
            for index, row in enumerate(snapshots):
                faults = row.get("faults")
                if isinstance(faults, list):
                    new_faults.extend(faults[lane_fault_offsets[index] :])
            attempts.append(
                {
                    "attempt": attempt_number,
                    "status": "complete",
                    "elapsed_seconds": time.monotonic() - attempt_started,
                    "lane_processes": snapshots,
                    "new_lane_faults": new_faults,
                }
            )
            recovered = attempt_number > 1
            telemetry = {
                "schema": SCHEMA,
                "serial_lane_count": 1,
                "root_fingerprint": root_sha,
                "attempt_count": attempt_number,
                "recovered_search": recovered,
                "exhausted_direct_fallback": False,
                "attempts": attempts,
                "recovery_elapsed_seconds": time.monotonic() - started,
            }
            self.last_decision_recovery = telemetry
            self.total_decisions += 1
            self.total_recovered_searches += int(recovered)
            self.total_lane_faults += len(new_faults)
            return receipt

        raise AssertionError("unreachable r250 attempt loop")

    def summary(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "serial_lane_count": 1,
            "total_decisions": self.total_decisions,
            "total_recovered_searches": self.total_recovered_searches,
            "total_exhausted_fallbacks": self.total_exhausted_fallbacks,
            "total_lane_faults": self.total_lane_faults,
            "max_attempts_per_decision": MAX_ATTEMPTS,
            "last_decision_recovery": dict(self.last_decision_recovery),
        }

    def close(self) -> None:
        error = self._close_core()
        self._lanes = []
        if error is not None:
            raise AsyncEightWorkerError(f"r250 serial-tree close failed: {error}")


__all__ = [
    "MAX_ATTEMPTS",
    "R250RecoveringSerialTree",
    "R250SerialRecoveryExhausted",
    "SCHEMA",
]
