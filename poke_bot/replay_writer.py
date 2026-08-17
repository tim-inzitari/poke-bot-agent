"""Bounded ordered replay writer with exactly-once crash recovery."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


class ReplayWriterError(RuntimeError):
    pass


@dataclass(frozen=True)
class _WriterCommand:
    action: str
    reason: str = ""


class OrderedReplayWriter:
    """Write unordered worker results in deterministic job-index order.

    A commit consists of replay bytes, a compact result journal, fsync of both,
    and finally an atomic offset checkpoint. Recovery truncates both streams to
    the last checkpoint, so a crash cannot duplicate or partially accept a job.
    """

    def __init__(
        self,
        replay_partial: Path,
        *,
        expected_jobs: int,
        queue_depth: int = 64,
        fsync_batch: int = 8,
    ) -> None:
        self.replay_partial = Path(replay_partial)
        self.expected_jobs = int(expected_jobs)
        self.queue_depth = max(1, int(queue_depth))
        self.fsync_batch = max(1, int(fsync_batch))
        self.journal_path = self.replay_partial.with_suffix(
            self.replay_partial.suffix + ".journal"
        )
        self.state_path = self.replay_partial.with_suffix(
            self.replay_partial.suffix + ".writer.json"
        )
        self.replay_partial.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load_or_create_state()
        self._next_index = int(self._state["next_index"])
        self._written_records = int(self._state["written_records"])
        self._queue: queue.Queue = queue.Queue(maxsize=self.queue_depth)
        self._pending: dict[int, tuple[Optional[str], dict[str, Any]]] = {}
        self._error: Optional[BaseException] = None
        self._closed = False
        self._aborted = False
        self._abort_reason: Optional[str] = None
        self._submitted: set[int] = set()
        self._queue_wait_total = 0.0
        self._queue_wait_max = 0.0
        self._max_queue_depth = 0
        self._started = time.perf_counter()
        self._replay = self.replay_partial.open("r+b")
        self._journal = self.journal_path.open("r+b")
        self._thread = threading.Thread(
            target=self._run, name="ordered-replay-writer", daemon=True
        )
        self._thread.start()

    def _load_or_create_state(self) -> dict[str, Any]:
        if self.state_path.is_file():
            state = json.loads(self.state_path.read_text())
            if int(state.get("expected_jobs", -1)) != self.expected_jobs:
                raise ReplayWriterError("writer expected-job count changed on resume")
            for path, key in (
                (self.replay_partial, "replay_offset"),
                (self.journal_path, "journal_offset"),
            ):
                if not path.is_file():
                    raise ReplayWriterError(f"missing writer recovery file {path}")
                with path.open("r+b") as handle:
                    handle.truncate(int(state[key]))
            return state
        if self.replay_partial.exists() or self.journal_path.exists():
            raise ReplayWriterError(
                "writer partial exists without an atomic recovery checkpoint"
            )
        self.replay_partial.touch()
        self.journal_path.touch()
        state = {
            "schema": 1,
            "expected_jobs": self.expected_jobs,
            "next_index": 0,
            "written_records": 0,
            "replay_offset": 0,
            "journal_offset": 0,
        }
        self._save_state(state)
        return state

    def _save_state(self, state: dict[str, Any]) -> None:
        tmp = self.state_path.with_suffix(
            self.state_path.suffix + f".tmp.{os.getpid()}"
        )
        tmp.write_text(json.dumps(state, sort_keys=True) + "\n")
        os.replace(tmp, self.state_path)

    @property
    def resume_index(self) -> int:
        return self._next_index

    @property
    def written_records(self) -> int:
        return self._written_records

    def submit(
        self,
        job_index: int,
        record_json: Optional[str],
        result_metadata: dict[str, Any],
        *,
        timeout: float = 30.0,
    ) -> bool:
        if self._closed:
            raise ReplayWriterError("submit after writer close")
        if self._error is not None:
            raise ReplayWriterError(f"writer failed: {self._error}")
        index = int(job_index)
        if not 0 <= index < self.expected_jobs:
            raise ReplayWriterError(f"job index out of range: {index}")
        if index < self._next_index:
            return False  # already durably committed before restart
        if index in self._submitted:
            raise ReplayWriterError(f"duplicate job submission: {index}")
        if record_json is not None:
            parsed = json.loads(record_json)
            if not isinstance(parsed, dict) or not parsed.get("episode_id"):
                raise ReplayWriterError("replay payload is not a game record")
        self._submitted.add(index)
        started = time.perf_counter()
        self._queue.put(
            (index, record_json, dict(result_metadata)), timeout=float(timeout)
        )
        waited = time.perf_counter() - started
        self._queue_wait_total += waited
        self._queue_wait_max = max(self._queue_wait_max, waited)
        self._max_queue_depth = max(self._max_queue_depth, self._queue.qsize())
        return True

    def _run(self) -> None:
        stopping = False
        aborting = False
        try:
            while not stopping:
                item = self._queue.get()
                if isinstance(item, _WriterCommand):
                    if item.action not in ("close", "abort"):
                        raise ReplayWriterError(
                            f"unknown writer command {item.action}"
                        )
                    stopping = True
                    aborting = item.action == "abort"
                else:
                    index, record_json, metadata = item
                    if index in self._pending:
                        raise ReplayWriterError(f"duplicate pending job {index}")
                    self._pending[index] = (record_json, metadata)
                self._drain_ready(force=stopping)
            if self._pending and not aborting:
                raise ReplayWriterError(
                    f"writer closed with ordering gaps before {min(self._pending)}"
                )
            if aborting:
                # Out-of-order rows after a missing job were never committed and
                # are intentionally discarded. Recovery resumes from the last
                # fsynced contiguous index without duplicates.
                self._pending.clear()
        except BaseException as exc:  # noqa: BLE001
            self._error = exc

    def _drain_ready(self, *, force: bool) -> None:
        batch: list[tuple[int, Optional[str], dict[str, Any]]] = []
        while self._next_index in self._pending:
            record_json, metadata = self._pending.pop(self._next_index)
            batch.append((self._next_index, record_json, metadata))
            self._next_index += 1
            if len(batch) >= self.fsync_batch:
                self._commit(batch)
                batch = []
        if batch and (force or self._queue.empty()):
            self._commit(batch)
        elif batch:
            # Do not advance durable ordering until this batch commits.
            self._next_index -= len(batch)
            for index, record_json, metadata in batch:
                self._pending[index] = (record_json, metadata)

    def _commit(
        self, batch: list[tuple[int, Optional[str], dict[str, Any]]]
    ) -> None:
        for index, record_json, metadata in batch:
            if record_json is not None:
                self._replay.seek(0, os.SEEK_END)
                self._replay.write((record_json + "\n").encode())
                self._written_records += 1
            journal = {
                "job_index": index,
                "record_written": record_json is not None,
                "record_sha256": (
                    hashlib.sha256(record_json.encode()).hexdigest()
                    if record_json is not None
                    else None
                ),
                "result": metadata,
            }
            self._journal.seek(0, os.SEEK_END)
            self._journal.write(
                (json.dumps(journal, separators=(",", ":")) + "\n").encode()
            )
        self._replay.flush()
        self._journal.flush()
        os.fsync(self._replay.fileno())
        os.fsync(self._journal.fileno())
        self._state = {
            "schema": 1,
            "expected_jobs": self.expected_jobs,
            "next_index": self._next_index,
            "written_records": self._written_records,
            "replay_offset": self._replay.seek(0, os.SEEK_END),
            "journal_offset": self._journal.seek(0, os.SEEK_END),
        }
        self._save_state(self._state)

    def close(self) -> dict[str, Any]:
        return self._finish(_WriterCommand("close"))

    def abort(self, reason: str) -> dict[str, Any]:
        """Stop without requiring all jobs; retain crash-resumable sidecars."""
        return self._finish(_WriterCommand("abort", str(reason)))

    def _finish(self, command: _WriterCommand) -> dict[str, Any]:
        if self._closed:
            return self.telemetry()
        self._closed = True
        self._aborted = command.action == "abort"
        self._abort_reason = command.reason if self._aborted else None
        self._queue.put(command)
        self._thread.join(timeout=120)
        if self._thread.is_alive():
            raise ReplayWriterError("writer thread did not stop")
        self._replay.close()
        self._journal.close()
        if self._error is not None:
            raise ReplayWriterError(f"writer failed: {self._error}")
        if self._aborted:
            self._state = {
                **self._state,
                "aborted_at": time.time(),
                "abort_reason": self._abort_reason,
            }
            self._save_state(self._state)
        elif self._next_index != self.expected_jobs:
            raise ReplayWriterError(
                f"writer committed {self._next_index}/{self.expected_jobs} jobs"
            )
        return self.telemetry()

    def telemetry(self) -> dict[str, Any]:
        elapsed = max(time.perf_counter() - self._started, 1e-9)
        return {
            "next_index": self._next_index,
            "expected_jobs": self.expected_jobs,
            "written_records": self._written_records,
            "aborted": self._aborted,
            "abort_reason": self._abort_reason,
            "queue_depth": self._queue.qsize(),
            "max_queue_depth": self._max_queue_depth,
            "queue_put_wait_ms_total": self._queue_wait_total * 1000.0,
            "queue_put_wait_ms_max": self._queue_wait_max * 1000.0,
            "jobs_per_s": self._next_index / elapsed,
            "records_per_s": self._written_records / elapsed,
        }

    def finalize(self, final_path: Path) -> None:
        if not self._closed:
            raise ReplayWriterError("close writer before finalize")
        if self._aborted or self._next_index != self.expected_jobs:
            raise ReplayWriterError("cannot finalize an aborted/incomplete writer")
        final = Path(final_path)
        if final.exists():
            raise ReplayWriterError(f"refusing to overwrite replay {final}")
        os.replace(self.replay_partial, final)

    def quarantine(self, suffix: str) -> Path:
        if not self._closed:
            raise ReplayWriterError("close writer before quarantine")
        destinations: list[Path] = []
        for source in (
            self.replay_partial,
            self.journal_path,
            self.state_path,
        ):
            if not source.exists():
                continue
            dest = source.with_name(source.name + suffix)
            os.replace(source, dest)
            destinations.append(dest)
        if not destinations:
            raise ReplayWriterError("no writer artifacts remain to quarantine")
        dest = destinations[0]
        return dest

