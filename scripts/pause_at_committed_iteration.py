#!/usr/bin/env python3
"""Stop a Pure-RL service at one exact append-only iteration boundary.

The watcher never edits a checkpoint or ledger. It waits until the immutable
commit and mutable loop pointer are byte-for-byte identical, verifies the
learner checkpoint checksum, then asks systemd to stop the owning service.
Uncommitted work that races the stop is intentionally left for the trainer's
normal recovery/quarantine path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "poke_bot.committed_iteration_pause/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish(path: Path, **values: Any) -> None:
    _atomic_json(
        path,
        {
            "schema": SCHEMA,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            **values,
        },
    )


def _systemctl(*args: str, timeout: float = 90.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["systemctl", "--user", *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )


def _service_value(unit: str, key: str) -> str:
    result = _systemctl("show", unit, "-p", key, "--value", timeout=15.0)
    if result.returncode:
        raise RuntimeError(
            f"cannot read systemd {key} for {unit}: {result.stdout.strip()}"
        )
    return result.stdout.strip()


def _boundary(
    run_dir: Path,
    completed_iteration: int,
) -> tuple[Path, dict[str, Any], Path, str] | None:
    next_iteration = completed_iteration + 1
    commit_path = run_dir / "commits" / f"iter_{completed_iteration:05d}.json"
    next_commit = run_dir / "commits" / f"iter_{next_iteration:05d}.json"
    if next_commit.exists():
        raise RuntimeError(
            f"iteration {next_iteration} already committed; refusing rollback"
        )
    if not commit_path.is_file():
        return None
    commit = _read_json(commit_path)
    loop_state = _read_json(run_dir / "loop_state.json")
    if (
        int(commit.get("last_completed_iteration", -1)) != completed_iteration
        or int(commit.get("next_iteration", -1)) != next_iteration
    ):
        raise RuntimeError("immutable commit does not describe the requested boundary")
    if loop_state != commit:
        previous_path = (
            run_dir / "commits" / f"iter_{completed_iteration - 1:05d}.json"
        )
        if previous_path.is_file() and loop_state == _read_json(previous_path):
            return None
        raise RuntimeError("loop state disagrees with the immutable boundary commit")
    learner = dict(commit.get("learner") or {})
    checkpoint = Path(str(learner.get("path") or "")).expanduser().resolve()
    digest = str(learner.get("digest") or "").lower()
    if (
        not checkpoint.is_file()
        or len(digest) != 71
        or not digest.startswith("sha256:")
        or _sha256(checkpoint) != digest
    ):
        raise RuntimeError("boundary learner checkpoint identity does not verify")
    return commit_path.resolve(), commit, checkpoint, digest


def pause(
    *,
    run_dir: Path,
    unit: str,
    completed_iteration: int,
    status_path: Path,
    poll_seconds: float,
    timeout_seconds: float,
    next_unit: str | None,
) -> None:
    run_dir = run_dir.expanduser().resolve()
    if _service_value(unit, "RefuseManualStop").strip().lower() != "no":
        raise RuntimeError("boundary watcher requires RefuseManualStop=no")
    if _service_value(unit, "ActiveState") not in {"active", "activating"}:
        raise RuntimeError("trainer is not active before the boundary watch")
    if next_unit and _service_value(next_unit, "LoadState") != "loaded":
        raise RuntimeError("boundary successor unit is not loaded")
    next_iteration = completed_iteration + 1
    _publish(
        status_path,
        status="waiting",
        unit=unit,
        run_dir=str(run_dir),
        completed_iteration=completed_iteration,
        next_iteration=next_iteration,
    )
    deadline = time.monotonic() + max(1.0, timeout_seconds)
    proof = None
    while time.monotonic() < deadline:
        proof = _boundary(run_dir, completed_iteration)
        if proof is not None:
            break
        if _service_value(unit, "ActiveState") not in {"active", "activating"}:
            raise RuntimeError("trainer stopped before the requested boundary")
        time.sleep(max(0.01, poll_seconds))
    if proof is None:
        raise RuntimeError("timed out waiting for the requested boundary")
    commit_path, commit, checkpoint, digest = proof
    _publish(
        status_path,
        status="stopping",
        unit=unit,
        commit=str(commit_path),
        commit_digest=_sha256(commit_path),
        checkpoint=str(checkpoint),
        checkpoint_digest=digest,
        completed_iteration=completed_iteration,
        next_iteration=next_iteration,
    )
    result = _systemctl("stop", unit)
    if result.returncode:
        raise RuntimeError(f"systemd refused boundary stop: {result.stdout.strip()}")
    stop_deadline = time.monotonic() + 90.0
    while time.monotonic() < stop_deadline:
        if (
            _service_value(unit, "ActiveState") in {"inactive", "failed"}
            and int(_service_value(unit, "MainPID") or 0) == 0
        ):
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("trainer did not stop at the committed boundary")
    verified = _boundary(run_dir, completed_iteration)
    if verified is None or verified[1] != commit:
        raise RuntimeError("boundary identity changed during service stop")
    runtime_path = run_dir / "iteration_runtime.json"
    raced_iteration = None
    if runtime_path.is_file():
        try:
            raced_iteration = int(_read_json(runtime_path).get("iteration", -1))
        except (TypeError, ValueError):
            raced_iteration = next_iteration
    raced = bool(raced_iteration is not None and raced_iteration >= next_iteration)
    paused_values = {
        "unit": unit,
        "commit": str(commit_path),
        "commit_digest": _sha256(commit_path),
        "checkpoint": str(checkpoint),
        "checkpoint_digest": digest,
        "completed_iteration": completed_iteration,
        "next_iteration": next_iteration,
        "uncommitted_next_iteration_started": raced,
        "recovery_required": raced,
        "service_active_state": _service_value(unit, "ActiveState"),
    }
    if next_unit:
        # Publish the final checksum-stable handoff receipt before asking
        # systemd to launch the successor.  The successor may start
        # immediately, so rewriting this receipt after start would race its
        # source-identity checksum.
        _publish(
            status_path,
            status="paused_successor_start_requested",
            **paused_values,
            successor_unit=next_unit,
            successor_start_requested=True,
        )
        started = _systemctl("start", "--no-block", next_unit)
        if started.returncode:
            raise RuntimeError(
                f"could not start boundary successor {next_unit}: "
                f"{started.stdout.strip()}"
            )
    else:
        _publish(
            status_path,
            status="paused",
            **paused_values,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--completed-iteration", type=int, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.02)
    parser.add_argument("--timeout-seconds", type=float, default=21600.0)
    parser.add_argument("--next-unit")
    args = parser.parse_args()
    if args.completed_iteration < 0:
        raise ValueError("--completed-iteration must be non-negative")
    pause(
        run_dir=args.run_dir,
        unit=args.unit,
        completed_iteration=args.completed_iteration,
        status_path=args.status.expanduser().resolve(),
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
        next_unit=args.next_unit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
