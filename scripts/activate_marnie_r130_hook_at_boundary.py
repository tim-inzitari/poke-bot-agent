#!/usr/bin/env python3
"""Load the revision-130 Marnie boundary hook at one clean RL commit."""

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


SCHEMA = "poke_bot.marnie_new_system_hook_activation/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _systemctl(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["systemctl", "--user", *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _show(service: str, prop: str) -> str:
    result = _systemctl("show", service, "-p", prop, "--value")
    if result.returncode != 0:
        raise RuntimeError(result.stdout.strip() or f"cannot inspect {service}")
    return result.stdout.strip()


def _process_environment(pid: int) -> dict[str, str]:
    raw = Path(f"/proc/{pid}/environ").read_bytes()
    result: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        result[key.decode(errors="strict")] = value.decode(errors="strict")
    return result


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--after-iteration", type=int, required=True)
    parser.add_argument("--service", required=True)
    parser.add_argument("--trigger-timer", required=True)
    parser.add_argument("--drop-in", type=Path, required=True)
    parser.add_argument("--trainer-source", type=Path, required=True)
    parser.add_argument("--activation-source", type=Path, required=True)
    parser.add_argument("--trigger-path", type=Path, required=True)
    parser.add_argument("--activation-request", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.20)
    parser.add_argument("--timeout-seconds", type=float, default=21600.0)
    args = parser.parse_args()

    if args.receipt.exists():
        existing = _read_json(args.receipt)
        if existing.get("schema") != SCHEMA or existing.get("status") != "activated":
            raise RuntimeError(f"invalid existing activation receipt: {args.receipt}")
        print(f"MARNIE_R130_HOOK_ALREADY_ACTIVATED receipt={args.receipt}")
        return 0

    required = [args.drop_in, args.trainer_source, args.activation_source]
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"required activation input missing: {path}")
    artifact_sha256 = {str(path): _sha256(path) for path in required}

    timer_state = _show(args.trigger_timer, "ActiveState")
    timer_substate = _show(args.trigger_timer, "SubState")
    if (timer_state, timer_substate) != ("active", "waiting"):
        raise RuntimeError(
            f"iteration-9 trigger timer is not active/waiting: "
            f"{timer_state}/{timer_substate}"
        )

    loop_state_path = args.run_dir / "loop_state.json"
    deadline = time.monotonic() + max(1.0, args.timeout_seconds)
    while time.monotonic() < deadline:
        try:
            loop_state = _read_json(loop_state_path)
            completed = int(loop_state.get("last_completed_iteration", -1))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            completed = -1
            loop_state = {}
        if completed >= args.after_iteration:
            break
        time.sleep(max(0.02, args.poll_seconds))
    else:
        raise RuntimeError("timed out waiting for the immutable RL boundary")

    if completed != args.after_iteration:
        raise RuntimeError(
            f"missed required activation boundary: completed={completed} "
            f"expected={args.after_iteration}"
        )
    if int(loop_state.get("next_iteration", -1)) != args.after_iteration + 1:
        raise RuntimeError("loop state does not point at the next exact iteration")

    commit_path = args.run_dir / "commits" / f"iter_{args.after_iteration:05d}.json"
    commit = _read_json(commit_path)
    if int(commit.get("last_completed_iteration", -1)) != args.after_iteration:
        raise RuntimeError("immutable iteration commit does not match boundary")

    if _show(args.service, "ActiveState") != "active":
        raise RuntimeError("Marnie service is not active at boundary")
    old_pid = int(_show(args.service, "MainPID") or "0")
    if old_pid <= 0:
        raise RuntimeError("Marnie service has no live main PID")

    restart = _systemctl("restart", args.service)
    if restart.returncode != 0:
        raise RuntimeError(restart.stdout.strip() or "managed restart failed")

    new_pid = 0
    new_env: dict[str, str] = {}
    restart_deadline = time.monotonic() + 600.0
    while time.monotonic() < restart_deadline:
        active = _show(args.service, "ActiveState")
        candidate = int(_show(args.service, "MainPID") or "0")
        if active == "active" and candidate > 0 and candidate != old_pid:
            try:
                environment = _process_environment(candidate)
            except OSError:
                time.sleep(0.2)
                continue
            expected = {
                "POKEBOT_MARNIE_ITERATION9_UPLOAD_TRIGGER": str(args.trigger_path),
                "POKEBOT_MARNIE_FAMILY_ACTIVATION_REQUEST": str(
                    args.activation_request
                ),
            }
            if all(environment.get(key) == value for key, value in expected.items()):
                new_pid = candidate
                new_env = expected
                break
        time.sleep(0.2)
    if new_pid <= 0:
        raise RuntimeError("restarted Marnie process did not load revision-130 hook")

    after = _read_json(loop_state_path)
    if int(after.get("last_completed_iteration", -1)) != args.after_iteration:
        raise RuntimeError("lineage moved while activation receipt was being bound")

    receipt = {
        "schema": SCHEMA,
        "status": "activated",
        "activated_at_utc": datetime.now(timezone.utc).isoformat(),
        "owner_boundary_revision": 130,
        "boundary": {
            "completed_iteration": args.after_iteration,
            "next_iteration": args.after_iteration + 1,
            "commit": str(commit_path),
            "commit_sha256": _sha256(commit_path),
        },
        "managed_service": args.service,
        "old_main_pid": old_pid,
        "new_main_pid": new_pid,
        "loaded_environment": new_env,
        "trigger_timer": {
            "unit": args.trigger_timer,
            "active_state": timer_state,
            "sub_state": timer_substate,
        },
        "activation_inputs_sha256": artifact_sha256,
        "successful_iteration9_upload_is_last_old_system_boundary": True,
        "first_post_upload_collection_system": "new_system",
        "old_system_collection_after_successful_upload_allowed": False,
        "scheduler_changed": False,
        "fleet_allocation_changed": False,
        "self_play_tail_rule_changed": False,
    }
    _write_exclusive(args.receipt, receipt)
    print(
        f"MARNIE_R130_HOOK_ACTIVATED old_pid={old_pid} new_pid={new_pid} "
        f"receipt={args.receipt}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
