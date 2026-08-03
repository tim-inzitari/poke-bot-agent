#!/usr/bin/env python3
"""Activate a staged Marnie expert corpus at the iteration-4→5 boundary.

The controller waits for both immutable staging evidence and the exact commit,
then changes only managed-unit drop-ins.  It never signals a process directly.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any, Iterable


STAGE_SCHEMA = "poke_bot.marnie_latest20_runtime_migration_stage/v1"
ACTIVATION_SCHEMA = "poke_bot.marnie_latest20_runtime_activation/v1"
SPECIALIST_ID = "marnie-s-grimmsnarl-ex"
EARLY_POOL_RELEASE_ENV = "POKEBOT_RELEASE_LOCAL_POOL_BEFORE_RESULT_DRAIN=1"
EARLY_POOL_RELEASE_IMPLEMENTATION = {
    "worker_pool": (
        "poke_bot/worker_pool.py",
        (
            "def release(self) -> None:",
            "self.release()",
            "pool.join()",
        ),
    ),
    "scheduled_results": (
        "poke_bot/remote_jobs.py",
        (
            "on_producers_drained: Optional[Callable[[], None]] = None",
            "def _producer_finished() -> None:",
            "callback = on_producers_drained",
        ),
    ),
    "trainer_integration": (
        "scripts/train_pure_rl.py",
        (
            '"POKEBOT_RELEASE_LOCAL_POOL_BEFORE_RESULT_DRAIN", "0"',
            "def _release_exhausted_local_pool() -> None:",
            "pool.release()",
            "on_producers_drained=(",
        ),
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def write_once(path: Path, value: dict[str, Any]) -> None:
    body = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != body:
            raise RuntimeError(f"refusing to replace immutable artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text(path: Path, body: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(argv: Iterable[str], *, timeout: float = 180.0) -> str:
    values = [str(value) for value in argv]
    result = subprocess.run(
        values,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if result.stdout:
        print(result.stdout.rstrip(), flush=True)
    if result.returncode:
        raise RuntimeError(
            f"command exited {result.returncode}: {shlex.join(values)}"
        )
    return result.stdout.strip()


def service_value(unit: str, key: str) -> str:
    return run(
        ["systemctl", "--user", "show", unit, "-p", key, "--value"],
        timeout=15,
    ).strip()


def process_environment(pid: int) -> set[str]:
    """Read the exact environment inherited by a managed process."""
    if int(pid) <= 0:
        raise RuntimeError("managed process PID must be positive")
    path = Path(f"/proc/{int(pid)}/environ")
    try:
        return {
            value.decode("utf-8", errors="strict")
            for value in path.read_bytes().split(b"\0")
            if value
        }
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(
            f"cannot verify managed process environment: pid={int(pid)}"
        ) from exc


def process_command_line(pid: int) -> list[str]:
    """Read the exact argv of a managed process without shell parsing."""
    if int(pid) <= 0:
        raise RuntimeError("managed process PID must be positive")
    path = Path(f"/proc/{int(pid)}/cmdline")
    try:
        return [
            value.decode("utf-8", errors="strict")
            for value in path.read_bytes().split(b"\0")
            if value
        ]
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(
            f"cannot verify managed process command line: pid={int(pid)}"
        ) from exc


def replace_exact(text: str, old: str, new: str, *, count: int) -> str:
    if text.count(old) != count:
        raise RuntimeError(
            f"expected {count} occurrences of {old!r}, found {text.count(old)}"
        )
    return text.replace(old, new)


def service_commands(
    unit_text: str,
    *,
    key: str,
    contains: str,
) -> list[str]:
    prefix = key + "="
    rows = [
        row
        for row in unit_text.splitlines()
        if row.startswith(prefix) and contains in row
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"expected one {key} containing {contains!r}, found {len(rows)}"
        )
    return shlex.split(rows[0][len(prefix) :])


def render_dropins(
    *,
    rl_unit_text: str,
    gate_unit_text: str,
    completion_unit_text: str,
    old_registry: str,
    candidate_registry: str,
    old_registration: str,
    candidate_registration: str,
) -> tuple[str, str, str, list[str], list[str], list[str]]:
    rl_start = service_commands(
        rl_unit_text,
        key="ExecStart",
        contains="launch_active_specialist.py",
    )
    gate_check = service_commands(
        gate_unit_text,
        key="ExecStartPre",
        contains="launch_active_specialist_gate_handler.py",
    )
    gate_start = service_commands(
        gate_unit_text,
        key="ExecStart",
        contains="launch_active_specialist_gate_handler.py",
    )
    completion_check = service_commands(
        completion_unit_text,
        key="ExecStartPre",
        contains="complete_final_format_marnie_refresh.py",
    )
    completion_start = service_commands(
        completion_unit_text,
        key="ExecStart",
        contains="complete_final_format_marnie_refresh.py",
    )
    rl_check = list(rl_start) + ["--check"]

    def swap(values: list[str], old: str, new: str) -> list[str]:
        joined = "\0".join(values)
        changed = replace_exact(joined, old, new, count=1)
        return changed.split("\0")

    rl_start = swap(rl_start, old_registry, candidate_registry)
    rl_check = swap(rl_check, old_registry, candidate_registry)
    gate_check = swap(gate_check, old_registry, candidate_registry)
    gate_start = swap(gate_start, old_registry, candidate_registry)
    completion_check = swap(
        completion_check, old_registration, candidate_registration
    )
    completion_start = swap(
        completion_start, old_registration, candidate_registration
    )

    rl_dropin = (
        "[Service]\n"
        f"Environment={EARLY_POOL_RELEASE_ENV}\n"
        f"ExecStartPre={shlex.join(rl_check)}\n"
        f"ExecStartPre={shlex.join(gate_check)}\n"
        f"ExecStartPre={shlex.join(completion_check)}\n"
        "ExecStart=\n"
        f"ExecStart={shlex.join(rl_start)}\n"
    )
    gate_dropin = (
        "[Service]\n"
        "ExecStartPre=\n"
        f"ExecStartPre={shlex.join(gate_check)}\n"
        "ExecStart=\n"
        f"ExecStart={shlex.join(gate_start)}\n"
    )
    completion_dropin = (
        "[Service]\n"
        "ExecStartPre=\n"
        f"ExecStartPre={shlex.join(completion_check)}\n"
        "ExecStart=\n"
        f"ExecStart={shlex.join(completion_start)}\n"
    )
    return (
        rl_dropin,
        gate_dropin,
        completion_dropin,
        rl_check,
        gate_check,
        completion_check,
    )


def validate_stage(path: Path) -> dict[str, Any]:
    stage = read_json(path)
    sync_path = Path(str(stage.get("sync_receipt") or ""))
    registry_path = Path(str(stage.get("candidate_registry") or ""))
    registration_path = Path(str(stage.get("candidate_registration") or ""))
    pointer_path = Path(str(stage.get("candidate_expert_pointer") or ""))
    if (
        stage.get("schema") != STAGE_SCHEMA
        or stage.get("status") != "ready_for_clean_rehearsal_boundary"
        or stage.get("specialist_id") != SPECIALIST_ID
        or stage.get("active_registry_modified") is not False
        or stage.get("training_interrupted") is not False
        or not sync_path.is_file()
        or sha256(sync_path) != stage.get("sync_receipt_sha256")
        or not registry_path.is_file()
        or not registration_path.is_file()
        or not pointer_path.is_file()
        or sha256(registry_path) != stage.get("candidate_registry_sha256")
        or sha256(registration_path)
        != stage.get("candidate_registration_sha256")
        or sha256(pointer_path) != stage.get("candidate_expert_pointer_sha256")
    ):
        raise RuntimeError("Marnie latest-20 migration stage is invalid")
    registry = read_json(registry_path)
    registration = read_json(registration_path)
    sync = read_json(sync_path)
    row = dict((registry.get("specialists") or {}).get(SPECIALIST_ID) or {})
    if (
        sync.get("schema") != "poke_bot.latest20_specialist_sync/v1"
        or sync.get("status") != "ready"
        or registration.get("runtime_registry") != str(registry_path)
        or registration.get("runtime_registry_sha256") != sha256(registry_path)
        or row.get("expert_manifest") != str(pointer_path)
        or row.get("expert_manifest_sha256")
        != sha256(pointer_path).removeprefix("sha256:")
    ):
        raise RuntimeError("staged registry and registration do not agree")
    return stage


def validate_early_pool_release_implementation(
    deployment_root: Path,
) -> dict[str, Any]:
    """Bind the boundary flag to the exact already-validated implementation."""

    root = deployment_root.resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError(f"deployment root is not a directory: {root}")
    modules: dict[str, dict[str, Any]] = {}
    for name, (relative, markers) in EARLY_POOL_RELEASE_IMPLEMENTATION.items():
        path = (root / relative).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(
                f"early-pool-release module escapes deployment root: {path}"
            ) from exc
        if not path.is_file():
            raise RuntimeError(f"early-pool-release module is not a file: {path}")
        source = path.read_text(encoding="utf-8")
        missing = [marker for marker in markers if marker not in source]
        if missing:
            raise RuntimeError(
                f"early-pool-release implementation is incomplete in {path}: "
                f"missing {missing!r}"
            )
        modules[name] = {
            "path": str(path),
            "relative_path": relative,
            "sha256": sha256(path),
            "required_markers": list(markers),
        }
    return {
        "deployment_root": str(root),
        "enabled_environment": EARLY_POOL_RELEASE_ENV,
        "modules": modules,
    }


def committed_boundary(path: Path, completed_iteration: int) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = read_json(path)
    learner = dict(value.get("learner") or {})
    checkpoint = Path(str(learner.get("path") or ""))
    if (
        int(value.get("last_completed_iteration") or -1) != completed_iteration
        or int(value.get("next_iteration") or -1) != completed_iteration + 1
        or not checkpoint.is_file()
        or sha256(checkpoint) != learner.get("digest")
    ):
        raise RuntimeError("iteration boundary commit is malformed")
    return value


def wait_file(path: Path, timeout: float, poll: float) -> None:
    deadline = time.monotonic() + timeout
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(max(0.05, poll))


def wait_service(unit: str, expected: set[str], timeout: float) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = service_value(unit, "ActiveState")
        if state in expected:
            return state
        time.sleep(0.25)
    raise TimeoutError(f"timed out waiting for {unit}: {sorted(expected)}")


def wait_service_pid(
    unit: str,
    previous_pid: int,
    timeout: float,
    *,
    stable_seconds: float = 2.0,
) -> int:
    """Wait through ExecStartPre until a new positive MainPID is stable.

    A Type=oneshot service enters ``activating`` before systemd assigns its
    long-lived ExecStart PID.  Treating that state alone as ready caused the
    original iteration-4→5 controller to observe MainPID=0 and roll back
    already-applied drop-ins.
    """

    deadline = time.monotonic() + timeout
    candidate = 0
    candidate_since = 0.0
    while time.monotonic() < deadline:
        state = service_value(unit, "ActiveState")
        pid = int(service_value(unit, "MainPID") or 0)
        now = time.monotonic()
        if state in {"active", "activating"} and pid > 0 and pid != previous_pid:
            if pid != candidate:
                candidate = pid
                candidate_since = now
            elif now - candidate_since >= stable_seconds:
                return pid
        else:
            candidate = 0
            candidate_since = 0.0
        time.sleep(0.25)
    raise TimeoutError(
        f"timed out waiting for stable new MainPID for {unit}; "
        f"previous={previous_pid}"
    )


def wait_migration(run_dir: Path, reason: str, boundary: int, after: float) -> Path:
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        for path in sorted((run_dir / "design_migrations").glob("migration_*.json")):
            if path.stat().st_mtime < after:
                continue
            value = read_json(path)
            if (
                value.get("reason") == reason
                and int(value.get("boundary_next_iteration") or -1) == boundary
            ):
                return path
        time.sleep(0.25)
    raise TimeoutError("new expert corpus design migration receipt did not appear")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-receipt", type=Path, required=True)
    parser.add_argument("--activation-receipt", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--deployment-root", type=Path, required=True)
    parser.add_argument("--completed-iteration", type=int, default=4)
    parser.add_argument("--rl-unit", required=True)
    parser.add_argument("--gate-unit", required=True)
    parser.add_argument("--completion-unit", required=True)
    parser.add_argument("--rl-unit-file", type=Path, required=True)
    parser.add_argument("--gate-unit-file", type=Path, required=True)
    parser.add_argument("--completion-unit-file", type=Path, required=True)
    parser.add_argument("--rl-dropin", type=Path, required=True)
    parser.add_argument("--gate-dropin", type=Path, required=True)
    parser.add_argument("--completion-dropin", type=Path, required=True)
    parser.add_argument("--maintenance-dropin", type=Path, required=True)
    parser.add_argument("--old-registry", required=True)
    parser.add_argument("--old-registration", required=True)
    parser.add_argument("--migration-reason", required=True)
    parser.add_argument("--stage-timeout", type=float, default=21600)
    parser.add_argument("--boundary-timeout", type=float, default=86400)
    parser.add_argument("--poll-seconds", type=float, default=0.1)
    parser.add_argument("--recover-applied-boundary", action="store_true")
    parser.add_argument("--recovery-old-pid", type=int, default=0)
    parser.add_argument("--recovery-old-gate-pid", type=int, default=0)
    args = parser.parse_args()

    if args.activation_receipt.is_file():
        existing = read_json(args.activation_receipt)
        if existing.get("schema") != ACTIVATION_SCHEMA:
            raise RuntimeError("existing activation receipt has wrong schema")
        print(json.dumps(existing, sort_keys=True))
        return 0

    wait_file(args.stage_receipt, args.stage_timeout, 1.0)
    stage = validate_stage(args.stage_receipt)
    early_pool_release_implementation = validate_early_pool_release_implementation(
        args.deployment_root
    )
    candidate_registry = str(stage["candidate_registry"])
    candidate_registration = str(stage["candidate_registration"])
    candidate_pointer = str(stage["candidate_expert_pointer"])
    (
        rl_dropin,
        gate_dropin,
        completion_dropin,
        rl_check,
        gate_check,
        completion_check,
    ) = render_dropins(
        rl_unit_text=args.rl_unit_file.read_text(encoding="utf-8"),
        gate_unit_text=args.gate_unit_file.read_text(encoding="utf-8"),
        completion_unit_text=args.completion_unit_file.read_text(encoding="utf-8"),
        old_registry=args.old_registry,
        candidate_registry=candidate_registry,
        old_registration=args.old_registration,
        candidate_registration=candidate_registration,
    )
    for command in (rl_check, gate_check, completion_check):
        run(command, timeout=180)

    commit_path = (
        args.run_dir
        / "commits"
        / f"iter_{args.completed_iteration:05d}.json"
    )
    deadline = time.monotonic() + args.boundary_timeout
    boundary: dict[str, Any] | None = None
    while boundary is None:
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for rehearsal boundary")
        boundary = committed_boundary(commit_path, args.completed_iteration)
        if boundary is None:
            time.sleep(max(0.05, args.poll_seconds))

    # The waiter can remain resident for hours. Re-open every staged identity
    # and implementation file at the actual boundary so the receipt cannot
    # bind startup-time hashes after an intervening deployment mutation.
    boundary_stage = validate_stage(args.stage_receipt)
    if boundary_stage != stage:
        raise RuntimeError("Marnie latest-20 migration stage changed while waiting")
    boundary_implementation = validate_early_pool_release_implementation(
        args.deployment_root
    )
    if boundary_implementation != early_pool_release_implementation:
        raise RuntimeError(
            "early-pool-release implementation changed while waiting"
        )

    loop_path = args.run_dir / "loop_state.json"
    if loop_path.read_bytes() != commit_path.read_bytes():
        if not args.recover_applied_boundary:
            raise RuntimeError("trainer advanced beyond the exact rehearsal boundary")
        if args.recovery_old_pid <= 0 or args.recovery_old_gate_pid <= 0:
            raise RuntimeError(
                "applied-boundary recovery requires the journal-backed old PIDs"
            )
        loop = read_json(loop_path)
        learner = dict(loop.get("learner") or {})
        if (
            int(loop.get("last_completed_iteration") or -1)
            != args.completed_iteration
            or int(loop.get("next_iteration") or -1)
            != args.completed_iteration + 1
            or learner != boundary.get("learner")
        ):
            raise RuntimeError(
                "applied-boundary recovery found a later committed iteration"
            )
        matching_migrations = [
            row
            for row in list(loop.get("design_migration_history") or [])
            if isinstance(row, dict)
            and row.get("reason") == args.migration_reason
            and int(row.get("boundary_next_iteration") or -1)
            == args.completed_iteration + 1
        ]
        if len(matching_migrations) != 1:
            raise RuntimeError(
                "applied-boundary recovery requires exactly one matching migration"
            )
        migration_receipt = Path(str(matching_migrations[0].get("receipt") or ""))
        if not migration_receipt.is_file():
            raise RuntimeError("applied-boundary migration receipt is missing")
        migration = read_json(migration_receipt)
        if (
            migration.get("reason") != args.migration_reason
            or int(migration.get("boundary_next_iteration") or -1)
            != args.completed_iteration + 1
        ):
            raise RuntimeError("applied-boundary migration receipt is malformed")

        new_pid = int(service_value(args.rl_unit, "MainPID") or 0)
        if new_pid <= 0 or service_value(args.rl_unit, "ActiveState") != "active":
            raise RuntimeError("managed Marnie trainer is not active during recovery")
        if new_pid == args.recovery_old_pid:
            raise RuntimeError("Marnie trainer PID did not advance at the boundary")
        if EARLY_POOL_RELEASE_ENV not in process_environment(new_pid):
            raise RuntimeError(
                "recovered Marnie trainer lacks mandatory pool-release environment"
            )
        trainer_argv = process_command_line(new_pid)
        if (
            candidate_pointer not in trainer_argv
            or args.migration_reason not in trainer_argv
        ):
            raise RuntimeError(
                "recovered Marnie trainer is not the staged Latest20 command"
            )

        # The failed controller removed these after the trainer had already
        # committed its design migration. Restore restart persistence without
        # touching the healthy trainer, then rotate only the stale gate watcher.
        atomic_text(args.rl_dropin, rl_dropin)
        atomic_text(args.gate_dropin, gate_dropin)
        atomic_text(args.completion_dropin, completion_dropin)
        args.maintenance_dropin.unlink(missing_ok=True)
        run(["systemctl", "--user", "daemon-reload"], timeout=30)
        observed_gate_pid = int(service_value(args.gate_unit, "MainPID") or 0)
        if observed_gate_pid <= 0:
            raise RuntimeError("managed Marnie gate watcher is absent during recovery")
        run(["systemctl", "--user", "stop", args.gate_unit], timeout=90)
        wait_service(args.gate_unit, {"inactive", "failed"}, 45)
        run(["systemctl", "--user", "reset-failed", args.gate_unit], timeout=15)
        run(
            ["systemctl", "--user", "start", "--no-block", args.gate_unit],
            timeout=15,
        )
        new_gate_pid = wait_service_pid(args.gate_unit, observed_gate_pid, 45)
        loaded_gate_unit = run(
            ["systemctl", "--user", "cat", args.gate_unit, "--no-pager"],
            timeout=15,
        )
        if candidate_registry not in loaded_gate_unit:
            raise RuntimeError(
                "recovered Marnie gate watcher is not using the staged registry"
            )
        if (
            service_value(args.rl_unit, "ActiveState") != "active"
            or int(service_value(args.rl_unit, "MainPID") or 0) != new_pid
        ):
            raise RuntimeError("healthy Marnie trainer changed during gate recovery")

        receipt = {
            "schema": ACTIVATION_SCHEMA,
            "status": "activated",
            "activated_at_utc": datetime.now(timezone.utc).isoformat(),
            "specialist_id": SPECIALIST_ID,
            "window_start": stage["window_start"],
            "window_end": stage["window_end"],
            "active_training_corpus": True,
            "stage_receipt": str(args.stage_receipt),
            "stage_receipt_sha256": sha256(args.stage_receipt),
            "candidate_registry": candidate_registry,
            "candidate_registry_sha256": sha256(Path(candidate_registry)),
            "candidate_registration": candidate_registration,
            "candidate_registration_sha256": sha256(Path(candidate_registration)),
            "candidate_expert_pointer": candidate_pointer,
            "candidate_expert_pointer_sha256": sha256(Path(candidate_pointer)),
            "boundary_iteration": args.completed_iteration,
            "boundary_commit": str(commit_path),
            "boundary_commit_sha256": sha256(commit_path),
            "checkpoint": boundary["learner"]["path"],
            "checkpoint_sha256": boundary["learner"]["digest"],
            "old_pid": args.recovery_old_pid,
            "new_pid": new_pid,
            "old_gate_pid": args.recovery_old_gate_pid,
            "new_gate_pid": new_gate_pid,
            "recovery_observed_gate_pid": observed_gate_pid,
            "recovered_after_gate_pid_observation_race": True,
            "trainer_restarted_during_recovery": False,
            "gate_restarted_during_recovery": True,
            "design_migration_receipt": str(migration_receipt),
            "design_migration_receipt_sha256": sha256(migration_receipt),
            "rl_dropin_sha256": sha256(args.rl_dropin),
            "gate_dropin_sha256": sha256(args.gate_dropin),
            "completion_dropin_sha256": sha256(args.completion_dropin),
            "release_local_pool_before_result_drain": True,
            "active_process_environment_verified": True,
            "early_pool_release_implementation": boundary_implementation,
            "training_history_replayed": False,
            "selector_identity_changed": False,
        }
        write_once(args.activation_receipt, receipt)
        print(json.dumps(receipt, sort_keys=True))
        return 0
    old_pid = int(service_value(args.rl_unit, "MainPID") or 0)
    old_gate_pid = int(service_value(args.gate_unit, "MainPID") or 0)
    gate_was_active = service_value(args.gate_unit, "ActiveState") in {
        "active",
        "activating",
    }
    if old_pid <= 0 or service_value(args.rl_unit, "ActiveState") != "active":
        raise RuntimeError("managed Marnie trainer is not active at boundary")
    if not gate_was_active or old_gate_pid <= 0:
        raise RuntimeError("managed Marnie gate watcher is not active at boundary")

    atomic_text(args.maintenance_dropin, "[Unit]\nOnSuccess=\n")
    run(["systemctl", "--user", "daemon-reload"], timeout=30)
    # Stop the long-lived gate watcher before the intentional trainer stop.
    # It was launched with the old registry and otherwise could both retain
    # stale terminal identity and interpret this clean migration as a terminal
    # trainer exit.
    run(["systemctl", "--user", "stop", args.gate_unit], timeout=90)
    wait_service(args.gate_unit, {"inactive", "failed"}, 45)
    run(["systemctl", "--user", "stop", args.rl_unit], timeout=90)
    wait_service(args.rl_unit, {"inactive", "failed"}, 45)
    if loop_path.read_bytes() != commit_path.read_bytes():
        raise RuntimeError("loop state changed during managed boundary stop")

    migration_receipt: Path | None = None
    try:
        atomic_text(args.rl_dropin, rl_dropin)
        atomic_text(args.gate_dropin, gate_dropin)
        atomic_text(args.completion_dropin, completion_dropin)
        args.maintenance_dropin.unlink(missing_ok=True)
        run(["systemctl", "--user", "daemon-reload"], timeout=30)
        started = time.time() - 1
        run(["systemctl", "--user", "reset-failed", args.rl_unit], timeout=15)
        run(["systemctl", "--user", "start", args.rl_unit], timeout=90)
        wait_service(args.rl_unit, {"active"}, 45)
        new_pid = int(service_value(args.rl_unit, "MainPID") or 0)
        if new_pid <= 0 or new_pid == old_pid:
            raise RuntimeError("Marnie trainer PID did not advance")
        if EARLY_POOL_RELEASE_ENV not in process_environment(new_pid):
            raise RuntimeError(
                "Marnie iteration-5 trainer did not inherit the mandatory "
                "producer-complete pool-release environment"
            )
        run(["systemctl", "--user", "reset-failed", args.gate_unit], timeout=15)
        run(
            ["systemctl", "--user", "start", "--no-block", args.gate_unit],
            timeout=15,
        )
        new_gate_pid = wait_service_pid(args.gate_unit, old_gate_pid, 45)
        migration_receipt = wait_migration(
            args.run_dir,
            args.migration_reason,
            args.completed_iteration + 1,
            started,
        )
        stability_deadline = time.monotonic() + 30
        while time.monotonic() < stability_deadline:
            if (
                service_value(args.rl_unit, "ActiveState") != "active"
                or int(service_value(args.rl_unit, "MainPID") or 0) != new_pid
                or service_value(args.gate_unit, "ActiveState")
                not in {"active", "activating"}
                or int(service_value(args.gate_unit, "MainPID") or 0)
                != new_gate_pid
            ):
                raise RuntimeError(
                    "Marnie trainer or gate watcher failed post-activation stability"
                )
            time.sleep(1)
        receipt = {
            "schema": ACTIVATION_SCHEMA,
            "status": "activated",
            "activated_at_utc": datetime.now(timezone.utc).isoformat(),
            "specialist_id": SPECIALIST_ID,
            "window_start": stage["window_start"],
            "window_end": stage["window_end"],
            "active_training_corpus": True,
            "stage_receipt": str(args.stage_receipt),
            "stage_receipt_sha256": sha256(args.stage_receipt),
            "candidate_registry": candidate_registry,
            "candidate_registry_sha256": sha256(Path(candidate_registry)),
            "candidate_registration": candidate_registration,
            "candidate_registration_sha256": sha256(Path(candidate_registration)),
            "candidate_expert_pointer": candidate_pointer,
            "candidate_expert_pointer_sha256": sha256(Path(candidate_pointer)),
            "boundary_iteration": args.completed_iteration,
            "boundary_commit": str(commit_path),
            "boundary_commit_sha256": sha256(commit_path),
            "checkpoint": boundary["learner"]["path"],
            "checkpoint_sha256": boundary["learner"]["digest"],
            "old_pid": old_pid,
            "new_pid": new_pid,
            "old_gate_pid": old_gate_pid,
            "new_gate_pid": new_gate_pid,
            "design_migration_receipt": str(migration_receipt),
            "design_migration_receipt_sha256": sha256(migration_receipt),
            "rl_dropin_sha256": sha256(args.rl_dropin),
            "gate_dropin_sha256": sha256(args.gate_dropin),
            "completion_dropin_sha256": sha256(args.completion_dropin),
            "release_local_pool_before_result_drain": True,
            "active_process_environment_verified": True,
            "early_pool_release_implementation": boundary_implementation,
            "training_history_replayed": False,
            "selector_identity_changed": False,
        }
        write_once(args.activation_receipt, receipt)
        print(json.dumps(receipt, sort_keys=True))
        return 0
    except BaseException:
        args.maintenance_dropin.unlink(missing_ok=True)
        if migration_receipt is None:
            args.rl_dropin.unlink(missing_ok=True)
            args.gate_dropin.unlink(missing_ok=True)
            args.completion_dropin.unlink(missing_ok=True)
        run(["systemctl", "--user", "daemon-reload"], timeout=30)
        if service_value(args.rl_unit, "ActiveState") != "active":
            run(["systemctl", "--user", "reset-failed", args.rl_unit], timeout=15)
            run(["systemctl", "--user", "start", args.rl_unit], timeout=90)
        if gate_was_active and service_value(args.gate_unit, "ActiveState") not in {
            "active",
            "activating",
        }:
            run(
                ["systemctl", "--user", "reset-failed", args.gate_unit],
                timeout=15,
            )
            run(
                [
                    "systemctl",
                    "--user",
                    "start",
                    "--no-block",
                    args.gate_unit,
                ],
                timeout=15,
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
