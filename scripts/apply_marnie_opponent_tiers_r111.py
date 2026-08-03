#!/usr/bin/env python3
"""Activate the staged Marnie tier registry at the iteration-5 commit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def systemctl(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["systemctl", "--user", *args], check=check, capture_output=True, text=True
    )


def main_pid(unit: str) -> int:
    return int(systemctl("show", unit, "-p", "MainPID", "--value").stdout.strip() or 0)


def wait_boundary(commit_path: Path, run_dir: Path, poll_seconds: float) -> dict[str, Any]:
    while True:
        if commit_path.is_file():
            commit = read_json(commit_path)
            loop = read_json(run_dir / "loop_state.json")
            if (
                int(commit.get("last_completed_iteration", commit.get("iteration", -1))) == 5
                and int(loop.get("last_completed_iteration", -1)) == 5
                and int(loop.get("next_iteration", -1)) == 6
            ):
                return commit
        time.sleep(max(0.1, poll_seconds))


def wait_stable(unit: str, *, old_pid: int = 0, timeout: float = 120.0) -> int:
    """Return a stable managed PID for simple or long-running oneshot units.

    The gate handler is intentionally a polling ``Type=oneshot`` service and
    therefore remains ``ActiveState=activating`` while its ExecStart process
    is healthy. Treat that state as live; requiring only ``active`` would wait
    to timeout and incorrectly roll back a successful boundary transaction.
    """
    deadline = time.monotonic() + timeout
    candidate = 0
    since = 0.0
    while time.monotonic() < deadline:
        active_state = systemctl(
            "show", unit, "-p", "ActiveState", "--value", check=False
        ).stdout.strip()
        live = active_state in {"active", "activating"}
        pid = main_pid(unit) if live else 0
        if pid > 0 and pid != old_pid:
            if pid != candidate:
                candidate, since = pid, time.monotonic()
            elif time.monotonic() - since >= 5.0:
                return pid
        else:
            candidate, since = 0, 0.0
        time.sleep(0.5)
    raise RuntimeError(f"managed unit did not stabilize: {unit}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-receipt", type=Path, required=True)
    parser.add_argument("--ceiling-stage-receipt", type=Path, required=True)
    parser.add_argument("--ceiling-activation-receipt", type=Path, required=True)
    parser.add_argument("--activation-receipt", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--commit", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--rl-unit", required=True)
    parser.add_argument("--gate-unit", required=True)
    parser.add_argument("--rl-dropin", type=Path, required=True)
    parser.add_argument("--gate-dropin", type=Path, required=True)
    parser.add_argument("--maintenance-dropin", type=Path, required=True)
    parser.add_argument("--implementation-stage-receipt", type=Path, required=True)
    parser.add_argument("--implementation-activation-receipt", type=Path, required=True)
    parser.add_argument("--implementation-source", type=Path, required=True)
    parser.add_argument("--implementation-target", type=Path, required=True)
    parser.add_argument("--implementation-sha256", required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.1)
    args = parser.parse_args()

    stage = read_json(args.stage_receipt)
    if (
        stage.get("schema") != "poke_bot.marnie_opponent_tier_stage/v1"
        or stage.get("status") != "ready_for_iteration_5_to_6_boundary"
        or int(stage.get("owner_decision_revision", -1)) != 111
        or not Path(stage.get("candidate_registry", "")).is_file()
        or sha256(Path(stage["candidate_registry"]))
        != stage.get("candidate_registry_sha256")
        or sha256(Path(stage["candidate_gate"])) != stage.get("candidate_gate_sha256")
    ):
        raise RuntimeError("staged tier receipt or candidate checksum is invalid")
    ceiling_stage = read_json(args.ceiling_stage_receipt)
    if (
        ceiling_stage.get("schema") != "poke_bot.marnie_iteration20_stage/v1"
        or ceiling_stage.get("status") != "ready_for_iteration_5_to_6_boundary"
        or int(ceiling_stage.get("owner_decision_revision", -1)) != 113
        or Path(ceiling_stage.get("parent_registry", ""))
        != Path(stage["candidate_registry"])
        or ceiling_stage.get("parent_registry_sha256")
        != stage["candidate_registry_sha256"]
        or Path(ceiling_stage.get("candidate_registry", "")) != args.registry
        or sha256(args.registry) != ceiling_stage.get("candidate_registry_sha256")
        or int(ceiling_stage.get("iteration_ceiling", -1)) != 20
        or int(ceiling_stage.get("maximum_iterations", -1)) != 21
    ):
        raise RuntimeError("staged iteration-20 receipt or candidate checksum is invalid")
    implementation_stage = read_json(args.implementation_stage_receipt)
    if (
        implementation_stage.get("schema")
        != "poke_bot.scheduled_fleet_tail_work_steal_stage/v1"
        or implementation_stage.get("status")
        != "ready_for_iteration_5_to_6_boundary"
        or int(implementation_stage.get("owner_decision_revision", -1)) != 112
        or implementation_stage.get("implementation_sha256")
        != args.implementation_sha256
        or sha256(args.implementation_source) != args.implementation_sha256
    ):
        raise RuntimeError("staged tail-work-steal receipt or checksum is invalid")
    if (
        args.activation_receipt.exists()
        and args.ceiling_activation_receipt.exists()
        and args.implementation_activation_receipt.exists()
    ):
        print(args.activation_receipt.read_text(encoding="utf-8"), end="")
        print(args.ceiling_activation_receipt.read_text(encoding="utf-8"), end="")
        print(
            args.implementation_activation_receipt.read_text(encoding="utf-8"),
            end="",
        )
        return 0

    commit = wait_boundary(args.commit, args.run_dir, args.poll_seconds)
    python = "/home/inzi/miniconda3/envs/poke-bot-agent/bin/python"
    launcher = args.runtime_root / "scripts/launch_active_specialist.py"
    gate_launcher = args.runtime_root / "scripts/launch_active_specialist_gate_handler.py"
    env = dict(os.environ)
    env["POKEBOT_ACTIVE_SPECIALIST"] = "marnie-s-grimmsnarl-ex"
    env["PYTHONPATH"] = str(args.runtime_root)
    for program in (launcher, gate_launcher):
        subprocess.run(
            [python, "-u", str(program), "--registry", str(args.registry), "--check"],
            cwd=args.runtime_root,
            env=env,
            check=True,
        )

    rl_command = f"{python} -u {launcher} --registry {args.registry}"
    gate_command = f"{python} -u {gate_launcher} --registry {args.registry}"
    rl_dropin = (
        "[Service]\n"
        f"ExecStartPre={rl_command} --check\n"
        f"ExecStartPre={gate_command} --check\n"
        "ExecStart=\n"
        f"ExecStart={rl_command}\n"
    ).encode()
    gate_dropin = (
        "[Service]\n"
        f"ExecStartPre={gate_command} --check\n"
        "ExecStart=\n"
        f"ExecStart={gate_command}\n"
    ).encode()
    prior_rl = args.rl_dropin.read_bytes() if args.rl_dropin.exists() else None
    prior_gate = args.gate_dropin.read_bytes() if args.gate_dropin.exists() else None
    prior_implementation = (
        args.implementation_target.read_bytes()
        if args.implementation_target.exists()
        else None
    )
    prior_implementation_sha256 = (
        sha256(args.implementation_target)
        if args.implementation_target.exists()
        else None
    )
    old_rl_pid = main_pid(args.rl_unit)
    old_gate_pid = main_pid(args.gate_unit)
    if old_rl_pid <= 0:
        raise RuntimeError("managed trainer is not active at the boundary")

    atomic_bytes(args.maintenance_dropin, b"[Unit]\nRefuseManualStop=no\n")
    atomic_bytes(args.rl_dropin, rl_dropin)
    atomic_bytes(args.gate_dropin, gate_dropin)
    systemctl("daemon-reload")
    try:
        systemctl("stop", args.gate_unit, check=False)
        atomic_bytes(
            args.implementation_target,
            args.implementation_source.read_bytes(),
        )
        if sha256(args.implementation_target) != args.implementation_sha256:
            raise RuntimeError("activated tail-work-steal implementation checksum drift")
        systemctl("restart", args.rl_unit)
        new_rl_pid = wait_stable(args.rl_unit, old_pid=old_rl_pid)
        # The gate handler is a long-running Type=oneshot poller.  A normal
        # `systemctl start` waits for ExecStart to exit, which would deadlock
        # this boundary transaction before receipts can be written.
        systemctl("start", "--no-block", args.gate_unit)
        new_gate_pid = wait_stable(args.gate_unit, old_pid=old_gate_pid)
    except BaseException:
        if prior_rl is None:
            args.rl_dropin.unlink(missing_ok=True)
        else:
            atomic_bytes(args.rl_dropin, prior_rl)
        if prior_gate is None:
            args.gate_dropin.unlink(missing_ok=True)
        else:
            atomic_bytes(args.gate_dropin, prior_gate)
        if prior_implementation is None:
            args.implementation_target.unlink(missing_ok=True)
        else:
            atomic_bytes(args.implementation_target, prior_implementation)
        systemctl("daemon-reload")
        systemctl("restart", args.rl_unit, check=False)
        systemctl("start", "--no-block", args.gate_unit, check=False)
        raise
    finally:
        args.maintenance_dropin.unlink(missing_ok=True)
        systemctl("daemon-reload")

    receipt = {
        "schema": "poke_bot.marnie_opponent_tier_activation/v1",
        "status": "activated",
        "owner_decision_revision": 111,
        "activated_at_utc": datetime.now(timezone.utc).isoformat(),
        "boundary_commit": str(args.commit),
        "boundary_commit_sha256": sha256(args.commit),
        "boundary_checkpoint": commit.get("learner_after", commit.get("learner")),
        "stage_receipt": str(args.stage_receipt),
        "stage_receipt_sha256": sha256(args.stage_receipt),
        "active_registry": str(args.registry),
        "active_registry_sha256": sha256(args.registry),
        "active_gate": stage["candidate_gate"],
        "active_gate_sha256": stage["candidate_gate_sha256"],
        "old_rl_pid": old_rl_pid,
        "new_rl_pid": new_rl_pid,
        "old_gate_pid": old_gate_pid,
        "new_gate_pid": new_gate_pid,
        "first_affected_iteration": 6,
    }
    atomic_json(args.activation_receipt, receipt)
    ceiling_receipt = {
        "schema": "poke_bot.marnie_iteration20_activation/v1",
        "status": "activated",
        "owner_decision_revision": 113,
        "activated_at_utc": receipt["activated_at_utc"],
        "boundary_commit": str(args.commit),
        "boundary_commit_sha256": receipt["boundary_commit_sha256"],
        "first_affected_iteration": 6,
        "iteration_ceiling": 20,
        "maximum_iterations": 21,
        "collect_iteration_21": False,
        "stage_receipt": str(args.ceiling_stage_receipt),
        "stage_receipt_sha256": sha256(args.ceiling_stage_receipt),
        "active_registry": str(args.registry),
        "active_registry_sha256": sha256(args.registry),
        "old_rl_pid": old_rl_pid,
        "new_rl_pid": new_rl_pid,
        "old_gate_pid": old_gate_pid,
        "new_gate_pid": new_gate_pid,
    }
    atomic_json(args.ceiling_activation_receipt, ceiling_receipt)
    implementation_receipt = {
        "schema": "poke_bot.scheduled_fleet_tail_work_steal_activation/v1",
        "status": "activated",
        "owner_decision_revision": 112,
        "activated_at_utc": receipt["activated_at_utc"],
        "boundary_commit": str(args.commit),
        "boundary_commit_sha256": receipt["boundary_commit_sha256"],
        "first_affected_iteration": 6,
        "implementation_source": str(args.implementation_source),
        "implementation_target": str(args.implementation_target),
        "implementation_sha256": sha256(args.implementation_target),
        "prior_implementation_sha256": prior_implementation_sha256,
        "stage_receipt": str(args.implementation_stage_receipt),
        "stage_receipt_sha256": sha256(args.implementation_stage_receipt),
        "tail_entry_unclaimed_fraction": 0.20,
        "remote_claim_games_in_tail": 1,
        "blackwell_local_workers_remain_eligible": 96,
        "result_transport": "per_game_streaming",
        "network_rpc_batching_introduced": False,
        "exact_once_concurrency_tests": implementation_stage.get("tests"),
        "old_rl_pid": old_rl_pid,
        "new_rl_pid": new_rl_pid,
    }
    atomic_json(args.implementation_activation_receipt, implementation_receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    print(json.dumps(ceiling_receipt, indent=2, sort_keys=True))
    print(json.dumps(implementation_receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
