#!/usr/bin/env python3
"""Activate derivative no-progress shaping at the sealed iter-5 boundary."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def run(argv: list[str]) -> str:
    return subprocess.run(
        argv, check=True, text=True, capture_output=True
    ).stdout.strip()


def service_property(unit: str, name: str) -> str:
    return run(["systemctl", "--user", "show", unit, f"-p{name}", "--value"])


def write_json_create_only(path: Path, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    body += b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(fd, body)
        os.fsync(fd)
    finally:
        os.close(fd)


def _validate_commit(path: Path, run_name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "run_name": run_name,
        "last_completed_iteration": 5,
        "next_iteration": 6,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(
                f"iteration-5 boundary mismatch for {key}: "
                f"expected={value!r} actual={payload.get(key)!r}"
            )
    learner = dict(payload.get("learner") or {})
    if not str(learner.get("digest") or "").startswith("sha256:"):
        raise RuntimeError("iteration-5 commit lacks a learner digest")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--training-unit", required=True)
    parser.add_argument("--expected-main-pid", type=int, required=True)
    parser.add_argument("--deployment-root", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--goal-contract", type=Path, required=True)
    parser.add_argument("--goal-contract-sha256", required=True)
    parser.add_argument("--drop-in-source", type=Path, required=True)
    parser.add_argument("--drop-in-destination", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--max-no-progress-turns", type=int, default=64)
    parser.add_argument("--wait-for-commit-seconds", type=float, default=0.0)
    parser.add_argument("--commit-poll-seconds", type=float, default=0.05)
    parser.add_argument("--restart-timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()

    if args.max_no_progress_turns != 64:
        raise RuntimeError("revision 31 requires exactly 64 no-progress turns")
    if args.receipt.exists() or args.receipt.is_symlink():
        raise FileExistsError("activation receipt is create-only")
    if sha256_file(args.goal_contract) != args.goal_contract_sha256:
        raise RuntimeError("goal contract identity drifted")
    manifest = args.deployment_root / "SOURCE_FILES.sha256"
    if sha256_file(manifest) != args.source_manifest_sha256:
        raise RuntimeError("deployment source manifest identity drifted")
    required_sources = {
        "poke_bot/pure_rl/no_progress.py",
        "poke_bot/agent.py",
        "poke_bot/remote_sim_jobs.py",
        "poke_bot/pure_rl/multi_env_self_play.py",
        "poke_bot/pure_rl/expert_rehearsal.py",
        "scripts/train_pure_rl.py",
    }
    manifest_rows = {
        line.split("  ", 1)[1].removeprefix("./"): "sha256:"
        + line.split("  ", 1)[0]
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if "  " in line
    }
    if not required_sources.issubset(manifest_rows):
        raise RuntimeError("deployment manifest lacks required stall sources")
    for relative in sorted(required_sources):
        if sha256_file(args.deployment_root / relative) != manifest_rows[relative]:
            raise RuntimeError(f"deployment source drifted: {relative}")

    if args.wait_for_commit_seconds < 0.0 or args.commit_poll_seconds <= 0.0:
        raise ValueError("commit wait values must be nonnegative with positive poll")
    if args.wait_for_commit_seconds > 0.0:
        commit_deadline = time.monotonic() + args.wait_for_commit_seconds
        next_health_check = 0.0
        while not args.commit.is_file():
            now = time.monotonic()
            if now >= commit_deadline:
                raise TimeoutError("iteration-5 commit did not arrive before deadline")
            if now >= next_health_check:
                if int(service_property(args.training_unit, "MainPID") or "0") != (
                    args.expected_main_pid
                ):
                    raise RuntimeError("trainer PID changed while awaiting commit")
                if service_property(args.training_unit, "ActiveState") != "active":
                    raise RuntimeError("trainer stopped while awaiting commit")
                next_health_check = now + 5.0
            time.sleep(args.commit_poll_seconds)

    commit = _validate_commit(args.commit, args.run_name)
    run_root = args.commit.parent.parent
    next_receipt = run_root / "collection_receipts" / "iter_00006.json"
    next_shard = run_root / "shards" / "iter_00006.jsonl"
    if next_receipt.exists() or next_shard.exists():
        raise RuntimeError("iteration-6 collection already started before activation")
    pre_pid = int(service_property(args.training_unit, "MainPID") or "0")
    if pre_pid != args.expected_main_pid:
        raise RuntimeError("trainer PID changed before boundary activation")
    if service_property(args.training_unit, "ActiveState") != "active":
        raise RuntimeError("trainer is not active at the sealed boundary")

    args.drop_in_destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.drop_in_destination.with_suffix(".tmp")
    shutil.copyfile(args.drop_in_source, temporary)
    os.chmod(temporary, 0o444)
    os.replace(temporary, args.drop_in_destination)
    run(["systemctl", "--user", "daemon-reload"])
    run(["systemctl", "--user", "restart", args.training_unit])

    deadline = time.monotonic() + args.restart_timeout_seconds
    post_pid = 0
    while time.monotonic() < deadline:
        active = service_property(args.training_unit, "ActiveState")
        post_pid = int(service_property(args.training_unit, "MainPID") or "0")
        if active == "active" and post_pid > 0 and post_pid != pre_pid:
            break
        time.sleep(0.5)
    else:
        raise RuntimeError("trainer did not become active after boundary restart")

    active_cwd = Path(f"/proc/{post_pid}/cwd").resolve()
    if active_cwd != args.deployment_root.resolve():
        raise RuntimeError(
            f"trainer restarted from unexpected source: {active_cwd}"
        )
    active_environment = service_property(args.training_unit, "Environment")
    if "PURE_RL_NO_PROGRESS_MAX_TURNS=64" not in active_environment.split():
        raise RuntimeError("trainer restarted without the 64-turn environment gate")
    receipt = {
        "schema": "poke_bot.alakazam_stall_reward_boundary_activation/v1",
        "status": "passed_clean_iteration_5_boundary_before_iteration_6",
        "owner_goal_revision": 31,
        "root_owner_revision": 333,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "activation_boundary_iteration": 5,
        "first_eligible_collection_iteration": 6,
        "boundary_commit_path": str(args.commit.resolve()),
        "boundary_commit_sha256": sha256_file(args.commit),
        "learner_checkpoint_sha256": str(commit["learner"]["digest"]),
        "goal_contract_sha256": sha256_file(args.goal_contract),
        "deployment_root": str(args.deployment_root.resolve()),
        "source_manifest_sha256": sha256_file(manifest),
        "source_module_sha256": manifest_rows[
            "poke_bot/pure_rl/no_progress.py"
        ],
        "maximum_complete_turns_without_win_progress": 64,
        "training_return_override_by_seat": {"0": -1.0, "1": -1.0},
        "historical_shards_relabelled": False,
        "prize_plan_v2_h3_changed": False,
        "runtime_or_submission_path_changed": False,
        "drop_in_sha256": sha256_file(args.drop_in_destination),
        "pre_activation_main_pid": pre_pid,
        "post_activation_main_pid": post_pid,
        "iteration_6_collection_receipt_fields_required": [
            "stall_games",
            "stall_rows_by_seat",
            "stagnant_turn_distribution",
            "training_return_override_counts",
            "ordinary_completed_result_counts",
            "disabled_path_parity",
        ],
    }
    write_json_create_only(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
