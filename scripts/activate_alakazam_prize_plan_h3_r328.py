#!/usr/bin/env python3
"""Adopt the revision-26 H3 actor reward after iteration 3 commits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def run(argv: list[str]) -> str:
    return subprocess.run(argv, check=True, text=True, capture_output=True).stdout.strip()


def service_property(unit: str, name: str) -> str:
    return run(["systemctl", "--user", "show", unit, f"-p{name}", "--value"])


def write_json_create_only(path: Path, payload: dict[str, object]) -> None:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(fd, body)
        os.fsync(fd)
    finally:
        os.close(fd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", type=Path, required=True)
    parser.add_argument("--expected-main-pid", type=int, required=True)
    parser.add_argument("--training-unit", required=True)
    parser.add_argument("--drop-in-source", type=Path, required=True)
    parser.add_argument("--drop-in-destination", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--sidecar-sha256", required=True)
    parser.add_argument("--scale-support", type=Path, required=True)
    parser.add_argument("--scale-support-sha256", required=True)
    parser.add_argument("--c3-support", type=Path, required=True)
    parser.add_argument("--c3-support-sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()

    expected = {
        args.contract: args.contract_sha256,
        args.sidecar: args.sidecar_sha256,
        args.scale_support: args.scale_support_sha256,
        args.c3_support: args.c3_support_sha256,
    }
    for path, digest in expected.items():
        if not path.is_file() or path.is_symlink() or sha256_file(path) != digest:
            raise RuntimeError(f"missing or stale activation input: {path}")
    if args.receipt.exists() or args.receipt.is_symlink():
        raise FileExistsError("activation receipt is create-only")

    while not args.commit.is_file():
        if int(service_property(args.training_unit, "MainPID") or "0") != args.expected_main_pid:
            raise RuntimeError("trainer PID changed before iteration-3 commit")
        time.sleep(args.poll_seconds)

    commit = json.loads(args.commit.read_text(encoding="utf-8"))
    if int(commit.get("iteration", 3)) != 3 or int(commit.get("next_iteration", 4)) != 4:
        raise RuntimeError("boundary file is not the iteration-3 commit")
    pre_pid = int(service_property(args.training_unit, "MainPID") or "0")
    if pre_pid != args.expected_main_pid or service_property(args.training_unit, "ActiveState") != "active":
        raise RuntimeError("trainer changed before clean-boundary adoption")

    args.drop_in_destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.drop_in_destination.with_suffix(".tmp")
    shutil.copyfile(args.drop_in_source, temporary)
    os.chmod(temporary, 0o444)
    os.replace(temporary, args.drop_in_destination)
    run(["systemctl", "--user", "daemon-reload"])

    receipt: dict[str, object] = {
        "schema": "poke_bot.alakazam_prize_plan_h3_boundary_activation/v1",
        "status": "authorized_at_clean_iteration_3_boundary",
        "owner_goal_revision": 26,
        "root_handoff_revision": 328,
        "boundary_commit_sha256": sha256_file(args.commit),
        "contract_sha256": sha256_file(args.contract),
        "sidecar_sha256": sha256_file(args.sidecar),
        "scale_support_sha256": sha256_file(args.scale_support),
        "c3_support_sha256": sha256_file(args.c3_support),
        "drop_in_sha256": sha256_file(args.drop_in_destination),
        "pre_activation_main_pid": pre_pid,
        "first_rewarded_optimizer_iteration": 4,
        "reward_formula": "(z-V_existing)+0.025*m3*c3*(Q_plan_3-V_plan_3)",
        "runtime_or_submission_critic": False,
    }
    write_json_create_only(args.receipt, receipt)
    run(["systemctl", "--user", "restart", args.training_unit])
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
