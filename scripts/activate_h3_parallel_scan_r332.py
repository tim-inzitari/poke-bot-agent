#!/usr/bin/env python3
"""Adopt parallel H3 replay scanning after an exact iteration commit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _run(argv: list[str]) -> str:
    return subprocess.run(
        argv, check=True, text=True, capture_output=True
    ).stdout.strip()


def _property(unit: str, name: str) -> str:
    return _run(["systemctl", "--user", "show", unit, f"-p{name}", "--value"])


def _write_create_only(path: Path, payload: dict[str, object]) -> None:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(descriptor, body)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--expected-main-pid", type=int, required=True)
    parser.add_argument("--training-unit", required=True)
    parser.add_argument("--deployment-root", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--module-sha256", required=True)
    parser.add_argument("--benchmark-receipt", type=Path, required=True)
    parser.add_argument("--benchmark-receipt-sha256", required=True)
    parser.add_argument("--drop-in-source", type=Path, required=True)
    parser.add_argument("--drop-in-destination", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args()

    manifest = args.deployment_root / "SOURCE_FILES.sha256"
    module = args.deployment_root / "poke_bot/prize_plan_live_cache.py"
    expected = (
        (manifest, args.source_manifest_sha256),
        (module, args.module_sha256),
        (args.benchmark_receipt, args.benchmark_receipt_sha256),
    )
    for path, digest in expected:
        if not path.is_file() or path.is_symlink() or _sha256(path) != digest:
            raise RuntimeError(f"parallel H3 activation input is stale: {path}")
    if args.workers != 8:
        raise RuntimeError("the measured H3 worker count is exactly eight")
    if args.receipt.exists() or args.receipt.is_symlink():
        raise FileExistsError("parallel H3 activation receipt is create-only")

    while not args.commit.is_file():
        current = int(_property(args.training_unit, "MainPID") or "0")
        if current != args.expected_main_pid:
            raise RuntimeError("trainer PID changed before the requested commit")
        time.sleep(args.poll_seconds)

    commit = json.loads(args.commit.read_text(encoding="utf-8"))
    if (
        int(commit.get("iteration", -1)) != args.iteration
        or int(commit.get("next_iteration", -1)) != args.iteration + 1
    ):
        raise RuntimeError("boundary file is not the requested iteration commit")
    pre_pid = int(_property(args.training_unit, "MainPID") or "0")
    if pre_pid != args.expected_main_pid or _property(args.training_unit, "ActiveState") != "active":
        raise RuntimeError("trainer changed before parallel H3 adoption")

    args.drop_in_destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.drop_in_destination.with_suffix(".tmp")
    shutil.copyfile(args.drop_in_source, temporary)
    os.chmod(temporary, 0o444)
    os.replace(temporary, args.drop_in_destination)
    _run(["systemctl", "--user", "daemon-reload"])
    _run(["systemctl", "--user", "restart", args.training_unit])

    post_pid = 0
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        post_pid = int(_property(args.training_unit, "MainPID") or "0")
        if post_pid and post_pid != pre_pid and _property(args.training_unit, "ActiveState") == "active":
            break
        time.sleep(0.25)
    else:
        raise RuntimeError("parallel H3 trainer restart did not become healthy")
    if Path(f"/proc/{post_pid}/cwd").resolve() != args.deployment_root.resolve():
        raise RuntimeError("parallel H3 restart did not adopt the sealed deployment")

    benchmark = json.loads(args.benchmark_receipt.read_text(encoding="utf-8"))
    receipt: dict[str, object] = {
        "schema": "poke_bot.alakazam_prize_plan_h3_parallel_scan_activation/v1",
        "status": "activated_at_clean_iteration_boundary",
        "boundary_iteration": args.iteration,
        "first_parallel_scan_iteration": args.iteration + 1,
        "boundary_commit_sha256": _sha256(args.commit),
        "source_manifest_sha256": _sha256(manifest),
        "module_sha256": _sha256(module),
        "benchmark_receipt_sha256": _sha256(args.benchmark_receipt),
        "measured_speedup": (benchmark.get("measurement") or {}).get("speedup"),
        "workers": args.workers,
        "pre_activation_main_pid": pre_pid,
        "post_activation_main_pid": post_pid,
        "drop_in_sha256": _sha256(args.drop_in_destination),
        "reward_semantics_changed": False,
        "replay_membership_changed": False,
        "r274_or_production_changed": False,
    }
    _write_create_only(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
