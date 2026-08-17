#!/usr/bin/env python3
"""Adopt the opt-in GPS fast path at a sealed post-r336 boundary."""

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
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _property(unit: str, name: str) -> str:
    result = subprocess.run(
        ["systemctl", "--user", "show", unit, f"--property={name}", "--value"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _write_create_only(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r336-receipt", type=Path, required=True)
    parser.add_argument("--commit", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=5)
    parser.add_argument("--training-unit", required=True)
    parser.add_argument("--deployment-root", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--prestage-receipt", type=Path, required=True)
    parser.add_argument("--prestage-receipt-sha256", required=True)
    parser.add_argument("--drop-in-source", type=Path, required=True)
    parser.add_argument("--drop-in-sha256", required=True)
    parser.add_argument("--drop-in-destination", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args()

    while not args.r336_receipt.is_file():
        time.sleep(args.poll_seconds)
    predecessor = json.loads(args.r336_receipt.read_text(encoding="utf-8"))
    if (
        predecessor.get("status") != "activated_at_clean_iteration_boundary"
        or int(predecessor.get("boundary_iteration", -1)) != 4
    ):
        raise RuntimeError("r336 predecessor receipt is not the sealed iteration-4 adoption")
    expected_pid = int(predecessor.get("post_activation_main_pid", 0))
    if expected_pid <= 0:
        raise RuntimeError("r336 predecessor receipt lacks its post-activation PID")

    while not args.commit.is_file():
        current_pid = int(_property(args.training_unit, "MainPID") or "0")
        if current_pid != expected_pid or _property(args.training_unit, "ActiveState") != "active":
            raise RuntimeError("trainer changed before the r337 clean boundary")
        time.sleep(args.poll_seconds)
    commit = json.loads(args.commit.read_text(encoding="utf-8"))
    if (
        int(commit.get("iteration", -1)) != args.iteration
        or int(commit.get("next_iteration", -1)) != args.iteration + 1
    ):
        raise RuntimeError("r337 boundary file has the wrong iteration")

    manifest = args.deployment_root / "SOURCE_FILES.sha256"
    expected = {
        manifest: args.source_manifest_sha256,
        args.prestage_receipt: args.prestage_receipt_sha256,
        args.drop_in_source: args.drop_in_sha256,
    }
    for path, digest in expected.items():
        if _sha256(path) != str(digest):
            raise RuntimeError(f"r337 immutable input changed: {path}")

    pre_pid = int(_property(args.training_unit, "MainPID") or "0")
    if pre_pid != expected_pid:
        raise RuntimeError("trainer PID changed before r337 activation")
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
        raise RuntimeError("r337 trainer restart did not become healthy")
    if Path(f"/proc/{post_pid}/cwd").resolve() != args.deployment_root.resolve():
        raise RuntimeError("r337 restart did not adopt its sealed deployment")
    environment = Path(f"/proc/{post_pid}/environ").read_bytes().split(b"\0")
    required = {
        b"PURE_RL_SHARD_WRITE_BUFFER_BYTES=16777216",
        b"POKEBOT_IMMUTABLE_DECK_ENCODING_CACHE=1",
    }
    if not required.issubset(set(environment)):
        raise RuntimeError("r337 restart did not adopt both throughput knobs")

    receipt: dict[str, object] = {
        "schema": "poke_bot.alakazam_rule_derivative_gps_throughput_activation/v1",
        "status": "activated_at_clean_iteration_boundary",
        "boundary_iteration": args.iteration,
        "first_gps_trial_iteration": args.iteration + 1,
        "boundary_commit_sha256": _sha256(args.commit),
        "r336_receipt_sha256": _sha256(args.r336_receipt),
        "source_manifest_sha256": _sha256(manifest),
        "prestage_receipt_sha256": _sha256(args.prestage_receipt),
        "drop_in_sha256": _sha256(args.drop_in_destination),
        "pre_activation_main_pid": pre_pid,
        "post_activation_main_pid": post_pid,
        "shard_write_buffer_bytes": 16777216,
        "immutable_deck_encoding_cache": True,
        "reward_semantics_changed": False,
        "replay_membership_changed": False,
        "r274_or_production_changed": False,
    }
    _write_create_only(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
