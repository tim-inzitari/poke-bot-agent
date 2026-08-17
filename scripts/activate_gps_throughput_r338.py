#!/usr/bin/env python3
"""Activate the owner-authorized GPS fast path at sealed iteration 4."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def prop(unit: str, name: str) -> str:
    return subprocess.run(
        ["systemctl", "--user", "show", unit, f"--property={name}", "--value"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def install_dropin(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    shutil.copyfile(source, temporary)
    os.chmod(temporary, 0o444)
    os.replace(temporary, destination)


def write_create_only(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", type=Path, required=True)
    parser.add_argument("--training-unit", required=True)
    parser.add_argument("--deployment-root", type=Path, required=True)
    parser.add_argument("--gps-dropin-source", type=Path, required=True)
    parser.add_argument("--gps-dropin-destination", type=Path, required=True)
    parser.add_argument("--submission-receipt", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    commit = json.loads(args.commit.read_text(encoding="utf-8"))
    if (
        int(commit.get("last_completed_iteration", -1)) != 4
        or int(commit.get("next_iteration", -1)) != 5
    ):
        raise RuntimeError("iteration-4 durable commit is not sealed")
    learner = dict(commit.get("learner") or {})
    checkpoint = Path(str(learner.get("path") or ""))
    if not checkpoint.is_file() or sha256(checkpoint) != learner.get("digest"):
        raise RuntimeError("iteration-4 learner identity is invalid")
    submission = json.loads(args.submission_receipt.read_text(encoding="utf-8"))
    if (
        submission.get("status") != "queued"
        or submission.get("checkpoint_sha256") != learner.get("digest")
    ):
        raise RuntimeError("iteration-4 submission receipt is not bound to the learner")
    for path in (
        args.deployment_root / "SOURCE_FILES.sha256",
        args.gps_dropin_source,
    ):
        if not path.is_file():
            raise RuntimeError(f"missing activation input: {path}")

    install_dropin(args.gps_dropin_source, args.gps_dropin_destination)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "restart", args.training_unit], check=True)

    deadline = time.monotonic() + 90
    post_pid = 0
    while time.monotonic() < deadline:
        post_pid = int(prop(args.training_unit, "MainPID") or "0")
        if post_pid and prop(args.training_unit, "ActiveState") == "active":
            break
        time.sleep(0.5)
    else:
        raise RuntimeError("trainer did not become healthy after GPS activation")
    if Path(f"/proc/{post_pid}/cwd").resolve() != args.deployment_root.resolve():
        raise RuntimeError("trainer did not adopt the GPS deployment")
    environment = set(Path(f"/proc/{post_pid}/environ").read_bytes().split(b"\0"))
    required = {
        b"PURE_RL_SHARD_WRITE_BUFFER_BYTES=16777216",
        b"POKEBOT_IMMUTABLE_DECK_ENCODING_CACHE=1",
    }
    if not required.issubset(environment):
        raise RuntimeError("trainer did not adopt the complete throughput environment")

    payload: dict[str, object] = {
        "schema": "poke_bot.alakazam_rule_derivative_gps_throughput_activation/v2",
        "status": "owner_authorized_activated_at_sealed_iteration_4",
        "boundary_iteration": 4,
        "first_gps_trial_iteration": 5,
        "boundary_commit_sha256": sha256(args.commit),
        "learner_checkpoint_sha256": learner["digest"],
        "submission_receipt_sha256": sha256(args.submission_receipt),
        "deployment_manifest_sha256": sha256(
            args.deployment_root / "SOURCE_FILES.sha256"
        ),
        "gps_dropin_sha256": sha256(args.gps_dropin_destination),
        "post_activation_main_pid": post_pid,
        "shard_write_buffer_bytes": 16777216,
        "immutable_deck_encoding_cache": True,
        "minimum_free_disk_gb": 100,
        "reward_semantics_changed": False,
        "replay_membership_changed": False,
        "r274_or_production_changed": False,
    }
    write_create_only(args.receipt, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
