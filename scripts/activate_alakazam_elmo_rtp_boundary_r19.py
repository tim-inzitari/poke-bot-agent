#!/usr/bin/env python3
"""Admit Elmo and launch RTP shadow continuation at one committed boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import socket
import subprocess
import time
from pathlib import Path


SCHEMA = "poke_bot.alakazam_elmo_rtp_boundary_activation/v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def run(argv: list[str], *, check: bool = True) -> str:
    result = subprocess.run(argv, check=check, text=True, capture_output=True)
    return result.stdout.strip()


def remote(host: str, command: str) -> str:
    return run(["ssh", host, "set -eu; " + command])


def service_property(unit: str, name: str) -> str:
    return run(["systemctl", "--user", "show", unit, f"-p{name}", "--value"])


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", type=Path, required=True)
    parser.add_argument("--expected-main-pid", type=int, required=True)
    parser.add_argument("--training-unit", required=True)
    parser.add_argument("--elmo-host", default="elmo")
    parser.add_argument("--elmo-endpoint", default="elmo:8765")
    parser.add_argument("--compose-host", required=True)
    parser.add_argument("--compose-host-sha256", required=True)
    parser.add_argument("--compose-production", required=True)
    parser.add_argument("--compose-production-sha256", required=True)
    parser.add_argument("--compose-libcg", required=True)
    parser.add_argument("--compose-libcg-sha256", required=True)
    parser.add_argument("--worker-image-sha256", required=True)
    parser.add_argument("--rtp-checkpoint", required=True)
    parser.add_argument("--rtp-checkpoint-sha256", required=True)
    parser.add_argument("--rtp-completion", required=True)
    parser.add_argument("--rtp-completion-sha256", required=True)
    parser.add_argument("--rtp-shadow-unit", required=True)
    parser.add_argument("--rtp-shadow-unit-path", required=True)
    parser.add_argument("--rtp-shadow-unit-sha256", required=True)
    parser.add_argument("--rtp-shadow-output", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    parser.add_argument("--remote-health-timeout-seconds", type=float, default=300.0)
    args = parser.parse_args()

    if args.receipt.exists() or args.receipt.is_symlink():
        raise FileExistsError("activation receipt is create-only")
    while not args.commit.is_file():
        current = int(service_property(args.training_unit, "MainPID") or "0")
        if current != args.expected_main_pid:
            raise RuntimeError("training PID changed before boundary commit")
        time.sleep(args.poll_seconds)

    pre_pid = int(service_property(args.training_unit, "MainPID") or "0")
    if pre_pid != args.expected_main_pid:
        raise RuntimeError("training PID changed at boundary")
    if service_property(args.training_unit, "ActiveState") != "active":
        raise RuntimeError("training service is not active at boundary")
    unit_text = run(["systemctl", "--user", "cat", args.training_unit])
    if f"--remote-worker-endpoints {args.elmo_endpoint}" not in unit_text:
        raise RuntimeError("next-start training unit does not bind the Elmo endpoint")

    remote_hashes = remote(
        args.elmo_host,
        "sha256sum "
        + " ".join(
            q(path)
            for path in (
                args.compose_host,
                args.compose_production,
                args.compose_libcg,
                args.rtp_checkpoint,
                args.rtp_completion,
                args.rtp_shadow_unit_path,
            )
        ),
    ).splitlines()
    observed = ["sha256:" + line.split()[0] for line in remote_hashes]
    expected = [
        args.compose_host_sha256,
        args.compose_production_sha256,
        args.compose_libcg_sha256,
        args.rtp_checkpoint_sha256,
        args.rtp_completion_sha256,
        args.rtp_shadow_unit_sha256,
    ]
    if observed != expected:
        raise RuntimeError(f"Elmo boundary identity mismatch: {observed!r}")
    image = remote(
        args.elmo_host,
        "sudo -n docker inspect poke-bot-truenas-worker --format '{{.Image}}'",
    )
    if "sha256:" + image.removeprefix("sha256:") != args.worker_image_sha256:
        raise RuntimeError("Elmo worker image identity changed")

    compose = "sudo -n docker compose " + " ".join(
        f"-f {q(path)}"
        for path in (args.compose_host, args.compose_production, args.compose_libcg)
    )
    remote(args.elmo_host, compose + " up -d --no-build worker")
    deadline = time.time() + args.remote_health_timeout_seconds
    health = ""
    while time.time() < deadline:
        health = remote(
            args.elmo_host,
            "sudo -n docker inspect poke-bot-truenas-worker "
            "--format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}'",
        )
        if health == "healthy":
            break
        time.sleep(2.0)
    if health != "healthy":
        raise RuntimeError(f"Elmo worker did not become healthy: {health}")
    endpoint_host, endpoint_port = args.elmo_endpoint.rsplit(":", 1)
    with socket.create_connection((endpoint_host, int(endpoint_port)), timeout=5.0):
        pass

    remote(
        args.elmo_host,
        f"test ! -e {q(args.rtp_shadow_output)}; "
        f"mkdir -m 0755 {q(args.rtp_shadow_output)}; "
        f"systemctl --user reset-failed {q(args.rtp_shadow_unit)} 2>/dev/null || true; "
        f"systemctl --user start --no-block {q(args.rtp_shadow_unit)}; "
        f"state=$(systemctl --user show {q(args.rtp_shadow_unit)} -p ActiveState --value); "
        "test \"$state\" = activating -o \"$state\" = active",
    )

    run(["systemctl", "--user", "restart", args.training_unit])
    deadline = time.time() + 120.0
    post_pid = 0
    while time.time() < deadline:
        post_pid = int(service_property(args.training_unit, "MainPID") or "0")
        if (
            post_pid > 0
            and post_pid != pre_pid
            and service_property(args.training_unit, "ActiveState") == "active"
        ):
            break
        time.sleep(1.0)
    if post_pid < 1 or post_pid == pre_pid:
        raise RuntimeError("boundary restart did not produce a new active learner PID")
    time.sleep(10.0)
    if (
        service_property(args.training_unit, "ActiveState") != "active"
        or int(service_property(args.training_unit, "MainPID") or "0") != post_pid
    ):
        raise RuntimeError("restarted learner did not remain stable after Elmo admission")

    receipt = {
        "schema": SCHEMA,
        "goal_revision": 19,
        "root_goal_revision": 321,
        "boundary_commit_path": str(args.commit.resolve()),
        "boundary_commit_sha256": sha256_file(args.commit),
        "pre_boundary_main_pid": pre_pid,
        "post_boundary_main_pid": post_pid,
        "elmo_endpoint": args.elmo_endpoint,
        "elmo_worker_health_before_restart": health,
        "elmo_worker_image_sha256": args.worker_image_sha256,
        "elmo_compose_sha256s": expected[:3],
        "rtp_bootstrap_checkpoint_sha256": args.rtp_checkpoint_sha256,
        "rtp_bootstrap_completion_sha256": args.rtp_completion_sha256,
        "rtp_shadow_unit": args.rtp_shadow_unit,
        "rtp_shadow_unit_sha256": args.rtp_shadow_unit_sha256,
        "rtp_shadow_output": args.rtp_shadow_output,
        "rtp_shadow_action_authority": False,
        "policy_checkpoint_mutated_by_activation": False,
        "mid_collection_restart_performed": False,
        "activated_at_unix_seconds": time.time(),
    }
    body = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(fd, body)
        os.fsync(fd)
    finally:
        os.close(fd)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
