#!/usr/bin/env python3
"""Activate Elmo's bounded service rotation at a committed RL boundary."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _publish(path: Path, **payload: Any) -> None:
    _atomic_json(
        path,
        {
            "schema": "poke_bot.elmo_throughput_rotation_boundary/v1",
            "updated_at_utc": _utc_now(),
            **payload,
        },
    )


def _run(
    argv: list[str],
    *,
    timeout: float = 180.0,
    check: bool = True,
) -> str:
    result = subprocess.run(
        argv,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if result.stdout:
        print(result.stdout.rstrip(), flush=True)
    if check and result.returncode:
        raise RuntimeError(
            f"command exited {result.returncode}: {shlex.join(argv)}"
        )
    return result.stdout.strip()


def _ssh(host: str, command: str, *, timeout: float = 180.0) -> str:
    return _run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            host,
            command,
        ],
        timeout=timeout,
    )


def _next_iteration(path: Path) -> int:
    try:
        return int(json.loads(path.read_text(encoding="utf-8")).get("next_iteration", -1))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return -1


def _remote_sha256(host: str, path: str) -> str:
    output = _ssh(host, f"sha256sum {shlex.quote(path)}", timeout=20)
    return output.split()[0] if output else ""


def _remote_health(host: str) -> dict[str, Any]:
    source = (
        "import json;"
        "from poke_bot.remote_jobs import RemoteJobClient;"
        "c=RemoteJobClient('127.0.0.1',8765,timeout_s=10,"
        "connect_timeout_s=5,control_timeout_s=8);"
        "c.connect();h=c.health();"
        "print(json.dumps({k:h.get(k) for k in "
        "('ok','controller_healthy','leaf_alive','leaf_identity_ok',"
        "'accepting_jobs','active_jobs','jobs_completed','checkpoint_digest',"
        "'tree_rss_gb','tree_rss_limit_gb','free_ram_gb')}));"
        "c.close()"
    )
    output = _ssh(
        host,
        "sudo -n docker exec poke-bot-truenas-worker "
        f"python -c {shlex.quote(source)}",
        timeout=30,
    )
    line = output.splitlines()[-1] if output else ""
    value = json.loads(line)
    if not isinstance(value, dict):
        raise RuntimeError("Elmo health response was not an object")
    return value


def _wait_healthy(
    host: str,
    *,
    timeout_s: float,
    require_idle: bool,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        try:
            health = _remote_health(host)
            healthy = all(
                bool(health.get(key))
                for key in (
                    "ok",
                    "controller_healthy",
                    "leaf_alive",
                    "leaf_identity_ok",
                )
            )
            idle = int(health.get("active_jobs") or 0) == 0
            if healthy and (idle or not require_idle):
                return health
            last_error = f"healthy={healthy} active_jobs={health.get('active_jobs')}"
        except Exception as exc:  # noqa: BLE001 - bounded health polling
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(0.5)
    raise RuntimeError(f"Elmo health deadline expired: {last_error}")


def _compose_command(
    directory: str,
    *,
    action: str,
) -> str:
    prefix = (
        f"cd {shlex.quote(directory)} && "
        "sudo -n env "
        "POKEBOT_CHECKPOINT_HOST="
        "/mnt/Main/Elmo/poke-bot-agent/containers/truenas-worker/checkpoint "
        "POKEBOT_RUNTIME_HOST="
        "/mnt/Main/Elmo/poke-bot-agent/containers/truenas-worker/runtime-logs "
        "docker compose -f docker-compose.host.yml "
        "-f docker-compose.production.yml "
    )
    return prefix + action


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop-state", required=True, type=Path)
    parser.add_argument("--target-next-iteration", required=True, type=int)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--remote-directory", required=True)
    parser.add_argument("--staged-compose-name", required=True)
    parser.add_argument("--expected-staged-sha256", required=True)
    parser.add_argument("--expected-rotation-jobs", required=True, type=int)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--maintenance-lock", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    parser.add_argument("--health-timeout-seconds", type=float, default=150.0)
    args = parser.parse_args()

    if args.expected_rotation_jobs < 512 or args.expected_rotation_jobs > 1024:
        raise ValueError("rotation jobs must remain within the production allowlist")
    expected_sha = args.expected_staged_sha256.removeprefix("sha256:")
    remote_dir = args.remote_directory.rstrip("/")
    staged = f"{remote_dir}/{args.staged_compose_name}"
    active = f"{remote_dir}/docker-compose.production.yml"
    backup = f"{remote_dir}/docker-compose.production.pre-rev60.yml"

    while _next_iteration(args.loop_state) < args.target_next_iteration:
        _publish(
            args.status,
            status="waiting_for_committed_boundary",
            observed_next_iteration=_next_iteration(args.loop_state),
            target_next_iteration=args.target_next_iteration,
            expected_rotation_jobs=args.expected_rotation_jobs,
        )
        time.sleep(max(0.1, args.poll_seconds))

    args.maintenance_lock.parent.mkdir(parents=True, exist_ok=True)
    with args.maintenance_lock.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        observed = _next_iteration(args.loop_state)
        staged_sha = _remote_sha256(args.remote_host, staged)
        if staged_sha != expected_sha:
            raise RuntimeError(
                f"staged compose digest mismatch: {staged_sha} != {expected_sha}"
            )
        staged_text = _ssh(
            args.remote_host,
            f"sed -n '1,180p' {shlex.quote(staged)}",
            timeout=20,
        )
        expected_line = (
            f'POKEBOT_REMOTE_MAX_SERVICE_JOBS: "{args.expected_rotation_jobs}"'
        )
        for required in (
            expected_line,
            'POKEBOT_REMOTE_TREE_RSS_LIMIT_GB: "30"',
            "mem_limit: 64g",
            "memswap_limit: 64g",
            'restart: "on-failure:3"',
        ):
            if required not in staged_text:
                raise RuntimeError(f"staged compose lost required contract: {required}")

        _publish(
            args.status,
            status="waiting_for_idle_elmo_boundary",
            observed_next_iteration=observed,
            target_next_iteration=args.target_next_iteration,
            staged_compose_sha256=f"sha256:{staged_sha}",
        )
        before = _wait_healthy(
            args.remote_host,
            timeout_s=20.0,
            require_idle=True,
        )
        before_digest = str(before.get("checkpoint_digest") or "")
        if not before_digest.startswith("sha256:"):
            raise RuntimeError("pre-activation checkpoint digest is missing")

        _publish(
            args.status,
            status="activating_managed_elmo_rotation",
            observed_next_iteration=observed,
            pre_activation_health=before,
            staged_compose_sha256=f"sha256:{staged_sha}",
        )
        install = (
            f"sudo -n cp -p {shlex.quote(active)} {shlex.quote(backup)} && "
            f"sudo -n install -m 0644 {shlex.quote(staged)} {shlex.quote(active)}"
        )
        _ssh(args.remote_host, install, timeout=30)

        try:
            _ssh(
                args.remote_host,
                _compose_command(
                    remote_dir,
                    action="config --quiet",
                ),
                timeout=30,
            )
            _ssh(
                args.remote_host,
                _compose_command(
                    remote_dir,
                    action="up -d --force-recreate --wait worker",
                ),
                timeout=180,
            )
            after = _wait_healthy(
                args.remote_host,
                timeout_s=args.health_timeout_seconds,
                require_idle=False,
            )
            if str(after.get("checkpoint_digest") or "") != before_digest:
                raise RuntimeError(
                    "Elmo checkpoint digest changed across managed recreation"
                )
            environment = _ssh(
                args.remote_host,
                "sudo -n docker inspect --format '{{json .Config.Env}}' "
                "poke-bot-truenas-worker",
                timeout=20,
            )
            if (
                f"POKEBOT_REMOTE_MAX_SERVICE_JOBS={args.expected_rotation_jobs}"
                not in environment
            ):
                raise RuntimeError("running Elmo container did not adopt rotation limit")
        except BaseException as exc:
            _publish(
                args.status,
                status="rolling_back",
                observed_next_iteration=observed,
                error=f"{type(exc).__name__}: {exc}",
            )
            rollback = (
                f"sudo -n install -m 0644 {shlex.quote(backup)} {shlex.quote(active)}"
            )
            _ssh(args.remote_host, rollback, timeout=30)
            _ssh(
                args.remote_host,
                _compose_command(
                    remote_dir,
                    action="up -d --force-recreate --wait worker",
                ),
                timeout=180,
            )
            _wait_healthy(
                args.remote_host,
                timeout_s=args.health_timeout_seconds,
                require_idle=False,
            )
            raise

        _publish(
            args.status,
            status="activated",
            activated_at_utc=_utc_now(),
            observed_next_iteration=observed,
            target_next_iteration=args.target_next_iteration,
            expected_rotation_jobs=args.expected_rotation_jobs,
            staged_compose_sha256=f"sha256:{staged_sha}",
            active_compose_sha256=f"sha256:{_remote_sha256(args.remote_host, active)}",
            checkpoint_digest_preserved=before_digest,
            pre_activation_health=before,
            post_activation_health=after,
            local_trainer_restart_performed=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
