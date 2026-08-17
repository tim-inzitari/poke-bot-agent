#!/usr/bin/env python3
"""Apply a staged Inzi/Elmo capacity profile at a committed RL boundary."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


def publish(path: Path, **values: Any) -> None:
    payload = {"schema": "poke_bot.next_iter_capacity/v1", "updated_at": time.time(), **values}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def run(argv: list[str], *, timeout: float = 180.0, check: bool = True) -> str:
    print("+", " ".join(argv), flush=True)
    result = subprocess.run(
        argv,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    if result.stdout:
        print(result.stdout.rstrip(), flush=True)
    if check and result.returncode:
        raise RuntimeError(f"command exited {result.returncode}: {' '.join(argv)}")
    return result.stdout.strip()


def next_iteration(path: Path) -> int:
    try:
        value = json.loads(path.read_text())
        return int(value.get("next_iteration", 0))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 0


def elmo_health() -> str:
    return run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            "elmo",
            "sudo docker inspect --format '{{.State.Health.Status}}' poke-bot-truenas-worker",
        ],
        timeout=12,
        check=False,
    ).strip()


def wait_elmo_healthy(timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if elmo_health() == "healthy":
            return
        time.sleep(2)
    raise RuntimeError(f"Elmo did not become healthy within {timeout_s:.0f}s")


def elmo_compose(directory: str, staged_name: str, *, rollback: bool = False) -> None:
    active = "docker-compose.production.yml"
    backup = "docker-compose.production.pre-capacity.yml"
    if rollback:
        command = (
            f"cd {directory} && sudo cp {backup} {active} && "
            "sudo env POKEBOT_CHECKPOINT_HOST=/mnt/Main/Elmo/poke-bot-agent/containers/truenas-worker/checkpoint "
            "POKEBOT_RUNTIME_HOST=/mnt/Main/Elmo/poke-bot-agent/containers/truenas-worker/runtime-logs "
            f"docker compose -f docker-compose.host.yml -f {active} "
            "up -d --force-recreate worker"
        )
    else:
        command = (
            f"cd {directory} && sudo cp {active} {backup} && "
            f"sudo cp {staged_name} {active} && "
            "sudo env POKEBOT_CHECKPOINT_HOST=/mnt/Main/Elmo/poke-bot-agent/containers/truenas-worker/checkpoint "
            "POKEBOT_RUNTIME_HOST=/mnt/Main/Elmo/poke-bot-agent/containers/truenas-worker/runtime-logs "
            f"docker compose -f docker-compose.host.yml -f {active} "
            "up -d --force-recreate worker"
        )
    run(["ssh", "-o", "BatchMode=yes", "elmo", command], timeout=180)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop-state", type=Path, required=True)
    parser.add_argument("--target-next-iteration", type=int, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--staged-unit", type=Path, required=True)
    parser.add_argument("--active-unit", type=Path, required=True)
    parser.add_argument("--elmo-dir", required=True)
    parser.add_argument("--elmo-staged-name", required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--health-timeout", type=float, default=150.0)
    args = parser.parse_args()

    publish(
        args.status,
        status="waiting_for_committed_boundary",
        target_next_iteration=args.target_next_iteration,
        observed_next_iteration=next_iteration(args.loop_state),
    )
    while next_iteration(args.loop_state) < args.target_next_iteration:
        publish(
            args.status,
            status="waiting_for_committed_boundary",
            target_next_iteration=args.target_next_iteration,
            observed_next_iteration=next_iteration(args.loop_state),
        )
        time.sleep(0.25)

    publish(args.status, status="stopping_production_at_boundary", observed_next_iteration=next_iteration(args.loop_state))
    run(["systemctl", "--user", "stop", args.unit], timeout=60)
    try:
        publish(args.status, status="recreating_elmo_48x2")
        elmo_compose(args.elmo_dir, args.elmo_staged_name)
        wait_elmo_healthy(args.health_timeout)
        publish(args.status, status="installing_inzi_80x24")
        shutil.copy2(args.staged_unit, args.active_unit)
        run(["systemctl", "--user", "daemon-reload"], timeout=30)
        run(["systemctl", "--user", "start", args.unit], timeout=60)
        if run(["systemctl", "--user", "is-active", args.unit], timeout=15).strip() != "active":
            raise RuntimeError("production unit did not become active")
        publish(args.status, status="complete", profile="inzi=80/24 elmo=48/2 batch=256")
        return 0
    except BaseException as exc:
        print(f"capacity activation failed: {type(exc).__name__}: {exc}", flush=True)
        publish(args.status, status="rolling_back", error=f"{type(exc).__name__}: {exc}")
        try:
            elmo_compose(args.elmo_dir, args.elmo_staged_name, rollback=True)
            wait_elmo_healthy(args.health_timeout)
        finally:
            run(["systemctl", "--user", "start", args.unit], timeout=60, check=False)
        publish(args.status, status="rolled_back_and_production_restarted", error=f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
