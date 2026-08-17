#!/usr/bin/env python3
"""Print exactly ADMITTED only when an r229 host has protected spare capacity."""

from __future__ import annotations

import argparse
import json
import subprocess
from typing import Sequence


def _endpoint(endpoint: str) -> bool:
    from poke_bot.remote_jobs import RemoteJobClient, parse_endpoint

    client = RemoteJobClient(*parse_endpoint(endpoint), connect_timeout_s=5, control_timeout_s=10)
    try:
        info = client.connect()
        health = client.health()
    finally:
        client.close()
    return (
        int(health.get("active_jobs") or 0) == 0
        and not health.get("error")
        and int(info.workers) > 0
        and (info.free_ram_gb is None or float(info.free_ram_gb) >= 8.0)
    )


def _train(host: str | None, gpu_uuid: str) -> bool:
    script = r'''set -eu
state=$(systemctl --user show pokebot-pure-rl-specialist.service -p ActiveState --value 2>/dev/null || true)
test "$state" != active
nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits
'''
    command = (
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host, script]
        if host
        else ["bash", "-lc", script]
    )
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=15, check=False)
    if result.returncode != 0:
        return False
    for line in result.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 3 and fields[0] == gpu_uuid:
            return int(fields[1]) <= 2048 and int(fields[2]) <= 20
    return False


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--endpoint")
    group.add_argument("--train-host")
    group.add_argument("--train-local", action="store_true")
    parser.add_argument("--gpu-uuid")
    args = parser.parse_args(argv)
    try:
        admitted = (
            _endpoint(args.endpoint)
            if args.endpoint
            else bool(args.gpu_uuid) and _train(args.train_host, args.gpu_uuid)
        )
    except Exception:
        admitted = False
    if admitted:
        print("ADMITTED")
        return 0
    print("REFUSED")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
