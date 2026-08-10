#!/usr/bin/env python3
"""Diagnose and repair Elmo worker runtime baseline visibility for b3307."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ELMO = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "admin@192.168.1.143"]
BASE_ID = "specialist-marnie-final-format-h10-b3307cf1bd67"
SHORT = "marnie-final-format-h10-b3307cf1bd67"
HOST_SYNC = "/mnt/Main/main/poke-bot-agent/containers/truenas-worker/baseline-sync"


def sh(cmd: str) -> str:
    print("+", cmd, flush=True)
    return subprocess.check_output(ELMO + [cmd], text=True)


def main() -> int:
    print(sh(f"ls -la {HOST_SYNC}/specialists/{SHORT}/model.pt; grep -c b3307cf1bd67 {HOST_SYNC}/manifest.json || true"))
    mounts = sh(
        "sudo docker inspect poke-bot-truenas-worker --format '{{json .Mounts}}'"
    ).strip()
    for m in json.loads(mounts):
        print(f"mount {m.get('Source')} -> {m.get('Destination')} ({m.get('Type')})")

    # Probe common in-container locations.
    probe = sh(
        "sudo docker exec poke-bot-truenas-worker sh -lc "
        "'for p in /baselines/manifest.json /baseline-sync/manifest.json "
        "/app/baselines/manifest.json /data/baselines/manifest.json "
        "/workspace/baselines/manifest.json; do "
        "if [ -f \"$p\" ]; then echo FOUND:$p; "
        "grep -c b3307cf1bd67 \"$p\" || true; "
        "grep -c f20efb20f5c3 \"$p\" || true; "
        "ls \"$(dirname \"$p\")/specialists\" 2>/dev/null | grep marnie-final | head; "
        "fi; done; "
        "echo ENV; env | grep -i baseline || true; "
        "ls -la /baselines /baseline-sync 2>/dev/null | head'"
    )
    print(probe)

    # If host sync is mounted but container still lacks id, recreate compose service.
    # Prefer compose recreate of the worker service.
    compose_dir = "/mnt/Main/main/poke-bot-agent/containers/truenas-worker"
    try:
        print(
            sh(
                f"cd {compose_dir} && sudo docker compose ps && "
                f"sudo docker compose up -d --force-recreate --no-deps "
                f"$(sudo docker compose config --services | head -1)"
            )
        )
    except subprocess.CalledProcessError:
        print(sh("sudo docker restart poke-bot-truenas-worker"))

    import time

    time.sleep(12)
    probe2 = sh(
        "sudo docker exec poke-bot-truenas-worker sh -lc "
        "'for p in /baselines/manifest.json /baseline-sync/manifest.json; do "
        "if [ -f \"$p\" ]; then echo FOUND:$p; grep -c b3307cf1bd67 \"$p\" || true; fi; done'"
    )
    print(probe2)
    if "b3307cf1bd67" not in probe2 and "FOUND:" in probe2:
        # Explicitly copy host manifest into whichever mount destination is writable.
        # If mounts are bind mounts from HOST_SYNC, container should already see it.
        # Fall back: write into host path that maps to the discovered FOUND path.
        found_lines = [ln for ln in probe2.splitlines() if ln.startswith("FOUND:")]
        print("still missing inside container after recreate; found=", found_lines)
        # Dump host manifest ids for proof
        print(sh(f"python3 -c \"import json; m=json.load(open('{HOST_SYNC}/manifest.json')); print('host_keys_sample', [k for k in (m if isinstance(m,dict) else {})][:12]); print('has', 'b3307cf1bd67' in open('{HOST_SYNC}/manifest.json').read())\""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
