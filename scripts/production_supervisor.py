#!/usr/bin/env python3
"""Conservative supervisor for the canonical sequential-specialist pipeline.

This supervisor only operates declared user services.  It never inspects,
signals, or terminates interactive processes.  Successful terminal trainer
exits are routed through the idempotent gate handler instead of being restarted.
"""

from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import socket
import shutil
import subprocess
import urllib.request


TRAINER = "pokebot-pure-rl-trevenant-staged.service"
GATE_HANDLER = "pokebot-specialist-passed-gate-handler.service"
HANDOFF = "pokebot-specialist-cycle-handoff.service"
KAGGLE_QUEUE = "pokebot-kaggle-submission-queue.service"
ROOT = Path("/home/inzi/poke-bot-agent")
STATE = ROOT / "outputs/state/production-supervisor.json"
LOCK = Path("/home/inzi/.local/state/pokebot/production-supervisor.lock")
MIN_FREE_GIB = 120.0


def systemctl(*args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["systemctl", "--user", *args],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )


def unit_state(unit: str) -> dict[str, str]:
    result = systemctl(
        "show",
        unit,
        "-p",
        "ActiveState",
        "-p",
        "SubState",
        "-p",
        "Result",
        "-p",
        "ExecMainStatus",
    )
    fields: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key] = value
    return fields


def endpoint_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            return 200 <= response.status < 500
    except Exception:
        return False


def tcp_endpoint_ok(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=3):
            return True
    except OSError:
        return False


def main() -> int:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0

        units = {
            "trainer": unit_state(TRAINER),
            "gate_handler": unit_state(GATE_HANDLER),
            "handoff": unit_state(HANDOFF),
            "kaggle_queue": unit_state(KAGGLE_QUEUE),
        }
        free_gib = shutil.disk_usage(ROOT).free / (1024**3)
        actions: list[str] = []

        # Each service already has Restart=on-failure.  This reset/start path
        # covers exhausted start limits and manager restarts.
        for key, unit in (
            ("handoff", HANDOFF),
            ("gate_handler", GATE_HANDLER),
            ("kaggle_queue", KAGGLE_QUEUE),
        ):
            if units[key].get("ActiveState") == "failed":
                systemctl("reset-failed", unit)
                systemctl("start", unit)
                actions.append(f"recovered_failed:{unit}")

        trainer = units["trainer"]
        if trainer.get("ActiveState") == "failed" and free_gib >= MIN_FREE_GIB:
            systemctl("reset-failed", TRAINER)
            systemctl("start", TRAINER)
            actions.append(f"recovered_failed:{TRAINER}")

        active_states = {
            units[name].get("ActiveState") for name in ("trainer", "gate_handler", "handoff")
        }
        if not (active_states & {"active", "activating", "reloading"}):
            # A successful trainer exit may mean a terminal gate pass.  The
            # handler is receipt-backed and idempotent, so it is the only safe
            # entry point here; never blindly restart the completed trainer.
            systemctl("start", GATE_HANDLER)
            actions.append(f"advanced_idle_pipeline:{GATE_HANDLER}")

        payload = {
            "schema": "poke_bot.production_supervisor/v1",
            "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "host": os.uname().nodename,
            "minimum_free_gib": MIN_FREE_GIB,
            "free_gib": round(free_gib, 2),
            "units": units,
            "health": {
                "elmo_worker": tcp_endpoint_ok("192.168.1.143", 8765),
                "bert_worker": tcp_endpoint_ok("192.168.1.158", 8766),
                "bert_dashboard": endpoint_ok("http://192.168.1.158:8780/"),
            },
            "actions": actions,
        }
        STATE.parent.mkdir(parents=True, exist_ok=True)
        temporary = STATE.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, STATE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
