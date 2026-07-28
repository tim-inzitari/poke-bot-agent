#!/usr/bin/env python3
"""Emit compact Elmo telemetry for the additive rare-route preparation."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time
from typing import Any


DAYS = (
    "2026-06-26",
    "2026-06-27",
    "2026-06-28",
    "2026-06-29",
    "2026-06-30",
    "2026-07-01",
    "2026-07-22",
    "2026-07-23",
)
ARCHIVE_ROOT = Path("/mnt/Main/main/poke-bot-agent/archive/episode-days")
EXPERT_ROOT = Path(
    "/mnt/Main/main/poke-bot-agent/archive/"
    "rare-route-expert-history-20260626-20260701"
)
ROUTER_ROOT = Path(
    "/mnt/Main/main/poke-adapter-oracle-v29/output/"
    "public-matchup-tree-calibration-v35"
)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _service(name: str) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "systemctl",
            "show",
            name,
            "-p",
            "LoadState",
            "-p",
            "ActiveState",
            "-p",
            "SubState",
            "-p",
            "MainPID",
            "-p",
            "NRestarts",
            "-p",
            "Result",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    result: dict[str, Any] = {"name": name}
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    result["active"] = result.get("ActiveState") in {"active", "activating"}
    return result


def _download_container() -> dict[str, Any]:
    completed = subprocess.run(
        [
            "docker",
            "inspect",
            "pokebot-expert-episode-download",
            "--format",
            "{{json .State}}",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    try:
        state = json.loads(completed.stdout)
    except json.JSONDecodeError:
        state = {}
    archives = []
    for day in DAYS:
        path = ARCHIVE_ROOT / f"pokemon-tcg-ai-battle-episodes-{day}.zip"
        archives.append(
            {
                "day": day,
                "ready": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else 0,
            }
        )
    return {
        "name": "pokebot-expert-episode-download",
        "running": state.get("Running") is True,
        "status": state.get("Status"),
        "exit_code": state.get("ExitCode"),
        "ready_days": sum(1 for row in archives if row["ready"]),
        "total_days": len(DAYS),
        "days": archives,
    }


def main() -> int:
    expert = _json(EXPERT_ROOT / "status.json")
    router_audit = _json(ROUTER_ROOT / "public-matchup-tree.audit.json")
    split = _json(
        EXPERT_ROOT / "specialists-v1/SPECIALIST_CORPORA_READY.json"
    )
    payload = {
        "schema": "poke_bot.rare_route_pipeline_dashboard/v1",
        "observed_at": time.time(),
        "host": "Elmo",
        "production_blocking": False,
        "activation_policy": "specialist_boundary_only",
        "download": _download_container(),
        "router": {
            "service": _service(
                "pokebot-public-tree-v35-after-sim.service"
            ),
            "audit_ready": bool(router_audit),
            "accepted_count": int(router_audit.get("accepted_count") or 0),
            "accepted_specialist_ids": list(
                router_audit.get("accepted_specialist_ids") or ()
            ),
        },
        "expert": {
            "service": _service(
                "pokebot-rare-route-expert-shards-v1.service"
            ),
            "phase": expert.get("phase") or "waiting",
            "current": int(expert.get("current") or 0),
            "total": int(expert.get("total") or len(DAYS)),
            "day": expert.get("day"),
            "completed": list(expert.get("completed") or ()),
            "split_ready": bool(split),
        },
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
