#!/usr/bin/env python3
"""Wait for four route workers, merge them, and publish canonical final.pt."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

DEPLOY = Path(__file__).resolve().parents[1]
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

from poke_bot import checkpoint


ROOT = Path("/home/inzi/poke-bot-agent")
SOURCE_DIR = ROOT / "outputs/matchup_adapters/alakazam-iter26-all22-v31"
FLEET_DIR = ROOT / "outputs/matchup_adapters/alakazam-iter26-fleet-v31"
SOURCE = SOURCE_DIR / "best.pt"
STATUS = FLEET_DIR / "fleet-finalizer.json"
WORKERS = ("blackwell", "rtx3080", "elmo", "bert")


def _atomic_json(payload: dict) -> None:
    temporary = STATUS.with_name(f".{STATUS.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, STATUS)


def _progress(name: str) -> dict:
    path = FLEET_DIR / f"{name}.pt.progress.json"
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _pull_elmo() -> None:
    remote = "/mnt/Main/main/poke-adapter-fleet-v31/output"
    subprocess.run(
        [
            "rsync",
            "-a",
            "elmo:" + remote + "/elmo.pt",
            "elmo:" + remote + "/elmo.pt.progress.json",
            str(FLEET_DIR) + "/",
        ],
        timeout=120,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def _service_active(name: str) -> bool:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", name],
        check=False,
    )
    return result.returncode == 0


def main() -> int:
    FLEET_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        _pull_elmo()
        progress = {name: _progress(name) for name in WORKERS}
        complete = {
            name: bool(row.get("complete") is True)
            for name, row in progress.items()
        }
        _atomic_json(
            {
                "schema": "poke_bot.matchup_adapter_fleet_finalizer/v1",
                "phase": "ready_to_merge" if all(complete.values()) else "waiting_for_workers",
                "workers_complete": complete,
                "updated_at": time.time(),
            }
        )
        if all(complete.values()):
            break
        time.sleep(30)

    command = [
        sys.executable,
        str(DEPLOY / "scripts/merge_matchup_adapter_route_workers.py"),
        "--checkpoint",
        str(SOURCE),
    ]
    for name in WORKERS:
        command.extend(["--worker", str(FLEET_DIR / f"{name}.pt")])
    command.extend(["--output-dir", str(FLEET_DIR)])
    _atomic_json(
        {
            "schema": "poke_bot.matchup_adapter_fleet_finalizer/v1",
            "phase": "merging",
            "workers_complete": {name: True for name in WORKERS},
            "updated_at": time.time(),
        }
    )
    subprocess.run(command, cwd=DEPLOY, check=True)
    final = torch.load(FLEET_DIR / "final.pt", map_location="cpu", weights_only=False)
    state = dict(dict(final.get("extra") or {}).get("streaming_matchup_adapter_state") or {})
    fleet_execution = dict(
        dict(final.get("extra") or {}).get("matchup_adapter_fleet_execution") or {}
    )
    expected_source_digest = str(
        fleet_execution.get("source_checkpoint_digest") or ""
    )
    if not (
        int(final.get("epoch", -1)) == 25
        and int(state.get("epoch", -1)) == 25
        and state.get("complete") is True
        and dict(final.get("extra") or {}).get("matchup_adapter_fit_complete") is True
        and expected_source_digest
        and checkpoint.checkpoint_digest(SOURCE) == expected_source_digest
    ):
        raise RuntimeError("fleet merge did not produce the exact epoch-25 contract")

    if _service_active("pokebot-matchup-adapter-v31-recovery.service"):
        raise RuntimeError(
            "refusing fleet publication while the canonical recovery fit is active"
        )

    backup_suffix = expected_source_digest.removeprefix("sha256:")[:16]
    for name in ("latest.pt", "best.pt", "final.pt", "progress.json"):
        destination = SOURCE_DIR / name
        if destination.is_file():
            backup = SOURCE_DIR / f"{name}.before-fleet-{backup_suffix}"
            if not backup.exists():
                _atomic_copy(destination, backup)
        _atomic_copy(FLEET_DIR / name, SOURCE_DIR / name)
    _atomic_json(
        {
            "schema": "poke_bot.matchup_adapter_fleet_finalizer/v1",
            "phase": "published",
            "workers_complete": {name: True for name in WORKERS},
            "final_checkpoint": str(SOURCE_DIR / "final.pt"),
            "epoch": 25,
            "updated_at": time.time(),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
