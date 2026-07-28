#!/usr/bin/env python3
"""Publish one concise progress stream for the four-device adapter fit."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import time
from pathlib import Path
from typing import Any


INZI_ROOT = Path("/home/inzi/poke-bot-agent/outputs/matchup_adapters/alakazam-iter26-fleet-v31")
BERT_ROOT = Path("/Users/tsinzitari/workspace/poke-adapter-fleet-v31/output")
REMOTE_LOG = INZI_ROOT / "fleet-status.log"
REMOTE_JSON = INZI_ROOT / "fleet-progress.json"
WORKERS = {
    "blackwell": {"host": "Inzi", "device": "RTX PRO 5000 Blackwell", "routes": [3, 4, 7]},
    "rtx3080": {"host": "Inzi", "device": "RTX 3080 Ti", "routes": [1, 5, 8]},
    "elmo3060": {"host": "Elmo", "device": "RTX 3060", "routes": [0, 2]},
    "bert_mps": {"host": "Bert", "device": "Apple M4 MPS", "routes": [6, 9, 10, 11, 12, 13, 14]},
}


def _command(*args: str) -> str:
    result = subprocess.run(args, text=True, capture_output=True, timeout=8, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def _parse(raw: str) -> dict[str, Any] | None:
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _local(path: Path) -> dict[str, Any] | None:
    try:
        return _parse(path.read_text())
    except OSError:
        return None


def _inzi(name: str) -> dict[str, Any] | None:
    return _parse(
        _command(
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=4",
            "inzi@192.168.1.151",
            "cat",
            str(INZI_ROOT / f"{name}.pt.progress.json"),
        )
    )


def _elmo() -> dict[str, Any] | None:
    return _parse(
        _command(
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=4",
            "inzi@192.168.1.151",
            "ssh elmo cat /mnt/Main/main/poke-adapter-fleet-v31/output/elmo.pt.progress.json",
        )
    )


def _label(name: str, row: dict[str, Any] | None) -> str:
    if row is None:
        return f"{name}=epoch2/25(starting)"
    epoch = int(row.get("epoch", 2))
    target = int(row.get("target_epochs", 25))
    elapsed = float(row.get("elapsed_seconds", 0.0))
    completed = max(0, epoch - 2)
    remaining = max(0, target - epoch)
    eta = (elapsed / completed * remaining) if completed and elapsed > 0 else 0.0
    suffix = "done" if row.get("complete") else f"eta={eta/3600:.2f}h"
    return f"{name}=epoch{epoch}/{target}({suffix})"


def main() -> int:
    last = ""
    last_publish = 0.0
    bert_artifact_pushed = False
    while True:
        rows = {
            "blackwell": _inzi("blackwell"),
            "rtx3080": _inzi("rtx3080"),
            "elmo3060": _elmo(),
            "bert_mps": _local(BERT_ROOT / "bert.pt.progress.json"),
        }
        status = " ".join(_label(name, row) for name, row in rows.items())
        now = time.time()
        if status != last or now - last_publish >= 30.0:
            stamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")
            line = f"[{stamp}] {status}\n"
            if status != last:
                print(line, end="", flush=True)
            subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=4",
                    "inzi@192.168.1.151",
                    f"cat >> {REMOTE_LOG}",
                ],
                input=line,
                text=True,
                timeout=8,
                check=False,
            )
            payload = {
                "schema": "poke_bot.matchup_adapter_fleet_progress/v1",
                "observed_at": now,
                "source_epoch": 2,
                "target_epochs": 25,
                "workers": [
                    {
                        "id": name,
                        **WORKERS[name],
                        "progress": row,
                        "state": (
                            "complete"
                            if row and row.get("complete") is True
                            else "running"
                        ),
                    }
                    for name, row in rows.items()
                ],
            }
            subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=4",
                    "inzi@192.168.1.151",
                    f"cat > {REMOTE_JSON}.tmp && mv {REMOTE_JSON}.tmp {REMOTE_JSON}",
                ],
                input=json.dumps(payload, sort_keys=True) + "\n",
                text=True,
                timeout=8,
                check=False,
            )
            last = status
            last_publish = now
        bert_row = rows.get("bert_mps")
        if (
            not bert_artifact_pushed
            and bert_row
            and bert_row.get("complete") is True
            and (BERT_ROOT / "bert.pt").is_file()
        ):
            pushed = subprocess.run(
                [
                    "rsync",
                    "-a",
                    str(BERT_ROOT / "bert.pt"),
                    str(BERT_ROOT / "bert.pt.progress.json"),
                    f"inzi@192.168.1.151:{INZI_ROOT}/",
                ],
                timeout=120,
                check=False,
            )
            bert_artifact_pushed = pushed.returncode == 0
        time.sleep(15)


if __name__ == "__main__":
    raise SystemExit(main())
