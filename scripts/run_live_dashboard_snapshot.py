#!/usr/bin/env python3
"""Run dashboard telemetry from the selector-owned specialist runtime."""

from __future__ import annotations

import os
from pathlib import Path
import sys


DEFAULT_SELECTOR = Path("/home/inzi/.config/pokebot/specialist_runtime.env")
RUNTIME_KEY = "POKEBOT_SPECIALIST_RUNTIME_ROOT"


def resolve_runtime_snapshot(selector: Path) -> Path:
    rows = [
        line.split("=", 1)[1].strip()
        for line in selector.read_text(encoding="utf-8").splitlines()
        if line.startswith(RUNTIME_KEY + "=")
    ]
    if len(rows) != 1 or not rows[0]:
        raise RuntimeError(
            "canonical specialist runtime root is absent or duplicated"
        )
    runtime_root = Path(rows[0]).expanduser()
    if not runtime_root.is_absolute():
        raise RuntimeError("canonical specialist runtime root is not absolute")
    snapshot = runtime_root / "scripts" / "dashboard_snapshot.py"
    if not snapshot.is_file():
        raise RuntimeError(
            f"selector-owned dashboard snapshot is missing: {snapshot}"
        )
    return snapshot


def main() -> int:
    selector = Path(
        os.environ.get("POKEBOT_SPECIALIST_SELECTOR", str(DEFAULT_SELECTOR))
    )
    snapshot = resolve_runtime_snapshot(selector)
    os.execv(sys.executable, [sys.executable, str(snapshot)])
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
