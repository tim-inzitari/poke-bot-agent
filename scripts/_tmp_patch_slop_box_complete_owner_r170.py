#!/usr/bin/env python3
"""Keep completed Slop Box final-refresh as dashboard service/runtime owner."""

from __future__ import annotations

import datetime
import shutil
import sys
from pathlib import Path


TARGETS = [
    Path("/home/inzi/poke-bot-agent/scripts/dashboard_snapshot.py"),
    Path(
        "/home/inzi/poke-bot-agent-deployments/final-format-marnie-postupload-r136/"
        "scripts/dashboard_snapshot.py"
    ),
    Path(
        "/home/inzi/poke-bot-agent-deployments/state-core-v1/scripts/"
        "dashboard_snapshot.py"
    ),
    Path(
        "/home/inzi/poke-bot-agent-deployments/"
        "pure-rl-resident-v41-specialist-matchup-runtime/scripts/"
        "dashboard_snapshot.py"
    ),
]


def patch_text(text: str) -> tuple[str, list[str]]:
    changed: list[str] = []
    replacements = [
        (
            """            if active_final_refresh.get("status") in {"running", "stopped"}:
                service = final_service
""",
            """            if active_final_refresh.get("status") in {
                "running",
                "stopped",
                "complete",
            }:
                service = final_service
""",
            "service_owner_complete",
        ),
        (
            """    if (
        active_final_refresh.get("status") in {"running", "stopped"}
        and str(active_final_refresh.get("specialist_id") or "").strip()
    ):
""",
            """    if (
        active_final_refresh.get("status")
        in {"running", "stopped", "complete"}
        and str(active_final_refresh.get("specialist_id") or "").strip()
    ):
""",
            "runtime_identity_complete",
        ),
        (
            """    protocol_service = (
        active_final_refresh.get("service") or service
        if active_final_refresh.get("status") in {"running", "stopped"}
        else service
    )
""",
            """    protocol_service = (
        active_final_refresh.get("service") or service
        if active_final_refresh.get("status")
        in {"running", "stopped", "complete"}
        else service
    )
""",
            "protocol_service_complete",
        ),
    ]
    for old, new, name in replacements:
        if new in text:
            continue
        if old not in text:
            raise SystemExit(f"missing anchor: {name}")
        text = text.replace(old, new, 1)
        changed.append(name)
    return text, changed


def main() -> int:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for path in TARGETS:
        if not path.is_file():
            print(f"skip missing {path}")
            continue
        text = path.read_text(encoding="utf-8")
        new_text, changed = patch_text(text)
        if not changed:
            print(f"no change {path}")
            continue
        bak = path.with_name(path.name + f".pre-complete-owner-{stamp}")
        shutil.copy2(path, bak)
        path.write_text(new_text, encoding="utf-8")
        print(f"patched {path} changed={changed} bak={bak.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
