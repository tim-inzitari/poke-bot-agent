#!/usr/bin/env python3
"""Launch the exact-parent revision-134/138 bootstrap from its upload trigger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.archetype_family_activation import (  # noqa: E402
    validate_iteration9_upload_trigger,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--bootstrap-script", type=Path, required=True)
    parser.add_argument("--trigger", type=Path, required=True)
    parser.add_argument("bootstrap_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    state = subprocess.run(
        ["systemctl", "--user", "is-active", args.service],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.strip()
    if state in {"active", "activating", "reloading"}:
        raise RuntimeError(f"managed trainer is still {state}")
    trigger_path = args.trigger.expanduser().resolve()
    trigger = json.loads(trigger_path.read_text(encoding="utf-8"))
    validate_iteration9_upload_trigger(trigger)
    child = Path(
        str(((trigger.get("bindings") or {}).get("checkpoint") or {}).get("path") or "")
    ).resolve()
    if not child.is_file():
        raise RuntimeError("iteration-9 trigger checkpoint is absent")
    tail = list(args.bootstrap_args)
    if tail and tail[0] == "--":
        tail = tail[1:]
    command = [
        str(args.python.expanduser().resolve()),
        "-u",
        str(args.bootstrap_script.expanduser().resolve()),
        "--child",
        str(child),
        "--post-upload-trigger",
        str(trigger_path),
        *tail,
    ]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
