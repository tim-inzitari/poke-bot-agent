#!/usr/bin/env python3
"""Bind the completed family study to the trainer's original status-75 pause."""

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
    materialize_activation_ready_pause,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", required=True)
    parser.add_argument("--original-pause", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = subprocess.run(
        ["systemctl", "--user", "is-active", args.service],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    state = result.stdout.strip() or "unknown"
    payload = materialize_activation_ready_pause(
        original_pause_path=args.original_pause.resolve(),
        request_path=args.request.resolve(),
        output_path=args.output.resolve(),
        managed_training_active=state in {"active", "activating", "reloading"},
    )
    print(json.dumps({**payload, "managed_service_state": state}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
