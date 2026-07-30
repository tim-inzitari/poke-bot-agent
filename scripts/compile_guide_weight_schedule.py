#!/usr/bin/env python3
"""Compile one isolated paired guide study into an immutable weight schedule."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.pure_rl.guide_weight_evidence import (  # noqa: E402
    compile_schedule,
    immutable_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    if not isinstance(evidence, dict):
        raise TypeError("paired guide evidence must be a JSON object")
    schedule = compile_schedule(evidence)
    target = immutable_json(args.output, schedule)
    print(
        json.dumps(
            {
                "ok": True,
                "artifact": str(target),
                "status": schedule["status"],
                "next_weight": schedule["next_state"]["weight"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
