#!/usr/bin/env python3
"""Atomically set the canonical specialist iteration floor and ceiling."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "poke_bot.specialist_runtime_registry/v1"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def update(path: Path, *, floor: int, ceiling: int) -> dict[str, Any]:
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != SCHEMA
        or not isinstance(payload.get("specialists"), dict)
        or floor < 0
        or ceiling < floor
    ):
        raise RuntimeError("specialist runtime registry contract is invalid")
    overrides = {
        specialist_id: {
            key: row[key]
            for key in ("minimum_terminal_iteration", "iteration_ceiling")
            if key in row
        }
        for specialist_id, row in payload["specialists"].items()
        if isinstance(row, dict)
        and any(
            key in row
            for key in ("minimum_terminal_iteration", "iteration_ceiling")
        )
    }
    if overrides:
        raise RuntimeError(
            "per-specialist iteration-window overrides are forbidden: "
            + ",".join(sorted(overrides))
        )
    prior = {
        "minimum_terminal_iteration": payload.get(
            "minimum_terminal_iteration"
        ),
        "iteration_ceiling": payload.get("iteration_ceiling"),
    }
    payload["minimum_terminal_iteration"] = int(floor)
    payload["iteration_ceiling"] = int(ceiling)
    _atomic_json(path, payload)
    return {
        "schema": SCHEMA,
        "path": str(path),
        "prior": prior,
        "current": {
            "minimum_terminal_iteration": int(floor),
            "iteration_ceiling": int(ceiling),
        },
        "specialist_count": len(payload["specialists"]),
        "per_specialist_overrides": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--floor", type=int, required=True)
    parser.add_argument("--ceiling", type=int, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            update(
                args.registry,
                floor=int(args.floor),
                ceiling=int(args.ceiling),
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
