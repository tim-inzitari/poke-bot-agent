"""Add or retire one V6 matchup identity without changing checkpoint shape."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from poke_bot.matchup_adapters_v6 import (
    DEFAULT_REGISTRY_PATH,
    allocate_archetype,
    load_slot_registry,
    retire_archetype,
)


def _atomic_json(path: Path, payload: dict) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add")
    add.add_argument("archetype_id")
    add.add_argument("--status", choices=("active", "dormant"), default="dormant")
    add.add_argument("--lineage", default="operator-registry-allocation")
    retire = commands.add_parser("retire")
    retire.add_argument("archetype_id")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    registry = load_slot_registry(args.registry)
    if args.command == "add":
        revised = allocate_archetype(
            registry,
            args.archetype_id,
            status=args.status,
            lineage=args.lineage,
        )
    else:
        revised = retire_archetype(registry, args.archetype_id)
    if args.dry_run:
        print(json.dumps(revised, indent=2, ensure_ascii=False))
    else:
        _atomic_json(args.registry.resolve(), revised)
        print(f"updated {args.registry.resolve()} to revision {revised['revision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
