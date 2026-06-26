#!/usr/bin/env python3
"""Copy deck lists whose filename matches an archetype pattern into a focused directory.

Used to build single-archetype field/our-side pools for specialization runs, e.g.:

    python scripts/filter_deck_dir.py --pattern dragapult \
        --source decks/competitive/high_performing --dest decks/dragapult-only

Matching is a case-insensitive substring test against each deck filename, so
``--pattern dragapult`` captures dragapult, dragapult-dusknoir, dragapult-blaziken, etc.
Pass ``--pattern`` multiple times to union several archetypes.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_agent.deck_pool import read_deck_file

DECK_SUFFIXES = {".csv", ".txt", ".deck"}


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def filter_deck_dir(
    *,
    patterns: list[str],
    source: Path,
    dest: Path,
    symlink: bool,
) -> list[Path]:
    source = _resolve(source)
    dest = _resolve(dest)
    if not source.is_dir():
        raise FileNotFoundError(f"source directory not found: {source}")

    lowered = [p.lower() for p in patterns]
    matches = sorted(
        path
        for path in source.iterdir()
        if path.is_file()
        and path.suffix.lower() in DECK_SUFFIXES
        and any(pattern in path.stem.lower() for pattern in lowered)
    )
    if not matches:
        raise SystemExit(f"no deck files in {source} matched patterns {patterns}")

    dest.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for src_file in matches:
        try:
            read_deck_file(src_file)
        except ValueError as exc:
            print(f"skipping invalid deck {src_file.name}: {exc}")
            continue
        target = dest / src_file.name
        if target.exists() or target.is_symlink():
            target.unlink()
        if symlink:
            target.symlink_to(src_file.resolve())
        else:
            shutil.copy2(src_file, target)
        copied.append(target)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a single-archetype deck directory.")
    parser.add_argument(
        "--pattern",
        action="append",
        required=True,
        help="case-insensitive filename substring (repeatable), e.g. dragapult",
    )
    parser.add_argument("--source", type=Path, required=True, help="source deck directory")
    parser.add_argument("--dest", type=Path, required=True, help="destination directory")
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="symlink instead of copying (saves disk; breaks if source moves)",
    )
    args = parser.parse_args()

    copied = filter_deck_dir(
        patterns=args.pattern,
        source=args.source,
        dest=args.dest,
        symlink=args.symlink,
    )
    print(f"{len(copied)} decks -> {_resolve(args.dest)} (patterns={args.pattern})")


if __name__ == "__main__":
    main()
