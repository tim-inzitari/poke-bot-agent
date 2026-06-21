#!/usr/bin/env python3
"""Merge multiple rollout JSONL sources into one deck-agnostic training file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_agent.rewards import filter_complete_episode_rows


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dedupe_key(row: dict) -> tuple[str, str, int, int]:
    return (
        str(row.get("source", "")),
        str(row.get("source_episode_id", row.get("episode", ""))),
        int(row.get("step", 0)),
        int(row.get("player", 0)),
    )


def merge_sources(paths: list[Path], *, require_complete: bool = True) -> list[dict]:
    merged: list[dict] = []
    seen: set[tuple[str, str, int, int]] = set()
    for path in paths:
        if not path.is_file():
            print(f"skip missing {path}")
            continue
        rows = load_jsonl(path)
        for row in rows:
            row.setdefault("source", path.stem)
            key = dedupe_key(row)
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)

    if require_complete:
        before = len({int(row["episode"]) for row in merged})
        merged = filter_complete_episode_rows(merged)
        after = len({int(row["episode"]) for row in merged})
        print(f"complete games filter: {before} -> {after} episodes")

    merged = _reindex_episodes(merged)
    return merged


def _reindex_episodes(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (str(row.get("source", "")), str(row.get("source_episode_id", row.get("episode", ""))))
        groups.setdefault(key, []).append(row)

    ordered: list[dict] = []
    for episode_id, key in enumerate(sorted(groups)):
        episode_rows = sorted(groups[key], key=lambda row: int(row["step"]))
        for row in episode_rows:
            row["episode"] = episode_id
        ordered.extend(episode_rows)
    return ordered


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge rollout JSONL files for deck-agnostic training.")
    parser.add_argument("sources", nargs="+", type=Path, help="input JSONL paths")
    parser.add_argument("--out", type=Path, default="data/training_rollouts_merged.jsonl")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="keep truncated/draw/timeout episodes (default: complete decisive games only)",
    )
    args = parser.parse_args()

    rows = merge_sources(args.sources, require_complete=not args.allow_incomplete)
    write_jsonl(args.out, rows)
    games = len({int(row["episode"]) for row in rows})
    print(f"merged {len(args.sources)} sources -> {games} games / {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
