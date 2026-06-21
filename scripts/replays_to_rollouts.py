#!/usr/bin/env python3
"""Convert scraped or daily-dataset replay JSON files into training JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_agent.archetypes import load_archetype_registry
from poke_agent.episodes_index import (
    EpisodeRecord,
    default_index_path,
    export_episode_ids,
    filter_top_percent,
    load_episode_pool,
    load_replay_score,
)
from poke_agent.replay_import import convert_replay_file


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def records_from_replays_dir(directory: Path) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    for replay_path in sorted(directory.glob("*.json")):
        try:
            episode_id, score = load_replay_score(replay_path)
        except (json.JSONDecodeError, OSError, ValueError):
            continue
        records.append(
            EpisodeRecord(
                episode_id=episode_id,
                replay_path=replay_path,
                score=score,
                source=str(directory),
            )
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert replay JSON files to CABT rollout JSONL.")
    parser.add_argument("--replays-dir", type=Path, default=None, help="directory of replay JSON files")
    parser.add_argument("--index-csv", type=Path, default=None, help="scrape index.csv with replay_file column")
    parser.add_argument("--episodes-index", type=Path, default=None, help="episodes index manifest.csv")
    parser.add_argument("--top-percent", type=float, default=0.0, help="keep top N%% episodes by score")
    parser.add_argument("--max-episodes", type=int, default=0, help="cap episodes converted (0 = all)")
    parser.add_argument("--out", type=Path, default="data/scraped_rollouts.jsonl")
    parser.add_argument("--episode-ids-out", type=Path, default=None, help="optional export of selected episode ids")
    args = parser.parse_args()

    registry = load_archetype_registry(ROOT)
    if args.replays_dir is not None:
        records = records_from_replays_dir(args.replays_dir)
    else:
        records = load_episode_pool(
            ROOT,
            index_path=args.episodes_index or default_index_path(ROOT),
            scrape_index_csv=args.index_csv,
        )

    records = [record for record in records if record.replay_path is not None and record.replay_path.is_file()]
    if args.top_percent > 0:
        records = filter_top_percent(records, args.top_percent)
    if args.max_episodes > 0:
        records = records[: args.max_episodes]

    if args.episode_ids_out is not None:
        export_episode_ids(records, args.episode_ids_out)

    all_rows: list[dict] = []
    for episode_index, record in enumerate(records):
        assert record.replay_path is not None
        rows = convert_replay_file(
            record.replay_path,
            episode=episode_index,
            registry=registry,
            root=ROOT,
            source=record.source or "replay",
        )
        all_rows.extend(rows)

    write_jsonl(args.out, all_rows)
    print(f"converted {len(records)} episodes -> {len(all_rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
