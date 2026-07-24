#!/usr/bin/env python3
"""Build one immutable Alakazam temporal expert shard from a raw daily zip."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import config, paths
from poke_bot.authoritative_visual_trace import materialize_day
from poke_bot.ladder_replay import LadderReplayClassifier


DEFAULT_MIX = ROOT / "data" / "training_mixes" / "top_ladder.v1.json"
DEFAULT_REPRESENTATIVES = (
    ROOT / "data" / "training_mixes" / "top_ladder_representatives.v1.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="YYYY-MM-DD source date")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mix", type=Path, default=DEFAULT_MIX)
    parser.add_argument(
        "--representatives", type=Path, default=DEFAULT_REPRESENTATIVES
    )
    parser.add_argument(
        "--card-csv",
        type=Path,
        default=None,
        help="Competition EN_Card_Data.csv (defaults to the checkout resolver).",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-in-flight", type=int, default=12)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--max-context", type=int, default=config.MODEL.max_context)
    parser.add_argument("--memory-floor-gib", type=float, default=8.0)
    parser.add_argument("--min-records", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    classifier = LadderReplayClassifier.from_paths(
        args.mix,
        args.representatives,
        card_csv=args.card_csv or paths.en_card_data_path(),
    )
    receipt = materialize_day(
        args.archive,
        args.out,
        classifier=classifier,
        source_date=args.date,
        workers=args.workers,
        max_in_flight=args.max_in_flight,
        max_episodes=args.max_episodes,
        max_context=args.max_context,
        resume=not args.no_resume,
        min_available_bytes=int(args.memory_floor_gib * 1024**3),
        min_records=args.min_records,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
