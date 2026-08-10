#!/usr/bin/env python3
"""Extract Slowking decision JSONL (+ heuristic features) from daily archives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.slowking_distill.corpus import extract_archive_to_jsonl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-date",
        nargs=2,
        action="append",
        metavar=("ARCHIVE", "DATE"),
        required=True,
        help="Repeatable pair: path/to/day.zip YYYY-MM-DD",
    )
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--max-episodes-per-archive", type=int, default=0)
    parser.add_argument("--summary-out", type=Path, default=None)
    args = parser.parse_args()

    if args.out_jsonl.exists():
        args.out_jsonl.unlink()
    summaries = []
    for archive_s, date in args.archive_date:
        archive = Path(archive_s)
        stats = extract_archive_to_jsonl(
            archive,
            source_date=date,
            out_jsonl=args.out_jsonl,
            max_episodes=int(args.max_episodes_per_archive),
        )
        summaries.append({"archive": str(archive), "source_date": date, **stats})
        print(f"{date}: {stats}", flush=True)
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(
            json.dumps(
                {
                    "schema": "poke_bot.slowking_distill.corpus_build/v1",
                    "research_only": True,
                    "training_authority": False,
                    "out_jsonl": str(args.out_jsonl),
                    "archives": summaries,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
