#!/usr/bin/env python
"""Build a bounded portable feature shard from ladder bootstrap JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import config
from poke_bot.feature_shards import write_feature_shard


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-date", action="append", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-in-flight", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-context", type=int, default=config.MODEL.max_context)
    args = parser.parse_args()
    metadata = write_feature_shard(
        args.jsonl,
        args.out,
        source_dates=list(args.source_date),
        max_context=args.max_context,
        workers=args.workers,
        max_in_flight=args.max_in_flight,
        max_records=args.max_records,
        verify_info_set=True,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
