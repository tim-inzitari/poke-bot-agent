#!/usr/bin/env python3
"""Aggregate daily Slowking distill receipts (research-only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.slowking_distill.multi_day import (  # noqa: E402
    aggregate_daily_receipts,
    write_aggregate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily", type=Path, nargs="+", required=True)
    parser.add_argument("--window-start", required=True)
    parser.add_argument("--window-end", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = aggregate_daily_receipts(
        list(args.daily),
        window_start=args.window_start,
        window_end=args.window_end,
    )
    write_aggregate(payload, args.out)
    print(
        f"wrote {args.out} seats={payload['identity']['slowking_seats']} "
        f"lists={payload['identity']['unique_deck_lists']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
