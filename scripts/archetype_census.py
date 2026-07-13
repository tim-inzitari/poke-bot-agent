#!/usr/bin/env python
"""Fast parallel archetype census over downloaded ladder episode zips.

Drives the "primary archetype fallback" decision: counts, per day, how many
seats classify as each archetype (pure ``dragapult`` vs ``dragapult-dudunsparce``
vs the rest). Reads directly from the daily zips already under
``data/episodes/raw/`` with a process pool (the single-threaded pre-filter in
``bootstrap_replays.py`` is too slow for a full-day census).

Usage::

    python scripts/archetype_census.py                 # all downloaded days
    python scripts/archetype_census.py --days 2026-07-12 2026-07-11
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import archetypes
from poke_bot.episodes_index import EPISODES_RAW_DIR
from poke_bot.replay_import import classify_episode_seats, _final_winner

_ZIP_CACHE: dict[str, zipfile.ZipFile] = {}


def _classify_member(job: tuple[str, str]) -> tuple[str, str, str, int]:
    """Return (day, arch_seat0, arch_seat1, winner) for one episode member."""
    zip_path, member = job
    zf = _ZIP_CACHE.get(zip_path)
    if zf is None:
        zf = zipfile.ZipFile(zip_path)
        _ZIP_CACHE[zip_path] = zf
    try:
        payload = json.loads(zf.read(member).decode("utf-8"))
    except Exception:
        return ("", archetypes.UNKNOWN, archetypes.UNKNOWN, -1)
    day = Path(zip_path).stem.replace("pokemon-tcg-ai-battle-episodes-", "")
    _, arches = classify_episode_seats(payload)
    a0 = arches[0] if len(arches) > 0 else archetypes.UNKNOWN
    a1 = arches[1] if len(arches) > 1 else archetypes.UNKNOWN
    try:
        winner = _final_winner(payload)
    except Exception:
        winner = -1
    return (day, a0, a1, winner)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", nargs="*", default=None, help="Specific YYYY-MM-DD days (default: all downloaded).")
    p.add_argument("--workers", type=int, default=28)
    p.add_argument("--complete-only", action="store_true", help="Count only decisive games (winner>=0).")
    args = p.parse_args(argv)

    zips = sorted(EPISODES_RAW_DIR.glob("pokemon-tcg-ai-battle-episodes-*.zip"))
    if args.days:
        wanted = set(args.days)
        zips = [z for z in zips if z.stem.replace("pokemon-tcg-ai-battle-episodes-", "") in wanted]
    if not zips:
        print("ERROR: no episode zips found", file=sys.stderr)
        return 2

    jobs: list[tuple[str, str]] = []
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            for nm in zf.namelist():
                if nm.endswith(".json"):
                    jobs.append((str(z), nm))
    print(f"== census over {len(zips)} day(s), {len(jobs)} episodes, workers={args.workers}", flush=True)

    # Per-day, per-archetype SEAT counts (one deck instance per seat).
    per_day_seat: dict[str, Counter] = {}
    per_day_games: Counter = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (day, a0, a1, winner) in enumerate(pool.map(_classify_member, jobs, chunksize=64)):
            if not day:
                continue
            if args.complete_only and winner < 0:
                continue
            per_day_games[day] += 1
            c = per_day_seat.setdefault(day, Counter())
            c[a0] += 1
            c[a1] += 1
            if (i + 1) % 5000 == 0:
                print(f"   ...{i+1}/{len(jobs)}", flush=True)

    focus = ["dragapult", "dragapult-dudunsparce", "dragapult-dusknoir", "dragapult-blaziken", "hammer-pult"]
    total_seat: Counter = Counter()
    print("\n== per-day seat counts (seat = one deck instance) ==", flush=True)
    for day in sorted(per_day_seat):
        c = per_day_seat[day]
        total_seat.update(c)
        row = "  ".join(f"{k}={c.get(k,0)}" for k in focus)
        print(f"  {day}: games={per_day_games[day]}  {row}  unknown={c.get('unknown',0)}", flush=True)

    print("\n== TOTAL across days (seat instances) ==", flush=True)
    for k in focus:
        print(f"  {k:26} {total_seat.get(k,0)}", flush=True)
    print(f"  {'unknown':26} {total_seat.get('unknown',0)}", flush=True)
    print(f"  total games scanned: {sum(per_day_games.values())}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
