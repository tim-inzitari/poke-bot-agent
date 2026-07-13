#!/usr/bin/env python
"""Efficient parallel per-archetype bootstrap collector (expand-day).

Faster replacement for ``bootstrap_replays.py``'s single-threaded pre-filter:

  1. Ensure the latest ``--days`` daily episode zips are downloaded.
  2. **Parallel classify** every episode (28 workers) → which seats match each
     focus archetype (records a full census for the fallback decision).
  3. **Parallel convert** only the matching episodes → per-seat info-set JSONL.

Primary-archetype fallback: pass ``--primary dragapult`` with
``--fallback dragapult-dudunsparce``; if pure ``dragapult`` cannot reach
``--min-games`` across the pulled days, the chosen primary automatically becomes
the fallback. The census counts driving the decision are printed and written to
the ``.meta.json``.

Usage::

    python scripts/collect_archetype.py --primary dragapult \\
        --fallback dragapult-dudunsparce --min-games 5000 --days 40 --workers 28
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm.auto import tqdm

from poke_bot import archetypes, paths
from poke_bot.episodes_index import (
    EPISODES_RAW_DIR,
    download_daily_dataset,
    ensure_episodes_index,
    iter_episode_paths,
    latest_n_days,
    load_daily_manifest,
)
from poke_bot.replay_import import (
    classify_episode_seats,
    convert_episodes_parallel,
    load_episode_payload,
    opponent_archetype_diversity,
    write_jsonl,
)

FOCUS = (
    "dragapult",
    "dragapult-dudunsparce",
    "dragapult-dusknoir",
    "dragapult-blaziken",
    "hammer-pult",
)


def _classify_ref(ref: str) -> tuple[str, str, str]:
    """Return (ref, arch_seat0, arch_seat1) for one episode ref."""
    try:
        payload = load_episode_payload(ref)
        _, arches = classify_episode_seats(payload)
        a0 = arches[0] if len(arches) > 0 else archetypes.UNKNOWN
        a1 = arches[1] if len(arches) > 1 else archetypes.UNKNOWN
        return (ref, a0, a1)
    except Exception:
        return (ref, archetypes.UNKNOWN, archetypes.UNKNOWN)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # FINAL selection rule: primary = highest ladder share among
    # {hammer-pult, dragapult, dragapult-dudunsparce}; hammer eligible only if
    # seats >= hammer_floor; when shares are close (within close_frac of the
    # leader) break toward hammer first, then pure, then dunsparce.
    p.add_argument("--hammer-floor", type=int, default=938,
                   help="Eligibility floor for hammer-pult (filtered seat count).")
    p.add_argument("--close-frac", type=float, default=0.9,
                   help="Counts >= close_frac * leader are 'close' for the tiebreak.")
    p.add_argument("--target", type=int, default=2000,
                   help="Bootstrap game target for the selected primary.")
    p.add_argument("--days", type=int, default=40, help="Latest N calendar days to pull (capped at what exists).")
    p.add_argument("--workers", type=int, default=28)
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--out-dir", type=Path, default=paths.DATA_DIR / "bootstrap")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    paths.ensure_runtime_dirs()
    EPISODES_RAW_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_daily_manifest(ensure_episodes_index())
    days = latest_n_days(manifest, args.days)
    print(f"== collect_archetype FINAL rule: highest share of "
          f"{{hammer-pult, dragapult, dudunsparce}}; hammer_floor={args.hammer_floor}; "
          f"tiebreak hammer>pure>dunsparce; target={args.target}", flush=True)
    print(f">> day window ({len(days)}): {days[0].date} .. {days[-1].date}", flush=True)

    # --- gather refs (download as needed) ---
    all_refs: list[str] = []
    for entry in days:
        if not args.skip_download:
            try:
                download_daily_dataset(entry, root=EPISODES_RAW_DIR, unzip=False)
            except Exception as exc:  # noqa: BLE001
                print(f"  WARN download {entry.slug}: {exc}", flush=True)
                continue
        refs = [ref for _id, ref in iter_episode_paths(entry, root=EPISODES_RAW_DIR)]
        all_refs.extend(refs)
        print(f"  {entry.date}: refs={len(refs)} (cum={len(all_refs)})", flush=True)

    # --- phase 1: parallel classify ---
    print(f">> classify {len(all_refs)} episodes with {args.workers} workers ...", flush=True)
    census: Counter = Counter()
    match_refs: dict[str, list[str]] = {a: [] for a in FOCUS}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for ref, a0, a1 in tqdm(
            pool.map(_classify_ref, all_refs, chunksize=64),
            total=len(all_refs), desc="classify", unit="ep",
        ):
            census[a0] += 1
            census[a1] += 1
            seat_arches = {a0, a1}
            for a in FOCUS:
                if a in seat_arches:
                    match_refs[a].append(ref)

    print("\n== census (seat instances) ==", flush=True)
    for a in FOCUS:
        print(f"  {a:26} seats={census.get(a,0):6}  episodes_with_seat={len(match_refs[a])}", flush=True)
    print(f"  {'unknown':26} seats={census.get('unknown',0)}", flush=True)

    # --- FINAL decision: highest share, hammer floor, tiebreak hammer>pure>dunsparce ---
    hammer = census.get("hammer-pult", 0)
    pult = census.get("dragapult", 0)
    dunsparce = census.get("dragapult-dudunsparce", 0)
    total_seats = max(1, sum(census.values()))
    print("\n== candidate filtered seat counts / ladder share ==", flush=True)
    print(f"  hammer-pult            = {hammer:6}  ({hammer/total_seats:.2%})  floor>={args.hammer_floor}", flush=True)
    print(f"  dragapult (pure)       = {pult:6}  ({pult/total_seats:.2%})", flush=True)
    print(f"  dragapult-dudunsparce  = {dunsparce:6}  ({dunsparce/total_seats:.2%})", flush=True)

    hammer_eligible = hammer >= args.hammer_floor
    candidates = {
        "hammer-pult": hammer if hammer_eligible else -1,
        "dragapult": pult,
        "dragapult-dudunsparce": dunsparce,
    }
    leader_count = max(candidates.values())
    close = {a for a, c in candidates.items() if c >= 0 and c >= args.close_frac * leader_count}
    # Preference order among "close" candidates.
    for pref in ("hammer-pult", "dragapult", "dragapult-dudunsparce"):
        if pref in close:
            chosen = pref
            break
    reason = (
        f"counts: hammer={hammer} (eligible={hammer_eligible}, floor {args.hammer_floor}), "
        f"pure={pult}, dunsparce={dunsparce}; leader_count={leader_count}; "
        f"close set (>= {args.close_frac:.0%} of leader) = {sorted(close)}; "
        f"tiebreak hammer>pure>dunsparce → {chosen}"
    )
    print(f"\n>> DECISION: chosen primary archetype = {chosen}\n   {reason}", flush=True)
    if len(match_refs.get(chosen, [])) < args.target:
        print(f"   NOTE: chosen archetype episodes={len(match_refs.get(chosen, []))} "
              f"< target {args.target} even after {len(days)} days (all readily "
              f"available days pulled)", flush=True)

    # --- phase 2: convert the chosen primary ---
    to_collect = {chosen}

    written: dict[str, dict] = {}
    for arch in to_collect:
        refs = match_refs[arch]
        if not refs:
            print(f">> {arch}: no episodes, skip", flush=True)
            continue
        jobs = [
            {"ref": r, "source": "ladder", "archetype_filter": arch,
             "require_complete": True, "strict_info_set": True}
            for r in refs
        ]
        print(f">> converting {arch}: {len(jobs)} episodes ...", flush=True)
        records = convert_episodes_parallel(jobs, workers=args.workers, desc=f"convert {arch}")
        # dedup by (episode_id, seat)
        seen = set()
        unique = []
        for rec in records:
            key = (rec.get("episode_id"), rec.get("seat"))
            if key in seen:
                continue
            seen.add(key)
            unique.append(rec)
        records = unique
        info_ok = sum(1 for r in records if r.get("info_set_ok"))
        if records and any(not r.get("info_set_ok") for r in records):
            print(f"ERROR: info-set integrity failed for {arch}", file=sys.stderr)
            return 1
        n_dec = sum(int(r.get("n_decisions") or 0) for r in records)
        opp_div = opponent_archetype_diversity(records)
        out_path = args.out_dir / f"{arch}.jsonl"
        n = write_jsonl(records, out_path)
        meta = {
            "archetype": arch,
            "chosen_primary": chosen,
            "decision_reason": reason,
            "n_sequences": n,
            "n_decisions": n_dec,
            "info_set_ok": info_ok,
            "opp_diversity": len(opp_div),
            "day_window": [days[0].date, days[-1].date],
            "n_days": len(days),
            "census_seat_counts": {a: census.get(a, 0) for a in list(FOCUS) + ["unknown"]},
            "out": str(out_path),
        }
        (out_path.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2) + "\n")
        written[arch] = meta
        print(f">> wrote {n} sequences ({n_dec} decisions) → {out_path} "
              f"[info_set_ok={info_ok}/{n}, opp_div={len(opp_div)}]", flush=True)

    print(f"\n>> DONE chosen={chosen}", flush=True)
    for arch, meta in written.items():
        print(f"   {arch}: {meta['n_sequences']} seq, {meta['n_decisions']} decisions", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
