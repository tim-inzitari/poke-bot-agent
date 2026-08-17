#!/usr/bin/env python
"""Per-archetype ladder bootstrap: episodes index → filter → JSONL.

Phase 4 / 2B entry point. Defaults target pure Dragapult:

  1. Ensure episodes index is present (download if missing).
  2. Take the latest ``--days`` calendar days (default 3).
  3. Download daily episode zips under ``data/episodes/raw/`` as needed.
  4. Filter episodes whose **card signature** matches ``--archetype``
     (pure ``dragapult`` = Dragapult line via ``classify_deck`` **without**
     Hammer-Pult signature). Hammer-Pult remains a separate bucket for later.
     Other Dragapult variants are separate buckets and never steal Hammer games.
  5. Convert matching seats to training JSONL with ``--workers`` (default 28).
  6. If filtered game count < ``--min-games`` (~5000) OR opponent diversity
     < ``--min-opp-archetypes`` (~5), pull another prior day and repeat.

Smoke / capped runs::

    python scripts/bootstrap_replays.py --archetype hammer-pult --max-games 20 \\
        --min-games 0 --days 1 --local-dir data/episodes/smoke

Full scale (multi-hour download)::

    python scripts/bootstrap_replays.py --archetype hammer-pult --min-games 5000 \\
        --min-opp-archetypes 5 --days 3 --workers 28
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import archetypes, config, paths
from poke_bot.episodes_index import (
    EPISODES_RAW_DIR,
    download_daily_dataset,
    ensure_episodes_index,
    expand_prior_days,
    iter_episode_paths,
    latest_n_days,
    load_daily_manifest,
)
from poke_bot.replay_import import (
    BOOTSTRAP_ARCHETYPES,
    convert_episodes_parallel,
    filter_episode_quick,
    load_episode_payload,
    opponent_archetype_diversity,
    write_jsonl,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--archetype",
        default="dragapult",
        choices=list(BOOTSTRAP_ARCHETYPES),
        help="Archetype bucket to filter (default: pure dragapult; hammer-pult is signature-based).",
    )
    p.add_argument("--days", type=int, default=3, help="Initial calendar-day window (latest N).")
    p.add_argument(
        "--min-games",
        type=int,
        default=5000,
        help="Expand prior days until this many matching games (0 disables).",
    )
    p.add_argument(
        "--min-opp-archetypes",
        type=int,
        default=5,
        help="Minimum unique opposing archetypes before stopping expansion.",
    )
    p.add_argument(
        "--max-games",
        type=int,
        default=0,
        help="Cap matching games converted (0 = no cap). Use for smoke runs.",
    )
    p.add_argument(
        "--max-scan",
        type=int,
        default=0,
        help="Max episode files to scan per day (0 = all). Useful for smoke.",
    )
    p.add_argument("--workers", type=int, default=config.HARDWARE.sim_workers)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSONL path (default data/bootstrap/<archetype>.jsonl).",
    )
    p.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="If set, scan this directory of episode JSON instead of Kaggle days.",
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not call kaggle; only use already-local zips/dirs.",
    )
    p.add_argument(
        "--dry-filter",
        action="store_true",
        help="Only count matches / diversity; do not write JSONL.",
    )
    return p.parse_args(argv)


def _scan_local_dir(
    directory: Path,
    archetype: str,
    *,
    max_scan: int = 0,
    max_games: int = 0,
) -> list[dict]:
    """Return convert jobs for matching episodes under ``directory``."""
    from tqdm.auto import tqdm

    files = sorted(directory.glob("*.json"))
    if max_scan > 0:
        files = files[:max_scan]
    jobs: list[dict] = []
    bar = tqdm(files, desc=f"scan {directory.name}", unit="file")
    for path in bar:
        try:
            payload = load_episode_payload(path)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if not filter_episode_quick(payload, archetype):
            continue
        jobs.append(
            {
                "ref": str(path),
                "source": f"local:{directory.name}",
                "archetype_filter": archetype,
                "require_complete": True,
                "strict_info_set": True,
            }
        )
        bar.set_postfix(kept=len(jobs))
        if max_games > 0 and len(jobs) >= max_games:
            break
    return jobs


def _scan_day(
    entry,
    archetype: str,
    *,
    root: Path,
    max_scan: int = 0,
    max_games: int = 0,
    skip_download: bool = False,
) -> list[dict]:
    from tqdm.auto import tqdm

    if not skip_download:
        print(f"  >> downloading day {entry.date} ({entry.slug}) ...", flush=True)
        try:
            download_daily_dataset(entry, root=root, unzip=False)
            print(f"  >> download done {entry.date}", flush=True)
        except Exception as exc:  # noqa: BLE001 — surface and continue other days
            print(f"  WARN: download failed for {entry.slug}: {exc}", flush=True)
            return []
    else:
        print(f"  >> skip-download; scanning local for {entry.date}", flush=True)

    jobs: list[dict] = []
    scanned = 0
    # Materialize paths so tqdm knows the total when possible.
    paths_iter = list(iter_episode_paths(entry, root=root, max_files=max_scan))
    bar = tqdm(paths_iter, desc=f"filter {entry.date}", unit="ep")
    for ep_id, ref in bar:
        scanned += 1
        try:
            payload = load_episode_payload(ref)
        except (OSError, json.JSONDecodeError, ValueError, KeyError):
            continue
        if not filter_episode_quick(payload, archetype):
            continue
        jobs.append(
            {
                "ref": ref if isinstance(ref, str) else str(ref),
                "source": entry.slug,
                "archetype_filter": archetype,
                "require_complete": True,
                "strict_info_set": True,
            }
        )
        bar.set_postfix(hits=len(jobs), arch=archetype)
        if max_games > 0 and len(jobs) >= max_games:
            break
    print(f"  {entry.date} scanned={scanned} matched={len(jobs)}", flush=True)
    return jobs


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    paths.ensure_runtime_dirs()
    EPISODES_RAW_DIR.mkdir(parents=True, exist_ok=True)

    out_path = args.out or (paths.DATA_DIR / "bootstrap" / f"{args.archetype}.jsonl")
    print(f"== bootstrap_replays archetype={args.archetype}", flush=True)
    print(f"   out={out_path}", flush=True)
    print(f"   min_games={args.min_games} min_opp_archetypes={args.min_opp_archetypes} "
          f"max_games={args.max_games} workers={args.workers}", flush=True)

    all_jobs: list[dict] = []

    if args.local_dir is not None:
        local = Path(args.local_dir)
        if not local.is_dir():
            print(f"ERROR: --local-dir not a directory: {local}", file=sys.stderr)
            return 2
        print(f">> scanning local dir {local}", flush=True)
        all_jobs = _scan_local_dir(
            local,
            args.archetype,
            max_scan=args.max_scan,
            max_games=args.max_games if args.max_games > 0 else 0,
        )
    else:
        manifest_path = ensure_episodes_index()
        manifest = load_daily_manifest(manifest_path)
        if not manifest:
            print("ERROR: empty episodes index manifest", file=sys.stderr)
            return 2
        days = latest_n_days(manifest, args.days)
        print(f">> initial window: {[d.date for d in days]}", flush=True)

        seen_dates = set()
        while True:
            for entry in days:
                if entry.date in seen_dates:
                    continue
                seen_dates.add(entry.date)
                remaining = 0
                if args.max_games > 0:
                    remaining = max(0, args.max_games - len(all_jobs))
                    if remaining == 0:
                        break
                day_jobs = _scan_day(
                    entry,
                    args.archetype,
                    root=EPISODES_RAW_DIR,
                    max_scan=args.max_scan,
                    max_games=remaining if args.max_games > 0 else 0,
                    skip_download=args.skip_download,
                )
                all_jobs.extend(day_jobs)
                if args.max_games > 0 and len(all_jobs) >= args.max_games:
                    all_jobs = all_jobs[: args.max_games]
                    break

            # Convert a probe set to measure diversity / count when expanding.
            need_more = False
            if args.min_games > 0 or args.min_opp_archetypes > 0:
                # Cheap count = matched jobs; diversity needs a convert pass on a sample.
                n_matched = len(all_jobs)
                if args.min_games > 0 and n_matched < args.min_games:
                    need_more = True
                if args.min_opp_archetypes > 0 and not need_more:
                    # Probe convert up to 200 jobs for diversity estimate.
                    probe = all_jobs[: min(len(all_jobs), 200)]
                    if probe:
                        probe_recs = convert_episodes_parallel(
                            probe, workers=min(args.workers, 8)
                        )
                        div = opponent_archetype_diversity(probe_recs)
                        print(f"   probe games={len(probe_recs)} opp_archetypes={sorted(div)}", flush=True)
                        if len(div) < args.min_opp_archetypes:
                            need_more = True
                    elif args.min_opp_archetypes > 0:
                        need_more = True

            if not need_more or args.max_games > 0:
                # Cap mode: do not expand days.
                break
            if args.min_games <= 0 and args.min_opp_archetypes <= 0:
                break

            expanded = expand_prior_days(manifest, days, extra=1)
            if len(expanded) == len(days):
                print(">> no more prior days available; stopping expansion", flush=True)
                break
            new_day = [d for d in expanded if d.date not in seen_dates]
            print(
                f">> expanding with prior day {[d.date for d in new_day]} "
                f"(matched so far={len(all_jobs)}; days_expanded={len(seen_dates)})",
                flush=True,
            )
            days = expanded

    print(f">> matched episode jobs: {len(all_jobs)}", flush=True)
    if args.dry_filter:
        print(">> --dry-filter set; skipping convert/write", flush=True)
        return 0
    if not all_jobs:
        print("ERROR: no matching episodes found", file=sys.stderr)
        return 1

    print(f">> converting with {args.workers} workers ...", flush=True)
    records = convert_episodes_parallel(
        all_jobs, workers=args.workers, desc=f"convert {args.archetype}"
    )
    # Deduplicate by (episode_id, seat)
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
    remasked = sum(1 for r in records if r.get("info_set_remasked"))
    opp_div = opponent_archetype_diversity(records)
    named_opp = sorted(
        {str(r.get("opp_archetype") or "unknown") for r in records}
    )
    arch_counts = Counter(r.get("archetype") for r in records)
    n_decisions = sum(int(r.get("n_decisions") or 0) for r in records)

    print(f">> records={len(records)} info_set_ok={info_ok}/{len(records)} "
          f"remasked={remasked} decisions={n_decisions}", flush=True)
    print(f"   archetypes={dict(arch_counts)}", flush=True)
    print(
        f"   opp_diversity={len(opp_div)} named_opp_archetypes={named_opp}",
        flush=True,
    )

    if any(not r.get("info_set_ok") for r in records):
        print("ERROR: info-set integrity failed for some records", file=sys.stderr)
        return 1

    n = write_jsonl(records, out_path)
    print(f">> wrote {n} sequences → {out_path}", flush=True)

    meta = {
        "archetype": args.archetype,
        "n_sequences": n,
        "n_decisions": n_decisions,
        "opp_diversity": len(opp_div),
        "named_opp_archetypes": named_opp,
        "info_set_ok": info_ok,
        "info_set_remasked": remasked,
        "out": str(out_path),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f">> meta → {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
