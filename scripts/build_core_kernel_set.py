#!/usr/bin/env python
"""Build a **stratified, hard-capped** deck-agnostic core-kernel training set.

The deck-agnostic core kernel (``poke_bot.core_kernel``) must warm up on a
*diverse* corpus without letting disk fill with unbounded RL dumps from
collapsed-win-rate models. This script curates a single training JSONL that:

  * pulls records from the **curated bootstrap archetype buckets** (info-set
    verified ladder data) — NOT the growing on-policy ``data/rl/*`` dumps;
  * is **stratified** across archetypes (round-robin draw so no single deck
    dominates), preferring diversity over raw volume;
  * enforces a **HARD CAP** of ``--max-games`` (default 5000) records by
    construction — the output file physically contains ``<= max_games`` game
    trajectories, so nothing downstream can blow past the cap;
  * de-duplicates by ``(episode_id, seat)`` so a game never appears twice;
  * is **re-runnable** to rotate/refresh: rerun to fold in freshly-built
    buckets and drop stale ones, keeping the set small and current.

Each "game" == one seat's whole-game trajectory record (the unit the streaming
corpus and ``record_to_sequence`` consume). Output goes to
``data/bootstrap/core_kernel_train.jsonl`` (+ ``.meta.json``) by default.

Examples
--------
    # default: discover buckets, stratify, cap at 5000, write curated set
    python scripts/build_core_kernel_set.py

    # explicit inputs + tighter cap
    python scripts/build_core_kernel_set.py --jsonl data/bootstrap/*.jsonl \
        --max-games 4000 --out data/bootstrap/core_kernel_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict, OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import paths
from poke_bot.core_kernel import StreamingArchetypeCorpus


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--jsonl", nargs="*", default=None,
                   help="Explicit bucket JSONL paths/globs (default: auto-discover buckets).")
    p.add_argument("--bucket-dir", type=Path, default=paths.DATA_DIR / "bootstrap",
                   help="Directory of per-archetype bucket JSONLs to discover.")
    p.add_argument("--out", type=Path,
                   default=paths.DATA_DIR / "bootstrap" / "core_kernel_train.jsonl",
                   help="Output curated capped JSONL.")
    p.add_argument("--max-games", type=int, default=5000,
                   help="HARD CAP on total game trajectories in the set (default 5000).")
    p.add_argument("--include-smoke", action="store_true",
                   help="Include *.smoke.jsonl buckets in discovery.")
    p.add_argument("--require-info-set-ok", action="store_true", default=True,
                   help="Keep only records whose info_set_ok is truthy (default on).")
    p.add_argument("--allow-info-set-bad", dest="require_info_set_ok",
                   action="store_false", help="Do not filter on info_set_ok.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _resolve_jsonls(args: argparse.Namespace) -> list[Path]:
    if args.jsonl:
        out: list[Path] = []
        for pattern in args.jsonl:
            pp = Path(pattern)
            if pp.is_file():
                out.append(pp)
            else:
                out.extend(sorted(Path().glob(pattern)))
        # never let the output file feed back into itself
        return [p for p in out if p.resolve() != args.out.resolve()]
    found = StreamingArchetypeCorpus.discover_bucket_jsonls(
        args.bucket_dir, include_smoke=args.include_smoke
    )
    return [p for p in found if p.resolve() != args.out.resolve()]


def main(argv=None) -> int:
    args = _parse_args(argv)
    paths.ensure_runtime_dirs()
    rng = random.Random(args.seed)

    jsonls = _resolve_jsonls(args)
    if not jsonls:
        print(f"ERROR: no bucket JSONL found under {args.bucket_dir}", file=sys.stderr)
        return 2
    print(f">> inputs ({len(jsonls)}): {[p.name for p in jsonls]}", flush=True)

    # 1) Load + de-dup by (episode_id, seat), grouped by archetype.
    by_arch: dict[str, list[dict]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    dropped_info = 0
    dropped_dup = 0
    total_read = 0
    for path in jsonls:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total_read += 1
                if args.require_info_set_ok and not rec.get("info_set_ok", True):
                    dropped_info += 1
                    continue
                key = (str(rec.get("episode_id", "")), int(rec.get("seat", 0)))
                if key in seen:
                    dropped_dup += 1
                    continue
                seen.add(key)
                arch = rec.get("archetype") or "unknown"
                by_arch[arch].append(rec)

    for arch in by_arch:
        rng.shuffle(by_arch[arch])

    avail = {a: len(v) for a, v in by_arch.items()}
    print(f">> read={total_read} unique={len(seen)} "
          f"dropped(info_set)={dropped_info} dropped(dup)={dropped_dup}", flush=True)
    print(f">> per-archetype available: {dict(sorted(avail.items()))}", flush=True)

    # 2) Stratified round-robin draw across archetypes up to the hard cap.
    cap = max(0, int(args.max_games))
    cursors = {a: 0 for a in by_arch}
    order = sorted(by_arch.keys())
    chosen: list[dict] = []
    while len(chosen) < cap:
        progressed = False
        for arch in order:
            if len(chosen) >= cap:
                break
            i = cursors[arch]
            bucket = by_arch[arch]
            if i < len(bucket):
                chosen.append(bucket[i])
                cursors[arch] = i + 1
                progressed = True
        if not progressed:  # every bucket exhausted
            break

    rng.shuffle(chosen)
    picked = Counter(r.get("archetype") or "unknown" for r in chosen)

    # 3) Atomic write of the curated set.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in chosen:
            fh.write(json.dumps(rec) + "\n")
    tmp.replace(args.out)

    n_decisions = sum(int(r.get("n_decisions", len(r.get("steps", []) or []))) for r in chosen)
    meta = {
        "out": str(args.out),
        "max_games_cap": cap,
        "n_games": len(chosen),
        "n_decisions": n_decisions,
        "under_cap": len(chosen) <= cap,
        "per_archetype": dict(sorted(picked.items())),
        "per_archetype_available": dict(sorted(avail.items())),
        "inputs": [str(p) for p in jsonls],
        "require_info_set_ok": bool(args.require_info_set_ok),
        "dropped_info_set": dropped_info,
        "dropped_dup": dropped_dup,
        "seed": args.seed,
    }
    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    assert len(chosen) <= cap, "HARD CAP violated"
    print(f">> wrote {len(chosen)} games ({n_decisions} decisions) → {args.out}", flush=True)
    print(f">> stratified picks: {dict(sorted(picked.items()))}", flush=True)
    print(f">> HARD CAP <= {cap} games: {'OK' if len(chosen) <= cap else 'VIOLATED'}", flush=True)
    print(f">> meta → {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
