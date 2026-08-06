#!/usr/bin/env python3
"""Co-train Recursive Turn Planner while a pure-RL run collects shards.

Watches ``--run-dir/shards`` for new or grown shard files and launches the
archetype RTP pipeline against the live seed/champion checkpoint. Writes a
stable ``--live-checkpoint`` path that the RL service can load via
``POKEBOT_RTP_CHECKPOINT`` (use when possible; missing → greedy).

Prefer ``--device cuda:0`` (3080) so Blackwell stays free for RL.

Does not grant selector/serving authority. Sidecar only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.recursive_turn_planner.pipeline import (  # noqa: E402
    ArchetypeRTPJob,
    run_archetype_rtp_pipeline,
)


def _shard_fingerprint(paths: list[Path]) -> tuple[tuple[str, int, int], ...]:
    rows = []
    for path in sorted(paths, key=lambda p: p.name):
        if not path.is_file():
            continue
        st = path.stat()
        rows.append((path.name, int(st.st_mtime_ns), int(st.st_size)))
    return tuple(rows)


def _list_rl_shards(run_dir: Path) -> list[Path]:
    shard_dir = run_dir / "shards"
    if not shard_dir.is_dir():
        return []
    return sorted(shard_dir.glob("iter_*.jsonl"))


def _resolve_parent_checkpoint(run_dir: Path, fallback: Path | None) -> Path:
    loop = run_dir / "loop_state.json"
    if loop.is_file():
        state = json.loads(loop.read_text(encoding="utf-8"))
        for key in ("champion", "learner", "lineage_base"):
            row = state.get(key) or {}
            path = Path(str(row.get("path") or ""))
            if path.is_file():
                return path
    seed = run_dir / "checkpoints" / "seed.pt"
    if seed.is_file():
        return seed
    if fallback is not None and fallback.is_file():
        return fallback
    raise FileNotFoundError(f"no parent checkpoint under {run_dir}")


def _publish_live_checkpoint(src: Path, live: Path) -> None:
    live.parent.mkdir(parents=True, exist_ok=True)
    tmp = live.with_suffix(live.suffix + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, live)
    receipt = src.with_name(src.name + ".receipt.json")
    if receipt.is_file():
        shutil.copy2(receipt, live.with_name(live.name + ".receipt.json"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--specialist-id", type=str, default="crustle")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--live-checkpoint",
        type=Path,
        required=True,
        help="Stable path RL workers read via POKEBOT_RTP_CHECKPOINT",
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-games", type=int, default=512)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Prefer cuda:0 (3080) so Blackwell stays free for RL",
    )
    parser.add_argument("--also-poke-rlm", action="store_true")
    parser.add_argument("--min-shard-bytes", type=int, default=1_000_000)
    parser.add_argument("--status", type=Path, default=None)
    parser.add_argument(
        "--seed-checkpoint",
        type=Path,
        default=None,
        help="Optional RTP checkpoint to publish as live before first train",
    )
    parser.add_argument(
        "--seed-shard",
        action="append",
        default=[],
        help="Bootstrap shard(s) used until RL produces shards/",
    )
    parser.add_argument(
        "--parent-checkpoint",
        type=Path,
        default=None,
        help="Fallback parent/expert checkpoint when run-dir has none yet",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    out_root = args.out_dir.expanduser().resolve()
    live = args.live_checkpoint.expanduser().resolve()
    parent_fallback = (
        args.parent_checkpoint.expanduser().resolve()
        if args.parent_checkpoint
        else None
    )
    seed_shards = [
        Path(p).expanduser().resolve()
        for p in args.seed_shard
        if Path(p).expanduser().is_file()
    ]
    status_path = (
        args.status.expanduser().resolve()
        if args.status
        else out_root / args.specialist_id / "cotrain_status.json"
    )
    last_fp: tuple[tuple[str, int, int], ...] | None = None
    seeded_once = False
    pass_idx = 0

    if args.seed_checkpoint is not None:
        seed_ckpt = args.seed_checkpoint.expanduser().resolve()
        if seed_ckpt.is_file():
            _publish_live_checkpoint(seed_ckpt, live)
            print(f"RTP_COTRAIN seeded live from {seed_ckpt}", flush=True)

    print(
        f"RTP_COTRAIN watching run_dir={run_dir} live={live} "
        f"device={args.device} poll={args.poll_seconds}s",
        flush=True,
    )
    while True:
        try:
            rl_shards = _list_rl_shards(run_dir)
            using_seed = False
            if rl_shards:
                candidates = rl_shards
            elif seed_shards and not seeded_once:
                candidates = seed_shards
                using_seed = True
            else:
                candidates = []

            fp = _shard_fingerprint(candidates)
            ready = [
                (name, mt, size)
                for name, mt, size in fp
                if size >= int(args.min_shard_bytes) or using_seed
            ]
            if ready and fp != last_fp:
                parent = _resolve_parent_checkpoint(run_dir, parent_fallback)
                newest_name = sorted(ready, key=lambda row: row[0])[-1][0]
                shard = next(p for p in candidates if p.name == newest_name)
                pass_idx += 1
                print(
                    f"RTP_COTRAIN pass={pass_idx} source={'seed' if using_seed else 'rl'} "
                    f"shard={shard.name} parent={parent} size={shard.stat().st_size}",
                    flush=True,
                )
                job = ArchetypeRTPJob(
                    specialist_id=str(args.specialist_id),
                    parent_checkpoint=str(parent),
                    training_shard=str(shard),
                    profile="pure_rl",
                    d_model=96,
                    epochs=int(args.epochs),
                    max_games=int(args.max_games),
                    device=str(args.device),
                    also_poke_rlm=bool(args.also_poke_rlm),
                    notes="Concurrent co-train with pure-RL collection",
                )
                result = run_archetype_rtp_pipeline(
                    job,
                    out_root=out_root / f"cotrain_pass_{pass_idx:04d}",
                    synthetic=False,
                )
                _publish_live_checkpoint(Path(result.rtp_checkpoint), live)
                payload = {
                    "schema": "poke_bot.rtp_cotrain_with_rl/v1",
                    "updated_at_unix": time.time(),
                    "pass": pass_idx,
                    "source": "seed" if using_seed else "rl",
                    "run_dir": str(run_dir),
                    "shard": str(shard),
                    "parent_checkpoint": str(parent),
                    "live_checkpoint": str(live),
                    "rtp_checkpoint": result.rtp_checkpoint,
                    "device": str(args.device),
                    "metrics": result.metrics,
                    "env": {
                        "POKEBOT_USE_RECURSIVE_TURN_PLANNER": "1",
                        "POKEBOT_RTP_CHECKPOINT": str(live),
                        "POKEBOT_RTP_SPECIALIST_ID": str(args.specialist_id),
                    },
                    "serving_eligible": False,
                }
                status_path.parent.mkdir(parents=True, exist_ok=True)
                status_path.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                print(
                    f"RTP_COTRAIN published live={live} "
                    f"loss={((result.metrics or {}).get('rtp') or {}).get('mean_loss')}",
                    flush=True,
                )
                last_fp = fp
                if using_seed:
                    seeded_once = True
            else:
                print(
                    f"RTP_COTRAIN idle rl_shards={len(rl_shards)} "
                    f"ready={len(ready)} seeded_once={seeded_once}",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001
            print(f"RTP_COTRAIN error: {type(exc).__name__}: {exc}", flush=True)
        time.sleep(float(args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
