#!/usr/bin/env python3
"""Train RTP for one specialist (thin wrapper over the archetype pipeline).

Prefer ``scripts/run_rtp_archetype_pipeline.py`` for multi-deck fleets.
This script remains for single-shot / CI smoke use.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.recursive_turn_planner.pipeline import (  # noqa: E402
    ArchetypeRTPJob,
    run_archetype_rtp_pipeline,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--specialist-id",
        type=str,
        default="experimental",
        help="Canonical specialist id (alakazam, crustle, …)",
    )
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--shard", type=Path, default=None)
    parser.add_argument("--max-games", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--profile", type=str, default="pure_rl")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-synthetic", type=int, default=64)
    parser.add_argument("--complexity-option-threshold", type=int, default=8)
    parser.add_argument("--complexity-entropy-threshold", type=float, default=1.5)
    parser.add_argument("--also-poke-rlm", action="store_true")
    args = parser.parse_args()

    job = ArchetypeRTPJob(
        specialist_id=str(args.specialist_id),
        parent_checkpoint=str(args.checkpoint or ""),
        training_shard=str(args.shard or ""),
        profile=str(args.profile),
        d_model=int(args.d_model),
        max_games=int(args.max_games),
        epochs=int(args.epochs),
        lr=float(args.lr),
        seed=int(args.seed),
        device=str(args.device),
        also_poke_rlm=bool(args.also_poke_rlm),
        complexity_option_threshold=int(args.complexity_option_threshold),
        complexity_entropy_threshold=float(args.complexity_entropy_threshold),
    )
    synthetic = bool(args.synthetic) or not job.ready_for_host_train
    # Keep prior CLI layout: artifacts directly under --out-dir when specialist
    # is the default experimental id; otherwise nest under specialist_id.
    out_root = args.out_dir
    if job.specialist_id != "experimental":
        result = run_archetype_rtp_pipeline(
            job,
            out_root=out_root,
            synthetic=synthetic,
            n_synthetic=int(args.n_synthetic),
        )
    else:
        # Nest as experimental/ for consistency with fleet layout.
        result = run_archetype_rtp_pipeline(
            job,
            out_root=out_root,
            synthetic=synthetic,
            n_synthetic=int(args.n_synthetic),
        )
    print(json.dumps(result.to_json(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
