#!/usr/bin/env python3
"""Run the archetype-generic RTP pipeline for one or many specialists.

Examples:
  # Smoke all example archetypes synthetically
  python3 scripts/run_rtp_archetype_pipeline.py \\
    --registry config/rtp_archetype_pipeline.example.yaml \\
    --out-dir outputs/rtp_fleet_smoke --synthetic

  # Host: Alakazam only (paths filled in your registry copy)
  python3 scripts/run_rtp_archetype_pipeline.py \\
    --registry config/rtp_archetype_pipeline.yaml \\
    --out-dir outputs/rtp_fleet \\
    --specialist alakazam

  # Host: every ready archetype (checkpoint+shard set)
  python3 scripts/run_rtp_archetype_pipeline.py \\
    --registry config/rtp_archetype_pipeline.yaml \\
    --out-dir outputs/rtp_fleet --only-ready
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
    example_registry_jobs,
    load_archetype_registry,
    run_archetype_rtp_pipeline,
    run_registry,
    select_jobs,
)


def _apply_defaults(job: ArchetypeRTPJob, args: argparse.Namespace) -> ArchetypeRTPJob:
    if args.epochs is not None:
        job.epochs = int(args.epochs)
    if args.max_games is not None:
        job.max_games = int(args.max_games)
    if args.device:
        job.device = str(args.device)
    if args.also_poke_rlm:
        job.also_poke_rlm = True
    if args.checkpoint:
        job.parent_checkpoint = str(args.checkpoint)
    if args.shard:
        job.training_shard = str(args.shard)
    if args.profile:
        job.profile = str(args.profile)
    if args.d_model is not None:
        job.d_model = int(args.d_model)
    return job


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="YAML/JSON archetype registry (default: built-in example jobs)",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--specialist",
        action="append",
        default=[],
        help="specialist_id to run (repeatable). Default: all enabled in registry",
    )
    parser.add_argument("--only-ready", action="store_true",
                        help="Skip jobs missing checkpoint/shard")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--n-synthetic", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--also-poke-rlm", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Override parent checkpoint for selected jobs")
    parser.add_argument("--shard", type=Path, default=None,
                        help="Override training shard for selected jobs")
    parser.add_argument("--profile", type=str, default="")
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument(
        "--list",
        action="store_true",
        help="List registry jobs and exit",
    )
    args = parser.parse_args()

    if args.registry is not None:
        jobs = load_archetype_registry(args.registry)
    else:
        jobs = example_registry_jobs()

    jobs = select_jobs(
        jobs,
        specialist_ids=args.specialist or None,
        only_ready=bool(args.only_ready) and not args.synthetic,
        include_disabled=False,
    )
    jobs = [_apply_defaults(j, args) for j in jobs]

    if args.list:
        print(
            json.dumps(
                {
                    "n": len(jobs),
                    "jobs": [
                        {
                            "specialist_id": j.specialist_id,
                            "enabled": j.enabled,
                            "ready_for_host_train": j.ready_for_host_train,
                            "parent_checkpoint": j.parent_checkpoint,
                            "training_shard": j.training_shard,
                        }
                        for j in jobs
                    ],
                },
                indent=2,
            )
        )
        return 0

    if not jobs:
        raise SystemExit("no archetype jobs selected")

    if len(jobs) == 1 and (args.checkpoint or args.shard or args.specialist):
        result = run_archetype_rtp_pipeline(
            jobs[0],
            out_root=args.out_dir,
            synthetic=bool(args.synthetic) or not jobs[0].ready_for_host_train,
            n_synthetic=int(args.n_synthetic),
        )
        print(json.dumps(result.to_json(), indent=2, sort_keys=True))
        return 0

    fleet = run_registry(
        jobs,
        out_root=args.out_dir,
        synthetic=bool(args.synthetic),
        n_synthetic=int(args.n_synthetic),
    )
    print(json.dumps(fleet, indent=2, sort_keys=True))
    return 1 if fleet.get("n_errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
