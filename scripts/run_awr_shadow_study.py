#!/usr/bin/env python3
"""Prepare and run an isolated, promotion-disabled AWR shadow study.

The default action is preparation only.  Candidate fitting requires the
literal ``--ack-shadow-only`` acknowledgement and the frozen manifest's
separate device.  This script has no systemd, dashboard, checkpoint-publish,
or production-ledger code path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.pure_rl.awr_shadow_study import (  # noqa: E402
    prepare_beta_manifest,
    prepare_stage2_manifest,
    finalize_beta_report,
    run_beta_study,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-beta", help="freeze beta-study inputs only")
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--parent-checkpoint", type=Path, required=True)
    prepare.add_argument(
        "--replay-shard", type=Path, action="append", required=True
    )
    prepare.add_argument(
        "--replay-receipt",
        type=Path,
        action="append",
        required=True,
        help="completed-collection receipt matching each --replay-shard in order",
    )
    prepare.add_argument("--baseline-contract", type=Path, required=True)
    prepare.add_argument("--shadow-device", default="cuda:0")
    prepare.add_argument("--production-device", default="cuda:1")
    prepare.add_argument("--seed", type=int, default=101)
    prepare.add_argument("--epochs", type=int, default=2)
    prepare.add_argument("--games-per-batch", type=int, default=240)
    prepare.add_argument("--val-frac", type=float, default=0.1)
    prepare.add_argument("--max-context", type=int, choices=(320,), default=320)
    prepare.add_argument("--archetype-id", default="alakazam")
    prepare.add_argument("--value-loss-weight", type=float, default=1.0)
    prepare.add_argument("--archetype-aux-loss-weight", type=float, default=0.05)
    prepare.add_argument("--opp-hand-loss-weight", type=float, default=0.05)
    prepare.add_argument("--opp-remainder-loss-weight", type=float, default=0.05)
    prepare.add_argument("--lethal-threat-loss-weight", type=float, default=0.025)
    prepare.add_argument("--prize-race-loss-weight", type=float, default=0.025)
    prepare.add_argument("--alakazam-guide-loss-weight", type=float, default=0.0)

    run = sub.add_parser(
        "run-beta",
        help="fit beta candidates on the isolated device; never promote/apply",
    )
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--device", required=True)
    run.add_argument(
        "--ack-shadow-only",
        action="store_true",
        help="confirm candidates stay outside production and promotion is disabled",
    )

    finalize = sub.add_parser(
        "finalize-beta",
        help="validate matched rows and emit the immutable final beta report",
    )
    finalize.add_argument("--manifest", type=Path, required=True)
    finalize.add_argument("--evaluation-rows", type=Path, required=True)

    stage2 = sub.add_parser(
        "prepare-stage2",
        help="prepare, but do not start, the guarded post-beta matrix",
    )
    stage2.add_argument("--beta-manifest", type=Path, required=True)
    stage2.add_argument("--final-beta-report", type=Path, required=True)
    stage2.add_argument("--output-dir", type=Path, required=True)
    stage2.add_argument(
        "--replay-shard", type=Path, action="append", required=True
    )
    stage2.add_argument(
        "--replay-receipt",
        type=Path,
        action="append",
        required=True,
        help="completed-collection receipt matching each --replay-shard in order",
    )
    stage2.add_argument("--seed", type=int, default=202)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare-beta":
        path = prepare_beta_manifest(
            output_dir=args.output_dir,
            parent_checkpoint=args.parent_checkpoint,
            replay_shards=args.replay_shard,
            replay_receipts=args.replay_receipt,
            baseline_contract=args.baseline_contract,
            shadow_device=args.shadow_device,
            production_device=args.production_device,
            seed=args.seed,
            epochs=args.epochs,
            games_per_batch=args.games_per_batch,
            val_frac=args.val_frac,
            max_context=args.max_context,
            archetype_id=args.archetype_id,
            loss_weights={
                "value": args.value_loss_weight,
                "archetype_aux": args.archetype_aux_loss_weight,
                "opponent_hand": args.opp_hand_loss_weight,
                "opponent_remainder": args.opp_remainder_loss_weight,
                "lethal_threat": args.lethal_threat_loss_weight,
                "prize_race": args.prize_race_loss_weight,
                "alakazam_guide": args.alakazam_guide_loss_weight,
            },
        )
    elif args.command == "run-beta":
        if not args.ack_shadow_only:
            raise SystemExit("run-beta requires --ack-shadow-only")
        path = run_beta_study(args.manifest, device=args.device)
    elif args.command == "finalize-beta":
        path = finalize_beta_report(
            args.manifest,
            evaluation_rows_path=args.evaluation_rows,
        )
    elif args.command == "prepare-stage2":
        path = prepare_stage2_manifest(
            beta_manifest_path=args.beta_manifest,
            final_beta_report_path=args.final_beta_report,
            output_dir=args.output_dir,
            replay_shards=args.replay_shard,
            replay_receipts=args.replay_receipt,
            seed=args.seed,
        )
    else:  # pragma: no cover - argparse enforces the command set.
        raise AssertionError(args.command)
    print(json.dumps({"ok": True, "artifact": str(path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
