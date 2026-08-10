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
    parser.add_argument(
        "--complete-action-corpus",
        type=Path,
        default=None,
        help=(
            "Immutable materialized r197 corpus directory (or MANIFEST.json). "
            "Required with --profile pure_rl_r197; never combined with --shard."
        ),
    )
    parser.add_argument(
        "--parent-digest",
        type=str,
        default="",
        help="Expected sha256 of --checkpoint; verified against bytes before load.",
    )
    parser.add_argument(
        "--shard-digest",
        type=str,
        default="",
        help="Expected sha256 of --shard; verified and bound when supplied.",
    )
    parser.add_argument(
        "--complete-action-corpus-manifest-digest",
        type=str,
        default="",
        help="Expected sha256 of the r197 corpus MANIFEST.json; required for r197.",
    )
    parser.add_argument(
        "--complete-action-corpus-receipt-digest",
        type=str,
        default="",
        help="Expected sha256 of the r197 corpus RECEIPT.json; required for r197.",
    )
    parser.add_argument(
        "--complete-action-corpus-source-pointer-digest",
        type=str,
        default="",
        help="Protected source-pointer sha256 in the r197 manifest; required for r197.",
    )
    parser.add_argument(
        "--complete-action-corpus-selection-plan-digest",
        type=str,
        default="",
        help="Preflight r197 whole-episode selection plan sha256; required for r197.",
    )
    parser.add_argument(
        "--complete-action-corpus-train-selection-digest",
        type=str,
        default="",
        help="Preflight r197 retained train episode-set sha256; required for r197.",
    )
    parser.add_argument(
        "--complete-action-corpus-heldout-selection-digest",
        type=str,
        default="",
        help="Preflight r197 retained heldout episode-set sha256; required for r197.",
    )
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
    parser.add_argument("--num-plan-candidates", type=int, default=4)
    parser.add_argument("--max-recursion-depth", type=int, default=2)
    parser.add_argument(
        "--max-neural-passes",
        type=int,
        default=4,
        help="Legacy default; pure_rl_r197 requires exactly 256.",
    )
    parser.add_argument("--heldout-fraction", type=float, default=0.20)
    parser.add_argument(
        "--max-runtime-action-combos",
        type=int,
        default=256,
        help="Legacy default; pure_rl_r197 requires exactly 1024.",
    )
    parser.add_argument("--max-train-games", type=int, default=0)
    parser.add_argument("--max-heldout-games", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-heldout-batches", type=int, default=0)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help=(
            "Whole-game split seed. pure_rl_r197 requires exactly 5000000; "
            "legacy jobs retain their historical seed behavior."
        ),
    )
    parser.add_argument(
        "--allow-factorized-only",
        action="store_true",
        help=(
            "Permit legacy factorized-prefix rows for offline diagnostics. "
            "Such a run remains shadow-only and is not runtime-action aligned."
        ),
    )
    parser.add_argument("--also-poke-rlm", action="store_true")
    args = parser.parse_args()
    if args.shard is not None and args.complete_action_corpus is not None:
        parser.error("--shard and --complete-action-corpus are mutually exclusive")

    job = ArchetypeRTPJob(
        specialist_id=str(args.specialist_id),
        parent_checkpoint=str(args.checkpoint or ""),
        training_shard=str(args.shard or ""),
        parent_digest=str(args.parent_digest),
        training_shard_digest=str(args.shard_digest),
        complete_action_corpus=str(args.complete_action_corpus or ""),
        complete_action_corpus_manifest_digest=str(
            args.complete_action_corpus_manifest_digest
        ),
        complete_action_corpus_receipt_digest=str(
            args.complete_action_corpus_receipt_digest
        ),
        complete_action_corpus_source_pointer_digest=str(
            args.complete_action_corpus_source_pointer_digest
        ),
        complete_action_corpus_selection_plan_digest=str(
            args.complete_action_corpus_selection_plan_digest
        ),
        complete_action_corpus_train_selection_digest=str(
            args.complete_action_corpus_train_selection_digest
        ),
        complete_action_corpus_heldout_selection_digest=str(
            args.complete_action_corpus_heldout_selection_digest
        ),
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
        num_plan_candidates=int(args.num_plan_candidates),
        max_recursion_depth=int(args.max_recursion_depth),
        max_neural_passes=int(args.max_neural_passes),
        heldout_fraction=float(args.heldout_fraction),
        require_complete_ordered_actions=not bool(args.allow_factorized_only),
        max_runtime_action_combos=int(args.max_runtime_action_combos),
        split_seed=int(args.split_seed),
        max_train_games=int(args.max_train_games),
        max_heldout_games=int(args.max_heldout_games),
        max_train_batches=int(args.max_train_batches),
        max_heldout_batches=int(args.max_heldout_batches),
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
