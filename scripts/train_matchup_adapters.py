#!/usr/bin/env python3
"""Fit dormant Alakazam matchup adapters from staged route shards.

This command is intentionally offline and isolated.  It never edits or
restarts pure-RL production, and its checkpoints remain runtime-disabled.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.pure_rl.matchup_adapter_trainer import (
    StreamingAdapterTrainConfig,
    train_matchup_adapters_streaming,
)


def _device(value: str) -> torch.device:
    try:
        result = torch.device(str(value).strip())
    except (RuntimeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if result.type not in {"cpu", "cuda", "mps"}:
        raise argparse.ArgumentTypeError("device must be cpu, cuda[:N], or mps")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged-manifest", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--activation-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device",
        type=_device,
        default=_device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    parser.add_argument("--run-name", default="alakazam_matchup_adapters_iter15")
    parser.add_argument("--resume", default="auto")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--games-per-batch", type=int, default=4)
    parser.add_argument("--max-decisions-per-batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-5)
    parser.add_argument("--checkpoint-every-steps", type=int, default=25)
    parser.add_argument("--log-every-steps", type=int, default=10)
    parser.add_argument("--max-process-rss-gib", type=float, default=8.0)
    parser.add_argument("--min-available-ram-gib", type=float, default=16.0)
    parser.add_argument("--memory-check-every-batches", type=int, default=1)
    parser.add_argument(
        "--stop-after-steps",
        type=int,
        default=None,
        help="controlled smoke-test stop after a fully verified optimizer step",
    )
    parser.add_argument(
        "--permit-post-boundary-use",
        action="store_true",
        help=(
            "Use a previously issued clean-boundary rehearsal authorization "
            "after the authorized iteration has started."
        ),
    )
    parser.add_argument(
        "--restore-parent-optimizer-state",
        action="store_true",
        help=(
            "Continue the isolated adapter AdamW moments embedded in the "
            "authorized parent checkpoint."
        ),
    )
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    cfg = StreamingAdapterTrainConfig(
        epochs=args.epochs,
        games_per_batch=args.games_per_batch,
        max_decisions_per_batch=args.max_decisions_per_batch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        value_loss_weight=args.value_loss_weight,
        grad_clip=args.grad_clip,
        amp=not args.no_amp,
        seed=args.seed,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        checkpoint_every_steps=args.checkpoint_every_steps,
        log_every_steps=args.log_every_steps,
        max_process_rss_gib=args.max_process_rss_gib,
        min_available_ram_gib=args.min_available_ram_gib,
        memory_check_every_batches=args.memory_check_every_batches,
    )
    result = train_matchup_adapters_streaming(
        staged_manifest=args.staged_manifest,
        parent_checkpoint=args.parent_checkpoint,
        activation_receipt=args.activation_receipt,
        output_dir=args.output_dir,
        train_config=cfg,
        device=args.device,
        resume=args.resume,
        run_name=args.run_name,
        stop_after_steps=args.stop_after_steps,
        permit_post_boundary_use=bool(args.permit_post_boundary_use),
        restore_parent_optimizer_state=bool(
            args.restore_parent_optimizer_state
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
