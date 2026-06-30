#!/usr/bin/env python3
"""Train/warm-start the neural policy from a rollout JSONL file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_agent.checkpoint import print_training_report, save_checkpoint
from poke_agent.config import build_config
from poke_agent.device import torch_device
from poke_agent.self_play import train_on_rollouts


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm-start train on active rollout JSONL.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint-in", type=Path, default=ROOT / "outputs/checkpoints/temporal_current.pt")
    parser.add_argument("--checkpoint-out", type=Path, required=True)
    parser.add_argument("--deck", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-games", type=int, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--early-stop-min-delta", type=float, default=None)
    args = parser.parse_args()

    overrides: dict[str, object] = {
        "primary_rollout_data": str(args.data),
        "merged_rollout_data": str(args.data),
        "training_rollout_sources": [str(args.data)],
        "require_cabt_eval_data": False,
        "require_training_matchup_diversity": False,
        "dataset_games": 0,
        "model_output_path": str(args.checkpoint_out),
    }
    if args.deck is not None:
        overrides["agent_deck_path"] = str(args.deck)
    if args.epochs is not None:
        overrides["self_play_train_epochs"] = args.epochs
    if args.batch_games is not None:
        overrides["batch_games"] = args.batch_games
    if args.early_stop_patience is not None:
        overrides["early_stop_patience"] = args.early_stop_patience
    if args.early_stop_min_delta is not None:
        overrides["early_stop_min_delta"] = args.early_stop_min_delta

    config = build_config(ROOT, overrides=overrides)
    device = torch_device()
    print("device", device)
    print("data", args.data)
    print("checkpoint in", args.checkpoint_in)
    print("checkpoint out", args.checkpoint_out)
    model, tensors, report = train_on_rollouts(
        config,
        device,
        data_path=args.data,
        checkpoint_path=args.checkpoint_in if args.checkpoint_in.exists() else None,
    )
    report = save_checkpoint(
        model=model,
        tensors=tensors,
        config=config,
        training_report=report,
        output_path=args.checkpoint_out,
    )
    print_training_report(report, args.checkpoint_out)


if __name__ == "__main__":
    main()
