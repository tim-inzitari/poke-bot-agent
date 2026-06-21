#!/usr/bin/env python3
"""Minimal end-to-end training smoke test (1 game, few epochs)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from poke_agent.checkpoint import print_training_report, save_checkpoint
from poke_agent.config import build_config
from poke_agent.dataset import load_jsonl, prepare_training_tensors
from poke_agent.device import torch_device
from poke_agent.paths import resolve_root
from poke_agent.training import build_model, train_model

DEFAULT_DATA = "data/scraped_rollouts_smoke.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA, help="Rollout JSONL path")
    parser.add_argument("--games", type=int, default=1, help="Games to train on")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument(
        "--out",
        default="outputs/checkpoints/smoke_train.pt",
        help="Checkpoint output path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_root()
    data_path = str(args.data)

    overrides = {
        "primary_rollout_data": data_path,
        "merged_rollout_data": data_path,
        "training_rollout_sources": [data_path],
        "cabt_generated_path": data_path,
        "require_cabt_eval_data": False,
        "require_complete_games": True,
        "require_training_matchup_diversity": False,
        "dataset_games": args.games,
        "train_epochs": args.epochs,
        "batch_games": 1,
        "early_stop_patience": max(1, args.epochs - 1),
        "train_print_every": 1,
        "model_output_path": args.out,
        "model_id": "smoke_train",
    }

    config = build_config(root, overrides=overrides)
    device = torch_device()
    rows = load_jsonl(config["merged_rollout_path"])
    episodes = len({int(row["episode"]) for row in rows})
    print(f"smoke data: {len(rows)} rows, {episodes} episodes from {config['merged_rollout_path']}")

    tensors = prepare_training_tensors(config, device)
    print(f"feature dim {tensors.x.shape[1]}, games {tensors.num_games}, steps {tensors.x.shape[0]}")

    model = build_model(config, tensors, device)
    report = train_model(model, tensors, config, device)
    report = save_checkpoint(
        model=model,
        tensors=tensors,
        config=config,
        training_report=report,
        output_path=config["output_path"],
    )
    print_training_report(report, config["output_path"])


if __name__ == "__main__":
    main()
