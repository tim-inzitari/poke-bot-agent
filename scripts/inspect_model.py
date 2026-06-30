#!/usr/bin/env python3
"""Print checkpoint/model/deck details for visible long-running jobs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_agent.models.temporal_transformer import TemporalTransformer


def read_deck(path: Path) -> list[int]:
    return [int(token) for token in path.read_text(encoding="utf-8").replace(",", "\n").split()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a trained poke-agent checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--deck", type=Path, default=None)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["model_config"]
    model = TemporalTransformer(
        int(checkpoint["input_dim"]),
        int(checkpoint["policy_dim"]),
        d_model=int(cfg["d_model"]),
        nhead=int(cfg["heads"]),
        num_layers=int(cfg["layers"]),
        dim_feedforward=int(cfg["dim_feedforward"]),
        dropout=float(cfg["dropout"]),
        window_size=int(cfg["window_size"]),
        kan_grid_size=int(cfg.get("kan_grid_size", 8)),
        use_kan=bool(cfg.get("use_kan", False)),
    )
    params = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)

    print("Model info")
    print("----------")
    print(f"checkpoint: {args.checkpoint}")
    if args.deck is not None:
        deck = read_deck(args.deck)
        print(f"deck: {args.deck} ({len(deck)} cards, first={deck[:12]})")
    print(f"architecture: {checkpoint.get('architecture') or checkpoint.get('model_type') or 'unknown'}")
    print(f"input_dim: {checkpoint['input_dim']}")
    print(f"policy_dim: {checkpoint['policy_dim']}")
    print(f"window_size: {cfg['window_size']}")
    print(f"d_model: {cfg['d_model']}")
    print(f"heads: {cfg['heads']}")
    print(f"layers: {cfg['layers']}")
    print(f"dim_feedforward: {cfg['dim_feedforward']}")
    print(f"dropout: {cfg['dropout']}")
    print(f"use_kan: {cfg.get('use_kan', False)}")
    print(f"kan_grid_size: {cfg.get('kan_grid_size', 8)}")
    print(f"parameters: {params:,}")
    print(f"trainable_parameters: {trainable:,}")
    if checkpoint.get("data_path"):
        print(f"trained_from: {checkpoint['data_path']}")


if __name__ == "__main__":
    main()
