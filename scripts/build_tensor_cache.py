#!/usr/bin/env python3
"""Build or refresh the on-disk training tensor cache (no model training).

Use this after data prep so train_agent.py can skip JSONL parse + feature build:

  python scripts/build_tensor_cache.py
  python scripts/train_agent.py   # loads cache if valid

Force rebuild after data changes:

  python scripts/build_tensor_cache.py --rebuild
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from poke_agent.config import build_config
from poke_agent.dataset import prepare_training_tensors
from poke_agent.paths import resolve_root
from poke_agent.tensor_cache import (
    describe_training_tensor_cache,
    has_valid_training_tensor_cache,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="ignore existing cache and rebuild from JSONL",
    )
    parser.add_argument(
        "--require",
        action="store_true",
        help="error if cache is missing or stale (no JSONL build)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.rebuild:
        os.environ["TRAIN_TENSOR_CACHE"] = "rebuild"
    elif args.require:
        os.environ["TRAIN_TENSOR_CACHE"] = "require"
    else:
        os.environ.setdefault("TRAIN_TENSOR_CACHE", "auto")

    root = resolve_root()
    config = build_config(root)
    status = describe_training_tensor_cache(config)
    if status:
        print(status)

    if args.require and not has_valid_training_tensor_cache(config):
        raise SystemExit(
            "tensor cache required but not valid; run without --require to build from JSONL"
        )

    if not args.rebuild and has_valid_training_tensor_cache(config):
        print("tensor cache already valid; nothing to do")
        return

    device = torch.device("cpu")
    print("building training tensor cache...")
    tensors = prepare_training_tensors(config, device)
    print(f"done: {tensors.num_seqs:,} seat-sequences cached for training")


if __name__ == "__main__":
    main()
