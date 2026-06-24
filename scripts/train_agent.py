#!/usr/bin/env python3
"""Run the poke agent training pipeline outside the notebook."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_agent.main import main


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__
        + "\n\nTensor cache (TRAIN_TENSOR_CACHE): auto loads disk cache when valid "
        "so resume skips JSONL parse + feature build. Pre-build with "
        "scripts/build_tensor_cache.py.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--resume",
        action="store_true",
        help="require resuming from <checkpoint>.latest.pt (errors if missing)",
    )
    group.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore any existing <checkpoint>.latest.pt and train from scratch",
    )

    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--rebuild-tensors",
        action="store_true",
        help="rebuild training tensors from JSONL (ignore disk cache)",
    )
    cache_group.add_argument(
        "--require-tensor-cache",
        action="store_true",
        help="require a valid disk tensor cache (error if missing)",
    )
    cache_group.add_argument(
        "--no-tensor-cache",
        action="store_true",
        help="always build tensors from JSONL; do not read or write disk cache",
    )
    parser.add_argument(
        "--tensors-only",
        action="store_true",
        help="build or load tensor cache and exit (no training)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.resume:
        os.environ["TRAIN_RESUME"] = "1"
    elif args.no_resume:
        os.environ["TRAIN_RESUME"] = "0"

    if args.rebuild_tensors:
        os.environ["TRAIN_TENSOR_CACHE"] = "rebuild"
    elif args.require_tensor_cache:
        os.environ["TRAIN_TENSOR_CACHE"] = "require"
    elif args.no_tensor_cache:
        os.environ["TRAIN_TENSOR_CACHE"] = "0"
    else:
        os.environ.setdefault("TRAIN_TENSOR_CACHE", "auto")

    main(tensors_only=args.tensors_only)
