#!/usr/bin/env python3
"""Validate an offline Prize-plan-v2 sidecar on sealed validation actions only.

This is deliberately a read-only companion to
``train_alakazam_prize_plan_v2.py``.  It loads only the strict separately
versioned sidecar checkpoint, verifies the same current-wrapper/r23 authority
and portable inputs, and emits a non-activation validation receipt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_alakazam_prize_plan_v2 import (  # noqa: E402
    PrizePlanV2TrainingError,
    Recent20OverlayError,
    validate_checkpoint,
)
from poke_bot.prize_plan_v2_sidecar import PrizePlanV2SidecarError  # noqa: E402


def build_argument_parser() -> argparse.ArgumentParser:
    # Keep the exact input names/types aligned with the trainer while avoiding
    # a required output-dir or optimizer flags in this read-only utility.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-view", type=Path, required=True)
    parser.add_argument("--training-view-sha256", required=True)
    parser.add_argument("--target-view", type=Path, default=None)
    parser.add_argument("--target-view-sha256", default="")
    parser.add_argument("--target-set-root", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--target-manifest", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--target-manifest-sha256", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--target-set-receipt", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--target-set-receipt-sha256", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "goals/alakazam-elmo-rule-derivative/contract.json",
    )
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-validation-programs", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--test-mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--test-allow-noncanonical-split", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--test-skip-input-shard-sha256", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.batch_size < 1 or args.max_validation_programs < 0:
        print("error: batch size and max validation programs are invalid", file=sys.stderr)
        return 2
    try:
        result = validate_checkpoint(args, output_receipt=args.output_receipt)
    except (PrizePlanV2TrainingError, Recent20OverlayError, PrizePlanV2SidecarError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
