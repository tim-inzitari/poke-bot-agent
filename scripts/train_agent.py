#!/usr/bin/env python3
"""Run the poke agent training pipeline outside the notebook."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_agent.main import main


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.resume:
        os.environ["TRAIN_RESUME"] = "1"
    elif args.no_resume:
        os.environ["TRAIN_RESUME"] = "0"
    main()
