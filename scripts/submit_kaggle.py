#!/usr/bin/env python3
"""Build the Kaggle submission bundle from a checkpoint and submit it.

Wraps scripts/build_submission.sh (sets VALUE_MODEL_PATH) + `kaggle competitions
submit`. Used standalone or by scripts/run_full_pipeline.sh.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_agent.kaggle_submit import DEFAULT_SUBMISSION_MESSAGE, submit_champion_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs/checkpoints/temporal_current.pt",
        help="checkpoint to package and submit",
    )
    parser.add_argument(
        "--message",
        default=DEFAULT_SUBMISSION_MESSAGE,
        help="Kaggle submission message",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")

    submit_champion_checkpoint(checkpoint, root=ROOT, message=args.message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
