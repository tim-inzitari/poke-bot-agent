#!/usr/bin/env python3
"""Build the shadow-only, complete-action Alakazam RTP r197 corpus.

This command has no checkpoint, trainer, selector, service, or submission
arguments.  It only creates a content-addressed derivative below
``--output-dir`` from the protected expert identity pointer and raw daily
episode archives.  Existing matching output is verified and reused; partial or
mismatched output is rejected without replacement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.recursive_turn_planner.r197_corpus import (  # noqa: E402
    DEFAULT_HELDOUT_FRACTION,
    MAX_ACTION_COMBOS,
    PRODUCTION_ARCHIVE_ROOT,
    SPECIALIST_ID,
    SPLIT_SEED,
    materialize_r197_complete_action_corpus,
    verify_r197_complete_action_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pointer", type=Path, required=True)
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=PRODUCTION_ARCHIVE_ROOT,
        help=(
            "raw daily episode zip root; defaults to the r175 production "
            "TrueNAS mount"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-pointer-sha256",
        default="",
        help="optional fail-closed digest for the protected pointer itself",
    )
    parser.add_argument("--specialist-id", default=SPECIALIST_ID)
    parser.add_argument("--heldout-fraction", default=DEFAULT_HELDOUT_FRACTION)
    # These are visible in the CLI so an operator sees the fixed contract.  The
    # library rejects a different value rather than silently creating a nearby
    # r197 variant.
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--max-action-combos", type=int, default=MAX_ACTION_COMBOS)
    parser.add_argument(
        "--verify-manifest",
        type=Path,
        default=None,
        help="read-only verification mode; no corpus files are created",
    )
    args = parser.parse_args()

    if args.verify_manifest is not None:
        receipt = verify_r197_complete_action_manifest(
            args.verify_manifest,
            archive_root=args.archive_root,
            require_current_generator=True,
        )
    else:
        receipt = materialize_r197_complete_action_corpus(
            args.source_pointer,
            args.archive_root,
            args.output_dir,
            specialist_id=args.specialist_id,
            split_seed=int(args.split_seed),
            heldout_fraction=args.heldout_fraction,
            max_action_combos=int(args.max_action_combos),
            expected_pointer_sha256=args.expected_pointer_sha256 or None,
        )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
