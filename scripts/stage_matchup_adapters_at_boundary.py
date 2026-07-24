#!/usr/bin/env python3
"""Create the fail-closed iteration-15 matchup-adapter activation receipt.

This command is intentionally non-orchestrating: it never stops/restarts the
production service and never edits the run.  It pins the committed iteration-15
cumulative learner so a separate adapter-only trainer can start in an isolated
output directory.  Re-running against an existing receipt fails rather than
overwriting boundary evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.matchup_adapter_activation import build_activation_receipt


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _args()
    receipt = build_activation_receipt(
        run_dir=args.run_dir,
        output_path=args.receipt,
    )
    print(
        json.dumps(
            {
                "status": "STAGED",
                "completed_iteration": receipt.completed_iteration,
                "first_eligible_iteration": receipt.first_eligible_iteration,
                "parent_checkpoint": str(receipt.parent_checkpoint),
                "parent_checkpoint_digest": receipt.parent_checkpoint_digest,
                "receipt": str(receipt.path),
                "runtime_enabled": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
