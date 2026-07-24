#!/usr/bin/env python3
"""Merge a validated adapter-only fit into its exact parent as runtime-off.

This command never edits the parent and never imports the adapter optimizer.
The output is immutable and remains a dormant continuation artifact; it is not
a promoted production learner.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import checkpoint  # noqa: E402
from poke_bot.matchup_adapter_activation import (  # noqa: E402
    merge_dormant_adapter_checkpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--activation-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permit-post-boundary-use", action="store_true")
    parser.add_argument("--import-optimizer-state", action="store_true")
    parser.add_argument("--accumulate-parent-fit", action="store_true")
    args = parser.parse_args()

    output = merge_dormant_adapter_checkpoint(
        parent_checkpoint=args.parent_checkpoint,
        adapter_checkpoint=args.adapter_checkpoint,
        activation_receipt=args.activation_receipt,
        output_path=args.output,
        permit_post_boundary_use=bool(args.permit_post_boundary_use),
        import_optimizer_state=bool(args.import_optimizer_state),
        accumulate_parent_fit=bool(args.accumulate_parent_fit),
    )
    print(
        json.dumps(
            {
                "status": "DORMANT_MERGE_COMPLETE",
                "output": str(output),
                "output_digest": checkpoint.checkpoint_digest(output),
                "runtime_enabled": False,
                "optimizer_imported": bool(args.import_optimizer_state),
                "production_promoted": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
