#!/usr/bin/env python3
"""Immutably add a frozen zero-output matchup bank to the iter-15 parent."""

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
    materialize_zero_dormant_adapter_checkpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--activation-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = materialize_zero_dormant_adapter_checkpoint(
        parent_checkpoint=args.parent_checkpoint,
        activation_receipt=args.activation_receipt,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "status": "ZERO_DORMANT_BANK_MATERIALIZED",
                "output": str(output),
                "output_digest": checkpoint.checkpoint_digest(output),
                "runtime_enabled": False,
                "training_enabled": False,
                "optimizer_imported": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
