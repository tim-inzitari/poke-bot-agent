#!/usr/bin/env python3
"""Materialize the checksum-bound r274 zero-safe successor on Inzi."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from poke_bot.r241_own_deck_successor import (
    load_r260_owner_contract,
    materialize_r260_own_deck_successor,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--source-closure", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    result = materialize_r260_own_deck_successor(
        parent_checkpoint=args.parent,
        output_checkpoint=args.output,
        receipt_path=args.receipt,
        source_closure=args.source_closure,
        owner_contract=load_r260_owner_contract(),
    )
    print(
        json.dumps(
            {
                "checkpoint": result.checkpoint.as_dict(),
                "receipt_path": str(result.receipt_path),
                "receipt_sha256": result.receipt_sha256,
                "added_tensor_keys": list(result.added_tensor_keys),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
