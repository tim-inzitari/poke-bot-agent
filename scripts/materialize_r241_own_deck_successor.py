#!/usr/bin/env python3
"""Materialize the r260 pre-start OwnDeckLedger child; never starts a service."""

from __future__ import annotations

import argparse
from pathlib import Path

from poke_bot.r241_own_deck_successor import (
    load_r260_owner_contract,
    materialize_r260_own_deck_successor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", required=True, type=Path)
    parser.add_argument("--output-checkpoint", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument(
        "--source-closure-receipt",
        required=True,
        type=Path,
        help="Fresh sealed r260 source closure; r259 runtime trees are rejected.",
    )
    parser.add_argument(
        "--owner-contract",
        type=Path,
        default=Path("state/alakazam-new-list-direct-policy-r241.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = load_r260_owner_contract(args.owner_contract)
    result = materialize_r260_own_deck_successor(
        parent_checkpoint=args.parent_checkpoint,
        output_checkpoint=args.output_checkpoint,
        receipt_path=args.receipt,
        source_closure=args.source_closure_receipt,
        owner_contract=contract,
    )
    print(
        "r260 OwnDeckLedger successor materialized "
        f"checkpoint={result.checkpoint.path} sha256={result.checkpoint.sha256} "
        f"receipt={result.receipt_path} receipt_sha256={result.receipt_sha256}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
