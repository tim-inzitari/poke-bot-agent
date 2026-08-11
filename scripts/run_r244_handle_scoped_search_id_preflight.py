#!/usr/bin/env python3
"""Write an immutable offline R244 handle-scoped SearchId regression receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.r244_handle_scoped_search_id_preflight import (  # noqa: E402
    R244HandleScopedSearchIdFailure,
    R244HandleScopedSearchIdInputs,
    run_r244_handle_scoped_search_id_preflight,
)
from poke_bot.r235_kaggle_phase1_preflight import ImmutableReceiptError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--witness-json", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument(
        "--r225-contract",
        type=Path,
        default=ROOT / "state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json",
    )
    parser.add_argument("--r236-contract", type=Path, default=ROOT / "state/canonical-libcg-r236.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = run_r244_handle_scoped_search_id_preflight(
            inputs=R244HandleScopedSearchIdInputs(
                witness_path=args.witness_json,
                receipt_path=args.receipt,
                r225_contract_path=args.r225_contract,
                r236_contract_path=args.r236_contract,
            )
        )
    except R244HandleScopedSearchIdFailure as exc:
        print(json.dumps({"status": "failed", "receipt": str(exc.path), "error": str(exc)}), file=sys.stderr)
        return 2
    except ImmutableReceiptError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
