#!/usr/bin/env python3
"""Materialize the research-only 768-game Slowking learning metadata index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.slowking_archetype_learning import (
    build_game_rows,
    load_aggregate,
    load_contract,
    summarize_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("config/slowking_archetype_learning.v1.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("state/slowking_archetype_learning_index_v1.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = ROOT
    contract_path = args.contract if args.contract.is_absolute() else root / args.contract
    output_path = args.out if args.out.is_absolute() else root / args.out
    contract = load_contract(contract_path)
    aggregate = load_aggregate(contract, root=root)
    rows = build_game_rows(contract, aggregate)
    output = {
        "schema": "poke_bot.slowking_archetype_learning_index/v1",
        "status": "research_only_no_training_or_runtime_authority",
        "contract": str(contract_path.relative_to(root)),
        "summary": summarize_rows(rows),
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output_path), **output["summary"]}, indent=2))


if __name__ == "__main__":
    main()
