#!/usr/bin/env python3
"""Prepare or compile the fail-closed RTP three-arm evaluation receipt.

The input spec is intentionally artifact-first: it must name checksum-bound
parent/deck/matchup/runtime artifacts, one true RNG tape or snapshot per
opponent/seat cell, a true-RNG capability receipt, and the three exact runtime
profiles.  r198 requires bridge/recursive ``pure_rl_r197`` profiles with
exactly 256 neural passes and a 1,024 action-combination cap. Requested seeds
are not accepted as pairing evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.rtp_three_arm_evaluation import (  # noqa: E402
    RTPThreeArmEvaluationError,
    compile_three_arm_receipt,
    prepare_three_arm_manifest_from_spec,
)


def _object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RTPThreeArmEvaluationError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise RTPThreeArmEvaluationError(f"{label} must be a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="write an immutable schedule manifest")
    prepare.add_argument("--spec", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)

    compile_parser = subparsers.add_parser(
        "compile", help="validate results and write a hold/review receipt"
    )
    compile_parser.add_argument("--manifest", type=Path, required=True)
    compile_parser.add_argument("--results", type=Path, required=True)
    compile_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    try:
        if args.command == "prepare":
            output = prepare_three_arm_manifest_from_spec(
                _object(args.spec, "evaluation spec"),
                output_path=args.output,
            )
        else:
            output = compile_three_arm_receipt(
                manifest_path=args.manifest,
                results=args.results,
                output_path=args.output,
            )
    except RTPThreeArmEvaluationError as exc:
        parser.error(str(exc))
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
