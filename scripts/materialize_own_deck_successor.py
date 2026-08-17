"""Materialize the dormant r258 OwnDeckLedger successor checkpoint.

This CLI creates only an immutable child checkpoint and its detached receipt.
It does not touch a selector, service, training run, package, or submission.
The canonical migration gate rejects invocation until the terminal refresh and
every prior r258 stage receipt are present and valid.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.own_deck_migration import (
    OwnDeckMigrationError,
    materialize_own_deck_successor,
)
from poke_bot.own_deck_successor import OwnDeckSuccessorError, OwnDeckSuccessorStage


def _stage_receipt_argument(value: str) -> tuple[OwnDeckSuccessorStage, Path]:
    """Parse one exact ``stage=receipt.json`` argument without aliases."""

    if "=" not in value:
        raise argparse.ArgumentTypeError("stage receipt must be STAGE=PATH")
    raw_stage, raw_path = value.split("=", 1)
    try:
        stage = OwnDeckSuccessorStage(raw_stage.strip())
    except ValueError as exc:
        allowed = ", ".join(stage.value for stage in OwnDeckSuccessorStage)
        raise argparse.ArgumentTypeError(
            f"stage must be one of: {allowed}"
        ) from exc
    path = Path(raw_path).expanduser()
    if not str(path):
        raise argparse.ArgumentTypeError("stage receipt path is empty")
    return stage, path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--refresh-completion-receipt", type=Path, required=True)
    parser.add_argument(
        "--stage-receipt",
        action="append",
        type=_stage_receipt_argument,
        default=[],
        metavar="STAGE=PATH",
        help="repeat once for every pre-refresh r258 stage receipt",
    )
    parser.add_argument("--ledger-width", type=int, default=128)
    args = parser.parse_args(argv)
    if args.ledger_width <= 0:
        parser.error("--ledger-width must be positive")
    stage_receipts: dict[OwnDeckSuccessorStage, Path] = {}
    for stage, path in args.stage_receipt:
        if stage in stage_receipts:
            parser.error(f"duplicate --stage-receipt for {stage.value}")
        stage_receipts[stage] = path
    try:
        result = materialize_own_deck_successor(
            parent_checkpoint=args.parent,
            expected_parent_sha256=args.parent_sha256,
            output_checkpoint=args.output,
            receipt_path=args.receipt,
            refresh_completion_receipt=args.refresh_completion_receipt,
            stage_receipts=stage_receipts,
            ledger_width=args.ledger_width,
        )
    except (OwnDeckMigrationError, OwnDeckSuccessorError, FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result.as_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
