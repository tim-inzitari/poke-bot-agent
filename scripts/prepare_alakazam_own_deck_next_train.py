#!/usr/bin/env python3
"""Prepare, but never launch, the receipt-bound r258/r259 next-train plan.

This command only reads immutable inputs and atomically writes one plan JSON.
It intentionally has no trainer, systemd, Docker, selector, package, or Kaggle
integration.  A successful invocation means the future canary *configuration*
is ready for its separately authorized workflow; it does not start that work.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import own_deck_successor as successor
from poke_bot import own_deck_training_contract as contract


def _stage_receipts(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    valid = {stage.value for stage in successor.OwnDeckSuccessorStage}
    for raw in values:
        stage, separator, path = raw.partition("=")
        if not separator or not stage or not path:
            raise argparse.ArgumentTypeError(
                "--stage-receipt must use STAGE=PATH"
            )
        if stage not in valid:
            raise argparse.ArgumentTypeError(
                f"unknown r258 stage {stage!r}; expected one of {sorted(valid)}"
            )
        if stage in result:
            raise argparse.ArgumentTypeError(
                f"duplicate stage receipt for {stage}"
            )
        result[stage] = Path(path)
    return result


def _daily_meta_paths(values: list[str], root: str | None) -> list[Path]:
    paths = [Path(value) for value in values]
    if root:
        daily_root = Path(root)
        paths.extend(sorted(daily_root.glob("daily/*/meta.json")))
    # De-duplicate exact textual paths before the contract detects duplicate
    # source days.  This lets an operator provide a root plus one explicit path
    # without generating a noisy duplicate argument error.
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        normalized = path.expanduser()
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    if not result:
        raise argparse.ArgumentTypeError(
            "provide --daily-meta and/or --daily-root containing daily/*/meta.json"
        )
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Validate r258/r259 receipts and atomically prepare a dormant "
            "Alakazam OwnDeckLedger next-train plan. It never starts training."
        )
    )
    result.add_argument("--refresh-completion-receipt", required=True)
    result.add_argument(
        "--stage-receipt",
        action="append",
        default=[],
        metavar="STAGE=PATH",
        help="One receipt for each r258 pre-refresh stage.",
    )
    result.add_argument("--migration-receipt", required=True)
    result.add_argument("--daily-meta", action="append", default=[])
    result.add_argument(
        "--daily-root",
        help="Optional r259 output root; discovers daily/*/meta.json beneath it.",
    )
    result.add_argument("--join-receipt", required=True)
    result.add_argument("--parity-receipt", required=True)
    result.add_argument("--metric-support-receipt", required=True)
    result.add_argument(
        "--source-manifest",
        required=True,
        help="The exact r259 protected source manifest, read-only.",
    )
    result.add_argument(
        "--model-checkpoint",
        required=True,
        help="Isolated-migration child checkpoint whose digest matches its receipt.",
    )
    result.add_argument(
        "--code-path",
        required=True,
        help="Immutable staged source file/tree to content-hash into the plan.",
    )
    result.add_argument("--output", required=True)
    result.add_argument(
        "--plan-id", default="alakazam-own-deck-next-train-r258-r259"
    )
    result.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print canonical plan JSON without creating an output file.",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        stages = _stage_receipts(args.stage_receipt)
        daily = _daily_meta_paths(args.daily_meta, args.daily_root)
        plan = contract.prepare_next_train_plan(
            refresh_completion_receipt=Path(args.refresh_completion_receipt),
            stage_receipts=stages,
            migration_receipt=Path(args.migration_receipt),
            daily_meta_receipts=daily,
            join_receipt=Path(args.join_receipt),
            parity_receipt=Path(args.parity_receipt),
            metric_support_receipt=Path(args.metric_support_receipt),
            source_manifest_identity=Path(args.source_manifest),
            model_identity=Path(args.model_checkpoint),
            code_identity=Path(args.code_path),
            plan_id=args.plan_id,
        )
        if args.dry_run:
            sys.stdout.buffer.write(contract.canonical_json_bytes(plan))
            return 0
        output = contract.write_next_train_plan(args.output, plan)
    except (OSError, ValueError, contract.OwnDeckNextTrainContractError) as exc:
        print(f"next-train plan refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"prepared": str(output), "plan_sha256": plan["plan_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
