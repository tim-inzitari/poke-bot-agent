#!/usr/bin/env python3
"""Build the dormant r259 OwnDeckLedger expert-rollout side store.

This command deliberately has no collector/index/raw-dir mode.  Production
uses ``--archive-native`` to read only the checksum-bound ZIPs named by the
sealed r241 receipt.  ``--protected-records`` is a parity-only, already
protected JSONL input route; both modes re-hash every manifest-listed ZIP
before writing any immutable r259 sidecar.

The two in-memory deployment preflights are intentionally isolated:
``--archive-native-smoke`` checks only one normal archive-native Alakazam
record, while ``--archive-native-unknown-outcome-smoke`` checks only the
pinned ``87394115.json`` malformed-reward fallback.  Neither accepts an
output mount or publishes a sidecar.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

# This script is executed by absolute path in a read-only container.  Put the
# sealed snapshot root ahead of site packages rather than relying on image
# entrypoint side effects (which r259 explicitly overrides).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.own_deck_rollout_store import (
    EXPECTED_IMAGE_ID,
    EXPECTED_IMAGE_TAG,
    EXPECTED_SOURCE_MANIFEST_SHA256,
    EXPECTED_VERSIONED_RECEIPT_SHA256,
    build_archive_native_sidecar,
    build_protected_jsonl_sidecar,
    smoke_archive_native_one_record,
    smoke_archive_native_unknown_outcome_member,
)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument(
        "--original-manifest-path",
        required=True,
        help="Protected host path recorded in receipts; it is never opened in-container.",
    )
    parser.add_argument(
        "--versioned-receipt-lock",
        type=Path,
        required=True,
        help="Root-verified readable projection of current.json's versioned receipt.",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--source-snapshot")
    parser.add_argument("--source-snapshot-tree-sha256")
    parser.add_argument(
        "--expected-manifest-sha256",
        default=EXPECTED_SOURCE_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--expected-versioned-receipt-sha256",
        default=EXPECTED_VERSIONED_RECEIPT_SHA256,
    )
    parser.add_argument("--image-tag", default=EXPECTED_IMAGE_TAG)
    parser.add_argument("--image-id", default=EXPECTED_IMAGE_ID)
    parser.add_argument(
        "--only-day",
        action="append",
        default=None,
        help="Optional exact r241 day; repeat only for an explicitly bounded retry.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--archive-native", action="store_true")
    mode.add_argument("--archive-native-smoke", action="store_true")
    mode.add_argument(
        "--archive-native-unknown-outcome-smoke",
        "--archive-native-87394115-smoke",
        dest="archive_native_unknown_outcome_smoke",
        action="store_true",
        help="Run only the exact 87394115.json malformed-reward smoke.",
    )
    mode.add_argument(
        "--protected-records",
        type=Path,
        help="Protected JSONL(.gz) parity stream; never a generic collector cache.",
    )
    parser.add_argument("--classifier-mix", type=Path)
    parser.add_argument("--classifier-representatives", type=Path)
    parser.add_argument("--card-csv", type=Path)
    args = parser.parse_args(argv)
    is_smoke = args.archive_native_smoke or args.archive_native_unknown_outcome_smoke
    if not is_smoke and args.output_root is None:
        parser.error("--output-root is required when publishing a sidecar")
    if not is_smoke and (
        args.source_snapshot is None or args.source_snapshot_tree_sha256 is None
    ):
        parser.error(
            "--source-snapshot and --source-snapshot-tree-sha256 are required "
            "when publishing a sidecar"
        )
    if args.archive_native or is_smoke:
        missing = [
            name
            for name in ("classifier_mix", "classifier_representatives", "card_csv")
            if getattr(args, name) is None
        ]
        if missing:
            parser.error(
                "--archive-native or either archive-native smoke requires "
                + ", ".join("--" + name.replace("_", "-") for name in missing)
            )
    elif any(
        getattr(args, name) is not None
        for name in ("classifier_mix", "classifier_representatives", "card_csv")
    ):
        parser.error("classifier inputs are only valid with archive-native modes")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    if args.archive_native_smoke:
        result = smoke_archive_native_one_record(
            source_manifest=args.source_manifest,
            classifier_mix=args.classifier_mix,
            classifier_representatives=args.classifier_representatives,
            card_csv=args.card_csv,
            original_manifest_path=args.original_manifest_path,
            versioned_receipt_lock=args.versioned_receipt_lock,
            expected_manifest_sha256=args.expected_manifest_sha256,
            expected_versioned_receipt_sha256=args.expected_versioned_receipt_sha256,
        )
        print(
            json.dumps(
                {
                    "schema": "poke_bot.own_deck_rollout_update_result/v1",
                    "mode": "archive_native_smoke",
                    "smoke": result,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.archive_native_unknown_outcome_smoke:
        result = smoke_archive_native_unknown_outcome_member(
            source_manifest=args.source_manifest,
            classifier_mix=args.classifier_mix,
            classifier_representatives=args.classifier_representatives,
            card_csv=args.card_csv,
            original_manifest_path=args.original_manifest_path,
            versioned_receipt_lock=args.versioned_receipt_lock,
            expected_manifest_sha256=args.expected_manifest_sha256,
            expected_versioned_receipt_sha256=args.expected_versioned_receipt_sha256,
        )
        print(
            json.dumps(
                {
                    "schema": "poke_bot.own_deck_rollout_update_result/v1",
                    "mode": "archive_native_unknown_outcome_smoke",
                    "smoke": result,
                },
                sort_keys=True,
            )
        )
        return 0
    common = {
        "source_manifest": args.source_manifest,
        "output_root": args.output_root,
        "source_snapshot_path": args.source_snapshot,
        "source_snapshot_tree_sha256": args.source_snapshot_tree_sha256,
        "image_tag": args.image_tag,
        "image_id": args.image_id,
        "original_manifest_path": args.original_manifest_path,
        "versioned_receipt_lock": args.versioned_receipt_lock,
        "expected_manifest_sha256": args.expected_manifest_sha256,
        "expected_versioned_receipt_sha256": args.expected_versioned_receipt_sha256,
        "only_days": args.only_day,
    }
    if args.archive_native:
        results = build_archive_native_sidecar(
            **common,
            classifier_mix=args.classifier_mix,
            classifier_representatives=args.classifier_representatives,
            card_csv=args.card_csv,
        )
    else:
        results = build_protected_jsonl_sidecar(
            **common,
            protected_records=args.protected_records,
        )
    print(
        json.dumps(
            {
                "schema": "poke_bot.own_deck_rollout_update_result/v1",
                "mode": "archive_native" if args.archive_native else "protected_jsonl",
                "days": [
                    {
                        "day": result.day,
                        "directory": str(result.directory),
                        "shard": str(result.shard_path),
                        "meta": str(result.meta_path),
                        "shard_sha256": result.shard_sha256,
                        "meta_sha256": result.meta_sha256,
                        "rows": result.rows,
                        "source_records": result.source_records,
                        "skipped_existing": result.skipped_existing,
                    }
                    for result in results
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
