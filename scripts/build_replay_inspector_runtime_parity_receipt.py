#!/usr/bin/env python3
"""Create a checksum-bound runtime-parity receipt for one exact submission.

Run this only after an operator independently verified that the extracted
runtime is the package uploaded for the submission.  The receipt contains no
trusted path: the inspector hashes the configured extracted tree again before
each dynamic trace.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from replay_inspector.provenance import sha256_file, sha256_source_tree

SCHEMA = "poke_bot.replay_model_inspector_runtime_parity_receipt/v1"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-id", required=True, type=int)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--runtime-package", required=True, type=Path)
    parser.add_argument("--runtime-source-root", required=True, type=Path)
    parser.add_argument(
        "--verified-by",
        required=True,
        help="Named independent verifier responsible for the parity check.",
    )
    parser.add_argument(
        "--verified-at-utc",
        default=None,
        help="Optional ISO-8601 timestamp; defaults to the current UTC time.",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if args.submission_id <= 0:
        raise SystemExit("--submission-id must be positive")
    if not args.verified_by.strip():
        raise SystemExit("--verified-by must not be empty")
    verified_at = args.verified_at_utc or datetime.now(timezone.utc).replace(
        microsecond=0
    ).isoformat().replace("+00:00", "Z")
    payload = {
        "schema": SCHEMA,
        "version": 1,
        "status": "verified",
        "submission_id": args.submission_id,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "bundle_sha256": sha256_file(args.bundle),
        "runtime_package_sha256": sha256_file(args.runtime_package),
        "runtime_source_tree_sha256": sha256_source_tree(args.runtime_source_root),
        "verification": {
            "method": "independent_exact_runtime_parity",
            "verified_by": args.verified_by.strip(),
            "verified_at_utc": verified_at,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
