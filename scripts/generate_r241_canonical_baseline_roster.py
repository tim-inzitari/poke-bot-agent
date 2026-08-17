#!/usr/bin/env python3
"""Create the immutable r241 canonical 41-row baseline-roster receipt.

The four inputs must be the checksum-exact r175/r192 evidence roots.  This
tool has no service, selector, baseline-import, or transfer behavior; it only
derives a create-only JSON receipt at the path explicitly supplied by an
operator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import r241_baseline_payload_snapshot as baseline_payload  # noqa: E402
from poke_bot import r241_canonical_baseline_roster as canonical_roster  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--r175-manifest",
        type=Path,
        required=True,
        help=(
            "immutable r175 run manifest; must hash to "
            f"{canonical_roster.R175_RUN_MANIFEST_SHA256}"
        ),
    )
    parser.add_argument(
        "--r192-h10-contract",
        type=Path,
        required=True,
        help=(
            "immutable r192 H10 contract; must hash to "
            f"{canonical_roster.R192_H10_CONTRACT_SHA256}"
        ),
    )
    parser.add_argument(
        "--r175-iter20-plan",
        type=Path,
        required=True,
        help=(
            "immutable r175 iter20 strong-public plan; must hash to "
            f"{canonical_roster.R175_ITER20_PLAN_SHA256}"
        ),
    )
    parser.add_argument(
        "--r175-iter20-receipt",
        type=Path,
        required=True,
        help=(
            "immutable r175 iter20 collection receipt; must hash to "
            f"{canonical_roster.R175_ITER20_RECEIPT_SHA256}"
        ),
    )
    parser.add_argument(
        "--owner-contract-sha256",
        required=True,
        help="explicit SHA-256 of state/alakazam-new-list-direct-policy-r241.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new receipt path; create-only and byte-identical reruns are accepted",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        receipt = canonical_roster.build_receipt(
            r175_manifest=args.r175_manifest,
            r192_h10_contract=args.r192_h10_contract,
            r175_iter20_plan=args.r175_iter20_plan,
            r175_iter20_receipt=args.r175_iter20_receipt,
            owner_contract_sha256=args.owner_contract_sha256,
        )
        output = canonical_roster.write_receipt(args.output, receipt)
    except baseline_payload.R241BaselinePayloadError as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "receipt": str(output),
                "sha256": baseline_payload.sha256_file(output),
                "schema": receipt["schema"],
                "baseline_roster_rows": receipt["counts"]["canonical_baseline_union_rows"],
                "baseline_manifest_sha256": receipt["baseline_manifest_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
