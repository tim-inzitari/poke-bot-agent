#!/usr/bin/env python3
"""Build and verify the create-only disk-backed r274 sidecar index on Inzi."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from poke_bot.r241_own_deck_successor import (
    load_r260_owner_contract,
    validate_r260_inzi_dataset_binding,
    validate_r260_sidecar_binding,
)
from poke_bot.r260_inzi_sidecar_index import R260InziSidecarIndex
from poke_bot.own_deck_rollout_store import read_daily_meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar-binding", required=True, type=Path)
    parser.add_argument("--dataset-binding", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    owner = load_r260_owner_contract()
    sidecar = validate_r260_sidecar_binding(
        args.sidecar_binding, owner_contract=owner, verify_daily_receipt_files=True
    )
    dataset = validate_r260_inzi_dataset_binding(
        args.dataset_binding,
        sidecar_binding=sidecar,
        owner_contract=owner,
        require_local_dataset=True,
    )
    daily_receipt_files = {
        str(day): str(row["sha256"])
        for day, row in sorted(sidecar["daily_sidecar_meta_receipts"].items())
    }
    # The binding stores the SHA-256 of each immutable ``meta.json`` file.
    # The sidecar reader's identity is the self-checksum embedded inside that
    # already file-verified JSON.  Keep those two identities distinct.
    daily = {
        day: str(read_daily_meta(dataset["inzi_sidecar_root"], day)["meta_sha256"])
        for day in sorted(daily_receipt_files)
    }
    index = R260InziSidecarIndex.build(
        sidecar_root=dataset["inzi_sidecar_root"],
        output=args.output,
        source_manifest_sha256=owner.source_manifest_sha256,
        daily_meta_sha256s=daily,
    )
    index.assert_verified(
        expected_source_manifest_sha256=owner.source_manifest_sha256,
        daily_meta_sha256s=daily,
    )
    from poke_bot.r260_prestart_canary import file_identity

    print(
        json.dumps(
            {
                "schema": "poke_bot.r260_inzi_sidecar_index/v1",
                "identity": file_identity(index.path, immutable=True),
                "source_manifest_sha256": owner.source_manifest_sha256,
                "daily_meta_sha256s": daily,
                "daily_meta_file_sha256s": daily_receipt_files,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
