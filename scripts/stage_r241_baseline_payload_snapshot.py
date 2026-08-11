#!/usr/bin/env python3
"""Publish a create-only, immutable r241 baseline payload snapshot.

This tool copies a separately supplied baseline library into a content-addressed
read-only mount.  It is deliberately independent of the immutable code
snapshot and never alters the r241 runtime registry, starts a worker, or loads
baseline Python.  The emitted receipt is the input to the later external
activation overlay.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import r241_baseline_payload_snapshot as payload  # noqa: E402


def _create_only_json(path: Path, value: Mapping[str, object]) -> None:
    """Write an external receipt once; existing differing bytes fail closed."""

    raw = path.expanduser()
    if raw.is_symlink() or raw.parent.is_symlink() or not raw.parent.is_dir():
        raise payload.R241BaselinePayloadError(
            f"baseline staging receipt parent must be a real directory: {raw.parent}"
        )
    target = raw.parent.resolve() / raw.name
    encoded = payload.canonical_json(value)
    if target.exists():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != encoded:
            raise payload.R241BaselinePayloadError(
                f"baseline staging receipt already exists with different bytes: {target}"
            )
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.is_symlink() or not target.is_file() or target.read_bytes() != encoded:
                raise payload.R241BaselinePayloadError(
                    f"baseline staging receipt already exists with different bytes: {target}"
                )
    finally:
        temporary.unlink(missing_ok=True)


def _stage(
    *,
    source_root: Path,
    output_base: Path,
    outputs_root: Path,
    receipt_output: Path,
    host: str,
    owner_contract_sha256: str,
    canonical_roster_receipt: Path,
    canonical_roster_receipt_sha256: str,
) -> dict[str, object]:
    source = payload.regular_directory(source_root, label="baseline source root")
    base = payload.regular_directory(output_base, label="baseline snapshot output base")
    outputs = payload.regular_directory(outputs_root, label="external r241 outputs root")
    receipt_parent = payload.regular_directory(
        receipt_output.expanduser().parent, label="baseline staging receipt parent"
    )
    receipt = receipt_parent / receipt_output.name
    if outputs not in receipt.parents:
        raise payload.R241BaselinePayloadError(
            "baseline staging receipt must remain under external outputs"
        )
    if source == base or source in base.parents or base in source.parents:
        raise payload.R241BaselinePayloadError(
            "baseline snapshot output base must be separate from its source library"
        )
    if base == outputs or base in outputs.parents or outputs in base.parents:
        raise payload.R241BaselinePayloadError(
            "baseline snapshot output base must be separate from external outputs"
        )

    canonical_receipt_path, canonical_receipt = payload.validate_canonical_roster_receipt(
        canonical_roster_receipt,
        expected_sha256=canonical_roster_receipt_sha256,
        owner_contract_sha256=owner_contract_sha256,
    )
    canonical_roster = payload.normalized_roster(
        list(canonical_receipt.get("baseline_roster") or [])
    )
    minimal_manifest = payload.minimal_manifest_bytes(canonical_roster)
    canonical_manifest_sha256 = payload.sha256_bytes(minimal_manifest)
    if canonical_manifest_sha256 != canonical_receipt["baseline_manifest_sha256"]:
        raise payload.R241BaselinePayloadError(
            "canonical roster receipt does not bind its synthetic minimal manifest"
        )
    rows = payload.inventory_exact_roster(
        source,
        canonical_roster,
        generated_manifest=minimal_manifest,
    )
    manifest_payload = payload.manifest_payload(
        owner_contract_sha256=owner_contract_sha256,
        rows=rows,
        baseline_manifest_sha256=canonical_manifest_sha256,
        baseline_roster=canonical_roster,
    )
    manifest_bytes = payload.canonical_json(manifest_payload)
    manifest_sha256 = payload.sha256_bytes(manifest_bytes)
    snapshot_root = base / (
        payload.BASELINE_PAYLOAD_ROOT_PREFIX
        + manifest_sha256.removeprefix("sha256:")[:16]
    )
    if snapshot_root.exists() or snapshot_root.is_symlink():
        identity = payload.validate_snapshot(
            root=snapshot_root,
            manifest_path=snapshot_root / payload.BASELINE_PAYLOAD_MANIFEST_FILENAME,
            manifest_sha256=manifest_sha256,
            baseline_tree_sha256=str(manifest_payload["baseline_tree_sha256"]),
            owner_contract_sha256=owner_contract_sha256,
        )
    else:
        # ``mkdir`` is exclusive.  Never use rename here: on macOS a rename
        # can replace a concurrently-created empty directory.  A partial root
        # has no receipt and is intentionally left for explicit operator
        # inspection instead of an automated overwrite.
        try:
            os.mkdir(snapshot_root, 0o700)
        except FileExistsError:
            identity = payload.validate_snapshot(
                root=snapshot_root,
                manifest_path=snapshot_root / payload.BASELINE_PAYLOAD_MANIFEST_FILENAME,
                manifest_sha256=manifest_sha256,
                baseline_tree_sha256=str(manifest_payload["baseline_tree_sha256"]),
                owner_contract_sha256=owner_contract_sha256,
            )
        else:
            payload.sealed_copy(
                source_root=source,
                destination_root=snapshot_root,
                rows=rows,
                manifest_bytes=manifest_bytes,
                generated_files={"manifest.json": minimal_manifest},
            )
            identity = payload.validate_snapshot(
                root=snapshot_root,
                manifest_path=snapshot_root / payload.BASELINE_PAYLOAD_MANIFEST_FILENAME,
                manifest_sha256=manifest_sha256,
                baseline_tree_sha256=str(manifest_payload["baseline_tree_sha256"]),
                owner_contract_sha256=owner_contract_sha256,
            )
    snapshot = {
        "schema": payload.BASELINE_PAYLOAD_SNAPSHOT_SCHEMA,
        "revision": payload.REVISION,
        "candidate_id": payload.CANDIDATE_ID,
        "status": "authenticated_immutable_baseline_payload_snapshot",
        "authenticated": True,
        "host": host,
        "root": identity["root"],
        "manifest": identity["manifest"],
        "manifest_sha256": identity["manifest_sha256"],
        "baseline_tree_sha256": identity["baseline_tree_sha256"],
        "file_inventory_sha256": identity["file_inventory_sha256"],
        "baseline_manifest_sha256": identity["baseline_manifest_sha256"],
        "baseline_roster_sha256": identity["baseline_roster_sha256"],
        "baseline_roster": identity["baseline_roster"],
        "canonical_roster_receipt": str(canonical_receipt_path),
        "canonical_roster_receipt_sha256": payload.sha256_file(canonical_receipt_path),
        "canonical_public_contract_sha256s": dict(
            canonical_receipt["public_contract_sha256s"]
        ),
        "owner_contract_sha256": owner_contract_sha256,
    }
    result = {
        "schema": payload.BASELINE_PAYLOAD_STAGING_SCHEMA,
        "revision": payload.REVISION,
        "candidate_id": payload.CANDIDATE_ID,
        "status": "passed",
        "passed": True,
        "operation": "deterministic_stage_or_verify",
        "baseline_payload_snapshot": snapshot,
        "canonical_roster_receipt": {
            "path": str(canonical_receipt_path),
            "sha256": payload.sha256_file(canonical_receipt_path),
            "baseline_manifest_sha256": canonical_manifest_sha256,
            "baseline_roster_sha256": payload.roster_digest(canonical_roster),
            "public_contract_sha256s": dict(
                canonical_receipt["public_contract_sha256s"]
            ),
        },
        "receipt_outside_source_and_baseline_snapshot": True,
    }
    _create_only_json(receipt, result)
    return {
        "receipt_path": str(receipt.resolve()),
        "receipt_sha256": payload.sha256_file(receipt.resolve()),
        **result,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-base", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--owner-contract-sha256", required=True)
    parser.add_argument(
        "--canonical-roster-receipt",
        type=Path,
        required=True,
        help=(
            "externally derived r195 canonical full-baseline roster receipt; "
            "the local manifest is never an authority"
        ),
    )
    parser.add_argument(
        "--canonical-roster-receipt-sha256",
        required=True,
        help="checksum of the externally derived canonical-roster receipt",
    )
    parser.add_argument("--host", choices=("inzi", "elmo"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = _stage(
        source_root=args.source_root,
        output_base=args.output_base,
        outputs_root=args.outputs_root,
        receipt_output=args.receipt_output,
        owner_contract_sha256=args.owner_contract_sha256,
        canonical_roster_receipt=args.canonical_roster_receipt,
        canonical_roster_receipt_sha256=args.canonical_roster_receipt_sha256,
        host=args.host,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except payload.R241BaselinePayloadError as exc:
        print(f"r241 baseline payload staging failed: {exc}", file=sys.stderr)
        raise SystemExit(78)
