#!/usr/bin/env python3
"""Create-only CLI for the rev5/r303 rule-derivative freeze boundary.

The command only writes checksum-addressable evidence files.  It cannot train,
start a service, stage data to Inzi, mutate a checkpoint, or enable a policy
route.  `migration-proposal` deliberately produces a proposal that still needs
the checklist owner's config pin; `freeze` refuses to run until that pin is
already present.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

# Allow direct ``python scripts/...`` invocation from a clean shell; this does
# not change package installation or runtime imports.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from poke_bot.alakazam_rule_derivative_freeze_r303 import (
    R303FreezeError,
    build_r303_frozen_schema_manifest,
    build_r303_materialization_preflight,
    build_r303_schema_freeze_bundle,
    build_r303_schema_freeze_receipt,
    build_r303_sealed_catalog_binding,
    build_revision_5_consumer_migration_payload,
    make_r303_zero_bypass_receipt,
    write_r303_frozen_schema_manifest_create_only,
    write_r303_materialization_preflight_create_only,
    write_r303_schema_freeze_bundle_create_only,
    write_r303_schema_freeze_receipt_create_only,
    write_r303_zero_bypass_receipt_create_only,
    write_revision_5_consumer_migration_create_only,
)


def _read_json_value(path: str, *, label: str) -> Any:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise R303FreezeError(f"{label} must be a regular JSON file")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R303FreezeError(f"cannot read {label}") from exc
    return payload


def _read_json(path: str, *, label: str) -> Mapping[str, Any]:
    payload = _read_json_value(path, label=label)
    if not isinstance(payload, Mapping):
        raise R303FreezeError(f"{label} must contain a JSON object")
    return payload


def _read_bytes(path: str, *, label: str) -> bytes:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise R303FreezeError(f"{label} must be a regular file")
    try:
        return source.read_bytes()
    except OSError as exc:
        raise R303FreezeError(f"cannot read {label}") from exc


def _emit(result: Mapping[str, Any]) -> None:
    print(json.dumps(dict(result), sort_keys=True, separators=(",", ":")))


def _migration_proposal(args: argparse.Namespace) -> int:
    assertions = _read_json(args.assertions_json, label="migration assertion evidence")
    payload = build_revision_5_consumer_migration_payload(
        consumer_rebind_assertions=assertions,
        validated_at_utc=args.validated_at_utc,
    )
    digest = write_revision_5_consumer_migration_create_only(args.output, payload)
    _emit(
        {
            "status": "proposal_written_requires_checklist_owner_anchor_pin",
            "path": str(Path(args.output).resolve()),
            "file_sha256": digest,
            "payload_sha256": digest,
            "schema": payload["schema"],
        }
    )
    return 0


def _freeze(args: argparse.Namespace) -> int:
    catalog = build_r303_sealed_catalog_binding(
        catalog_sealer_receipt_path=args.catalog_sealer_receipt,
        revision_5_consumer_migration_receipt_path=args.migration_receipt,
    )
    manifest = build_r303_frozen_schema_manifest(
        catalog_binding=catalog,
        revision_5_consumer_migration_receipt_path=args.migration_receipt,
    )
    manifest_digest = write_r303_frozen_schema_manifest_create_only(
        args.schema_output,
        manifest,
        catalog_sealer_receipt_path=args.catalog_sealer_receipt,
        revision_5_consumer_migration_receipt_path=args.migration_receipt,
    )
    baseline_checkpoint = _read_json(
        args.baseline_checkpoint_identity, label="baseline checkpoint identity"
    )
    baseline_choice = _read_json_value(args.baseline_legal_choice, label="baseline legal choice")
    layer_off_choice = _read_json_value(args.layer_off_legal_choice, label="layer-off legal choice")
    evidence = _read_json(args.evidence_provenance, label="layer-off evidence provenance")
    bypass = make_r303_zero_bypass_receipt(
        schema_manifest=manifest,
        baseline_checkpoint_identity=baseline_checkpoint,
        baseline_logits=_read_bytes(args.baseline_logits_bytes, label="baseline logits bytes"),
        layer_off_logits=_read_bytes(args.layer_off_logits_bytes, label="layer-off logits bytes"),
        baseline_legal_choice=baseline_choice,
        layer_off_legal_choice=layer_off_choice,
        evidence_provenance=evidence,
    )
    bypass_digest = write_r303_zero_bypass_receipt_create_only(
        args.zero_bypass_output, manifest, bypass
    )
    schema_receipt = build_r303_schema_freeze_receipt(
        schema_manifest=manifest,
        zero_bypass_receipt=bypass,
        frozen_at_utc=args.frozen_at_utc,
    )
    schema_receipt_digest = write_r303_schema_freeze_receipt_create_only(
        args.schema_freeze_receipt_output,
        schema_manifest=manifest,
        zero_bypass_receipt=bypass,
        schema_freeze_receipt=schema_receipt,
    )
    bundle = build_r303_schema_freeze_bundle(
        schema_manifest=manifest,
        zero_bypass_receipt=bypass,
        schema_freeze_receipt=schema_receipt,
    )
    bundle_digest = write_r303_schema_freeze_bundle_create_only(args.bundle_output, bundle)
    _emit(
        {
            "status": "frozen_zero_inert_ready_for_elmo_refeaturization_only",
            "schema_manifest_sha256": manifest_digest,
            "zero_bypass_receipt_sha256": bypass_digest,
            "schema_freeze_receipt_sha256": schema_receipt_digest,
            "bundle_sha256": bundle_digest,
            "training_allowed": False,
            "inzi_authority": False,
        }
    )
    return 0


def _preflight(args: argparse.Namespace) -> int:
    preflight = build_r303_materialization_preflight(
        raw_manifest=_read_json(args.raw_manifest, label="raw 30-day manifest"),
        raw_corpus_receipt=_read_json(args.raw_receipt, label="raw corpus receipt"),
        schema_manifest=_read_json(args.schema_manifest, label="frozen schema manifest"),
        zero_bypass_receipt=_read_json(args.zero_bypass_receipt, label="zero-bypass receipt"),
        schema_freeze_receipt=_read_json(args.schema_freeze_receipt, label="schema-freeze receipt"),
    )
    digest = write_r303_materialization_preflight_create_only(args.output, preflight)
    _emit(
        {
            "status": preflight["status"],
            "path": str(Path(args.output).resolve()),
            "sha256": digest,
            "training_allowed": False,
            "selected_action_target_mode": preflight["selected_action_target_mode"],
        }
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    migration = commands.add_parser(
        "migration-proposal", help="write a create-only checklist-owner pin proposal"
    )
    migration.add_argument("--assertions-json", required=True)
    migration.add_argument("--validated-at-utc", required=True)
    migration.add_argument("--output", required=True)
    migration.set_defaults(handler=_migration_proposal)

    freeze = commands.add_parser(
        "freeze", help="write rev5 schema, zero-bypass, contract receipt, and bundle"
    )
    freeze.add_argument("--catalog-sealer-receipt", required=True)
    freeze.add_argument("--migration-receipt", required=True)
    freeze.add_argument("--schema-output", required=True)
    freeze.add_argument("--zero-bypass-output", required=True)
    freeze.add_argument("--schema-freeze-receipt-output", required=True)
    freeze.add_argument("--bundle-output", required=True)
    freeze.add_argument("--baseline-checkpoint-identity", required=True)
    freeze.add_argument("--baseline-logits-bytes", required=True)
    freeze.add_argument("--layer-off-logits-bytes", required=True)
    freeze.add_argument("--baseline-legal-choice", required=True)
    freeze.add_argument("--layer-off-legal-choice", required=True)
    freeze.add_argument("--evidence-provenance", required=True)
    freeze.add_argument("--frozen-at-utc", required=True)
    freeze.set_defaults(handler=_freeze)

    preflight = commands.add_parser(
        "preflight", help="write only a diagnostic-target 30-day preflight"
    )
    preflight.add_argument("--raw-manifest", required=True)
    preflight.add_argument("--raw-receipt", required=True)
    preflight.add_argument("--schema-manifest", required=True)
    preflight.add_argument("--zero-bypass-receipt", required=True)
    preflight.add_argument("--schema-freeze-receipt", required=True)
    preflight.add_argument("--output", required=True)
    preflight.set_defaults(handler=_preflight)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except R303FreezeError as exc:
        print(f"freeze-r303: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
