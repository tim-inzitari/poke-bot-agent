#!/usr/bin/env python3
"""Operator entrypoint for the isolated r298 re-featurization evidence path.

There is deliberately no ``train``, policy activation, checkpoint publish,
Inzi runtime load, or service-control subcommand here.  The mutating commands
require ``--execute`` and only create versioned materialization/receipt files.
They are intended to run on Elmo after the canonical schema gate passes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poke_bot.alakazam_public_rule_adapter_r298 import load_sealed_public_catalog
from poke_bot.alakazam_rule_derivative_training_r298 import (
    ExperimentLease,
    ResourceTelemetry,
    RuleDerivativeTrainingError,
    bounded_day_parallel_plan,
    build_frozen_schema_manifest,
    initialize_refeature_root,
    inspect_and_build_frozen_tensor_audit_create_only,
    load_sealed_structured_catalog,
    materialize_refeature_day,
    merge_refeatured_days,
    write_frozen_schema_manifest_create_only,
    write_frozen_tensor_audit_create_only,
)
from poke_bot.alakazam_rule_derivative_transfer_r298 import (
    build_transfer_plan,
    write_transfer_plan_create_only,
)


def _json_object(path: str, *, label: str) -> Mapping[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise RuleDerivativeTrainingError(f"{label} must be a regular JSON file")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuleDerivativeTrainingError(f"cannot read {label}") from exc
    if not isinstance(value, Mapping):
        raise RuleDerivativeTrainingError(f"{label} must contain a JSON object")
    return value


def _print(value: Any) -> None:
    print(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True))


def _require_execute(args: argparse.Namespace) -> None:
    if args.execute is not True:
        raise RuleDerivativeTrainingError("this create-only operation requires explicit --execute")


def _materialization_inputs(args: argparse.Namespace) -> tuple[Mapping[str, Any], ...]:
    return (
        _json_object(args.raw_manifest, label="raw manifest"),
        _json_object(args.raw_receipt, label="raw receipt"),
        _json_object(args.schema_manifest, label="frozen schema manifest"),
        _json_object(args.zero_bypass_receipt, label="zero-bypass receipt"),
    )


def _sealed_catalog(args: argparse.Namespace) -> Any:
    # First invoke the Phase-D binding wrapper, then reopen through the
    # adapter's own type boundary.  Both must agree before a day writer is
    # allowed to ask for metadata-dependent target labels.
    catalog, _binding = load_sealed_structured_catalog(
        args.catalog,
        args.catalog_receipt,
        fixed_vectors_path=args.catalog_vectors,
    )
    adapter_catalog = load_sealed_public_catalog(
        args.catalog,
        args.catalog_receipt,
        vectors_path=args.catalog_vectors,
    )
    if type(catalog) is not type(adapter_catalog):
        raise RuleDerivativeTrainingError("sealed catalog wrapper and adapter type disagree")
    return adapter_catalog


def _with_lease(root: Path, purpose: str) -> tuple[ExperimentLease, ResourceTelemetry]:
    lease = ExperimentLease.acquire(root / ".r298-memory-heavy.lease", purpose=purpose)
    return lease, ResourceTelemetry(lease=lease, require_exclusive_lease=True)


def command_schema_manifest(args: argparse.Namespace) -> None:
    _require_execute(args)
    _catalog, binding = load_sealed_structured_catalog(
        args.catalog,
        args.catalog_receipt,
        fixed_vectors_path=args.catalog_vectors,
    )
    manifest = build_frozen_schema_manifest(catalog_binding=binding)
    destination = Path(args.output)
    digest = write_frozen_schema_manifest_create_only(destination, manifest)
    _print(
        {
            "status": "created_schema_manifest_only",
            "path": str(destination),
            "file_sha256": digest,
            "manifest": manifest,
        }
    )


def command_init(args: argparse.Namespace) -> None:
    _require_execute(args)
    raw_manifest, raw_receipt, schema_manifest, bypass = _materialization_inputs(args)
    # Verify the exact sealed catalog again even though init itself does not
    # persist catalog metadata; this prevents a run root whose later day
    # writers could not satisfy its frozen catalog binding.
    _sealed_catalog(args)
    result = initialize_refeature_root(
        args.output_root,
        raw_manifest=raw_manifest,
        raw_receipt=raw_receipt,
        schema_manifest=schema_manifest,
        zero_bypass_receipt=bypass,
    )
    _print(result)


def command_day(args: argparse.Namespace) -> None:
    _require_execute(args)
    raw_manifest, raw_receipt, schema_manifest, bypass = _materialization_inputs(args)
    root = Path(args.output_root)
    lease, telemetry = _with_lease(root, f"r298-refeature-day:{args.day}")
    try:
        result = materialize_refeature_day(
            root,
            day=args.day,
            raw_manifest=raw_manifest,
            raw_receipt=raw_receipt,
            schema_manifest=schema_manifest,
            zero_bypass_receipt=bypass,
            archive_root=Path(args.archive_root) if args.archive_root else None,
            metadata_catalog=_sealed_catalog(args),
            telemetry=telemetry,
        )
        _print(result)
    except Exception:
        # Keep a failed lease visible to the operator; never erase a failed
        # run's serialization evidence automatically.
        raise
    else:
        lease.release()


def command_merge(args: argparse.Namespace) -> None:
    _require_execute(args)
    raw_manifest, raw_receipt, schema_manifest, bypass = _materialization_inputs(args)
    root = Path(args.output_root)
    lease, telemetry = _with_lease(root, "r298-refeature-merge")
    try:
        manifest, receipt, split = merge_refeatured_days(
            root,
            raw_manifest=raw_manifest,
            raw_receipt=raw_receipt,
            schema_manifest=schema_manifest,
            zero_bypass_receipt=bypass,
            telemetry=telemetry,
        )
        _print({"manifest": manifest, "receipt": receipt, "split": split})
    except Exception:
        raise
    else:
        lease.release()


def command_parallel_plan(args: argparse.Namespace) -> None:
    _print(bounded_day_parallel_plan(lane_count=args.lanes))


def command_transfer_dry_run(args: argparse.Namespace) -> None:
    plan = build_transfer_plan(
        args.output_root,
        refeature_manifest_path=args.refeature_manifest,
        raw_manifest_path=args.raw_manifest,
        raw_receipt_path=args.raw_receipt,
        frozen_schema_manifest_path=args.schema_manifest,
        zero_bypass_receipt_path=args.zero_bypass_receipt,
        lane_count=args.lanes,
    )
    if args.output:
        _require_execute(args)
        digest = write_transfer_plan_create_only(args.output, plan)
        _print({"plan_file_sha256": digest, "plan": plan})
    else:
        _print(plan)


def command_inspect_frozen_tensors(args: argparse.Namespace) -> None:
    _require_execute(args)
    components = _json_object(args.frozen_components, label="frozen component map")
    new_names = _json_object(args.new_sidecar_tensors, label="new sidecar tensor list")
    tensor_names = new_names.get("names")
    if not isinstance(tensor_names, list) or not all(isinstance(item, str) for item in tensor_names):
        raise RuleDerivativeTrainingError("new sidecar tensor JSON must be {'names': [string, ...]}")
    audit = inspect_and_build_frozen_tensor_audit_create_only(
        baseline_checkpoint_path=args.baseline_checkpoint,
        candidate_checkpoint_path=args.candidate_checkpoint,
        trusted_roots=args.trusted_root,
        frozen_component_tensor_names=components,
        trainable_new_tensor_names=tensor_names,
    )
    digest = write_frozen_tensor_audit_create_only(args.output, audit)
    _print({"audit_file_sha256": digest, "audit": audit})


def _add_materialization_arguments(parser: argparse.ArgumentParser, *, catalog: bool) -> None:
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--raw-manifest", required=True)
    parser.add_argument("--raw-receipt", required=True)
    parser.add_argument("--schema-manifest", required=True)
    parser.add_argument("--zero-bypass-receipt", required=True)
    if catalog:
        parser.add_argument("--catalog", required=True)
        parser.add_argument("--catalog-receipt", required=True)
        parser.add_argument("--catalog-vectors", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    schema = subparsers.add_parser("schema-manifest")
    schema.add_argument("--catalog", required=True)
    schema.add_argument("--catalog-receipt", required=True)
    schema.add_argument("--catalog-vectors", required=True)
    schema.add_argument("--output", required=True)
    schema.add_argument("--execute", action="store_true")
    schema.set_defaults(handler=command_schema_manifest)

    init = subparsers.add_parser("init")
    _add_materialization_arguments(init, catalog=True)
    init.add_argument("--execute", action="store_true")
    init.set_defaults(handler=command_init)

    day = subparsers.add_parser("day")
    _add_materialization_arguments(day, catalog=True)
    day.add_argument("--day", required=True)
    day.add_argument("--archive-root")
    day.add_argument("--execute", action="store_true")
    day.set_defaults(handler=command_day)

    merge = subparsers.add_parser("merge")
    _add_materialization_arguments(merge, catalog=False)
    merge.add_argument("--execute", action="store_true")
    merge.set_defaults(handler=command_merge)

    parallel = subparsers.add_parser("parallel-plan")
    parallel.add_argument("--lanes", type=int, default=1)
    parallel.set_defaults(handler=command_parallel_plan)

    transfer = subparsers.add_parser("transfer-dry-run")
    _add_materialization_arguments(transfer, catalog=False)
    transfer.add_argument("--refeature-manifest", required=True)
    transfer.add_argument("--lanes", type=int, default=4)
    transfer.add_argument("--output")
    transfer.add_argument("--execute", action="store_true")
    transfer.set_defaults(handler=command_transfer_dry_run)

    audit = subparsers.add_parser("inspect-frozen-tensors")
    audit.add_argument("--baseline-checkpoint", required=True)
    audit.add_argument("--candidate-checkpoint", required=True)
    audit.add_argument("--trusted-root", required=True, action="append")
    audit.add_argument("--frozen-components", required=True)
    audit.add_argument("--new-sidecar-tensors", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--execute", action="store_true")
    audit.set_defaults(handler=command_inspect_frozen_tensors)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.handler(args)
    except RuleDerivativeTrainingError as exc:
        print(f"r298 materialization blocked: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
