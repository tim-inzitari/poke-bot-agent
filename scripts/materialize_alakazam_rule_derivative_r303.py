#!/usr/bin/env python3
"""Create-only r303 diagnostic materialization on canonical Elmo.

This is intentionally not a training, checkpoint, service, staging, or
activation entry point.  It can only create isolated public-rule records after
the r303 freeze and exact 30-day preflight have passed.  Selected-action labels
remain diagnostic with every training mask false.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from poke_bot.alakazam_rule_derivative_materialize_r303 import (
    R303MaterializationError,
    _ElmoMemoryLease,
    _Telemetry,
    bounded_day_parallel_plan_r303,
    build_revision_6_scope_migration_from_inventory,
    initialize_r303_materialization,
    load_r303_sealed_catalog,
    materialize_all_r303_days,
    materialize_r303_day,
    merge_r303_materialized_days,
    scan_revision_6_scope_inventory,
    write_corpus_scope_migration_receipt_create_only,
)


def _read_json(path: str, *, label: str) -> Mapping[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise R303MaterializationError(f"{label} must be a regular JSON file")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R303MaterializationError(f"cannot read {label}") from exc
    if not isinstance(value, Mapping):
        raise R303MaterializationError(f"{label} must be a JSON object")
    return value


def _require_execute(args: argparse.Namespace) -> None:
    if args.execute is not True:
        raise R303MaterializationError("this create-only action requires --execute")


def _inputs(args: argparse.Namespace) -> dict[str, Mapping[str, Any]]:
    return {
        "raw_manifest": _read_json(args.raw_manifest, label="raw manifest"),
        "raw_receipt": _read_json(args.raw_receipt, label="raw corpus receipt"),
        "schema_manifest": _read_json(args.schema_manifest, label="frozen schema manifest"),
        "zero_bypass_receipt": _read_json(args.zero_bypass_receipt, label="zero-bypass receipt"),
        "schema_freeze_receipt": _read_json(args.schema_freeze_receipt, label="schema-freeze receipt"),
        "preflight": _read_json(args.preflight, label="r303 materialization preflight"),
        "corpus_scope_migration_receipt": _read_json(
            args.corpus_scope_migration_receipt, label="revision-6 corpus-scope migration receipt"
        ),
    }


def _catalog(args: argparse.Namespace, schema_manifest: Mapping[str, Any]) -> Any:
    return load_r303_sealed_catalog(
        catalog_path=args.catalog,
        catalog_receipt_path=args.catalog_receipt,
        catalog_vectors_path=args.catalog_vectors,
        schema_manifest=schema_manifest,
    )


def _emit(value: Any) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))


def command_plan(args: argparse.Namespace) -> None:
    _emit(bounded_day_parallel_plan_r303(workers=args.workers))


def command_init(args: argparse.Namespace) -> None:
    _require_execute(args)
    inputs = _inputs(args)
    # Reopening catalog here prevents a root whose day workers could not meet
    # the frozen structured-Catalog identity from being initialized.
    _catalog(args, inputs["schema_manifest"])
    result = initialize_r303_materialization(args.output_root, **inputs)
    _emit(result)


def command_day(args: argparse.Namespace) -> None:
    _require_execute(args)
    inputs = _inputs(args)
    catalog = _catalog(args, inputs["schema_manifest"])
    with _ElmoMemoryLease():
        telemetry = _Telemetry()
        result = materialize_r303_day(
            args.output_root,
            day=args.day,
            raw_manifest=inputs["raw_manifest"],
            schema_manifest=inputs["schema_manifest"],
            catalog=catalog,
            archive_root=Path(args.archive_root) if args.archive_root else None,
            telemetry=telemetry,
        )
    _emit(result)


def command_all(args: argparse.Namespace) -> None:
    _require_execute(args)
    inputs = _inputs(args)
    catalog = _catalog(args, inputs["schema_manifest"])
    result = materialize_all_r303_days(
        args.output_root,
        raw_manifest=inputs["raw_manifest"],
        schema_manifest=inputs["schema_manifest"],
        catalog=catalog,
        archive_root=Path(args.archive_root) if args.archive_root else None,
        workers=args.workers,
    )
    _emit(
        {
            "status": "all_days_complete",
            "completed_days": [row["utc_day_partition"] for row in result],
            "process_pool_execution": str(
                (Path(args.output_root) / "PROCESS_POOL_EXECUTION.json").resolve()
            ),
            "training_allowed": False,
        }
    )


def command_merge(args: argparse.Namespace) -> None:
    _require_execute(args)
    # The merge reopens only the immutable run contract/day markers; read these
    # inputs nonetheless to refuse an operator accidentally passing a stale
    # preflight alongside an otherwise similarly named run root.
    _inputs(args)
    with _ElmoMemoryLease():
        manifest, receipt, split = merge_r303_materialized_days(args.output_root, telemetry=_Telemetry())
    _emit({"manifest": manifest, "receipt": receipt, "split": split})


def command_scope_audit(args: argparse.Namespace) -> None:
    _require_execute(args)
    raw_manifest = _read_json(args.raw_manifest, label="raw manifest")
    with _ElmoMemoryLease():
        inventory, variants, split = scan_revision_6_scope_inventory(
            args.output_root,
            raw_manifest=raw_manifest,
            raw_manifest_path=args.raw_manifest,
            archive_root=Path(args.archive_root) if args.archive_root else None,
            telemetry=_Telemetry(),
            workers=args.workers,
        )
    _emit({"inventory": inventory, "variants": variants, "split": split})


def command_seal_scope_migration(args: argparse.Namespace) -> None:
    _require_execute(args)
    schema = _read_json(args.schema_manifest, label="frozen schema manifest")
    bypass = _read_json(args.zero_bypass_receipt, label="zero-bypass receipt")
    freeze = _read_json(args.schema_freeze_receipt, label="schema-freeze receipt")
    predecessor = _read_json(args.predecessor_receipts, label="predecessor receipt identity list")
    identities = predecessor.get("sha256s")
    if not isinstance(identities, list) or not all(isinstance(item, str) for item in identities):
        raise R303MaterializationError("predecessor receipt JSON must be {'sha256s': [sha256, ...]}")
    receipt = build_revision_6_scope_migration_from_inventory(
        args.scope_root,
        schema_manifest=schema,
        zero_bypass_receipt=bypass,
        schema_freeze_receipt=freeze,
        all_episode_collision_census_receipt_path=args.census_receipt,
        predecessor_corpus_and_refeaturization_receipt_sha256s=identities,
        sealed_at_utc=args.sealed_at_utc,
    )
    identity = write_corpus_scope_migration_receipt_create_only(args.output, receipt)
    _emit({"status": "sealed_revision_6_corpus_scope_migration", "receipt": identity, "training_allowed": False})


def _add_inputs(parser: argparse.ArgumentParser, *, catalog: bool) -> None:
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--raw-manifest", required=True)
    parser.add_argument("--raw-receipt", required=True)
    parser.add_argument("--schema-manifest", required=True)
    parser.add_argument("--zero-bypass-receipt", required=True)
    parser.add_argument("--schema-freeze-receipt", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--corpus-scope-migration-receipt", required=True)
    if catalog:
        parser.add_argument("--catalog", required=True)
        parser.add_argument("--catalog-receipt", required=True)
        parser.add_argument("--catalog-vectors", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("parallel-plan", help="show bounded, data-only whole-day assignments")
    plan.add_argument("--workers", type=int, default=1)
    plan.set_defaults(handler=command_plan)

    init = commands.add_parser("init", help="create a new immutable r303 materialization root")
    _add_inputs(init, catalog=True)
    init.add_argument("--execute", action="store_true")
    init.set_defaults(handler=command_init)

    day = commands.add_parser("day", help="materialize one exact UTC day under the global Elmo lease")
    _add_inputs(day, catalog=True)
    day.add_argument("--day", required=True)
    day.add_argument("--archive-root")
    day.add_argument("--execute", action="store_true")
    day.set_defaults(handler=command_day)

    all_days = commands.add_parser("all", help="materialize all 30 days in one bounded leased process")
    _add_inputs(all_days, catalog=True)
    all_days.add_argument("--archive-root")
    all_days.add_argument("--workers", type=int, default=1)
    all_days.add_argument("--execute", action="store_true")
    all_days.set_defaults(handler=command_all)

    merge = commands.add_parser("merge", help="validate every completed shard and write final manifests")
    _add_inputs(merge, catalog=False)
    merge.add_argument("--execute", action="store_true")
    merge.set_defaults(handler=command_merge)

    scope = commands.add_parser("scope-audit", help="scan all raw episodes and count literal same-seat card-743 eligibility")
    scope.add_argument("--output-root", required=True)
    scope.add_argument("--raw-manifest", required=True)
    scope.add_argument("--archive-root")
    scope.add_argument("--workers", type=int, default=1)
    scope.add_argument("--execute", action="store_true")
    scope.set_defaults(handler=command_scope_audit)

    seal = commands.add_parser("seal-scope-migration", help="write canonical rev6 scope receipt after the all-episode census")
    seal.add_argument("--scope-root", required=True)
    seal.add_argument("--schema-manifest", required=True)
    seal.add_argument("--zero-bypass-receipt", required=True)
    seal.add_argument("--schema-freeze-receipt", required=True)
    seal.add_argument("--census-receipt", required=True)
    seal.add_argument("--predecessor-receipts", required=True)
    seal.add_argument("--sealed-at-utc", required=True)
    seal.add_argument("--output", required=True)
    seal.add_argument("--execute", action="store_true")
    seal.set_defaults(handler=command_seal_scope_migration)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.handler(args)
    except R303MaterializationError as exc:
        print(f"r303 materialization blocked: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
