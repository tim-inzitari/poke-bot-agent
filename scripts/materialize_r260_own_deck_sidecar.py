#!/usr/bin/env python3
"""Create-only r260 sidecar evidence and Inzi-local transport operations.

This command has no service-manager, trainer, Docker, SSH, selector, package,
or submission integration.  Every subcommand only reads immutable inputs and
creates a new artifact; it refuses replacement and does not make partial
prefix staging training-eligible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.r241_own_deck_successor import load_r260_owner_contract
from poke_bot.r260_sidecar_materialization import (
    CompletionResult,
    R260SidecarMaterializationError,
    attest_r260_causal_local_remote_parity,
    attest_r274_local_post_transfer,
    attest_r274_local_rebuild_parity,
    audit_r260_sidecar,
    audit_r260_sidecar_prefix,
    bind_r260_inzi_dataset,
    file_identity,
    finalize_r260_inzi_sidecar,
    materialize_r260_sidecar,
    promote_r260_inzi_staging_root,
    stage_r260_sidecar_prefix_to_inzi,
    transport_r260_sidecar_to_inzi,
    write_r274_local_post_transfer_receipt,
)


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _contract(args: argparse.Namespace) -> Any:
    return load_r260_owner_contract(args.owner_contract)


def _identity(value: Any) -> dict[str, object]:
    return {
        "path": str(value.path),
        "sha256": value.sha256,
        "size_bytes": value.size_bytes,
    }


def _add_owner(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--owner-contract",
        type=_path,
        default=Path("state/alakazam-new-list-direct-policy-r241.json"),
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit", help="require/re-hash the completed 20/20 source")
    _add_owner(audit)
    audit.add_argument("--sidecar-root", required=True, type=_path)
    audit.add_argument("--source-manifest", required=True, type=_path)
    audit.add_argument("--source-window-receipt", required=True, type=_path)
    audit.add_argument("--expected-sidecar-root", type=_path)

    prefix_audit = commands.add_parser(
        "audit-prefix", help="re-hash a nonterminal gap-free committed source prefix"
    )
    _add_owner(prefix_audit)
    prefix_audit.add_argument("--sidecar-root", required=True, type=_path)
    prefix_audit.add_argument("--source-manifest", required=True, type=_path)
    prefix_audit.add_argument("--source-window-receipt", required=True, type=_path)
    prefix_audit.add_argument("--expected-sidecar-root", type=_path)

    materialize = commands.add_parser(
        "materialize", help="create deterministic joined data only after 20/20"
    )
    _add_owner(materialize)
    materialize.add_argument("--sidecar-root", required=True, type=_path)
    materialize.add_argument("--source-manifest", required=True, type=_path)
    materialize.add_argument("--source-window-receipt", required=True, type=_path)
    materialize.add_argument("--evidence-root", required=True, type=_path)
    materialize.add_argument("--expected-sidecar-root", type=_path)
    materialize.add_argument("--receipt-identity-root", type=_path)

    local_attest = commands.add_parser(
        "attest-local-transfer", help="rehash the sealed 20/20 Inzi transfer under r274"
    )
    _add_owner(local_attest)
    local_attest.add_argument("--inzi-staging-root", required=True, type=_path)
    local_attest.add_argument("--source-manifest", required=True, type=_path)
    local_attest.add_argument("--source-window-receipt", required=True, type=_path)
    local_attest.add_argument("--legacy-receipts-root", required=True, type=_path)
    local_attest.add_argument("--expected-elmo-sidecar-root", required=True, type=_path)

    local_transport = commands.add_parser(
        "bind-local-transfer", help="bind an Inzi-local join to the r274 transfer attestation"
    )
    _add_owner(local_transport)
    local_transport.add_argument("--inzi-sidecar-root", required=True, type=_path)
    local_transport.add_argument("--inzi-evidence-root", required=True, type=_path)
    local_transport.add_argument("--local-post-transfer-attestation", required=True, type=_path)
    local_transport.add_argument("--receipt-identity-root", required=True, type=_path)

    local_parity = commands.add_parser(
        "parity-local-rebuild", help="independently rebuild and compare the joined dataset on Inzi"
    )
    _add_owner(local_parity)
    local_parity.add_argument("--inzi-sidecar-root", required=True, type=_path)
    local_parity.add_argument("--inzi-evidence-root", required=True, type=_path)
    local_parity.add_argument("--receipt-identity-root", required=True, type=_path)
    local_parity.add_argument("--sample-limit", type=int, default=256)

    stage = commands.add_parser(
        "stage-prefix", help="append-only copy of committed non-dot days into Inzi staging"
    )
    _add_owner(stage)
    stage.add_argument("--source-sidecar-root", required=True, type=_path)
    stage.add_argument("--source-manifest", required=True, type=_path)
    stage.add_argument("--source-window-receipt", required=True, type=_path)
    stage.add_argument("--inzi-staging-root", required=True, type=_path)
    stage.add_argument("--expected-elmo-sidecar-root", required=True, type=_path)

    promote = commands.add_parser(
        "promote", help="atomically promote only complete 20/20 Inzi staging"
    )
    _add_owner(promote)
    promote.add_argument("--inzi-staging-root", required=True, type=_path)
    promote.add_argument("--inzi-final-root", required=True, type=_path)
    promote.add_argument("--expected-elmo-sidecar-root", required=True, type=_path)

    transport = commands.add_parser(
        "transport", help="full byte-identical 20/20 copy plus initial evidence"
    )
    _add_owner(transport)
    transport.add_argument("--source-sidecar-root", required=True, type=_path)
    transport.add_argument("--source-evidence-root", required=True, type=_path)
    transport.add_argument("--inzi-sidecar-root", required=True, type=_path)
    transport.add_argument("--expected-elmo-sidecar-root", required=True, type=_path)

    parity = commands.add_parser(
        "parity", help="write bounded local/remote causal parity evidence"
    )
    _add_owner(parity)
    parity.add_argument("--elmo-sidecar-root", required=True, type=_path)
    parity.add_argument("--elmo-evidence-root", required=True, type=_path)
    parity.add_argument("--inzi-sidecar-root", required=True, type=_path)
    parity.add_argument("--inzi-evidence-root", required=True, type=_path)
    parity.add_argument("--expected-elmo-sidecar-root", required=True, type=_path)
    parity.add_argument("--sample-limit", type=int, default=256)

    finalize = commands.add_parser(
        "finalize", help="write Inzi-local completion and aggregate binding"
    )
    _add_owner(finalize)
    finalize.add_argument("--inzi-sidecar-root", required=True, type=_path)
    finalize.add_argument("--inzi-evidence-root", required=True, type=_path)
    finalize.add_argument("--expected-elmo-sidecar-root", required=True, type=_path)
    finalize.add_argument("--receipt-identity-root", type=_path)

    bind = commands.add_parser(
        "bind-inzi", help="write the final Inzi dataset binding")
    _add_owner(bind)
    bind.add_argument("--inzi-sidecar-root", required=True, type=_path)
    bind.add_argument("--inzi-evidence-root", required=True, type=_path)
    bind.add_argument("--aggregate-binding", required=True, type=_path)
    bind.add_argument("--completion-receipt", required=True, type=_path)
    bind.add_argument("--parity-receipt", required=True, type=_path)
    bind.add_argument("--joined-dataset", required=True, type=_path)
    bind.add_argument("--transport-receipt", required=True, type=_path)
    return root


def _audit_output(audit: Any) -> dict[str, object]:
    return {
        "sidecar_root": str(audit.sidecar_root),
        "day_count": len(audit.daily),
        "days": [row.day for row in audit.daily],
        "daily_meta": audit.daily_meta_identities,
        "daily_shards": audit.daily_shard_identities,
        "daily_build_identity": dict(audit.daily_build_identity),
        "validated_episode_count": audit.validated_episode_count,
        "source_archive_bytes": audit.source_archive_bytes,
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        contract = _contract(args)
        if args.command == "audit":
            result = audit_r260_sidecar(
                sidecar_root=args.sidecar_root,
                source_manifest=args.source_manifest,
                source_window_receipt=args.source_window_receipt,
                expected_sidecar_root=args.expected_sidecar_root,
                owner_contract=contract,
            )
            output = _audit_output(result)
        elif args.command == "audit-prefix":
            result = audit_r260_sidecar_prefix(
                sidecar_root=args.sidecar_root,
                source_manifest=args.source_manifest,
                source_window_receipt=args.source_window_receipt,
                expected_sidecar_root=args.expected_sidecar_root,
                owner_contract=contract,
            )
            output = _audit_output(result)
            output["training_eligible"] = False
        elif args.command == "materialize":
            result = materialize_r260_sidecar(
                sidecar_root=args.sidecar_root,
                source_manifest=args.source_manifest,
                source_window_receipt=args.source_window_receipt,
                evidence_root=args.evidence_root,
                expected_sidecar_root=args.expected_sidecar_root,
                owner_contract=contract,
                receipt_identity_root=args.receipt_identity_root,
            )
            output = {
                "evidence_root": str(result.evidence_root),
                "joined_dataset": _identity(result.joined_dataset),
                "joined_manifest": _identity(result.joined_manifest),
                "join_receipt": _identity(result.join_receipt),
                "schema_receipt": _identity(result.schema_receipt),
                "count_receipt": _identity(result.count_receipt),
                "digest_receipt": _identity(result.digest_receipt),
                "row_count": result.row_count,
            }
        elif args.command == "attest-local-transfer":
            attestation, prefix = attest_r274_local_post_transfer(
                inzi_staging_root=args.inzi_staging_root,
                source_manifest=args.source_manifest,
                source_window_receipt=args.source_window_receipt,
                legacy_receipts_root=args.legacy_receipts_root,
                expected_elmo_sidecar_root=args.expected_elmo_sidecar_root,
                owner_contract=contract,
            )
            output = {
                "local_post_transfer_attestation": _identity(attestation),
                "complete_prefix_receipt": _identity(prefix),
                "training_eligible": False,
            }
        elif args.command == "bind-local-transfer":
            result = write_r274_local_post_transfer_receipt(
                inzi_sidecar_root=args.inzi_sidecar_root,
                inzi_evidence_root=args.inzi_evidence_root,
                local_post_transfer_attestation=args.local_post_transfer_attestation,
                receipt_identity_root=args.receipt_identity_root,
                owner_contract=contract,
            )
            output = {"transport_receipt": _identity(result)}
        elif args.command == "parity-local-rebuild":
            result = attest_r274_local_rebuild_parity(
                inzi_sidecar_root=args.inzi_sidecar_root,
                inzi_evidence_root=args.inzi_evidence_root,
                receipt_identity_root=args.receipt_identity_root,
                owner_contract=contract,
                sample_limit=args.sample_limit,
            )
            output = {"parity_receipt": _identity(result)}
        elif args.command == "stage-prefix":
            result = stage_r260_sidecar_prefix_to_inzi(
                source_sidecar_root=args.source_sidecar_root,
                source_manifest=args.source_manifest,
                source_window_receipt=args.source_window_receipt,
                inzi_staging_root=args.inzi_staging_root,
                expected_elmo_sidecar_root=args.expected_elmo_sidecar_root,
                owner_contract=contract,
            )
            output = {
                "inzi_staging_root": str(result.inzi_staging_root),
                "committed_days": list(result.committed_days),
                "copied_days": list(result.copied_days),
                "prefix_receipt": _identity(result.prefix_receipt),
                "training_eligible": False,
            }
        elif args.command == "promote":
            result = promote_r260_inzi_staging_root(
                inzi_staging_root=args.inzi_staging_root,
                inzi_final_root=args.inzi_final_root,
                expected_elmo_sidecar_root=args.expected_elmo_sidecar_root,
                owner_contract=contract,
            )
            output = {"inzi_final_root": str(result), "training_eligible": False}
        elif args.command == "transport":
            result = transport_r260_sidecar_to_inzi(
                source_sidecar_root=args.source_sidecar_root,
                source_evidence_root=args.source_evidence_root,
                inzi_sidecar_root=args.inzi_sidecar_root,
                expected_elmo_sidecar_root=args.expected_elmo_sidecar_root,
                owner_contract=contract,
            )
            output = {
                "inzi_sidecar_root": str(result.inzi_sidecar_root),
                "inzi_evidence_root": str(result.inzi_evidence_root),
                "joined_dataset": _identity(result.joined_dataset),
                "joined_manifest": _identity(result.joined_manifest),
                "transport_receipt": _identity(result.transport_receipt),
            }
        elif args.command == "parity":
            result = attest_r260_causal_local_remote_parity(
                elmo_sidecar_root=args.elmo_sidecar_root,
                elmo_evidence_root=args.elmo_evidence_root,
                inzi_sidecar_root=args.inzi_sidecar_root,
                inzi_evidence_root=args.inzi_evidence_root,
                expected_elmo_sidecar_root=args.expected_elmo_sidecar_root,
                owner_contract=contract,
                sample_limit=args.sample_limit,
            )
            output = {"parity_receipt": _identity(result)}
        elif args.command == "finalize":
            result = finalize_r260_inzi_sidecar(
                inzi_sidecar_root=args.inzi_sidecar_root,
                inzi_evidence_root=args.inzi_evidence_root,
                expected_elmo_sidecar_root=args.expected_elmo_sidecar_root,
                owner_contract=contract,
                receipt_identity_root=args.receipt_identity_root,
            )
            output = {
                "aggregate_binding": _identity(result.aggregate_binding),
                "completion_receipt": _identity(result.completion_receipt),
                "parity_receipt": _identity(result.parity_receipt),
                "joined_dataset": _identity(result.joined_dataset),
            }
        elif args.command == "bind-inzi":
            aggregate = file_identity(args.aggregate_binding, label="aggregate binding")
            completion = CompletionResult(
                audit=audit_r260_sidecar(
                    sidecar_root=args.inzi_sidecar_root,
                    source_manifest=args.inzi_evidence_root / "source-manifest.json",
                    source_window_receipt=args.inzi_evidence_root / "source-window-receipt.json",
                    owner_contract=contract,
                ),
                inzi_sidecar_root=args.inzi_sidecar_root.resolve(),
                evidence_root=args.inzi_evidence_root.resolve(),
                aggregate_binding=aggregate,
                completion_receipt=file_identity(args.completion_receipt, label="completion receipt"),
                parity_receipt=file_identity(args.parity_receipt, label="parity receipt"),
                joined_dataset=file_identity(args.joined_dataset, label="joined dataset"),
            )
            result = bind_r260_inzi_dataset(
                completion=completion,
                source_transport_receipt=args.transport_receipt,
                owner_contract=contract,
            )
            output = {"inzi_dataset_binding": _identity(result)}
        else:  # pragma: no cover - argparse guarantees this.
            raise AssertionError(args.command)
    except (OSError, ValueError, R260SidecarMaterializationError) as exc:
        print(f"r260 sidecar producer refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
