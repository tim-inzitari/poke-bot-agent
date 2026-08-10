"""Inspect the dormant, guide-agnostic Guide2Vec pipeline boundary.

``check`` is read-only and reports the important distinction between byte-sealed
content and a semantically proved causal split.  ``preflight`` is also
read-only: after a future manifest is semantically ready, it validates a
separate checksum-bound owner receipt and still does not invoke a trainer.

No command here writes a receipt, reads a Torch chunk, creates a candidate,
starts a service, or changes runtime, search, selector, serving, promotion, or
Kaggle state.  The current r226 template is deliberately not launchable.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class Guide2VecPipelinePreflightError(ValueError):
    """Raised when an inert pipeline cannot become a future launch preflight."""


def _load_manifest(manifest: Path) -> Any:
    from poke_bot.guide2vec_contracts import load_and_validate_training_manifest

    return load_and_validate_training_manifest(manifest, require_ready=False)


def _readiness_payload(resolved: Any) -> dict[str, object]:
    readiness = resolved.readiness
    return {
        "manifest": str(resolved.manifest_path),
        "manifest_sha256": resolved.manifest_sha256,
        "manifest_status": readiness.status,
        "declared_inputs": list(readiness.declared_inputs),
        "resolved_inputs": list(readiness.resolved_inputs),
        "unresolved_inputs": list(readiness.unresolved_inputs),
        "content_refs_sealed": readiness.content_refs_sealed,
        "split_semantics_proven": readiness.split_semantics_proven,
        "split_proof_required_schema": readiness.split_proof_required_schema,
        "blocking_reasons": list(readiness.blocking_reasons),
        "data_ready": readiness.data_ready,
        "training_authorized_by_manifest": readiness.training_authorized,
        "separate_activation_receipt_required": readiness.separate_activation_receipt_required,
    }


def _inert_authority_payload() -> dict[str, bool]:
    return {
        "training_executed": False,
        "training_service_started": False,
        "candidate_published": False,
        "runtime_attached": False,
        "selector_changed": False,
        "serving_changed": False,
        "bo1000_started": False,
        "mcts_changed": False,
        "rtp_changed": False,
        "kaggle_called": False,
    }


def _check(manifest: Path) -> dict[str, object]:
    resolved = _load_manifest(manifest)
    readiness = resolved.readiness
    if readiness.data_ready:
        status = "manifest_semantically_ready_but_unactivated"
    elif readiness.content_refs_sealed:
        status = "manifest_content_sealed_but_not_semantically_ready"
    else:
        status = "manifest_dormant"
    return {
        "command": "check",
        "status": status,
        **_readiness_payload(resolved),
        "authority": _inert_authority_payload(),
    }


def _preflight(manifest: Path, activation_receipt: Path) -> dict[str, object]:
    resolved = _load_manifest(manifest)
    readiness = resolved.readiness
    if not readiness.data_ready:
        reasons = "; ".join(readiness.blocking_reasons)
        raise Guide2VecPipelinePreflightError(
            "Guide2Vec preflight is non-launchable before semantic split proof: "
            f"{reasons}"
        )
    from poke_bot.guide2vec_activation import validate_activation_receipt

    grant = validate_activation_receipt(
        training_manifest_path=manifest,
        receipt_path=activation_receipt,
    )
    return {
        "command": "preflight",
        "status": "preflight_complete_training_not_implemented",
        **_readiness_payload(resolved),
        "activation_receipt": str(grant.receipt_path),
        "activation_receipt_sha256": grant.receipt_sha256,
        "owner_contract_revision": grant.owner_contract_revision,
        "authority": {
            "activation_receipt_valid": True,
            **_inert_authority_payload(),
        },
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", help="validate an inert manifest")
    check.add_argument("--manifest", type=Path, required=True)

    preflight = commands.add_parser(
        "preflight",
        help="validate future launch inputs only; this command never trains",
    )
    preflight.add_argument("--manifest", type=Path, required=True)
    preflight.add_argument("--activation-receipt", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        payload = (
            _check(args.manifest)
            if args.command == "check"
            else _preflight(args.manifest, args.activation_receipt)
        )
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "command": args.command,
                    "status": "validation_failed",
                    "error": str(exc),
                    "training_executed": False,
                    "authority": _inert_authority_payload(),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 2
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
