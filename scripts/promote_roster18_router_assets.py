#!/usr/bin/env python3
"""Create an immutable roster-18 derivative of a validated legacy router."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from poke_bot.matchup_adapters import EXPERT_IDS
from scripts.audit_public_matchup_tree_candidate import audit
from scripts.migrate_active_matchup_roster import migrate_tree


SCHEMA = "poke_bot.rare_route_asset_promotion/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def _atomic(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"existing immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", type=Path, required=True)
    parser.add_argument("--output-tree", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-digest", required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--parent-receipt", type=Path, required=True)
    args = parser.parse_args()

    source_tree = args.source_tree.expanduser().resolve()
    output_tree = args.output_tree.expanduser().resolve()
    output_audit = args.output_audit.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    corpus_root = args.corpus_root.expanduser().resolve()
    parent_receipt = args.parent_receipt.expanduser().resolve()
    if not all(
        path.exists()
        for path in (source_tree, checkpoint, corpus_root, parent_receipt)
    ):
        raise RuntimeError("router promotion input is missing")
    if _sha256(checkpoint) != args.checkpoint_digest:
        raise RuntimeError("router promotion checkpoint digest changed")

    if not output_tree.is_file():
        migrate_tree(
            source_tree,
            output_tree,
            checkpoint=checkpoint,
            checkpoint_digest=args.checkpoint_digest,
        )
    candidate = json.loads(output_tree.read_text(encoding="utf-8"))
    if (
        candidate.get("runtime_enabled") is not False
        or tuple(str(value) for value in candidate.get("targets") or ())
        != tuple(EXPERT_IDS)
    ):
        raise RuntimeError("roster-18 router derivative is not inactive/canonical")

    audit_payload = audit(
        output_tree,
        minimum_precision=0.93,
        minimum_weighted_support=10_000,
    )
    _atomic(output_audit, audit_payload)
    accepted = list(audit_payload["accepted_specialist_ids"])
    if "dudunsparce" not in accepted:
        raise RuntimeError("roster-18 router does not accept Dudunsparce")

    receipt = {
        "schema": SCHEMA,
        "status": "ready",
        "activation_policy": "specialist_boundary_only",
        "live_trainer_modified": False,
        "corpus_mode": "router_only_preserve_canonical_latest20",
        "corpus_root": str(corpus_root),
        "candidate_tree": str(output_tree),
        "candidate_tree_sha256": _sha256(output_tree),
        "candidate_audit": str(output_audit),
        "candidate_audit_sha256": _sha256(output_audit),
        "accepted_specialist_ids": accepted,
        "accepted_count": len(accepted),
        "ready_rare_archetype_ids": ["dudunsparce"],
        "parent_promotion_receipt": str(parent_receipt),
        "parent_promotion_receipt_sha256": _sha256(parent_receipt),
        "canonical_target_ids": list(EXPERT_IDS),
        "canonical_target_count": len(EXPERT_IDS),
    }
    _atomic(args.receipt.expanduser().resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
