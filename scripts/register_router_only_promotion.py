#!/usr/bin/env python3
"""Register an audited inactive router without replacing expert corpora."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _atomic(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"immutable promotion differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--parent-receipt", type=Path)
    parser.add_argument("--ready-archetype", action="append", default=[])
    args = parser.parse_args()

    tree_path = args.tree.resolve()
    audit_path = args.audit.resolve()
    roster_path = args.roster.resolve()
    corpus_root = args.corpus_root.resolve()
    tree = _read(tree_path)
    audit = _read(audit_path)
    roster = _read(roster_path)
    targets = tuple(str(value) for value in roster.get("active_expert_ids") or ())
    candidate_targets = tuple(str(value) for value in tree.get("targets") or ())
    accepted = tuple(str(value) for value in audit.get("accepted_specialist_ids") or ())
    ready = tuple(dict.fromkeys(str(value) for value in args.ready_archetype))
    if (
        roster.get("schema") != "poke_bot.matchup_adapter_roster/v1"
        or not targets
        or len(set(targets)) != len(targets)
        or tree.get("schema") != "poke_bot.public_matchup_decision_tree/v1"
        or tree.get("runtime_enabled") is not False
        or candidate_targets != targets
        or audit.get("schema") != "poke_bot.public_matchup_tree_candidate_audit/v1"
        or audit.get("runtime_enabled") is not False
        or audit.get("artifact_sha256") != _sha256(tree_path)
        or float(audit.get("minimum_precision") or 0.0) != 0.93
        or int(audit.get("minimum_weighted_support") or 0) != 10_000
        or int(audit.get("target_count") or 0) != len(targets)
        or int(audit.get("accepted_count") or 0) != len(accepted)
        or len(set(accepted)) != len(accepted)
        or not set(accepted).issubset(targets)
        or not set(ready).issubset(accepted)
        or not corpus_root.is_dir()
    ):
        raise RuntimeError("router-only promotion identity failed")
    for specialist_id in ready:
        if not (
            corpus_root / specialist_id / "PROTECTED_EXPERT_CORPUS.json"
        ).is_file():
            raise RuntimeError(f"ready corpus missing for {specialist_id}")

    receipt = {
        "schema": "poke_bot.rare_route_asset_promotion/v1",
        "status": "ready",
        "activation_policy": "specialist_boundary_only",
        "live_trainer_modified": False,
        "corpus_mode": "router_only_preserve_canonical_expert_generation",
        "corpus_root": str(corpus_root),
        "candidate_tree": str(tree_path),
        "candidate_tree_sha256": _sha256(tree_path),
        "candidate_audit": str(audit_path),
        "candidate_audit_sha256": _sha256(audit_path),
        "accepted_specialist_ids": list(accepted),
        "accepted_count": len(accepted),
        "canonical_target_ids": list(targets),
        "canonical_target_count": len(targets),
        "ready_rare_archetype_ids": list(ready),
        "parent_promotion_receipt": (
            str(args.parent_receipt.resolve()) if args.parent_receipt else None
        ),
        "parent_promotion_receipt_sha256": (
            _sha256(args.parent_receipt.resolve())
            if args.parent_receipt
            else None
        ),
    }
    _atomic(args.receipt.resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
