"""Resolve the newest checksum-bound specialist router/corpus generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from poke_bot.pure_rl.model_registry import sha256


PROMOTION_SCHEMA = "poke_bot.rare_route_asset_promotion/v1"
AUDIT_SCHEMA = "poke_bot.public_matchup_tree_candidate_audit/v1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def resolve_specialist_assets(
    *,
    default_corpus_root: Path,
    default_candidate_tree: Path,
    default_candidate_audit: Path,
    promotion_receipt: Path | None,
    promotion_scope: str = "full_bundle",
) -> dict[str, Any]:
    """Return defaults until a complete immutable promotion is available."""

    if promotion_scope not in {"full_bundle", "router_only"}:
        raise RuntimeError("unknown rare-route promotion scope")
    defaults = {
        "source": "contract_defaults",
        "corpus_root": default_corpus_root.resolve(),
        "candidate_tree": default_candidate_tree.resolve(),
        "candidate_audit": default_candidate_audit.resolve(),
        "promotion_receipt": None,
    }
    if promotion_receipt is None or not promotion_receipt.is_file():
        return defaults
    receipt = _read(promotion_receipt.resolve())
    tree = Path(str(receipt.get("candidate_tree") or "")).resolve()
    audit_path = Path(str(receipt.get("candidate_audit") or "")).resolve()
    corpus_root = Path(str(receipt.get("corpus_root") or "")).resolve()
    audit = _read(audit_path)
    accepted = tuple(
        str(value) for value in receipt.get("accepted_specialist_ids") or ()
    )
    ready_rare = set(
        str(value) for value in receipt.get("ready_rare_archetype_ids") or ()
    )
    if (
        receipt.get("schema") != PROMOTION_SCHEMA
        or receipt.get("status") != "ready"
        or receipt.get("activation_policy") != "specialist_boundary_only"
        or receipt.get("live_trainer_modified") is not False
        or not tree.is_file()
        or not audit_path.is_file()
        or not corpus_root.is_dir()
        or receipt.get("candidate_tree_sha256") != sha256(tree)
        or receipt.get("candidate_audit_sha256") != sha256(audit_path)
        or audit.get("schema") != AUDIT_SCHEMA
        or audit.get("runtime_enabled") is not False
        or audit.get("artifact_sha256") != sha256(tree)
        or tuple(str(value) for value in audit.get("accepted_specialist_ids") or ())
        != accepted
        or int(audit.get("accepted_count") or 0) != len(accepted)
        or not ready_rare.issubset(set(accepted))
        or any(
            not corpus_root.joinpath(
                archetype, "PROTECTED_EXPERT_CORPUS.json"
            ).is_file()
            for archetype in ready_rare
        )
    ):
        raise RuntimeError("rare-route asset promotion receipt changed")
    return {
        "source": "rare_route_promotion",
        "promotion_scope": promotion_scope,
        "corpus_root": (
            default_corpus_root.resolve()
            if promotion_scope == "router_only"
            else corpus_root
        ),
        "promoted_corpus_root": corpus_root,
        "candidate_tree": tree,
        "candidate_audit": audit_path,
        "promotion_receipt": promotion_receipt.resolve(),
        "promotion_receipt_sha256": sha256(promotion_receipt.resolve()),
    }
