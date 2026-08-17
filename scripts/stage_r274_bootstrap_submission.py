#!/usr/bin/env python3
"""Build and queue the exact route-off r274 bootstrap submission once."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tarfile
from typing import Any, Mapping

from poke_bot.r274_bootstrap_handoff import (
    BOOTSTRAP_SUBMISSION_SCHEMA,
    file_identity,
    receipt_digest,
)
from scripts.activate_public_matchup_tree import activate as activate_matchup_tree
from scripts.handle_passed_gate import (
    _copy_submission_slot,
    build_submission_bundle,
    queue_submission_copies,
)
from scripts.stage_alakazam_terminal_no_rtp_submission_r195 import (
    materialize_checkpoint_compatible_matchup_tree,
)


SCHEMA = "poke_bot.alakazam_r274_bootstrap_submission_stage/v1"
CANDIDATE_ID = "alakazam-new-list-direct-policy-r274"
OWNER_SOURCE = "GOAL.md#/revision-264-TRAINING"
ARCHIVE_MAX_BYTES = int(197.7 * 1024 * 1024)


def _json(path: Path) -> dict[str, Any]:
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


def _canonical_digest(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _write_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    body = json.dumps(
        dict(payload), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_text() != body:
            raise RuntimeError(f"immutable r274 receipt changed: {path}")
        return
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _deck_receipt(contract: Mapping[str, Any], contract_path: Path) -> dict[str, Any]:
    row = dict(contract.get("exact_deck") or {})
    source = Path(str(row.get("path") or "")).expanduser()
    if not source.is_absolute():
        source = contract_path.parent.parent / source
    source = source.resolve()
    cards = [
        int(line.split(",", 1)[0])
        for raw in source.read_text(encoding="utf-8").splitlines()
        if (line := raw.strip()) and not line.startswith("#")
    ]
    if (
        len(cards) != 60
        or _sha256(source) != row.get("file_sha256")
        or _canonical_digest(cards) != row.get("ordered_cards_sha256")
    ):
        raise RuntimeError("r274 exact submission deck identity changed")
    return {
        "path": str(source),
        "cards": 60,
        "cards_sha256": row["ordered_cards_sha256"],
        "file_sha256": row["file_sha256"],
        "representatives": str(contract_path),
        "representatives_sha256": _sha256(contract_path),
    }


def _audit_archive(
    archive: Path,
    *,
    checkpoint_sha256: str,
    deck_sha256: str,
    matchup_tree_sha256: str,
    official_libcg_sha256: str,
) -> dict[str, Any]:
    if archive.stat().st_size > ARCHIVE_MAX_BYTES:
        raise RuntimeError("r274 bootstrap archive exceeds 197.7 MiB")
    with tarfile.open(archive, "r:gz") as bundle:
        members = {
            member.name.removeprefix("./"): member
            for member in bundle.getmembers()
            if member.isfile()
        }

        def digest(name: str) -> str:
            member = members.get(name)
            stream = bundle.extractfile(member) if member is not None else None
            if stream is None:
                raise RuntimeError(f"r274 archive lacks {name}")
            return "sha256:" + hashlib.sha256(stream.read()).hexdigest()

        checks = {
            "model": digest("model.pt") == checkpoint_sha256,
            "deck": digest("deck.csv") == deck_sha256,
            "matchup_tree": digest("matchup_tree.json") == matchup_tree_sha256,
            "official_libcg": digest("cg/libcg.so") == official_libcg_sha256,
            "main": "main.py" in members,
            "runtime_profile": "runtime_profile.json" in members,
            "turn_order_profile": "turn_order_profile.json" in members,
            "search_config_absent": "search_config.json" not in members,
            "belief_decks_absent": "belief_decks.json" not in members,
            "rtp_sidecar_absent": not any(
                name.endswith("rtp_shadow_planner.pt") for name in members
            ),
        }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError("r274 archive audit failed: " + ",".join(failed))
    return {"passed": True, "checks": checks, "members": len(members)}


def stage(args: argparse.Namespace) -> dict[str, Any]:
    if args.receipt.is_file():
        existing = _json(args.receipt)
        if existing.get("schema") != SCHEMA or existing.get("status") != "queued":
            raise RuntimeError("existing r274 bootstrap stage receipt is invalid")
        return existing
    contract = _json(args.contract)
    if (
        contract.get("schema") != "poke_bot.alakazam_new_list_direct_policy_r274/v1"
        or contract.get("candidate_id") != CANDIDATE_ID
        or int(contract.get("latest_owner_clarification_revision", -1)) < 280
    ):
        raise RuntimeError("r274 canonical training contract changed")
    submission_receipt = _json(args.bootstrap_submission_receipt)
    checkpoint = file_identity(args.submission_checkpoint)
    if (
        submission_receipt.get("schema") != BOOTSTRAP_SUBMISSION_SCHEMA
        or submission_receipt.get("status") != "ready_for_first_if_allowed_submission"
        or submission_receipt.get("receipt_sha256") != receipt_digest(submission_receipt)
        or dict(submission_receipt.get("submission_checkpoint") or {}) != checkpoint
        or submission_receipt.get("direct_policy_only") is not True
        or submission_receipt.get("rtp_enabled") is not False
    ):
        raise RuntimeError("r274 route-off bootstrap checkpoint receipt is invalid")

    args.output_root.mkdir(parents=True, exist_ok=True)
    activated_tree = args.output_root / "matchup-tree-runtime-active.json"
    filtered_tree = args.output_root / "matchup-tree-checkpoint-compatible.json"
    if not activated_tree.exists():
        activate_matchup_tree(
            args.inactive_matchup_tree,
            args.submission_checkpoint,
            activated_tree,
            min_precision=0.93,
            min_support=10_000,
            min_leaf_confidence=0.90,
            consecutive_required=2,
        )
    tree_path, tree_receipt = materialize_checkpoint_compatible_matchup_tree(
        source=activated_tree,
        checkpoint=args.submission_checkpoint,
        output=filtered_tree,
    )
    tree_payload = _json(tree_path)
    runtime = dict(tree_payload.get("runtime_contract") or {})
    if (
        tree_payload.get("runtime_enabled") is not True
        or "alakazam" not in set(runtime.get("accepted_archetype_ids") or [])
        or runtime.get("one_route_per_decision") is not True
        or runtime.get("unknown_route_exact_bypass") is not True
    ):
        raise RuntimeError("r274 checkpoint-compatible matchup runtime is not active")

    deck = _deck_receipt(contract, args.contract)
    bundle = build_submission_bundle(
        repo_root=args.runtime_root,
        frozen_manifest={
            "model_path": checkpoint["path"],
            "checkpoint_digest": checkpoint["sha256"],
        },
        deck_receipt=deck,
        output_dir=args.output_root / "build",
        python=args.python,
        archetype="alakazam",
        matchup_tree=tree_path,
        matchup_roster=args.matchup_roster,
        cg_root=args.cg_root,
        turn_order_preference="first_if_allowed",
        rtp_mode="off",
        direct_no_search_assets=True,
    )
    archive = Path(str(bundle["path"]))
    simulator = dict(contract.get("canonical_simulator") or {})
    audit = _audit_archive(
        archive,
        checkpoint_sha256=checkpoint["sha256"],
        deck_sha256=deck["file_sha256"],
        matchup_tree_sha256=_sha256(tree_path),
        official_libcg_sha256=str(simulator["linux_x86_64_sha256"]),
    )
    copy = _copy_submission_slot(bundle, args.output_root, 1)
    copy["label_suffix"] = "r274 bootstrap NO RTP"
    queued = queue_submission_copies(
        queue_path=args.queue,
        copies=[copy],
        gate_plan={
            "checkpoint_digest": checkpoint["sha256"],
            "gate_id": "alakazam-r274-bootstrap-epoch25",
            "iteration": 0,
            "completion_authority": "training_milestone_snapshot",
            "owner_decision_source": OWNER_SOURCE,
        },
        specialist_id="alakazam",
        competition="pokemon-tcg-ai-battle",
    )
    payload = {
        "schema": SCHEMA,
        "status": "queued",
        "owner_revisions": [264, 268, 274, 275, 277, 279, 280],
        "candidate_id": CANDIDATE_ID,
        "boundary": "bootstrap",
        "bootstrap_submission_receipt": file_identity(
            args.bootstrap_submission_receipt
        ),
        "checkpoint": checkpoint,
        "deck": deck,
        "matchup_tree": {
            "identity": file_identity(tree_path),
            "activation": tree_receipt,
            "accepted_archetype_ids": runtime["accepted_archetype_ids"],
        },
        "bundle": bundle,
        "archive_audit": audit,
        "queue_entry": queued[0],
        "direct_policy_only": True,
        "rtp_enabled": False,
        "search_assets_packaged": False,
        "tactical_route_enabled": False,
        "submission_performed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_create_only(args.receipt, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--submission-checkpoint", type=Path, required=True)
    parser.add_argument("--bootstrap-submission-receipt", type=Path, required=True)
    parser.add_argument("--inactive-matchup-tree", type=Path, required=True)
    parser.add_argument("--matchup-roster", type=Path, required=True)
    parser.add_argument("--cg-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    result = stage(args)
    print(json.dumps({"status": result["status"], "receipt": str(args.receipt)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
