#!/usr/bin/env python3
"""Build and queue one exact tactical-active r274 five-update submission."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from poke_bot import checkpoint as checkpoint_mod
from poke_bot.r274_bootstrap_handoff import (
    RUNTIME_GATE_FIELDS,
    TACTICAL_ACTIVATION_SCHEMA,
    TACTICAL_ROUTE_FIELDS,
    file_identity,
)
from scripts.handle_passed_gate import (
    _copy_submission_slot,
    build_submission_bundle,
    queue_submission_copies,
)
from scripts.stage_alakazam_terminal_no_rtp_submission_r195 import (
    materialize_checkpoint_compatible_matchup_tree,
)
from scripts.stage_r274_bootstrap_submission import (
    ARCHIVE_MAX_BYTES,
    CANDIDATE_ID,
    OWNER_SOURCE,
    _audit_archive,
    _canonical_digest,
    _deck_receipt,
    _json,
    _sha256,
    _write_create_only,
)
from scripts.train_pure_rl import (
    R274_SUBMISSION_REQUEST_SCHEMA,
)


SCHEMA = "poke_bot.alakazam_r274_rl_submission_stage/v1"
ALLOWED_BOUNDARIES = {1, 5, 10, 15, 20}


def _validate_request(path: Path) -> dict[str, Any]:
    request = _json(path)
    unsigned = {
        key: value for key, value in request.items() if key != "request_sha256"
    }
    checkpoint = dict(request.get("checkpoint") or {})
    boundary = int(request.get("boundary_update", -1))
    if (
        request.get("schema") != R274_SUBMISSION_REQUEST_SCHEMA
        or request.get("candidate_id") != CANDIDATE_ID
        or boundary not in ALLOWED_BOUNDARIES
        or int(request.get("rl_updates_completed", -1)) != boundary
        or request.get("request_sha256") != _canonical_digest(unsigned)
        or request.get("direct_policy") is not True
        or request.get("rtp_enabled") is not False
        or request.get("next_collection_started") is not False
        or not str(request.get("owner_contract_sha256") or "").startswith("sha256:")
    ):
        raise RuntimeError("r274 RL submission request is invalid")
    observed = file_identity(str(checkpoint.get("path") or ""))
    if (
        checkpoint.get("digest") != observed["sha256"]
        or int(checkpoint.get("size_bytes", -1)) != observed["size_bytes"]
    ):
        raise RuntimeError("r274 RL submission request checkpoint changed")
    payload = checkpoint_mod.load_checkpoint(observed["path"], map_location="cpu")
    cfg = dict(payload.get("model_config") or {})
    extra = dict(payload.get("extra") or {})
    activation = dict(extra.get("r274_tactical_route_activation") or {})
    if boundary == 1:
        state = dict(payload.get("model_state") or payload.get("model") or {})
        if (
            any(bool(cfg.get(name)) for name in TACTICAL_ROUTE_FIELDS)
            or any(
                str(name).startswith(
                    ("tactical_sequence_outcome_head.", "tactical_sequence_outcome_route.")
                )
                for name in state
            )
        ):
            raise RuntimeError("r304 iteration-1 checkpoint retained removed tactical sequence tensors")
    elif (
        any(cfg.get(name) is not True for name in RUNTIME_GATE_FIELDS)
        or any(cfg.get(name) is not True for name in TACTICAL_ROUTE_FIELDS)
        or activation.get("schema") != TACTICAL_ACTIVATION_SCHEMA
        or activation.get("planner_dispatch_authority") is not False
    ):
        raise RuntimeError("r274 RL checkpoint is not tactical-route active")
    return request


def stage(args: argparse.Namespace) -> dict[str, Any]:
    if args.receipt.is_file():
        existing = _json(args.receipt)
        if existing.get("schema") != SCHEMA or existing.get("status") != "queued":
            raise RuntimeError("existing r274 RL stage receipt is invalid")
        return existing
    contract = _json(args.contract)
    if (
        contract.get("schema") != "poke_bot.alakazam_new_list_direct_policy_r274/v1"
        or contract.get("candidate_id") != CANDIDATE_ID
        or int(contract.get("latest_owner_clarification_revision", -1)) < 280
    ):
        raise RuntimeError("r274 canonical training contract changed")
    request = _validate_request(args.request)
    boundary = int(request["boundary_update"])
    checkpoint_request = dict(request["checkpoint"])
    checkpoint = file_identity(checkpoint_request["path"])

    args.output_root.mkdir(parents=True, exist_ok=True)
    tree_path, tree_receipt = materialize_checkpoint_compatible_matchup_tree(
        source=args.active_matchup_tree,
        checkpoint=Path(checkpoint["path"]),
        output=args.output_root / "matchup-tree-checkpoint-compatible.json",
    )
    tree_payload = _json(tree_path)
    runtime = dict(tree_payload.get("runtime_contract") or {})
    if (
        tree_payload.get("runtime_enabled") is not True
        or "alakazam" not in set(runtime.get("accepted_archetype_ids") or [])
        or runtime.get("one_route_per_decision") is not True
        or runtime.get("unknown_route_exact_bypass") is not True
    ):
        raise RuntimeError("r274 RL matchup runtime is not active")

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
    if archive.stat().st_size > ARCHIVE_MAX_BYTES:
        raise RuntimeError("r274 RL archive exceeds 197.7 MiB")
    simulator = dict(contract.get("canonical_simulator") or {})
    audit = _audit_archive(
        archive,
        checkpoint_sha256=checkpoint["sha256"],
        deck_sha256=deck["file_sha256"],
        matchup_tree_sha256=_sha256(tree_path),
        official_libcg_sha256=str(simulator["linux_x86_64_sha256"]),
    )
    copy = _copy_submission_slot(bundle, args.output_root, 1)
    copy["label_suffix"] = (
        "r274 iter1 updated learner NO RTP"
        if boundary == 1
        else f"r274 update{boundary:02d} NO RTP"
    )
    queued = queue_submission_copies(
        queue_path=args.queue,
        copies=[copy],
        gate_plan={
            "checkpoint_digest": checkpoint["sha256"],
            "gate_id": f"alakazam-r274-update-{boundary:02d}",
            "iteration": boundary,
            "completion_authority": (
                "revision_304_durable_iter_00001_commit"
                if boundary == 1
                else "five_epoch_refresh_receipt"
            ),
            "owner_decision_source": (
                "GOAL.md#/revision-304-TRAINING"
                if boundary == 1
                else OWNER_SOURCE
            ),
        },
        specialist_id="alakazam",
        competition="pokemon-tcg-ai-battle",
    )
    result = {
        "schema": SCHEMA,
        "status": "queued",
        "candidate_id": CANDIDATE_ID,
        "boundary_update": boundary,
        "request": file_identity(args.request),
        "request_sha256": request["request_sha256"],
        "checkpoint": checkpoint,
        "deck": deck,
        "matchup_tree": {
            "identity": file_identity(tree_path),
            "filter": tree_receipt,
            "accepted_archetype_ids": runtime["accepted_archetype_ids"],
        },
        "bundle": bundle,
        "archive_audit": audit,
        "queue_entry": queued[0],
        "direct_policy_only": True,
        "rtp_enabled": False,
        "search_assets_packaged": False,
        "tactical_route_enabled": boundary != 1,
        "submission_performed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_create_only(args.receipt, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "runtime_root",
        "python",
        "contract",
        "request",
        "active_matchup_tree",
        "matchup_roster",
        "cg_root",
        "output_root",
        "queue",
        "receipt",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    result = stage(args)
    print(json.dumps({"status": result["status"], "receipt": str(args.receipt)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
