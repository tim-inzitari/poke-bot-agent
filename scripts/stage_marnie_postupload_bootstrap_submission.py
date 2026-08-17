#!/usr/bin/env python3
"""Queue the exact revision-134/135 epoch-25 Marnie bootstrap checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.archetype_family_activation import (  # noqa: E402
    sha256 as family_sha256,
    validate_activation_request,
    validate_iteration9_upload_trigger,
    validate_migration_receipt,
)
from scripts.handle_passed_gate import (  # noqa: E402
    _copy_submission_slot,
    build_submission_bundle,
    materialize_pinned_specialist_deck,
    queue_submission_copies,
)


SCHEMA = "poke_bot.marnie_postupload_bootstrap_submission/v1"
READY_SCHEMA = "poke_bot.final_format_marnie_h10_bootstrap_ready/v1"
SPECIALIST_ID = "marnie-s-grimmsnarl-ex"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def write_once(path: Path, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_text(encoding="utf-8") != body:
            raise RuntimeError("post-upload bootstrap submission receipt changed")
        return
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(body, encoding="utf-8")
    os.link(temporary, path)
    temporary.unlink(missing_ok=True)


def validate_ready(
    *,
    ready_path: Path,
    trigger_path: Path,
    request_path: Path,
    migration_path: Path,
    importance_index: Path,
    continuity_receipt: Path,
    guide_retirement_receipt: Path,
    epoch_recovery_receipt: Path,
) -> tuple[dict[str, Any], Path, str]:
    ready = read_json(ready_path)
    trigger = read_json(trigger_path)
    validate_iteration9_upload_trigger(trigger)
    trigger_checkpoint = dict((trigger.get("bindings") or {}).get("checkpoint") or {})
    checkpoint = Path(str(ready.get("checkpoint") or "")).resolve()
    checkpoint_digest = str(ready.get("checkpoint_sha256") or "")
    pilot = dict(ready.get("expert_pilot_importance") or {})
    postupload = dict(ready.get("post_upload_new_system") or {})
    latent = dict(ready.get("latent_policy") or {})
    continuity = dict(ready.get("latent_policy_continuity") or {})
    retirement = read_json(guide_retirement_receipt)
    recovery = read_json(epoch_recovery_receipt)
    continuity_authority = read_json(continuity_receipt)
    continuity_recovery = dict(continuity.get("epoch_recovery") or {})
    recovered_epoch = Path(
        str(recovery.get("recovered_epoch_checkpoint") or "")
    ).resolve()
    expected_latent = {
        "schema": "poke_bot.action_conditioned_latent_lookahead/v1",
        "enabled": True,
        "action_authority_enabled": True,
        "width": 512,
        "policy_aid_cap": 0.25,
        "parameters": 412130,
    }
    if (
        ready.get("schema") != READY_SCHEMA
        or ready.get("status") != "ready_for_managed_rl_registration"
        or ready.get("specialist_id") != SPECIALIST_ID
        or int(ready.get("epochs_completed", -1)) != 25
        or ready.get("checkpoint_selection") != "final_epoch_25"
        or ready.get("capacity_profile") != "H10-I/v1"
        or ready.get("decision_fusion_schema")
        != "poke_bot.causal_decision_fusion/v3"
        or int(ready.get("learned_head_count", -1)) != 19
        or int(ready.get("learned_route_count", -1)) != 19
        or float(ready.get("guide_weight", -1.0)) != 0.0
        or ready.get("guide_enabled") is not False
        or not checkpoint.is_file()
        or not checkpoint_digest.startswith("sha256:")
        or sha256(checkpoint) != checkpoint_digest
        or ready.get("initial_checkpoint_sha256")
        != str(trigger_checkpoint.get("sha256") or "")
        or pilot.get("importance_index_sha256") != sha256(importance_index)
        or int(pilot.get("matched_top_100_train_games", -1)) != 33156
        or float(pilot.get("effective_training_weight_mass", -1.0)) != 237330.0
        or postupload.get("schema") != "poke_bot.marnie_postupload_weighted_bootstrap/v1"
        or postupload.get("status") != "validated_ready_for_weighted_bootstrap"
        or latent != expected_latent
        or Path(str(continuity.get("path") or "")).resolve()
        != continuity_receipt.resolve()
        or continuity.get("sha256") != sha256(continuity_receipt)
        or int(continuity.get("accepted_policy_generation", -1)) != 15
        or dict(continuity.get("required_checkpoint_inventory") or {})
        != expected_latent
        or continuity_authority.get("owner_decision_revision") != 140
        or (continuity_authority.get("guide_override") or {}).get("long_term_retired") is not True
        or float((continuity_authority.get("guide_override") or {}).get("weight", -1.0)) != 0.0
        or retirement.get("schema") != "poke_bot.marnie_guide_retirement/v1"
        or retirement.get("owner_revision") != 140
        or float(retirement.get("guide_weight", -1.0)) != 0.0
        or retirement.get("guide_target_generation_required") is not False
        or retirement.get("guide_conditioned_losses_enabled") is not False
        or retirement.get("guide_action_influence") is not False
        or Path(str(continuity_recovery.get("path") or "")).resolve()
        != epoch_recovery_receipt.resolve()
        or continuity_recovery.get("sha256") != sha256(epoch_recovery_receipt)
        or recovery.get("schema")
        != "poke_bot.marnie_postupload_epoch_recovery/v1"
        or recovery.get("status") != "validated_resume_without_retraining"
        or int(recovery.get("owner_revision", -1)) != 141
        or recovery.get("latent_policy_continuity_receipt_sha256")
        != sha256(continuity_receipt)
        or not recovered_epoch.is_file()
        or recovery.get("recovered_epoch_checkpoint_sha256")
        != sha256(recovered_epoch)
        or float(recovery.get("guide_weight", -1.0)) != 0.0
        or recovery.get("guide_enabled") is not False
        or recovery.get("patched_bootstrap_script_sha256")
        != sha256(ROOT / str(recovery.get("patched_bootstrap_script") or ""))
        or recovery.get("patched_expanded_validator_sha256")
        != sha256(ROOT / str(recovery.get("patched_expanded_validator") or ""))
    ):
        raise RuntimeError("post-upload epoch-25 bootstrap ready receipt is invalid")

    request = read_json(request_path)
    validate_activation_request(
        request,
        expected_learner_digest=str(trigger_checkpoint.get("sha256") or ""),
    )
    migration = read_json(migration_path)
    validate_migration_receipt(migration, request_path=request_path)
    for key, path in (
        ("trigger", trigger_path),
        ("activation_request", request_path),
        ("activation_migration", migration_path),
    ):
        row = dict(postupload.get(key) or {})
        if (
            Path(str(row.get("path") or "")).resolve() != path.resolve()
            or str(row.get("sha256") or "") != family_sha256(path)
        ):
            raise RuntimeError(f"bootstrap receipt binds another {key}")
    return ready, checkpoint, checkpoint_digest


def stage(args: argparse.Namespace) -> dict[str, Any]:
    if args.receipt.is_file():
        existing = read_json(args.receipt)
        if existing.get("schema") != SCHEMA or existing.get("status") != "queued":
            raise RuntimeError("existing post-upload bootstrap submission is invalid")
        return existing
    ready, checkpoint, checkpoint_digest = validate_ready(
        ready_path=args.ready,
        trigger_path=args.trigger,
        request_path=args.activation_request,
        migration_path=args.activation_migration,
        importance_index=args.pilot_importance_index,
        continuity_receipt=args.latent_policy_continuity_receipt,
        guide_retirement_receipt=args.guide_retirement_receipt,
        epoch_recovery_receipt=args.epoch_recovery_receipt,
    )
    deck = materialize_pinned_specialist_deck(
        run_dir=args.run_dir,
        representatives_path=args.representatives,
        archetype=SPECIALIST_ID,
        output_path=args.submission_root / "pinned-marnie-postupload-bootstrap.deck.csv",
    )
    bundle = build_submission_bundle(
        repo_root=args.runtime_root,
        frozen_manifest={
            "model_path": str(checkpoint),
            "checkpoint_digest": checkpoint_digest,
        },
        deck_receipt=deck,
        output_dir=args.submission_root / "build",
        python=args.python,
        archetype=SPECIALIST_ID,
        matchup_tree=args.matchup_tree,
        turn_order_preference="first_if_allowed",
    )
    copy = _copy_submission_slot(bundle, args.submission_root, 1)
    queued = queue_submission_copies(
        queue_path=args.queue,
        copies=[copy],
        gate_plan={
            "checkpoint_digest": checkpoint_digest,
            "gate_id": "marnie-postupload-expert-bootstrap-r135",
            "iteration": 9,
            "completion_authority": "postupload_expert_bootstrap_epoch25",
        },
        specialist_id=SPECIALIST_ID,
        competition="pokemon-tcg-ai-battle",
    )
    payload = {
        "schema": SCHEMA,
        "status": "queued",
        "owner_revisions": [130, 134, 135, 136, 137, 138, 139, 140, 141],
        "ready": {"path": str(args.ready), "sha256": sha256(args.ready)},
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_digest,
        "initial_uploaded_iteration9_checkpoint_sha256": ready[
            "initial_checkpoint_sha256"
        ],
        "pilot_importance_index": {
            "path": str(args.pilot_importance_index),
            "sha256": sha256(args.pilot_importance_index),
        },
        "latent_policy": ready["latent_policy"],
        "latent_policy_continuity": ready["latent_policy_continuity"],
        "guide_retirement": {
            "path": str(args.guide_retirement_receipt),
            "sha256": sha256(args.guide_retirement_receipt),
            "weight": 0.0,
            "long_term_retired": True,
        },
        "epoch_recovery": {
            "path": str(args.epoch_recovery_receipt),
            "sha256": sha256(args.epoch_recovery_receipt),
            "recovered_epoch": 1,
            "retrained": False,
        },
        "bundle": bundle,
        "queue_entry": queued[0],
        "self_play_authority": False,
        "training_stop_or_freeze_authority": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_once(args.receipt, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--representatives", type=Path, required=True)
    parser.add_argument("--matchup-tree", type=Path, required=True)
    parser.add_argument("--submission-root", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--trigger", type=Path, required=True)
    parser.add_argument("--activation-request", type=Path, required=True)
    parser.add_argument("--activation-migration", type=Path, required=True)
    parser.add_argument("--pilot-importance-index", type=Path, required=True)
    parser.add_argument(
        "--latent-policy-continuity-receipt", type=Path, required=True
    )
    parser.add_argument("--guide-retirement-receipt", type=Path, required=True)
    parser.add_argument("--epoch-recovery-receipt", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    result = stage(args)
    print(json.dumps({"status": result["status"], "receipt": str(args.receipt)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
