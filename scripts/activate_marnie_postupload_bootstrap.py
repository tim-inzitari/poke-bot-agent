#!/usr/bin/env python3
"""Activate the successfully uploaded epoch-25 bootstrap at iter9→10."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import checkpoint  # noqa: E402
from poke_bot.train import load_model_from_checkpoint  # noqa: E402


SCHEMA = "poke_bot.marnie_postupload_bootstrap_activation/v1"
UPLOAD_SCHEMA = "poke_bot.marnie_postupload_bootstrap_upload/v1"


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


def exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def require_trainer_inactive(service: str) -> str:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", service],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    state = result.stdout.strip() or "unknown"
    if state in {"active", "activating", "reloading"}:
        raise RuntimeError(f"managed trainer is still {state}")
    return state


def activate(args: argparse.Namespace) -> dict[str, Any]:
    upload_path = args.upload_receipt.expanduser().resolve()
    loop_path = args.loop_state.expanduser().resolve()
    backup_path = args.loop_state_backup.expanduser().resolve()
    receipt_path = args.receipt.expanduser().resolve()
    retirement_path = args.guide_retirement_receipt.expanduser().resolve()
    recovery_path = args.epoch_recovery_receipt.expanduser().resolve()
    runtime_path = args.runtime_registry.expanduser().resolve()
    runtime_receipt_path = args.guide_shadow_runtime_receipt.expanduser().resolve()
    runtime_chain_path = args.guide_shadow_chain_receipt.expanduser().resolve()
    if receipt_path.is_file():
        existing = read_json(receipt_path)
        if existing.get("schema") != SCHEMA or existing.get("status") != "activated":
            raise RuntimeError("existing post-upload activation receipt is invalid")
        return existing
    if backup_path.exists():
        raise RuntimeError("loop-state backup exists without activation receipt")
    service_state = require_trainer_inactive(args.service)
    runtime_chain = read_json(runtime_chain_path)
    canonical_root = runtime_chain_path.parent.parent
    chain_files = {
        "base_epoch_recovery_chain": "base_epoch_recovery_chain_sha256",
        "runtime_receipt": "runtime_receipt_sha256",
        "stage_script": "stage_script_sha256",
        "activation_script": "activation_script_sha256",
        "monitor_resolution_script": "monitor_resolution_script_sha256",
        "activation_unit": "activation_unit_sha256",
        "monitor_resolution_unit": "monitor_resolution_unit_sha256",
    }
    if (
        runtime_chain.get("schema")
        != "poke_bot.marnie_family_guide_shadow_chain/v1"
        or runtime_chain.get("status") != "armed_fail_closed"
        or int(runtime_chain.get("owner_revision", -1)) != 142
    ):
        raise RuntimeError("iteration-10 family/guide-shadow chain is invalid")
    for path_key, digest_key in chain_files.items():
        bound = (canonical_root / str(runtime_chain.get(path_key) or "")).resolve()
        if not bound.is_file() or runtime_chain.get(digest_key) != sha256(bound):
            raise RuntimeError(f"iteration-10 chain changed: {path_key}")
    if (
        sha256(Path(__file__).resolve())
        != runtime_chain.get("activation_script_sha256")
        or (canonical_root / str(runtime_chain["runtime_receipt"])).resolve()
        != runtime_receipt_path
    ):
        raise RuntimeError("running activation bytes differ from armed chain")
    runtime_receipt = read_json(runtime_receipt_path)
    runtime_registry = read_json(runtime_path)
    runtime_row = dict(
        (runtime_registry.get("specialists") or {}).get(
            "marnie-s-grimmsnarl-ex"
        )
        or {}
    )
    bound_runtime = dict(runtime_receipt.get("merged_registry") or {})
    bound_overlay = dict(runtime_receipt.get("environment_drop_in") or {})
    overlay_path = Path(str(bound_overlay.get("path") or "")).resolve()
    runtime_proof = dict(runtime_receipt.get("proof") or {})
    if (
        runtime_receipt.get("schema")
        != "poke_bot.marnie_family_guide_shadow_runtime/v1"
        or runtime_receipt.get("status") != "active_next_start_overlay"
        or int(runtime_receipt.get("owner_revision", -1)) != 142
        or Path(str(bound_runtime.get("path") or "")).resolve() != runtime_path
        or bound_runtime.get("sha256") != sha256(runtime_path)
        or not overlay_path.is_file()
        or bound_overlay.get("sha256") != sha256(overlay_path)
        or f"--registry {runtime_path}" not in overlay_path.read_text(
            encoding="utf-8"
        )
        or float(runtime_row.get("guide_loss_weight", -1.0)) != 0.0
        or runtime_row.get("guide_retired") is not True
        or runtime_row.get("guide_shadow_only") is not True
        or runtime_row.get("guide_shadow_blocking") is not False
        or runtime_row.get("guide_shadow_runtime_authority") is not False
        or runtime_row.get("guide_target_generation_required") is not False
        or runtime_row.get("guide_conditioned_losses_enabled") is not False
        or runtime_row.get("guide_action_influence") is not False
        or runtime_proof.get("family_and_typed_loss_system_preserved") is not True
        or runtime_proof.get("latent_policy_and_fusion_preserved") is not True
        or runtime_proof.get("all_non_guide_registry_fields_unchanged") is not True
        or runtime_proof.get("guide_blocking_authority") is not False
    ):
        raise RuntimeError("iteration-10 runtime lost family/guide-shadow merge")
    upload = read_json(upload_path)
    checkpoint_row = dict(upload.get("checkpoint") or {})
    bootstrap = Path(str(checkpoint_row.get("path") or "")).resolve()
    bootstrap_digest = str(checkpoint_row.get("sha256") or "")
    if (
        upload.get("schema") != UPLOAD_SCHEMA
        or upload.get("status") != "successful_upload"
        or upload.get("first_new_system_self_play_authorized") is not True
        or not bootstrap.is_file()
        or sha256(bootstrap) != bootstrap_digest
    ):
        raise RuntimeError("successful bootstrap upload authority is invalid")

    payload = checkpoint.load_checkpoint(bootstrap, map_location="cpu")
    model = load_model_from_checkpoint(bootstrap, device=torch.device("cpu"))
    fusion = model.decision_fusion_inventory()
    latent_inventory = model.latent_lookahead_inventory()
    # The runtime inventory intentionally carries additional implementation
    # attestations and older inventories did not repeat the configured width.
    # Activation compares the stable checkpoint contract, so canonicalize that
    # exact six-field projection and recover width from the loaded model config
    # when the inventory omits it.
    latent = {
        "schema": latent_inventory.get("schema"),
        "enabled": latent_inventory.get("enabled"),
        "action_authority_enabled": latent_inventory.get(
            "action_authority_enabled"
        ),
        "width": latent_inventory.get(
            "width", getattr(model.cfg, "latent_lookahead_width", None)
        ),
        "policy_aid_cap": latent_inventory.get("policy_aid_cap"),
        "parameters": latent_inventory.get("parameters"),
    }
    pilot = dict(
        ((payload.get("extra") or {}).get("expert_rehearsal") or {}).get(
            "expert_pilot_importance"
        )
        or {}
    )
    if (
        payload.get("archetype_id") != "marnie-s-grimmsnarl-ex"
        or model.cfg.h10_capacity_enabled is not True
        or fusion.get("schema") != "poke_bot.causal_decision_fusion/v3"
        or len(fusion.get("required_heads") or ()) != 19
        or int(pilot.get("matched_top_100_train_games", -1)) != 33156
        or float(pilot.get("effective_training_weight_mass", -1.0)) != 237330.0
        or latent.get("schema")
        != "poke_bot.action_conditioned_latent_lookahead/v1"
        or latent.get("enabled") is not True
        or latent.get("action_authority_enabled") is not True
        or int(latent.get("width", -1)) != 512
        or float(latent.get("policy_aid_cap", -1.0)) != 0.25
        or int(latent.get("parameters", -1)) != 412130
    ):
        raise RuntimeError("uploaded bootstrap checkpoint contract changed")

    loop = read_json(loop_path)
    learner = dict(loop.get("learner") or {})
    source_submission = read_json(
        Path(str((upload.get("source_submission") or {}).get("path") or ""))
    )
    ready_row = dict(source_submission.get("ready") or {})
    ready_path = Path(str(ready_row.get("path") or "")).resolve()
    ready = read_json(ready_path)
    retirement = read_json(retirement_path)
    recovery = read_json(recovery_path)
    source_recovery = dict(source_submission.get("epoch_recovery") or {})
    continuity_recovery = dict(
        (ready.get("latent_policy_continuity") or {}).get("epoch_recovery") or {}
    )
    recovered_epoch = Path(
        str(recovery.get("recovered_epoch_checkpoint") or "")
    ).resolve()
    if (
        sha256(ready_path) != str(ready_row.get("sha256") or "")
        or dict(ready.get("latent_policy") or {}) != latent
        or float(ready.get("guide_weight", -1.0)) != 0.0
        or ready.get("guide_enabled") is not False
        or retirement.get("schema") != "poke_bot.marnie_guide_retirement/v1"
        or retirement.get("owner_revision") != 140
        or float(retirement.get("guide_weight", -1.0)) != 0.0
        or retirement.get("guide_target_generation_required") is not False
        or retirement.get("guide_conditioned_losses_enabled") is not False
        or retirement.get("guide_action_influence") is not False
        or Path(str(source_recovery.get("path") or "")).resolve()
        != recovery_path
        or source_recovery.get("sha256") != sha256(recovery_path)
        or Path(str(continuity_recovery.get("path") or "")).resolve()
        != recovery_path
        or continuity_recovery.get("sha256") != sha256(recovery_path)
        or recovery.get("schema")
        != "poke_bot.marnie_postupload_epoch_recovery/v1"
        or recovery.get("status") != "validated_resume_without_retraining"
        or int(recovery.get("owner_revision", -1)) != 141
        or not recovered_epoch.is_file()
        or recovery.get("recovered_epoch_checkpoint_sha256")
        != sha256(recovered_epoch)
        or float(recovery.get("guide_weight", -1.0)) != 0.0
        or recovery.get("guide_enabled") is not False
        or int(
            (ready.get("latent_policy_continuity") or {}).get(
                "accepted_policy_generation", -1
            )
        )
        != 15
    ):
        raise RuntimeError("bootstrap upload lost latent-policy continuity evidence")
    iter9_digest = str(
        source_submission.get("initial_uploaded_iteration9_checkpoint_sha256") or ""
    )
    if (
        int(loop.get("last_completed_iteration", -1)) != 9
        or int(loop.get("next_iteration", -1)) != 10
        or learner.get("digest") != iter9_digest
        or not Path(str(learner.get("path") or "")).is_file()
        or sha256(Path(str(learner["path"]))) != iter9_digest
    ):
        raise RuntimeError("activation is not at the exact iter9→10 learner boundary")

    exclusive_json(backup_path, loop)
    now = datetime.now(timezone.utc).isoformat()
    updated = copy.deepcopy(loop)
    identity = {"path": str(bootstrap), "digest": bootstrap_digest}
    updated["learner"] = dict(identity)
    updated["heldout_champion"] = dict(identity)
    # The uploaded bootstrap has not yet played the formal heldout gate.  The
    # prior champion's evidence must remain historical and cannot be attached
    # to the replacement checkpoint merely because the pointer changed.
    updated["heldout_champion_evidence"] = {}
    if "champion" in updated:
        updated["champion"] = dict(identity)
    updated["postupload_bootstrap_activation"] = {
        "schema": SCHEMA,
        "receipt": str(receipt_path),
        "upload_receipt": str(upload_path),
        "upload_receipt_sha256": sha256(upload_path),
        "source_iteration9_learner_sha256": iter9_digest,
        "activated_bootstrap_sha256": bootstrap_digest,
        "epochs": 25,
        "next_iteration": 10,
    }
    updated["updated_at_utc"] = now
    atomic_json(loop_path, updated)

    receipt = {
        "schema": SCHEMA,
        "status": "activated",
        "activated_at_utc": now,
        "owner_revisions": [130, 134, 135, 136, 137, 138, 139, 140, 141, 142],
        "managed_trainer_state_before_activation": service_state,
        "boundary": {"last_completed_iteration": 9, "next_iteration": 10},
        "source_iteration9_learner": dict(learner),
        "activated_bootstrap": dict(identity),
        "upload_receipt": {"path": str(upload_path), "sha256": sha256(upload_path)},
        "guide_retirement_receipt": {
            "path": str(retirement_path),
            "sha256": sha256(retirement_path),
        },
        "epoch_recovery_receipt": {
            "path": str(recovery_path),
            "sha256": sha256(recovery_path),
        },
        "family_guide_shadow_runtime": {
            "chain_receipt": str(runtime_chain_path),
            "chain_receipt_sha256": sha256(runtime_chain_path),
            "receipt": str(runtime_receipt_path),
            "receipt_sha256": sha256(runtime_receipt_path),
            "registry": str(runtime_path),
            "registry_sha256": sha256(runtime_path),
        },
        "loop_state_backup": {"path": str(backup_path), "sha256": sha256(backup_path)},
        "proof": {
            "exact_25_epochs": True,
            "final_epoch_25_selected": True,
            "top100_weight_index_bound": True,
            "successful_checksum_exact_upload": True,
            "h10_fusion_v3_19_heads": True,
            "accepted_policy_generation_15_latent_authority": True,
            "marnie_guide_permanently_retired": True,
            "marnie_guide_weight": 0.0,
            "marnie_guide_shadow_only_nonblocking": True,
            "heldout_evidence_reset_until_new_exact_gate": True,
            "family_guide_shadow_registry_merged": True,
            "epoch_1_recovered_without_retraining": True,
            "first_new_system_self_play_authorized": True,
            "old_system_collection_authorized": False,
        },
    }
    exclusive_json(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", required=True)
    parser.add_argument("--upload-receipt", type=Path, required=True)
    parser.add_argument("--loop-state", type=Path, required=True)
    parser.add_argument("--loop-state-backup", type=Path, required=True)
    parser.add_argument("--guide-retirement-receipt", type=Path, required=True)
    parser.add_argument("--epoch-recovery-receipt", type=Path, required=True)
    parser.add_argument("--runtime-registry", type=Path, required=True)
    parser.add_argument("--guide-shadow-runtime-receipt", type=Path, required=True)
    parser.add_argument("--guide-shadow-chain-receipt", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    result = activate(parser.parse_args())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
