from __future__ import annotations

import json
from pathlib import Path
from argparse import Namespace
from types import SimpleNamespace

import pytest

from scripts.materialize_marnie_iteration9_upload_trigger import (
    QUEUE_IDENTITY_FIELDS,
    sha256,
)
from scripts.materialize_marnie_postupload_bootstrap_upload import materialize
import scripts.stage_marnie_postupload_bootstrap_submission as stage_module
import scripts.activate_marnie_postupload_bootstrap as activate_module


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_postupload_submission_ready_binds_exact_parent_weights_and_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "epoch25.pt"
    checkpoint.write_bytes(b"epoch25")
    parent = tmp_path / "iter9.pt"
    parent.write_bytes(b"iter9")
    trigger = write_json(
        tmp_path / "trigger.json",
        {"bindings": {"checkpoint": {"path": str(parent), "sha256": sha256(parent)}}},
    )
    request = write_json(tmp_path / "request.json", {})
    migration = write_json(tmp_path / "migration.json", {})
    index = write_json(tmp_path / "importance.json", {"ready": True})
    continuity = write_json(
        tmp_path / "continuity.json",
        {
            "owner_decision_revision": 140,
            "guide_override": {"long_term_retired": True, "weight": 0.0},
        },
    )
    retirement = write_json(
        tmp_path / "retirement.json",
        {
            "schema": "poke_bot.marnie_guide_retirement/v1",
            "owner_revision": 140,
            "guide_weight": 0.0,
            "guide_target_generation_required": False,
            "guide_conditioned_losses_enabled": False,
            "guide_action_influence": False,
        },
    )
    recovered_epoch = tmp_path / "epoch01.pt"
    recovered_epoch.write_bytes(b"epoch01")
    recovery = write_json(
        tmp_path / "recovery.json",
        {
            "schema": "poke_bot.marnie_postupload_epoch_recovery/v1",
            "status": "validated_resume_without_retraining",
            "owner_revision": 141,
            "latent_policy_continuity_receipt_sha256": sha256(continuity),
            "recovered_epoch_checkpoint": str(recovered_epoch),
            "recovered_epoch_checkpoint_sha256": sha256(recovered_epoch),
            "guide_weight": 0.0,
            "guide_enabled": False,
            "patched_bootstrap_script": "scripts/run_final_format_marnie_h10_bootstrap.py",
            "patched_bootstrap_script_sha256": sha256(
                stage_module.ROOT / "scripts/run_final_format_marnie_h10_bootstrap.py"
            ),
            "patched_expanded_validator": "scripts/run_starmie_expert_bootstrap.py",
            "patched_expanded_validator_sha256": sha256(
                stage_module.ROOT / "scripts/run_starmie_expert_bootstrap.py"
            ),
        },
    )
    latent = {
        "schema": "poke_bot.action_conditioned_latent_lookahead/v1",
        "enabled": True,
        "action_authority_enabled": True,
        "width": 512,
        "policy_aid_cap": 0.25,
        "parameters": 412130,
    }
    postupload = {
        "schema": "poke_bot.marnie_postupload_weighted_bootstrap/v1",
        "status": "validated_ready_for_weighted_bootstrap",
        "trigger": {"path": str(trigger), "sha256": sha256(trigger)},
        "activation_request": {"path": str(request), "sha256": sha256(request)},
        "activation_migration": {
            "path": str(migration),
            "sha256": sha256(migration),
        },
    }
    ready = write_json(
        tmp_path / "ready.json",
        {
            "schema": "poke_bot.final_format_marnie_h10_bootstrap_ready/v1",
            "status": "ready_for_managed_rl_registration",
            "specialist_id": "marnie-s-grimmsnarl-ex",
            "epochs_completed": 25,
            "checkpoint_selection": "final_epoch_25",
            "capacity_profile": "H10-I/v1",
            "decision_fusion_schema": "poke_bot.causal_decision_fusion/v3",
            "learned_head_count": 19,
            "learned_route_count": 19,
            "guide_weight": 0.0,
            "guide_enabled": False,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "initial_checkpoint_sha256": sha256(parent),
            "expert_pilot_importance": {
                "importance_index_sha256": sha256(index),
                "matched_top_100_train_games": 33156,
                "effective_training_weight_mass": 237330.0,
            },
            "post_upload_new_system": postupload,
            "latent_policy": latent,
            "latent_policy_continuity": {
                "path": str(continuity),
                "sha256": sha256(continuity),
                "accepted_policy_generation": 15,
                "required_checkpoint_inventory": latent,
                "epoch_recovery": {
                    "path": str(recovery),
                    "sha256": sha256(recovery),
                },
            },
        },
    )
    monkeypatch.setattr(
        stage_module, "validate_iteration9_upload_trigger", lambda _x: {"valid": True}
    )
    monkeypatch.setattr(
        stage_module, "validate_activation_request", lambda *_a, **_k: {"valid": True}
    )
    monkeypatch.setattr(
        stage_module, "validate_migration_receipt", lambda *_a, **_k: {}
    )
    result, selected, digest = stage_module.validate_ready(
        ready_path=ready,
        trigger_path=trigger,
        request_path=request,
        migration_path=migration,
        importance_index=index,
        continuity_receipt=continuity,
        guide_retirement_receipt=retirement,
        epoch_recovery_receipt=recovery,
    )
    assert result["initial_checkpoint_sha256"] == sha256(parent)
    assert selected == checkpoint
    assert digest == sha256(checkpoint)


def test_bootstrap_upload_receipt_is_fail_closed_until_exact_success(tmp_path: Path) -> None:
    checkpoint = tmp_path / "epoch25.pt"
    checkpoint.write_bytes(b"epoch25")
    ready = write_json(tmp_path / "ready.json", {"status": "ready"})
    bundle = tmp_path / "submission.tar.gz"
    bundle.write_bytes(b"bundle")
    uploaded = bundle.resolve()
    auth = write_json(
        tmp_path / "auth.json",
        {
            "schema": "poke_bot.kaggle_submission_authorization/v1",
            "nonce": "nonce-25",
            "consumed_before_upload": True,
            "remaining_uses": 0,
            "submission_file_checksum": sha256(uploaded),
            "frozen_checkpoint_checksum": sha256(checkpoint),
        },
    )
    attempts = tmp_path / "attempts"
    attempt = write_json(
        attempts / "attempt.json",
        {
            "schema": "poke_bot.kaggle_submission_attempt/v1",
            "nonce": "nonce-25",
            "returncode": 0,
            "authorization_consumed": str(auth),
            "identity": {
                "file": str(uploaded),
                "file_sha256": sha256(uploaded),
                "competition": "pokemon-tcg-ai-battle",
                "message": "marnie postupload bootstrap epoch25",
            },
        },
    )
    assert attempt.is_file()
    identity = {field: None for field in QUEUE_IDENTITY_FIELDS}
    identity.update(
        {
            "specialist_id": "marnie-s-grimmsnarl-ex",
            "copy_number": 1,
            "turn_order_preference": "first_if_allowed",
            "label": "marnie postupload bootstrap epoch25",
            "checkpoint_checksum": sha256(checkpoint),
            "model_checksum": sha256(checkpoint),
            "competition": "pokemon-tcg-ai-battle",
            "file": str(uploaded),
            "file_sha256": sha256(uploaded),
            "iteration": 9,
        }
    )
    stage = write_json(
        tmp_path / "submission.json",
        {
            "schema": "poke_bot.marnie_postupload_bootstrap_submission/v1",
            "status": "queued",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "ready": {"path": str(ready), "sha256": sha256(ready)},
            "bundle": {"path": str(bundle), "sha256": sha256(bundle)},
            "queue_entry": dict(identity),
        },
    )
    queue_path = write_json(
        tmp_path / "queue.json",
        {
            "schema": "poke_bot.kaggle_submission_queue/v1",
            "queue": [
                {
                    **identity,
                    "queue_status": "submitted",
                    "kaggle_status": "RUNNING",
                    "failure_reason": None,
                    "submission_id": None,
                    "one_shot_authorization_nonce": "nonce-25",
                }
            ],
        },
    )
    output = tmp_path / "upload.json"
    pending = materialize(
        queue_path=queue_path,
        submission_receipt=stage,
        attempt_receipts=attempts,
        output=output,
    )
    assert pending == {"status": "not_ready", "reason": "exact_upload_not_complete"}
    assert not output.exists()

    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    queue["queue"][0].update(
        {
            "queue_status": "accepted",
            "kaggle_status": "COMPLETE",
            "submission_id": 999,
        }
    )
    write_json(queue_path, queue)
    complete = materialize(
        queue_path=queue_path,
        submission_receipt=stage,
        attempt_receipts=attempts,
        output=output,
    )
    assert complete["status"] == "materialized"
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["first_new_system_self_play_authorized"] is True
    assert receipt["checkpoint"]["sha256"] == sha256(checkpoint)


def test_successful_upload_atomically_reparents_iter10_to_bootstrap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    iter9 = tmp_path / "iter_00009.pt"
    iter9.write_bytes(b"iter9")
    bootstrap = tmp_path / "epoch25.pt"
    bootstrap.write_bytes(b"epoch25")
    latent = {
        "schema": "poke_bot.action_conditioned_latent_lookahead/v1",
        "enabled": True,
        "action_authority_enabled": True,
        "width": 512,
        "policy_aid_cap": 0.25,
        "parameters": 412130,
    }
    ready = write_json(
        tmp_path / "ready.json",
        {
            "latent_policy": latent,
            "latent_policy_continuity": {"accepted_policy_generation": 15},
            "guide_weight": 0.0,
            "guide_enabled": False,
        },
    )
    retirement = write_json(
        tmp_path / "retirement.json",
        {
            "schema": "poke_bot.marnie_guide_retirement/v1",
            "owner_revision": 140,
            "guide_weight": 0.0,
            "guide_target_generation_required": False,
            "guide_conditioned_losses_enabled": False,
            "guide_action_influence": False,
        },
    )
    recovered_epoch = tmp_path / "epoch01.pt"
    recovered_epoch.write_bytes(b"epoch01")
    recovery = write_json(
        tmp_path / "recovery.json",
        {
            "schema": "poke_bot.marnie_postupload_epoch_recovery/v1",
            "status": "validated_resume_without_retraining",
            "owner_revision": 141,
            "recovered_epoch_checkpoint": str(recovered_epoch),
            "recovered_epoch_checkpoint_sha256": sha256(recovered_epoch),
            "guide_weight": 0.0,
            "guide_enabled": False,
        },
    )
    ready_payload = json.loads(ready.read_text(encoding="utf-8"))
    ready_payload["latent_policy_continuity"]["epoch_recovery"] = {
        "path": str(recovery),
        "sha256": sha256(recovery),
    }
    write_json(ready, ready_payload)
    source_submission = write_json(
        tmp_path / "submission.json",
        {
            "initial_uploaded_iteration9_checkpoint_sha256": sha256(iter9),
            "ready": {"path": str(ready), "sha256": sha256(ready)},
            "epoch_recovery": {
                "path": str(recovery),
                "sha256": sha256(recovery),
            },
        },
    )
    upload = write_json(
        tmp_path / "upload.json",
        {
            "schema": "poke_bot.marnie_postupload_bootstrap_upload/v1",
            "status": "successful_upload",
            "first_new_system_self_play_authorized": True,
            "checkpoint": {"path": str(bootstrap), "sha256": sha256(bootstrap)},
            "source_submission": {
                "path": str(source_submission),
                "sha256": sha256(source_submission),
            },
        },
    )
    loop = write_json(
        tmp_path / "loop_state.json",
        {
            "last_completed_iteration": 9,
            "next_iteration": 10,
            "learner": {"path": str(iter9), "digest": sha256(iter9)},
            "heldout_champion": {"path": str(iter9), "digest": sha256(iter9)},
            "heldout_champion_evidence": {
                "checkpoint_digest": sha256(iter9),
                "games": 4250,
            },
        },
    )
    runtime_registry = write_json(
        tmp_path / "runtime-r142.json",
        {
            "specialists": {
                "marnie-s-grimmsnarl-ex": {
                    "guide_loss_weight": 0.0,
                    "guide_retired": True,
                    "guide_shadow_only": True,
                    "guide_shadow_blocking": False,
                    "guide_shadow_runtime_authority": False,
                    "guide_target_generation_required": False,
                    "guide_conditioned_losses_enabled": False,
                    "guide_action_influence": False,
                }
            }
        },
    )
    runtime_receipt = write_json(
        tmp_path / "runtime-r142-receipt.json",
        {
            "schema": "poke_bot.marnie_family_guide_shadow_runtime/v1",
            "status": "active_next_start_overlay",
            "owner_revision": 142,
            "merged_registry": {
                "path": str(runtime_registry.resolve()),
                "sha256": sha256(runtime_registry),
            },
            "environment_drop_in": {
                "path": str((tmp_path / "guide-shadow.conf").resolve()),
                "sha256": "pending",
            },
            "proof": {
                "family_and_typed_loss_system_preserved": True,
                "latent_policy_and_fusion_preserved": True,
                "all_non_guide_registry_fields_unchanged": True,
                "guide_blocking_authority": False,
            },
        },
    )
    overlay = tmp_path / "guide-shadow.conf"
    overlay.write_text(
        f"[Service]\nExecStart=/python --registry {runtime_registry.resolve()}\n",
        encoding="utf-8",
    )
    runtime_receipt_payload = json.loads(runtime_receipt.read_text(encoding="utf-8"))
    runtime_receipt_payload["environment_drop_in"]["sha256"] = sha256(overlay)
    write_json(runtime_receipt, runtime_receipt_payload)
    base_chain = write_json(tmp_path / "state" / "base-chain.json", {"base": True})
    stage_script = tmp_path / "scripts" / "stage.py"
    resolver_script = tmp_path / "scripts" / "resolve.py"
    activation_unit = tmp_path / "deploy" / "activation.service"
    resolver_unit = tmp_path / "deploy" / "resolver.service"
    for path, body in (
        (stage_script, "stage\n"),
        (resolver_script, "resolve\n"),
        (activation_unit, "activation\n"),
        (resolver_unit, "resolver\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    running_activation = Path(activate_module.__file__).resolve()
    runtime_chain = write_json(
        tmp_path / "state" / "runtime-chain.json",
        {
            "schema": "poke_bot.marnie_family_guide_shadow_chain/v1",
            "status": "armed_fail_closed",
            "owner_revision": 142,
            "base_epoch_recovery_chain": str(base_chain),
            "base_epoch_recovery_chain_sha256": sha256(base_chain),
            "runtime_receipt": str(runtime_receipt.resolve()),
            "runtime_receipt_sha256": sha256(runtime_receipt),
            "stage_script": str(stage_script.resolve()),
            "stage_script_sha256": sha256(stage_script),
            "activation_script": str(running_activation),
            "activation_script_sha256": sha256(running_activation),
            "monitor_resolution_script": str(resolver_script.resolve()),
            "monitor_resolution_script_sha256": sha256(resolver_script),
            "activation_unit": str(activation_unit.resolve()),
            "activation_unit_sha256": sha256(activation_unit),
            "monitor_resolution_unit": str(resolver_unit.resolve()),
            "monitor_resolution_unit_sha256": sha256(resolver_unit),
        },
    )
    monkeypatch.setattr(activate_module, "require_trainer_inactive", lambda _s: "inactive")
    monkeypatch.setattr(
        activate_module.checkpoint,
        "load_checkpoint",
        lambda *_a, **_k: {
            "archetype_id": "marnie-s-grimmsnarl-ex",
            "extra": {
                "expert_rehearsal": {
                    "expert_pilot_importance": {
                        "matched_top_100_train_games": 33156,
                        "effective_training_weight_mass": 237330.0,
                    }
                }
            },
        },
    )
    monkeypatch.setattr(
        activate_module,
        "load_model_from_checkpoint",
        lambda *_a, **_k: SimpleNamespace(
            cfg=SimpleNamespace(
                h10_capacity_enabled=True,
                latent_lookahead_width=512,
            ),
            decision_fusion_inventory=lambda: {
                "schema": "poke_bot.causal_decision_fusion/v3",
                "required_heads": [f"head-{i}" for i in range(19)],
            },
            # Match the real runtime inventory: width is owned by model cfg,
            # while implementation attestations are inventory-only metadata.
            latent_lookahead_inventory=lambda: {
                key: value for key, value in latent.items() if key != "width"
            }
            | {
                "neural_only": True,
                "single_forward_pass": True,
                "mcts_allowed": False,
            },
        ),
    )
    backup = tmp_path / "loop_state.before.json"
    activation_receipt = tmp_path / "activation.json"
    result = activate_module.activate(
        Namespace(
            service="trainer.service",
            upload_receipt=upload,
            loop_state=loop,
            loop_state_backup=backup,
            guide_retirement_receipt=retirement,
            epoch_recovery_receipt=recovery,
            runtime_registry=runtime_registry,
            guide_shadow_runtime_receipt=runtime_receipt,
            guide_shadow_chain_receipt=runtime_chain,
            receipt=activation_receipt,
        )
    )
    updated = json.loads(loop.read_text(encoding="utf-8"))
    expected = {"path": str(bootstrap), "digest": sha256(bootstrap)}
    assert updated["learner"] == expected
    assert updated["heldout_champion"] == expected
    assert updated["heldout_champion_evidence"] == {}
    assert updated["next_iteration"] == 10
    assert result["status"] == "activated"
    assert backup.is_file() and activation_receipt.is_file()
