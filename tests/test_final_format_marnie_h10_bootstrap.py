from __future__ import annotations

import inspect
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from poke_bot.train import BatchMetrics, _merge_metrics, supervised_rehearsal_step
from scripts.run_final_format_marnie_h10_bootstrap import (
    _combo_rows,
    _directional_rows,
    _validate_latent_policy_checkpoint,
    _validate_latent_policy_continuity_receipt,
    _validate_post_upload_boundary,
)
from scripts.run_starmie_expert_bootstrap import validate_expanded_epoch_checkpoint
import scripts.run_final_format_marnie_h10_bootstrap as bootstrap_module
from scripts.register_final_format_marnie_h10_rl import (
    _materialize_adapter_authorization,
    _route_reliability_telemetry,
    _selector_env_is_authorized,
    _validate_runtime_assets,
    _validate_selected_bootstrap_training,
)
from poke_bot.matchup_adapter_activation import (
    validate_adapter_training_authorization,
)
from poke_bot.strategic_schedule import EXPANDED_HEAD_IDS


def test_epoch_validator_includes_family_residual_gradient_heads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_weights = {name: 0.0 for name in EXPANDED_HEAD_IDS}
    for name, weight in {
        "action_q": 0.1,
        "action_type": 0.05,
        "action_target": 0.025,
        "action_resource": 0.025,
        "action_utility": 0.05,
    }.items():
        base_weights[name] = weight
    residuals = {
        "core_setup_continuity": 0.015,
        "resource_attack_readiness": 0.015,
        "long_horizon_prize_pressure": 0.010416666666666668,
    }
    effective = dict(base_weights)
    effective["action_resource"] += residuals["resource_attack_readiness"]
    effective["resource_forecast"] += residuals["resource_attack_readiness"]
    effective["outcome_distribution"] += residuals[
        "long_horizon_prize_pressure"
    ]
    effective["remaining_turns"] += residuals["long_horizon_prize_pressure"]
    enabled = tuple(name for name, weight in base_weights.items() if weight > 0.0)
    gradient = [name for name, weight in effective.items() if weight > 0.0]
    metric_heads = {
        "labeled": {name: 1 for name in enabled},
        "total": {name: 1 for name in enabled},
        "losses": {name: 0.5 for name in enabled},
    }
    payload = {
        "extra": {
            "expanded_head_training": {
                "schema": "poke_bot.expanded_head_training/v1",
                "target_schema_version": "targets-v1",
                "target_schema_digest": "sha256:targets",
                "schedule_version": "schedule-v1",
                "schedule_digest": "sha256:schedule",
                "epoch": 1,
                "epochs_total": 25,
                "architecture_present_heads": list(EXPANDED_HEAD_IDS),
                "trained_heads": gradient,
                "gradient_enabled_heads": gradient,
                "runtime_enabled_heads": [],
                "loss_weights": base_weights,
                "effective_loss_weights": effective,
                "archetype_residual_loss_weights": residuals,
            }
        }
    }
    monkeypatch.setattr(
        bootstrap_module.checkpoint,
        "load_checkpoint",
        lambda *_a, **_k: payload,
    )
    contract = validate_expanded_epoch_checkpoint(
        tmp_path / "epoch_01.pt",
        plan=SimpleNamespace(epoch=1, enabled_heads=enabled, loss_weights=base_weights),
        identity={
            "target_schema": "targets-v1",
            "target_schema_digest": "sha256:targets",
            "schedule_schema": "schedule-v1",
            "schedule_digest": "sha256:schedule",
        },
        train_metrics={"expanded_head_metrics": metric_heads},
        validation_metrics={"expanded_head_metrics": metric_heads},
        archetype_residual_loss_weights=residuals,
    )
    assert set(contract["gradient_enabled_heads"]) == set(gradient)


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_postupload_bootstrap_binds_uploaded_iter9_and_atomic_family_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = tmp_path / "iter_00009.pt"
    child.write_bytes(b"exact-iter9")
    child_digest = _digest(child)
    selected = _write_json(
        tmp_path / "selected-loss.json",
        {"residual_objectives": {"terminal_hazard": 0.2}},
    )
    trigger = _write_json(
        tmp_path / "trigger.json",
        {
            "iteration": 9,
            "bindings": {
                "checkpoint": {"path": str(child), "sha256": child_digest}
            },
        },
    )
    request = _write_json(
        tmp_path / "request.json",
        {
            "trigger": {"path": str(trigger), "sha256": _digest(trigger)},
            "bindings": {
                "selected_loss_vector": {
                    "path": str(selected),
                    "sha256": _digest(selected),
                }
            },
        },
    )
    migration = _write_json(tmp_path / "migration.json", {})
    roles = _write_json(
        tmp_path / "roles.json",
        {
            "learned_head_count": 19,
            "learned_route_count": 19,
            "all_learned_heads_influence_actions": True,
        },
    )
    prestage = _write_json(
        tmp_path / "prestage.json",
        {
            "schema": "poke_bot.final_format_marnie_refresh_prestage/v1",
            "specialist_id": "marnie-s-grimmsnarl-ex",
            "final_capacity_profile": "H10-I/v1",
            "final_decision_fusion_schema": "poke_bot.causal_decision_fusion/v3",
            "guide_training_mode": "strategic_directional_v2",
        },
    )
    monkeypatch.setattr(
        bootstrap_module, "validate_iteration9_upload_trigger", lambda _x: {"valid": True}
    )
    monkeypatch.setattr(
        bootstrap_module,
        "validate_activation_request",
        lambda _x, expected_learner_digest=None: {
            "valid": expected_learner_digest == child_digest
        },
    )
    monkeypatch.setattr(
        bootstrap_module, "validate_migration_receipt", lambda *_a, **_k: {}
    )
    monkeypatch.setattr(
        bootstrap_module.checkpoint,
        "checkpoint_digest",
        lambda path: child_digest if Path(path) == child else _digest(Path(path)),
    )
    monkeypatch.setattr(
        bootstrap_module.checkpoint,
        "load_checkpoint",
        lambda *_a, **_k: {
            "archetype_id": "marnie-s-grimmsnarl-ex",
            "model_config": {
                "h10_capacity_enabled": True,
                "decision_fusion_typed_output_centered_routes_enabled": True,
                "latent_lookahead_enabled": True,
                "latent_lookahead_action_authority_enabled": True,
                "latent_lookahead_width": 512,
                "latent_lookahead_policy_aid_cap": 0.25,
            },
            "model_state_dict": {
                "latent_lookahead.inventory": torch.zeros(412_130),
            },
            "provenance": {
                "decision_fusion": {
                    "schema": "poke_bot.causal_decision_fusion/v3",
                    "required_heads": [f"head-{i}" for i in range(19)],
                }
            },
        },
    )
    monkeypatch.setattr(
        bootstrap_module,
        "canonical_residual_weights",
        lambda values: {str(k): float(v) for k, v in dict(values).items()},
    )

    _prestage, parent, postupload, residuals = _validate_post_upload_boundary(
        prestage_path=prestage,
        child_path=child,
        roles_path=roles,
        trigger_path=trigger,
        request_path=request,
        activation_migration_path=migration,
    )
    assert parent["checkpoint_digest"] == child_digest
    assert parent["kind"] == "checksum_exact_successfully_uploaded_iteration_9_learner"
    assert postupload["schema"] == "poke_bot.marnie_postupload_weighted_bootstrap/v1"
    assert postupload["activation_identity"]["valid"] is True
    assert residuals == {"terminal_hazard": 0.2}


def test_postupload_bootstrap_rejects_deauthorized_latent_policy() -> None:
    payload = {
        "model_config": {
            "latent_lookahead_enabled": True,
            "latent_lookahead_action_authority_enabled": False,
            "latent_lookahead_width": 512,
            "latent_lookahead_policy_aid_cap": 0.25,
        },
        "model_state_dict": {
            "latent_lookahead.inventory": torch.zeros(412_130),
        },
    }
    with pytest.raises(RuntimeError, match="Accepted Policy Generation 15"):
        _validate_latent_policy_checkpoint(payload, label="test checkpoint")


def test_postupload_bootstrap_binds_latent_policy_continuity_receipt(
    tmp_path: Path,
) -> None:
    receipt = _write_json(
        tmp_path / "continuity.json",
        {
            "schema": "poke_bot.marnie_postupload_latent_policy_continuity/v1",
            "status": "validated_runtime_inert_until_post_iteration_9_bootstrap",
            "accepted_policy_generation": 15,
            "required_checkpoint_inventory": {
                "schema": "poke_bot.action_conditioned_latent_lookahead/v1",
                "enabled": True,
                "action_authority_enabled": True,
                "width": 512,
                "policy_aid_cap": 0.25,
                "parameters": 412_130,
            },
            "failure_behavior": "fail_closed_before_bootstrap_submission_or_self_play",
            "bootstrap_script_sha256": bootstrap_module.checkpoint.checkpoint_digest(
                Path(bootstrap_module.__file__).resolve()
            ),
        },
    )
    identity = _validate_latent_policy_continuity_receipt(receipt)
    assert identity["accepted_policy_generation"] == 15
    assert identity["sha256"] == _digest(receipt)


def test_marnie_adapter_authorization_binds_inherited_training_evidence(
    tmp_path: Path,
) -> None:
    expert = tmp_path / "expert.pt"
    router = tmp_path / "router.pt"
    expert.write_bytes(b"25-epoch-marnie")
    router.write_bytes(b"router-v6-marnie")

    authorization = _materialize_adapter_authorization(
        state_root=tmp_path / "state",
        router_checkpoint=router,
        router_checkpoint_digest=_digest(router),
        expert_checkpoint=expert,
        expert_checkpoint_digest=_digest(expert),
    )
    proof = validate_adapter_training_authorization(
        authorization,
        parent_checkpoint=router,
        permit_post_boundary_use=True,
    )

    assert proof.parent_checkpoint == router.resolve()
    assert proof.completed_iteration == -1


def test_marnie_directional_bootstrap_requires_all_five_guided_routes() -> None:
    metrics = {
        "guide_curriculum_head_metrics": {
            "directional_route_ranking": {
                "eligible_rows": 17,
                "heads": [
                    "action_q",
                    "action_resource",
                    "action_utility",
                    "setup_board_outcome",
                    "combo_state",
                ],
            }
        }
    }
    assert _directional_rows(metrics) == 17


def test_marnie_directional_bootstrap_rejects_missing_combo_route() -> None:
    metrics = {
        "guide_curriculum_head_metrics": {
            "directional_route_ranking": {
                "eligible_rows": 17,
                "heads": [
                    "action_q",
                    "action_resource",
                    "action_utility",
                    "setup_board_outcome",
                ],
            }
        }
    }
    with pytest.raises(RuntimeError, match="omitted a required Marnie route"):
        _directional_rows(metrics)


def test_directional_route_metrics_survive_multi_batch_merge() -> None:
    merged = _merge_metrics(
        [
            BatchMetrics(
                n_decisions=2,
                guide_curriculum_head_metrics={
                    "total": {"action_q": 2},
                    "labeled": {"action_q": 2},
                    "losses": {"action_q": 0.25},
                    "directional_route_ranking": {
                        "eligible_rows": 2,
                        "heads": {
                            "action_q": 0.5,
                            "action_resource": 0.6,
                            "action_utility": 0.7,
                            "setup_board_outcome": 0.8,
                            "combo_state": 0.9,
                        },
                        "margin": 0.1,
                    },
                },
            ),
            BatchMetrics(
                n_decisions=3,
                guide_curriculum_head_metrics={
                    "total": {"action_q": 3},
                    "labeled": {"action_q": 3},
                    "losses": {"action_q": 0.5},
                    "directional_route_ranking": {
                        "eligible_rows": 3,
                        "heads": {
                            "action_q": 1.0,
                            "action_resource": 1.1,
                            "action_utility": 1.2,
                            "setup_board_outcome": 1.3,
                            "combo_state": 1.4,
                        },
                        "margin": 0.1,
                    },
                },
            ),
        ]
    )
    directional = merged.guide_curriculum_head_metrics[
        "directional_route_ranking"
    ]
    assert directional["eligible_rows"] == 5
    assert set(directional["heads"]) == {
        "action_q",
        "action_resource",
        "action_utility",
        "setup_board_outcome",
        "combo_state",
    }
    assert directional["heads"]["action_q"] == pytest.approx(0.8)


def test_marnie_combo_rows_preserve_valid_all_masked_corpus() -> None:
    assert _combo_rows(
        {"combo_state_metrics": {"total_rows": 101, "eligible_rows": 0}}
    ) == 0


def test_marnie_combo_rows_reject_invalid_masking_telemetry() -> None:
    with pytest.raises(RuntimeError, match="masking telemetry is invalid"):
        _combo_rows(
            {"combo_state_metrics": {"total_rows": 3, "eligible_rows": 4}}
        )


def test_marnie_bootstrap_can_reset_inherited_core_head_history() -> None:
    parameter = inspect.signature(supervised_rehearsal_step).parameters[
        "reset_expanded_training_history"
    ]
    assert parameter.default is False

    source = inspect.getsource(
        __import__(
            "scripts.run_final_format_marnie_h10_bootstrap",
            fromlist=["main"],
        ).main
    )
    assert "reset_expanded_training_history=(epoch == 1)" in source


def _selected_payload(epoch: int = 16) -> dict:
    heads = {
        name: {
            "present": True,
            "gradient_enabled": True,
            "trained": True,
            "trained_this_epoch": True,
            "train_labeled_rows": 10,
            "validation_labeled_rows": 2,
        }
        for name in EXPANDED_HEAD_IDS
    }
    return {
        "extra": {
            "final_format_marnie_h10_bootstrap": {
                "schema": "poke_bot.final_format_marnie_h10_bootstrap_epoch/v1",
                "epoch": epoch,
                "guide_mode": "strategic_directional_v2",
                "guide_weight": 0.0,
                "decision_fusion_schema": "poke_bot.causal_decision_fusion/v3",
            },
            "expanded_head_training": {
                "schema": "poke_bot.expanded_head_training/v1",
                "epoch": epoch,
                "epochs_total": 25,
                "gradient_enabled_heads": list(EXPANDED_HEAD_IDS),
                "trained_this_epoch": list(EXPANDED_HEAD_IDS),
                "heads": heads,
            },
        }
    }


def test_marnie_registration_accepts_only_full_head_selection_epochs() -> None:
    _validate_selected_bootstrap_training(_selected_payload())

    historical = _selected_payload()
    historical["extra"]["final_format_marnie_h10_bootstrap"]["guide_weight"] = 0.05
    _validate_selected_bootstrap_training(historical)

    incomplete = _selected_payload(epoch=15)
    with pytest.raises(RuntimeError, match="full-head epoch evidence"):
        _validate_selected_bootstrap_training(incomplete)


def test_marnie_registration_rejects_missing_per_head_training_evidence() -> None:
    incomplete = _selected_payload()
    incomplete["extra"]["expanded_head_training"]["heads"][
        "remaining_turns"
    ]["validation_labeled_rows"] = 0
    with pytest.raises(RuntimeError, match="remaining_turns"):
        _validate_selected_bootstrap_training(incomplete)


def test_marnie_registration_records_exact_effective_route_reliabilities() -> None:
    payload = {
        "model_state_dict": {
            "decision_fusion.dedicated_route_log_reliability." + name:
            torch.tensor(0.0)
            for name in (
                "value",
                "archetype",
                "opponent_hand",
                "opponent_remainder",
                "lethal_threat",
                "prize_race",
                "action_q",
                "action_type",
                "action_target",
                "action_resource",
                "action_utility",
                "tactical_outcomes",
                "opponent_response",
                "resource_forecast",
                "game_phase",
                "outcome_distribution",
                "remaining_turns",
                "setup_board_outcome",
                "combo_state",
            )
        }
    }
    telemetry = _route_reliability_telemetry(payload)
    assert telemetry["value"] == pytest.approx(1.0)
    assert telemetry["action_type"] == pytest.approx(0.25)
    assert len(telemetry) == 19


def test_marnie_registration_rejects_nonfinite_route_reliability() -> None:
    payload = {
        "model_state_dict": {
            "decision_fusion.dedicated_route_log_reliability." + name:
            torch.tensor(float("nan") if name == "action_q" else 0.0)
            for name in (
                "value",
                "archetype",
                "opponent_hand",
                "opponent_remainder",
                "lethal_threat",
                "prize_race",
                "action_q",
                "action_type",
                "action_target",
                "action_resource",
                "action_utility",
                "tactical_outcomes",
                "opponent_response",
                "resource_forecast",
                "game_phase",
                "outcome_distribution",
                "remaining_turns",
                "setup_board_outcome",
                "combo_state",
            )
        }
    }
    with pytest.raises(RuntimeError, match="not finite: action_q"):
        _route_reliability_telemetry(payload)


def test_marnie_registration_requires_checksum_bound_ladder_mix(
    tmp_path, monkeypatch
) -> None:
    deployment = tmp_path / "deployment"
    import scripts.register_final_format_marnie_h10_rl as registration

    expected = {}
    for relative in (
        "submission/main.py",
        "submission/search_config.json",
        "data/training_mixes/top_ladder.v1.json",
        "data/training_mixes/top_ladder_representatives.v1.json",
        "data/training_mixes/specialist_representatives.v1.json",
    ):
        path = deployment / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'{{"schema":"{relative}"}}\n', encoding="utf-8")
        expected[relative] = registration._sha256(path)
    monkeypatch.setattr(registration, "RUNTIME_ASSET_SHA256", expected)
    assert _validate_runtime_assets(deployment) == expected
    mix = deployment / "data/training_mixes/top_ladder.v1.json"
    mix.write_text('{"schema":"changed"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="runtime training-mix asset"):
        _validate_runtime_assets(deployment)


def test_marnie_registration_accepts_only_exact_selector_r125_migration(
    tmp_path: Path,
) -> None:
    import json

    registration = tmp_path / "registration.json"
    registration.write_text('{"schema":"base"}\n', encoding="utf-8")
    deployment = tmp_path / "deployment"
    selector = tmp_path / "selector.env"
    exact_environment = {
        "POKEBOT_ACTIVE_SPECIALIST": "marnie-s-grimmsnarl-ex",
        "POKEBOT_SPECIALIST_RUNTIME_ROOT": str(deployment),
        "PYTHONPATH": str(deployment),
        "POKEBOT_REMOTE_SOCKET_PREFETCH": "1",
        "POKEBOT_REMOTE_SOCKET_PREFETCH_MAX": "1",
    }
    exact_service_environment = {
        "POKEBOT_REMOTE_SOCKET_PREFETCH": "1",
        "POKEBOT_REMOTE_SOCKET_PREFETCH_MAX": "1",
        "POKEBOT_SELF_PLAY_ELMO_TAIL_ONLY": "1",
        "POKEBOT_SELF_PLAY_TAIL_WORK_STEAL_GAMES": "20",
    }
    selector.write_text(
        "\n".join(f"{key}={value}" for key, value in exact_environment.items())
        + "\n",
        encoding="utf-8",
    )
    scheduler_dropin = tmp_path / "scheduler.conf"
    scheduler_dropin.write_text(
        "[Service]\n"
        + "\n".join(
            f"Environment={key}={value}"
            for key, value in exact_service_environment.items()
        )
        + "\n",
        encoding="utf-8",
    )
    migration = tmp_path / "migration.json"
    migration.write_text(
        json.dumps(
            {
                "schema": "poke_bot.marnie_selector_env_migration/v1",
                "status": "activated_at_stopped_uncommitted_boundary",
                "goal_revision": 125,
                "specialist_id": "marnie-s-grimmsnarl-ex",
                "run_name": "final_format_marnie_r104_h10_i_v6_8k",
                "base_registration_receipt": str(registration),
                "base_registration_sha256": _digest(registration),
                "base_selector_env_sha256": "sha256:" + "a" * 64,
                "selector_env": str(selector),
                "selector_env_sha256": _digest(selector),
                "exact_environment": exact_environment,
                "scheduler_dropin": str(scheduler_dropin),
                "scheduler_dropin_sha256": _digest(scheduler_dropin),
                "exact_service_environment": exact_service_environment,
                "prior_attempt_rejected": True,
                "observed_remote_sockets": 104,
                "maximum_remote_sockets": 52,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert _selector_env_is_authorized(
        registration_receipt=registration,
        selector_path=selector,
        expected_digest="sha256:" + "a" * 64,
        deployment=deployment,
        migration_receipt=migration,
        scheduler_dropin=scheduler_dropin,
    )

    selector.write_text(
        selector.read_text(encoding="utf-8").replace(
            "POKEBOT_REMOTE_SOCKET_PREFETCH_MAX=1",
            "POKEBOT_REMOTE_SOCKET_PREFETCH_MAX=2",
        ),
        encoding="utf-8",
    )
    assert not _selector_env_is_authorized(
        registration_receipt=registration,
        selector_path=selector,
        expected_digest="sha256:" + "a" * 64,
        deployment=deployment,
        migration_receipt=migration,
        scheduler_dropin=scheduler_dropin,
    )
