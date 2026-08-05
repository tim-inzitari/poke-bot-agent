from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from poke_bot.matchup_adapter_routes import (
    resolve_matchup_adapter_route_contract,
)
from poke_bot.matchup_adapters_v6 import (
    ADAPTER_CHECKPOINT_FORMAT as V6_ADAPTER_CHECKPOINT_FORMAT,
    SLOT_CAPACITY,
    load_slot_registry,
    registry_digest,
)
from poke_bot.model import (
    DECISION_FUSION_REQUIRED_HEADS,
    DECISION_FUSION_SCHEMA,
)
from scripts.launch_active_specialist import (
    _build_command,
    _load_registry,
    _required_runtime_fusion_heads,
    _resolve,
    _sha256,
    _validate_crustle_persistent_guide_policy,
    _validate_guide_training_contract,
    _validate_guide_weight_policy,
)


def test_crustle_persistent_guide_policy_is_fail_closed() -> None:
    policy = {
        "schema": "poke_bot.crustle_persistent_guide_hold/v1",
        "owner_decision_revision": 165,
        "scope": "crustle_persistent_training_only",
        "held_weight": 0.05,
        "automatic_review_after_each_five_iteration_commit": False,
        "automatic_ramp_allowed": False,
        "automatic_decay_allowed": False,
        "change_requires_explicit_owner_decision": True,
        "change_requires_checksum_bound_boundary_receipt": True,
        "direct_policy_cross_entropy_allowed": False,
        "runtime_action_override_allowed": False,
        "serving_authority": False,
        "gate_authority": False,
    }
    _validate_crustle_persistent_guide_policy(policy)
    policy["automatic_decay_allowed"] = True
    with pytest.raises(RuntimeError, match="Crustle persistent guide policy"):
        _validate_crustle_persistent_guide_policy(policy)


def _fixture(tmp_path: Path, *, status: str = "ready") -> Path:
    root = tmp_path / "runtime"
    (root / "scripts").mkdir(parents=True)
    (root / "ops").mkdir()
    checkpoint = tmp_path / "model.pt"
    expert = tmp_path / "expert.json"
    slot_registry = load_slot_registry()
    adapter_config = {
        "format": V6_ADAPTER_CHECKPOINT_FORMAT,
        "slot_capacity": SLOT_CAPACITY,
        "slot_registry_digest": registry_digest(slot_registry),
        "slot_registry": slot_registry,
    }
    route_contract = resolve_matchup_adapter_route_contract(adapter_config)
    routes = list(route_contract.target_ids)
    torch.save(
        {
            "model_state_dict": {
                f"matchup_adapter_bank.experts.{index}.up.weight": torch.zeros(1)
                for index in range(SLOT_CAPACITY)
            },
            "extra": {"matchup_adapter_config": adapter_config},
        },
        checkpoint,
    )
    expert.write_text("{}", encoding="utf-8")
    runtime_tree = tmp_path / "runtime-tree.json"
    runtime_tree.write_text(
        json.dumps(
            {
                "runtime_enabled": True,
                "targets": routes,
                "runtime_contract": {
                    "accepted_archetype_ids": ["dragapult-dusknoir"],
                    "one_route_per_decision": True,
                    "unknown_route_exact_bypass": True,
                    **route_contract.runtime_binding(),
                },
            }
        ),
        encoding="utf-8",
    )
    authorization = tmp_path / "adapter-authorization.json"
    authorization.write_text(
        json.dumps(
            {
                "schema": "poke_bot.matchup_adapter_rehearsal_authorization/v1",
                "optimizer_scope": "matchup_adapter_bank_only",
                "runtime_enabled": False,
            }
        ),
        encoding="utf-8",
    )
    for relative in ("scripts/launch_pure_rl.py", "ops/research.json"):
        path = root / relative
        path.write_text("{}", encoding="utf-8")
    digest = "sha256:" + "1" * 64
    (root / "ops/frozen.json").write_text(
        json.dumps(
            {
                "schema": "poke_bot.frozen_specialist_registry/v1",
                "specialists": [
                    {
                        "specialist_id": "alakazam",
                        "opponent_id": "specialist-alakazam",
                        "checkpoint_digest": digest,
                        "frozen": True,
                        "public_mix_eligible": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "ops/gate.json").write_text(
        json.dumps(
            {
                "next_gate": {
                    "id": "gate-v1+frozen-specialists-r1",
                    "evaluation": {
                        "games_total": 2250,
                        "games_per_opponent": 250,
                        "seat0_games_per_opponent": 125,
                        "seat1_games_per_opponent": 125,
                    },
                    "roster": [
                        *[
                            {"opponent_id": f"public-{index}"}
                            for index in range(8)
                        ],
                        {
                            "opponent_id": "specialist-alakazam",
                            "tier": "S+",
                            "frozen_specialist": True,
                            "frozen_checkpoint_digest": digest,
                        },
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "schema": "poke_bot.specialist_runtime_registry/v1",
        "version": 1,
        "selector_environment_variable": "POKEBOT_ACTIVE_SPECIALIST",
        "runtime_root": str(root),
        "python": "/usr/bin/python3",
        "launcher": "scripts/launch_pure_rl.py",
        "active_gate_contract": "ops/gate.json",
        "research_control_registry": "ops/research.json",
        "frozen_specialist_registry": "ops/frozen.json",
        "terminal_active_gate_id": "gate-v1",
        "minimum_terminal_iteration": 5,
        "iteration_ceiling": 15,
        "common_launcher_args": ["--mode", "specialist"],
        "common_trainer_args": [
            "--expert-min-decisions",
            "20000",
            "--continue-after-gate",
            "--resume",
            "auto",
        ],
        "specialists": {
            "dragapult-dusknoir": {
                "status": status,
                "reason": "not bootstrapped",
                "run_name": "run-dragapult-dusknoir",
                "log": str(tmp_path / "run.log"),
                "initial_checkpoint": str(checkpoint),
                "initial_checkpoint_sha256": _sha256(checkpoint),
                "expert_manifest": str(expert),
                "expert_manifest_sha256": _sha256(expert),
                "expert_minimum_decisions": 10_000,
                "expert_required_target_coverage": [
                    "temporal_action_rows",
                    "opponent_remainder_rows",
                ],
                "matchup_runtime_tree": str(runtime_tree),
                "matchup_runtime_tree_sha256": _sha256(runtime_tree),
                "matchup_adapter_authorization": str(authorization),
                "matchup_adapter_authorization_sha256": _sha256(authorization),
                "matchup_adapter_epochs_per_rl_iteration": 1,
                "measurement_decks": "dragapult-dusknoir",
                "guide_loss_weight": 0.0,
                "terminal_gate_marker": "SPECIALIST_GATE_PASSED.dragapult-dusknoir-v1",
            }
        },
    }
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_registry_accepts_monotonic_content_version(
    tmp_path: Path,
) -> None:
    path = _fixture(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["version"] = 3
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert _load_registry(path)["version"] == 3


def test_load_registry_rejects_nonpositive_content_version(
    tmp_path: Path,
) -> None:
    path = _fixture(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["version"] = 0
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid specialist runtime registry"):
        _load_registry(path)


def test_runtime_fusion_inventory_allows_registered_specialist_heads() -> None:
    mandatory = list(
        dict.fromkeys(
            (*DECISION_FUSION_REQUIRED_HEADS, "setup_board_outcome")
        )
    )
    row = {
        "guide_training_mode": "strategic_curriculum_v1",
        "decision_fusion": {
            "required_heads": [*mandatory, "combo_state"],
        },
    }
    assert _required_runtime_fusion_heads(row) == [
        *mandatory,
        "combo_state",
    ]

    row["decision_fusion"]["required_heads"] = [
        *mandatory[:-1],
        "combo_state",
    ]
    with pytest.raises(RuntimeError, match="weaken or reorder"):
        _required_runtime_fusion_heads(row)


def _strategic_training_row(
    tmp_path: Path,
    *,
    specialist_id: str = "archaludon-ex",
    additional_sources: tuple[str, ...] = (),
) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    guide = tmp_path / "guide.yaml"
    guide.write_text("schema: guide\n", encoding="utf-8")
    guide_digest = _sha256(guide)
    implementation = tmp_path / "strategic_training.py"
    implementation.write_text("STRATEGIC = True\n", encoding="utf-8")
    validation_test = tmp_path / "test_strategic_training.py"
    validation_test.write_text(
        "def test_strategic(): assert True\n",
        encoding="utf-8",
    )
    sources = list(
        dict.fromkeys(
            (
                *DECISION_FUSION_REQUIRED_HEADS,
                "setup_board_outcome",
                *additional_sources,
            )
        )
    )
    heads = {
        head_id: {
            "computation_role": "independent_head",
            "fusion_role": "fused_input",
            "trainable": True,
            "causal_training_targets_only": True,
            "guide_action_target_allowed": False,
            "enters_decision_fusion": True,
            "action_influence": "bounded_option_conditioned_route",
            "causal_input": "board_state_and_legal_option",
            "direct_action_selection_authority": False,
            "runtime_activation_requirement": "receipt_backed_validation",
            "route_id": f"{head_id}-route",
            "route_input": "option_hidden_plus_typed_output",
            "route_reduction": "fixed_mean",
            "zero_safe_final_projection": True,
            "maximum_absolute_logit_contribution": 0.25,
        }
        for head_id in sources
    }
    role_map = tmp_path / "head-roles.json"
    role_map.write_text(
        json.dumps(
            {
                "schema": (
                    "poke_bot.future_specialist_strategic_head_roles/v1"
                ),
                "specialist_id": specialist_id,
                "training_mode": "strategic_curriculum_v1",
                "guide_curriculum_revision": 51,
                "strategic_branch_scope_revision": 56,
                "action_influence_revision": 56,
                "guide_contract_sha256": f"sha256:{guide_digest}",
                "decision_fusion_schema": (
                    "poke_bot.causal_decision_fusion/v2"
                ),
                "preserve_v1_additive_residual": True,
                "canonical_learned_decision_sources": sources,
                "one_route_per_learned_source": True,
                "route_input": "option_hidden_plus_typed_output",
                "route_reduction": "fixed_mean",
                "aggregate_absolute_logit_cap": 1.0,
                "zero_safe_final_projection": True,
                "guide_is_only_action_route_exception": True,
                "heads": heads,
            }
        ),
        encoding="utf-8",
    )
    spec = tmp_path / "curriculum.json"
    spec.write_text(
        json.dumps(
            {
                "schema": (
                    "poke_bot.future_specialist_strategic_curriculum/v1"
                ),
                "specialist_id": specialist_id,
                "training_mode": "strategic_curriculum_v1",
                "guide_curriculum_revision": 51,
                "strategic_branch_scope_revision": 56,
                "action_influence_revision": 56,
                "guide_contract_sha256": f"sha256:{guide_digest}",
                "head_role_map_sha256": f"sha256:{_sha256(role_map)}",
                "curriculum_heads": sorted(sources),
                "guide_targets": "observed_causal_strategic_heads_only",
                "direct_policy_cross_entropy_allowed": False,
                "guide_runtime_input_allowed": False,
                "guide_action_selection_allowed": False,
                "replace_observed_outcome_targets_allowed": False,
                "all_curriculum_heads_must_influence_actions": True,
                "computation_role": "independent_head",
                "fusion_role": "fused_input",
                "action_influence": "bounded_option_conditioned_route",
                "causal_input": "board_state_and_legal_option",
                "direct_action_selection_authority": False,
                "runtime_activation_requirement": "receipt_backed_validation",
                "decision_fusion_schema": (
                    "poke_bot.causal_decision_fusion/v2"
                ),
                "preserve_v1_additive_residual": True,
                "canonical_learned_decision_sources": sources,
                "one_route_per_learned_source": True,
                "route_input": "option_hidden_plus_typed_output",
                "route_reduction": "fixed_mean",
                "aggregate_absolute_logit_cap": 1.0,
                "zero_safe_final_projection": True,
                "guide_is_only_action_route_exception": True,
                "pre_fleet_h10_compute_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    validation = tmp_path / "validation.json"
    validation.write_text(
        json.dumps(
            {
                "schema": (
                    "poke_bot.future_specialist_strategic_"
                    "curriculum_validation/v1"
                ),
                "status": "validated",
                "specialist_id": specialist_id,
                "training_mode": "strategic_curriculum_v1",
                "guide_curriculum_revision": 51,
                "strategic_branch_scope_revision": 56,
                "action_influence_revision": 56,
                "guide_contract_sha256": f"sha256:{guide_digest}",
                "curriculum_spec_sha256": f"sha256:{_sha256(spec)}",
                "head_role_map_sha256": f"sha256:{_sha256(role_map)}",
                "required_training_paths": [
                    "supervised_bootstrap",
                    "pure_rl",
                    "resident_expert_rehearsal",
                ],
                "decision_fusion_schema": (
                    "poke_bot.causal_decision_fusion/v2"
                ),
                "validated_route_ids": [
                    heads[head_id]["route_id"] for head_id in sorted(sources)
                ],
                "checks": {
                    "guide_supervision_terminates_at_strategic_heads": True,
                    "fused_policy_remains_outcome_and_win_trained": True,
                    "direct_policy_cross_entropy_absent": True,
                    "observed_outcome_targets_not_replaced": True,
                    "all_curriculum_heads_have_valid_fusion_roles": True,
                    "every_curriculum_head_has_bounded_action_scoring_route": True,
                    "per_head_action_influence_ablation_passed": True,
                    "exact_parent_parity_at_initialization": True,
                    "one_option_conditioned_route_per_learned_head": True,
                    "causal_suffix_invariance": True,
                    "legal_option_dependence": True,
                    "bounded_aggregate_residual": True,
                    "all_training_paths_use_the_declared_mode": True,
                    "active_and_historical_specialists_unchanged": True,
                },
                "measurements": {
                    "guide_to_strategic_head_gradient_norm": 0.125,
                    "guide_labeled_rows": 128,
                    "maximum_absolute_aggregate_residual_observed": 0.5,
                },
                "action_influence_ablations": {
                    head_id: {
                        "decisions_evaluated": 128,
                        "mean_absolute_action_logit_delta": 0.02,
                        "maximum_absolute_logit_contribution_observed": 0.1,
                        "selection_change_rate": 0.04,
                    }
                    for head_id in sources
                },
                "route_validation": {
                    head_id: {
                        "route_id": heads[head_id]["route_id"],
                        "route_input": "option_hidden_plus_typed_output",
                        "post_training_route_gradient_norm": 0.01,
                        "legal_option_dependence_delta": 0.02,
                        "causal_suffix_max_logit_delta": 0.0,
                        "zero_safe_initial_max_logit_delta": 0.0,
                    }
                    for head_id in sources
                },
                "implementation_artifacts": [
                    {
                        "role": "training_implementation",
                        "path": str(implementation.resolve()),
                        "sha256": f"sha256:{_sha256(implementation)}",
                    },
                    {
                        "role": "validation_test",
                        "path": str(validation_test.resolve()),
                        "sha256": f"sha256:{_sha256(validation_test)}",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return {
        "guide_id": specialist_id,
        "guide_loss_weight": 0.05,
        "guide_contract": str(guide.resolve()),
        "guide_contract_sha256": guide_digest,
        "guide_version": "north-star-v1",
        "guide_training_mode": "strategic_curriculum_v1",
        "guide_weight_policy": {},
        "strategic_curriculum": {
            "schema": "poke_bot.specialist_guide_training_contract/v1",
            "training_mode": "strategic_curriculum_v1",
            "guide_curriculum_revision": 51,
            "strategic_branch_scope_revision": 56,
            "action_influence_revision": 56,
            "decision_fusion_schema": "poke_bot.causal_decision_fusion/v2",
            "require_all_registered_learned_sources": bool(
                additional_sources
            ),
            "curriculum_spec": str(spec.resolve()),
            "curriculum_spec_sha256": _sha256(spec),
            "head_role_map": str(role_map.resolve()),
            "head_role_map_sha256": _sha256(role_map),
            "validation_receipt": str(validation.resolve()),
            "validation_receipt_sha256": _sha256(validation),
        },
    }


def test_strategic_curriculum_binds_registered_specialist_head(
    tmp_path: Path,
) -> None:
    row = _strategic_training_row(
        tmp_path,
        specialist_id="alakazam",
        additional_sources=("combo_state",),
    )
    registered = tuple(
        dict.fromkeys(
            (
                *DECISION_FUSION_REQUIRED_HEADS,
                "setup_board_outcome",
                "combo_state",
            )
        )
    )
    assert (
        _validate_guide_training_contract(row, "alakazam", registered)
        == "strategic_curriculum_v1"
    )

    with pytest.raises(RuntimeError, match="head-role map is invalid"):
        _validate_guide_training_contract(
            row,
            "alakazam",
            registered[:-1],
        )


def test_ready_specialist_resolves_one_complete_command(tmp_path: Path) -> None:
    registry = _load_registry(_fixture(tmp_path))
    row, checkpoint, expert, runtime_tree, authorization = _resolve(
        registry, "dragapult-dusknoir"
    )
    command = _build_command(
        registry,
        "dragapult-dusknoir",
        row,
        checkpoint,
        expert,
        runtime_tree,
        authorization,
    )
    assert command.count("--specialist-archetype") == 1
    assert command[command.index("--specialist-archetype") + 1] == "dragapult-dusknoir"
    assert command[command.index("--minimum-terminal-iteration") + 1] == "5"
    assert command[command.index("--iterations") + 1] == "16"
    assert command[command.index("--terminal-active-gate-id") + 1] == (
        "gate-v1+frozen-specialists-r1"
    )
    assert command[command.index("--heldout-games") + 1] == "2250"
    assert command.count("--expert-min-decisions") == 1
    assert command[command.index("--expert-min-decisions") + 1] == "10000"
    assert command.count("--expert-required-target") == 2
    assert "--frozen-specialist-registry" in command
    assert command[command.index("--combo-state-loss-weight") + 1] == "0.0"


def test_slowking_command_enables_typed_combo_loss(tmp_path: Path) -> None:
    registry = _load_registry(_fixture(tmp_path))
    row, checkpoint, expert, runtime_tree, authorization = _resolve(
        registry, "dragapult-dusknoir"
    )

    command = _build_command(
        registry,
        "slowking",
        row,
        checkpoint,
        expert,
        runtime_tree,
        authorization,
    )

    assert command[command.index("--combo-state-loss-weight") + 1] == "0.025"
    required_targets = [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--expert-required-target"
    ]
    assert required_targets[-1] == "combo_state_rows"


def test_registered_combo_loss_weight_is_honored_for_non_slowking_refresh(
    tmp_path: Path,
) -> None:
    registry = _load_registry(_fixture(tmp_path))
    row, checkpoint, expert, runtime_tree, authorization = _resolve(
        registry, "dragapult-dusknoir"
    )
    row["combo_state_loss_weight"] = 0.025

    command = _build_command(
        registry,
        "marnie-s-grimmsnarl-ex",
        row,
        checkpoint,
        expert,
        runtime_tree,
        authorization,
    )

    assert command[command.index("--combo-state-loss-weight") + 1] == "0.025"


def test_revision44_prospective_guide_weight_policy_is_checksum_bound(
    tmp_path: Path,
) -> None:
    module = tmp_path / "deck_guide_schedule.py"
    module.write_text("MAXIMUM_AUXILIARY_WEIGHT = 0.50\n", encoding="utf-8")
    evidence_module = tmp_path / "guide_weight_evidence.py"
    evidence_module.write_text("EVIDENCE_SCHEMA = 'v1'\n", encoding="utf-8")
    review_module = tmp_path / "guide_weight_review.py"
    review_module.write_text("REQUEST_SCHEMA = 'v1'\n", encoding="utf-8")
    boundary_controller = tmp_path / "apply_future_guide_weight_at_boundary.py"
    boundary_controller.write_text("BOUNDARY_SCHEMA = 'v1'\n", encoding="utf-8")
    shadow_pair_module = tmp_path / "guide_weight_shadow_pair.py"
    shadow_pair_module.write_text("PAIR_SCHEMA = 'v1'\n", encoding="utf-8")
    shadow_pair_runner = tmp_path / "run_future_guide_weight_shadow_pair.py"
    shadow_pair_runner.write_text("PAIR_RUNNER = True\n", encoding="utf-8")
    shadow_queue_processor = (
        tmp_path / "process_future_guide_weight_review_queue.py"
    )
    shadow_queue_processor.write_text("QUEUE = True\n", encoding="utf-8")
    row = {
        "guide_weight_policy": {
            "schema": "poke_bot.current_deck_guide_weight_policy/v1",
                "owner_decision_revision": 43,
                "prospective_scope_revision": 44,
                "learning_semantics_revision": 46,
                "guide_curriculum_revision": 51,
                "strategic_branch_scope_revision": 56,
                "action_influence_revision": 56,
                "decision_fusion_schema": (
                    "poke_bot.causal_decision_fusion/v2"
                ),
            "scope": "future_specialist_training_runs_only",
            "prospective_effective_specialist": "archaludon-ex",
            "retroactive_application_to_completed_frozen_or_started_runs": False,
            "historical_weight_or_receipt_rewrite_allowed": False,
            "active_teal_revision42_exception_preserved": True,
            "schedule_module": str(module),
            "schedule_module_sha256": _sha256(module),
            "evidence_module": str(evidence_module),
            "evidence_module_sha256": _sha256(evidence_module),
            "review_request_module": str(review_module),
            "review_request_module_sha256": _sha256(review_module),
            "boundary_controller": str(boundary_controller),
            "boundary_controller_sha256": _sha256(boundary_controller),
            "shadow_pair_module": str(shadow_pair_module),
            "shadow_pair_module_sha256": _sha256(shadow_pair_module),
            "shadow_pair_runner": str(shadow_pair_runner),
            "shadow_pair_runner_sha256": _sha256(shadow_pair_runner),
            "shadow_queue_processor": str(shadow_queue_processor),
            "shadow_queue_processor_sha256": _sha256(shadow_queue_processor),
            "shadow_pair_manifest_schema": (
                "poke_bot.future_guide_weight_shadow_pair/v1"
            ),
            "boundary_receipt_schema": (
                "poke_bot.future_specialist_guide_weight_boundary/v1"
            ),
            "review_request_schema": (
                "poke_bot.current_deck_guide_weight_review_request/v1"
            ),
                "automatic_review_after_each_five_iteration_commit": True,
                "learning_effect": (
                    "literal_multiplier_on_bounded_guide_conditioned_"
                    "strategic_head_curriculum"
                ),
                "multiplier_applied_before_backpropagation": True,
                "gradient_effect": (
                    "scales_guide_conditioned_strategic_head_gradient_contribution"
                ),
                "training_target_mode": "bounded_strategic_head_curriculum",
                "direct_policy_cross_entropy_allowed": False,
                "activation_requires_prestage_validation_receipt": True,
                "allowed_fusion_roles": ["fused_input"],
                "every_head_declares_fusion_role": True,
                "allowed_computation_roles": ["independent_head"],
                "every_head_declares_computation_role": True,
                "independent_pre_fusion_branches_trainable": True,
                "action_influence": "bounded_option_conditioned_route",
                "causal_input": "board_state_and_legal_option",
                "direct_action_selection_authority": False,
                "runtime_activation_requirement": "receipt_backed_validation",
                "per_head_action_logit_ablation_required": True,
                "preserve_v1_additive_residual": True,
                "canonical_learned_decision_sources": list(
                    dict.fromkeys(
                        (
                            *DECISION_FUSION_REQUIRED_HEADS,
                            "setup_board_outcome",
                        )
                    )
                ),
                "one_route_per_learned_source": True,
                "route_input": "option_hidden_plus_typed_output",
                "route_reduction": "fixed_mean",
                "aggregate_absolute_logit_cap": 1.0,
                "zero_safe_final_projection": True,
                "guide_is_only_action_route_exception": True,
                "elapsed_time_only_progression_allowed": False,
                "guide_head_only_bookkeeping_allowed": False,
                "dashboard_only_or_runtime_action_bias_allowed": False,
            "evidence_receipt_schema": (
                "poke_bot.current_deck_guide_paired_evaluation/v1"
            ),
            "schedule_receipt_schema": (
                "poke_bot.current_deck_guide_weight_schedule/v1"
            ),
            "bootstrap_ramp": [0.01, 0.05],
            "post_bootstrap_positive_ramp_steps": [0.15, 0.25, 0.35, 0.50],
            "maximum_auxiliary_loss_weight": 0.50,
            "review_every_completed_iterations": 5,
            "minimum_paired_games_per_review": 1000,
            "minimum_paired_games_per_matchup": 50,
            "realized_win_delta_confidence_level": 0.90,
            "positive_ramp_evidence": (
                "training_ineligible_paired_guide_on_guide_off_"
                "realized_win_delta_lower_confidence_bound_above_zero"
            ),
            "decay_after_consecutive_nonpositive_reviews": 2,
            "decay_steps": [0.15, 0.075, 0.0],
                "clean_boundary_schedule_receipt_required": True,
                "schedule_records_earliest_eligible_next_iteration": True,
                "application_boundary": (
                    "first_available_future_five_iteration_hard_pause"
                ),
                "actual_application_iteration_recorded_only_by_boundary_receipt": True,
                "consecutive_nonpositive_evaluations": 0,
            "training_replay_and_formal_gate_tuning_allowed": False,
            "counterfactual_checkpoint_serving_allowed": False,
            "counterfactual_checkpoint_promotion_allowed": False,
            "runtime_action_override_allowed": False,
        }
    }
    _validate_guide_weight_policy(row)
    module.write_text("MAXIMUM_AUXILIARY_WEIGHT = 0.25\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        _validate_guide_weight_policy(row)


def test_future_strategic_curriculum_is_checksum_and_route_bound(
    tmp_path: Path,
) -> None:
    assert (
        _validate_guide_training_contract(
            {"guide_loss_weight": 0.05},
            "teal-mask-ogerpon-ex",
        )
        == "legacy_policy_ce_v1"
    )
    with pytest.raises(RuntimeError, match="explicit training mode"):
        _validate_guide_training_contract(
            {
                "guide_loss_weight": 0.05,
                "guide_weight_policy": {},
            },
            "archaludon-ex",
        )

    assert (
        _validate_guide_training_contract(
            {
                "guide_training_mode": "strategic_directional_v2",
                "guide_loss_weight": 0.0,
                "guide_retired": True,
                "guide_retirement_revision": 140,
                "guide_target_generation_required": False,
                "guide_conditioned_losses_enabled": False,
                "guide_action_influence": False,
            },
            "marnie-s-grimmsnarl-ex",
        )
        == "strategic_directional_v2"
    )

    row = _strategic_training_row(tmp_path)
    assert (
        _validate_guide_training_contract(row, "archaludon-ex")
        == "strategic_curriculum_v1"
    )
    registry = _load_registry(_fixture(tmp_path / "command"))
    command_row = dict(registry["specialists"]["dragapult-dusknoir"])
    command_row.update(
        _strategic_training_row(
            tmp_path / "command-contract",
            specialist_id="dragapult-dusknoir",
        )
    )
    command = _build_command(
        registry,
        "dragapult-dusknoir",
        command_row,
        Path(command_row["initial_checkpoint"]),
        Path(command_row["expert_manifest"]),
        Path(command_row["matchup_runtime_tree"]),
        Path(command_row["matchup_adapter_authorization"]),
    )
    assert command[
        command.index("--current-deck-guide-training-mode") + 1
    ] == "strategic_curriculum_v1"
    assert "--current-deck-guide-curriculum-spec" in command
    assert "--current-deck-guide-head-role-map" in command
    assert "--current-deck-guide-curriculum-validation-receipt" in command

    validation = Path(row["strategic_curriculum"]["validation_receipt"])
    validation.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        _validate_guide_training_contract(row, "archaludon-ex")


def test_successor_runtime_fails_closed_without_exact_17_head_fusion(
    tmp_path: Path,
) -> None:
    path = _fixture(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = payload["specialists"]["dragapult-dusknoir"]
    row["decision_fusion"] = {
        "schema": DECISION_FUSION_SCHEMA,
        "required": True,
        "runtime_enabled": True,
        "required_heads": list(DECISION_FUSION_REQUIRED_HEADS),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="mandatory all-head"):
        _resolve(_load_registry(path), "dragapult-dusknoir")

    checkpoint = Path(row["initial_checkpoint"])
    checkpoint_payload = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    checkpoint_payload["model_config"] = {
        "expanded_heads_enabled": True,
        "decision_fusion_enabled": True,
        "decision_fusion_runtime_enabled": True,
    }
    checkpoint_payload["provenance"] = {
        "decision_fusion": {
            "schema": DECISION_FUSION_SCHEMA,
            "enabled": True,
            "runtime_enabled": True,
            "required_heads": list(DECISION_FUSION_REQUIRED_HEADS),
        }
    }
    checkpoint_payload["model_state_dict"][
        "decision_fusion.residual.2.weight"
    ] = torch.ones(1)
    torch.save(checkpoint_payload, checkpoint)
    row["initial_checkpoint_sha256"] = _sha256(checkpoint)
    path.write_text(json.dumps(payload), encoding="utf-8")

    _resolve(_load_registry(path), "dragapult-dusknoir")


def test_v6_checkpoint_must_bind_exact_runtime_registry(tmp_path: Path) -> None:
    path = _fixture(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = payload["specialists"]["dragapult-dusknoir"]
    checkpoint = Path(row["initial_checkpoint"])
    slot_registry = load_slot_registry()
    torch.save(
        {
            "model_state_dict": {
                f"matchup_adapter_bank.experts.{index}.up.weight": torch.zeros(1)
                for index in range(SLOT_CAPACITY)
            },
            "extra": {
                "matchup_adapter_config": {
                    "format": V6_ADAPTER_CHECKPOINT_FORMAT,
                    "slot_capacity": SLOT_CAPACITY,
                    "slot_registry_digest": registry_digest(slot_registry),
                    "slot_registry": slot_registry,
                }
            },
        },
        checkpoint,
    )
    row["initial_checkpoint_sha256"] = _sha256(checkpoint)
    path.write_text(json.dumps(payload), encoding="utf-8")
    _resolve(_load_registry(path), "dragapult-dusknoir")

    mismatched = json.loads(json.dumps(slot_registry))
    mismatched["revision"] = int(mismatched["revision"]) + 1
    checkpoint_payload = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    checkpoint_payload["extra"]["matchup_adapter_config"][
        "slot_registry"
    ] = mismatched
    torch.save(checkpoint_payload, checkpoint)
    payload["specialists"]["dragapult-dusknoir"][
        "initial_checkpoint_sha256"
    ] = _sha256(checkpoint)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="exact runtime registry"):
        _resolve(_load_registry(path), "dragapult-dusknoir")


def test_existing_run_uses_validated_loop_state_not_initial_seed(
    tmp_path: Path,
) -> None:
    registry = _load_registry(_fixture(tmp_path))
    row, checkpoint, expert, runtime_tree, authorization = _resolve(
        registry, "dragapult-dusknoir"
    )
    loop_state = (
        Path(registry["runtime_root"])
        / "outputs"
        / "pure_rl"
        / row["run_name"]
        / "loop_state.json"
    )
    loop_state.parent.mkdir(parents=True)
    loop_state.write_text(
        json.dumps(
            {
                "next_iteration": 0,
                "learner": {
                    "path": str(checkpoint),
                    "digest": f"sha256:{_sha256(checkpoint)}",
                },
            }
        ),
        encoding="utf-8",
    )

    command = _build_command(
        registry,
        "dragapult-dusknoir",
        row,
        checkpoint,
        expert,
        runtime_tree,
        authorization,
    )

    assert "--initial-learner-checkpoint" not in command
    assert command[command.index("--resume") + 1] == "auto"


def test_future_specialist_uses_registry_floor_and_ceiling(tmp_path: Path) -> None:
    registry = _load_registry(_fixture(tmp_path))
    registry["minimum_terminal_iteration"] = 5
    registry["iteration_ceiling"] = 15
    row, checkpoint, expert, runtime_tree, authorization = _resolve(
        registry, "dragapult-dusknoir"
    )
    command = _build_command(
        registry,
        "dragapult-dusknoir",
        row,
        checkpoint,
        expert,
        runtime_tree,
        authorization,
    )
    assert command[command.index("--minimum-terminal-iteration") + 1] == "5"
    assert command[command.index("--iterations") + 1] == "16"


def test_not_ready_specialist_fails_closed(tmp_path: Path) -> None:
    registry = _load_registry(_fixture(tmp_path, status="not_ready"))
    with pytest.raises(RuntimeError, match="not ready"):
        _resolve(registry, "dragapult-dusknoir")


def test_registered_digest_mismatch_fails_closed(tmp_path: Path) -> None:
    registry_path = _fixture(tmp_path)
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    payload["specialists"]["dragapult-dusknoir"]["initial_checkpoint_sha256"] = "0" * 64
    registry_path.write_text(json.dumps(payload), encoding="utf-8")
    registry = _load_registry(registry_path)
    with pytest.raises(RuntimeError, match="digest mismatch"):
        _resolve(registry, "dragapult-dusknoir")


def test_gate_missing_frozen_predecessor_fails_closed(tmp_path: Path) -> None:
    registry = _load_registry(_fixture(tmp_path))
    gate_path = Path(registry["runtime_root"]) / registry["active_gate_contract"]
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["next_gate"]["roster"].pop()
    gate["next_gate"]["evaluation"]["games_total"] = 2000
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    row, checkpoint, expert, runtime_tree, authorization = _resolve(
        registry, "dragapult-dusknoir"
    )
    with pytest.raises(RuntimeError, match="gate/registry"):
        _build_command(
            registry,
            "dragapult-dusknoir",
            row,
            checkpoint,
            expert,
            runtime_tree,
            authorization,
        )


def test_gate_total_scales_with_every_frozen_predecessor(tmp_path: Path) -> None:
    registry = _load_registry(_fixture(tmp_path))
    runtime = Path(registry["runtime_root"])
    frozen_path = runtime / registry["frozen_specialist_registry"]
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    trevenant_digest = "sha256:" + "2" * 64
    frozen["specialists"].append(
        {
            "specialist_id": "hops-trevenant",
            "opponent_id": "specialist-trevenant",
            "checkpoint_digest": trevenant_digest,
            "frozen": True,
            "public_mix_eligible": True,
        }
    )
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
    gate_path = runtime / registry["active_gate_contract"]
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["next_gate"]["id"] = "gate-v1+frozen-specialists-r2"
    gate["next_gate"]["evaluation"]["games_total"] = 2500
    gate["next_gate"]["roster"].append(
        {
            "opponent_id": "specialist-trevenant",
            "tier": "S+",
            "frozen_specialist": True,
            "frozen_checkpoint_digest": trevenant_digest,
        }
    )
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    row, checkpoint, expert, runtime_tree, authorization = _resolve(
        registry, "dragapult-dusknoir"
    )
    command = _build_command(
        registry,
        "dragapult-dusknoir",
        row,
        checkpoint,
        expert,
        runtime_tree,
        authorization,
    )
    assert command[command.index("--terminal-active-gate-id") + 1] == (
        "gate-v1+frozen-specialists-r2"
    )
    assert command[command.index("--heldout-games") + 1] == "2500"
