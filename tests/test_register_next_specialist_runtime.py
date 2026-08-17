from __future__ import annotations

import json
from pathlib import Path

import pytest

from poke_bot.matchup_adapters import EXPERT_IDS
from scripts import register_next_specialist_runtime as register


def test_crustle_persistent_guide_policy_forbids_automatic_decay() -> None:
    policy = register._persistent_crustle_guide_policy()
    assert policy == {
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


def _json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _strategic_curriculum(
    tmp_path: Path,
    *,
    specialist_id: str,
    guide: Path,
) -> dict:
    guide_digest = register.sha256(guide)
    implementation = tmp_path / "strategic_training_impl.py"
    implementation.write_text("STRATEGIC_CURRICULUM = True\n", encoding="utf-8")
    validation_test = tmp_path / "test_strategic_training_impl.py"
    validation_test.write_text(
        "def test_strategic_curriculum(): assert True\n",
        encoding="utf-8",
    )
    sources = list(register.CANONICAL_LEARNED_DECISION_SOURCES)
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
    role_map = _json(
        tmp_path / "strategic-head-roles.json",
        {
            "schema": "poke_bot.future_specialist_strategic_head_roles/v1",
            "specialist_id": specialist_id,
            "training_mode": "strategic_curriculum_v1",
            "guide_curriculum_revision": 51,
            "strategic_branch_scope_revision": 56,
            "action_influence_revision": 56,
            "guide_contract_sha256": guide_digest,
            "decision_fusion_schema": "poke_bot.causal_decision_fusion/v2",
            "preserve_v1_additive_residual": True,
            "canonical_learned_decision_sources": sources,
            "one_route_per_learned_source": True,
            "route_input": "option_hidden_plus_typed_output",
            "route_reduction": "fixed_mean",
            "aggregate_absolute_logit_cap": 1.0,
            "zero_safe_final_projection": True,
            "guide_is_only_action_route_exception": True,
            "heads": heads,
        },
    )
    spec = _json(
        tmp_path / "strategic-curriculum.json",
        {
            "schema": "poke_bot.future_specialist_strategic_curriculum/v1",
            "specialist_id": specialist_id,
            "training_mode": "strategic_curriculum_v1",
            "guide_curriculum_revision": 51,
            "strategic_branch_scope_revision": 56,
            "action_influence_revision": 56,
            "guide_contract_sha256": guide_digest,
            "head_role_map_sha256": register.sha256(role_map),
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
            "decision_fusion_schema": "poke_bot.causal_decision_fusion/v2",
            "preserve_v1_additive_residual": True,
            "canonical_learned_decision_sources": sources,
            "one_route_per_learned_source": True,
            "route_input": "option_hidden_plus_typed_output",
            "route_reduction": "fixed_mean",
            "aggregate_absolute_logit_cap": 1.0,
            "zero_safe_final_projection": True,
            "guide_is_only_action_route_exception": True,
            "pre_fleet_h10_compute_allowed": False,
        },
    )
    receipt = _json(
        tmp_path / "strategic-validation.json",
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
            "guide_contract_sha256": guide_digest,
            "curriculum_spec_sha256": register.sha256(spec),
            "head_role_map_sha256": register.sha256(role_map),
            "required_training_paths": [
                "supervised_bootstrap",
                "pure_rl",
                "resident_expert_rehearsal",
            ],
            "decision_fusion_schema": "poke_bot.causal_decision_fusion/v2",
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
                    "maximum_absolute_logit_contribution_observed": 0.10,
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
                    "sha256": register.sha256(implementation),
                },
                {
                    "role": "validation_test",
                    "path": str(validation_test.resolve()),
                    "sha256": register.sha256(validation_test),
                },
            ],
        },
    )
    return {
        "guide_training_mode": "strategic_curriculum_v1",
        "strategic_curriculum_spec": spec,
        "strategic_curriculum_spec_sha256": register.sha256(spec),
        "strategic_head_role_map": role_map,
        "strategic_head_role_map_sha256": register.sha256(role_map),
        "strategic_validation_receipt": receipt,
        "strategic_validation_receipt_sha256": register.sha256(receipt),
    }


def test_registration_is_single_source_and_idempotent(
    tmp_path: Path, monkeypatch
) -> None:
    family = tmp_path / "family"
    model = _json(family / "model.pt", {"model": True})
    _json(family / "manifest.json", {"checkpoint": str(model)})
    expert = _json(
        tmp_path / "expert.json",
        {
            "schema": "poke_bot.pinned_expert_corpus/v1",
            "protected": True,
            "totals": {"decisions_kept": 173_490},
        },
    )
    targets = list(EXPERT_IDS)
    tree = _json(
        tmp_path / "tree.json",
        {
            "runtime_enabled": True,
            "targets": targets,
            "runtime_contract": {
                "accepted_archetype_ids": targets,
                "one_route_per_decision": True,
                "unknown_route_exact_bypass": True,
            },
        },
    )
    registry = _json(
        tmp_path / "registry.json",
        {
            "schema": "poke_bot.specialist_runtime_registry/v1",
            "runtime_root": str(tmp_path),
            "specialists": {},
        },
    )
    selector = tmp_path / "specialist.env"
    selector.write_text(
        "# canonical\n"
        "POKEBOT_ACTIVE_SPECIALIST=starmie\n"
        "PURE_RL_SIM_WORKERS=48\n"
        "PURE_RL_GAMES_IN_FLIGHT=48\n"
        "POKEBOT_LIVE_POOL_MAX_WORKERS=48\n"
        "PURE_RL_REBALANCE_MAX_WORKERS=48\n"
        "PURE_RL_REBALANCE_MIN_WORKERS=32\n"
        "PURE_RL_REBALANCE_RAM_FLOOR_GB=12\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        register,
        "verify_frozen_model",
        lambda _family: {
            "model_path": str(model),
            "checkpoint_digest": "sha256:" + "a" * 64,
        },
    )

    kwargs = {
        "specialist_id": "lucario",
        "family": family,
        "expert": expert,
        "runtime_tree": tree,
        "runtime_registry": registry,
        "selector_env": selector,
        "state_root": tmp_path / "state",
        "run_name": "pure_rl_lucario_temporal1_8k_v1",
        "handoff_service": "pokebot-specialist-cycle-handoff.service",
    }
    first = register.register(**kwargs)
    second = register.register(**kwargs)

    assert first["identity_sha256"] == second["identity_sha256"]
    assert selector.read_text(encoding="utf-8").count(
        "POKEBOT_ACTIVE_SPECIALIST="
    ) == 1
    assert "POKEBOT_ACTIVE_SPECIALIST=lucario" in selector.read_text(
        encoding="utf-8"
    )
    assert f"POKEBOT_SPECIALIST_RUNTIME_ROOT={tmp_path}" in selector.read_text(
        encoding="utf-8"
    )
    assert f"PYTHONPATH={tmp_path}" in selector.read_text(encoding="utf-8")
    assert "POKEBOT_EXPANDED_HEADS_ENABLED=1" in selector.read_text(
        encoding="utf-8"
    )
    assert "POKEBOT_SETUP_BOARD_OUTCOME_HEAD_ENABLED=0" in selector.read_text(
        encoding="utf-8"
    )
    assert "POKEBOT_COMBO_STATE_HEAD_ENABLED=0" in selector.read_text(
        encoding="utf-8"
    )
    assert "POKEBOT_DECISION_FUSION_ENABLED=1" in selector.read_text(
        encoding="utf-8"
    )
    assert "POKEBOT_DECISION_FUSION_RUNTIME_ENABLED=1" in selector.read_text(
        encoding="utf-8"
    )
    for memory_guard in (
        "PURE_RL_SIM_WORKERS=48",
        "PURE_RL_GAMES_IN_FLIGHT=48",
        "POKEBOT_LIVE_POOL_MAX_WORKERS=48",
        "PURE_RL_REBALANCE_MAX_WORKERS=48",
        "PURE_RL_REBALANCE_MIN_WORKERS=32",
        "PURE_RL_REBALANCE_RAM_FLOOR_GB=12",
    ):
        assert memory_guard in selector.read_text(encoding="utf-8")
    row = json.loads(registry.read_text(encoding="utf-8"))["specialists"][
        "lucario"
    ]
    assert row["status"] == "ready"
    assert row["measurement_decks"] == "lucario"
    assert row["decision_fusion"]["required"] is True
    assert row["decision_fusion"]["runtime_enabled"] is True
    assert len(row["decision_fusion"]["required_heads"]) == 17
    assert row["pass_handler"]["handoff_service"] == (
        "pokebot-specialist-cycle-handoff.service"
    )
    authorization = json.loads(
        Path(row["matchup_adapter_authorization"]).read_text(encoding="utf-8")
    )
    assert authorization["optimizer_scope"] == "matchup_adapter_bank_only"
    assert authorization["runtime_enabled"] is False

    legacy_registry = json.loads(registry.read_text(encoding="utf-8"))
    legacy_runtime_row = legacy_registry["specialists"]["lucario"]
    legacy_runtime_row.pop("guide_training_mode")
    legacy_runtime_row.pop("strategic_curriculum")
    registry.write_text(json.dumps(legacy_registry), encoding="utf-8")
    preserved = register.register(**kwargs)
    assert "guide_training_mode" not in preserved["runtime_row"]
    assert "strategic_curriculum" not in preserved["runtime_row"]

    corrected = register.register(
        **{
            **kwargs,
            "run_name": "pure_rl_lucario_temporal1_8k_corrected_v2",
            "authorization_name": (
                "lucario-matchup-adapter-bootstrap-corrected-v2.json"
            ),
            "replace_unpassed": True,
        }
    )
    corrected_row = corrected["runtime_row"]
    assert corrected_row["run_name"].endswith("corrected_v2")
    assert corrected_row["matchup_adapter_authorization"].endswith(
        "corrected-v2.json"
    )

    expanded_targets = (*EXPERT_IDS, "teal-mask-ogerpon-ex")
    expanded_tree = _json(
        tmp_path / "expanded-tree.json",
        {
            "runtime_enabled": True,
            "targets": list(expanded_targets),
            "runtime_contract": {
                "accepted_archetype_ids": list(expanded_targets),
                "one_route_per_decision": True,
                "unknown_route_exact_bypass": True,
            },
        },
    )
    expanded = register.register(
        **{
            **kwargs,
            "specialist_id": "teal-mask-ogerpon-ex",
            "runtime_tree": expanded_tree,
            "run_name": "pure_rl_teal_mask_ogerpon_ex_v1",
            "matchup_target_ids": expanded_targets,
        }
    )
    assert expanded["specialist_id"] == "teal-mask-ogerpon-ex"


def test_selector_enables_registered_specialist_head_architecture(
    tmp_path: Path,
) -> None:
    selector = tmp_path / "specialist.env"
    selector.write_text(
        "POKEBOT_SETUP_BOARD_OUTCOME_HEAD_ENABLED=0\n"
        "POKEBOT_COMBO_STATE_HEAD_ENABLED=0\n",
        encoding="utf-8",
    )
    register._atomic_selector(
        selector,
        "slowking",
        tmp_path,
        (
            *register.CANONICAL_LEARNED_DECISION_SOURCES,
            "combo_state",
        ),
    )

    selector_text = selector.read_text(encoding="utf-8")
    assert "POKEBOT_SETUP_BOARD_OUTCOME_HEAD_ENABLED=1\n" in selector_text
    assert "POKEBOT_COMBO_STATE_HEAD_ENABLED=1\n" in selector_text


def test_registration_honors_specialist_specific_corpus_floor(
    tmp_path: Path, monkeypatch
) -> None:
    family = tmp_path / "family"
    model = _json(family / "model.pt", {"model": True})
    _json(family / "manifest.json", {"checkpoint": str(model)})
    expert = _json(
        tmp_path / "expert.json",
        {
            "schema": "poke_bot.pinned_expert_corpus/v1",
            "protected": True,
            "totals": {"decisions_kept": 10_946},
        },
    )
    targets = list(EXPERT_IDS)
    tree = _json(
        tmp_path / "tree.json",
        {
            "runtime_enabled": True,
            "targets": targets,
            "runtime_contract": {
                "accepted_archetype_ids": targets,
                "one_route_per_decision": True,
                "unknown_route_exact_bypass": True,
            },
        },
    )
    registry = _json(
        tmp_path / "registry.json",
        {
            "schema": "poke_bot.specialist_runtime_registry/v1",
            "runtime_root": str(tmp_path),
            "specialists": {},
        },
    )
    schedule_module = tmp_path / "poke_bot/pure_rl/deck_guide_schedule.py"
    schedule_module.parent.mkdir(parents=True, exist_ok=True)
    schedule_module.write_bytes(
        (
            Path(__file__).resolve().parents[1]
            / "poke_bot/pure_rl/deck_guide_schedule.py"
        ).read_bytes()
    )
    evidence_module = tmp_path / "poke_bot/pure_rl/guide_weight_evidence.py"
    evidence_module.write_bytes(
        (
            Path(__file__).resolve().parents[1]
            / "poke_bot/pure_rl/guide_weight_evidence.py"
        ).read_bytes()
    )
    selector = tmp_path / "specialist.env"
    selector.write_text(
        "POKEBOT_ACTIVE_SPECIALIST=lucario\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        register,
        "verify_frozen_model",
        lambda _family: {
            "model_path": str(model),
            "checkpoint_digest": "sha256:" + "b" * 64,
        },
    )

    receipt = register.register(
        specialist_id="dragapult-dusknoir",
        family=family,
        expert=expert,
        runtime_tree=tree,
        runtime_registry=registry,
        selector_env=selector,
        state_root=tmp_path / "state",
        run_name="pure_rl_dragapult-dusknoir_temporal1_8k_v1",
        handoff_service="pokebot-specialist-cycle-handoff.service",
        minimum_decisions=10_000,
        required_target_coverage=(
            "temporal_action_rows",
            "opponent_remainder_rows",
            "lethal_threat_rows",
            "prize_race_rows",
        ),
    )

    assert receipt["specialist_id"] == "dragapult-dusknoir"
    assert receipt["runtime_row"]["expert_minimum_decisions"] == 10_000
    assert receipt["runtime_row"]["expert_required_target_coverage"] == [
        "temporal_action_rows",
        "opponent_remainder_rows",
        "lethal_threat_rows",
        "prize_race_rows",
    ]


def test_registration_binds_successor_guide_to_runtime_row(
    tmp_path: Path, monkeypatch
) -> None:
    family = tmp_path / "family"
    model = _json(family / "model.pt", {"model": True})
    _json(family / "manifest.json", {"checkpoint": str(model)})
    expert = _json(
        tmp_path / "expert.json",
        {
            "schema": "poke_bot.pinned_expert_corpus/v1",
            "protected": True,
            "totals": {"decisions_kept": 20_000},
        },
    )
    targets = list(EXPERT_IDS)
    tree = _json(
        tmp_path / "tree.json",
        {
            "runtime_enabled": True,
            "targets": targets,
            "runtime_contract": {
                "accepted_archetype_ids": targets,
                "one_route_per_decision": True,
                "unknown_route_exact_bypass": True,
            },
        },
    )
    registry = _json(
        tmp_path / "registry.json",
        {
            "schema": "poke_bot.specialist_runtime_registry/v1",
            "runtime_root": str(tmp_path),
            "specialists": {},
        },
    )
    selector = tmp_path / "specialist.env"
    selector.write_text("", encoding="utf-8")
    guide = tmp_path / "grimmsnarl.yaml"
    guide.write_text("schema: guide\n", encoding="utf-8")
    schedule_module = tmp_path / "poke_bot/pure_rl/deck_guide_schedule.py"
    schedule_module.parent.mkdir(parents=True, exist_ok=True)
    schedule_module.write_bytes(
        (
            Path(__file__).resolve().parents[1]
            / "poke_bot/pure_rl/deck_guide_schedule.py"
        ).read_bytes()
    )
    evidence_module = tmp_path / "poke_bot/pure_rl/guide_weight_evidence.py"
    evidence_module.write_bytes(
        (
            Path(__file__).resolve().parents[1]
            / "poke_bot/pure_rl/guide_weight_evidence.py"
        ).read_bytes()
    )
    review_module = tmp_path / "poke_bot/pure_rl/guide_weight_review.py"
    review_module.write_bytes(
        (
            Path(__file__).resolve().parents[1]
            / "poke_bot/pure_rl/guide_weight_review.py"
        ).read_bytes()
    )
    boundary_controller = (
        tmp_path / "scripts/apply_future_guide_weight_at_boundary.py"
    )
    boundary_controller.parent.mkdir(parents=True, exist_ok=True)
    boundary_controller.write_bytes(
        (
            Path(__file__).resolve().parents[1]
            / "scripts/apply_future_guide_weight_at_boundary.py"
        ).read_bytes()
    )
    shadow_pair_module = (
        tmp_path / "poke_bot/pure_rl/guide_weight_shadow_pair.py"
    )
    shadow_pair_module.write_bytes(
        (
            Path(__file__).resolve().parents[1]
            / "poke_bot/pure_rl/guide_weight_shadow_pair.py"
        ).read_bytes()
    )
    for name in (
        "run_future_guide_weight_shadow_pair.py",
        "process_future_guide_weight_review_queue.py",
    ):
        target = tmp_path / "scripts" / name
        target.write_bytes(
            (Path(__file__).resolve().parents[1] / "scripts" / name).read_bytes()
        )
    monkeypatch.setattr(
        register,
        "verify_frozen_model",
        lambda _family: {
            "model_path": str(model),
            "checkpoint_digest": "sha256:" + "c" * 64,
        },
    )

    registration_args = {
        "specialist_id": "marnie-s-grimmsnarl-ex",
        "family": family,
        "expert": expert,
        "runtime_tree": tree,
        "runtime_registry": registry,
        "selector_env": selector,
        "state_root": tmp_path / "state",
        "run_name": "pure_rl_grimmsnarl",
        "handoff_service": "pokebot-specialist-cycle-handoff.service",
        "guide_id": "marnie-s-grimmsnarl-ex",
        "guide_loss_weight": 0.05,
        "guide_contract": guide,
        "guide_contract_sha256": register.sha256(guide),
        "guide_version": "marnie-grimmsnarl-north-star-v1",
    }
    with pytest.raises(RuntimeError, match="explicit guide training mode"):
        register.register(**registration_args)

    strategic = _strategic_curriculum(
        tmp_path,
        specialist_id="marnie-s-grimmsnarl-ex",
        guide=guide,
    )
    role_payload = json.loads(
        strategic["strategic_head_role_map"].read_text(encoding="utf-8")
    )
    register._validate_head_role_map(
        role_payload,
        specialist_id="marnie-s-grimmsnarl-ex",
        guide_contract_sha256=register.sha256(guide).removeprefix("sha256:"),
    )
    role_payload["heads"]["setup_board_outcome"][
        "fusion_role"
    ] = "shadow_unfused"
    with pytest.raises(RuntimeError, match="head role is invalid"):
        register._validate_head_role_map(
            role_payload,
            specialist_id="marnie-s-grimmsnarl-ex",
            guide_contract_sha256=register.sha256(guide).removeprefix(
                "sha256:"
            ),
        )
    receipt = register.register(**registration_args, **strategic)

    row = receipt["runtime_row"]
    assert row["guide_id"] == "marnie-s-grimmsnarl-ex"
    assert row["guide_loss_weight"] == 0.05
    assert row["guide_contract"] == str(guide.resolve())
    assert row["guide_contract_sha256"] == register.sha256(guide).removeprefix(
        "sha256:"
    )
    assert row["guide_training_mode"] == "strategic_curriculum_v1"
    assert row["decision_fusion"] == {
        "schema": "poke_bot.causal_decision_fusion/v2",
        "required": True,
        "runtime_enabled": True,
        "required_heads": list(register.CANONICAL_LEARNED_DECISION_SOURCES),
    }
    assert len(row["decision_fusion"]["required_heads"]) == 18
    assert "setup_board_outcome" in row["decision_fusion"]["required_heads"]
    assert row["strategic_curriculum"]["strategic_branch_scope_revision"] == 56
    assert (
        row["strategic_curriculum"]["validation_receipt_sha256"]
        == register.sha256(strategic["strategic_validation_receipt"]).removeprefix(
            "sha256:"
        )
    )
    policy = row["guide_weight_policy"]
    assert policy["schema"] == "poke_bot.current_deck_guide_weight_policy/v1"
    assert policy["owner_decision_revision"] == 43
    assert policy["prospective_scope_revision"] == 44
    assert policy["learning_semantics_revision"] == 46
    assert policy["guide_curriculum_revision"] == 51
    assert policy["strategic_branch_scope_revision"] == 56
    assert policy["action_influence_revision"] == 56
    assert policy["decision_fusion_schema"] == (
        "poke_bot.causal_decision_fusion/v2"
    )
    assert policy["scope"] == "future_specialist_training_runs_only"
    assert policy["prospective_effective_specialist"] == "archaludon-ex"
    assert policy[
        "retroactive_application_to_completed_frozen_or_started_runs"
    ] is False
    assert policy["historical_weight_or_receipt_rewrite_allowed"] is False
    assert policy["active_teal_revision42_exception_preserved"] is True
    assert policy["post_bootstrap_positive_ramp_steps"] == [
        0.15,
        0.25,
        0.35,
        0.5,
    ]
    assert policy["decay_steps"] == [0.15, 0.075, 0.0]
    assert policy["maximum_auxiliary_loss_weight"] == 0.5
    assert policy["minimum_paired_games_per_review"] == 1000
    assert policy["minimum_paired_games_per_matchup"] == 50
    assert policy["realized_win_delta_confidence_level"] == 0.90
    assert policy["evidence_receipt_schema"] == (
        "poke_bot.current_deck_guide_paired_evaluation/v1"
    )
    assert policy["schedule_receipt_schema"] == (
        "poke_bot.current_deck_guide_weight_schedule/v1"
    )
    assert policy["review_request_schema"] == (
        "poke_bot.current_deck_guide_weight_review_request/v1"
    )
    assert policy["automatic_review_after_each_five_iteration_commit"] is True
    assert policy["learning_effect"] == (
        "literal_multiplier_on_bounded_guide_conditioned_"
        "strategic_head_curriculum"
    )
    assert policy["multiplier_applied_before_backpropagation"] is True
    assert policy["gradient_effect"] == (
        "scales_guide_conditioned_strategic_head_gradient_contribution"
    )
    assert policy["training_target_mode"] == "bounded_strategic_head_curriculum"
    assert policy["direct_policy_cross_entropy_allowed"] is False
    assert policy["activation_requires_prestage_validation_receipt"] is True
    assert policy["allowed_fusion_roles"] == ["fused_input"]
    assert policy["every_head_declares_fusion_role"] is True
    assert policy["allowed_computation_roles"] == ["independent_head"]
    assert policy["every_head_declares_computation_role"] is True
    assert policy["independent_pre_fusion_branches_trainable"] is True
    assert policy["action_influence"] == "bounded_option_conditioned_route"
    assert policy["causal_input"] == "board_state_and_legal_option"
    assert policy["direct_action_selection_authority"] is False
    assert policy["runtime_activation_requirement"] == (
        "receipt_backed_validation"
    )
    assert policy["per_head_action_logit_ablation_required"] is True
    assert policy["preserve_v1_additive_residual"] is True
    assert len(policy["canonical_learned_decision_sources"]) == 18
    assert policy["one_route_per_learned_source"] is True
    assert policy["route_input"] == "option_hidden_plus_typed_output"
    assert policy["route_reduction"] == "fixed_mean"
    assert policy["aggregate_absolute_logit_cap"] == 1.0
    assert policy["zero_safe_final_projection"] is True
    assert policy["guide_is_only_action_route_exception"] is True
    assert policy["elapsed_time_only_progression_allowed"] is False
    assert policy["guide_head_only_bookkeeping_allowed"] is False
    assert policy["dashboard_only_or_runtime_action_bias_allowed"] is False
    assert policy["boundary_receipt_schema"] == (
        "poke_bot.future_specialist_guide_weight_boundary/v1"
    )
    assert policy["shadow_pair_manifest_schema"] == (
        "poke_bot.future_guide_weight_shadow_pair/v1"
    )
    assert Path(policy["boundary_controller"]).name == (
        "apply_future_guide_weight_at_boundary.py"
    )
    assert policy["clean_boundary_schedule_receipt_required"] is True
    assert policy["schedule_records_earliest_eligible_next_iteration"] is True
    assert policy["application_boundary"] == (
        "first_available_future_five_iteration_hard_pause"
    )
    assert policy[
        "actual_application_iteration_recorded_only_by_boundary_receipt"
    ] is True
    assert policy["runtime_action_override_allowed"] is False
    selector_text = selector.read_text()
    assert "POKEBOT_FUTURE_GUIDE_WEIGHT_POLICY_REVISION=44\n" in selector_text
    assert "POKEBOT_GUIDE_LEARNING_SEMANTICS_REVISION=46\n" in selector_text
    assert (
        "POKEBOT_GUIDE_CONSECUTIVE_NONPOSITIVE_EVALUATIONS=0\n"
        in selector_text
    )
