from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml
import pytest


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "state/alakazam-new-list-direct-policy-r241.json"
R262_OWNER_CONTRACT_SHA256 = (
    "sha256:57cbc0ac7ca7ee3791f7257899a16f6f0642749effa218323368e35940cdc202"
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_r241_exact_deck_guide_parent_and_libcg_bindings() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert contract["schema"] == "poke_bot.alakazam_new_list_direct_policy_r241/v1"
    assert contract["owner_decision_revision"] == 241

    parent = contract["parent"]
    assert parent["immutable"] is True
    assert parent["checkpoint_sha256"] == (
        "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
    )
    assert _sha256(ROOT / parent["typed_source"]) == parent["typed_source_sha256"]

    deck = contract["exact_deck"]
    deck_path = ROOT / deck["path"]
    cards = [int(line) for line in deck_path.read_text().splitlines() if line]
    assert len(cards) == 60
    assert Counter(cards) == Counter(
        {
            741: 4, 742: 4, 743: 3, 305: 3, 66: 2, 140: 1,
            1264: 4, 1086: 4, 1231: 4, 1081: 4, 1225: 4,
            1152: 4, 1079: 3, 1097: 2, 1182: 3, 1197: 2,
            1184: 1, 1129: 1, 19: 4, 5: 2, 13: 1,
        }
    )
    assert _sha256(deck_path) == deck["file_sha256"]
    multiset = hashlib.sha256(
        json.dumps(sorted(cards), separators=(",", ":")).encode()
    ).hexdigest()
    assert "sha256:" + multiset == deck["canonical_multiset_sha256"]

    guide = contract["owner_guide"]
    assert _sha256(ROOT / guide["guide_contract"]) == guide["guide_contract_sha256"]
    assert _sha256(ROOT / guide["human_guide"]) == guide["human_guide_sha256"]
    assert _sha256(ROOT / guide["teacher_module"]) == guide["teacher_module_sha256"]
    assert guide["ordinary_rl_guide_loss_weight"] == 0.05
    assert guide["expert_soft_refresh_guide_loss_weight"] == 0.0

    simulator = contract["canonical_simulator"]
    assert _sha256(ROOT / simulator["typed_source"]) == simulator["typed_source_sha256"]
    assert simulator["binding_environment"] == "CG_LIB_PATH"
    assert set(simulator["forbidden_environment"]) == {
        "POKEBOT_LIBCG_PATH",
        "POKEBOT_BATCH_LIBCG",
    }
    assert simulator["linux_x86_64_sha256"] == (
        "sha256:d16244a3157fc55c3314f08dcc7c5179168697d78c105b95c7debd556b764bb7"
    )


def test_r241_fixed_cycle_refresh_and_single_submit_boundary() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert contract["latest_owner_clarification_revision"] == 262
    topology = contract["activation_topology"]
    assert topology == {
        "owner_revision": 262,
        "activation_overlay_schema": (
            "poke_bot.alakazam_new_list_direct_r241_activation_overlay/v2"
        ),
        "activation_overlay_mirror_schema": (
            "poke_bot.alakazam_new_list_direct_r241_activation_overlay_mirror/v2"
        ),
        "activation_overlay_mirrors_schema": (
            "poke_bot.alakazam_new_list_direct_r241_activation_overlay_mirrors/v2"
        ),
        "owner_start_authorization_schema": (
            "poke_bot.alakazam_new_list_direct_r241_owner_start_authorization/v2"
        ),
        "owner_start_authorization_generator_schema": (
            "poke_bot.alakazam_new_list_direct_r241_owner_start_authorization_generator/v2"
        ),
        "execution_dag": [
            "immutable_owner_intent",
            "checksum_bound_source_and_baseline_payload_snapshots",
            "completed_elmo_expert_side_store_and_zero_safe_head_migration",
            "bounded_training_canary_and_runtime_influence_receipts",
            "offline_host_receipts",
            "one_logical_create_only_activation_overlay",
            "managed_services",
        ],
        "logical_overlay_cardinality": "one",
        "host_publications": ["inzi", "elmo"],
        "host_publication_identity_requirement": "byte_identical_with_one_shared_sha256",
        "overlay_scope": (
            "both_hosts_source_baseline_libcg_peak_r195_own_deck_head_migration_"
            "canary_evaluation_and_remote_worker_receipts"
        ),
        "source_snapshot_registry_policy": (
            "remain_pending_static_intent_until_the_external_overlay_is_published"
        ),
        "derived_readiness_policy": (
            "derived_readiness_and_operation_authorization_exist_only_in_the_create_only_activation_overlay"
        ),
        "managed_service_start_policy": (
            "requires_the_checksum_bound_external_overlay_its_owner_start_authorization_"
            "and_all_r260_head_import_gates"
        ),
    }
    preservation = contract["peak_r195_behavior_preservation"]
    assert preservation["learned_head_count_present"] == 19
    assert preservation["inherited_learned_head_count_present"] == 19
    assert preservation["inherited_active_non_combo_fusion_route_count"] == 18
    assert preservation["new_typed_option_head_count"] == 2
    assert preservation["total_architecture_head_count_after_import"] == 21
    assert preservation["new_routes_outside_inherited_fusion_denominator"] == 2
    assert preservation["every_architecture_present_non_combo_head_trainable"] is True
    assert preservation["every_architecture_present_non_combo_fusion_route_enabled"] is True
    assert preservation["combo_state_head_remains_present"] is True
    assert preservation["combo_state_loss_weight"] == 0.0
    assert preservation["combo_state_fusion_route_enabled"] is False
    assert preservation["matchup_adapter_bank_preserved"] is True
    assert preservation["matchup_adapter_training_enabled"] is True
    assert preservation["matchup_adapter_runtime_enabled"] is True

    refresh = contract["matchup_adapter_archetype_refresh"]
    assert refresh["owner_revision"] == 247
    assert refresh["deferred_by_owner_revision"] == 248
    assert refresh["status"] == "deferred_not_part_of_r241_cycle"
    assert refresh["authentication_secret_committed_logged_or_receipted"] is False
    assert (
        refresh[
            "source_may_supply_actions_gradients_hidden_state_outcomes_or_gate_evidence"
        ]
        is False
    )
    assert refresh["sealed_source_snapshot"] is None
    assert refresh["sealed_source_snapshot_receipt_required"] is False
    assert refresh["current_cycle_required_slot_migration_status"] == "no_slot_change"
    assert refresh["current_cycle_launch_training_terminal_or_submission_gate"] is False
    assert refresh["baseline_slot_registry"] == "state/matchup_adapter_roster.json"
    assert refresh["baseline_slot_registry_sha256"] == (
        "sha256:08322efe30c0f8b75d922aae8b882b4e78a20df03a63ed997ec8288165bfd1bc"
    )
    assert (
        refresh[
            "existing_slots_0_through_19_identity_and_tensor_values_must_remain_bit_identical"
        ]
        is True
    )
    assert refresh["new_archetype_allocation_policy"] == "lowest_never_used_router_format_6_slot"
    assert refresh["retired_or_existing_slot_reuse_reindex_or_rename_allowed"] is False
    assert refresh["new_slot_initial_state"] == "exact_zero_dormant_without_optimizer_state"
    assert refresh["new_archetype_slots"] == []
    assert (
        refresh[
            "already_running_exact20_roster18_corpus_jobs_must_not_be_mutated_or_restarted"
        ]
        is True
    )
    assert refresh["eligible_adapter_learning_uses_only_checksum_backed_training_data"] is True
    assert refresh["fixed_ten_update_schedule_unchanged"] is True
    assert refresh["games_per_update_unchanged"] is True
    assert refresh["expert_refresh_schedule_unchanged"] is True
    assert refresh["exact_deck_and_guide_unchanged"] is True
    assert refresh["direct_policy_only_and_one_terminal_submit_boundaries_unchanged"] is True

    cycle = contract["training_cycle"]
    assert cycle["rl_updates_exact"] == 10
    assert cycle["zero_indexed_iteration_commits"] == list(range(10))
    assert cycle["next_iteration_after_loop"] == 10
    assert cycle["iteration_10_collection_allowed"] is False
    assert cycle["games_per_update"] == 8196
    assert cycle["self_play_games_exact"] == 1024
    assert cycle["public_mix_games_exact"] == 7172
    assert cycle["marnie_h10_games_minimum"] == 1024
    assert cycle["established_diverse_public_mix_preserved"] is True
    assert cycle["marnie_h10_is_minimum_not_exclusive_public_opponent"] is True
    assert cycle["established_research_control_phase_preserved"] is True
    assert cycle["training_seats"] == {
        "first": 4098,
        "second": 4098,
        "exact_split_required": True,
    }

    marnie = contract["marnie_practice_opponent"]
    assert marnie["checkpoint_sha256"].startswith("sha256:f20efb20f5c3")
    assert marnie["content_sha256"].startswith("sha256:f7c25cfd0bba")
    assert marnie["learner_and_opponent_runtime"] == "frozen_direct_policy"
    assert marnie["mcts_or_rtp_provenance_allowed"] is False

    refresh = contract["expert_soft_refresh"]
    assert refresh["rolling_calendar_window_start"] == "2026-07-22"
    assert refresh["rolling_calendar_window_end"] == "2026-08-10"
    assert refresh["calendar_days_inclusive"] == 20
    assert refresh["update_boundaries"] == [5, 10]
    assert refresh["epochs_each_boundary"] == 5
    assert refresh["terminal_checkpoint_name"] == "expert_before_iter_00010.pt"
    assert refresh["terminal_refresh_must_not_collect_an_eleventh_wave"] is True
    assert "today_source_status" not in refresh
    assert "activation_gates" not in contract
    assert refresh["exact_window_evidence_binding"] == {
        "staging_receipt": "state/alakazam-new-list-direct-r241-expert-window-staging.json",
        "canonical_manifest_sha256": (
            "sha256:09848f04a6c863a02c517fdcd5b7a61a139eceafd3348aa2a08705fd6e971a16"
        ),
        "expected_latest_calendar_date": "2026-08-10",
        "evidence_role": (
            "immutable_exact_window_identity_only; readiness_is_derived_externally"
        ),
    }

    exclusion = contract["search_and_planning_exclusion"]
    assert exclusion["mcts"] == "forbidden_for_scoped_direct_roles"
    assert exclusion["recursive_turn_planner"] == "forbidden_for_scoped_direct_roles"
    assert exclusion["submission_action_selector"] == "direct_policy_only"
    assert exclusion["scope"] == {
        "learner": "direct_policy_only",
        "pinned_h10_marnie_opponent": "direct_policy_only",
        "target_generation": "direct_policy_only",
        "terminal_package_and_submission": "direct_policy_only",
        "frozen_non_h10_diverse_public_opponent_packages_and_selectors": (
            "preserve_unchanged_per_r245"
        ),
    }
    assert exclusion["public_opponent_selector_change"] == "forbidden"
    assert exclusion["public_search_firewall"] == "not_introduced"
    submission = contract["submission"]
    assert submission["exact_count"] == 1
    assert submission["turn_order_preference"] == "first_if_allowed"
    assert submission["intermediate_iteration_5_submission_allowed"] is False
    assert submission["retry_copy_or_duplicate_allowed"] is False


def test_r262_own_deck_head_structure_import_and_inzi_placement_are_exact() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    imported = contract["own_deck_head_structure_import"]
    assert imported["owner_revision"] == 260
    assert imported["status"] == "authorized_for_pre_start_successor_integration"
    assert imported["provenance_manifest"] == "state/alakazam-own-deck-ledger-successor-r258.json"
    assert imported["pre_start_override"] == {
        "supersedes_r258_wait_for_r241_completion_only_for_this_successor_boundary": True,
        "evidence_required": "no_r241_worker_arm_listener_training_update_or_submission_ever_started",
        "prior_1c34_h10_v8_peak_v6_r13_quartet_and_overlay": "immutable_inactive_history",
    }
    assert imported["expert_corpus"] == {
        "host": "elmo",
        "source_manifest": "/mnt/Main/main/poke-bot-agent/archive/expert-r241-20260722-20260810/current.json",
        "source_manifest_sha256": (
            "sha256:09848f04a6c863a02c517fdcd5b7a61a139eceafd3348aa2a08705fd6e971a16"
        ),
        "source_window_receipt": (
            "/mnt/Main/main/poke-bot-agent/archive/expert-r241-20260722-20260810/"
            "windows/2026-07-22_2026-08-10.json"
        ),
        "source_window_receipt_sha256": (
            "sha256:d377cd5b4558150588d1461539d50bcfb2ca46898120b4e3ad97e9d95e479551"
        ),
        "start_date": "2026-07-22",
        "end_date": "2026-08-10",
        "day_count": 20,
        "validated_episode_count": 91_253,
        "source_archive_bytes": 14_842_033_482,
        "source_access": "read_only",
        "derived_side_store_root": (
            "/mnt/Main/main/poke-bot-agent/archive/expert-r258-own-deck-ledger-sidecar/"
            "2026-07-22_2026-08-10"
        ),
        "derived_side_store_eligibility": (
            "all_20_immutable_daily_shards_plus_source_join_schema_count_digest_"
            "causal_local_remote_parity_and_completion_receipts_required"
        ),
        "partial_or_unreceipted_side_store_training_eligible": False,
    }
    assert imported["training_placement"] == {
        "owner_revision": 262,
        "sole_managed_training_host": "inzi",
        "elmo_role": "read_only_source_preprocessing_and_bounded_disposable_parity_only",
        "elmo_may_train_learner": False,
        "canonical_inzi_training_root": (
            "/home/inzi/poke-bot-agent/outputs/pure_rl/"
            "alakazam_new_list_direct_policy_r241/runtime/"
            "r260-own-deck-training-dataset"
        ),
        "inzi_prefix_staging_root": (
            "/home/inzi/poke-bot-agent/outputs/pure_rl/"
            "alakazam_new_list_direct_policy_r241/runtime/"
            "r260-own-deck-training-dataset-staging-09848f04"
        ),
        "prefix_transfer_while_elmo_builder_runs": True,
        "prefix_transfer_scope": "committed_non_dot_daily_directories_only",
        "per_day_transfer": "create_only_byte_identical_rehash_and_read_only_seal",
        "partial_staging_root_training_eligible": False,
        "final_promotion": (
            "atomic_only_after_20_of_20_join_parity_and_transport_receipts_pass"
        ),
        "trainer_may_consume_elmo_mnt_main_path": False,
        "trainer_input": "local_inzi_disk_backed_exact_four_key_streaming_index_only",
        "healthy_r259_service_may_be_stopped_restarted_or_reconfigured": False,
    }
    assert imported["architecture"] == {
        "shared_adapter_schema": "poke_bot.own_deck_ledger_adapter/v2",
        "shared_adapter_width": 128,
        "shared_adapter_applies_before_policy_value_every_existing_learned_head_option_decoding_and_fusion": True,
        "option_adapter_schema": "poke_bot.own_deck_ledger_option_adapter/v1",
        "option_feature_dim": 8,
        "visible_tutor_completion_head_schema": "poke_bot.visible_tutor_completion_head/v1",
        "visible_tutor_completion_output_dim": 7,
        "terminal_conversion_head_schema": "poke_bot.terminal_conversion_head/v1",
        "terminal_conversion_output_dim": 6,
        "typed_option_route_schema": "poke_bot.own_deck_option_route/v1",
        "typed_option_route_width": 16,
        "typed_option_route_aggregate_delta_cap": 1.0,
        "new_option_routes_are_outside_inherited_18_route_fusion_denominator": True,
        "new_tensor_prefixes": [
            "own_deck_ledger_adapter.",
            "own_deck_ledger_option_adapter.",
            "visible_tutor_completion_head.",
            "terminal_conversion_head.",
            "visible_tutor_completion_route.",
            "terminal_conversion_route.",
        ],
        "total_auxiliary_loss_weight": 0.05,
        "visible_tutor_completion_loss_weight": 0.025,
        "terminal_conversion_loss_weight": 0.025,
    }
    assert imported["migration"] == {
        "parent": "revision_195_checkpoint",
        "zero_safe": True,
        "every_inherited_tensor_must_be_bit_identical": True,
        "pre_training_parent_behavior_must_be_exact": True,
        "zero_safe_final_projection_keys": [
            "own_deck_ledger_adapter.output.weight",
            "own_deck_ledger_adapter.output.bias",
            "own_deck_ledger_option_adapter.network.3.weight",
            "own_deck_ledger_option_adapter.network.3.bias",
            "visible_tutor_completion_route.network.2.weight",
            "visible_tutor_completion_route.network.2.bias",
            "terminal_conversion_route.network.2.weight",
            "terminal_conversion_route.network.2.bias",
        ],
        "typed_parent_file_identity_keys": ["path", "sha256", "size_bytes"],
        "host_local_parent_paths_allowed_when_sha256_and_size_match_registry": True,
    }
    assert imported["runtime_boundaries"] == {
        "direct_policy_only": True,
        "mcts_rtp_or_search": False,
        "hidden_deck_prize_or_opponent_private_state": False,
        "guide_runtime_action_authority": False,
        "fabricated_counterfactual_targets": False,
        "evaluation_or_kaggle_replay_training": False,
    }
    assert imported["promotion_requirements"] == [
        "finite_gradient_training_canary_receipt",
        "coverage_receipt",
        "calibration_receipt",
        "bounded_influence_receipt",
        "local_remote_replay_parity_receipt",
        "source_disjoint_evaluation_receipt",
        "new_source_bound_h10_peak_worker_image_preflight_and_overlay_receipts",
    ]
    assert imported["activation_after_all_requirements_pass"] == (
        "immediate_without_further_owner_decision"
    )


def test_r241_trainer_resolves_only_the_checksum_bound_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.train_pure_rl import _our_decks

    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    deck = contract["exact_deck"]
    monkeypatch.setenv("POKEBOT_SPECIALIST_DECK_PATH", str(ROOT / deck["path"]))
    monkeypatch.setenv("POKEBOT_SPECIALIST_DECK_SHA256", deck["file_sha256"])
    monkeypatch.setenv(
        "POKEBOT_SPECIALIST_DECK_MULTISET_SHA256",
        deck["canonical_multiset_sha256"],
    )
    resolved = _our_decks("specialist", "alakazam")
    assert len(resolved) == 1
    assert resolved[0][0] == "alakazam"
    assert len(resolved[0][1]) == 60

    monkeypatch.setenv("POKEBOT_SPECIALIST_DECK_SHA256", "sha256:" + "0" * 64)
    with pytest.raises(ValueError, match="override file identity mismatch"):
        _our_decks("specialist", "alakazam")


def test_r241_authoritative_and_compatibility_projections_agree() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    exclusion = contract["search_and_planning_exclusion"]
    topology = contract["activation_topology"]
    imported = contract["own_deck_head_structure_import"]
    contract_sha = _sha256(CONTRACT_PATH)
    goal = (ROOT / "GOAL.md").read_text(encoding="utf-8")
    revision = int(goal.split("Revision: `", 1)[1].split("`", 1)[0])
    assert contract_sha == R262_OWNER_CONTRACT_SHA256
    assert revision >= 262
    assert "Under revision 241-TRAINING" in goal
    assert "Under revision 251-TRAINING" in goal
    assert "Under revision 260-TRAINING" in goal
    assert "Under revision 262-TRAINING" in goal
    assert "state/alakazam-new-list-direct-policy-r241.json" in goal

    requirements = json.loads(
        (ROOT / "ops/current_goal_requirements.json").read_text(encoding="utf-8")
    )["current_owner_overrides"]["alakazam_new_list_direct_policy_r241"]
    protocol = yaml.safe_load(
        (ROOT / "config/rl_protocol.yaml").read_text(encoding="utf-8")
    )["alakazam_new_list_direct_policy_r241"]
    specialists = yaml.safe_load(
        (ROOT / "state/specialists.yaml").read_text(encoding="utf-8")
    )["alakazam_new_list_direct_policy_r241"]

    assert requirements["typed_source_sha256"] == contract_sha
    assert protocol["values_owned_by_sha256"] == contract_sha
    assert specialists["owner_contract_sha256"] == contract_sha
    assert requirements["matchup_adapters_enabled"] is True
    assert requirements["authenticated_ptcgreplay_matchup_archetype_refresh_revision"] == 247
    assert requirements["authenticated_ptcgreplay_matchup_archetype_refresh_deferred_by_owner_revision"] == 248
    assert requirements["latest_owner_clarification_revision"] == 262
    assert requirements["matchup_archetype_refresh_cycle_status"] == "deferred_not_part_of_r241_cycle"
    assert requirements["matchup_archetype_refresh_launch_gate"] == "not_required_due_to_r248_deferral"
    assert requirements["adapter_slot_migration_required_status"] == "no_slot_change"
    assert requirements["existing_matchup_slot_identities_and_peak_r195_tensors_immutable"] is True
    assert requirements["new_matchup_slots_start_exact_zero_and_dormant"] is True
    assert requirements["ptcgreplay_meta_may_supply_training_actions_or_gate_evidence"] is False
    assert requirements["every_non_combo_head_and_fusion_route_live_and_trainable"] is True
    assert requirements["combo_state_loss_and_fusion_route_enabled"] is False
    assert protocol["matchup_adapters_enabled"] is True
    assert protocol["latest_owner_clarification_revision"] == 262
    assert protocol["matchup_archetype_refresh_cycle_status"] == "deferred_not_part_of_r241_cycle"
    assert protocol["matchup_archetype_refresh_launch_gate"] == "not_required_due_to_r248_deferral"
    assert protocol["adapter_slot_migration_required_status"] == "no_slot_change"
    assert protocol["new_matchup_slots_start_exact_zero_and_dormant"] is True
    assert specialists["matchup_adapters_enabled"] is True
    assert specialists["latest_owner_clarification_revision"] == 262
    assert specialists["matchup_archetype_refresh_cycle_status"] == "deferred_not_part_of_r241_cycle"
    assert specialists["matchup_archetype_refresh_launch_gate"] == "not_required_due_to_r248_deferral"
    assert specialists["adapter_slot_migration_required_status"] == "no_slot_change"
    assert specialists["new_matchup_slots_start_exact_zero_and_dormant"] is True
    for projection in (requirements, protocol, specialists):
        assert projection["direct_policy_scope"] == exclusion["scope"]
        assert projection["public_opponent_selector_change"] == "forbidden"
        assert projection["public_search_firewall"] == "not_introduced"
        for schema_key in (
            "activation_overlay_schema",
            "activation_overlay_mirror_schema",
            "activation_overlay_mirrors_schema",
            "owner_start_authorization_schema",
            "owner_start_authorization_generator_schema",
        ):
            assert projection[schema_key] == topology[schema_key]
        assert projection["logical_activation_overlay_cardinality"] == "one"
        assert projection["overlay_host_publication_identity"] == (
            "byte_identical_with_one_shared_sha256"
        )
        assert projection["external_activation_overlay_required"] is True
        assert projection["managed_training_start"] == (
            "requires_inzi_local_completed_dataset_transport_receipt_external_"
            "activation_overlay_and_completed_r260_side_store_migration_canary_"
            "evaluation_and_runtime_parity_receipts"
        )
        assert projection["parent_checkpoint_file_identity"] == {
            "path": (
                "/home/inzi/poke-bot-agent/outputs/pure_rl/"
                "alakazam_terminal_expert_bootstrap_no_rtp_r195/checkpoints/"
                "expert_before_iter_00021.pt"
            ),
            "sha256": "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a",
            "size_bytes": 127914385,
        }
        assert projection["inherited_learned_head_count_present"] == 19
        assert projection["inherited_active_non_combo_fusion_route_count"] == 18
        assert projection["new_typed_option_head_count"] == 2
        assert projection["total_architecture_head_count_after_import"] == 21
        assert projection["new_routes_outside_inherited_fusion_denominator"] == 2
        assert projection["own_deck_head_structure_import"] == imported
