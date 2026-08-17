from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_current_deck_guide_training_log_is_not_deck_mislabeled() -> None:
    source = (ROOT / "poke_bot/train.py").read_text(encoding="utf-8")
    assert "[rl-train] current-deck guide rows=" in source
    assert "Alakazam guide" not in source


def test_goal_gateway_is_stable_authority() -> None:
    goal = (ROOT / "GOAL.md").read_text(encoding="utf-8")
    compatibility = json.loads(
        (ROOT / "ops/current_goal_requirements.json").read_text(encoding="utf-8")
    )
    assert "Schema: `poke_bot.goal_gateway/v1`" in goal
    assert "## Design-change procedure" in goal
    assert compatibility["status"] == "compatibility_projection"
    assert compatibility["authoritative_sources"]["goal_gateway"] == "GOAL.md"
    assert "GOAL.md" in compatibility["replacement_goal_objective"]


def test_goal_gateway_references_existing_canonical_sources() -> None:
    for relative in (
        "docs/RL_TRAINING_PROTOCOL.md",
        "config/rl_protocol.yaml",
        "state/specialists.yaml",
        "state/matchup_adapter_roster.json",
        "ops/specialist_runtime_registry_v1.json",
        "ops/frozen_specialist_registry_v1.json",
        "ops/specialist_transition_graph.json",
    ):
        assert (ROOT / relative).is_file(), relative


def test_revision_192_stages_exact_marnie_splusplus_without_runtime_authority() -> None:
    goal = (ROOT / "GOAL.md").read_text(encoding="utf-8")
    compatibility = json.loads(
        (ROOT / "ops/current_goal_requirements.json").read_text(
            encoding="utf-8"
        )
    )
    typed = json.loads(
        (ROOT / "state/alakazam-marnie-splusplus-opponent-r192.json").read_text(
            encoding="utf-8"
        )
    )
    projected = compatibility["current_owner_overrides"][
        "alakazam_marnie_splusplus_opponent"
    ]

    assert "| 192 |" in goal
    assert typed["schema"] == "poke_bot.alakazam_marnie_splusplus_opponent_r192/v1"
    assert typed["status"] == (
        "staged_candidate_not_armed_pending_trainer_owned_fence_or_"
        "proven_inactive_boundary"
    )
    assert typed["opponent"]["opponent_id"] == (
        "specialist-marnie-final-format-h10-f20efb20f5c3"
    )
    assert typed["opponent"]["checkpoint_sha256"] == (
        "sha256:f20efb20f5c30820c7e23004e529d326ec87f91b026c1fe3"
        "bbb431f9c8b44381"
    )
    assert typed["opponent"]["tier"] == "S++"
    assert typed["opponent"]["weight"] == 4.0
    assert typed["opponent"]["floor_games_per_set"] == 1024
    assert typed["collection_contract"]["strong_public_practice_games"] == 4586
    assert typed["collection_contract"]["diverse_public_games"] == 2586
    assert typed["collection_contract"]["ordinary_strong_public_minimum_share"] == 0.04
    assert typed["historical_marnie"]["must_remain_distinct"] is True
    assert typed["activation"]["boundary"] == (
        "receipt_backed_inactive_boundary_or_trainer_owned_fence_enabled_"
        "clean_pause_after_completed_iteration5"
    )
    assert typed["activation"]["first_guaranteed_activation_boundary_completed_iteration"] == 5
    assert typed["activation"]["boundary_pause_seconds"] == 30
    assert typed["activation"]["allow_clean_boundary_design_migration"] is True
    assert typed["activation"]["boundary_design_migration_reason"] == (
        "owner_r192_marnie_splusplus_post_iteration5_receipt_backed_migration"
    )
    assert (
        typed["activation"][
            "managed_restart_during_verified_post_iteration5_hard_pause_allowed"
        ]
        is False
    )
    assert typed["activation"]["automatic_managed_restart_armed"] is False
    assert typed["activation"]["trainer_owned_handoff_fence_required"] is True
    assert (
        typed["activation"]["current_r175_source_has_trainer_owned_handoff_fence"]
        is False
    )
    assert (
        typed["activation"][
            "proven_inactive_receipt_boundary_alternative_required"
        ]
        is True
    )
    assert typed["activation"]["interrupt_active_collection_allowed"] is False
    assert typed["transport"] == {
        "r182_default_deny_unchanged": True,
        "other_r182_pairs_unchanged": True,
        "prior_pack4_eligible_group": "diverse_public",
        "activation_training_group": "strong_public_practice",
        "dispatch_mode": "singleton_remote_play",
        "pack4_attested_for_activation_group": False,
        "separate_exact_group_retention_attestation_required_for_pack4": True,
    }

    assert projected["goal_revision"] == 192
    assert projected["status"] == typed["status"]
    assert projected["typed_source"] == (
        "state/alakazam-marnie-splusplus-opponent-r192.json"
    )
    assert projected["opponent"]["opponent_id"] == typed["opponent"][
        "opponent_id"
    ]
    assert {
        key: projected["opponent"][key] for key in typed["opponent"]
    } == typed["opponent"]
    assert projected["opponent"]["checkpoint_sha256"] == typed["opponent"][
        "checkpoint_sha256"
    ]
    assert projected["opponent"]["content_digest"] == typed["opponent"][
        "content_digest"
    ]
    assert projected["opponent"]["tier"] == "S++"
    assert projected["opponent"]["weight"] == 4.0
    assert projected["opponent"]["floor_games_per_set"] == 1024
    assert projected["collection_contract"] == typed["collection_contract"]
    assert projected["transport"] == typed["transport"]
    assert projected["activation"]["boundary"] == typed["activation"]["boundary"]
    assert (
        projected["activation"][
            "first_guaranteed_activation_boundary_completed_iteration"
        ]
        == 5
    )
    assert projected["activation"]["boundary_pause_seconds"] == 30
    assert projected["activation"]["allow_clean_boundary_design_migration"] is True
    assert projected["activation"]["boundary_design_migration_reason"] == (
        "owner_r192_marnie_splusplus_post_iteration5_receipt_backed_migration"
    )
    assert projected["activation"]["activation_receipt"] is None
    assert projected["activation"]["active_before_receipt_backed_activation"] is False
    assert (
        projected["activation"][
            "managed_restart_during_verified_post_iteration5_hard_pause_allowed"
        ]
        is False
    )
    assert projected["activation"]["automatic_managed_restart_armed"] is False
    assert projected["activation"]["trainer_owned_handoff_fence_required"] is True
    assert (
        projected["activation"][
            "current_r175_source_has_trainer_owned_handoff_fence"
        ]
        is False
    )
    assert (
        projected["activation"][
            "proven_inactive_receipt_boundary_alternative_required"
        ]
        is True
    )
    assert projected["activation"]["training_restart_before_validation_allowed"] is False
    assert projected["activation"]["interrupt_active_collection_allowed"] is False


def test_active_core_fallback_matches_latest_accepted_v9_receipt() -> None:
    goal = (ROOT / "GOAL.md").read_text(encoding="utf-8")
    dashboard = (ROOT / "dashboard/lan/index.html").read_text(
        encoding="utf-8"
    )
    snapshot = (ROOT / "scripts/dashboard_snapshot.py").read_text(
        encoding="utf-8"
    )
    protocol = yaml.safe_load(
        (ROOT / "config/rl_protocol.yaml").read_text(encoding="utf-8")
    )
    state = yaml.safe_load(
        (ROOT / "state/specialists.yaml").read_text(encoding="utf-8")
    )
    compatibility = json.loads(
        (ROOT / "ops/current_goal_requirements.json").read_text(
            encoding="utf-8"
        )
    )

    handoff = protocol["shared_core_refresh"]["next_specialist_handoff"]
    accepted = state["current_cumulative_core_refresh"]["latest_accepted_core"]
    projected = compatibility["verified_snapshot"]["cumulative_core"]
    current_goal = goal.split("## Decision ledger", 1)[0]

    assert "Matchup Router Format 6" in current_goal
    assert "Accepted Policy Generation 9" in current_goal
    assert "latest checksum-accepted core is Core Generation 9" not in current_goal
    assert "V6 cumulative-core" not in goal
    assert "then cumulative core v6" not in goal
    assert handoff["fallback_core_selection"] == "latest_checksum_accepted_core"
    assert handoff["fallback_current_version"] == 9
    assert handoff["fallback_current_checkpoint"] == accepted["checkpoint"]
    assert handoff["fallback_current_checkpoint_checksum"] == accepted["checksum"]
    assert projected["latest_accepted_version"] == 9
    assert projected["latest_accepted_checkpoint"] == accepted["checkpoint"]
    assert projected["latest_accepted_checkpoint_digest"] == accepted["checksum"]
    assert projected["post_thwackey_v10_status"] == (
        "rejected_gameplay_regression"
    )
    namespaces = compatibility["current_owner_overrides"][
        "version_namespaces"
    ]
    assert namespaces["goal_revision"] == 66
    assert namespaces["core_system"] == {
        "display_namespace": "Training Core Revision",
        "current_revision": 10,
        "meaning": "current_training_and_control_implementation",
    }
    assert namespaces["cumulative_core"] == {
        "display_namespace": "Accepted Policy Generation",
        "latest_accepted_version": 9,
        "latest_attempted_version": 14,
        "v10_status": "rejected_gameplay_regression",
        "v11_status": "rejected_pretraining_validation",
        "v12_status": "rejected_pretraining_validation",
        "v13_status": "rejected_pretraining_validation",
        "v14_status": "rejected_pretraining_validation",
    }
    assert namespaces["matchup_adapter"]["checkpoint_format_version"] == 6
    assert namespaces["runtime_modified"] is False
    recovery = compatibility["current_owner_overrides"][
        "teal_iteration_7_memory_recovery"
    ]
    assert recovery["goal_revision"] == 47
    assert recovery["next_start_worker_ceiling"] == 48
    assert recovery["next_start_games_in_flight_ceiling"] == 48
    assert [
        recovery["rebalance_worker_floor"],
        recovery["rebalance_worker_ceiling"],
    ] == [32, 48]
    assert recovery["free_ram_floor_gib"] == 12
    assert recovery["memory_high_gib"] == 100
    assert recovery["memory_max_gib"] == 116
    assert recovery["manual_restart_used"] is False
    assert recovery["first_automatic_recovery_main_pid"] == 2307796
    assert recovery["current_main_pid"] == 2864742
    assert recovery["automatic_restart_count"] == 2
    assert recovery["next_start_safety_contract_activated"] is True
    collection_failure = recovery["iteration_8_exact_collection_failure"]
    assert collection_failure["retained_games"] == 8146
    assert collection_failure["required_games"] == 8192
    assert collection_failure["public_mix_games"] == 7122
    assert collection_failure["required_public_mix_games"] == 7168
    assert collection_failure["immutable_commit_created"] is False
    current_profile = protocol["training_hardware_throughput"][
        "next_boundary_blackwell_profile"
    ]
    assert current_profile["simulator_workers"] == 96
    assert current_profile["games_in_flight"] == 96
    assert [
        current_profile["worker_floor"],
        current_profile["worker_target"],
        current_profile["worker_ceiling"],
    ] == [96, 96, 96]
    assert current_profile["adaptive_worker_rebalance_allowed"] is False
    assert current_profile["automatic_worker_fallback_allowed"] is False
    assert current_profile["lower_profile_substitution_allowed"] is False
    assert current_profile["minimum_host_available_ram_gib"] == 12
    assert current_profile["memory_high_gib"] == 100
    active_run = state["current"]["active_run"]
    assert active_run["active_specialist"] == "crustle"
    assert active_run["decision_fusion_schema"] == (
        "poke_bot.causal_decision_fusion/v3"
    )
    assert active_run["required_fused_head_count"] == 19
    assert "attemptedCoreStatuses" in dashboard
    assert "Training Core Revision" in dashboard
    assert "CURRENT CORE V6" not in dashboard
    assert "Accepted V6" not in dashboard
    assert "VERSIONED SYSTEMS" in dashboard
    assert "checkpointDisplayNamespace" in dashboard
    assert (
        "ATTEMPTED THROUGH '+checkpointDisplayNamespace.toUpperCase()+' '+attemptedCoreVersion"
        in dashboard
        or "attemptedCoreSummary" in dashboard
    )
    assert (
        "Matchup Router Format 6" in dashboard
        or "separate namespace; not a cumulative-core generation or plan step"
        in dashboard
    )
    assert "NEXT BOUNDARY V6 STRATEGIC CORPUS" not in snapshot
    assert "NEXT BOUNDARY EXPANDED STRATEGIC CORPUS" in snapshot
    assert '"Accepted Policy Generation 15 · "' in snapshot


def test_teal_mask_has_two_turn_order_submission_profiles() -> None:
    goal = (ROOT / "GOAL.md").read_text(encoding="utf-8")
    protocol = yaml.safe_load(
        (ROOT / "config/rl_protocol.yaml").read_text(encoding="utf-8")
    )
    state = yaml.safe_load(
        (ROOT / "state/specialists.yaml").read_text(encoding="utf-8")
    )
    registry = json.loads(
        (ROOT / "ops/specialist_runtime_registry_v1.json").read_text(
            encoding="utf-8"
        )
    )
    projection = json.loads(
        (ROOT / "ops/current_goal_requirements.json").read_text(
            encoding="utf-8"
        )
    )
    expected = [
        {"copy_number": 1, "turn_order_preference": "first_if_allowed"},
        {"copy_number": 2, "turn_order_preference": "second_if_allowed"},
    ]

    assert "| 31 | 2026-07-29 |" in goal
    profile = protocol["kaggle"]["specialist_submission_profiles"][
        "teal-mask-ogerpon-ex"
    ]
    assert profile["required_copies"] == 2
    assert profile["copy_profiles"] == expected
    teal = next(
        row
        for row in state["specialists"]
        if row["id"] == "teal-mask-ogerpon-ex"
    )
    assert teal["kaggle_submission_profile"]["copies"] == [
        {
            **expected[0],
            "status": "submitted",
            "submission_id": 55114866,
        },
        {
            **expected[1],
            "status": "submitted",
            "submission_id": 55114885,
        },
    ]
    runtime_profile = registry["pass_handler"][
        "specialist_submission_profiles"
    ]["teal-mask-ogerpon-ex"]
    assert runtime_profile["submission_count"] == 2
    assert runtime_profile["turn_order_preferences"] == [
        "first_if_allowed",
        "second_if_allowed",
    ]
    assert projection["current_owner_overrides"][
        "teal_mask_dual_turn_order_submissions"
    ]["copy_profiles"] == expected


def test_hops_has_one_owner_authorized_second_preference_submission() -> None:
    goal = (ROOT / "GOAL.md").read_text(encoding="utf-8")
    registry = json.loads(
        (ROOT / "ops/specialist_runtime_registry_v1.json").read_text(
            encoding="utf-8"
        )
    )
    projection = json.loads(
        (ROOT / "ops/current_goal_requirements.json").read_text(
            encoding="utf-8"
        )
    )

    assert "| 33 | 2026-07-29 |" in goal
    request = registry["pass_handler"][
        "owner_authorized_one_off_submissions"
    ]["hops-trevenant-second-if-allowed-v1"]
    assert request["specialist_id"] == "hops-trevenant"
    assert request["submission_count"] == 1
    assert request["turn_order_preferences"] == ["second_if_allowed"]
    assert request["submission_id"] == 55088551
    assert request["kaggle_status"] in {"pending", "complete"}
    assert request["exact_frozen_checkpoint_checksum"] == (
        "sha256:462f201f8de6c07eef07b3e8f58229360972d1d64308db9c155f211d2ce3faf1"
    )
    projected = projection["current_owner_overrides"][
        "hops_second_preference_submission"
    ]
    assert projected["turn_order_preference"] == "second_if_allowed"
    assert projected["historical_submissions_replaced"] is False
    assert projected["submission_id"] == 55088551
    assert projected["authorization_consumed_before_upload"] is True


def test_hammer_terminal_failure_submits_latest_and_moves_to_teal() -> None:
    goal = (ROOT / "GOAL.md").read_text(encoding="utf-8")
    protocol = yaml.safe_load(
        (ROOT / "config/rl_protocol.yaml").read_text(encoding="utf-8")
    )
    state = yaml.safe_load(
        (ROOT / "state/specialists.yaml").read_text(encoding="utf-8")
    )
    registry = json.loads(
        (ROOT / "ops/specialist_runtime_registry_v1.json").read_text(
            encoding="utf-8"
        )
    )
    projection = json.loads(
        (ROOT / "ops/current_goal_requirements.json").read_text(
            encoding="utf-8"
        )
    )

    assert "| 37 | 2026-07-29 |" in goal
    gate = protocol["specialist_training"]["gate_window"]
    assert gate["ceiling_completed_iteration"] == 15
    assert gate["ceiling_behavior"] == (
        "freeze_submit_and_continue_without_false_pass"
    )
    assert gate["ceiling_checkpoint_must_be_exact"] is True
    assert gate["ceiling_failed_gate_results_must_be_preserved"] is True
    assert registry["pass_handler"]["ceiling_behavior"] == (
        "freeze_submit_and_continue_without_false_pass"
    )
    fallback = state["current"]["hammer_pult_terminal_owner_fallback"]
    assert fallback["checkpoint_selection"] == (
        "latest_exact_immutable_iteration_15_checkpoint"
    )
    assert fallback["completion_status"] == "ceiling_accepted"
    assert fallback["next_specialist"] == "teal-mask-ogerpon-ex"
    assert fallback["retry_or_extend_hammer_pult"] is False
    projected = projection["current_owner_overrides"][
        "hammer_pult_terminal_fallback"
    ]
    assert projected["runtime_handler_flag"] == (
        "--accept-ceiling-and-continue"
    )
    assert projected["invalid_or_incomplete_checkpoint_or_receipt_behavior"] == (
        "fail_closed"
    )


def test_kaggle_spacing_uses_second_most_recent_submission() -> None:
    goal = (ROOT / "GOAL.md").read_text(encoding="utf-8")
    policy = json.loads(
        (ROOT / "ops/kaggle_submission_policy.json").read_text(
            encoding="utf-8"
        )
    )
    projection = json.loads(
        (ROOT / "ops/current_goal_requirements.json").read_text(
            encoding="utf-8"
        )
    )
    processor = (
        ROOT / "scripts/process_kaggle_submission_queue.py"
    ).read_text(encoding="utf-8")

    assert "| 38 | 2026-07-29 |" in goal
    assert policy["minimum_hours_between_submissions"] == 4
    assert policy["spacing_anchor"] == (
        "second_most_recent_logical_submission"
    )
    projected = projection["current_owner_overrides"][
        "kaggle_spacing_anchor"
    ]
    assert projected["goal_revision"] == 38
    assert projected["anchor"] == policy["spacing_anchor"]
    assert projected["daily_submission_limit_unchanged"] == 5
    assert "spacing_anchor_submission_at" in processor


def test_hammer_iteration_15_is_ceiling_accepted_and_frozen() -> None:
    state = yaml.safe_load(
        (ROOT / "state/specialists.yaml").read_text(encoding="utf-8")
    )
    projection = json.loads(
        (ROOT / "ops/current_goal_requirements.json").read_text(
            encoding="utf-8"
        )
    )
    hammer = next(
        row for row in state["specialists"] if row["id"] == "hammer-pult"
    )
    terminal = hammer["terminal_completion"]
    completed = hammer["terminal_iteration_15"]
    projected = projection["verified_snapshot"]["hammer_pult_terminal_completion"]

    assert hammer["status"] == "passed_frozen"
    assert hammer["active"] is False
    assert hammer["counters"]["last_completed_iteration"] == 15
    assert hammer["counters"]["next_iteration"] is None
    assert completed["iteration"] == 15
    assert completed["stage"] == (
        "iteration_15_committed_gate_failed_ceiling_accepted_frozen"
    )
    assert completed["collection_status"] == "committed"
    assert terminal["completion_authority"] == (
        "explicit_owner_ceiling_acceptance"
    )
    assert terminal["measured_gate_passed"] is False
    assert terminal["exact_checkpoint_checksum"] == (
        "sha256:c256a0ababee147a09d87773c74e2aa11cf46f04c492a1c20b3fb6ead8da0dce"
    )
    assert terminal["skill_weighted_win_rate"] == 0.48738461538461536
    assert terminal["confidence_lower"] == 0.4743076923076923
    assert terminal["causal_runtime_all_enabled"] is True
    assert terminal["kaggle_queue_status"] == "accepted"
    assert terminal["kaggle_submission_id"] == 55090161
    assert terminal["kaggle_public_score"] == 594.5
    assert projected["terminal_iteration"] == 15
    assert projected["status"] == "ceiling_accepted"
    assert projected["checkpoint_sha256"] == terminal["exact_checkpoint_checksum"]


def test_spidops_is_the_fail_closed_successor_after_thwackey() -> None:
    goal = (ROOT / "GOAL.md").read_text(encoding="utf-8")
    human = (ROOT / "docs/RL_TRAINING_PROTOCOL.md").read_text(
        encoding="utf-8"
    )
    state = yaml.safe_load(
        (ROOT / "state/specialists.yaml").read_text(encoding="utf-8")
    )
    cycle = json.loads(
        (ROOT / "ops/specialist_cycle_handoff_v1.json").read_text(
            encoding="utf-8"
        )
    )
    compatibility = json.loads(
        (ROOT / "ops/current_goal_requirements.json").read_text(
            encoding="utf-8"
        )
    )

    goal_revision = int(goal.split("Revision: `", 1)[1].split("`", 1)[0])
    assert goal_revision >= 117
    assert "mandatory and sole successor after Thwackey" in goal
    assert state["current"]["active_specialist"] is None
    assert state["current"]["active_run"]["active_specialist"] == "crustle"
    assert state["current"]["transition_source_specialist"] == "slowking"
    assert state["current"]["staged_successor_specialist"] is None
    assert state["training_priority"][
        "ordered_unfinished_ids_after_active"
    ] == []
    assert cycle["selection"]["strict_priority_prefix"][-4:] == [
        "team-rockets-spidops",
        "hammer-pult",
        "teal-mask-ogerpon-ex",
        "archaludon-ex",
    ]
    assert cycle["selection"]["minimum_records_by_specialist"] == {
        "team-rockets-spidops": 16_639
    }
    assert "never permits fall-through" in human
    projected = compatibility["current_owner_overrides"][
        "team_rockets_spidops_successor"
    ]
    assert projected["mandatory_next_specialist"] == "team-rockets-spidops"
    assert projected["minimum_acting_seat_games"] == 16_639
    assert projected["lower_priority_fallthrough_allowed"] is False
    assert projected["activation_status"] == (
        "completed_frozen_successor_handoff_complete"
    )
    assert projected["hammer_pult_selected"] is True


def test_owner_required_plan_and_post_spidops_order_cannot_regress() -> None:
    goal = (ROOT / "GOAL.md").read_text(encoding="utf-8")
    protocol = yaml.safe_load(
        (ROOT / "config/rl_protocol.yaml").read_text(encoding="utf-8")
    )
    state = yaml.safe_load(
        (ROOT / "state/specialists.yaml").read_text(encoding="utf-8")
    )
    compatibility = json.loads(
        (ROOT / "ops/current_goal_requirements.json").read_text(
            encoding="utf-8"
        )
    )

    removed = [
        "dragapult",
        "dragapult-blaziken",
        "dragapult-dudunsparce",
        "walrein",
    ]
    order = [
        "hammer-pult",
        "teal-mask-ogerpon-ex",
        "archaludon-ex",
    ]
    required_specialist_ids = [
        row["id"]
        for row in state["specialists"]
        if row.get("required_specialist") is not False
    ]
    priority = state["training_priority"]
    selection = protocol["specialist_training"]["selection_order"]
    projected = compatibility["current_owner_overrides"][
        "required_specialist_plan"
    ]

    assert "Remove `dragapult-blaziken` and `dragapult-dudunsparce`" in goal
    assert state["current"]["program_progress"]["required_specialists_total"] == 15
    assert state["current"]["active_specialist"] is None
    assert state["current"]["active_run"]["active_specialist"] == "crustle"
    assert state["current"]["transition_source_specialist"] == "slowking"
    assert state["current"]["staged_successor_specialist"] is None
    assert priority["ordered_unfinished_ids_after_active"] == []
    assert priority["strict_post_spidops_prefix"]["ids"] == order
    assert priority["strict_post_spidops_prefix"]["status"] == (
        "strict_prefix_completed_postfleet_marnie_active_revision109"
    )
    assert priority["owner_removal"]["status"] == (
        "active_selector_and_dashboard_projection"
    )
    assert all(
        specialist_id not in required_specialist_ids
        for specialist_id in removed
    )
    matchup_roster = json.loads(
        (ROOT / "state/matchup_adapter_roster.json").read_text(
            encoding="utf-8"
        )
    )
    assert matchup_roster["specialist_priority"][7:10] == order
    assert all(
        specialist_id in matchup_roster["expert_ids"]
        for specialist_id in removed
    )
    assert "crustle" in matchup_roster["expert_ids"]
    crustle_slot = next(
        row
        for row in matchup_roster["slots"]
        if row["archetype_id"] == "crustle"
    )
    assert crustle_slot == {
        "slot": 0,
        "archetype_id": "crustle",
        "status": "active",
        "lineage": "v5:0",
    }
    assert selection["owner_removed_specialist_ids"] == removed
    assert selection["owner_removed_specialists_are_selection_eligible"] is False
    assert selection["owner_removed_specialists_count_toward_completion"] is False
    assert projected["strict_post_spidops_prefix"] == order
    assert projected["active_specialist"] is None
    assert projected["transition_source_specialist"] == order[2]
    assert projected["staged_successor_specialist"] is None
    assert projected["removed_specialist_ids"] == removed
    assert projected["required_specialists_total"] == 15


def test_post_fleet_refresh_is_ordered_versioned_and_non_intrusive() -> None:
    goal = (ROOT / "GOAL.md").read_text(encoding="utf-8")
    protocol = yaml.safe_load(
        (ROOT / "config/rl_protocol.yaml").read_text(encoding="utf-8")
    )
    state = yaml.safe_load(
        (ROOT / "state/specialists.yaml").read_text(encoding="utf-8")
    )
    compatibility = json.loads(
        (ROOT / "ops/current_goal_requirements.json").read_text(
            encoding="utf-8"
        )
    )
    cycle = json.loads(
        (ROOT / "ops/specialist_cycle_handoff_v1.json").read_text(
            encoding="utf-8"
        )
    )
    runtime_registry = (
        ROOT / "ops/specialist_runtime_registry_v1.json"
    ).read_text(encoding="utf-8")
    frozen_registry = (
        ROOT / "ops/frozen_specialist_registry_v1.json"
    ).read_text(encoding="utf-8")
    prestage = (ROOT / "ops/next_specialist_prestage.env").read_text(
        encoding="utf-8"
    )
    order = ["alakazam", "marnie-s-grimmsnarl-ex", "crustle"]
    canonical = protocol["specialist_training"]["post_fleet_refresh"]
    projected = compatibility["current_owner_overrides"][
        "post_fleet_alakazam_grimms_refresh"
    ]
    mutable = state["post_fleet_refresh"]

    goal_revision = int(goal.split("Revision: `", 1)[1].split("`", 1)[0])
    assert goal_revision >= 117
    assert "new separately versioned refreshes in strict order" in goal
    assert protocol["allowed_phases"][-3] == (
        "post_fleet_specialist_refresh"
    )
    assert canonical["ordered_specialist_ids"] == order
    assert mutable["ordered_specialist_ids"] == order
    assert projected["ordered_specialist_ids"] == order
    assert cycle["post_fleet_refresh"]["ordered_specialist_ids"] == order
    trigger_key = (
        "slowking_terminal_disposition_immediately_triggers_first_refresh"
    )
    assert canonical["trigger"][trigger_key] is True
    assert mutable["trigger"][trigger_key] is True
    assert projected["trigger"][trigger_key] is True
    assert cycle["post_fleet_refresh"][trigger_key] is True
    cycle_refresh = cycle["post_fleet_refresh"]
    for contract in (canonical, mutable, projected, cycle_refresh):
        first = contract["first_refresh"]
        migration = first["preferred_parent_migration"]
        turn_order = first["turn_order"]
        gates = contract["release_gates"]
        assert first["specialist_id"] == "alakazam"
        assert first["start_timing"] == (
            "immediately_after_slowking_failed_experiment_receipt_and_g0_g1"
        )
        assert first["model_format"] == "final_submission_format"
        assert first["first_final_format_model"] is True
        assert first[
            "final_format_computation_before_slowking_terminal_disposition_allowed"
        ] is False
        assert first["silent_legacy_format_fallback_allowed"] is False
        assert migration["parent_checkpoint"] == (
            "immutable_existing_alakazam"
        )
        assert migration[
            "require_checksum_bound_shape_and_key_coverage"
        ] is True
        assert migration["genuinely_new_structures_initialization"] == (
            "zero_safe"
        )
        assert migration["step_zero_parity_receipt_required"] is True
        assert migration["causal_validation_receipt_required"] is True
        assert migration["original_checkpoint_may_be_rewritten"] is False
        assert migration["failure_fallback"] == {
            "migration_failure_receipt_preserved": True,
            "ordinary_same_archetype_alakazam_refresh_initialized_from": (
                "then_latest_checksum_accepted_core"
            ),
            "expand_only_that_completed_alakazam_derivative_to_final_format": (
                True
            ),
            "latest_core_direct_final_format_tensor_parent_allowed": False,
            "partial_old_alakazam_core_overlay_allowed": False,
        }
        assert turn_order["training_seat_split"] == {
            "first": 0.5,
            "second": 0.5,
        }
        assert turn_order["exact_even_split_required"] is True
        assert turn_order["deterministic_assignment_required"] is True
        assert turn_order["seat_count_parity_receipt_required"] is True
        assert turn_order["seat_count_parity_receipt_schema"] == (
            "poke_bot.alakazam_refresh_seat_split/v1"
        )
        assert turn_order["seat_count_receipt_required_stages"] == [
            "assigned",
            "actual",
            "consumed",
        ]
        assert turn_order[
            "equal_first_second_counts_required_at_each_stage"
        ] is True
        assert turn_order["package_preference"] == "first_if_allowed"
        assert turn_order["second_focus_1_to_7_allowed"] is False
        assert turn_order["always_second_arm_allowed"] is False
        assert turn_order["second_preferring_refresh_copy_allowed"] is False
        assert gates["final_alakazam_model_computation"][
            "required_receipts"
        ] == [
            "required_specialist_fleet_complete_for_final_alakazam_v1",
            "capacity_research_resource_lease_v1",
        ]
        assert gates["broader_multi_archetype_capacity_program"][
            "required_receipt"
        ] == "post_refresh_sequence_complete_for_capacity_v2"
    assert canonical["core_hot_start"]["resolve_at_each_refresh_start"] == (
        "latest_checksum_accepted_cumulative_core"
    )
    assert mutable["core_hot_start"]["pinned_core_generation"] is None
    assert projected["pinned_core_generation"] is None
    assert canonical["training_structure_resolution"][
        "resolve_at_each_refresh_start"
    ] == "current_canonical_training_contracts"
    assert projected["pinned_training_schema_digests"] == []
    assert mutable["completed_refresh_specialist_ids"] == ["alakazam"]
    assert mutable["active_refresh_specialist_id"] == "marnie-s-grimmsnarl-ex"
    assert mutable["next_refresh_specialist_id"] == "marnie-s-grimmsnarl-ex"
    assert mutable["refresh_model_versions"]["alakazam"] == (
        "final-format-alakazam-r79-h10-v1"
    )
    assert projected["completed_refresh_specialist_ids"] == ["alakazam"]
    assert projected["active_refresh_specialist_id"] == (
        "marnie-s-grimmsnarl-ex"
    )
    assert state["current"]["program_progress"][
        "required_specialists_total"
    ] == 15
    assert len(state["specialists"]) == 17
    assert state["current"]["active_specialist"] is None
    assert state["current"]["active_run"]["active_specialist"] == "crustle"
    assert state["current"]["staged_successor_specialist"] is None
    assert state["training_priority"][
        "ordered_unfinished_ids_after_active"
    ] == []
    assert "PRESTAGE_SPECIALIST_ID=alakazam" not in prestage
    assert compatibility["authoritative_sources"]["live_selector"].startswith(
        "/home/pokebot/poke-bot-agent/outputs/final_format_marnie_r104/runtime/"
    )
    assert projected["required_specialist_count_modified"] is False
    assert projected["current_selector_modified"] is False
    assert projected["current_prestage_modified"] is False
    assert projected["current_runtime_registry_modified"] is False
    assert state["population_training"]["enabled"] is False
    assert state["population_training"][
        "blocked_until_post_fleet_refresh_complete"
    ] is True
    assert protocol["population_phase"][
        "requires_post_fleet_refresh_complete"
    ] is True

    originals = mutable["original_checkpoint_identities"]
    assert originals["alakazam"]["checksum"] == (
        "sha256:270b5156781b0a95f703abe3e8fe13866d2fbb4c85a8f325"
        "34f99af74aece2ea"
    )
    assert originals["marnie-s-grimmsnarl-ex"]["checksum"] == (
        "sha256:52a5207e4c98dce80b49b6403cbb17f14d6fc4d2ac5b6255"
        "32020a1a25f233ac"
    )
    assert all(row["immutable"] for row in originals.values())
    phase_id = "post-fleet-alakazam-grimms-crustle-v2"
    assert phase_id not in runtime_registry
    assert phase_id not in frozen_registry
    assert phase_id not in prestage


def test_teal_mask_ogerpon_uses_exact_slop_box_source_identity() -> None:
    state = yaml.safe_load(
        (ROOT / "state/specialists.yaml").read_text(encoding="utf-8")
    )
    row = next(
        value
        for value in state["specialists"]
        if value["id"] == "teal-mask-ogerpon-ex"
    )

    assert row["source_archetype_id"] == "slop-box"
    assert row["competitive_family_alias"] == "Raging Bolt Ogerpon"
    assert row["datasets"]["source_indexed_public_acting_seat_games"] == 1_442
    assert row["datasets"]["materialized_acting_seat_games"] == 1_442
    assert row["datasets"]["materialized_decisions"] == 97_762
    assert row["datasets"]["materialized_guide_rows"] == 8_672
    assert row["datasets"]["daily_receipts_verified"] == 33
    assert row["datasets"]["duplicate_episode_seat_keys"] == 0
    assert row["datasets"]["corpus_identity_status"] == (
        "slop_box_full33_ready_checksum_validated_imported"
    )
    assert row["datasets"]["clean_rebuild"]["status"] == (
        "ready_checksum_validated_imported"
    )
    assert row["datasets"]["contaminated_v1"]["status"] == (
        "quarantined_not_promotable"
    )
    assert row["datasets"]["contaminated_v1"]["records"] == 2_300
    assert row["datasets"]["contaminated_v1"]["decisions"] == 156_692
    assert row["datasets"]["contaminated_v1"]["guide_rows"] == 10_495
    assert row["datasets"]["public_deck_catalog"].endswith(
        "teal-mask-ogerpon-ex-public-full33.v1.json"
    )
    projected = json.loads(
        (ROOT / "ops/current_goal_requirements.json").read_text(
            encoding="utf-8"
        )
    )["current_owner_overrides"]["teal_mask_ogerpon_ex"]
    assert projected["competitive_taxonomy_id"] == "slop-box"
    assert projected["physical_public_archetype_id"] == 151
    assert projected["physical_public_archetype_name"] == (
        "Teal Mask Ogerpon ex"
    )
    assert projected["source_indexed_public_acting_seat_games"] == 1_442
    assert projected["materialized_public_acting_seat_games"] == 1_442
    assert projected["materialized_decisions"] == 97_762
    assert projected["materialized_guide_rows"] == 8_672
    assert projected["daily_receipts_verified"] == 33
    assert projected["corpus_promotion_status"] == (
        "full33_v4_active_bootstrap_bound"
    )
    assert projected["public_deck_catalog"].endswith(
        "teal-mask-ogerpon-ex-public-full33.v1.json"
    )
    assert projected["pending_full33_extension"]["expected_records"] == 1_442
    assert projected["pending_full33_extension"]["prestage_status"] == (
        "completed_imported_and_activated"
    )
    assert projected["contaminated_v1"]["status"] == (
        "quarantined_not_promotable"
    )
    assert row["representative"]["exact_card_count"] == 60
    assert row["representative"]["physical_source_identity"] == "slop-box"
    assert row["representative"]["competitive_family_alias"] == (
        "raging-bolt-ogerpon"
    )
    assert row["transition"]["strict_predecessor"] == "hammer-pult"
    assert row["transition"]["strict_successor"] == "archaludon-ex"


def test_archaludon_revision69_existing_corpus_override_is_explicit() -> None:
    state = yaml.safe_load(
        (ROOT / "state/specialists.yaml").read_text(encoding="utf-8")
    )
    row = next(
        value
        for value in state["specialists"]
        if value["id"] == "archaludon-ex"
    )
    projected = json.loads(
        (ROOT / "ops/current_goal_requirements.json").read_text(
            encoding="utf-8"
        )
    )["current_owner_overrides"]["archaludon_ex"]

    assert row["status"] == "passed_frozen"
    assert row["active"] is False
    assert row["prestage_status"] == (
        "historical_completed_owner_revision69_existing_corpus_override"
    )
    assert row["heads"]["current_deck_guide"]["guide_corpus_status"] == (
        "historical_training_input_from_existing_protected_import"
    )
    assert row["datasets"]["records"] == 1_458
    assert row["datasets"]["decisions"] == 83_980
    assert row["datasets"]["guide_rows"] == 1_454
    assert row["datasets"]["current_deck_guide_corpus_ready"] is True
    assert row["datasets"]["satisfies_revision68_minimum_game_floor"] is False
    assert row["datasets"]["launch_authority"] == "owner_revision69"
    assert row["datasets"]["dataset_schema_required"] == 7
    assert row["datasets"]["audited_source_matching_acting_seats"] == 21_278
    assert row["superseded_schema6_datasets"]["records"] == 1_458
    assert row["superseded_schema6_datasets"][
        "current_deck_guide_corpus_ready"
    ] is True
    assert row["superseded_schema6_datasets"]["status"] == (
        "historical_completed_run_exception_by_owner_revision69"
    )
    assert projected["training_status"] == "completed_frozen"
    assert projected["guide_corpus_status"] == (
        "historical_training_input_from_existing_protected_import"
    )
    assert projected["materialized_records"] == 1_458
    assert projected["materialized_decisions"] == 83_980
    assert projected["materialized_guide_rows"] == 1_454
    assert projected["dataset_schema_required"] == 7
    assert projected["audited_source_matching_acting_seats"] == 21_278
    assert projected["active_teal_mask_ogerpon_ex_modified"] is False
    assert projected["prestage_status"] == (
        "historical_completed_owner_revision69_existing_corpus_override"
    )
    assert projected["full_public_identity_ready"] is False
    assert projected["full_public_guide_ready"] is False
    assert projected["schema7_import_ready"] is False
    assert projected["activation_status"] == (
        "completed_frozen_handoff_to_slowking"
    )
    assert projected["superseded_schema6"][
        "satisfies_revision68_minimum_game_floor"
    ] is False


def test_clean_specialist_transition_contract_cannot_regress() -> None:
    goal = (ROOT / "GOAL.md").read_text(encoding="utf-8")
    human = (ROOT / "docs/RL_TRAINING_PROTOCOL.md").read_text(encoding="utf-8")
    protocol = yaml.safe_load(
        (ROOT / "config/rl_protocol.yaml").read_text(encoding="utf-8")
    )
    transition = protocol["shared_core_refresh"]["next_specialist_handoff"][
        "clean_transition"
    ]

    assert "Specialist transitions must be clean and automatic" in goal
    assert "Every specialist transition must be clean and automatic" in human
    assert transition == {
        "required": True,
        "preflight_before_specialist_start": True,
        "preflight_training_launch_path": True,
        "preflight_terminal_freeze_package_submission_handoff_path": True,
        "exact_logical_specialist_representative_required": True,
        "representative_card_count": 60,
        "deterministic_transition_input_error_timing": "before_rl_training",
        "successful_trainer_exit_action": "start_idempotent_gate_handler",
        "periodic_gate_supervisor_role": "recovery_only",
        "periodic_gate_supervisor_is_normal_transition_path": False,
        "submission_completion_blocks_handoff": False,
    }


def test_deck_guide_sme_workflow_cannot_regress() -> None:
    goal = (ROOT / "GOAL.md").read_text(encoding="utf-8")
    human = (ROOT / "docs/RL_TRAINING_PROTOCOL.md").read_text(encoding="utf-8")
    human_flat = " ".join(human.split())
    protocol = yaml.safe_load(
        (ROOT / "config/rl_protocol.yaml").read_text(encoding="utf-8")
    )
    compatibility = json.loads(
        (ROOT / "ops/current_goal_requirements.json").read_text(encoding="utf-8")
    )

    guide = protocol["specialist_training"]["current_deck_guide"]
    writeup = guide["expert_writeup"]
    execution = guide["research_execution"]
    projection = compatibility["verified_snapshot"]["current_deck_guides"]

    assert "no more than 10,000 words" in goal
    assert "exactly those two responsibilities" in goal
    assert "no more than 10,000 words" in human_flat
    assert "exactly two tasks" in human_flat
    assert writeup["required"] is True
    assert writeup["maximum_words"] == 10000
    assert writeup["primary_document_type"] == (
        "practical_human_deck_pilot_guide"
    )
    assert writeup["how_to_play_is_primary"] is True
    assert writeup["heuristic_audit_location"] == "short_appendix_only"
    assert writeup["heuristic_audit_may_be_primary_content"] is False
    assert set(writeup["required_play_sections"]) >= {
        "setup_and_opening_priorities",
        "going_first_plan",
        "going_second_plan",
        "turn_by_turn_sequencing",
        "resource_and_energy_planning",
        "attack_selection",
        "bench_management",
        "prize_mapping",
        "matchup_plans",
        "recovery_lines",
        "common_mistakes",
        "decision_checklists",
    }
    assert writeup["checksum_validation_required_before_prestage_ready"] is True
    assert execution["executor"] == "dedicated_subagent"
    assert execution["reasoning_tier"] == "highest_available"
    assert execution["exact_task_scope"] == [
        "expert_facing_guide",
        "causal_heuristic_extraction",
    ]
    assert execution["additional_tasks_allowed"] is False
    assert execution["production_runtime_mutation_allowed"] is False
    corpus_binding = guide["corpus_binding"]
    assert corpus_binding["ready_receipt_schema"] == (
        "poke_bot.current_deck_guide_corpus_ready/v1"
    )
    assert corpus_binding["selected_expert_manifest_checksum_required"] is True
    assert corpus_binding["selected_protected_pointer_checksum_required"] is True
    assert corpus_binding[
        "contract_guide_rows_must_equal_selected_manifest_guide_rows"
    ] is True
    assert corpus_binding[
        "all_daily_shard_checksums_required_before_atomic_promotion"
    ] is True
    assert corpus_binding["cpu_pack_build_allowed_before_binding_ready"] is False
    assert corpus_binding["active_training_modification_allowed"] is False
    assert projection["expert_writeup"]["maximum_words"] == 10000
    assert projection["expert_writeup"]["how_to_play_is_primary"] is True
    assert projection["expert_writeup"]["heuristic_audit_location"] == (
        "short_appendix_only"
    )
    assert projection["research_execution"]["exact_task_scope"] == (
        execution["exact_task_scope"]
    )
    assert projection["corpus_binding"][
        "selected_manifest_and_pointer_checksums_required"
    ] is True
    assert projection["corpus_binding"][
        "contract_rows_must_equal_selected_manifest_rows"
    ] is True
    assert projection["corpus_binding"][
        "cpu_pack_deferred_until_binding_ready"
    ] is True

    goal_path = guide["goal_path_guidance"]
    assert goal_path["owner_vision_required"] is True
    assert goal_path["applies_to_every_future_specialist_training_run"] is True
    assert goal_path[
        "applies_retroactively_to_completed_frozen_or_started_runs"
    ] is False
    assert goal_path["owner_decision_revision"] == 43
    assert goal_path["prospective_scope_revision"] == 44
    assert goal_path["prospective_effective_specialist"] == "archaludon-ex"
    assert goal_path["historical_weight_or_receipt_rewrite_allowed"] is False
    assert goal_path["active_teal_revision42_exception_preserved"] is True
    assert goal_path["bootstrap_behavior"] == "ramp_in"
    assert goal_path["positive_contribution_behavior"] == "ramp_then_hold"
    assert goal_path["internalized_or_nonpositive_behavior"] == (
        "anneal_toward_zero"
    )
    assert goal_path["curve_shape"] == (
        "rapid_ramp_then_positive_plateau_then_evidence_driven_decay"
    )
    assert goal_path["maximum_auxiliary_loss_weight"] == 0.50
    assert goal_path["post_bootstrap_positive_ramp_steps"] == [
        0.15,
        0.25,
        0.35,
        0.50,
    ]
    assert goal_path["all_weight_changes_require_clean_boundary_receipt"] is True
    ramp = goal_path["corrective_clean_boundary_ramp"]
    assert ramp["owner_decision_revision"] == 42
    assert ramp["specialist_id"] == "teal-mask-ogerpon-ex"
    assert ramp["observed_weight"] == 0.05
    assert ramp["target_weight"] == 0.25
    assert ramp["activation_boundary_next_iteration"] == 6
    assert ramp["preserve_inflight_iteration_design"] is True
    assert goal_path["evidence_source"] == (
        "training_ineligible_guide_on_guide_off_evaluation_pairs"
    )
    assert goal_path["training_outcomes_may_control_weight"] is False
    assert goal_path["formal_gate_games_may_control_weight"] is False
    assert goal_path["silent_removal_or_fixed_weight_replacement_allowed"] is False
    assert guide["maximum_loss_weight"] == 0.05
    assert guide["maximum_loss_weight_scope"] == "bootstrap_default"
    assert guide["maximum_post_bootstrap_auxiliary_loss_weight"] == 0.50
    adaptive = guide["adaptive_annealing"]
    assert adaptive["scope"] == "future_specialist_training_runs_only"
    assert adaptive["prospective_scope_revision"] == 44
    assert adaptive["prospective_effective_specialist"] == "archaludon-ex"
    assert adaptive[
        "retroactive_application_to_completed_frozen_or_started_runs"
    ] is False
    assert adaptive["historical_weight_or_receipt_rewrite_allowed"] is False
    assert adaptive["active_teal_revision42_exception_preserved"] is True
    assert adaptive["installation_boundary"] == (
        "after_predecessor_training_service_is_inactive_and_before_"
        "future_runtime_registration"
    )
    assert adaptive["installation_receipt_schema"] == (
        "poke_bot.future_specialist_guide_weight_policy_install/v1"
    )
    assert adaptive["service_control_during_install_allowed"] is False
    assert adaptive["multiplier_applied_before_backpropagation"] is True
    assert adaptive["gradient_effect"] == (
        "scales_guide_conditioned_strategic_head_gradient_contribution"
    )
    assert adaptive["guide_curriculum_revision"] == 51
    assert adaptive["strategic_branch_scope_revision"] == 56
    assert adaptive["direct_policy_cross_entropy_allowed"] is False
    assert adaptive["activation_requires_prestage_validation_receipt"] is True
    assert adaptive["allowed_fusion_roles"] == ["fused_input"]
    assert adaptive["every_head_declares_fusion_role"] is True
    assert adaptive["ascent_semantics"] == (
        "evidence_governed_increase_in_supervised_learning_pressure"
    )
    assert adaptive["summit_semantics"] == (
        "hold_only_while_realized_win_contribution_remains_positive"
    )
    assert adaptive["descent_semantics"] == (
        "evidence_governed_reduction_of_guide_gradient_toward_zero"
    )
    assert adaptive["elapsed_time_only_progression_allowed"] is False
    assert adaptive["guide_head_only_bookkeeping_allowed"] is False
    assert adaptive["statistic"] == "paired_realized_win_rate_delta"
    assert adaptive["estimator"] == (
        "training_ineligible_paired_guide_on_guide_off_evaluation"
    )
    assert adaptive["decay_schedule"] == [0.15, 0.075, 0.0]
    assert adaptive["positive_ramp_schedule"] == [0.15, 0.25, 0.35, 0.50]
    assert adaptive["minimum_paired_games_per_review"] == 1000
    assert adaptive["minimum_paired_games_per_matchup"] == 50
    assert adaptive["realized_win_delta_confidence_level"] == 0.90
    trainer = (ROOT / "scripts" / "train_pure_rl.py").read_text(
        encoding="utf-8"
    )
    review_gate = trainer[
        trainer.index("POKEBOT_FUTURE_GUIDE_WEIGHT_POLICY_REVISION"):
        trainer.index("review_request = emit_review_request")
    ]
    assert '== "44"' in review_gate
    assert '== "46"' in review_gate
    assert '== "45"' not in review_gate
    assert adaptive["evidence_receipt_schema"] == (
        "poke_bot.current_deck_guide_paired_evaluation/v1"
    )
    assert adaptive["schedule_receipt_schema"] == (
        "poke_bot.current_deck_guide_weight_schedule/v1"
    )
    assert adaptive["weight_increase_after_bootstrap_allowed"] is True
    assert adaptive[
        "weight_increase_requires_owner_authorized_clean_boundary_receipt"
    ] is False
    assert adaptive["fleet_owner_authorization_revision"] == 43
    assert adaptive["clean_boundary_schedule_receipt_required"] is True
    assert adaptive["autonomous_weight_increase_allowed"] is True
    assert adaptive["counterfactual_checkpoint_serving_allowed"] is False
    assert adaptive["counterfactual_checkpoint_promotion_allowed"] is False
    projected_goal_path = projection["goal_path_guidance"]
    assert projected_goal_path["owner_vision_required"] is True
    assert projected_goal_path[
        "applies_to_every_future_specialist_training_run"
    ] is True
    assert projected_goal_path[
        "applies_retroactively_to_completed_frozen_or_started_runs"
    ] is False
    assert projected_goal_path["owner_decision_revision"] == 43
    assert projected_goal_path["prospective_scope_revision"] == 44
    assert projected_goal_path["prospective_effective_specialist"] == (
        "archaludon-ex"
    )
    assert projected_goal_path[
        "historical_weight_or_receipt_rewrite_allowed"
    ] is False
    assert projected_goal_path["active_teal_revision42_exception_preserved"] is True
    assert projected_goal_path["bootstrap_behavior"] == "ramp_in"
    assert projected_goal_path["positive_contribution_behavior"] == (
        "ramp_then_hold"
    )
    assert projected_goal_path["internalized_or_nonpositive_behavior"] == (
        "anneal_toward_zero"
    )
    assert projected_goal_path["curve_shape"] == goal_path["curve_shape"]
    assert projected_goal_path["maximum_auxiliary_loss_weight"] == 0.5
    assert projected_goal_path["post_bootstrap_positive_ramp_steps"] == [
        0.15,
        0.25,
        0.35,
        0.5,
    ]
    assert projected_goal_path[
        "all_weight_changes_require_clean_boundary_receipt"
    ] is True
    paired = projected_goal_path["paired_evidence_contract"]
    assert paired["minimum_paired_games_per_review"] == 1000
    assert paired["minimum_paired_games_per_matchup"] == 50
    assert paired["realized_win_delta_confidence_level"] == 0.9
    assert paired["training_replay_and_formal_gate_eligible"] is False
    assert paired["counterfactual_serving_or_promotion_allowed"] is False
    active_teal_ramp = projected_goal_path["active_teal_corrective_ramp"]
    assert active_teal_ramp["owner_decision_revision"] == 42
    assert active_teal_ramp["observed_weight"] == 0.05
    assert active_teal_ramp["target_weight"] == 0.25
    assert active_teal_ramp["activation_boundary_next_iteration"] == 6
    assert active_teal_ramp["status"] == "activated"
    assert active_teal_ramp["iteration_5_commit_sha256"] == (
        "sha256:db28b6681eee5f989a3a8f98e6f7299fbdce2135c05fc1df1b23a8bba61283e4"
    )
    assert projected_goal_path["evidence_source"] == (
        "training_ineligible_guide_on_guide_off_evaluation_pairs"
    )
    assert (
        projected_goal_path["silent_removal_or_fixed_weight_replacement_allowed"]
        is False
    )
    human_protocol = (ROOT / "docs/RL_TRAINING_PROTOCOL.md").read_text(
        encoding="utf-8"
    )
    assert "The weight may never rise above its bootstrap maximum." not in (
        human_protocol
    )
    assert "from 0.05 to 0.25 after the immutable iteration-5 commit" in (
        human_protocol
    )


def test_research_controls_stay_official_four_while_frozen_stay_holdouts() -> None:
    protocol = yaml.safe_load(
        (ROOT / "config/rl_protocol.yaml").read_text(encoding="utf-8")
    )
    registry = json.loads(
        (ROOT / "ops/research_control_registry_v1.json").read_text(
            encoding="utf-8"
        )
    )
    frozen_registry = json.loads(
        (ROOT / "ops/frozen_specialist_registry_v1.json").read_text(
            encoding="utf-8"
        )
    )
    state = yaml.safe_load(
        (ROOT / "state/specialists.yaml").read_text(encoding="utf-8")
    )
    compatibility = json.loads(
        (ROOT / "ops/current_goal_requirements.json").read_text(
            encoding="utf-8"
        )
    )
    research = protocol["research_evaluations"]

    assert research["official"]["roster_policy"] == "fixed_official_four_only"
    assert research["official"]["agents"] == 4
    assert research["official"]["frozen_specialists_allowed"] is False
    assert (
        research["official"]["dynamic_growth_from_frozen_specialist_registry"]
        is False
    )
    assert len(registry["controls"]) == 4
    assert {row["opponent_id"] for row in registry["controls"]} == {
        "iono",
        "dragapult-ex",
        "mega-abomasnow-ex",
        "mega-lucario-ex",
    }
    premium = research["premium_competition"]
    assert premium["frozen_specialists_added"] == (
        "all_registered_completed_specialists"
    )
    assert premium["eligible_frozen_specialists_remain_required"] is True
    frozen_count = sum(
        row.get("frozen") is True
        for row in frozen_registry["specialists"]
    )
    expected_opponents = (
        premium["active_external_agents_after_supersession"] + frozen_count
    )
    expected_premium_games = expected_opponents * premium["games_per_agent"]
    expected_all_games = (
        research["official"]["games_total"] + expected_premium_games
    )
    assert premium["current_frozen_specialists"] == frozen_count
    assert premium["current_agents"] == expected_opponents
    assert premium["current_games_total"] == expected_premium_games
    assert research["current_all_holdouts_games_total"] == expected_all_games
    state_gate = state["gates"]["premium_competition"]
    assert state_gate["frozen_specialist_opponents"] == frozen_count
    assert state_gate["opponents"] == expected_opponents
    assert state_gate["games"] == expected_premium_games
    projection = compatibility["verified_snapshot"]["gate_contract"]
    assert projection["premium_current_frozen_specialists"] == frozen_count
    assert projection["premium_current_games"] == expected_premium_games
    assert projection["all_current_holdout_games"] == expected_all_games


def test_production_selector_uses_owner_pinned_96_worker_scheduler() -> None:
    selector = (ROOT / "config/specialist_runtime.env").read_text(
        encoding="utf-8"
    )
    unit = (
        ROOT / "ops/systemd/pokebot-pure-rl-specialist.service"
    ).read_text(encoding="utf-8")

    assert "PURE_RL_SIM_WORKERS=96\n" in selector
    assert "PURE_RL_GAMES_IN_FLIGHT=96\n" in selector
    assert "POKEBOT_LIVE_POOL_MAX_WORKERS=96\n" in selector
    assert "PURE_RL_REBALANCE_MAX_WORKERS=96\n" in selector
    assert "PURE_RL_REBALANCE_MIN_WORKERS=96\n" in selector
    assert "PURE_RL_MID_ITER_SCHEDULER=1\n" in selector
    assert "POKEBOT_REMOTE_SOCKET_PREFETCH=1\n" in selector
    # Revision 124 deliberately removed the speculative second socket wave.
    assert "POKEBOT_REMOTE_SOCKET_PREFETCH_MAX=1\n" in selector
    assert "POKEBOT_RELEASE_LOCAL_POOL_BEFORE_RESULT_DRAIN=1\n" in selector
    assert "PURE_RL_REBALANCE_RAM_FLOOR_GB=12\n" in selector
    assert "MemoryHigh=100G\n" in unit
    assert "MemoryMax=116G\n" in unit


def test_future_result_drains_release_phase_owned_simulators() -> None:
    protocol = yaml.safe_load(
        (ROOT / "config/rl_protocol.yaml").read_text(encoding="utf-8")
    )
    compatibility = json.loads(
        (ROOT / "ops/current_goal_requirements.json").read_text(
            encoding="utf-8"
        )
    )
    contract = protocol["training_hardware_throughput"][
        "producer_complete_result_drain"
    ]
    projected = compatibility["current_owner_overrides"][
        "producer_complete_result_drain"
    ]

    assert contract["goal_revision"] == 110
    assert contract["required_environment"] == (
        "POKEBOT_RELEASE_LOCAL_POOL_BEFORE_RESULT_DRAIN=1"
    )
    assert set(contract["applies_to"]) == {
        "self_play",
        "public_mix",
        "promotion",
        "research_control",
        "formal_holdout",
    }
    assert contract[
        "release_phase_owned_local_pool_before_buffer_compaction"
    ] is True
    assert contract["scheduled_and_legacy_additive_dispatch_required"] is True
    assert contract["next_iteration_activation_required"] == 5
    assert projected["goal_revision"] == 110
    assert projected[
        "release_phase_owned_local_pool_before_buffer_compaction"
    ] is True
    assert projected["next_iteration_activation_required"] == 5


def test_marnie_iteration9_upload_is_immutable_new_system_boundary() -> None:
    protocol = yaml.safe_load(
        (ROOT / "config/rl_protocol.yaml").read_text(encoding="utf-8")
    )
    state = yaml.safe_load(
        (ROOT / "state/specialists.yaml").read_text(encoding="utf-8")
    )
    compatibility = json.loads(
        (ROOT / "ops/current_goal_requirements.json").read_text(
            encoding="utf-8"
        )
    )

    activation = protocol["marnie_archetype_family_generalization"]["activation"]
    projected = compatibility["current_owner_overrides"][
        "marnie_archetype_family_generalization"
    ]
    mutable = state["post_fleet_refresh"]["active_refresh_new_system_boundary"]
    for contract in (activation, projected, mutable):
        assert contract["successful_upload_is_last_old_system_boundary"] is True
        assert contract["first_post_upload_training_collection_system"] == (
            "new_system"
        )
        assert contract["old_system_collection_after_successful_upload_allowed"] is False
        assert contract["boundary_is_owner_immutable"] is True
        assert contract["activation_inputs_must_be_prepared_before_trigger"] is True
        assert contract["missing_or_inconclusive_evidence_after_successful_upload"] == (
            "pause_before_next_collection"
        )
    assert activation["owner_boundary_revision"] == 130
    assert activation["trigger_iteration"] == 9
    assert projected["activation_trigger"] == (
        "successful_checksum_exact_iteration_9_kaggle_upload_receipt"
    )
    assert mutable["trigger"] == (
        "successful_checksum_exact_iteration_9_kaggle_upload_receipt"
    )
    assert mutable["current_iteration_modified"] is False
    assert mutable["operational_reconciliation_revision"] == 131
    assert mutable["hook_activation_boundary"] == (
        "after_iteration_6_before_iteration_7_collection"
    )
    assert projected["operational_reconciliation_revision"] == 131
    assert projected["scheduler_changed"] is False
    assert projected["fleet_allocation_changed"] is False
    assert projected["self_play_tail_rule_changed"] is False
