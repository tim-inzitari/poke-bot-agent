from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


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

    assert "Revision: `24`" in goal
    assert "mandatory and sole successor after Thwackey" in goal
    assert state["current"]["active_specialist"] == "team-rockets-spidops"
    assert state["current"]["staged_successor_specialist"] is None
    assert state["training_priority"][
        "ordered_unfinished_ids_after_active"
    ][0] == "hammer-pult"
    assert cycle["selection"]["strict_priority_prefix"][-1] == (
        "team-rockets-spidops"
    )
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
    assert projected["activation_status"] == "active_training_verified"
    assert projected["hammer_pult_selected"] is False


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
    assert goal_path["bootstrap_behavior"] == "ramp_in"
    assert goal_path["positive_contribution_behavior"] == "hold"
    assert goal_path["internalized_or_nonpositive_behavior"] == (
        "anneal_toward_zero"
    )
    assert goal_path["evidence_source"] == (
        "training_ineligible_guide_on_guide_off_evaluation_pairs"
    )
    assert goal_path["training_outcomes_may_control_weight"] is False
    assert goal_path["formal_gate_games_may_control_weight"] is False
    assert goal_path["silent_removal_or_fixed_weight_replacement_allowed"] is False
    assert projection["goal_path_guidance"] == {
        "owner_vision_required": True,
        "bootstrap_behavior": "ramp_in",
        "positive_contribution_behavior": "hold",
        "internalized_or_nonpositive_behavior": "anneal_toward_zero",
        "evidence_source": (
            "training_ineligible_guide_on_guide_off_evaluation_pairs"
        ),
        "silent_removal_or_fixed_weight_replacement_allowed": False,
    }


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
