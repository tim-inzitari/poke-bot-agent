from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / "ops/final_submission/final_format_alakazam_static_preparation_v1.json"
)
EXPECTED_PARENT = (
    "sha256:270b5156781b0a95f703abe3e8fe13866"
    "d2fbb4c85a8f32534f99af74aece2ea"
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_static_contract_is_noncomputing_and_checksum_bound() -> None:
    contract = _contract()
    authority = contract["authority"]
    assert contract["schema"] == (
        "poke_bot.final_format_alakazam_static_preparation/v1"
    )
    assert contract["status"] == (
        "static_preparation_complete_slowking_failed_experiment_boundary_reached_waiting_for_g0_g1"
    )
    assert contract["owner_goal_revision"] == 79
    assert contract["canonical_plan_sha256"] == _sha256(
        ROOT / contract["canonical_plan"]
    )
    assert authority["static_implementation"] is True
    for key in (
        "model_computation",
        "checkpoint_loading",
        "model_instantiation",
        "replay_scanning",
        "corpus_materialization",
        "benchmarking",
        "training",
        "evaluation",
        "package_construction",
        "submission_construction",
    ):
        assert authority[key] is False
    for key in ("runtime", "selector", "registry", "kaggle_queue"):
        assert authority[key] == "none"
    assert contract["active_runtime_modified"] is False
    assert contract["active_selector_modified"] is False
    assert contract["active_registry_modified"] is False


def test_every_static_template_matches_its_bound_checksum() -> None:
    contract = _contract()
    assert len(contract["templates"]) == 8
    for row in contract["templates"].values():
        path = ROOT / row["path"]
        assert path.is_file()
        assert row["sha256"] == _sha256(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["template_only"] is True
        assert payload["status"] == "unissued"
        assert payload.get("runtime_authority") == "none"


def test_alakazam_bridge_is_even_seat_first_preferring_and_same_lineage() -> None:
    contract = _contract()
    bridge = contract["bridge"]
    assert bridge["specialist_id"] == "alakazam"
    assert bridge["model_format"] == "final_submission_format"
    assert bridge["start_after_terminal_disposition"] == (
        "slowking_failed_experiment"
    )
    assert bridge["required_fleet_count"] == 15
    assert bridge["required_frozen_specialist_count"] == 14
    assert bridge["terminal_failed_experiment_exception_count"] == 1
    assert bridge["preferred_parent_checkpoint_sha256"] == EXPECTED_PARENT
    assert bridge["training_seat_split"] == {"first": 0.5, "second": 0.5}
    assert bridge["package_preference"] == "first_if_allowed"
    assert bridge["second_focus_1_to_7_allowed"] is False
    assert bridge["partial_old_alakazam_core_overlay_allowed"] is False
    assert bridge["guide_runtime_route_count"] == 0
    assert contract["model_computation_release_receipts"] == [
        "required_specialist_fleet_complete_for_final_alakazam_v1",
        "capacity_research_resource_lease_v1",
    ]


def test_final_format_model_spec_binds_exact_h10_shape_without_running() -> None:
    contract = _contract()
    row = contract["specifications"]["final_format_alakazam_model_spec_v1"]
    path = ROOT / row["path"]
    assert row["sha256"] == _sha256(path)
    spec = json.loads(path.read_text(encoding="utf-8"))
    assert spec["runnable"] is False
    assert spec["model_computation_authority"] is False
    assert spec["capacity_profile"] == "H10-I/v1"
    assert spec["search_schema"] == "none"
    architecture = spec["architecture"]
    assert architecture == {
        "d_model": 96,
        "attention_heads": 8,
        "spatial_layers": 7,
        "temporal_layers": 3,
        "option_layers": 7,
        "feed_forward_width": 2496,
        "history_context": 320,
        "fusion_width": 16,
        "minimum_learned_head_count": 19,
        "exact_learned_head_count": None,
        "minimum_distinct_bounded_action_routes": 19,
        "exact_distinct_bounded_action_routes": None,
        "head_route_count_equality_required": True,
        "exact_counts_late_bound_by": (
            "final_format_learned_role_route_inventory_v1"
        ),
        "setup_board_outcome_required": True,
        "combo_state_head_required": True,
        "slowking_passing_checkpoint_required": False,
        "slowking_failed_experiment_receipt_required": True,
        "one_distinct_option_conditioned_route_per_learned_head": True,
        "every_learned_head_influences_action_scores": True,
        "guide_runtime_route_count": 0,
    }
    assert spec["training"]["supervised_bootstrap_epochs"] == 25
    assert spec["training"]["baseline_games_per_iteration"] == 8192
    assert spec["training"]["seat_assignment"] == {
        "first_fraction": 0.5,
        "second_fraction": 0.5,
        "exact_even_split_required_for": ["assigned", "actual", "consumed"],
    }
    assert spec["training"]["second_focus_1_to_7_allowed"] is False
    assert spec["hardware_entry_invariant"][
        "blackwell_local_simulator_workers"
    ] == 96
    assert spec["hardware_entry_invariant"][
        "blackwell_local_games_in_flight"
    ] == 96
    assert spec["package"]["turn_order_preference"] == "first_if_allowed"


def test_source_compatibility_is_checksum_bound_and_static_only() -> None:
    contract = _contract()
    row = contract["specifications"][
        "final_format_alakazam_source_compatibility_v1"
    ]
    path = ROOT / row["path"]
    assert row["sha256"] == _sha256(path)
    audit = json.loads(path.read_text(encoding="utf-8"))
    assert audit["runnable"] is False
    assert audit["model_instantiated"] is False
    assert audit["checkpoint_loaded"] is False
    assert audit["parameter_count_computed"] is False
    for source, digest in audit["source_files"].items():
        assert digest == _sha256(ROOT / source)
    checks = audit["static_checks"]
    assert checks["base_learned_head_count"] == 17
    assert checks["setup_board_outcome_route_supported"] is True
    assert checks["combo_state_route_supported"] is True
    assert checks["minimum_final_route_count_supported"] == 19
    assert checks[
        "route_uses_typed_output_and_cross_attended_option_hidden"
    ] is True
    assert checks["one_module_per_head_route"] is True
    assert checks["route_final_projection_zero_safe"] is True
    assert checks["guide_excluded_from_runtime_fusion"] is True
    assert "direct_migration_compatibility" in audit["not_proven_before_release"]


def test_non_runnable_package_template_cannot_enter_production() -> None:
    contract = _contract()
    row = contract["templates"][
        "final_format_alakazam_offline_package_v1"
    ]
    package = json.loads((ROOT / row["path"]).read_text(encoding="utf-8"))
    assert package["runnable"] is False
    assert package["submittable"] is False
    assert package["kaggle_eligible"] is False
    assert package["selector_eligible"] is False
    assert package["queue_authorization"] is None
    assert package["turn_order_preference"] == "first_if_allowed"
    assert package["guide_runtime_route_count"] == 0


def test_parent_lock_binds_complete_identity_and_explicit_failure_path() -> None:
    contract = _contract()
    row = contract["templates"]["final_format_alakazam_parent_lock_v1"]
    lock = json.loads((ROOT / row["path"]).read_text(encoding="utf-8"))
    parent = lock["preferred_parent"]
    assert parent["checkpoint_sha256"] == EXPECTED_PARENT
    for key in (
        "checkpoint_status",
        "run_id",
        "iteration",
        "lineage_sha256",
        "accepted_cumulative_core_ancestry_sha256",
        "model_config_sha256",
        "feature_schema_sha256",
        "option_schema_sha256",
        "temporal_schema_sha256",
        "strategic_head_schema_sha256",
        "target_schema_sha256",
        "guide_schema_sha256",
        "decision_fusion_schema_sha256",
        "matchup_adapter_format_sha256",
        "matchup_adapter_roster_sha256",
        "guide_sha256",
        "corpus_sha256",
        "learned_head_role_map_sha256",
        "learned_route_inventory_sha256",
        "learned_parameter_count",
        "package_size_bytes",
        "trainer_code_revision",
        "parent_selection_evidence_sha256",
    ):
        assert key in parent
    assert lock["direct_migration_failure"]["partial_tensor_overlay_used"] is False
    assert lock["same_archetype_fallback"]["partial_old_alakazam_core_overlay_used"] is False
    assert lock["historical_alakazam_checkpoint_rewritten"] is False


def test_role_route_inventory_requires_action_influence_and_zero_guide_routes() -> None:
    contract = _contract()
    row = contract["templates"]["final_format_learned_role_route_inventory_v1"]
    inventory = json.loads((ROOT / row["path"]).read_text(encoding="utf-8"))
    required = set(inventory["required_role_fields"])
    assert {
        "head_id",
        "route_id",
        "option_conditioned",
        "causal_board_state_cross_attention",
        "finite_bound",
        "zero_safe",
        "action_influence",
    } <= required
    assert inventory["guide"] == {
        "classification": "training_only_curriculum_metadata",
        "learned_decision_head": False,
        "runtime_input": False,
        "runtime_route_count": 0,
        "action_logit_influence": False,
    }


def test_direct_migration_failure_template_preserves_parent_and_forbids_overlay() -> None:
    contract = _contract()
    row = contract["templates"][
        "final_format_alakazam_direct_migration_failure_v1"
    ]
    failure = json.loads((ROOT / row["path"]).read_text(encoding="utf-8"))
    assert failure["preferred_parent_checkpoint_sha256"] == EXPECTED_PARENT
    assert failure["preferred_parent_mutated"] is False
    assert failure["partial_tensor_overlay_used"] is False
    assert failure["fallback_authorized"] is False


def test_canonical_projections_bind_the_exact_static_contract() -> None:
    expected = _sha256(CONTRACT_PATH)
    protocol = yaml.safe_load(
        (ROOT / "config/rl_protocol.yaml").read_text(encoding="utf-8")
    )
    state = yaml.safe_load(
        (ROOT / "state/specialists.yaml").read_text(encoding="utf-8")
    )
    goals = json.loads(
        (ROOT / "ops/current_goal_requirements.json").read_text(
            encoding="utf-8"
        )
    )
    protocol_row = protocol["specialist_training"]["post_fleet_refresh"][
        "static_preparation"
    ]
    state_row = state["post_fleet_refresh"]["static_preparation"]
    goal_row = goals["current_owner_overrides"][
        "post_fleet_alakazam_grimms_refresh"
    ]["static_preparation"]
    assert protocol_row["contract_sha256"] == expected
    assert state_row["contract_checksum"] == expected
    assert goal_row["contract_sha256"] == expected
    for row in (protocol_row, state_row, goal_row):
        assert row["receipt_template_count"] == 8
        assert row["model_computation_started"] is True
        assert row["active_runtime_modified"] is False
