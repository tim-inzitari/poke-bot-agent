from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "state/alakazam-full-turn-belief-mcts-bo1000-r215.json"
CONTRACT_SHA256 = "5423dde739785cdbd75ddee60bfaa2caeb20f70cd841111ff05fbedd920f1681"


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_r215_full_turn_clock_cache_and_frozen_identity_contract() -> None:
    contract = _contract()
    frozen = contract["frozen_r195_package"]
    timing = contract["timing"]
    cache = contract["experimental_arm"]["deterministic_state_evaluation_cache"]

    assert hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest() == CONTRACT_SHA256
    assert contract["owner_decision_revision"] == 215
    assert contract["relationship_to_existing_work"][
        "supersedes_r214_execution_semantics_before_r214_launch"
    ] is True
    assert frozen["checkpoint_sha256"] == (
        "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
    )
    assert frozen["bundle_sha256"] == (
        "sha256:dfa8bfccf9ee41d2205c7e30d817489391bb6295fa1ed1eff78c36fd8a8b7145"
    )
    assert timing["source_backed_outer_game_clock"]["default_total_game_wall_seconds"] == 600.0
    assert timing["default_planner_wall_seconds_per_actual_turn"] == 20.0
    assert timing["default_model_or_simulator_operation_wall_seconds"] == 5.0
    assert timing["atomic_steps_do_not_receive_a_fresh_twenty_second_or_five_second_search_pool"] is True
    assert timing["minimum_valid_simulations_before_search_authority"] == 1
    assert timing["emergency_simulation_safety_ceiling"] == 1_000_000
    assert timing["requested_fixed_simulation_target_or_target_completion_gate_allowed"] is False
    assert cache["public_observation_equality_alone_may_merge_transpositions"] is False
    assert cache["native_simulator_exact_complete_semantic_state_equality_required_for_transposition_merge"] is True
    assert cache["current_packaged_engine_exposes_trusted_complete_transposition_identity"] is False
    assert cache["current_packaged_engine_exposes_trusted_actions_commute_certificate"] is False
    assert cache["absent_or_invalid_native_identity_or_commutation_certificate_behavior"] == (
        "expand_orders_separately_fail_closed"
    )


def test_r215_projections_match_typed_timing_and_transposition_boundary() -> None:
    contract = _contract()
    protocol = yaml.safe_load((ROOT / "config/rl_protocol.yaml").read_text())[ 
        "alakazam_full_turn_belief_mcts_bo1000_r215"
    ]
    specialists = yaml.safe_load((ROOT / "state/specialists.yaml").read_text())[ 
        "alakazam_full_turn_belief_mcts_bo1000_r215"
    ]
    compatibility = json.loads(
        (ROOT / "ops/current_goal_requirements.json").read_text()
    )["current_owner_overrides"]["alakazam_full_turn_belief_mcts_bo1000_r215"]

    for projection in (protocol, specialists, compatibility):
        revision = projection.get("owner_decision_revision", projection.get("goal_revision"))
        assert revision == 215
    assert protocol["values_owned_by_sha256"] == "sha256:" + CONTRACT_SHA256
    assert specialists["owner_contract_sha256"] == "sha256:" + CONTRACT_SHA256
    assert compatibility["typed_source_sha256"] == "sha256:" + CONTRACT_SHA256
    assert protocol["timing"]["default_total_game_wall_seconds"] == 600.0
    assert protocol["timing"]["default_planner_wall_seconds_per_actual_turn"] == 20.0
    assert protocol["timing"]["fixed_simulation_target_or_completion_gate_allowed"] is False
    assert protocol["experimental_arm"]["current_native_complete_transposition_identity_available"] is False
    assert specialists["full_turn_search"]["missing_native_identity_or_certificate_behavior"] == (
        "expand_orders_separately_fail_closed"
    )
    assert compatibility["timing"]["effective_operation_allowance_formula"] == (
        "min(per_operation_ceiling, remaining_actual_turn_pool, remaining_game_clock)"
    )
    assert contract["evaluation_design"]["total_games"] == 1000
    assert compatibility["evaluation"]["belief_mcts_actual_second"] == 500

