from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "state/alakazam-local-approximate-belief-mcts-bo1000-r216.json"
CONTRACT_SHA256 = "2e260755c33d9fa8a2f821f7eb5e6edb8cd609112d8e01e7c94937aefbe776f3"
R215_PATH = ROOT / "state/alakazam-full-turn-belief-mcts-bo1000-r215.json"
R215_SHA256 = "5423dde739785cdbd75ddee60bfaa2caeb20f70cd841111ff05fbedd920f1681"


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_r216_local_approximate_bo1000_boundary_and_frozen_identity() -> None:
    contract = _contract()
    frozen = contract["frozen_r195_package"]
    timing = contract["timing"]
    approximation = contract["experimental_arm"]["approximation_boundary"]

    assert hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest() == CONTRACT_SHA256
    assert hashlib.sha256(R215_PATH.read_bytes()).hexdigest() == R215_SHA256
    assert contract["owner_decision_revision"] == 216
    assert contract["status"] == "authorized_local_exploratory_bo1000_approximate_search"
    assert contract["relationship_to_existing_work"]["r215_contract_must_be_preserved_byte_for_byte"]
    assert frozen["checkpoint_sha256"] == (
        "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
    )
    assert frozen["bundle_sha256"] == (
        "sha256:dfa8bfccf9ee41d2205c7e30d817489391bb6295fa1ed1eff78c36fd8a8b7145"
    )
    assert all(frozen["disabled_runtime_components"].values())
    assert contract["matchup_adapter"]["required_on_both_arms"]
    assert contract["matchup_adapter"]["runtime_enabled_required"]
    assert approximation["existing_available_state_search_and_cache_api_may_be_used"]
    assert not approximation[
        "perfect_native_complete_semantic_state_equivalence_proof_required_before_local_launch"
    ]
    assert not approximation["perfect_native_actions_commute_certificate_required_before_local_launch"]
    assert not approximation["r215_exact_full_turn_cache_or_transposition_receipt_required_before_local_launch"]
    outer_clock = timing["source_backed_outer_game_clock"]
    assert outer_clock["default_total_game_wall_seconds"] == 600.0
    assert outer_clock["turn_pool_divisor"] == 8.0
    assert outer_clock["dynamic_allocation_formula"] == (
        "min(20.0, max(0.0, (remaining_game_seconds - 30.0) / 8.0))"
    )
    assert outer_clock["healthy_game_receives_full_default_turn_pool"]
    assert outer_clock["turn_pool_shrink_begins_only_when_remaining_game_seconds_below"] == 190.0
    assert outer_clock["r215_stale_600_30_64_fair_share_formula_superseded_for_r216"]
    assert timing["default_planner_wall_seconds_per_actual_turn"] == 20.0
    assert timing["default_model_or_simulator_operation_wall_seconds"] == 5.0
    assert timing["effective_actual_turn_pool_formula"] == (
        "min(20.0, max(0.0, (remaining_game_seconds - 30.0) / 8.0))"
    )
    assert timing["atomic_steps_do_not_receive_a_fresh_twenty_second_or_five_second_search_pool"]


def test_r216_projections_and_no_kaggle_authority_match_typed_contract() -> None:
    contract = _contract()
    protocol_doc = yaml.safe_load((ROOT / "config/rl_protocol.yaml").read_text())
    protocol = protocol_doc["alakazam_local_approximate_belief_mcts_bo1000_r216"]
    specialists_doc = yaml.safe_load((ROOT / "state/specialists.yaml").read_text())
    specialists = specialists_doc["alakazam_local_approximate_belief_mcts_bo1000_r216"]
    compatibility = json.loads(
        (ROOT / "ops/current_goal_requirements.json").read_text()
    )["current_owner_overrides"]["alakazam_local_approximate_belief_mcts_bo1000_r216"]

    for projection in (protocol, specialists, compatibility):
        revision = projection.get("owner_decision_revision", projection.get("goal_revision"))
        assert revision == 216
    assert protocol["values_owned_by_sha256"] == "sha256:" + CONTRACT_SHA256
    assert specialists["owner_contract_sha256"] == "sha256:" + CONTRACT_SHA256
    assert compatibility["typed_source_sha256"] == "sha256:" + CONTRACT_SHA256
    assert compatibility["evaluation"]["total_games"] == 1000
    assert compatibility["evaluation"]["matched_rng_pairs"] == 500
    assert compatibility["evaluation"]["belief_mcts_as_seat_0"] == 500
    assert compatibility["evaluation"]["belief_mcts_as_seat_1"] == 500
    assert compatibility["evaluation"]["belief_mcts_actual_first"] == 500
    assert compatibility["evaluation"]["belief_mcts_actual_second"] == 500
    assert compatibility["timing"]["dynamic_allocation_formula"] == (
        "min(20.0, max(0.0, (remaining_game_seconds - 30.0) / 8.0))"
    )
    assert compatibility["timing"]["turn_pool_shrink_begins_only_when_remaining_game_seconds_below"] == 190.0
    assert not compatibility["approximate_search"][
        "perfect_native_complete_semantic_state_equivalence_proof_required_before_local_launch"
    ]
    for key in (
        "additional_training_authorized",
        "evaluation_games_training_eligible",
        "serving_eligible",
        "production_action_authority_enabled",
        "selector_change_authorized",
        "checkpoint_publication_authorized",
        "promotion_authorized",
        "kaggle_api_calls_authorized",
        "kaggle_upload_authorized",
        "kaggle_queue_authorized",
        "kaggle_submission_authorized",
        "r175_restart_authorized",
        "iteration_21_collection_authorized",
    ):
        assert compatibility[key] is False
    assert contract["authority"]["kaggle_api_calls_authorized"] is False
