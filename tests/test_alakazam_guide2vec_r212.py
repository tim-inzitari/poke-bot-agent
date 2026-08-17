"""Contract coverage for the isolated r212 Alakazam Guide2Vec experiment."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "state/alakazam-guide2vec-no-mcts-bo1000-r212.json"
R195_PATH = ROOT / "state/alakazam-terminal-expert-bootstrap-no-rtp-submit-r195.json"
PROJECTION_KEY = "alakazam_guide2vec_no_mcts_bo1000_r212"
STATUS = "authorized_isolated_blackwell_distillation_and_no_mcts_bo1000_pending_prerequisites"
RETIRED_STATUS = "abandoned_unlaunched_preserved_by_r226_general_pipeline"
R226_PIPELINE_PATH = "state/guide2vec-general-training-pipeline-r226.json"
R195_CONTRACT_SHA256 = "e37cf1d3e638c3aed56230c9fa970c61e6c1ed8b4bd3024de259cb9847c31e48"
R195_CHECKPOINT_SHA256 = (
    "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
)
R195_BUNDLE_SHA256 = (
    "sha256:dfa8bfccf9ee41d2205c7e30d817489391bb6295fa1ed1eff78c36fd8a8b7145"
)
R195_MATCHUP_TREE_SHA256 = (
    "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049"
)
R212_CONTRACT_SHA256 = (
    "sha256:aa9c7b8158c91d183c092b92bab3047c7bd7af705d539c68cdd3e9c206c0c2b9"
)


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _projections() -> tuple[dict, dict, dict]:
    compatibility = json.loads(
        (ROOT / "ops/current_goal_requirements.json").read_text(encoding="utf-8")
    )["current_owner_overrides"][PROJECTION_KEY]
    protocol = yaml.safe_load(
        (ROOT / "config/rl_protocol.yaml").read_text(encoding="utf-8")
    )[PROJECTION_KEY]
    specialists = yaml.safe_load(
        (ROOT / "state/specialists.yaml").read_text(encoding="utf-8")
    )[PROJECTION_KEY]
    return compatibility, protocol, specialists


def test_r212_locks_the_exact_no_rtp_base_and_tiny_frozen_head() -> None:
    contract = _contract()
    base = contract["frozen_base"]
    head = contract["guide2vec_head"]

    assert contract["schema"] == "poke_bot.alakazam_guide2vec_no_mcts_bo1000_r212/v1"
    assert contract["owner_decision_revision"] == 212
    assert contract["latest_owner_clarification_revision"] == 217
    assert contract["status"] == STATUS
    assert hashlib.sha256(R195_PATH.read_bytes()).hexdigest() == R195_CONTRACT_SHA256
    assert base == {
        "base_checkpoint_or_parameter_mutation_allowed": False,
        "base_hidden_state_must_remain_frozen": True,
        "bundle_sha256": R195_BUNDLE_SHA256,
        "checkpoint_bytes": 127_914_385,
        "checkpoint_sha256": R195_CHECKPOINT_SHA256,
        "deck_cards_sha256": "sha256:660c1274aac19d88c40fd2bb52187f53dc639d944506760e386f2686b91cc247",
        "deck_id": "alakazam-owner-rtp-pilot-r175",
        "deck_path": "decks/archetype-samples/alakazam-owner-rtp-pilot-r175.csv",
        "matchup_adapter_bank_and_public_tree_must_be_identical_enabled_frozen_and_route_parity_attested": True,
        "matchup_adapter_runtime_enabled_for_training_latent_extraction_and_both_bo1000_arms": True,
        "matchup_tree_sha256": R195_MATCHUP_TREE_SHA256,
        "r195_contract_path": "state/alakazam-terminal-expert-bootstrap-no-rtp-submit-r195.json",
        "r195_contract_sha256": f"sha256:{R195_CONTRACT_SHA256}",
        "rtp_enabled": False,
        "same_checkpoint_deck_model_config_matchup_tree_and_non_guide_runtime_required_for_both_arms": True,
        "submission_id": 55_378_392,
        "submission_message": "alakazam training milestone iter 21 copy 1/2 first 261d367e131e NO RTP",
    }
    assert head["strategy_id"] == "frozen_hidden_state_option_conditioned_guide2vec_ranker"
    assert head["parameter_count_min"] == 100_000
    assert head["parameter_count_max"] == 500_000
    assert head["base_model_trainable"] is False
    assert head["only_guide2vec_head_parameters_receive_gradients"] is True
    assert head["frozen_r195_hidden_state_input_required"] is True
    assert head["current_legal_option_conditioning_required"] is True
    assert head["language_model_or_text_corpus_allowed"] is False


def test_r212_data_targets_are_causal_compact_and_evaluation_is_ineligible() -> None:
    data = _contract()["guide_teacher_and_data"]

    assert data["guide_id"] == "alakazam"
    assert data["acting_archetype_required"] == "alakazam"
    assert data["acting_seat_only"] is True
    assert data["exact_teacher_compatible_alakazam_deck_required"] is True
    assert data["training_target"] == (
        "confidence_weighted_listwise_cross_entropy_over_current_legal_stage_options"
    )
    assert data["compact_target_representation"] == {
        "full_teacher_score_vector_available_or_required": False,
        "guide_confidence": True,
        "guide_target_index": True,
        "nonpositive_confidence_or_out_of_range_target_behavior": "mask_entire_stage",
        "target_index_and_legal_option_width_must_match": True,
    }
    assert data["causal_inputs"] == {
        "acting_seat_policy_visible_observation_only": True,
        "current_factorized_legal_option_representation_only": True,
        "opponent_hidden_cards_deck_order_prizes_or_future_state_allowed": False,
        "select_context_and_effect_representation_required_when_present": True,
    }
    assert data["human_action_base_policy_logit_outcome_or_future_action_target_allowed"] is False
    assert set(data["excluded_from_training"]) == {
        "kaggle_submission_replays_and_scores",
        "submission_55378392_replays",
        "r197_r198_r199_legacy_rtp_rows",
        "r205_r207_mcts_rows_or_receipts",
        "all_guide2vec_bo1000_games_and_receipts",
        "slop_box_or_non_alakazam_rows",
    }
    assert data["split"] == {
        "deck_list_disjoint": False,
        "deck_list_overlap_expected_or_allowed": True,
        "heldout_used_for_model_selection_or_threshold_tuning": False,
        "retained_row_episode_and_source_day_fingerprints_required_per_partition": True,
        "train_validation_heldout_overlap_allowed": False,
        "whole_episode_and_source_day_split_fingerprints_must_be_reported": True,
        "whole_episode_disjoint": True,
        "whole_source_day_disjoint": True,
    }


def test_r212_runtime_is_bounded_and_cannot_call_search_or_rtp() -> None:
    contract = _contract()
    runtime = contract["candidate_runtime"]

    assert runtime == {
        "direct_action_override_without_bounded_logit_recomputation_allowed": False,
        "finite_legal_scores_and_unique_confident_top_choice_required": True,
        "invalid_flat_tied_or_low_confidence_behavior": "exact_frozen_r195_direct_policy_fallback",
        "maximum_logit_bonus": 0.05,
        "historical_guide_linear_or_guide_logit_layer_allowed": False,
        "mcts_expectimax_rollout_recursive_turn_planner_rtp_or_simulator_leaf_reranking_allowed": False,
        "mode": "bounded_guide_logit_bonus",
        "per_factorized_stage_score_normalization": "min_max_0_to_1",
    }
    relationship = contract["relationship_to_existing_work"]
    assert relationship["separate_from_r205_r207_mcts"] is True
    assert relationship["legacy_rtp_sidecar_executor_or_partial_rows_allowed"] is False
    assert relationship["slop_box_data_model_runtime_or_submission_authority"] is False


def test_sidecar_and_trainer_have_no_direct_search_or_rtp_import_boundary() -> None:
    """Keep the offline candidate path structurally separate from r205/r207."""

    forbidden = (
        "mcts",
        "expectimax",
        "recursive_turn_planner",
        "simulator",
        "rtp",
    )
    for relative in (
        Path("poke_bot/guide2vec.py"),
        Path("scripts/train_alakazam_guide2vec_r212.py"),
    ):
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert not [
            module
            for module in imports
            if any(token in module.casefold() for token in forbidden)
        ], relative


def test_r212_bo1000_is_a_separate_even_first_second_direct_policy_mirror() -> None:
    evaluation = _contract()["bo1000_evaluation"]

    assert evaluation["evaluation_id"] == "alakazam-r212-guide2vec-no-mcts-bo1000"
    assert evaluation["total_games"] == 1_000
    assert evaluation["matched_rng_pairs"] == 500
    assert evaluation["games_per_pair"] == 2
    assert evaluation["complete_all_games_without_early_best_of_stop"] is True
    assert evaluation["arms"] == [
        "frozen_r195_direct_policy_plus_frozen_guide2vec_bounded_logit_bonus",
        "frozen_r195_no_rtp_direct_policy",
    ]
    assert evaluation["seat_balance"] == {
        "guide2vec_as_seat_0": 500,
        "guide2vec_as_seat_1": 500,
        "no_rtp_as_seat_0": 500,
        "no_rtp_as_seat_1": 500,
    }
    assert evaluation["actual_turn_order_balance"] == {
        "first_actor_arm_seat_and_turn_order_digest_required_in_every_receipt": True,
        "guide2vec_actual_first": 500,
        "guide2vec_actual_second": 500,
        "initial_actor_is_explicit_sealed_pair_material_not_inferred_from_seat": True,
        "missing_duplicate_crossed_or_unbalanced_pair_fails_closed": True,
        "no_rtp_actual_first": 500,
        "no_rtp_actual_second": 500,
        "one_guide2vec_first_and_one_guide2vec_second_game_per_pair": True,
    }
    assert evaluation["pairing"] == {
        "candidate_and_control_use_same_frozen_base_runtime": True,
        "identical_initial_rng_and_deck_order_material_within_pair": True,
        "unique_pair_ids_and_pair_game_nonces_required": True,
    }
    assert evaluation["mcts_or_r207_schedule_runner_or_receipt_reuse_allowed"] is False
    assert evaluation["evaluation_games_training_eligible"] is False
    assert evaluation["runtime_graph_difference"] == {
        "candidate_guide2vec_frozen": True,
        "candidate_guide2vec_instances": 1,
        "control_disabled_or_zeroed_guide2vec_component_allowed": False,
        "control_guide2vec_forward_hooks": 0,
        "control_guide2vec_linear_transforms": 0,
        "control_guide2vec_module_instances": 0,
        "control_guide2vec_parameters": 0,
        "control_guide2vec_presence": "absent_from_runtime_graph",
        "control_guide2vec_state_keys": 0,
        "only_candidate_control_runtime_difference": "one_frozen_guide2vec_bounded_logit_bonus_component",
        "per_game_runtime_graph_absence_and_difference_receipts_required": True,
    }


def test_r212_has_a_dedicated_noninterfering_blackwell_boundary_and_no_serving_authority() -> None:
    contract = _contract()
    isolation = contract["managed_blackwell_isolation"]
    prerequisites = contract["launch_prerequisites"]
    authority = contract["authority"]

    assert isolation["managed_service"] == "pokebot-alakazam-guide2vec-r212.service"
    assert isolation["required_device_name"] == "NVIDIA RTX PRO 5000 Blackwell"
    assert isolation["managed_systemd_boundary_required"] is True
    assert isolation["safe_noninterference_preflight_required_before_start"] is True
    assert isolation["existing_protected_workload_stop_restart_reconfigure_or_worker_reduction_allowed"] is False
    assert isolation["interactive_sessions_may_be_signalled_terminated_or_replaced"] is False
    assert isolation["r197_r198_r205_r207_or_other_running_test_output_reuse_allowed"] is False
    assert isolation["blackwell_96_worker_invariant_may_be_changed"] is False
    assert prerequisites["launch_before_every_prerequisite_is_immutable_and_valid"] is False
    assert all(
        prerequisites[key] is True
        for key in (
            "blackwell_safe_noninterference_and_dedicated_output_receipt",
            "bo1000_actual_first_second_schedule_and_receipt_integrity_tests",
            "deterministic_head_and_exact_direct_fallback_parity_receipt",
            "guide2vec_materialization_causality_and_target_alignment_receipt",
            "heldout_teacher_agreement_gate",
            "no_mcts_no_rtp_dependency_and_runtime_receipt",
            "parameter_count_frozen_base_and_teacher_identity_receipt",
            "retained_row_episode_and_source_day_partition_fingerprint_receipt",
            "runtime_on_frozen_matchup_adapter_tree_parity_receipt_for_training_and_both_arms",
            "control_runtime_graph_complete_guide2vec_absence_receipt",
            "whole_game_day_split_and_exclusion_audit",
        )
    )
    assert authority["guide2vec_head_gradient_updates_authorized"] is True
    assert authority["dedicated_guide2vec_training_service_start_authorized_after_preflight"] is True
    assert authority["exact_no_mcts_bo1000_authorized_after_prerequisites"] is True
    assert all(
        authority[key] is False
        for key in (
            "base_checkpoint_publication_authorized",
            "base_model_training_or_parameter_updates_authorized",
            "iteration_21_collection_authorized",
            "kaggle_submission_authorized",
            "production_action_authority_enabled",
            "promotion_authorized",
            "r175_restart_authorized",
            "selector_change_authorized",
            "serving_eligible",
            "training_service_start_authorized_for_any_other_lineage",
        )
    )


def test_r212_source_is_preserved_while_live_projections_retire_its_launches() -> None:
    contract = _contract()
    compatibility, protocol, specialists = _projections()

    assert "sha256:" + hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest() == (
        R212_CONTRACT_SHA256
    )
    assert compatibility["typed_source"] == str(CONTRACT_PATH.relative_to(ROOT))
    assert compatibility["typed_source_sha256"] == R212_CONTRACT_SHA256
    assert protocol["values_owned_by"] == str(CONTRACT_PATH.relative_to(ROOT))
    assert protocol["values_owned_by_sha256"] == R212_CONTRACT_SHA256
    assert specialists["owner_contract"] == str(CONTRACT_PATH.relative_to(ROOT))
    assert specialists["owner_contract_sha256"] == R212_CONTRACT_SHA256

    for projection in (compatibility, specialists):
        assert projection.get("goal_revision", projection.get("owner_decision_revision")) == 212
        assert projection["status"] == RETIRED_STATUS
        assert projection["latest_owner_clarification_revision"] == 226
        assert projection["superseded_by_typed_source"] == R226_PIPELINE_PATH
        assert projection["training_or_bo1000_launch_authority_retired"] is True
        assert projection["strategy_id"] == contract["guide2vec_head"]["strategy_id"]
        assert projection["submission_id"] == contract["frozen_base"]["submission_id"]
        assert projection["checkpoint_sha256"] == R195_CHECKPOINT_SHA256
        assert projection["bundle_sha256"] == R195_BUNDLE_SHA256
        assert projection["rtp_enabled"] is False
        assert projection[
            "matchup_adapter_runtime_enabled_for_training_latent_extraction_and_both_bo1000_arms"
        ] is True
        assert projection["parameter_count_min"] == 100_000
        assert projection["parameter_count_max"] == 500_000
        assert projection["maximum_logit_bonus"] == 0.05
        assert projection["base_model_trainable"] is False
        assert projection["only_guide2vec_head_parameters_receive_gradients"] is True
        assert projection[
            "mcts_expectimax_rollout_recursive_turn_planner_rtp_or_simulator_leaf_reranking_allowed"
        ] is False
        assert projection["r207_schedule_runner_or_receipt_reuse_allowed"] is False
        assert projection["total_games"] == 1_000
        assert projection["matched_rng_pairs"] == 500
        assert projection["games_per_pair"] == 2
        assert projection["guide2vec_as_seat_0"] == 500
        assert projection["guide2vec_as_seat_1"] == 500
        assert projection["guide2vec_actual_first"] == 500
        assert projection["guide2vec_actual_second"] == 500
        assert projection["no_rtp_actual_first"] == 500
        assert projection["no_rtp_actual_second"] == 500
        assert projection["evaluation_games_training_eligible"] is False
        assert projection["candidate_guide2vec_instances"] == 1
        assert projection["control_guide2vec_presence"] == "absent_from_runtime_graph"
        assert projection["control_guide2vec_linear_transforms"] == 0
        assert projection["control_disabled_or_zeroed_guide2vec_component_allowed"] is False
        assert all(
            projection[key] is False
            for key in (
                "guide2vec_head_gradient_updates_authorized",
                "dedicated_guide2vec_training_service_start_authorized_after_preflight",
                "immutable_guide2vec_artifact_publication_authorized",
                "exact_no_mcts_bo1000_authorized_after_prerequisites",
            )
        )

    assert protocol["owner_decision_revision"] == 212
    assert protocol["status"] == STATUS
    assert protocol["frozen_base"]["submission_id"] == 55_378_392
    assert protocol["frozen_base"]["rtp_enabled"] is False
    assert protocol["guide2vec"]["parameter_count_min"] == 100_000
    assert protocol["guide2vec"]["parameter_count_max"] == 500_000
    assert protocol["guide2vec"]["maximum_logit_bonus"] == 0.05
    assert protocol["guide2vec"]["base_model_trainable"] is False
    assert protocol["data_isolation"]["deck_list_overlap_expected_or_allowed"] is True
    assert protocol["data_isolation"]["kaggle_r197_r198_r199_r205_r207_and_bo1000_rows_training_eligible"] is False
    assert protocol["evaluation"]["guide2vec_actual_first"] == 500
    assert protocol["evaluation"]["guide2vec_actual_second"] == 500
    assert protocol["evaluation"]["no_rtp_actual_first"] == 500
    assert protocol["evaluation"]["no_rtp_actual_second"] == 500
    assert protocol["evaluation"][
        "mcts_expectimax_rollout_recursive_turn_planner_rtp_or_simulator_leaf_reranking_allowed"
    ] is False
    assert protocol["evaluation"]["r207_schedule_runner_or_receipt_reuse_allowed"] is False
    assert protocol["evaluation"]["evaluation_games_training_eligible"] is False


def test_r226_retirement_blocks_every_r212_gradient_entry_before_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The preserved r212 implementation cannot regain training authority."""

    from scripts import stage_alakazam_guide2vec_r212_source_snapshot as stager
    from scripts import train_alakazam_guide2vec_r212 as trainer

    applied_job_specs: list[object] = []
    monkeypatch.setattr(
        trainer,
        "_apply_job_spec",
        lambda args: applied_job_specs.append(args),
    )
    monkeypatch.setattr(
        trainer,
        "validate_inputs",
        lambda _args: pytest.fail("r226 retirement must precede input validation"),
    )
    with pytest.raises(RuntimeError, match="r212 Guide2Vec training is retired by r226"):
        trainer.main(["--run"])
    assert applied_job_specs == []

    with pytest.raises(RuntimeError, match="r212 Guide2Vec training is retired by r226"):
        trainer._run_training(
            SimpleNamespace(device="cuda:0", seed=0),
            object(),
        )

    with pytest.raises(RuntimeError, match="r212 Guide2Vec training is retired by r226"):
        trainer.run_epoch(
            object(),
            (),
            device=trainer.torch.device("cpu"),
            batch_rows=1,
            coverage_weight=0.0,
            optimizer=object(),
            seed=0,
        )

    service = (
        ROOT / "deploy/systemd/pokebot-alakazam-guide2vec-r212.service"
    ).read_text(encoding="utf-8")
    assert "ConditionPathExists=!/" in service
    assert service.index("ConditionPathExists=!/") < service.index("[Service]")
    assert service.index("ConditionPathExists=!/") < service.index("ExecStartPre=")

    entries = [
        {
            "path": str(stager.UNIT_TEMPLATE_RELATIVE),
            "type": "file",
        }
    ]
    assert "ConditionPathExists=!/" in stager._expected_unit_lines()[0]
    stager._validate_unit_template(ROOT, entries)


def test_goal_records_r212_as_a_separate_no_mcts_no_rtp_shadow_experiment() -> None:
    goal = " ".join((ROOT / "GOAL.md").read_text(encoding="utf-8").split())

    assert int(goal.split("Revision: `", 1)[1].split("`", 1)[0]) >= 212
    assert "| 212 |" in goal
    assert "| 217 |" in goal
    assert "state/alakazam-guide2vec-no-mcts-bo1000-r212.json" in goal
    assert "no-MCTS/no-RTP mirror" in goal
    assert "500 matched RNG/deck-order pairs" in goal
    assert "one candidate-first and one candidate-second game" in goal
    assert "control runtime contains no Guide2Vec object at all" in goal
    assert "| 226-GUIDE2VEC |" in goal
    assert R226_PIPELINE_PATH in goal
