from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pytest

from poke_bot.alakazam_turn_checklist_bo250_r289 import (
    CALIBRATION_ELIGIBLE_GATE_NAMES,
    CHECKLIST_CHANNELS,
    CHECKLIST_GATE_NAMES,
    COMPARISON_IDS,
    CONTROL_ARMS,
    EVALUATION_ID,
    GAME_COUNT,
    PAIR_COUNT,
    R289BO250Error,
    build_run_identity,
    build_schedule,
    compile_three_comparison_report,
    comparison_evaluation_id,
    canonical_digest,
    derive_comparison_seed_identity,
    compile_report,
    empty_checklist_telemetry,
    load_r289_config,
    make_game_receipt,
    read_exact_new_list_deck,
    read_r195_native_deck,
    schedule_identity,
    stage_verified_copy,
    validate_game_receipt,
    validate_optional_calibration_receipt,
    validate_r298_collision_census_receipt,
    validate_r298_raw_corpus_receipt,
    validate_derivative_goal_contract,
    validate_required_calibration_receipt,
    validate_r293_overlap_audit_receipt,
    validate_schedule,
    write_create_only_json,
    file_identity,
)


ROOT = Path(__file__).resolve().parents[1]
SEED = "sha256:" + "a" * 64
RUN = "sha256:" + "b" * 64
PREFLIGHT = "sha256:" + "c" * 64


def _runner_module():
    path = ROOT / "scripts" / "run_alakazam_turn_checklist_bo250_r289.py"
    spec = importlib.util.spec_from_file_location("r289_bo250_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_stage() -> dict[str, object]:
    zeros = [0.0]
    channels = [
        {
            "name": name,
            "raw": list(zeros),
            "normalized": list(zeros),
            "option_availability": [False],
            "available": False,
            "status": "unavailable",
            "reason": "fixture",
        }
        for name in CHECKLIST_CHANNELS
    ]
    overlap = {
        name: {
            "existing_route_overlap_or_distinct_reason": "fixture audit",
            "attenuation_or_suppression_decision": "fixture bounded gate",
            "post_deduplication_signed_residual": list(zeros),
        }
        for name in CHECKLIST_CHANNELS
    }
    scalar_gates = {
        name: (0.01 if name in CALIBRATION_ELIGIBLE_GATE_NAMES else 0.0)
        for name in CHECKLIST_GATE_NAMES
    }
    return {
        "enabled": True,
        "applied": True,
        "evaluation_mode": "evaluated",
        "action_authority": True,
        "channel_names": list(CHECKLIST_CHANNELS),
        "channels": channels,
        "guide_support": {
            "name": "guide_support",
            "raw": list(zeros),
            "normalized": list(zeros),
            "option_availability": [False],
            "available": False,
            "status": "unavailable",
            "reason": "trace_only",
        },
        "scalar_gates": scalar_gates,
        "module_residuals_before_whole_decision_cap": list(zeros),
        "residuals": list(zeros),
        "facts": {},
        "available": False,
        "reason": "fixture",
        "active": False,
        "score_space": "local_logits",
        "base_argmax_index": 0,
        "adjusted_argmax_index": 0,
        "stage_argmax_changed_at_prefix": False,
        "action_changed_from_base_policy": False,
        "factorized_stage_index": 0,
        "factorized_prefix": [],
        "candidate_rows": [[0]],
        "selected_candidate_index": 0,
        "selected_stage_index": 0,
        "selected_candidate": [0],
        "whole_decision_budget_initial": 0.10,
        "whole_decision_budget_before": 0.10,
        "whole_decision_budget_consumed": 0.0,
        "whole_decision_budget_remaining": 0.10,
        "channel_overlap_audit": overlap,
        "guide_support_trace_only": True,
        "guide_support_runtime_residual": 0.0,
    }


def _identity(path: str) -> dict[str, object]:
    return {"path": path, "sha256": "sha256:" + "d" * 64, "size_bytes": 1}


def _receipt(spec, *, winner: int) -> dict[str, object]:
    return make_game_receipt(
        run_identity_sha256=RUN,
        spec=spec,
        first_player_seat=0,
        winner_seat=winner,
        steps=12,
        checklist_telemetry=empty_checklist_telemetry(),
        runtime_preflight_sha256=PREFLIGHT,
        direct_policy_flags={
            "rtp": False,
            "search": False,
            "mcts": False,
            "rollout": False,
            "hidden_information_inference": False,
            "candidate_checklist_layer": True,
            "control_checklist_layer": False,
        },
        pair_first_player_seal_sha256=canonical_digest({"pair": spec.pair_index}),
        stage_trace_digest=canonical_digest([]),
    )


def test_r289_config_and_exact_deck_bindings_are_loadable() -> None:
    config, _ = load_r289_config(
        ROOT / "config/evaluations/alakazam-turn-checklist-bo250-r289.json"
    )
    deck, identity = read_exact_new_list_deck(
        ROOT / "decks/archetype-samples/alakazam-new-list-direct-r241.csv"
    )

    assert config["evaluation_id"] == EVALUATION_ID
    assert len(deck) == 60
    assert identity["card_count"] == 60
    assert config["checklist_channels"] == list(CHECKLIST_CHANNELS)


def test_revision_4_derivative_contract_is_exactly_bound() -> None:
    identity = validate_derivative_goal_contract(
        ROOT / "goals/alakazam-elmo-rule-derivative/contract.json"
    )
    assert identity["sha256"] == (
        "sha256:f65e023d454375cfd59324306044da10a116201a187415f0534e24c239bd2dc2"
    )


def test_runner_refuses_to_reinterpret_a_receipted_wrapper_as_r288_runtime(
    tmp_path,
) -> None:
    runner = _runner_module()
    receipt = {
        "checklist_provenance": {
            "candidate_action_time_wrapper_sealed": True,
            "acting_player_public_information_only": True,
            "strict_q1_q2_q8_provenance_enforced": True,
            "q5_q6_exact_zero": True,
            "legacy_r288_runtime_residual": 0.0,
        }
    }
    receipt_path = tmp_path / "r298-validation.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing the legacy r288 fallback"):
        runner._require_wired_strict_r298_runtime_wrapper(
            r298_receipt_path=receipt_path
        )


def test_r289_schedule_and_report_have_exact_seat_and_actual_order_balance() -> None:
    schedule = build_schedule(SEED)
    assert len(schedule) == GAME_COUNT
    assert sum(spec.experimental_seat == 0 for spec in schedule) == PAIR_COUNT
    assert sum(spec.experimental_seat == 1 for spec in schedule) == PAIR_COUNT

    receipts = [
        _receipt(
            spec,
            # Candidate wins its seat-0 game and loses its seat-1 game.
            winner=spec.experimental_seat if spec.game_index == 0 else spec.control_seat,
        )
        for spec in schedule
    ]
    report = compile_report(
        run_identity_sha256=RUN,
        runtime_preflight_sha256=PREFLIGHT,
        seed_identity_sha256=SEED,
        schedule=schedule,
        game_receipts=receipts,
        input_identities={"fixture": True},
    )

    assert report["pair_count"] == PAIR_COUNT
    assert report["game_count"] == GAME_COUNT
    assert report["candidate_results"] == {"wins": 125, "losses": 125, "ties": 0}
    assert all(
        sum(row.values()) == PAIR_COUNT
        for row in report["candidate_results_by_seat"].values()
    )
    assert all(
        sum(row.values()) == PAIR_COUNT
        for row in report["candidate_results_by_actual_turn_order"].values()
    )


def test_r289_report_rejects_double_credit() -> None:
    schedule = build_schedule(SEED)
    receipts = [_receipt(spec, winner=spec.experimental_seat) for spec in schedule]
    receipts[-1] = dict(receipts[0])

    with pytest.raises(R289BO250Error, match="double-credited"):
        compile_report(
            run_identity_sha256=RUN,
            runtime_preflight_sha256=PREFLIGHT,
            seed_identity_sha256=SEED,
            schedule=schedule,
            game_receipts=receipts,
            input_identities={"fixture": True},
        )


def test_r289_schedule_rejects_a_structurally_plausible_seed_substitution() -> None:
    schedule = list(build_schedule(SEED))
    # A different valid pair schedule is the meaningful substitution attempt.
    other = build_schedule("sha256:" + "f" * 64)
    schedule[0] = other[0]
    with pytest.raises(R289BO250Error, match="deterministic seed identity"):
        validate_schedule(schedule, seed_identity_sha256=SEED)


def test_run_identity_is_content_addressed_not_source_path_addressed() -> None:
    identity_a = _identity("/elmo/a/candidate.pt")
    identity_b = _identity("/elmo/b/candidate.pt")
    common = _identity("/elmo/common/file")
    schedule = build_schedule(SEED)
    kwargs = {
        "config_identity": common,
        "owner_contract_identity": common,
        "candidate_receipt": common,
        "control_checkpoint": common,
        "r195_contract": common,
        "r195_bundle": common,
        "exact_deck": {**common, "canonical_multiset_sha256": "sha256:" + "e" * 64},
        "checklist_config": common,
        "candidate_matchup_tree": common,
        "control_matchup_tree": common,
        "seeded_engine": common,
        "seeded_engine_receipt": common,
        "r293_overlap_audit_receipt": common,
        "calibration_receipt": None,
        "runtime_sources": {"poke_bot/agent.py": common},
        "seed_identity_sha256": SEED,
        "schedule_sha256": schedule_identity(schedule, seed_identity_sha256=SEED),
    }

    first = build_run_identity(candidate_checkpoint=identity_a, **kwargs)
    second = build_run_identity(candidate_checkpoint=identity_b, **kwargs)
    assert first == second


def test_create_only_stage_and_receipt_validation(tmp_path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"r289 immutable input")
    expected = file_identity(source, label="fixture source")
    target = tmp_path / "out" / "copy.bin"
    first = stage_verified_copy(source, target, expected_identity=expected, label="fixture")
    second = stage_verified_copy(source, target, expected_identity=expected, label="fixture")
    assert first["sha256"] == second["sha256"] == expected["sha256"]

    payload = {"schema": "fixture", "value": 1}
    write_create_only_json(tmp_path / "out" / "receipt.json", payload, label="fixture")
    write_create_only_json(tmp_path / "out" / "receipt.json", payload, label="fixture")

    spec = build_schedule(SEED)[0]
    row = _receipt(spec, winner=spec.experimental_seat)
    assert (
        validate_game_receipt(
            row,
            spec=spec,
            run_identity_sha256=RUN,
            runtime_preflight_sha256=PREFLIGHT,
        )
        == row
    )


def test_calibration_receipt_must_explicitly_bind_disjoint_bo250_seed(tmp_path) -> None:
    receipt = {
        "schema": "poke_bot.alakazam_turn_checklist_gate_calibration_receipt/v1",
        "status": "completed",
        "source_disjoint_exact_new_list_data": True,
        "all_neural_model_and_checkpoint_tensors_frozen": True,
        "training_eligible": False,
        "replay_eligible": False,
        "production_authority": False,
        "bo250_seed_disjoint": True,
        "bo250_seed_identities_excluded": [SEED],
        "config": {
            "corrected_guide_attachment_sha256": "sha256:5cc092c9ed93b3e0e4ecae9fca9d50409bea6979e8d92e358f684091e0cdff8b",
        },
        "overlap_deduplication": {
            "post_deduplication_vectors_required": True,
            "post_deduplication_vectors_attested_for_all_rows": True,
            "existing_learned_logic_modified": False,
        },
        "attestations": {
            "exactly_six_checklist_scalar_gates_fitted": True,
            "all_eight_checklist_channels_traced": True,
            "bench_prize_exposure_gate_trace_only_exact_zero": True,
            "immediate_disruption_outcome_gate_trace_only_exact_zero": True,
            "guide_gate_separate_trace_only_exact_zero": True,
            "guide_support_calibration_eligible": False,
            "strongest_per_group_aggregation_exact": True,
            "winner_identity_trace_bound_in_artifact": True,
            "post_deduplication_vectors_only": True,
            "corrected_r295_guide_attachment_bound": True,
            "input_training_eligible": False,
            "input_production_eligible": False,
            "input_replay_eligible": False,
            "learner_or_checkpoint_training_authority": False,
            "runtime_activation_authority": False,
        },
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    assert validate_optional_calibration_receipt(
        path, evaluation_seed_identity_sha256=SEED
    )["path"] == str(path.resolve())

    receipt["bo250_seed_disjoint"] = False
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(R289BO250Error, match="seed/source-disjoint"):
        validate_optional_calibration_receipt(path, evaluation_seed_identity_sha256=SEED)


def test_r293_audit_requires_trace_only_broad_guide_and_all_channels(tmp_path) -> None:
    checklist = file_identity(
        ROOT / "config/policy_layers/alakazam-turn-checklist-r288.json",
        label="r288 checklist fixture",
    )
    receipt = {
        "schema": "poke_bot.alakazam_turn_checklist_r293_overlap_audit_receipt/v1",
        "status": "completed",
        "scope": "elmo_only_nonproduction",
        "candidate": {
            "checkpoint_sha256": "sha256:645f8e6a0bc5e0cb98695e5a65151f38eef2e5011ca63f3ea4eded42cf4a11a2",
            "checkpoint_size_bytes": 134661950,
            "matchup_tree_sha256": "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049",
        },
        "exact_new_list_multiset_sha256": "sha256:a42e047c45c419a599a31f2e20a6209d324558082f27e12091ade8918376d182",
        "checklist_base_config_sha256": checklist["sha256"],
        "r288_candidate_package_parity_passed": True,
        "r292_bench_only_regression_passed": True,
        "r292_local_elmo_parity_passed": True,
        "r293_overlap_audit_passed": True,
        "r293_per_channel_trace_contract_passed": True,
        "r295_corrected_attachment_sha256": "sha256:5cc092c9ed93b3e0e4ecae9fca9d50409bea6979e8d92e358f684091e0cdff8b",
        "legacy_broad_guide": {"trace_only": True, "runtime_residual": 0.0},
        "direct_policy_boundary": {
            "rtp": False,
            "search": False,
            "mcts": False,
            "rollout": False,
            "hidden_information_inference": False,
        },
        "authority": {
            "training_eligible": False,
            "replay_eligible": False,
            "production_authority": False,
            "promotion_authority": False,
            "selector_authority": False,
            "kaggle_authority": False,
            "submission_authority": False,
            "elmo_production_readmission_authority": False,
        },
        "channel_audits": {
            name: {
                "existing_route_overlap_or_distinct_reason": "new heuristic is separately audited",
                "attenuation_or_suppression_decision": "bounded r293 new-layer gate",
                "post_deduplication_signed_residual": [0.0],
            }
            for name in CHECKLIST_CHANNELS
        },
    }
    path = tmp_path / "r293-audit.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    assert validate_r293_overlap_audit_receipt(
        path, checklist_config_identity=checklist
    )["path"] == str(path.resolve())

    receipt["legacy_broad_guide"]["runtime_residual"] = 0.01
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(R289BO250Error, match="exact zero"):
        validate_r293_overlap_audit_receipt(path, checklist_config_identity=checklist)


def test_runner_trace_revalidation_binds_six_gates_and_whole_decision_budget() -> None:
    runner = _runner_module()
    live_trace = {
        "enabled": True,
        "selected_action": [0],
        "stages": [_valid_stage()],
        "whole_decision_budget_initial": 0.10,
        "whole_decision_budget_consumed": 0.0,
        "whole_decision_budget_remaining": 0.10,
    }
    telemetry = empty_checklist_telemetry()
    stable = runner._record_candidate_trace(
        telemetry=telemetry,
        trace=live_trace,
        selected_action=[0],
        game_step=7,
    )
    spec = build_schedule(SEED)[0]
    document = runner._trace_document(
        run_identity_sha256=RUN,
        runtime_preflight_sha256=PREFLIGHT,
        spec=spec,
        stage_traces=[stable],
    )
    digest, recovered_telemetry = runner._validate_trace_document(
        document,
        spec=spec,
        run_identity_sha256=RUN,
        runtime_preflight_sha256=PREFLIGHT,
    )
    assert digest == document["candidate_stage_trace_sha256"]
    assert recovered_telemetry == telemetry

    bad = _valid_stage()
    bad["scalar_gates"]["bench_prize_exposure_gate"] = 0.01
    with pytest.raises(RuntimeError, match="trace-only checklist gate"):
        runner._record_candidate_trace(
            telemetry=empty_checklist_telemetry(),
            trace={
                "enabled": True,
                "selected_action": [0],
                "stages": [bad],
                "whole_decision_budget_initial": 0.10,
                "whole_decision_budget_consumed": 0.0,
                "whole_decision_budget_remaining": 0.10,
            },
            selected_action=[0],
            game_step=7,
        )


def test_three_comparison_seed_namespaces_and_combined_report_are_separate() -> None:
    flags = {
        "rtp": False,
        "search": False,
        "mcts": False,
        "rollout": False,
        "hidden_information_inference": False,
        "candidate_checklist_layer": True,
        "control_checklist_layer": False,
    }
    reports: dict[str, dict[str, object]] = {}
    seeds = {
        comparison_id: derive_comparison_seed_identity(
            SEED, comparison_id=comparison_id
        )
        for comparison_id in COMPARISON_IDS
    }
    assert len(set(seeds.values())) == 3
    for comparison_id in COMPARISON_IDS:
        schedule = build_schedule(seeds[comparison_id], comparison_id=comparison_id)
        assert {row.evaluation_id for row in schedule} == {
            comparison_evaluation_id(comparison_id)
        }
        receipts = [
            make_game_receipt(
                run_identity_sha256=RUN,
                spec=spec,
                first_player_seat=0,
                winner_seat=2,
                steps=1,
                checklist_telemetry=empty_checklist_telemetry(),
                runtime_preflight_sha256=PREFLIGHT,
                direct_policy_flags=flags,
                pair_first_player_seal_sha256=canonical_digest(
                    {"game": spec.game_nonce_sha256}
                ),
                stage_trace_digest=canonical_digest([]),
            )
            for spec in schedule
        ]
        reports[comparison_id] = compile_report(
            run_identity_sha256=RUN,
            runtime_preflight_sha256=PREFLIGHT,
            seed_identity_sha256=seeds[comparison_id],
            schedule=schedule,
            game_receipts=receipts,
            input_identities={"fixture": comparison_id},
            comparison_id=comparison_id,
        )
        assert reports[comparison_id]["control_arm"] == CONTROL_ARMS[comparison_id]
    combined = compile_three_comparison_report(
        run_identity_sha256=RUN,
        benchmark_seed_identity_sha256=SEED,
        comparison_reports=reports,
        input_identities={"fixture": True},
    )
    assert combined["total_game_count"] == 750
    assert combined["limitations"]["cohort_C_deck_shift_confound"] is True


def test_r195_native_deck_is_order_and_multiset_bound() -> None:
    cards, identity = read_r195_native_deck(
        ROOT / "decks/archetype-samples/alakazam-owner-rtp-pilot-r175.csv"
    )
    assert len(cards) == 60
    assert identity["ordered_cards_sha256"] == (
        "sha256:660c1274aac19d88c40fd2bb52187f53dc639d944506760e386f2686b91cc247"
    )


def test_rev4_raw_and_collision_receipts_are_exactly_bound(tmp_path) -> None:
    raw = {
        "schema": "poke_bot.alakazam_collision_census_r298_raw_expert_corpus_receipt/v1",
        "status": "passed",
        "owner_goal_sha256": "sha256:0f440fc71043b4352e6401a3187c9d582c1c5614d76e186095e0eef51017af6f",
        "rule_derivative_contract_sha256": "sha256:f65e023d454375cfd59324306044da10a116201a187415f0534e24c239bd2dc2",
        "rule_derivative_gateway_sha256": "sha256:2af67560510ca7ffd9fe0bc6ff37cdbbd74f5a78d6c5237091bb527d49ce4ed8",
        "mechanics_attachment_sha256": "sha256:d3f06071663dde2ae7012da72b407b410c7facd06d09ab723cad05af44ddb2cb",
        "raw_expert_corpus_manifest_sha256": "sha256:" + "1" * 64,
        "window_start_utc": "2026-07-13",
        "window_end_utc": "2026-08-11",
        "distinct_utc_day_count": 30,
        "recollection_authorized": False,
        "source_disjointness": {
            "archive_date_source_sha256_unique": True,
            "episode_identity_unique": True,
            "episode_id_content_unique": True,
            "source_window_blending_permitted": False,
        },
        "runtime_authority": {
            "elmo_only": True,
            "create_only": True,
            "production_activation": False,
            "inzi_mutation": False,
            "archive_mutation": False,
        },
    }
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    raw_identity = validate_r298_raw_corpus_receipt(raw_path)
    collision = {
        "schema": "poke_bot.alakazam_collision_census_r298_receipt/v1",
        "status": "passed_no_actionable_public_semantic_collision",
        "pass": True,
        "owner_goal_sha256": raw["owner_goal_sha256"],
        "rule_derivative_contract_sha256": raw["rule_derivative_contract_sha256"],
        "rule_derivative_gateway_sha256": raw["rule_derivative_gateway_sha256"],
        "mechanics_attachment_sha256": raw["mechanics_attachment_sha256"],
        "raw_expert_corpus_receipt_sha256": raw_identity["sha256"],
        "raw_expert_corpus": {
            "window_start_utc": "2026-07-13",
            "window_end_utc": "2026-08-11",
            "distinct_utc_day_count": 30,
            "source_disjointness": raw["source_disjointness"],
        },
        "pinned_simulator": {
            "libcg_sha256": "sha256:d16244a3157fc55c3314f08dcc7c5179168697d78c105b95c7debd556b764bb7",
            "complete_evidence_required_for_pass": True,
            "hidden_branching_permitted": False,
        },
        "runtime_authority": {
            "elmo_only": True,
            "create_only": True,
            "training_eligible": False,
            "production_activation": False,
            "inzi_mutation": False,
            "learned_baseline_mutation": False,
        },
    }
    collision_path = tmp_path / "collision.json"
    collision_path.write_text(json.dumps(collision), encoding="utf-8")
    assert validate_r298_collision_census_receipt(
        collision_path, raw_corpus_receipt_identity=raw_identity
    )["path"] == str(collision_path.resolve())


def test_required_calibration_excludes_all_derived_comparison_seeds(tmp_path) -> None:
    seeds = {
        comparison_id: derive_comparison_seed_identity(SEED, comparison_id=comparison_id)
        for comparison_id in COMPARISON_IDS
    }
    receipt = {
        "schema": "poke_bot.alakazam_turn_checklist_gate_calibration_receipt/v1",
        "status": "completed",
        "source_disjoint_exact_new_list_data": True,
        "all_neural_model_and_checkpoint_tensors_frozen": True,
        "training_eligible": False,
        "replay_eligible": False,
        "production_authority": False,
        "bo250_seed_disjoint": True,
        "bo250_seed_identities_excluded": list(seeds.values()),
        "config": {
            "corrected_guide_attachment_sha256": "sha256:5cc092c9ed93b3e0e4ecae9fca9d50409bea6979e8d92e358f684091e0cdff8b"
        },
        "overlap_deduplication": {
            "post_deduplication_vectors_required": True,
            "post_deduplication_vectors_attested_for_all_rows": True,
            "existing_learned_logic_modified": False,
        },
        "attestations": {
            "exactly_six_checklist_scalar_gates_fitted": True,
            "all_eight_checklist_channels_traced": True,
            "bench_prize_exposure_gate_trace_only_exact_zero": True,
            "immediate_disruption_outcome_gate_trace_only_exact_zero": True,
            "guide_gate_separate_trace_only_exact_zero": True,
            "guide_support_calibration_eligible": False,
            "strongest_per_group_aggregation_exact": True,
            "winner_identity_trace_bound_in_artifact": True,
            "post_deduplication_vectors_only": True,
            "corrected_r295_guide_attachment_bound": True,
            "input_training_eligible": False,
            "input_production_eligible": False,
            "input_replay_eligible": False,
            "learner_or_checkpoint_training_authority": False,
            "runtime_activation_authority": False,
        },
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    assert validate_required_calibration_receipt(
        path, comparison_seed_identities=seeds
    )["path"] == str(path.resolve())
