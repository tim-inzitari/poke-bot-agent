from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "state/alakazam-rtp-abandonment-r210.json"
RETIREMENT_RECEIPT_PATH = (
    ROOT / "state/alakazam-rtp-abandonment-retirement-guard-r210.json"
)
RETIREMENT_DROP_IN_PATH = (
    ROOT
    / "deploy/systemd/pokebot-alakazam-rtp-r198-three-arm-eval.service.d"
    / "99-r210-retired.conf"
)
OLD_CONTRACTS = {
    "state/alakazam-rtp-realignment-r197.json": (
        "ea032624be23341fbae6e0b9b9debf6695a7a3b5a51613cf7294248bdba39c05"
    ),
    "state/alakazam-rtp-continuation-r199.json": (
        "9de3cce02940bec190dd5d7028036e6943511889c85164d165a40c979d9f7869"
    ),
    "state/alakazam-chance-aware-inter-turn-mcts-r202.json": (
        "5df1eadedb342e90c56aa24c5b59f9887c229243411a4112acb6e69562841d32"
    ),
    "state/alakazam-chance-aware-inter-turn-mcts-bo1000-r205.json": (
        "90d1018d67fddc0565adc195f56830aca2f92d60f57d20b6bb1494d956e74a1d"
    ),
    "state/alakazam-chance-aware-inter-turn-mcts-bo1000-r207.json": (
        "d9cb5f8d15e2bebbcbf943f5a273a4116703c3e8549a3328b7d78d161f7b5dce"
    ),
}
PROJECTION_KEY = "alakazam_legacy_recursive_rtp_abandonment_r210"


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_r210_preserves_predecessor_bytes_and_abandons_only_legacy_rtp() -> None:
    contract = _contract()
    assert contract["schema"] == (
        "poke_bot.alakazam_legacy_recursive_rtp_abandonment_r210/v1"
    )
    assert contract["owner_decision_revision"] == 210
    assert contract["supersedes"]["continuation_rescinded"] is True

    for relative_path, expected in OLD_CONTRACTS.items():
        assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == expected

    abandoned = contract["abandoned_scope"]
    assert abandoned["strategy_family"] == "legacy_recursive_turn_planner_rtp"
    assert abandoned["r195_rtp_submission_and_sidecar_are_historical_only"] is True
    assert abandoned["new_recursive_rtp_candidate_or_evaluation_allowed"] is False
    assert abandoned["legacy_rtp_training_or_data_collection_allowed"] is False
    assert abandoned["legacy_rtp_runtime_attachment_allowed"] is False
    assert abandoned["historical_rtp_artifacts_must_be_preserved"] is True


def test_r210_binds_exact_owner_stopped_incomplete_attempt10_snapshot() -> None:
    contract = _contract()
    stop = contract["attempt10_managed_stop"]
    assert stop["owner_ordered_immediate_stop"] is True
    assert stop["managed_service_stop_only"] is True
    assert stop["concurrent_idempotent_stop_requests_observed"] == 2
    assert stop["manual_process_signal_or_kill_command_used"] is False
    assert stop["reset_restart_enable_or_retry_used"] is False
    assert stop["post_stop"] == {
        "observed_at_local": "2026-08-10T11:56:18-04:00",
        "load_state": "loaded",
        "active_state": "failed",
        "sub_state": "failed",
        "main_pid": 0,
        "exec_main_pid": 1431670,
        "result": "signal",
        "exec_main_code": 2,
        "exec_main_status": 15,
        "n_restarts": 0,
        "restart_policy": "no",
        "systemd_term_is_owner_authorized_managed_stop_not_evaluator_failure": True,
    }

    evidence = contract["preserved_partial_evidence"]
    assert evidence["completed_transcripts"] == 761
    assert evidence["completed_execution_receipts"] == 761
    assert evidence["fully_completed_matched_cells"] == 253
    assert evidence["completed_rows_by_arm"] == {
        "no_rtp": 254,
        "direct_bridge_recursive_disabled": 254,
        "recursive_rtp": 253,
    }
    assert evidence["cell_000253_recursive_rtp"]["scored"] is False
    assert evidence["completed_evidence_snapshot_digest"] == (
        "sha256:a561ed820d00b8b9460c0ea0d9aa17c8e0fa82c7834451e7bb13c370b742628b"
    )
    assert evidence["evaluation_tree_content_and_mode_snapshot_digest"] == (
        "sha256:6868ec957f9dd266c18e5d40f2f09fef391dce67a67864e4869be848a3e39ad7"
    )
    assert evidence["roots_are_terminal_sealed"] is False
    assert evidence["only_exact_snapshot_digests_may_be_used_as_preservation_boundary"] is True
    assert evidence["terminal_evaluation_result_count"] == 0
    assert evidence["compiler_result_count"] == 0
    assert evidence["promotion_or_hold_receipt_count"] == 0
    assert evidence["complete_evaluation_claim_allowed"] is False
    assert evidence["terminal_efficacy_claim_allowed"] is False


def test_r210_denies_every_legacy_rtp_execution_and_authority_path() -> None:
    contract = _contract()
    future = contract["legacy_rtp_future"]
    assert future
    assert all(value is False for value in future.values())

    boundary = contract["preservation_and_production_boundary"]
    assert boundary["preserve_r175_and_r195"] is True
    assert boundary["preserve_all_r197_r198_r199_source_evaluation_and_failure_evidence"] is True
    assert boundary["restart_r175_allowed"] is False
    assert boundary["collect_iteration_21_allowed"] is False


def test_r210_retirement_drop_in_preserves_unit_and_denies_every_start_path() -> None:
    assert RETIREMENT_DROP_IN_PATH.read_text(encoding="utf-8") == (
        "# Revision 210 retirement guard.\n"
        "# Contract: sha256:bb9eaa02398175fc5c9bd8e29ce290f102afff234b6d27bf7588fc1e53f09961\n"
        "[Unit]\n"
        "RefuseManualStart=yes\n"
        "\n"
        "[Service]\n"
        "ExecCondition=/usr/bin/false\n"
    )


def test_r210_retirement_receipt_binds_guard_and_stopped_evidence() -> None:
    receipt = json.loads(RETIREMENT_RECEIPT_PATH.read_text(encoding="utf-8"))
    declared = receipt.pop("receipt_payload_sha256")
    payload = json.dumps(
        receipt,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    assert declared == "sha256:" + hashlib.sha256(payload).hexdigest()

    assert receipt["abandonment_contract"] == {
        "path": "state/alakazam-rtp-abandonment-r210.json",
        "sha256": "sha256:bb9eaa02398175fc5c9bd8e29ce290f102afff234b6d27bf7588fc1e53f09961",
        "bytes": 6978,
    }
    drop_in = receipt["retirement_drop_in"]
    assert drop_in["sha256"] == (
        "sha256:2dda98aeee06b3488bf32ade03aaca06f1fcd7a4de8e3d7617425fca55d5f24c"
    )
    assert drop_in["file_mode"] == "0444"
    assert drop_in["refuse_manual_start"] is True
    assert drop_in["exec_condition"] == "/usr/bin/false"

    activation = receipt["activation"]
    assert activation["user_daemon_reload_count"] == 1
    assert activation["production_refusal_probe_performed"] is False
    assert activation[
        "start_stop_restart_reset_failed_enable_disable_mask_unmask_or_unlink_during_guard_install"
    ] is False
    service = receipt["post_activation_service"]
    assert service["main_pid"] == 0
    assert service["n_restarts"] == 0
    assert service["refuse_manual_start"] is True
    assert service["exec_condition"] == "/usr/bin/false"


def test_r210_preserves_separate_r207_mcts_without_legacy_rtp_dependencies() -> None:
    separation = _contract()["separate_non_rtp_turn_planning"]
    assert separation["r202_r205_r207_simulator_backed_mcts_is_legacy_rtp"] is False
    assert separation["offline_implementation_and_preflight_remain_authorized"] is True
    assert separation[
        "exact_shadow_bo1000_remains_authorized_only_after_all_r207_prerequisites"
    ] is True
    assert separation["legacy_rtp_candidate_sidecar_or_executor_may_be_used"] is False
    assert separation["legacy_r198_partial_rows_may_train_calibrate_or_rank_mcts"] is False
    assert separation["attempt10_stop_automatically_grants_train_host_capacity"] is False
    assert separation["per_host_noninterference_preflight_still_required"] is True
    assert separation["comparator_remains_exact_r195_no_rtp"] is True
    assert separation["production_action_or_serving_authority_enabled"] is False


def test_r210_compatibility_projections_are_fail_closed() -> None:
    compatibility = json.loads(
        (ROOT / "ops/current_goal_requirements.json").read_text(encoding="utf-8")
    )["current_owner_overrides"]
    protocol = yaml.safe_load((ROOT / "config/rl_protocol.yaml").read_text())
    specialists = yaml.safe_load((ROOT / "state/specialists.yaml").read_text())

    projections = (
        compatibility[PROJECTION_KEY],
        protocol[PROJECTION_KEY],
        specialists[PROJECTION_KEY],
    )
    for projection in projections:
        revision = projection.get("goal_revision", projection.get("owner_decision_revision"))
        assert revision == 210
        assert "abandoned" in projection["status"]

    compatibility_projection = compatibility[PROJECTION_KEY]
    assert compatibility_projection["persistent_retirement_guard_active"] is True
    assert compatibility_projection["retirement_guard_receipt_sha256"] == (
        "sha256:d0ee2255bf2b5e4abd2c1b9eaaff39343997c2452578d53347927fa5b2f75db0"
    )
    for key in (
        "attempt10_restart_or_retry_allowed",
        "legacy_recursive_rtp_research_enabled",
        "new_recursive_rtp_candidate_or_evaluation_allowed",
        "legacy_rtp_corpus_materialization_or_sidecar_training_allowed",
        "legacy_rtp_stage_run_resume_retry_reset_or_unmask_allowed",
        "training_service_start_authorized",
        "evaluation_service_start_authorized",
        "replay_collection_authorized",
        "serving_eligible",
        "action_authority_enabled",
        "selector_change_authorized",
        "checkpoint_publication_authorized",
        "kaggle_submission_authorized",
        "promotion_authorized",
        "r175_restart_allowed",
        "iter_00021_allowed",
    ):
        assert compatibility_projection[key] is False

    protocol_projection = protocol[PROJECTION_KEY]
    assert protocol_projection["attempt10"]["restart_or_retry_allowed"] is False
    assert protocol_projection["retirement_guard"] == {
        "active": True,
        "receipt": "state/alakazam-rtp-abandonment-retirement-guard-r210.json",
        "receipt_sha256": (
            "sha256:d0ee2255bf2b5e4abd2c1b9eaaff39343997c2452578d53347927fa5b2f75db0"
        ),
        "drop_in_sha256": (
            "sha256:2dda98aeee06b3488bf32ade03aaca06f1fcd7a4de8e3d7617425fca55d5f24c"
        ),
        "refuse_manual_start": True,
        "indirect_activation_exec_condition": "/usr/bin/false",
        "removing_drop_in_alone_authorizes_restart": False,
    }
    assert all(
        value is False for value in protocol_projection["legacy_rtp"].values()
    )
    assert all(
        value is False for value in protocol_projection["authority"].values()
    )

    specialist_projection = specialists[PROJECTION_KEY]
    assert specialist_projection["persistent_retirement_guard_active"] is True
    assert specialist_projection["retirement_guard_receipt_sha256"] == (
        "sha256:d0ee2255bf2b5e4abd2c1b9eaaff39343997c2452578d53347927fa5b2f75db0"
    )
    for key in (
        "active",
        "attempt10_restart_or_retry_allowed",
        "legacy_recursive_rtp_research_enabled",
        "new_recursive_rtp_candidate_or_evaluation_allowed",
        "legacy_rtp_sidecar_attachment_allowed",
        "legacy_rtp_corpus_materialization_or_sidecar_training_allowed",
        "legacy_rtp_stage_run_resume_retry_reset_or_unmask_allowed",
        "training_service_start_authorized",
        "evaluation_service_start_authorized",
        "selector_eligible",
        "serving_eligible",
        "action_authority_enabled",
        "checkpoint_publication_authorized",
        "promotion_authorized",
        "r175_restart_allowed",
        "iter_00021_allowed",
        "automatic_kaggle_submission_allowed",
    ):
        assert specialist_projection[key] is False

    for source in (compatibility, protocol, specialists):
        historical = source["alakazam_rtp_continuation_r199"]
        assert historical["status"] == (
            "superseded_by_r210_owner_abandoned_legacy_recursive_rtp"
        )

    for source, key in (
        (compatibility, "alakazam_chance_aware_inter_turn_mcts_r202"),
        (compatibility, "alakazam_chance_aware_inter_turn_mcts_bo1000_r205"),
        (compatibility, "alakazam_chance_aware_inter_turn_mcts_bo1000_r207"),
        (protocol, "alakazam_chance_aware_inter_turn_mcts_r202"),
        (protocol, "alakazam_chance_aware_inter_turn_mcts_bo1000_r205"),
        (protocol, "alakazam_chance_aware_inter_turn_mcts_bo1000_r207"),
        (specialists, "alakazam_chance_aware_inter_turn_mcts_r202"),
        (specialists, "alakazam_chance_aware_inter_turn_mcts_bo1000_r205"),
        (specialists, "alakazam_chance_aware_inter_turn_mcts_bo1000_r207"),
    ):
        separation = source[key]["r210_separation"]
        assert separation["legacy_recursive_rtp_abandoned"] is True
        assert separation["legacy_rtp_sidecar_executor_or_partial_rows_allowed"] is False
        assert separation["r207_authorization_cancelled"] is False
        assert separation["attempt10_stop_waives_launch_or_host_preflight"] is False


def test_r210_goal_and_human_protocol_record_immediate_abandonment() -> None:
    goal = (ROOT / "GOAL.md").read_text(encoding="utf-8")
    protocol = (ROOT / "docs/RL_TRAINING_PROTOCOL.md").read_text(encoding="utf-8")
    goal_revision = int(goal.split("Revision: `", 1)[1].split("`", 1)[0])
    assert goal_revision >= 210
    assert "| 210 |" in goal
    assert "fully abandons the legacy recursive RTP line" in goal
    assert "state/alakazam-rtp-abandonment-r210.json" in goal
    assert "state/alakazam-rtp-abandonment-retirement-guard-r210.json" in goal
    assert "Revision 210 supersedes revision 199's continuation policy" in protocol
    assert "manual starts are refused" in protocol
    assert "removing that\ndrop-in alone never authorizes a restart" in protocol
    assert "attempt-10 stop alone does not grant train-host capacity" in protocol
