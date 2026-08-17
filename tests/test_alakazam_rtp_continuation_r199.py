from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "state/alakazam-rtp-continuation-r199.json"
R198_CONTRACT_PATH = ROOT / "state/alakazam-rtp-realignment-r197.json"
R198_CONTRACT_SHA256 = (
    "ea032624be23341fbae6e0b9b9debf6695a7a3b5a51613cf7294248bdba39c05"
)
STATUS = "attempt10_in_progress_shadow_only_pending_terminal_evidence"
EVALUATION_ID = (
    "r198-three-arm-"
    "6a0d99cb02d5ca02318f8725c6523517fee1b9a97d76b05aa6bb0fddfc680105"
)


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_r199_continues_shadow_research_without_mutating_frozen_r198() -> None:
    contract = _contract()

    assert contract["schema"] == "poke_bot.alakazam_rtp_continuation_r199/v1"
    assert contract["owner_decision_revision"] == 199
    assert contract["status"] == STATUS
    assert contract["provisional_abandonment_instruction"] == {
        "rescinded_before_activation": True,
        "live_prefix_is_abandonment_efficacy_promotion_or_stop_authority": False,
        "owner_authorized_abandonment_threshold": None,
    }

    frozen = contract["frozen_r198_contract"]
    assert frozen["owner_decision_revision"] == 198
    assert frozen["must_remain_byte_identical_for_attempt10"] is True
    assert frozen["sha256"] == f"sha256:{R198_CONTRACT_SHA256}"
    assert hashlib.sha256(R198_CONTRACT_PATH.read_bytes()).hexdigest() == (
        R198_CONTRACT_SHA256
    )

    attempt = contract["attempt10"]
    assert attempt["evaluation_id"] == EVALUATION_ID
    assert attempt["frozen"] is True
    assert attempt["must_complete_to_terminal_evidence_or_fail_closed"] is True
    assert attempt["preemption_allowed"] is False
    assert attempt["alteration_allowed"] is False
    assert attempt["retry_in_place_allowed"] is False
    assert attempt["prefix_is_nonterminal"] is True
    assert attempt["prefix_can_change_selector_or_authority"] is False
    assert attempt["evaluation_input_contract"]["canonical_sha256"] == (
        "sha256:6a0d99cb02d5ca02318f8725c6523517fee1b9a97d76b05aa6bb0fddfc680105"
    )


def test_r199_preserves_gates_and_requires_new_followup_identities() -> None:
    contract = _contract()
    terminal = contract["terminal_evidence"]
    assert terminal["matched_cells_required_for_completed_evaluation"] == 1000
    assert terminal["expected_arm_rows_for_completed_evaluation"] == 3000
    assert terminal["immutable_evaluation_receipt_required"] is True
    assert terminal["compiler_and_binding_audit_required"] is True
    assert terminal["failed_boundary_is_efficacy_evidence"] is False
    assert terminal["hold_or_rejection_is_abandonment_authority"] is False
    assert terminal["valid_compiler_statuses"] == [
        "hold",
        "ready_for_separate_promotion_review",
    ]

    followup = contract["post_terminal_r_and_d"]
    assert followup["enabled"] is True
    assert followup["new_separately_versioned_contract_required"] is True
    assert followup["new_content_addressed_source_snapshot_required"] is True
    assert followup["new_candidate_id_required"] is True
    assert followup["new_evaluation_identity_required"] is True
    assert followup["new_output_root_required"] is True
    assert followup["reuse_or_rewrite_failed_attempt_allowed"] is False
    assert followup["gate_weakening_allowed"] is False

    limits = contract["inherited_r198_limits_and_gates"]
    assert limits["max_action_combos"] == 1024
    assert limits["max_neural_passes"] == 256
    assert limits["required_neural_passes_normal"] == 6
    assert limits["required_neural_passes_forced_replan"] == 5
    assert limits["automatic_budget_escalation_allowed"] is False
    for gate in (
        "paired_three_arm_evaluation_required",
        "reliability_gate_required",
        "latency_gate_required",
        "legality_and_forfeit_nonregression_required",
        "source_excluded_recursive_over_direct_improvement_required",
        "trustworthy_counterfactual_targets_required_for_action_authority",
        "separate_immutable_promotion_receipt_required",
    ):
        assert limits[gate] is True

    assert all(value is False for value in contract["authority"].values())
    boundary = contract["production_boundary"]
    assert boundary["preserve_r175_and_r195"] is True
    assert boundary["restart_r175_allowed"] is False
    assert boundary["collect_iteration_21_allowed"] is False


def test_r199_compatibility_projections_are_preserved_but_superseded_by_r210() -> None:
    contract = _contract()
    compatibility = json.loads(
        (ROOT / "ops/current_goal_requirements.json").read_text(encoding="utf-8")
    )["current_owner_overrides"]
    protocol = yaml.safe_load(
        (ROOT / "config/rl_protocol.yaml").read_text(encoding="utf-8")
    )
    specialists = yaml.safe_load(
        (ROOT / "state/specialists.yaml").read_text(encoding="utf-8")
    )

    r198_projections = (
        compatibility["alakazam_rtp_realignment_r197"],
        protocol["alakazam_rtp_realignment_r197"],
        specialists["alakazam_rtp_realignment_r197"],
    )
    for projection in r198_projections:
        revision = projection.get("goal_revision", projection.get("owner_decision_revision"))
        assert revision == 198
        assert projection["status"] == "authorized_shadow_only_pending_materialization"

    r199_projections = (
        compatibility["alakazam_rtp_continuation_r199"],
        protocol["alakazam_rtp_continuation_r199"],
        specialists["alakazam_rtp_continuation_r199"],
    )
    for projection in r199_projections:
        revision = projection.get("goal_revision", projection.get("owner_decision_revision"))
        assert revision == contract["owner_decision_revision"]
        assert projection["status"] == (
            "superseded_by_r210_owner_abandoned_legacy_recursive_rtp"
        )
        assert projection["historical_status"] == contract["status"]

    assert compatibility["alakazam_rtp_continuation_r199"]["retry_in_place_allowed"] is False
    assert protocol["alakazam_rtp_continuation_r199"]["attempt10"]["retry_in_place_allowed"] is False
    assert specialists["alakazam_rtp_continuation_r199"]["retry_in_place_allowed"] is False
    assert compatibility["alakazam_rtp_continuation_r199"][
        "must_complete_to_terminal_evidence_or_fail_closed"
    ] is False
    assert protocol["alakazam_rtp_continuation_r199"]["attempt10"][
        "must_complete_to_terminal_evidence_or_fail_closed"
    ] is False
    assert specialists["alakazam_rtp_continuation_r199"][
        "must_complete_to_terminal_evidence_or_fail_closed"
    ] is False

    compatibility_r199 = compatibility["alakazam_rtp_continuation_r199"]
    for key in (
        "serving_eligible",
        "action_authority_enabled",
        "selector_change_authorized",
        "checkpoint_publication_authorized",
        "kaggle_submission_authorized",
        "promotion_authorized",
        "r175_restart_allowed",
        "iter_00021_allowed",
    ):
        assert compatibility_r199[key] is False

    protocol_authority = protocol["alakazam_rtp_continuation_r199"]["authority"]
    assert all(value is False for value in protocol_authority.values())

    specialists_r199 = specialists["alakazam_rtp_continuation_r199"]
    for key in (
        "selector_eligible",
        "serving_eligible",
        "action_authority_enabled",
        "checkpoint_publication_authorized",
        "promotion_authorized",
        "r175_restart_allowed",
        "iter_00021_allowed",
        "automatic_kaggle_submission_allowed",
    ):
        assert specialists_r199[key] is False

    stage = (ROOT / "scripts/stage_alakazam_rtp_r198_three_arm_eval.py").read_text(
        encoding="utf-8"
    )
    assert 'R198_TYPED_CONTRACT_PATH = ROOT / "state/alakazam-rtp-realignment-r197.json"' in stage
    assert "alakazam-rtp-continuation-r199" not in stage


def test_r199_goal_and_protocol_text_preserve_history_under_r210() -> None:
    goal = (ROOT / "GOAL.md").read_text(encoding="utf-8")
    protocol = (ROOT / "docs/RL_TRAINING_PROTOCOL.md").read_text(encoding="utf-8")
    flat_goal = " ".join(goal.split())
    flat_protocol = " ".join(protocol.split())

    goal_revision = int(goal.split("Revision: `", 1)[1].split("`", 1)[0])
    assert goal_revision >= 199
    assert "| 199 |" in goal
    assert "Continue iterating on Alakazam RTP" in goal
    assert "do not preempt, restart, alter, or retry it in place" in flat_goal
    assert "state/alakazam-rtp-continuation-r199.json" in goal
    assert "Revision 199 historically continued RTP" in protocol
    assert "live prefix is not terminal" in flat_protocol
    assert "new content-addressed source snapshot" in flat_protocol
    assert "| 210 |" in goal
    assert "state/alakazam-rtp-abandonment-r210.json" in goal
