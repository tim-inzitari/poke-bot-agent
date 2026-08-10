from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "state/alakazam-rtp-realignment-r197.json"
PARENT_SHA256 = (
    "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
)
POINTER_SHA256 = (
    "sha256:2427c2b51cc93beccc3618085d9c77c83f49fb69cabf0208040608c384a659cd"
)
MANIFEST_SHA256 = (
    "sha256:192fce8878db8a3f7c65d898d2d5e32e9ebf9a011f37c61e267e17a70da57990"
)


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_r197_owner_contract_is_shadow_only_and_checksum_bound() -> None:
    contract = _contract()

    assert contract["schema"] == "poke_bot.alakazam_rtp_realignment_r197/v1"
    assert contract["owner_decision_revision"] == 198
    assert contract["status"] == "authorized_shadow_only_pending_materialization"
    assert contract["parent"]["checkpoint_sha256"] == PARENT_SHA256
    assert contract["data"]["protected_pointer_sha256"] == POINTER_SHA256
    assert contract["data"]["split_unit"] == "whole_game"
    assert contract["data"]["split_seed"] == 5_000_000
    assert contract["data"]["heldout_fraction"] == 0.2
    assert contract["data"]["complete_ordered_legal_action_combos_required"] is True
    assert contract["data"]["max_action_combos"] == 1024
    assert contract["data"]["factorized_policy_stage_substitution_allowed"] is False

    planner = contract["planner"]
    assert planner["sizing_profile"] == "pure_rl_r197"
    assert planner["legacy_pure_rl_profile_unchanged"] is True
    assert planner["num_plan_candidates"] == 4
    assert planner["max_recursion_depth"] == 2
    assert planner["max_neural_passes"] == 256
    assert planner["required_neural_passes_normal"] == 6
    assert planner["required_neural_passes_forced_replan"] == 5
    assert planner["absolute_owner_max_neural_passes"] == 256
    assert planner["automatic_pass_ceiling_escalation_allowed"] is False
    assert planner["above_256_passes_allowed"] is False
    assert planner["revision_197_32_pass_draft_superseded"] is True

    promotion = contract["promotion"]
    assert "max_action_combos" in promotion["required_receipt_fields"]
    assert promotion["initial_candidate_max_neural_passes"] == 256
    assert promotion["required_max_action_combos"] == 1024

    boundary = contract["production_boundary"]
    assert boundary["restart_r175_allowed"] is False
    assert boundary["collect_iteration_21_allowed"] is False
    assert boundary["selector_change_during_shadow_build_allowed"] is False
    assert boundary["new_kaggle_submission_authorized"] is False
    assert contract["artifact_contract"]["initial_serving_eligible"] is False
    assert contract["artifact_contract"]["initial_action_authority_enabled"] is False


def test_r197_compatibility_projections_match_the_typed_contract() -> None:
    contract = _contract()
    compatibility = json.loads(
        (ROOT / "ops/current_goal_requirements.json").read_text(encoding="utf-8")
    )["current_owner_overrides"]["alakazam_rtp_realignment_r197"]
    protocol = yaml.safe_load(
        (ROOT / "config/rl_protocol.yaml").read_text(encoding="utf-8")
    )["alakazam_rtp_realignment_r197"]
    specialists = yaml.safe_load(
        (ROOT / "state/specialists.yaml").read_text(encoding="utf-8")
    )["alakazam_rtp_realignment_r197"]

    for projection in (compatibility, protocol, specialists):
        assert projection["status"] == contract["status"]
        assert projection["parent_checkpoint_sha256"] == PARENT_SHA256
        sizing_profile = projection.get("sizing_profile")
        if sizing_profile is None:
            sizing_profile = projection["runtime"]["sizing_profile"]
        assert sizing_profile == "pure_rl_r197"

    assert compatibility["source_pointer_sha256"] == POINTER_SHA256
    assert compatibility["source_manifest_sha256"] == MANIFEST_SHA256
    assert compatibility["split_seed"] == 5_000_000
    assert compatibility["heldout_fraction"] == 0.2
    assert compatibility["initial_max_neural_passes"] == 256
    assert compatibility["max_action_combos"] == 1024
    assert compatibility["absolute_owner_max_neural_passes"] == 256
    assert compatibility["automatic_budget_escalation"] is False
    assert compatibility["revision_197_32_pass_draft_superseded"] is True

    assert protocol["runtime"]["initial_max_neural_passes"] == 256
    assert protocol["action_space"]["max_action_combos"] == 1024
    assert protocol["training"]["split_seed"] == 5_000_000
    assert protocol["training"]["heldout_fraction"] == 0.2
    assert protocol["runtime"]["absolute_owner_max_neural_passes"] == 256
    assert protocol["runtime"]["automatic_budget_escalation"] is False
    assert protocol["runtime"]["revision_197_32_pass_draft_superseded"] is True
    assert protocol["activation"]["promotion_receipt"] is None

    assert specialists["active"] is False
    assert specialists["split_seed"] == 5_000_000
    assert specialists["heldout_fraction"] == 0.2
    assert specialists["selector_eligible"] is False
    assert specialists["serving_eligible"] is False
    assert specialists["action_authority_enabled"] is False
    assert specialists["revision_197_32_pass_draft_superseded"] is True
    assert specialists["max_action_combos"] == 1024
    assert specialists["r175_restart_allowed"] is False
    assert specialists["iter_00021_allowed"] is False


def test_r197_goal_and_shadow_unit_preserve_terminal_r175() -> None:
    goal = (ROOT / "GOAL.md").read_text(encoding="utf-8")
    unit = (
        ROOT / "deploy/systemd/pokebot-alakazam-rtp-r197-shadow.service"
    ).read_text(encoding="utf-8")

    assert "| 198 |" in goal
    assert "hard neural-pass ceiling to exactly 256" in goal
    assert "1,024 complete ordered legal-action combinations" in goal
    assert "Do not restart" in goal
    assert "pokebot-final-format-alakazam-rtp-r175-rl.service" not in unit
    assert "Kaggle" not in unit
    assert "Restart=no" in unit
