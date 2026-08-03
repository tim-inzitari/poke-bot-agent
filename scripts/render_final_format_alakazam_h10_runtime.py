#!/usr/bin/env python3
"""Render the isolated, receipt-bound Alakazam H10 training registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping


AUTH_SCHEMA = "poke_bot.matchup_adapter_specialist_bootstrap_authorization/v1"
REGISTRY_SCHEMA = "poke_bot.specialist_runtime_registry/v1"
RENDER_SCHEMA = "poke_bot.final_format_alakazam_h10_runtime_render/v1"
LEARNED_HEADS = [
    "value",
    "archetype",
    "opponent_hand",
    "opponent_remainder",
    "lethal_threat",
    "prize_race",
    "action_q",
    "action_type",
    "action_target",
    "action_resource",
    "action_utility",
    "tactical_outcomes",
    "opponent_response",
    "resource_forecast",
    "game_phase",
    "outcome_distribution",
    "remaining_turns",
    "setup_board_outcome",
    "combo_state",
]
TARGET_COVERAGE = [
    "temporal_action_rows",
    "opponent_hand_rows",
    "opponent_remainder_rows",
    "opponent_private_prize_rows",
    "lethal_threat_rows",
    "prize_race_rows",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    body = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != body:
            raise RuntimeError(f"refusing to replace different immutable artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _require(path: Path, expected: str | None = None) -> str:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"required H10 runtime input is missing: {path}")
    digest = _sha256(path)
    if expected is not None and digest != expected:
        raise RuntimeError(
            f"H10 runtime input digest changed: {path} expected={expected} actual={digest}"
        )
    return digest


def _replace_option(values: list[str], option: str, value: str) -> list[str]:
    output: list[str] = []
    index = 0
    while index < len(values):
        if values[index] == option:
            if index + 1 >= len(values):
                raise RuntimeError(f"trainer option lacks a value: {option}")
            index += 2
            continue
        output.append(values[index])
        index += 1
    output.extend([option, value])
    return output


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if _sha256(target) != _sha256(source):
            raise RuntimeError(f"immutable source-family checkpoint changed: {target}")
        return
    try:
        os.link(source, target)
    except OSError:
        temporary = target.with_name(f".{target.name}.partial.{os.getpid()}")
        shutil.copyfile(source, temporary)
        os.link(temporary, target)
        temporary.unlink(missing_ok=True)


def render(args: argparse.Namespace) -> dict[str, Any]:
    runtime_root = args.runtime_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    ordinary = args.ordinary_checkpoint.expanduser().resolve()
    h10_handoff = args.h10_handoff.expanduser().resolve()
    expert = args.expert_manifest.expanduser().resolve()
    tree = args.matchup_tree.expanduser().resolve()
    guide = (runtime_root / "config/deck_guides/alakazam-final-refresh.yaml").resolve()
    checkpoint_digest = _require(checkpoint, args.expected_checkpoint_sha256)
    ordinary_digest = _require(ordinary, args.expected_ordinary_sha256)
    handoff_digest = _require(h10_handoff, args.expected_handoff_sha256)
    expert_digest = _require(expert)
    tree_digest = _require(tree)
    guide_digest = _require(guide)

    handoff = json.loads(h10_handoff.read_text(encoding="utf-8"))
    if (
        handoff.get("schema")
        != "poke_bot.final_format_alakazam_h10_router_v6_handoff/v1"
        or handoff.get("status") != "h10_router_v6_migrated_training_pending"
        or handoff.get("child_checkpoint_sha256") != checkpoint_digest
        or handoff.get("learned_head_count") != 19
        or handoff.get("learned_route_count") != 19
        or handoff.get("adapter_slot_capacity") != 64
        or handoff.get("active_logical_matchup_routes") != 20
        or handoff.get("matchup_adapters_runtime_enabled") is not False
        or handoff.get("training_seat_split") != {"first": 0.5, "second": 0.5}
        or handoff.get("package_preference") != "first_if_allowed"
        or handoff.get("second_focused_arm_allowed") is not False
        or handoff.get("selector_eligible") is not False
        or handoff.get("kaggle_eligible") is not False
    ):
        raise RuntimeError("Router Format 6 H10 handoff is not launch-authoritative")

    authorization_root = args.authorization_root.expanduser().resolve()
    source_family = authorization_root / "ordinary_source"
    source_model = source_family / "model.pt"
    _link_or_copy(ordinary, source_model)
    source_manifest = source_family / "manifest.json"
    _write_once(
        source_manifest,
        {
            "schema": "poke_bot.final_format_alakazam_ordinary_training_manifest/v1",
            "checkpoint_digest": ordinary_digest,
            "model_path": str(source_model),
            "provenance": {
                "acting_seat_archetype": "alakazam",
                "epochs_max": 25,
                "trained_target_coverage": TARGET_COVERAGE,
                "ordinary_same_archetype_refresh": True,
            },
            "evidence": {
                "epochs_completed": 25,
                "ordinary_checkpoint_sha256": ordinary_digest,
                "ordinary_parent_preserved": True,
            },
        },
    )
    source_manifest_digest = _sha256(source_manifest)
    derivative_manifest = authorization_root / "router_v6_derivative_manifest.json"
    _write_once(
        derivative_manifest,
        {
            "schema": "poke_bot.final_format_alakazam_h10_router_v6_training_manifest/v1",
            "checkpoint_digest": checkpoint_digest,
            "model_path": str(checkpoint),
            "provenance": {
                "kind": "matchup_adapter_v6_runtime_derivative",
                "source_family": str(source_family),
                "source_family_immutable": True,
                "source_family_manifest_sha256": source_manifest_digest,
                "source_checkpoint_digest": ordinary_digest,
                "h10_router_v6_handoff": str(h10_handoff),
                "h10_router_v6_handoff_sha256": handoff_digest,
            },
            "evidence": {
                "training_evidence_inherited_from_source": True,
                "h10_new_capacity_zero_safe_pending_rl_training": True,
                "matchup_adapter_bank_dormant_at_launch": True,
            },
        },
    )
    derivative_manifest_digest = _sha256(derivative_manifest)
    authorization = authorization_root / "matchup_adapter_bootstrap_authorization.json"
    _write_once(
        authorization,
        {
            "schema": AUTH_SCHEMA,
            "specialist_id": "alakazam",
            "runtime_enabled": False,
            "parent_untouched": True,
            "optimizer_scope": "matchup_adapter_bank_only",
            "first_eligible_iteration": 0,
            "completed_iteration": -1,
            "parent_checkpoint": str(checkpoint),
            "parent_checkpoint_digest": checkpoint_digest,
            "protected_manifest": str(derivative_manifest),
            "protected_manifest_digest": derivative_manifest_digest,
            "required_target_coverage": TARGET_COVERAGE,
        },
    )
    authorization_digest = _sha256(authorization)

    base_path = (runtime_root / "ops/specialist_runtime_registry_v1.json").resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if base.get("schema") != REGISTRY_SCHEMA:
        raise RuntimeError("base specialist runtime registry schema changed")
    slowking = dict(dict(base.get("specialists") or {}).get("slowking") or {})
    if not slowking:
        raise RuntimeError("base registry lacks the current strategic-row scaffold")

    # The base registry predates the final-format gate derivatives.  Do not
    # inherit its historical terminal-gate label: bind both the relative
    # contract path and the exact gate identity from the same immutable file.
    active_gate_path = Path(args.active_gate_contract).expanduser()
    if not active_gate_path.is_absolute():
        active_gate_path = runtime_root / active_gate_path
    active_gate_path = active_gate_path.resolve()
    _require(active_gate_path)
    active_gate = json.loads(active_gate_path.read_text(encoding="utf-8"))
    terminal_active_gate_id = str(active_gate.get("active_gate_id") or "").strip()
    if not terminal_active_gate_id:
        raise RuntimeError("active final-format gate lacks active_gate_id")

    curriculum_root = runtime_root / "state/final_format_alakazam_curriculum_r79"
    role_map = curriculum_root / "alakazam-strategic-head-roles-r104.json"
    curriculum = curriculum_root / "alakazam-strategic-curriculum-r104.json"
    curriculum_receipt = (
        args.curriculum_validation_receipt.expanduser().resolve()
        if args.curriculum_validation_receipt is not None
        else curriculum_root / "alakazam-strategic-curriculum-validation-r104.json"
    )
    role_digest = _require(role_map)
    curriculum_digest = _require(curriculum)
    curriculum_receipt_digest = _require(curriculum_receipt)

    policy = dict(slowking["guide_weight_policy"])
    policy["canonical_learned_decision_sources"] = LEARNED_HEADS
    module_fields = {
        "schedule_module": "poke_bot/pure_rl/deck_guide_schedule.py",
        "evidence_module": "poke_bot/pure_rl/guide_weight_evidence.py",
        "review_request_module": "poke_bot/pure_rl/guide_weight_review.py",
        "boundary_controller": "scripts/apply_future_guide_weight_at_boundary.py",
        "shadow_pair_module": "poke_bot/pure_rl/guide_weight_shadow_pair.py",
        "shadow_pair_runner": "scripts/run_future_guide_weight_shadow_pair.py",
        "shadow_queue_processor": "scripts/process_future_guide_weight_review_queue.py",
    }
    for field, relative in module_fields.items():
        module = (runtime_root / relative).resolve()
        policy[field] = str(module)
        policy[field + "_sha256"] = _require(module).removeprefix("sha256:")

    row = {
        "status": "ready",
        "reason": None,
        "run_name": args.run_name,
        "log": str(args.log.expanduser().resolve()),
        "measurement_decks": "alakazam",
        "initial_checkpoint": str(checkpoint),
        "initial_checkpoint_sha256": checkpoint_digest.removeprefix("sha256:"),
        "expert_manifest": str(expert),
        "expert_manifest_sha256": expert_digest.removeprefix("sha256:"),
        "expert_minimum_decisions": 20000,
        "expert_required_target_coverage": TARGET_COVERAGE,
        "matchup_runtime_tree": str(tree),
        "matchup_runtime_tree_sha256": tree_digest.removeprefix("sha256:"),
        "matchup_adapter_authorization": str(authorization),
        "matchup_adapter_authorization_sha256": authorization_digest.removeprefix("sha256:"),
        "matchup_adapter_epochs_per_rl_iteration": 1,
        "guide_id": "alakazam",
        "guide_version": "powerful-hand-v1",
        "guide_contract": str(guide),
        "guide_contract_sha256": guide_digest.removeprefix("sha256:"),
        "guide_loss_weight": 0.05,
        "guide_training_mode": "strategic_directional_v2",
        "guide_weight_policy": policy,
        "strategic_curriculum": {
            "schema": "poke_bot.specialist_guide_training_contract/v1",
            "training_mode": "strategic_directional_v2",
            "guide_curriculum_revision": 104,
            "strategic_branch_scope_revision": 56,
            "action_influence_revision": 104,
            "decision_fusion_schema": "poke_bot.causal_decision_fusion/v3",
            "typed_output_centered_routes": True,
            "route_reliability_bounds": [0.25, 4.0],
            "action_type_reliability_cap": 0.25,
            "final_policy_logits_are_guide_targets": False,
            "require_all_registered_learned_sources": True,
            "head_role_map": str(role_map),
            "head_role_map_sha256": role_digest.removeprefix("sha256:"),
            "curriculum_spec": str(curriculum),
            "curriculum_spec_sha256": curriculum_digest.removeprefix("sha256:"),
            "validation_receipt": str(curriculum_receipt),
            "validation_receipt_sha256": curriculum_receipt_digest.removeprefix("sha256:"),
        },
        "decision_fusion": {
            "schema": "poke_bot.causal_decision_fusion/v3",
            "required": True,
            "runtime_enabled": True,
            "typed_output_centered_routes": True,
            "route_reliability_bounds": [0.25, 4.0],
            "action_type_reliability_cap": 0.25,
            "required_heads": LEARNED_HEADS,
        },
        "setup_board_outcome_loss_weight": 0.025,
        "combo_state_loss_weight": 0.0,
        "minimum_terminal_iteration": args.minimum_terminal_iteration,
        "iteration_ceiling": 20,
        "terminal_gate_marker": "SPECIALIST_GATE_PASSED.alakazam-final-h10-r79",
        "pass_handler": {
            "family": "final-format-alakazam-r79-h10-refresh-v1",
            "display_name": "Final-format Alakazam H10 Refresh Champion",
            "submission_root": "/home/inzi/poke-bot-agent/outputs/submissions/final-format-alakazam-r79-h10-refresh-v1",
            "state": "/home/inzi/poke-bot-agent/outputs/state/final-format-alakazam-r79-h10-passed-gate-handler-v1.json",
            "lock": "/home/inzi/.local/state/pokebot/final-format-alakazam-r79-h10-passed-gate-handler-v1.lock",
            "handoff_service": "pokebot-final-format-alakazam-r79-completion.service",
        },
    }

    registry = dict(base)
    registry["version"] = max(4, int(base.get("version") or 0) + 1)
    registry["runtime_root"] = str(runtime_root)
    registry["specialists"] = {"alakazam": row}
    common_args = [str(value) for value in base.get("common_trainer_args") or []]
    if "--require-exact-training-seat-split" not in common_args:
        common_args.append("--require-exact-training-seat-split")
    common_args = _replace_option(
        common_args, "--setup-board-outcome-loss-weight", "0.025"
    )
    common_args = _replace_option(common_args, "--games-per-iter", "16384")
    common_args = _replace_option(common_args, "--train-epochs", "1")
    common_args = _replace_option(common_args, "--train-games-per-batch", "240")
    # Gate arguments are appended canonically by launch_active_specialist.
    # Keeping them out of the common vector prevents a stale registry from
    # overriding the receipt-bound row/gate pair during a managed resume.
    common_args = _replace_option(
        common_args,
        "--train-max-decisions-per-batch",
        str(args.train_decisions_cap),
    )
    common_args = _replace_option(
        common_args,
        "--train-warmup-max-decisions-per-batch",
        str(args.train_warmup_decisions_cap),
    )
    common_args = _replace_option(
        common_args,
        "--dormant-matchup-adapter-max-decisions-per-batch",
        str(args.dormant_matchup_adapter_decisions_cap),
    )
    common_args = _replace_option(
        common_args,
        "--boundary-design-migration-reason",
        str(args.boundary_design_migration_reason),
    )
    common_args = _replace_option(
        common_args,
        "--remote-worker-endpoints",
        "192.168.1.143:8765,192.168.1.158:8766",
    )
    registry["common_trainer_args"] = common_args
    registry["minimum_terminal_iteration"] = args.minimum_terminal_iteration
    registry["iteration_ceiling"] = 20
    registry["active_gate_contract"] = args.active_gate_contract
    registry["terminal_active_gate_id"] = terminal_active_gate_id
    handler = dict(base["pass_handler"])
    handler["training_service"] = "pokebot-final-format-alakazam-r79-h10.service"
    # Revision 103 makes exact iteration 20 terminal. A measured pass remains
    # preferred; otherwise preserve the failed gate and use the explicit owner
    # ceiling authority without relabeling it as a pass.
    handler["ceiling_behavior"] = "freeze_submit_and_continue_without_false_pass"
    profiles = dict(handler.get("specialist_submission_profiles") or {})
    profiles["alakazam"] = {
        "owner_decision_revision": 104,
        "submission_count": 1,
        "turn_order_preferences": ["first_if_allowed"],
        "same_frozen_checkpoint_required": True,
        "same_exact_60_card_deck_required": True,
    }
    handler["specialist_submission_profiles"] = profiles
    registry["pass_handler"] = handler
    registry["isolated_refresh_contract"] = {
        "schema": "poke_bot.final_format_alakazam_h10_isolated_runtime/v1",
        "owner_decision_revision": 82,
        "allocator_recovery_revision": 105,
        "production_selector_write_authority": False,
        "historical_alakazam_replacement_allowed": False,
        "training_seat_split": {"first": 0.5, "second": 0.5},
        "exact_assigned_actual_consumed_receipts_required": True,
        "package_preference": "first_if_allowed",
        "second_focused_arm_allowed": False,
        "games_per_iteration": 16384,
        "self_play_fraction": 0.125,
        "self_play_games_per_iteration": 2048,
        "public_opponent_games_per_iteration": 14336,
        "learner_epochs_per_iteration": 1,
        "learner_batch": {
            "games_per_batch_cap": 240,
            "decisions_per_batch_cap": args.train_decisions_cap,
            "warmup_decisions_per_batch_cap": args.train_warmup_decisions_cap,
            "agreement_decisions_per_batch_cap": 6144,
            "agreement_forward": "policy_only",
            "dormant_matchup_adapter_decisions_per_batch_cap": args.dormant_matchup_adapter_decisions_cap,
            "dormant_matchup_adapter_oom_splitting": True,
        },
        "awr_baseline_preparation": {
            "value_only": True,
            "cpu_prefetch_batches": 1,
            "packed_temporal": "shadow_pending_exact_blackwell_parity",
        },
        "maximum_iterations": 21,
        "maximum_training_games": 344064,
        "minimum_terminal_iteration": args.minimum_terminal_iteration,
        "premium_skill_weighted_win_rate": 0.75,
        "premium_skill_weighted_confidence_lower": 0.60,
        "official_control_win_rate": 0.60,
        "kaggle_rating_simulation_projected_lower_bound": args.kaggle_rating_lower_bound,
        "strength_gate_and_rating_simulation_are_independent": True,
        "blackwell_workers": 96,
        "blackwell_games_in_flight": 96,
        "blackwell_leaf_replicas": {"gpu0": 4, "gpu1": 12},
        "elmo_workers": 36,
        "bert_workers": 16,
        "checkpoint_sha256": checkpoint_digest,
        "h10_router_v6_handoff_sha256": handoff_digest,
    }
    _write_once(args.output.expanduser().resolve(), registry)
    registry_digest = _sha256(args.output.expanduser().resolve())
    fusion_authority = {
        "schema": "poke_bot.causal_decision_fusion/v3",
        "runtime_enabled": True,
        "typed_output_centered_routes": True,
        "route_reliability_bounds": [0.25, 4.0],
        "action_type_reliability_cap": 0.25,
        "required_heads": LEARNED_HEADS,
    }
    activation = {
        "schema": "poke_bot.specialist_rl_activation/v2",
        "status": "ready",
        "specialist_id": "alakazam",
        "activation_scope": "isolated_final_format_refresh_training",
        "production_selector_write_authority": False,
        "identity": {
            "next_specialist_bootstrap": {
                "specialist_id": "alakazam",
                "checkpoint": str(checkpoint),
                "checkpoint_digest": checkpoint_digest,
                "decision_fusion": fusion_authority,
            },
            "runtime_registration": {
                "specialist_id": "alakazam",
                "runtime_registry": str(args.output.expanduser().resolve()),
                "runtime_registry_sha256": registry_digest,
                "runtime_row": {
                    "initial_checkpoint": str(checkpoint),
                    "initial_checkpoint_sha256": checkpoint_digest.removeprefix(
                        "sha256:"
                    ),
                    "decision_fusion": {
                        **fusion_authority,
                        "required": True,
                    },
                },
            },
        },
        "training_seat_split": {"first": 0.5, "second": 0.5},
        "package_preference": "first_if_allowed",
        "games_per_iteration": 16384,
        "self_play_fraction": 0.125,
        "self_play_games_per_iteration": 2048,
        "public_opponent_games_per_iteration": 14336,
        "maximum_iterations": 21,
        "maximum_training_games": 344064,
        "minimum_terminal_iteration": args.minimum_terminal_iteration,
        "active_gate_contract": str(active_gate_path),
        "terminal_active_gate_id": terminal_active_gate_id,
        "premium_skill_weighted_win_rate": 0.75,
        "premium_skill_weighted_confidence_lower": 0.60,
        "official_control_win_rate": 0.60,
        "kaggle_rating_simulation_projected_lower_bound": args.kaggle_rating_lower_bound,
        "h10_router_v6_handoff_sha256": handoff_digest,
    }
    _write_once(args.activation_receipt.expanduser().resolve(), activation)
    activation_digest = _sha256(args.activation_receipt.expanduser().resolve())
    receipt = {
        "schema": RENDER_SCHEMA,
        "status": "isolated_h10_runtime_rendered",
        "specialist_id": "alakazam",
        "registry": str(args.output.expanduser().resolve()),
        "registry_sha256": registry_digest,
        "checkpoint_sha256": checkpoint_digest,
        "h10_router_v6_handoff_sha256": handoff_digest,
        "matchup_adapter_authorization_sha256": authorization_digest,
        "specialist_rl_activation_receipt": str(
            args.activation_receipt.expanduser().resolve()
        ),
        "specialist_rl_activation_receipt_sha256": activation_digest,
        "learned_heads": LEARNED_HEADS,
        "guide_runtime_route_count": 0,
        "guide_loss_weight": 0.05,
        "setup_board_outcome_loss_weight": 0.025,
        "combo_state_loss_weight": 0.0,
        "training_seat_split": {"first": 0.5, "second": 0.5},
        "package_preference": "first_if_allowed",
        "production_selector_write_authority": False,
        "canonical_payload_sha256": _canonical_sha256(registry),
    }
    _write_once(args.receipt.expanduser().resolve(), receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--ordinary-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-ordinary-sha256", required=True)
    parser.add_argument("--h10-handoff", type=Path, required=True)
    parser.add_argument("--expected-handoff-sha256", required=True)
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--matchup-tree", type=Path, required=True)
    parser.add_argument("--authorization-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--activation-receipt", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--minimum-terminal-iteration", type=int, default=20)
    parser.add_argument(
        "--active-gate-contract",
        default="ops/final_format_alakazam_gate_r100_v1.json",
    )
    parser.add_argument("--kaggle-rating-lower-bound", type=float, default=1150.0)
    parser.add_argument("--train-decisions-cap", type=int, default=3072)
    parser.add_argument("--train-warmup-decisions-cap", type=int, default=3072)
    parser.add_argument(
        "--dormant-matchup-adapter-decisions-cap", type=int, default=3072
    )
    parser.add_argument(
        "--boundary-design-migration-reason",
        default="receipt_backed_completed_collection_resume_v1",
    )
    parser.add_argument("--curriculum-validation-receipt", type=Path)
    args = parser.parse_args()
    if args.minimum_terminal_iteration != 20:
        parser.error("--minimum-terminal-iteration must be exactly 20")
    if args.kaggle_rating_lower_bound <= 0:
        parser.error("--kaggle-rating-lower-bound must be positive")
    if not args.boundary_design_migration_reason.strip():
        parser.error("--boundary-design-migration-reason must be non-empty")
    for option, value in (
        ("--train-decisions-cap", args.train_decisions_cap),
        ("--train-warmup-decisions-cap", args.train_warmup_decisions_cap),
        (
            "--dormant-matchup-adapter-decisions-cap",
            args.dormant_matchup_adapter_decisions_cap,
        ),
    ):
        if value <= 0:
            parser.error(f"{option} must be positive")
    print(json.dumps(render(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
