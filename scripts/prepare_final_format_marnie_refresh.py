#!/usr/bin/env python3
"""Open the Marnie refresh pre-stage only after Alakazam is registered."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    body = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != body:
            raise RuntimeError(f"Marnie preparation receipt changed: {path}")
        return
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink(missing_ok=True)


def _static(args: argparse.Namespace) -> dict[str, str]:
    paths = {
        "state": args.specialist_state.expanduser().resolve(),
        "guide": args.guide.expanduser().resolve(),
        "guide_ready": args.guide_ready.expanduser().resolve(),
        "head_role_map": args.head_role_map.expanduser().resolve(),
        "curriculum_spec": args.curriculum_spec.expanduser().resolve(),
        "curriculum_validation": args.curriculum_validation.expanduser().resolve(),
        "expert": args.expert_manifest.expanduser().resolve(),
        "original": args.original_checkpoint.expanduser().resolve(),
        "refresh_registry": args.refresh_registry.expanduser().resolve(),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if args.check:
        missing = [name for name in missing if name != "refresh_registry"]
    if missing:
        raise RuntimeError("Marnie refresh input is missing: " + ",".join(missing))
    guide = yaml.safe_load(paths["guide"].read_text(encoding="utf-8"))
    ready = json.loads(paths["guide_ready"].read_text(encoding="utf-8"))
    roles = json.loads(paths["head_role_map"].read_text(encoding="utf-8"))
    curriculum = json.loads(paths["curriculum_spec"].read_text(encoding="utf-8"))
    validation = json.loads(
        paths["curriculum_validation"].read_text(encoding="utf-8")
    )
    target = dict(guide.get("policy_target") or {})
    learned_sources = list(roles.get("canonical_learned_decision_sources") or [])
    head_rows = dict(roles.get("heads") or {})
    every_strategic_head_is_causal = bool(learned_sources) and all(
        dict(head_rows.get(head_id) or {}).get("causal_input")
        == "board_state_and_legal_option"
        and dict(head_rows.get(head_id) or {}).get("enters_decision_fusion") is True
        and dict(head_rows.get(head_id) or {}).get("action_influence")
        == "bounded_option_conditioned_route"
        for head_id in learned_sources
    )
    if (
        guide.get("schema_version") != "poke_bot.current_deck_guide/v1"
        or guide.get("specialist_id") != "marnie-s-grimmsnarl-ex"
        or target.get("training_mode") != "strategic_directional_v2"
        or target.get("target_logits") != "none"
        or target.get("direct_policy_cross_entropy_allowed") is not False
        or target.get("guide_preference_index_may_affect_loss_or_gradient")
        is not True
        or target.get("final_policy_logits_are_guide_targets") is not False
        or target.get("serving_authority") is not False
        or target.get("override_authoritative_policy") is not False
        or ready.get("schema") != "poke_bot.current_deck_guide_corpus_ready/v1"
        or ready.get("status") not in {"ready", "validated"}
        or ready.get("specialist_id") != "marnie-s-grimmsnarl-ex"
        or ready.get("guide_version") != guide.get("guide_version")
        or int(ready.get("guide_rows") or 0) <= 0
        or roles.get("specialist_id") != "marnie-s-grimmsnarl-ex"
        or curriculum.get("specialist_id") != "marnie-s-grimmsnarl-ex"
        or validation.get("specialist_id") != "marnie-s-grimmsnarl-ex"
        or roles.get("training_mode") != "strategic_directional_v2"
        or curriculum.get("training_mode") != "strategic_directional_v2"
        or validation.get("training_mode") != "strategic_directional_v2"
        or validation.get("status") != "validated"
        or roles.get("guide_contract_sha256") != _sha256(paths["guide"])
        or curriculum.get("guide_contract_sha256") != _sha256(paths["guide"])
        or validation.get("guide_contract_sha256") != _sha256(paths["guide"])
        or validation.get("guide_ready_receipt_sha256")
        != _sha256(paths["guide_ready"])
        or int(validation.get("measurements", {}).get("guide_labeled_rows") or 0)
        != int(ready.get("guide_rows") or 0)
        or len(learned_sources) != 19
        or "combo_state" not in learned_sources
        or set(head_rows) != set(learned_sources)
        or not every_strategic_head_is_causal
    ):
        raise RuntimeError("Marnie final-format guide contract is not strategic-ready")
    return {name: _sha256(path) for name, path in paths.items() if path.is_file()}


def _execution_plan() -> list[dict[str, Any]]:
    return [
        {
            "ordinal": 1,
            "stage": "attempt_normal_cumulative_core_refresh",
            "core_generation_is_resolved_at_execution": True,
            "replace_historical_alakazam_teacher_with_refresh": True,
            "on_rejection": "preserve_candidate_and_use_latest_checksum_accepted_core",
        },
        {
            "ordinal": 2,
            "stage": "hot_start_and_expand_to_final_submission_h10",
            "resolved_core": "latest_checksum_accepted_after_stage_1",
            "training_before_h10_migration_allowed": False,
            "bootstrap_runs_entirely_after_h10_migration": True,
            "epochs": 25,
            "capacity_profile": "H10-I/v1",
            "spatial_layers": 7,
            "temporal_layers": 3,
            "option_decoder_layers": 7,
            "feed_forward_width": 2496,
            "strategic_head_residual_width": 512,
            "final_learned_head_count": 19,
            "final_distinct_action_route_count": 19,
            "all_final_heads_option_conditioned_and_action_influencing": True,
            "decision_fusion_schema": "poke_bot.causal_decision_fusion/v3",
            "typed_output_centered_routes": True,
            "positive_bounded_route_reliability": [0.25, 4.0],
            "action_type_reliability_cap_until_versioned_labels": 0.25,
            "guide_training_mode": "strategic_directional_v2",
            "guide_weight": 0.05,
            "guide_pairwise_route_heads": [
                "action_q",
                "action_resource",
                "action_utility",
                "setup_board_outcome",
                "combo_state",
            ],
            "guide_runtime_route_count": 0,
        },
        {
            "ordinal": 3,
            "stage": "migrate_router_and_launch_managed_rl",
            "router_format": 6,
            "matchup_adapters_required": True,
            "selector_authority": False,
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--check-receipt", type=Path)
    parser.add_argument("--specialist-state", type=Path, required=True)
    parser.add_argument("--guide", type=Path, required=True)
    parser.add_argument("--guide-ready", type=Path, required=True)
    parser.add_argument("--head-role-map", type=Path, required=True)
    parser.add_argument("--curriculum-spec", type=Path, required=True)
    parser.add_argument("--curriculum-validation", type=Path, required=True)
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--refresh-registry", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    static = _static(args)
    if args.check:
        if args.check_receipt is not None:
            _write_once(
                args.check_receipt.expanduser().resolve(),
                {
                    "schema": "poke_bot.final_format_marnie_refresh_preparation/v1",
                    "status": "static_inputs_validated_waiting_for_alakazam_iter20_boundary",
                    "owner_decision_revision": 104,
                    "specialist_id": "marnie-s-grimmsnarl-ex",
                    "boundary_iteration": 20,
                    "input_sha256": static,
                    "execution_plan": _execution_plan(),
                    "training_authority": False,
                    "selector_authority": False,
                },
            )
        print("FINAL_FORMAT_MARNIE_PRESTAGE_OK " + json.dumps(static, sort_keys=True))
        return 0
    state = yaml.safe_load(args.specialist_state.read_text(encoding="utf-8"))
    phase = dict(state.get("post_fleet_refresh") or {})
    if (
        phase.get("completed_refresh_specialist_ids") != ["alakazam"]
        or phase.get("pending_specialist_ids") != ["marnie-s-grimmsnarl-ex"]
        or phase.get("active_refresh_specialist_id") != "marnie-s-grimmsnarl-ex"
        or phase.get("next_refresh_specialist_id") != "marnie-s-grimmsnarl-ex"
    ):
        raise RuntimeError("Marnie refresh boundary is not authorized")
    registry = json.loads(args.refresh_registry.read_text(encoding="utf-8"))
    refreshes = list(registry.get("refreshes") or [])
    if (
        registry.get("schema") != "poke_bot.post_fleet_refresh_registry/v1"
        or len(refreshes) != 1
        or dict(refreshes[0]).get("specialist_id") != "alakazam"
    ):
        raise RuntimeError("Alakazam refresh registration is not authoritative")
    receipt = {
        "schema": "poke_bot.final_format_marnie_refresh_prestage/v1",
        "status": "authorized_preparation_started",
        "owner_decision_revision": 104,
        "specialist_id": "marnie-s-grimmsnarl-ex",
        "predecessor_refresh": "alakazam",
        "predecessor_registry_sha256": static["refresh_registry"],
        "guide_sha256": static["guide"],
        "guide_ready_receipt_sha256": static["guide_ready"],
        "head_role_map_sha256": static["head_role_map"],
        "curriculum_spec_sha256": static["curriculum_spec"],
        "curriculum_validation_sha256": static["curriculum_validation"],
        "guide_training_mode": "strategic_directional_v2",
        "guide_runtime_route_count": 0,
        "final_capacity_profile": "H10-I/v1",
        "final_decision_fusion_schema": "poke_bot.causal_decision_fusion/v3",
        "expert_manifest_sha256": static["expert"],
        "original_checkpoint_sha256": static["original"],
        "original_checkpoint_immutable": True,
        "boundary_iteration": 20,
        "predecessor_completion_required": {
            "frozen": True,
            "registered": True,
            "allowed_completion_authorities": [
                "measured_gate_pass",
                "explicit_owner_ceiling_acceptance",
            ],
            "ceiling_acceptance_may_be_labeled_measured_pass": False,
        },
        "execution_plan": _execution_plan(),
        "training_authority": False,
        "selector_authority": False,
        "next_boundary": "execute_core_refresh_then_25_epoch_bootstrap_h10_migration_and_managed_launch",
    }
    _write_once(args.receipt.expanduser().resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
