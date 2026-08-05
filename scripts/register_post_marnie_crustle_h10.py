#!/usr/bin/env python3
"""Register and launch the Revision-113 Crustle H10 specialist runtime."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from poke_bot.pure_rl.model_registry import sha256
from scripts.register_next_specialist_runtime import register


SPECIALIST_ID = "crustle"
MARNIE_COMPLETION_SCHEMA = "poke_bot.post_fleet_specialist_refresh_completion/v1"
MARNIE_ONLY_TOP_LEVEL_KEYS = {
    "archetype_loss_contract",
    "deck_family_distribution",
    "family_activation_authority",
    "family_manifest",
    "marnie_family_guide_shadow_runtime",
    "marnie_guide_retirement",
    "replay_list_sampler",
    "selected_loss_vector",
    "variant_provenance",
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_source_registry(
    completion_path: Path,
    opponent_stage_path: Path,
) -> tuple[dict[str, Any], Path]:
    """Build a Crustle-safe derivative of the exact handoff-time registry."""

    completion_path = completion_path.resolve()
    completion = read_json(completion_path)
    opponent_stage_path = opponent_stage_path.resolve()
    opponent_stage = read_json(opponent_stage_path)
    source_path = Path(str(completion.get("runtime_registry") or "")).resolve()
    if (
        completion.get("schema") != MARNIE_COMPLETION_SCHEMA
        or completion.get("specialist_id") != "marnie-s-grimmsnarl-ex"
        or completion.get("frozen") is not True
        or completion.get("registered") is not True
        or not source_path.is_file()
        or sha256(source_path) != completion.get("runtime_registry_sha256")
    ):
        raise RuntimeError("Marnie completion runtime registry binding is invalid")
    source = read_json(source_path)
    marnie = dict(
        (source.get("specialists") or {}).get("marnie-s-grimmsnarl-ex") or {}
    )
    if (
        source.get("schema") != "poke_bot.specialist_runtime_registry/v1"
        or marnie.get("status") != "ready"
        or float(marnie.get("guide_loss_weight") or 0.0) != 0.0
        or dict(marnie.get("decision_fusion") or {}).get("schema")
        != "poke_bot.causal_decision_fusion/v3"
        or dict(marnie.get("decision_fusion") or {}).get("runtime_enabled")
        is not True
    ):
        raise RuntimeError("then-current Marnie H10 registry is ineligible")
    frozen_registry = Path(str(opponent_stage.get("frozen_registry") or "")).resolve()
    active_gate = Path(str(opponent_stage.get("active_gate") or "")).resolve()
    if (
        opponent_stage.get("schema")
        != "poke_bot.crustle_marnie_expert_practice_stage/v1"
        or opponent_stage.get("status")
        != "ready_for_post_marnie_crustle_bootstrap"
        or opponent_stage.get("marnie_completion_sha256")
        != sha256(completion_path)
        or (
            opponent_stage.get("checkpoint_digest")
            != completion.get("refresh_checkpoint_checksum")
            and opponent_stage.get("owner_decision_revision") != 163
        )
        or not frozen_registry.is_file()
        or sha256(frozen_registry) != opponent_stage.get("frozen_registry_sha256")
        or not active_gate.is_file()
        or sha256(active_gate) != opponent_stage.get("active_gate_sha256")
    ):
        raise RuntimeError("Marnie H10 Crustle opponent stage is invalid")
    for key in MARNIE_ONLY_TOP_LEVEL_KEYS:
        source.pop(key, None)
    args_list = list(source.get("common_trainer_args") or [])
    if "--boundary-design-migration-reason" in args_list:
        index = args_list.index("--boundary-design-migration-reason")
        if index + 1 >= len(args_list):
            raise RuntimeError("source registry migration reason is malformed")
        args_list[index + 1] = "post_marnie_crustle_h10_registration_r143"
    source["common_trainer_args"] = args_list
    source["frozen_specialist_registry"] = str(
        opponent_stage["frozen_registry_relative"]
    )
    source["active_gate_contract"] = str(opponent_stage["active_gate_relative"])
    source["terminal_active_gate_id"] = str(opponent_stage["active_gate_id"])
    source["post_marnie_runtime_rebind"] = {
        "schema": "poke_bot.post_marnie_crustle_runtime_rebind/v1",
        "source_registry": str(source_path),
        "source_registry_sha256": sha256(source_path),
        "completion_receipt": str(completion_path),
        "completion_receipt_sha256": sha256(completion_path),
        "marnie_only_runtime_fields_removed": sorted(MARNIE_ONLY_TOP_LEVEL_KEYS),
        "marnie_expert_practice_stage": str(opponent_stage_path),
        "marnie_expert_practice_stage_sha256": sha256(opponent_stage_path),
    }
    return source, source_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-family", type=Path, required=True)
    parser.add_argument("--bootstrap-ready", type=Path, required=True)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--runtime-tree", type=Path, required=True)
    parser.add_argument("--marnie-completion", type=Path, required=True)
    parser.add_argument("--marnie-opponent-stage", type=Path, required=True)
    parser.add_argument("--runtime-registry", type=Path, required=True)
    parser.add_argument("--source-selector", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--guide", type=Path, required=True)
    parser.add_argument("--curriculum-root", type=Path, required=True)
    args = parser.parse_args()

    ready = read_json(args.bootstrap_ready.resolve())
    completion = read_json(args.marnie_completion.resolve())
    opponent_stage = read_json(args.marnie_opponent_stage.resolve())
    anchor = dict(ready.get("marnie_expert_practice_anchor") or {})
    if (
        ready.get("schema")
        != "poke_bot.specialist_expert_bootstrap_ready/v1"
        or ready.get("acting_seat_archetype") != SPECIALIST_ID
        or str(ready.get("family") or "")
        != args.bootstrap_family.resolve().name
        or int(ready.get("epochs_completed") or 0) != 35
        or int(ready.get("epochs_max") or 0) != 35
        or dict(
            (ready.get("current_deck_guide") or {}).get(
                "owner_epoch_schedule"
            )
            or {}
        )
        != {
            "schema": "poke_bot.crustle_guide_all_epochs/v1",
            "owner_decision_revision": 163,
            "guide_active_epochs": [1, 35],
            "guide_free_expert_refresh_epochs": [],
            "guide_free_expert_refresh_epoch_count": 0,
            "final_selection_eligible_epochs": [1, 35],
            "expert_pilot_importance_epochs": [1, 35],
        }
        or anchor.get("schema")
        != "poke_bot.crustle_marnie_expert_practice_anchor/v1"
        or anchor.get("owner_decision_revision") != 163
        or not isinstance(ready.get("expert_pilot_importance"), dict)
        or int(
            dict(ready.get("expert_pilot_importance") or {}).get(
                "matched_top_100_train_games", 0
            )
        )
        <= 0
        or anchor.get("checkpoint_digest")
        != opponent_stage.get("checkpoint_digest")
        or anchor.get("completion_receipt_sha256")
        != sha256(args.marnie_completion.resolve())
        or anchor.get("actions_relabelled_as_crustle_expert_targets")
        is not False
    ):
        raise RuntimeError("Crustle H10 bootstrap is not complete")
    source, source_path = prepare_source_registry(
        args.marnie_completion,
        args.marnie_opponent_stage,
    )
    source["owner_decision_revision"] = 143
    source["minimum_terminal_iteration"] = 5
    source["iteration_ceiling"] = 15
    source["isolated_refresh_contract"] = {
        "schema": "poke_bot.post_marnie_crustle_h10_isolated_runtime/v1",
        "specialist_id": SPECIALIST_ID,
        "games_per_iteration": 8192,
        "learner_epochs_per_iteration": 5,
        "minimum_terminal_iteration": 5,
        "maximum_iterations": 16,
        "maximum_training_games": 131072,
        "production_selector_write_authority": True,
    }
    atomic_json(args.runtime_registry.resolve(), source)
    atomic_text(
        args.selector.resolve(),
        args.source_selector.resolve().read_text(encoding="utf-8"),
    )

    root = args.curriculum_root.resolve()
    role = root / "crustle-strategic-head-roles-r104.json"
    spec = root / "crustle-strategic-curriculum-r104.json"
    validation = root / "crustle-strategic-curriculum-validation-r104.json"
    for path in (role, spec, validation):
        if not path.is_file():
            raise RuntimeError(f"Crustle curriculum artifact is missing: {path}")
    result = register(
        specialist_id=SPECIALIST_ID,
        family=args.bootstrap_family.resolve(),
        expert=args.expert.resolve(),
        runtime_tree=args.runtime_tree.resolve(),
        runtime_registry=args.runtime_registry.resolve(),
        selector_env=args.selector.resolve(),
        state_root=args.state_root.resolve(),
        run_name="final_format_crustle_r113_h10_i_v6_8k",
        handoff_service="pokebot-final-format-crustle-r113-completion.service",
        minimum_decisions=100_000,
        guide_id=SPECIALIST_ID,
        guide_loss_weight=0.05,
        guide_contract=args.guide.resolve(),
        guide_contract_sha256=sha256(args.guide.resolve()),
        guide_version="crustle-north-star-v2",
        guide_training_mode="strategic_directional_v2",
        strategic_curriculum_spec=spec,
        strategic_curriculum_spec_sha256=sha256(spec),
        strategic_head_role_map=role,
        strategic_head_role_map_sha256=sha256(role),
        strategic_validation_receipt=validation,
        strategic_validation_receipt_sha256=sha256(validation),
        authorization_name="crustle-h10-matchup-adapter-bootstrap-r113.json",
        replace_unpassed=True,
        guide_retired_after_bootstrap=False,
        expert_practice_anchor=anchor,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
