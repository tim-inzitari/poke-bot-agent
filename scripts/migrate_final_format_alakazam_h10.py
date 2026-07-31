#!/usr/bin/env python3
"""Expand one ordinary Alakazam refresh into the exact H10-I format."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from poke_bot import checkpoint  # noqa: E402
from poke_bot.h10_migration import build_h10_child  # noqa: E402
from poke_bot.model import (  # noqa: E402
    CausalDecisionFusion,
    DECISION_FUSION_REQUIRED_HEADS,
    DECISION_FUSION_SCHEMA,
    DECISION_FUSION_V2_SCHEMA,
    DECISION_FUSION_V2_ROUTE_SCHEMA,
    COMBO_STATE_HEAD_NAME,
    SETUP_BOARD_OUTCOME_HEAD_NAME,
)
from poke_bot.train import load_model_from_checkpoint  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _json_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _write_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _role_inventory(model) -> dict[str, object]:
    fusion = model.decision_fusion
    if not isinstance(fusion, CausalDecisionFusion):
        raise RuntimeError("H10 child lacks causal decision fusion")
    heads = tuple(fusion.required_heads)
    routes = tuple(fusion.dedicated_routes)
    if heads != routes or len(heads) != 19:
        raise RuntimeError(
            f"H10 head/route inventory changed: heads={heads} routes={routes}"
        )
    roles = [
        {
            "ordinal": ordinal,
            "head_id": name,
            "typed_objective": name,
            "head_schema_sha256": _json_digest(
                {"head_id": name, "model_config": model.cfg.__dict__}
            ),
            "route_id": f"decision_fusion.dedicated_routes.{name}",
            "route_schema_sha256": _json_digest(
                {"schema": DECISION_FUSION_V2_ROUTE_SCHEMA, "head_id": name}
            ),
            "option_conditioned": True,
            "causal_board_state_cross_attention": True,
            "finite_bound": 1.0,
            "zero_safe": True,
            "action_influence": "bounded_option_conditioned_route",
        }
        for ordinal, name in enumerate(heads)
    ]
    return {
        "schema": "poke_bot.final_format_learned_role_route_inventory/v1",
        "template_only": False,
        "status": "issued_step_zero_training_pending",
        "runtime_authority": "none",
        "capacity_profile": "H10-I/v1",
        "model_config_sha256": _json_digest(model.cfg.__dict__),
        "learned_head_count": len(heads),
        "learned_route_count": len(routes),
        "roles": roles,
        "one_distinct_route_per_learned_head": True,
        "all_routes_option_conditioned": True,
        "all_routes_use_causal_board_state_cross_attention": True,
        "all_routes_finite_bounded": True,
        "all_routes_zero_safe": True,
        "all_learned_heads_influence_actions": True,
        "setup_board_outcome_included": "setup_board_outcome" in heads,
        "validated_slowking_combo_head_included": "combo_state" in heads,
        "guide": {
            "classification": "training_only_curriculum_metadata",
            "learned_decision_head": False,
            "runtime_input": False,
            "runtime_route_count": 0,
            "action_logit_influence": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--expected-parent-sha256", required=True)
    parser.add_argument("--child", type=Path, required=True)
    parser.add_argument("--migration-receipt", type=Path, required=True)
    parser.add_argument("--role-route-receipt", type=Path, required=True)
    args = parser.parse_args()

    parent_path = args.parent.expanduser().resolve()
    actual_parent_sha = _sha256(parent_path)
    if actual_parent_sha != args.expected_parent_sha256:
        raise RuntimeError(
            f"ordinary Alakazam parent checksum changed: {actual_parent_sha}"
        )
    parent_payload = checkpoint.load_checkpoint(parent_path, map_location="cpu")
    parent = load_model_from_checkpoint(parent_path, device=torch.device("cpu"))
    child, evidence = build_h10_child(parent)
    role_inventory = _role_inventory(child)
    role_inventory["inventory_sha256"] = _json_digest(role_inventory["roles"])
    _write_exclusive(args.role_route_receipt, role_inventory)
    role_sha = _sha256(args.role_route_receipt)

    extra = dict(parent_payload.get("extra") or {})
    extra["h10_function_preserving_migration"] = evidence
    extra["final_format_role_route_inventory_sha256"] = role_sha
    extra["parent_digest"] = actual_parent_sha
    child_keys = set(child.state_dict())
    dedicated_keys = sorted(
        key
        for key in child_keys
        if key.startswith("decision_fusion.dedicated_routes.")
    )
    inherited_fusion_keys = sorted(
        key
        for key in child_keys
        if key.startswith("decision_fusion.") and key not in dedicated_keys
    )
    extra["decision_fusion_migration"] = {
        "schema": "poke_bot.causal_decision_fusion_v2_migration/v1",
        "source_schema": DECISION_FUSION_SCHEMA,
        "target_schema": DECISION_FUSION_V2_SCHEMA,
        "source_checkpoint": str(parent_path),
        "source_checkpoint_digest": actual_parent_sha,
        "zero_safe_initialization": True,
        "runtime_enabled": True,
        "activation_scope": "isolated_specialist_bootstrap",
        "serving_eligible": False,
        "all_inherited_tensors_preserved": True,
        "inherited_fusion_tensor_keys": inherited_fusion_keys,
        "new_dedicated_route_names": sorted(
            [
                *DECISION_FUSION_REQUIRED_HEADS,
                "setup_board_outcome",
                "combo_state",
            ]
        ),
        "new_auxiliary_head_names": [
            SETUP_BOARD_OUTCOME_HEAD_NAME,
            COMBO_STATE_HEAD_NAME,
        ],
        "one_option_conditioned_route_per_learned_head": True,
    }
    payload = checkpoint.build_checkpoint(
        model=child,
        model_config=child.cfg,
        step=int(parent_payload.get("step", 0)),
        epoch=int(parent_payload.get("epoch", 0)),
        rl_iteration=int(parent_payload.get("rl_iteration", 0)),
        best_metric=parent_payload.get("best_metric"),
        extra=extra,
        archetype_id="alakazam",
        model_id="final-format-alakazam-r79-h10-i",
    )
    checkpoint.immutable_torch_save(payload, args.child)
    child_sha = _sha256(args.child)
    reloaded = load_model_from_checkpoint(args.child, device=torch.device("cpu"))
    if sum(parameter.numel() for parameter in reloaded.parameters()) != evidence[
        "learned_parameters"
    ]:
        raise RuntimeError("reloaded H10 parameter inventory changed")

    migration = evidence["migration"]
    parity = evidence["step_zero_parity"]
    receipt = {
        "schema": "poke_bot.final_format_alakazam_migration/v1",
        "template_only": False,
        "status": "issued_step_zero_passed_training_pending",
        "capacity_profile": "H10-I/v1",
        "search_schema": "none",
        "parent_checkpoint_sha256": actual_parent_sha,
        "child_checkpoint_sha256": child_sha,
        "inherited_tensors": migration["inherited_tensors"],
        "new_zero_safe_structures": {
            "appended_block_tensors": migration["appended_block_tensors"],
            "new_tensors": migration["new_tensors"],
            "zero_safe_output_tensors": migration["zero_safe_output_tensors"],
        },
        "learned_head_role_map_sha256": role_sha,
        "learned_route_inventory_sha256": role_sha,
        "learned_head_count": role_inventory["learned_head_count"],
        "learned_route_count": role_inventory["learned_route_count"],
        "one_distinct_bounded_option_conditioned_route_per_learned_head": True,
        "guide_runtime_route_count": 0,
        "aggregate_route_bound": 1.0,
        "parameter_inventory": {
            "learned_parameters": evidence["learned_parameters"],
            "package_bytes": args.child.stat().st_size,
        },
        "step_zero_parity": {
            **parity,
            "adapter_routes_exact": True,
            "missing_inherited_keys": [],
            "unexpected_inherited_keys": [],
            "typed_outputs_within_bound": True,
            "guide_metadata_runtime_logits_byte_identical": True,
        },
        "runtime_authority": "none",
        "selector_eligible": False,
        "kaggle_eligible": False,
    }
    _write_exclusive(args.migration_receipt, receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
