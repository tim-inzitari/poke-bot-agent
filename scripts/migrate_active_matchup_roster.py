#!/usr/bin/env python3
"""Migrate one active v4 specialist checkpoint/tree to roster18 v5.

Retained adapter tensors are copied by route identity, five retired routes are
physically deleted, Festival Lead is renamed to Thwackey without changing its
weights, and Team Rocket's Spidops is initialized to exact zeros.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.checkpoint import immutable_torch_save
from poke_bot.matchup_adapters import (
    EXPERT_IDS,
    LEGACY_EXPERT_IDS_V4,
    MatchupAdapterBank,
)


RETIRED = {
    "cornerstone-ogerpon",
    "lopunny",
    "gardevoir",
    "ns-zoroark",
    "raging-bolt",
}
SOURCE_BY_TARGET = {
    **{value: value for value in EXPERT_IDS},
    "thwackey": "festival-lead",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return f"sha256:{value.hexdigest()}"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def migrate_route_mapping(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    output: dict[str, Any] = {}
    for target in EXPERT_IDS:
        source = SOURCE_BY_TARGET.get(target, target)
        output[target] = copy.deepcopy(value.get(source, 0))
    return output


def migrate_fit(value: Any) -> Any:
    fit = copy.deepcopy(value) if isinstance(value, dict) else {}
    for key in (
        "route_sequences",
        "route_decisions",
        "phase_route_sequences",
        "phase_route_decisions",
    ):
        if isinstance(fit.get(key), dict):
            fit[key] = migrate_route_mapping(fit[key])
    fit["trained_archetype_ids"] = [
        ("thwackey" if value == "festival-lead" else value)
        for value in fit.get("trained_archetype_ids") or []
        if value not in RETIRED
    ]
    fit["dormant_no_example_archetype_ids"] = [
        target
        for target in EXPERT_IDS
        if int((fit.get("route_decisions") or {}).get(target, 0)) <= 0
    ]
    fit["optimizer_state_restored"] = False
    fit["roster_migration"] = "v4_22_to_canonical_v5_18"
    return fit


def migrate_checkpoint(source: Path, output: Path) -> dict[str, Any]:
    payload = torch.load(source, map_location="cpu", weights_only=False)
    extra = dict(payload.get("extra") or {})
    config = dict(extra.get("matchup_adapter_config") or {})
    old_ids = tuple(str(value) for value in config.get("expert_ids") or ())
    if old_ids != LEGACY_EXPERT_IDS_V4:
        raise RuntimeError(f"source checkpoint is not exact v4 roster: {old_ids}")
    state = dict(payload.get("model_state_dict") or {})
    adapter_prefix = "matchup_adapter_bank.experts."
    adapter_state = {
        key: value for key, value in state.items() if key.startswith(adapter_prefix)
    }
    expected_old_keys = {
        f"{adapter_prefix}{route}.{suffix}"
        for route in range(len(LEGACY_EXPERT_IDS_V4))
        for suffix in ("down.weight", "down.bias", "up.weight", "up.bias")
    }
    if set(adapter_state) != expected_old_keys:
        raise RuntimeError("source checkpoint adapter tensor set is incomplete")
    migrated_state = {
        key: value
        for key, value in state.items()
        if not key.startswith(adapter_prefix)
    }
    old_index = {value: index for index, value in enumerate(LEGACY_EXPERT_IDS_V4)}
    retained_checks: dict[str, bool] = {}
    for new_route, target in enumerate(EXPERT_IDS):
        source_id = SOURCE_BY_TARGET.get(target)
        for suffix in ("down.weight", "down.bias", "up.weight", "up.bias"):
            new_key = f"{adapter_prefix}{new_route}.{suffix}"
            if target == "team-rockets-spidops":
                reference = adapter_state[f"{adapter_prefix}0.{suffix}"]
                migrated_state[new_key] = torch.zeros_like(reference)
                continue
            if source_id is None or source_id not in old_index:
                raise RuntimeError(f"no v4 source row for {target}")
            old_key = f"{adapter_prefix}{old_index[source_id]}.{suffix}"
            migrated_state[new_key] = adapter_state[old_key].detach().clone()
            retained_checks[f"{target}:{suffix}"] = torch.equal(
                migrated_state[new_key], adapter_state[old_key]
            )
    new_config = MatchupAdapterBank.config_dict()
    migrated = copy.deepcopy(payload)
    migrated["model_state_dict"] = migrated_state
    migrated_extra = dict(migrated.get("extra") or {})
    old_adapter_parameters = sum(int(value.numel()) for value in adapter_state.values())
    new_adapter_parameters = sum(
        int(value.numel())
        for key, value in migrated_state.items()
        if key.startswith(adapter_prefix)
    )
    if int(migrated_extra.get("param_count") or 0) > 0:
        migrated_extra["param_count"] = (
            int(migrated_extra["param_count"])
            - old_adapter_parameters
            + new_adapter_parameters
        )
    migrated_extra["matchup_adapter_config"] = new_config
    migrated_extra["dormant_matchup_adapter_fit"] = migrate_fit(
        migrated_extra.get("dormant_matchup_adapter_fit")
    )
    migrated_extra.pop("dormant_matchup_adapter_optimizer_state", None)
    dormant = dict(migrated_extra.get("dormant_matchup_adapter_bank") or {})
    dormant.update(
        adapter_config=new_config,
        parameter_count=new_adapter_parameters,
        optimizer_imported=False,
        optimizer_included=False,
    )
    migrated_extra["dormant_matchup_adapter_bank"] = dormant
    migrated_extra["roster_migration"] = {
        "schema": "poke_bot.matchup_adapter_roster_migration/v1",
        "source_checkpoint": str(source),
        "source_checkpoint_digest": digest(source),
        "source_expert_ids": list(LEGACY_EXPERT_IDS_V4),
        "target_expert_ids": list(EXPERT_IDS),
        "removed_expert_ids": sorted(RETIRED),
        "renamed_expert_ids": {"festival-lead": "thwackey"},
        "zero_initialized_expert_ids": ["team-rockets-spidops"],
        "retained_rows_byte_identical": all(retained_checks.values()),
    }
    migrated["extra"] = migrated_extra
    immutable_torch_save(migrated, output)
    return {
        "output": str(output),
        "output_digest": digest(output),
        "source_digest": digest(source),
        "old_adapter_parameters": old_adapter_parameters,
        "new_adapter_parameters": new_adapter_parameters,
        "retained_tensor_checks": len(retained_checks),
        "retained_rows_byte_identical": all(retained_checks.values()),
        "removed_rows_absent": not any(
            value in EXPERT_IDS for value in RETIRED
        ),
        "spidops_exact_zero": all(
            int(value.count_nonzero().item()) == 0
            for key, value in migrated_state.items()
            if key.startswith(
                f"{adapter_prefix}{EXPERT_IDS.index('team-rockets-spidops')}."
            )
        ),
        "target_expert_ids": list(EXPERT_IDS),
        "target_parameter_count": migrated_extra.get("param_count"),
    }


def migrate_tree(
    source: Path,
    output: Path,
    *,
    checkpoint: Path,
    checkpoint_digest: str,
) -> dict[str, Any]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    old_targets = tuple(str(value) for value in payload.get("targets") or ())
    if old_targets != LEGACY_EXPERT_IDS_V4:
        raise RuntimeError("source public tree is not exact v4 roster")
    tree = dict(payload.get("tree") or {})
    old_class_names = tuple(str(value) for value in tree.get("class_names") or ())
    if old_class_names != old_targets + ("unknown",):
        raise RuntimeError("source public tree class order changed")
    old_index = {value: index for index, value in enumerate(old_targets)}
    new_counts = []
    for row in tree.get("weighted_class_counts") or []:
        if len(row) != len(old_targets) + 1:
            raise RuntimeError("source public tree class width changed")
        mapped = []
        for target in EXPERT_IDS:
            source_id = SOURCE_BY_TARGET.get(target)
            mapped.append(
                0.0
                if target == "team-rockets-spidops"
                else float(row[old_index[source_id]])
            )
        unknown = float(row[len(old_targets)]) + sum(
            float(row[old_index[value]]) for value in RETIRED
        )
        mapped.append(unknown)
        new_counts.append(mapped)
    migrated = copy.deepcopy(payload)
    migrated["targets"] = list(EXPERT_IDS)
    prediction = dict(migrated.get("prediction_contract") or {})
    prediction.update(
        adapter_count=len(EXPERT_IDS),
        route_class_names=list(EXPERT_IDS),
        route_output_width=len(EXPERT_IDS),
        unknown_class_index=len(EXPERT_IDS),
    )
    migrated["prediction_contract"] = prediction
    migrated_tree = dict(migrated.get("tree") or {})
    migrated_tree["class_names"] = [*EXPERT_IDS, "unknown"]
    migrated_tree["weighted_class_counts"] = new_counts
    migrated["tree"] = migrated_tree
    runtime = dict(migrated.get("runtime_contract") or {})
    accepted = [
        ("thwackey" if value == "festival-lead" else value)
        for value in runtime.get("accepted_archetype_ids") or []
        if value not in RETIRED
    ]
    runtime["accepted_archetype_ids"] = accepted
    thresholds = dict(runtime.get("per_archetype_min_leaf_confidence") or {})
    runtime["per_archetype_min_leaf_confidence"] = {
        target: thresholds.get(SOURCE_BY_TARGET.get(target, target))
        for target in accepted
    }
    runtime["checkpoint"] = str(checkpoint)
    runtime["checkpoint_digest"] = checkpoint_digest
    runtime["source_tree_digest"] = digest(source)
    rejected = dict(runtime.get("rejected_archetype_ids") or {})
    rejected["team-rockets-spidops"] = "no validated visible-card support"
    runtime["rejected_archetype_ids"] = rejected
    migrated["runtime_contract"] = runtime
    calibration = dict(migrated.get("runtime_calibration") or {})
    per_archetype = dict(calibration.get("per_archetype") or {})
    calibration["per_archetype"] = {
        target: copy.deepcopy(per_archetype.get(SOURCE_BY_TARGET.get(target, target)))
        for target in EXPERT_IDS
        if target != "team-rockets-spidops"
        and per_archetype.get(SOURCE_BY_TARGET.get(target, target)) is not None
    }
    calibration["per_archetype"]["team-rockets-spidops"] = {
        "available": False,
        "accepted_weighted_observations": 0.0,
        "precision": None,
        "recall": None,
        "min_leaf_confidence": None,
        "reason": "no validated visible-card support",
    }
    migrated["runtime_calibration"] = calibration
    contract = dict(migrated.get("calibration_contract") or {})
    contract["canonical_class_count"] = len(EXPERT_IDS) + 1
    contract["fitted_class_indexes"] = list(range(len(EXPERT_IDS) + 1))
    migrated["calibration_contract"] = contract
    migrated["activation_status"] = "validated_identity_migration"
    migrated["roster_migration"] = {
        "schema": "poke_bot.public_matchup_tree_roster_migration/v1",
        "source": str(source),
        "source_digest": digest(source),
        "removed_expert_ids": sorted(RETIRED),
        "renamed_expert_ids": {"festival-lead": "thwackey"},
        "new_zero_support_expert_ids": ["team-rockets-spidops"],
    }
    atomic_json(output, migrated)
    return {
        "output": str(output),
        "output_digest": digest(output),
        "accepted_archetype_ids": accepted,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--tree-output", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--authorization-output", type=Path, required=True)
    parser.add_argument("--loop-state", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--apply-loop", action="store_true")
    args = parser.parse_args()
    checkpoint_result = migrate_checkpoint(
        args.checkpoint.resolve(), args.checkpoint_output.resolve()
    )
    tree_result = migrate_tree(
        args.tree.resolve(),
        args.tree_output.resolve(),
        checkpoint=args.checkpoint_output.resolve(),
        checkpoint_digest=checkpoint_result["output_digest"],
    )
    authorization = json.loads(args.authorization.read_text(encoding="utf-8"))
    authorization["parent_checkpoint"] = str(args.checkpoint_output.resolve())
    authorization["parent_checkpoint_digest"] = checkpoint_result["output_digest"]
    authorization["roster_source"] = "state/matchup_adapter_roster.json"
    authorization["roster_expert_ids"] = list(EXPERT_IDS)
    atomic_json(args.authorization_output.resolve(), authorization)
    result = {
        "schema": "poke_bot.active_matchup_roster_migration/v1",
        "checkpoint": checkpoint_result,
        "runtime_tree": tree_result,
        "authorization": {
            "output": str(args.authorization_output.resolve()),
            "output_digest": digest(args.authorization_output.resolve()),
        },
        "loop_state": str(args.loop_state.resolve()),
        "applied_to_loop": bool(args.apply_loop),
    }
    if args.apply_loop:
        loop = json.loads(args.loop_state.read_text(encoding="utf-8"))
        loop["learner"] = {
            "path": str(args.checkpoint_output.resolve()),
            "digest": checkpoint_result["output_digest"],
        }
        loop["dormant_matchup_adapter_fit"] = migrate_fit(
            loop.get("dormant_matchup_adapter_fit")
        )
        loop["dormant_matchup_adapter_fit"]["checkpoint_path"] = str(
            args.checkpoint_output.resolve()
        )
        loop["dormant_matchup_adapter_fit"]["checkpoint_digest"] = (
            checkpoint_result["output_digest"]
        )
        loop["roster_migration"] = {
            "receipt": str(args.receipt.resolve()),
            "schema": result["schema"],
        }
        atomic_json(args.loop_state.resolve(), loop)
    atomic_json(args.receipt.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
