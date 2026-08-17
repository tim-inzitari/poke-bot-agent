#!/usr/bin/env python3
"""Migrate a committed H10 specialist checkpoint to directional Fusion v3.

The historical Alakazam invocation remains compatible through defaults.  A
later final-format refresh must pass its own specialist/model/schema identity,
which prevents a Marnie child from inheriting an Alakazam label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import torch

from poke_bot import checkpoint
from poke_bot.model import build_model
from poke_bot.train import load_model_from_checkpoint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _write_once(path: Path, value: dict[str, object]) -> None:
    body = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != body:
            raise RuntimeError(f"fusion-v3 receipt changed: {path}")
        return
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(body, encoding="utf-8")
    os.link(temporary, path)
    temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--specialist-id", default="alakazam")
    parser.add_argument("--model-id", default="")
    parser.add_argument(
        "--migration-schema",
        default="poke_bot.alakazam_fusion_v3_migration/v1",
    )
    args = parser.parse_args()
    specialist_id = str(args.specialist_id).strip().casefold()
    if not specialist_id or any(
        char not in "abcdefghijklmnopqrstuvwxyz0123456789-"
        for char in specialist_id
    ):
        raise RuntimeError("unsafe final-format specialist id")
    migration_schema = str(args.migration_schema).strip()
    if (
        not migration_schema.startswith("poke_bot.final_format_")
        and migration_schema != "poke_bot.alakazam_fusion_v3_migration/v1"
    ) or not migration_schema.endswith("/v1"):
        raise RuntimeError("invalid Fusion-v3 migration schema")
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    actual = _sha256(source)
    if actual != args.expected_source_sha256:
        raise RuntimeError(f"source checkpoint changed: {actual}")

    payload = checkpoint.load_checkpoint(source, map_location="cpu")
    parent = load_model_from_checkpoint(source, device=torch.device("cpu"))
    cfg = replace(
        parent.cfg,
        decision_fusion_typed_output_centered_routes_enabled=True,
        decision_fusion_action_type_reliability_cap=0.25,
    )
    child = build_model(
        cfg,
        device=torch.device("cpu"),
        aux_archetype_classes=int(parent.aux_head[-1].out_features),
        encoder_vocab=int(parent.encoder_vocab),
        decoder_vocab=int(parent.decoder_vocab),
        belief_card_vocab=int(parent.belief_card_vocab),
    )
    parent_state = parent.state_dict()
    child_state = child.state_dict()
    missing_parent = sorted(set(parent_state) - set(child_state))
    if missing_parent:
        raise RuntimeError(f"fusion-v3 child omitted parent tensors: {missing_parent}")
    new_keys = sorted(set(child_state) - set(parent_state))
    if not new_keys or any(
        not key.startswith("decision_fusion.dedicated_route_log_reliability.")
        for key in new_keys
    ):
        raise RuntimeError(f"unexpected fusion-v3 tensors: {new_keys}")
    for key, value in parent_state.items():
        child_state[key] = value.detach().clone()
    for key in new_keys:
        child_state[key] = torch.zeros_like(child_state[key])
    child.load_state_dict(child_state, strict=True)

    migrated = dict(payload)
    migrated["model_config"] = asdict(cfg)
    migrated["model_state_dict"] = child.state_dict()
    migrated["archetype_id"] = specialist_id
    migrated["model_id"] = (
        str(args.model_id).strip()
        or str(payload.get("model_id") or "") + "-fusion-v3-r104"
    )
    provenance = dict(payload.get("provenance") or {})
    provenance["decision_fusion"] = child.decision_fusion_inventory()
    migrated["provenance"] = provenance
    extra = dict(payload.get("extra") or {})
    extra["fusion_v3_migration"] = {
        "schema": migration_schema,
        "specialist_id": specialist_id,
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": actual,
        "typed_output_centered_routes": True,
        "positive_bounded_reliability": [0.25, 4.0],
        "action_type_reliability_cap": 0.25,
        "new_tensors": new_keys,
        "all_parent_tensors_preserved": True,
        "optimizer_state_preserved": True,
    }
    migrated["extra"] = extra

    optimizer = dict(migrated.get("optimizer_state_dict") or {})
    groups = [dict(group) for group in optimizer.get("param_groups") or []]
    parent_trainable = [
        name for name, parameter in parent.named_parameters() if parameter.requires_grad
    ]
    child_trainable = [
        name for name, parameter in child.named_parameters() if parameter.requires_grad
    ]
    old_count = len(parent_trainable)
    new_count = len(child_trainable)
    flat = [int(value) for group in groups for value in group.get("params") or []]
    if flat != list(range(old_count)) or not groups:
        raise RuntimeError("optimizer parameter inventory does not match trainable parent")
    child_ids = {name: index for index, name in enumerate(child_trainable)}
    if any(name not in child_ids for name in parent_trainable):
        raise RuntimeError("optimizer migration omitted a trainable parent parameter")
    old_to_new = {
        old_id: child_ids[name] for old_id, name in enumerate(parent_trainable)
    }
    migrated_state = {
        old_to_new[int(old_id)]: value
        for old_id, value in dict(optimizer.get("state") or {}).items()
    }
    optimizer["state"] = migrated_state
    for group in groups:
        group["params"] = [old_to_new[int(value)] for value in group["params"]]
    new_trainable_ids = [
        child_ids[name] for name in child_trainable if name not in set(parent_trainable)
    ]
    groups[-1]["params"] = [*groups[-1]["params"], *new_trainable_ids]
    optimizer["param_groups"] = groups
    migrated["optimizer_state_dict"] = optimizer

    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.immutable_torch_save(migrated, output)
    reloaded = load_model_from_checkpoint(output, device=torch.device("cpu"))
    reloaded_payload = checkpoint.load_checkpoint(output, map_location="cpu")
    if (
        reloaded_payload.get("archetype_id") != specialist_id
        or reloaded_payload.get("model_id") != migrated["model_id"]
    ):
        raise RuntimeError("reloaded Fusion-v3 specialist identity changed")
    if not reloaded.decision_fusion_typed_output_centered_routes_enabled:
        raise RuntimeError("reloaded fusion-v3 flag is false")
    if reloaded.decision_fusion_action_type_reliability_cap != 0.25:
        raise RuntimeError("reloaded action_type reliability cap changed")
    output_sha = _sha256(output)
    receipt = {
        "schema": migration_schema,
        "status": "validated",
        "specialist_id": specialist_id,
        "model_id": migrated["model_id"],
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": actual,
        "output_checkpoint": str(output),
        "output_checkpoint_sha256": output_sha,
        "new_optimizer_parameter_count": new_count - old_count,
        "new_tensors": new_keys,
        "all_parent_tensors_preserved": True,
        "optimizer_state_preserved_and_extended": True,
        "typed_output_centered_routes": True,
        "route_reliability_bounds": [0.25, 4.0],
        "action_type_reliability_cap": 0.25,
        "guide_mode_required_for_next_iteration": "strategic_directional_v2",
    }
    _write_once(args.receipt.expanduser().resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
