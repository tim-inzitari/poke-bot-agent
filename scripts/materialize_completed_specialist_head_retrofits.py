#!/usr/bin/env python3
"""Materialize dormant expanded-head derivatives of frozen specialists.

This is an architecture-only compatibility operation.  It never rewrites a
passing checkpoint, never imports optimizer state, and never makes the
derivative eligible for serving, gates, public mix, or Kaggle.  The added
strategic-head tensors are copied byte-for-byte from one checksum-bound
cumulative core and remain runtime-disabled until a later explicit retrofit
training and activation phase.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import checkpoint
from poke_bot.model import EXPANDED_HEAD_KEY_PREFIXES
from poke_bot.strategic_schedule import EXPANDED_HEAD_IDS
from poke_bot.train import load_model_from_checkpoint
from scripts.run_starmie_expert_bootstrap import load_expanded_head_contract


RETROFIT_SCHEMA = "poke_bot.completed_specialist_head_retrofit/v1"
RECEIPT_SCHEMA = "poke_bot.completed_specialist_head_retrofit_set/v1"
MIGRATION_SCHEMA = "poke_bot.expanded_head_migration/v1"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _expanded_state(state: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        name: value
        for name, value in state.items()
        if name.startswith(EXPANDED_HEAD_KEY_PREFIXES)
        and isinstance(value, torch.Tensor)
    }


def _assert_digest(value: Any, *, field: str) -> str:
    digest = str(value or "")
    if (
        not digest.startswith("sha256:")
        or len(digest) != 71
        or any(char not in "0123456789abcdef" for char in digest[7:])
    ):
        raise RuntimeError(f"{field} is not a complete sha256 digest")
    return digest


def _core_identity(
    core_checkpoint: Path,
    *,
    protocol: Path,
) -> dict[str, Any]:
    core_checkpoint = core_checkpoint.expanduser().resolve()
    payload = checkpoint.load_checkpoint(core_checkpoint, map_location="cpu")
    state = dict(payload.get("model_state_dict") or {})
    model_config = dict(payload.get("model_config") or {})
    extra = dict(payload.get("extra") or {})
    training = dict(extra.get("expanded_head_training") or {})
    _raw, canonical = load_expanded_head_contract(protocol)
    head_state = _expanded_state(state)
    expected_tensor_count = len(EXPANDED_HEAD_IDS) * 2
    if (
        model_config.get("expanded_heads_enabled") is not True
        or len(head_state) != expected_tensor_count
        or set(training.get("architecture_present_heads") or ())
        != set(EXPANDED_HEAD_IDS)
        or set(training.get("trained_heads") or ()) != set(EXPANDED_HEAD_IDS)
        or training.get("runtime_enabled_heads") != []
        or training.get("target_schema_digest")
        != canonical["target_schema_digest"]
        or training.get("schedule_digest") != canonical["schedule_digest"]
    ):
        raise RuntimeError(
            "cumulative core is not a complete runtime-disabled expanded-head "
            "teacher"
        )
    return {
        "path": core_checkpoint,
        "digest": checkpoint.checkpoint_digest(core_checkpoint),
        "payload": payload,
        "head_state": head_state,
        "canonical": canonical,
    }


def materialize_derivative(
    *,
    specialist_id: str,
    source_checkpoint: Path,
    source_passing_checkpoint_digest: str,
    core: dict[str, Any],
    output_checkpoint: Path,
) -> dict[str, Any]:
    """Create and validate one immutable dormant derivative."""

    source_checkpoint = source_checkpoint.expanduser().resolve()
    output_checkpoint = output_checkpoint.expanduser().resolve()
    source_digest = checkpoint.checkpoint_digest(source_checkpoint)
    passing_digest = _assert_digest(
        source_passing_checkpoint_digest,
        field="source_passing_checkpoint_digest",
    )
    source_payload = checkpoint.load_checkpoint(
        source_checkpoint, map_location="cpu"
    )
    source_state = dict(source_payload.get("model_state_dict") or {})
    source_config = dict(source_payload.get("model_config") or {})
    if source_config.get("expanded_heads_enabled") is True or _expanded_state(
        source_state
    ):
        raise RuntimeError(
            f"{specialist_id} source already contains expanded heads"
        )
    source_archetype = str(source_payload.get("archetype_id") or "")
    if source_archetype != specialist_id:
        raise RuntimeError(
            f"{specialist_id} source archetype changed: {source_archetype!r}"
        )

    canonical = dict(core["canonical"])
    head_state = dict(core["head_state"])
    state = dict(source_state)
    for name, tensor in head_state.items():
        state[name] = tensor.detach().cpu().clone()

    model_config = dict(source_config)
    model_config["expanded_heads_enabled"] = True
    migration = {
        "schema": MIGRATION_SCHEMA,
        "status": "dormant_core_warmstart_untrained_for_specialist",
        "target_architecture_schema": canonical["architecture_schema"],
        "target_schema": canonical["target_schema"],
        "target_schema_digest": canonical["target_schema_digest"],
        "schedule_schema": canonical["schedule_schema"],
        "schedule_digest": canonical["schedule_digest"],
        "runtime_enabled_heads": [],
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_digest": source_digest,
        "source_passing_checkpoint_digest": passing_digest,
        "cumulative_core_checkpoint": str(core["path"]),
        "cumulative_core_checkpoint_digest": str(core["digest"]),
        "initialization_method": (
            "byte_exact_copy_from_checksum_bound_cumulative_core"
        ),
        "initialization_seed": None,
        "added_head_tensor_count": len(head_state),
        "inherited_source_tensor_count": len(source_state),
        "all_inherited_source_tensors_byte_identical": True,
        "specialist_training_metadata_reset": True,
        "architecture_migration_counts_as_training_complete": False,
        "architecture_migration_counts_as_gate_pass": False,
    }
    retrofit = {
        "schema": RETROFIT_SCHEMA,
        "status": "dormant_untrained",
        "specialist_id": specialist_id,
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_digest": source_digest,
        "source_passing_checkpoint_digest": passing_digest,
        "cumulative_core_checkpoint": str(core["path"]),
        "cumulative_core_checkpoint_digest": str(core["digest"]),
        "architecture_present_heads": list(EXPANDED_HEAD_IDS),
        "specialist_trained_heads": [],
        "gradient_enabled_heads": [],
        "runtime_enabled_heads": [],
        "flat_policy_authoritative": True,
        "serving_eligible": False,
        "public_mix_eligible": False,
        "gate_eligible": False,
        "kaggle_submission_eligible": False,
        "later_training_requires_explicit_retrofit_phase": True,
        "exact_future_bootstrap_epochs": 25,
        "target_schema": canonical["target_schema"],
        "target_schema_digest": canonical["target_schema_digest"],
        "schedule_schema": canonical["schedule_schema"],
        "schedule_digest": canonical["schedule_digest"],
        "initialization_method": migration["initialization_method"],
    }

    payload = dict(source_payload)
    payload["model_state_dict"] = state
    payload["model_config"] = model_config
    payload["step"] = 0
    payload["epoch"] = 0
    payload["rl_iteration"] = 0
    payload["model_id"] = (
        f"{specialist_id}.completed_specialist_head_retrofit_v1.dormant"
    )
    for key in (
        "optimizer_state_dict",
        "scaler_state_dict",
        "scheduler_state_dict",
        "rng_state",
        "early_stop_state",
    ):
        payload.pop(key, None)
    extra = dict(source_payload.get("extra") or {})
    extra.pop("expanded_head_training", None)
    extra["expanded_head_migration"] = migration
    extra["completed_specialist_head_retrofit"] = retrofit
    extra["runtime_enabled_heads"] = []
    payload["extra"] = extra
    provenance = dict(source_payload.get("provenance") or {})
    provenance["expanded_heads"] = {
        "schema": canonical["architecture_schema"],
        "version": 1,
        "enabled": True,
        "runtime_enabled_heads": [],
        "modules": {},
    }
    provenance["warm_started_expanded_heads"] = []
    payload["provenance"] = provenance

    if output_checkpoint.exists():
        existing = checkpoint.load_checkpoint(
            output_checkpoint, map_location="cpu"
        )
        existing_retrofit = dict(
            (existing.get("extra") or {}).get(
                "completed_specialist_head_retrofit"
            )
            or {}
        )
        if existing_retrofit != retrofit:
            raise RuntimeError(
                f"existing {specialist_id} retrofit identity changed"
            )
    else:
        checkpoint.immutable_torch_save(payload, output_checkpoint)

    derivative_digest = checkpoint.checkpoint_digest(output_checkpoint)
    loaded = load_model_from_checkpoint(output_checkpoint, device=torch.device("cpu"))
    loaded_state = loaded.state_dict()
    for name, tensor in source_state.items():
        if name not in loaded_state or not torch.equal(
            tensor.detach().cpu(), loaded_state[name].detach().cpu()
        ):
            raise RuntimeError(
                f"{specialist_id} inherited source tensor changed: {name}"
            )
    for name, tensor in head_state.items():
        if name not in loaded_state or not torch.equal(
            tensor.detach().cpu(), loaded_state[name].detach().cpu()
        ):
            raise RuntimeError(
                f"{specialist_id} cumulative-core head tensor changed: {name}"
            )
    inventory = loaded.expanded_head_inventory()
    if (
        inventory.get("runtime_enabled_heads") != []
        or set(inventory.get("modules") or {}) == set()
        or len(inventory.get("modules") or {}) != len(EXPANDED_HEAD_IDS)
    ):
        raise RuntimeError(
            f"{specialist_id} dormant expanded-head inventory is invalid"
        )
    if checkpoint.checkpoint_digest(source_checkpoint) != source_digest:
        raise RuntimeError(
            f"{specialist_id} source checkpoint changed during retrofit"
        )

    manifest = {
        **retrofit,
        "checkpoint": str(output_checkpoint),
        "checkpoint_digest": derivative_digest,
        "source_checkpoint_unchanged": True,
        "inherited_source_tensor_count": len(source_state),
        "inherited_source_tensors_byte_identical": len(source_state),
        "copied_core_head_tensor_count": len(head_state),
        "copied_core_head_tensors_byte_identical": len(head_state),
        "strict_model_load_passed": True,
        "runtime_enabled_heads": [],
    }
    manifest_path = output_checkpoint.with_name("manifest.json")
    if manifest_path.exists():
        existing_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if existing_manifest != manifest:
            raise RuntimeError(
                f"existing {specialist_id} retrofit manifest changed"
            )
    else:
        _atomic_json(manifest_path, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-checkpoint", type=Path, required=True)
    parser.add_argument("--frozen-registry", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "config/rl_protocol.yaml",
    )
    args = parser.parse_args(argv)

    core = _core_identity(args.core_checkpoint, protocol=args.protocol)
    registry = json.loads(
        args.frozen_registry.expanduser().resolve().read_text(encoding="utf-8")
    )
    rows = list(registry.get("specialists") or ())
    if not rows:
        raise RuntimeError("frozen specialist registry is empty")
    manifests: list[dict[str, Any]] = []
    for row in rows:
        specialist_id = str(row.get("specialist_id") or "")
        baseline_dir = str(row.get("baseline_dir") or "")
        if not specialist_id or not baseline_dir:
            raise RuntimeError("frozen specialist registry row is incomplete")
        source = (
            args.baseline_root.expanduser().resolve()
            / "specialists"
            / baseline_dir
            / "model.pt"
        )
        manifests.append(
            materialize_derivative(
                specialist_id=specialist_id,
                source_checkpoint=source,
                source_passing_checkpoint_digest=str(
                    row.get("source_passing_checkpoint_digest")
                    # A directly registered V5 passing package has no
                    # compatibility-derivative parent; its checkpoint digest
                    # is itself the original passing digest.
                    or row.get("checkpoint_digest")
                    or ""
                ),
                core=core,
                output_checkpoint=(
                    args.output_root.expanduser().resolve()
                    / specialist_id
                    / "model.pt"
                ),
            )
        )

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "ready_dormant_untrained",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "core_checkpoint": str(core["path"]),
        "core_checkpoint_digest": str(core["digest"]),
        "frozen_registry": str(args.frozen_registry.expanduser().resolve()),
        "frozen_registry_digest": checkpoint.checkpoint_digest(
            args.frozen_registry
        ),
        "derivative_count": len(manifests),
        "specialists": manifests,
        "all_original_source_checkpoints_unchanged": True,
        "all_runtime_enabled_heads": [],
        "training_started": False,
        "serving_changed": False,
        "selector_changed": False,
    }
    receipt_path = args.receipt.expanduser().resolve()
    if receipt_path.exists():
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        comparable_existing = dict(existing)
        comparable_existing.pop("created_at_utc", None)
        comparable_receipt = dict(receipt)
        comparable_receipt.pop("created_at_utc", None)
        if comparable_existing != comparable_receipt:
            raise RuntimeError("existing retrofit-set receipt identity changed")
        receipt = existing
    else:
        _atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
