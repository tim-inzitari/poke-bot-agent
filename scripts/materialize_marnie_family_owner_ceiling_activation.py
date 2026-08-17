#!/usr/bin/env python3
"""Seal revision-139 owner-ceiling family activation artifacts.

The measured two-round study remains immutable and failed.  This script gives
that study no measured-pass status; it materializes separate owner authority
for the exact tested round-1-plus vector and the ordinary atomic installer.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from poke_bot.archetype_family_activation import (  # noqa: E402
    OWNER_CEILING_SCHEMA,
    REQUEST_SCHEMA,
    sha256,
    validate_activation_request,
    validate_iteration9_upload_trigger,
    validate_owner_ceiling_acceptance,
)
from poke_bot.archetype_family_study import owner_ceiling_loss_vector  # noqa: E402
from poke_bot.archetype_loss_contract import (  # noqa: E402
    canonical_residual_weights,
    validate_loss_contract,
)
from poke_bot.specialist_archetype_family import validate_manifest  # noqa: E402


OWNER_DECISION = "Do family thats the whole point lmao"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _identity(path: Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return {
        "path": str(source),
        "sha256": sha256(source),
        "bytes": source.stat().st_size,
    }


def _write_immutable(path: Path, value: Any) -> Path:
    target = Path(path).expanduser().resolve()
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if target.exists():
        if target.read_bytes() != encoded:
            raise RuntimeError(f"immutable artifact identity changed: {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return target


def _canonical_digest(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _hash_torch_value(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode())
            digest.update(json.dumps(list(tensor.shape)).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=lambda raw: repr(raw)):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode() + b"\0")
            for row in item:
                visit(row)
        elif isinstance(item, bytes):
            digest.update(b"bytes\0" + item)
        else:
            digest.update(
                json.dumps(item, sort_keys=True, default=repr).encode("utf-8")
            )

    visit(value)
    return "sha256:" + digest.hexdigest()


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    trigger = _read(args.trigger)
    validate_iteration9_upload_trigger(trigger)
    study = _read(args.study)
    rounds = list(study.get("rounds") or ())
    if len(rounds) != 2:
        raise RuntimeError("owner ceiling requires the exact two-round study")
    weights = canonical_residual_weights(
        ((((rounds[0].get("perturbation") or {}).get("candidates") or {}).get("plus") or {}))
    )
    checkpoint = Path(
        str(((trigger.get("bindings") or {}).get("checkpoint") or {}).get("path") or "")
    ).resolve()
    checkpoint_identity = _identity(checkpoint)
    parent_digest = str(
        (study.get("same_parent_validation") or {}).get(
            "parent_checkpoint_sha256", ""
        )
    )
    if checkpoint_identity["sha256"] != parent_digest:
        raise RuntimeError("owner ceiling does not target the exact studied parent")
    manifest = _read(args.manifest)
    validate_manifest(manifest, require_activation_ready=True)
    loss_contract = _read(args.loss_contract)
    validate_loss_contract(loss_contract)
    acceptance = {
        "schema": OWNER_CEILING_SCHEMA,
        "owner_revision": 139,
        "status": "owner_ceiling_accepted_after_inconclusive_study",
        "owner_decision": OWNER_DECISION,
        "measured_study_passed": False,
        "measured_study_relabelled": False,
        "measured_study_status": str(study.get("status") or ""),
        "study": _identity(args.study),
        "parent_checkpoint": checkpoint_identity,
        "selection": {
            "round": 1,
            "direction": "plus",
            "weights": weights,
            "basis": "only tested direction with a positive overall point estimate",
        },
        "ordinary_safety_checks_retained": [
            "same_parent",
            "training_and_replay_isolation",
            "causal_integrity",
            "activation_ready_manifest",
            "typed_loss_contract",
            "sealed_checkpoint_state",
            "atomic_sampler_and_loss_vector_migration",
            "post_activation_monitor",
        ],
        "runtime_authority_before_atomic_migration": False,
    }
    acceptance_path = _write_immutable(args.owner_acceptance, acceptance)
    validate_owner_ceiling_acceptance(
        _read(acceptance_path), study_path=args.study, study=study
    )
    vector = owner_ceiling_loss_vector(
        weights=weights,
        manifest_sha256=sha256(args.manifest),
        loss_contract_sha256=sha256(args.loss_contract),
        study_sha256=sha256(args.study),
        activation_authority_sha256=sha256(acceptance_path),
    )
    vector_path = _write_immutable(args.selected_loss_vector, vector)
    before_registry = _read(args.before_registry)
    candidate_registry = copy.deepcopy(before_registry)
    runtime_root = args.candidate_runtime_root.expanduser().resolve()
    if not runtime_root.is_dir():
        raise FileNotFoundError(runtime_root)
    for field in (
        "launcher",
        "active_gate_contract",
        "research_control_registry",
        "frozen_specialist_registry",
    ):
        relative = Path(str(candidate_registry.get(field) or ""))
        if relative.is_absolute() or not str(relative):
            raise RuntimeError(f"candidate registry {field} must be relative")
        if not (runtime_root / relative).is_file():
            raise FileNotFoundError(runtime_root / relative)
    trainer_args = list(candidate_registry.get("common_trainer_args") or ())
    try:
        reason_index = trainer_args.index("--boundary-design-migration-reason") + 1
    except ValueError as exc:
        raise RuntimeError("runtime registry lacks migration reason") from exc
    if reason_index >= len(trainer_args):
        raise RuntimeError("runtime registry migration reason is empty")
    trainer_args[reason_index] = "owner_ceiling_marnie_family_activation_r139"
    candidate_registry.update(
        {
            "runtime_root": str(runtime_root),
            "common_trainer_args": trainer_args,
            "deck_family_distribution": "package_20_equal_cluster_80_v1",
            "family_manifest": _identity(args.manifest),
            "variant_provenance": "checksum_bound_exact_observed_lists_v1",
            "replay_list_sampler": "deterministic_hamilton_family_macro_v1",
            "archetype_loss_contract": _identity(args.loss_contract),
            "selected_loss_vector": _identity(vector_path),
            "family_activation_authority": _identity(acceptance_path),
        }
    )
    candidate_path = _write_immutable(
        args.candidate_registry, candidate_registry
    )
    selector_identity = _identity(args.selector)
    checkpoint_payload = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    absent_manifest = _canonical_digest({"status": "absent_before_activation"})
    absent_vector = _canonical_digest({"status": "absent_before_activation"})
    request = {
        "schema": REQUEST_SCHEMA,
        "status": "ready_for_atomic_owner_ceiling_activation",
        "activation_authority": _identity(acceptance_path),
        "trigger": _identity(args.trigger),
        "manifest": _identity(args.manifest),
        "loss_contract": _identity(args.loss_contract),
        "study": _identity(args.study),
        "bindings": {
            "learner_sha256": checkpoint_identity["sha256"],
            "checkpoint": checkpoint_identity,
            "registry": _identity(args.before_registry),
            "selector": selector_identity,
            "selected_loss_vector": _identity(vector_path),
            "candidate_registry": _identity(candidate_path),
        },
        "sealed_pre_activation": {
            "registry": sha256(args.before_registry),
            "selector": selector_identity["sha256"],
            "learner": checkpoint_identity["sha256"],
            "checkpoint": checkpoint_identity["sha256"],
            "optimizer": _hash_torch_value(
                checkpoint_payload.get("optimizer_state_dict")
            ),
            "scaler": _hash_torch_value(
                checkpoint_payload.get("scaler_state_dict")
            ),
            "rng": _hash_torch_value(
                {
                    key: checkpoint_payload.get(key)
                    for key in (
                        "rng_state",
                        "torch_rng_state",
                        "cuda_rng_state",
                        "python_rng_state",
                        "numpy_rng_state",
                    )
                }
            ),
            "design_contract": sha256(args.run_manifest),
            "manifest": absent_manifest,
            "loss_vector": absent_vector,
        },
        "atomic_changes_only": [
            "runtime_root",
            "deck_family_distribution",
            "family_manifest",
            "variant_provenance",
            "replay_list_sampler",
            "archetype_loss_contract",
            "selected_loss_vector",
            "family_activation_authority",
        ],
        "next_collection_started": False,
        "measured_study_passed": False,
    }
    request_path = _write_immutable(args.activation_request, request)
    validation = validate_activation_request(
        _read(request_path),
        expected_learner_digest=checkpoint_identity["sha256"],
    )
    return {
        "status": "ready_for_atomic_owner_ceiling_activation",
        "owner_acceptance": _identity(acceptance_path),
        "selected_loss_vector": _identity(vector_path),
        "candidate_registry": _identity(candidate_path),
        "activation_request": _identity(request_path),
        "validation": validation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "trigger",
        "manifest",
        "loss-contract",
        "study",
        "run-manifest",
        "before-registry",
        "candidate-runtime-root",
        "selector",
        "owner-acceptance",
        "selected-loss-vector",
        "candidate-registry",
        "activation-request",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    result = materialize(parser.parse_args())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
