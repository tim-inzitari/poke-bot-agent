#!/usr/bin/env python3
"""Validate the completed revision-10 full-model derivative bootstrap."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


R10_GOAL_SHA256 = "sha256:0e6a73b6e215ffe4f1c076462ff778935a39fa8d11592ad8141d3f264d962250"
R10_CONTRACT_SHA256 = "sha256:91e9c60e87fe093446ef9979f64464be18e77a5041afe82164c4c6ca80d2225f"
R11_GOAL_SHA256 = "sha256:8f8ad130107c0699e32356b0800472d4861276c508093782c5490e95aa9e8074"
R11_CONTRACT_SHA256 = "sha256:d74152bca415c80e4983172b5fdcd8c03313e0c8a0a18e24e0c68a7bcfe84245"
R195_SHA256 = "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
INITIAL_SHA256 = "sha256:16373d030de6aedd8407a936bdb590dcd9364068238bbc26acc846a2f9f5e2e2"
FINAL_SHA256 = "sha256:8b59af9af1d715639bd3d63a84df7d608cee686c27aadf1c6dac3c971631a248"
COMPLETION_SHA256 = "sha256:e6398071f1abc64a4e886dbb3b361b5c411dbd7116a1f4164619a72f6f917c51"
SEMANTIC_PACK_SHA256 = "sha256:9261bc6c52f55810db59c313631ec51966f71e49abcbdd43f6b3e1fd198965a1"
EXPECTED_DECISIONS = 2_351_208
EXPECTED_OPTIONS = 18_412_973


class ValidationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def canonical_sha(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(body).hexdigest()


def write_json_create_only(path: Path, payload: Any) -> str:
    body = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(fd, body)
        os.fsync(fd)
    finally:
        os.close(fd)
    return "sha256:" + hashlib.sha256(body).hexdigest()


def _stable_file(path: Path, expected: str, label: str) -> None:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
        raise ValidationError(f"{label} identity mismatch: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--initial-candidate", type=Path, required=True)
    parser.add_argument("--semantic-pack-completion", type=Path, required=True)
    parser.add_argument("--goal", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    _stable_file(args.checkpoint, FINAL_SHA256, "final checkpoint")
    _stable_file(args.completion, COMPLETION_SHA256, "bootstrap completion")
    _stable_file(args.initial_candidate, INITIAL_SHA256, "initial composite candidate")
    _stable_file(args.goal, R11_GOAL_SHA256, "current goal")
    _stable_file(args.contract, R11_CONTRACT_SHA256, "current contract")

    completion = json.loads(args.completion.read_text())
    if (
        completion.get("schema") != "poke_bot.alakazam_rule_derivative_r10_full_bootstrap_receipt/v1"
        or completion.get("status") != "completed"
        or completion.get("output_sha256") != FINAL_SHA256
        or completion.get("epochs_completed") != 25
        or completion.get("r195_source_sha256") != R195_SHA256
    ):
        raise ValidationError("bootstrap completion receipt is not the sealed revision-10 run")
    semantic = completion.get("semantic_training") or {}
    epochs = semantic.get("epochs") or []
    if (
        semantic.get("decision_occurrences_per_epoch") != EXPECTED_DECISIONS
        or semantic.get("option_occurrences_per_epoch") != EXPECTED_OPTIONS
        or len(epochs) != 25
        or [row.get("epoch") for row in epochs] != list(range(1, 26))
        or any(row.get("decisions") != EXPECTED_DECISIONS for row in epochs)
        or any(row.get("options") != EXPECTED_OPTIONS for row in epochs)
        or any(not math.isfinite(float(row.get("loss"))) for row in epochs)
    ):
        raise ValidationError("semantic every-occurrence 25-epoch evidence is incomplete")

    pack = json.loads(args.semantic_pack_completion.read_text())
    manifest_sha = (
        pack.get("source_manifest_sha256")
        or pack.get("corpus_manifest_sha256")
        or (pack.get("source") or {}).get("manifest_sha256")
    )
    if manifest_sha != SEMANTIC_PACK_SHA256:
        raise ValidationError("semantic pack manifest binding drifted")

    import torch
    from poke_bot.alakazam_rule_derivative_frozen_tensor_rev9 import _build_base_model

    initial = torch.load(args.initial_candidate, map_location="cpu", weights_only=False)
    final = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (
        final.get("schema") != "poke_bot.alakazam_rule_derivative_composite_candidate_initialization/v1"
        or final.get("goal_revision") != 10
        or final.get("goal_gateway_sha256") != R10_GOAL_SHA256
        or final.get("goal_contract_sha256") != R10_CONTRACT_SHA256
        or final.get("step") != 25_194
    ):
        raise ValidationError("final checkpoint revision-10 binding is invalid")

    initial_base = initial.get("base_model_state_dict") or {}
    final_base = final.get("base_model_state_dict") or {}
    if set(initial_base) != set(final_base):
        raise ValidationError("base-model tensor inventory drifted")
    changed_noncombo: list[str] = []
    changed_combo: list[str] = []
    for name in sorted(final_base):
        before, after = initial_base[name], final_base[name]
        if before.shape != after.shape or before.dtype != after.dtype:
            raise ValidationError(f"tensor shape/dtype drift: {name}")
        if not bool(torch.isfinite(after).all()):
            raise ValidationError(f"nonfinite final tensor: {name}")
        if not torch.equal(before, after):
            (changed_combo if "combo_state" in name else changed_noncombo).append(name)
    if not changed_noncombo or changed_combo:
        raise ValidationError("all-weight training or combo exclusion evidence failed")

    model = _build_base_model(final["model_config"], final_base)
    parameter_names = [name for name, _ in model.named_parameters()]
    noncombo_parameter_names = [name for name in parameter_names if "combo_state" not in name]
    combo_parameter_names = [name for name in parameter_names if "combo_state" in name]
    optimizer_parameter_count = sum(
        len(group.get("params") or [])
        for group in (final.get("optimizer_state_dict") or {}).get("param_groups", [])
    )
    if optimizer_parameter_count != len(noncombo_parameter_names):
        raise ValidationError("optimizer did not cover every non-combo model parameter exactly once")

    initial_sidecar = initial.get("public_rule_semantic_projection_state_dict") or {}
    final_sidecar = final.get("public_rule_semantic_projection_state_dict") or {}
    if set(initial_sidecar) != set(final_sidecar):
        raise ValidationError("semantic projection tensor inventory drifted")
    changed_sidecar = [
        name for name in final_sidecar
        if not torch.equal(initial_sidecar[name], final_sidecar[name])
    ]
    if not changed_sidecar or any(not bool(torch.isfinite(v).all()) for v in final_sidecar.values()):
        raise ValidationError("semantic projection did not train finitely")
    sidecar_optimizer_count = sum(
        len(group.get("params") or [])
        for group in (final.get("public_rule_semantic_projection_optimizer_state_dict") or {}).get("param_groups", [])
    )
    if sidecar_optimizer_count != len(final_sidecar):
        raise ValidationError("semantic projection optimizer coverage drifted")

    evidence = {
        "schema": "poke_bot.alakazam_rule_derivative_r10_full_candidate_validation_evidence/v1",
        "goal_revision": 11,
        "goal_contract_sha256": R11_CONTRACT_SHA256,
        "bootstrap_goal_revision": 10,
        "bootstrap_goal_contract_sha256": R10_CONTRACT_SHA256,
        "checkpoint_sha256": FINAL_SHA256,
        "checkpoint_size_bytes": args.checkpoint.stat().st_size,
        "completion_receipt_sha256": COMPLETION_SHA256,
        "semantic_pack_manifest_sha256": SEMANTIC_PACK_SHA256,
        "epochs_validated": 25,
        "decision_occurrences_per_epoch": EXPECTED_DECISIONS,
        "option_occurrences_per_epoch": EXPECTED_OPTIONS,
        "base_tensor_count": len(final_base),
        "model_parameter_count": len(parameter_names),
        "noncombo_optimizer_parameter_count": optimizer_parameter_count,
        "combo_parameter_count": len(combo_parameter_names),
        "changed_noncombo_tensor_count": len(changed_noncombo),
        "changed_combo_tensor_count": len(changed_combo),
        "changed_semantic_projection_tensor_count": len(changed_sidecar),
        "all_final_tensors_finite": True,
        "combo_route_runtime_enabled": False,
        "combo_loss": 0.0,
        "validation_passed": True,
        "validated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    args.output_root.mkdir(parents=True, exist_ok=False)
    evidence_sha = write_json_create_only(args.output_root / "candidate-validation-evidence.json", evidence)
    receipt = {
        "schema": "poke_bot.alakazam_rule_derivative_r10_full_candidate_validation_receipt/v1",
        "status": "passed",
        "goal_revision": 11,
        "goal_contract_sha256": R11_CONTRACT_SHA256,
        "bootstrap_goal_revision": 10,
        "bootstrap_goal_contract_sha256": R10_CONTRACT_SHA256,
        "candidate_checkpoint_sha256": FINAL_SHA256,
        "bootstrap_completion_receipt_sha256": COMPLETION_SHA256,
        "validation_evidence_sha256": evidence_sha,
        "parameter_inventory_sha256": canonical_sha([
            {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in sorted(final_base.items())
        ]),
        "every_occurrence_25_epoch_validation_passed": True,
        "all_noncombo_parameters_optimizer_covered": True,
        "combo_parameters_optimizer_excluded_and_unchanged": True,
        "submission_authorized": True,
        "production_serving_selector_authority": False,
    }
    receipt_sha = write_json_create_only(args.output_root / "candidate-validation-receipt.json", receipt)
    complete_sha = write_json_create_only(args.output_root / "COMPLETE.json", {
        "schema": "poke_bot.alakazam_rule_derivative_r10_full_candidate_validation_completion/v1",
        "status": "passed",
        "candidate_checkpoint_sha256": FINAL_SHA256,
        "candidate_validation_receipt_sha256": receipt_sha,
        "validation_evidence_sha256": evidence_sha,
    })
    print(json.dumps({"receipt_sha256": receipt_sha, "evidence_sha256": evidence_sha, "completion_sha256": complete_sha, **evidence}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
