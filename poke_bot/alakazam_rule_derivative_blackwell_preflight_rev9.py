"""Nonactivating revision-9 Blackwell load/gradient/optimizer canary.

The producer consumes the sealed composite initialization and frozen-tensor
receipt, runs one synthetic sidecar-only optimizer step on the contract-bound
Inzi Blackwell, and emits the exact closed preflight receipt.  It never loads
the corpus, controls a service, or mutates either parent checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from poke_bot.alakazam_rule_derivative_frozen_tensor_rev9 import (
    CONTRACT_PATH,
    CONTRACT_SHA256,
    FROZEN_RECEIPT_SCHEMA,
    GOAL_PATH,
    GOAL_REVISION,
    GOAL_SHA256,
    R274_SHA256,
    _build_base_model,
    _fixture,
    _tensor_raw_bytes,
    canonical_bytes,
    sha256_file,
    tensor_stream_sha256,
)


RECEIPT_SCHEMA = "poke_bot.alakazam_rule_derivative_blackwell_preflight_receipt/v1"
EVIDENCE_SCHEMA = "poke_bot.alakazam_rule_derivative_blackwell_preflight_evidence/v1"
EXPECTED_DEVICE = "cuda:1"
EXPECTED_DEVICE_MODEL = "NVIDIA RTX PRO 5000 Blackwell"


class BlackwellPreflightRev9Error(RuntimeError):
    pass


def _sha(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def _verify(path: Path, expected: str) -> None:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
        raise BlackwellPreflightRev9Error(f"artifact identity mismatch: {path}")


def _write_json(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise BlackwellPreflightRev9Error(f"output exists: {path}")
    body = canonical_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        offset = 0
        while offset < len(body):
            offset += os.write(descriptor, body[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return sha256_file(path)


def _source_manifest(repo_root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted((repo_root / "poke_bot").rglob("*.py")):
        if path.is_symlink() or not path.is_file():
            raise BlackwellPreflightRev9Error("runtime source contains a non-regular Python file")
        rows.append(
            {
                "path": str(path.relative_to(repo_root)),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    if not rows:
        raise BlackwellPreflightRev9Error("runtime source manifest is empty")
    return {"schema": "poke_bot.alakazam_rule_derivative_runtime_source_manifest/v1", "files": rows}


def _representation() -> dict[str, Any]:
    player = {
        "hand_count": 5,
        "deck_count": 45,
        "prize_count": 6,
        "bench": [],
        "effective_bench_maximum": 5,
    }
    selection = {
        "selection_type": "number",
        "min_count": 1,
        "max_count": 1,
        "remain_damage_counter": None,
        "remain_energy_cost": None,
    }
    options = []
    for number in (1, 2):
        options.append(
            {
                "semantic": {
                    "selection": selection,
                    "option": {
                        "option_type": "number",
                        "number": number,
                        "count": 1,
                    },
                },
                "referenced_card_ids": [],
                "referenced_attack_ids": [],
            }
        )
    return {
        "schema": "poke_bot.alakazam_public_rule_representation/v1",
        "state": {
            "turn": 2,
            "turn_action_count": 1,
            "supporter_played": False,
            "stadium_played": False,
            "energy_attached": False,
            "retreated": False,
            "players": {"acting": player, "opponent": dict(player)},
        },
        "selection": selection,
        "options": options,
    }


def _required_fields(contract: Mapping[str, Any]) -> set[str]:
    try:
        fields = contract["inzi_blackwell_training_handoff"]["receipt_contract"]["blackwell_preflight"]["required_fields"]
    except (KeyError, TypeError) as exc:
        raise BlackwellPreflightRev9Error("contract preflight receipt declaration is missing") from exc
    return set(fields)


def _driver_version() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    values = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    if len(values) != 1:
        raise BlackwellPreflightRev9Error("GPU driver versions are ambiguous")
    return next(iter(values))


def _device_uuid_for_pci_index(index: int) -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        raw_index, raw_uuid = [item.strip() for item in line.split(",", 1)]
        if int(raw_index) == index:
            return raw_uuid
    raise BlackwellPreflightRev9Error("cannot resolve Blackwell UUID")


def build_preflight(
    *,
    repo_root: Path,
    candidate_initialization_path: Path,
    candidate_initialization_sha256: str,
    frozen_tensor_receipt_path: Path,
    frozen_tensor_receipt_sha256: str,
    output_root: Path,
) -> dict[str, Any]:
    import torch
    from poke_bot.alakazam_rule_derivative_model_r298 import (
        R298PublicRuleSemanticProjection,
        R298SemanticProjectionConfig,
    )

    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise BlackwellPreflightRev9Error("CUDA_DEVICE_ORDER must be PCI_BUS_ID")
    _verify(repo_root / GOAL_PATH, GOAL_SHA256)
    _verify(repo_root / CONTRACT_PATH, CONTRACT_SHA256)
    _verify(candidate_initialization_path, candidate_initialization_sha256)
    _verify(frozen_tensor_receipt_path, frozen_tensor_receipt_sha256)
    contract = json.loads((repo_root / CONTRACT_PATH).read_text())
    frozen = json.loads(frozen_tensor_receipt_path.read_text())
    if (
        frozen.get("schema") != FROZEN_RECEIPT_SCHEMA
        or frozen.get("goal_contract_sha256") != CONTRACT_SHA256
        or frozen.get("goal_revision") != GOAL_REVISION
        or frozen.get("parent_checkpoint_sha256") != R274_SHA256
        or frozen.get("layer_off_bit_identical_baseline_logits") is not True
    ):
        raise BlackwellPreflightRev9Error("frozen-tensor receipt is not the sealed rev9 prerequisite")
    bundle = torch.load(candidate_initialization_path, map_location="cpu", weights_only=False)
    if (
        bundle.get("goal_contract_sha256") != CONTRACT_SHA256
        or bundle.get("r274_checkpoint_sha256") != R274_SHA256
        or bundle.get("eligible_trainable_branches") != ["public_rule_semantic_projection"]
        or bundle.get("training_eligible_before_activation") is not False
    ):
        raise BlackwellPreflightRev9Error("candidate initialization bundle is incompatible")
    if torch.cuda.device_count() < 2:
        raise BlackwellPreflightRev9Error("contract-bound cuda:1 is unavailable")
    device = torch.device(EXPECTED_DEVICE)
    properties = torch.cuda.get_device_properties(device)
    if properties.name != EXPECTED_DEVICE_MODEL or (properties.major, properties.minor) != (12, 0):
        raise BlackwellPreflightRev9Error("cuda:1 is not the contract-bound Blackwell")
    device_uuid = _device_uuid_for_pci_index(1)
    free_before, total_memory = torch.cuda.mem_get_info(device)
    torch.cuda.reset_peak_memory_stats(device)

    base_state = bundle["base_model_state_dict"]
    before_frozen_sha = tensor_stream_sha256(base_state)
    if before_frozen_sha != frozen["frozen_parameter_and_buffer_bytes_sha256_before"]:
        raise BlackwellPreflightRev9Error("candidate inherited bytes do not match frozen receipt")
    base_model = _build_base_model(bundle["model_config"], base_state).to(device)
    base_model.eval()
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)

    sidecar = R298PublicRuleSemanticProjection(
        R298SemanticProjectionConfig(**bundle["public_rule_semantic_projection_config"])
    ).to(device)
    sidecar.load_state_dict(bundle["public_rule_semantic_projection_state_dict"], strict=True)
    trainable = {name: parameter for name, parameter in sidecar.named_parameters()}
    expected_names = [name.removeprefix("public_rule_semantic_projection.") for name in frozen["trainable_parameter_names"]]
    if sorted(trainable) != sorted(expected_names):
        raise BlackwellPreflightRev9Error("sidecar trainable inventory drifted")
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=1e-3, weight_decay=0.0)
    optimizer_names = sorted(trainable)

    board, options, _fixture_payload = _fixture()
    with torch.no_grad():
        base_output = base_model.forward(board, options, n_options=[2])
    base_logits = base_output["policy_logits"]
    base_logits_before = base_logits.detach().clone()
    base_option_hidden = torch.zeros((1, 2, sidecar.d_model), dtype=base_logits.dtype, device=device)
    representations = [_representation()]
    armed_logits = sidecar.apply_to_logits(
        base_logits,
        base_option_hidden,
        representations,
        runtime_enabled=True,
        gate=torch.ones((), dtype=base_logits.dtype, device=device),
    )
    loss = torch.nn.functional.cross_entropy(armed_logits, torch.tensor([0], device=device))
    if not bool(torch.isfinite(loss)):
        raise BlackwellPreflightRev9Error("canary loss is nonfinite")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if any(parameter.grad is not None for parameter in base_model.parameters()):
        raise BlackwellPreflightRev9Error("frozen base received a gradient")
    grads = [parameter.grad for parameter in trainable.values() if parameter.grad is not None]
    if not grads or any(not bool(torch.isfinite(grad).all()) for grad in grads):
        raise BlackwellPreflightRev9Error("sidecar gradients are missing or nonfinite")
    if not any(float(grad.detach().abs().sum()) > 0.0 for grad in grads):
        raise BlackwellPreflightRev9Error("sidecar canary produced no nonzero gradient")
    sidecar_before = {name: value.detach().clone() for name, value in sidecar.state_dict().items()}
    optimizer.step()
    if not any(
        not torch.equal(sidecar_before[name], value.detach())
        for name, value in sidecar.state_dict().items()
    ):
        raise BlackwellPreflightRev9Error("optimizer canary did not update a sidecar tensor")
    off_logits = sidecar.apply_to_logits(
        base_logits,
        None,
        None,
        runtime_enabled=False,
        gate=0.0,
    )
    if off_logits is not base_logits or not torch.equal(off_logits, base_logits_before):
        raise BlackwellPreflightRev9Error("post-step layer-off logits changed")
    after_frozen_sha = tensor_stream_sha256(
        {name: value.detach().cpu() for name, value in base_model.state_dict().items()}
    )
    if after_frozen_sha != before_frozen_sha:
        raise BlackwellPreflightRev9Error("frozen base tensors changed during canary")
    torch.cuda.synchronize(device)
    peak_device = int(torch.cuda.max_memory_allocated(device))
    peak_host = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    del optimizer, sidecar, base_model
    torch.cuda.empty_cache()

    source_manifest = _source_manifest(repo_root)
    source_manifest_sha = _sha(source_manifest)
    python_environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nccl": list(torch.cuda.nccl.version()) if torch.cuda.nccl.is_available([]) else None,
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
    }
    python_environment_sha = _sha(python_environment)
    plan = {
        "batch_size": 1,
        "legal_option_count": 2,
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "optimizer_scope": "public_rule_semantic_projection_only",
        "corpus_loaded": False,
    }
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "goal_contract_path": CONTRACT_PATH,
        "goal_contract_sha256": CONTRACT_SHA256,
        "goal_revision": GOAL_REVISION,
        "host": "inzi",
        "device": EXPECTED_DEVICE,
        "device_model": EXPECTED_DEVICE_MODEL,
        "device_uuid": device_uuid,
        "compute_capability": [int(properties.major), int(properties.minor)],
        "driver_version": _driver_version(),
        "cuda_version": str(torch.version.cuda),
        "cudnn_version": str(torch.backends.cudnn.version()),
        "nccl_version": ".".join(str(item) for item in torch.cuda.nccl.version()),
        "torch_version": str(torch.__version__),
        "runtime_commit_sha256": source_manifest_sha,
        "python_environment_manifest_sha256": python_environment_sha,
        "device_total_memory_bytes": int(total_memory),
        "device_free_memory_bytes_before": int(free_before),
        "required_dtype_and_kernel_capabilities": {
            "float32": True,
            "linear": True,
            "gelu": True,
            "cross_entropy": True,
            "adamw": True,
        },
        "planned_batch_and_optimizer_config_sha256": _sha(plan),
        "load_canary_passed": True,
        "forward_backward_optimizer_canary_passed": True,
        "finite_loss_and_gradients": True,
        "frozen_tensor_gradient_norm_exact_zero": True,
        "frozen_parameters_absent_from_optimizer": True,
        "layer_off_parity_passed": True,
        "peak_device_memory_bytes": peak_device,
        "peak_host_ram_bytes": peak_host,
        "resource_headroom_passed": peak_device < int(total_memory) and peak_host < 96 * 1024**3,
        "no_runtime_or_service_change_performed": True,
        "validated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if set(receipt) != _required_fields(contract):
        raise BlackwellPreflightRev9Error("preflight receipt does not exactly match contract fields")
    if receipt["resource_headroom_passed"] is not True:
        raise BlackwellPreflightRev9Error("canary resource headroom failed")

    output_root.mkdir(parents=True, exist_ok=False)
    source_manifest_file_sha = _write_json(output_root / "runtime-source-manifest.json", source_manifest)
    environment_file_sha = _write_json(output_root / "python-environment-manifest.json", python_environment)
    plan_file_sha = _write_json(output_root / "planned-batch-and-optimizer.json", plan)
    receipt_sha = _write_json(output_root / "blackwell-preflight-receipt.json", receipt)
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "goal_revision": GOAL_REVISION,
        "candidate_initialization_sha256": candidate_initialization_sha256,
        "frozen_tensor_receipt_sha256": frozen_tensor_receipt_sha256,
        "runtime_source_manifest_payload_sha256": source_manifest_sha,
        "runtime_source_manifest_file_sha256": source_manifest_file_sha,
        "python_environment_manifest_payload_sha256": python_environment_sha,
        "python_environment_manifest_file_sha256": environment_file_sha,
        "planned_batch_and_optimizer_file_sha256": plan_file_sha,
        "blackwell_preflight_receipt_sha256": receipt_sha,
        "loss": float(loss.detach().cpu()),
        "sidecar_gradient_tensor_count": len(grads),
        "sidecar_optimizer_parameter_names": optimizer_names,
        "frozen_tensor_bytes_sha256_before_and_after": before_frozen_sha,
        "layer_off_logits_sha256": "sha256:" + hashlib.sha256(_tensor_raw_bytes(base_logits_before.detach().cpu())).hexdigest(),
        "service_control_performed": False,
        "training_corpus_loaded": False,
        "training_or_activation_performed": False,
    }
    evidence_sha = _write_json(output_root / "blackwell-preflight-evidence.json", evidence)
    complete = {
        "schema": "poke_bot.alakazam_rule_derivative_blackwell_preflight_completion/v1",
        "status": "passed_nonactivating_blackwell_canary",
        "receipt_sha256": receipt_sha,
        "evidence_sha256": evidence_sha,
        "service_control_performed": False,
        "training_or_activation_performed": False,
    }
    complete_sha = _write_json(output_root / "COMPLETE.json", complete)
    return {
        "output_root": str(output_root),
        "receipt_sha256": receipt_sha,
        "evidence_sha256": evidence_sha,
        "completion_sha256": complete_sha,
        "peak_device_memory_bytes": peak_device,
        "peak_host_ram_bytes": peak_host,
        "device_uuid": device_uuid,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--candidate-initialization", type=Path, required=True)
    parser.add_argument("--candidate-initialization-sha256", required=True)
    parser.add_argument("--frozen-tensor-receipt", type=Path, required=True)
    parser.add_argument("--frozen-tensor-receipt-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_preflight(
        repo_root=args.repo_root.resolve(),
        candidate_initialization_path=args.candidate_initialization.resolve(),
        candidate_initialization_sha256=args.candidate_initialization_sha256,
        frozen_tensor_receipt_path=args.frozen_tensor_receipt.resolve(),
        frozen_tensor_receipt_sha256=args.frozen_tensor_receipt_sha256,
        output_root=args.output_root.resolve(),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

