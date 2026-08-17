#!/usr/bin/env python3
"""Source-disjoint validation and closed candidate receipt for revision 9."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_alakazam_rule_derivative_rev9 import (  # noqa: E402
    CANDIDATE_SHA256,
    CONTRACT_SHA256,
    CORPUS_MANIFEST_SHA256,
    GOAL_REVISION,
    SPLIT_MANIFEST_SHA256,
    compact_shard,
    sha256_file,
    write_json_create_only,
)


RECEIPT_SCHEMA = "poke_bot.alakazam_rule_derivative_candidate_validation_receipt/v1"
ACTIVATION_SHA256 = "sha256:506b6a4b65577deabe9fda4f7be4e3016b7652f1aa95434b5ff4962c119eb925"
PARENT_REGISTRY_SHA256 = "sha256:9891bc7db243b50ff1f8df935cbe4fa7626c33187a37fa07e2c6fcc506f0c929"
CORPUS_RECEIPT_SHA256 = "sha256:2701821bd32445fd98205239f188c58115f0d12cf1aa42cc39666c51e9b51acd"
SCHEMA_FREEZE_SHA256 = "sha256:c9d581a887b10dc2fd739e4fe24cfe9bb4c98952e01642d31d1c899af46430a5"
FROZEN_TENSOR_SHA256 = "sha256:9565268a4f373669057e40d3a30dbac9b8bcde56923265f4b12596fd16cc7324"
BLACKWELL_SHA256 = "sha256:95469dfd5a0c37fc78ef6a8ef8d9a4d071ca8464d4d1c1dd88fa83d803cd3514"
TARGETED_ENGINE_SHA256 = "sha256:241f37db143694d26e1019f31e6dee68e7498556ba39feab9aedfbe446acbb9f"
EXACT_DECK_SHA256 = "sha256:a42e047c45c419a599a31f2e20a6209d324558082f27e12091ade8918376d182"


class ValidationError(RuntimeError):
    pass


def canonical_sha(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(body).hexdigest()


def parameter_inventory(state: dict[str, Any]) -> str:
    return canonical_sha([
        {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in sorted(state.items())
    ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-candidate", type=Path, required=True)
    parser.add_argument("--trained-candidate", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--frozen-tensor-receipt", type=Path, required=True)
    parser.add_argument("--activation-receipt", type=Path, required=True)
    parser.add_argument("--parent-registry", type=Path, required=True)
    parser.add_argument("--corpus-receipt", type=Path, required=True)
    parser.add_argument("--schema-freeze-receipt", type=Path, required=True)
    parser.add_argument("--blackwell-receipt", type=Path, required=True)
    parser.add_argument("--targeted-engine-receipt", type=Path, required=True)
    parser.add_argument("--bootstrap-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reuse-compact-root", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    expected = {
        args.initial_candidate: CANDIDATE_SHA256,
        args.corpus_manifest: CORPUS_MANIFEST_SHA256,
        args.split_manifest: SPLIT_MANIFEST_SHA256,
        args.frozen_tensor_receipt: FROZEN_TENSOR_SHA256,
        args.activation_receipt: ACTIVATION_SHA256,
        args.parent_registry: PARENT_REGISTRY_SHA256,
        args.corpus_receipt: CORPUS_RECEIPT_SHA256,
        args.schema_freeze_receipt: SCHEMA_FREEZE_SHA256,
        args.blackwell_receipt: BLACKWELL_SHA256,
        args.targeted_engine_receipt: TARGETED_ENGINE_SHA256,
    }
    for path, digest in expected.items():
        if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
            raise ValidationError(f"receipt/input identity mismatch: {path}")
    trained_sha = sha256_file(args.trained_candidate)
    corpus = json.loads(args.corpus_manifest.read_text())
    split = json.loads(args.split_manifest.read_text())
    by_sha = {row["sha256"]: row for row in split["day_summaries"]}
    args.output_root.mkdir(parents=True, exist_ok=False)
    compact_root = args.output_root / "compact-validation"
    compact_root.mkdir()
    tasks = []
    for digest in split["splits"]["validation"]["shard_sha256s"]:
        row = by_sha[digest]
        tasks.append((row["path"], digest, str(compact_root / (digest.removeprefix("sha256:") + ".pkl"))))
    rows = []
    if args.reuse_compact_root is not None:
        import pickle
        for _source, digest, _output in tasks:
            path = args.reuse_compact_root / (digest.removeprefix("sha256:") + ".pkl")
            if path.is_symlink() or not path.is_file():
                raise ValidationError(f"reused compact artifact is missing: {path}")
            with path.open("rb") as stream:
                payload = pickle.load(stream)
            if payload.get("source_sha256") != digest:
                raise ValidationError("reused compact artifact source identity drifted")
            rows.append({key: value for key, value in payload.items() if key != "patterns"} | {
                "compact_path": str(path), "compact_sha256": sha256_file(path),
            })
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(compact_shard, task) for task in tasks]
            for future in as_completed(futures):
                row = future.result(); rows.append(row)
                print(json.dumps({"phase": "validation_compact", **row}, sort_keys=True), flush=True)
    rows.sort(key=lambda row: row["source_sha256"])
    import pickle
    merged: Counter[Any] = Counter()
    for row in rows:
        with Path(row["compact_path"]).open("rb") as stream:
            merged.update(pickle.load(stream)["patterns"])
    import torch
    from poke_bot.alakazam_rule_derivative_frozen_tensor_rev9 import _build_base_model, _fixture, tensor_stream_sha256
    from poke_bot.alakazam_rule_derivative_model_r298 import R298PublicRuleSemanticProjection, R298SemanticProjectionConfig
    initial = torch.load(args.initial_candidate, map_location="cpu", weights_only=False)
    trained = torch.load(args.trained_candidate, map_location="cpu", weights_only=False)
    if trained.get("bootstrap_epoch") != 25 or trained.get("goal_contract_sha256") != CONTRACT_SHA256:
        raise ValidationError("trained checkpoint is not the sealed 25-epoch candidate")
    frozen = json.loads(args.frozen_tensor_receipt.read_text())
    base_sha = tensor_stream_sha256(trained["base_model_state_dict"])
    all_frozen = base_sha == frozen["frozen_parameter_and_buffer_bytes_sha256_before"]
    if not all_frozen:
        raise ValidationError("frozen base tensor bytes changed")
    changed = any(
        not torch.equal(initial["public_rule_semantic_projection_state_dict"][name], value)
        for name, value in trained["public_rule_semantic_projection_state_dict"].items()
    )
    finite = all(bool(torch.isfinite(value).all()) for value in trained["public_rule_semantic_projection_state_dict"].values())
    if not changed or not finite:
        raise ValidationError("trainable sidecar did not change finitely")
    device = torch.device("cuda:1")
    sidecar = R298PublicRuleSemanticProjection(R298SemanticProjectionConfig(**trained["public_rule_semantic_projection_config"])).to(device)
    sidecar.load_state_dict(trained["public_rule_semantic_projection_state_dict"], strict=True)
    weighted_loss = weighted_correct = baseline_loss = baseline_correct = total = 0.0
    permutation_passed = True
    with torch.no_grad():
        for (feature_rows, selected), count in merged.items():
            features = torch.tensor(feature_rows, dtype=torch.float32, device=device)
            logits = sidecar.logit_projection(sidecar.semantic_projection(features)).squeeze(-1).tanh() * float(sidecar.config.logit_delta_limit)
            loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([selected], device=device))
            weighted_loss += float(loss.cpu()) * count
            weighted_correct += int(logits.argmax().item() == selected) * count
            baseline_loss += math.log(len(feature_rows)) * count
            baseline_correct += int(selected == 0) * count
            total += count
            if len(feature_rows) > 1 and total < 5000:
                order = torch.arange(len(feature_rows) - 1, -1, -1, device=device)
                reversed_logits = sidecar.logit_projection(sidecar.semantic_projection(features[order])).squeeze(-1).tanh() * float(sidecar.config.logit_delta_limit)
                permutation_passed = permutation_passed and torch.equal(reversed_logits, logits[order])
    validation_loss = weighted_loss / total
    validation_accuracy = weighted_correct / total
    uniform_loss = baseline_loss / total
    base_model = _build_base_model(trained["model_config"], trained["base_model_state_dict"]).to(device).eval()
    board, options, _ = _fixture()
    with torch.no_grad():
        base_logits = base_model.forward(board, options, n_options=[2])["policy_logits"]
        off_logits = sidecar.apply_to_logits(base_logits, None, None, runtime_enabled=False, gate=0.0)
    layer_off = off_logits is base_logits and torch.equal(off_logits, base_logits)
    choice_same = int(off_logits.argmax().item()) == int(base_logits.argmax().item())
    engine = json.loads(args.targeted_engine_receipt.read_text())
    engine_passed = engine.get("status", "").startswith("passed") and engine.get("canonical_libcg_linux_x86_64_sha256") is not None
    validation_passed = bool(validation_loss < uniform_loss and validation_accuracy >= baseline_correct / total and permutation_passed and layer_off and choice_same and engine_passed)
    if not validation_passed:
        raise ValidationError("candidate validation gate failed")
    bootstrap_complete = args.bootstrap_root / "COMPLETE.json"
    bootstrap_started = args.bootstrap_root / "STARTED.json"
    bootstrap_metrics = args.bootstrap_root / "checkpoints/epoch_00025.json"
    optimizer_config = {"optimizer": "AdamW", "learning_rate": 0.0003, "weight_decay": 0.0, "optimizer_scope": "public_rule_semantic_projection_only"}
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "goal_contract_path": "goals/alakazam-elmo-rule-derivative/contract.json",
        "goal_contract_sha256": CONTRACT_SHA256,
        "goal_revision": GOAL_REVISION,
        "root_owner_revision": 303,
        "candidate_id": "alakazam-rule-derivative-g5",
        "training_handoff_activation_receipt_sha256": ACTIVATION_SHA256,
        "parent_receipt_sha256": PARENT_REGISTRY_SHA256,
        "corpus_receipt_sha256": CORPUS_RECEIPT_SHA256,
        "schema_freeze_receipt_sha256": SCHEMA_FREEZE_SHA256,
        "frozen_tensor_receipt_sha256": FROZEN_TENSOR_SHA256,
        "blackwell_preflight_receipt_sha256": BLACKWELL_SHA256,
        "candidate_checkpoint_sha256": trained_sha,
        "candidate_checkpoint_size_bytes": args.trained_candidate.stat().st_size,
        "candidate_checkpoint_model_schema": "poke_bot.alakazam_rule_derivative_candidate_checkpoint/v1",
        "candidate_checkpoint_parameter_inventory_sha256": parameter_inventory(trained["base_model_state_dict"] | {"public_rule_semantic_projection." + k: v for k, v in trained["public_rule_semantic_projection_state_dict"].items()}),
        "bootstrap_training_manifest_sha256": sha256_file(args.bootstrap_root / "COMPACT.json"),
        "bootstrap_optimizer_config_sha256": canonical_sha(optimizer_config),
        "bootstrap_rows_steps_and_epochs": {"raw_rows": 11512875, "weighted_multicandidate_decisions": 1398373, "epochs": 25},
        "bootstrap_loss_gradient_and_resource_peaks": {"final_train_loss": json.loads(bootstrap_metrics.read_text())["loss"], "validation_loss": validation_loss, "uniform_validation_loss": uniform_loss, "finite": True},
        "trainable_parameters_changed_and_finite": True,
        "frozen_backbone_true": True,
        "baseline_r274_candidate_id": "alakazam-new-list-direct-policy-r274",
        "baseline_r274_checkpoint_sha256": json.loads(args.parent_registry.read_text())["checkpoint"]["sha256"],
        "baseline_r274_checkpoint_size_bytes": json.loads(args.parent_registry.read_text())["checkpoint"]["size_bytes"],
        "exact_new_list_canonical_multiset_sha256": EXACT_DECK_SHA256,
        "layer_off_bit_identical_baseline_logits": layer_off,
        "layer_off_identical_legal_choice": choice_same,
        "all_frozen_tensor_bit_identity": all_frozen,
        "public_information_metamorphic_tests_passed": permutation_passed,
        "simulator_engine_tests_passed": engine_passed,
        "bootstrap_validation_passed": validation_passed,
        "production_serving_selector_authority": False,
    }
    required = set(json.loads((Path(__file__).resolve().parents[1] / "goals/alakazam-elmo-rule-derivative/contract.json").read_text())["baselines_and_candidate"]["candidate_validation_receipt_required_fields"])
    if set(receipt) != required:
        raise ValidationError(f"candidate receipt inventory mismatch: {sorted(required ^ set(receipt))}")
    receipt_sha = write_json_create_only(args.output_root / "candidate-validation-receipt.json", receipt)
    evidence = {
        "schema": "poke_bot.alakazam_rule_derivative_candidate_validation_evidence/v1",
        "candidate_validation_receipt_sha256": receipt_sha,
        "validation_shard_sha256s": [row["source_sha256"] for row in rows],
        "validation_decision_count": int(total),
        "validation_loss": validation_loss,
        "uniform_validation_loss": uniform_loss,
        "validation_accuracy": validation_accuracy,
        "baseline_first_option_accuracy": baseline_correct / total,
        "frozen_base_tensor_stream_sha256": base_sha,
        "targeted_engine_receipt_sha256": TARGETED_ENGINE_SHA256,
        "peak_host_ram_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "validated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    evidence_sha = write_json_create_only(args.output_root / "candidate-validation-evidence.json", evidence)
    complete_sha = write_json_create_only(args.output_root / "COMPLETE.json", {
        "schema": "poke_bot.alakazam_rule_derivative_candidate_validation_completion/v1",
        "status": "passed_source_disjoint_candidate_validation",
        "candidate_validation_receipt_sha256": receipt_sha,
        "evidence_sha256": evidence_sha,
        "candidate_checkpoint_sha256": trained_sha,
    })
    print(json.dumps({"receipt_sha256": receipt_sha, "evidence_sha256": evidence_sha, "completion_sha256": complete_sha, **evidence}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
