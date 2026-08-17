#!/usr/bin/env python3
"""Validate an isolated Alakazam action-critic checkpoint on held-out days.

The command performs no optimization and consumes only the sealed validation
days.  It replays the exact streaming complete-action join and emits a separate
JSON receipt; evaluation days, policy state, runtime paths, and search remain
untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # Running as ``python scripts/…``.
    from train_alakazam_action_critic import (  # type: ignore[import-not-found]
        CHECKPOINT_SCHEMA,
        RECEIPT_SCHEMA,
        VALIDATION_SCHEMA,
        ActionCriticTrainingError,
        _activation_metric_summary,
        _mapping,
        _require_sha256,
        _require_sidecar_module,
        _run_batches,
        _strict_sidecar_checkpoint_available,
        atomic_write_json,
        build_critic,
        iter_complete_action_examples,
        open_sealed_inputs,
        resolve_device,
        sha256_file,
    )
except ModuleNotFoundError:  # Imported by tests as ``scripts.…``.
    from scripts.train_alakazam_action_critic import (
        CHECKPOINT_SCHEMA,
        RECEIPT_SCHEMA,
        VALIDATION_SCHEMA,
        ActionCriticTrainingError,
        _activation_metric_summary,
        _mapping,
        _require_sha256,
        _require_sidecar_module,
        _run_batches,
        _strict_sidecar_checkpoint_available,
        atomic_write_json,
        build_critic,
        iter_complete_action_examples,
        open_sealed_inputs,
        resolve_device,
        sha256_file,
    )


def _read_checkpoint(path: Path, *, expected_sha256: str) -> Mapping[str, Any]:
    if sha256_file(path) != _require_sha256(expected_sha256, label="checkpoint SHA-256"):
        raise ActionCriticTrainingError("critic checkpoint digest mismatch")
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ActionCriticTrainingError("cannot load critic checkpoint") from exc
    if not isinstance(value, Mapping):
        raise ActionCriticTrainingError("critic checkpoint schema drifted")
    sidecar = _require_sidecar_module()
    strict_schema = str(getattr(sidecar, "ACTION_CRITIC_SIDECAR_CHECKPOINT_SCHEMA", ""))
    if value.get("schema") not in {CHECKPOINT_SCHEMA, strict_schema}:
        raise ActionCriticTrainingError("critic checkpoint schema drifted")
    return value


def _check_optional_training_receipt(
    args: argparse.Namespace,
    *,
    source_binding: Mapping[str, Any],
    checkpoint_sha256: str,
) -> dict[str, Any]:
    if args.training_receipt is None:
        return {"provided": False}
    path = Path(args.training_receipt).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ActionCriticTrainingError("training receipt must be a regular file")
    actual_sha = sha256_file(path)
    if args.training_receipt_sha256 and actual_sha != _require_sha256(
        args.training_receipt_sha256, label="training receipt SHA-256"
    ):
        raise ActionCriticTrainingError("training receipt digest mismatch")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ActionCriticTrainingError("training receipt JSON is invalid") from exc
    if not isinstance(receipt, Mapping) or receipt.get("schema") != RECEIPT_SCHEMA:
        raise ActionCriticTrainingError("training receipt schema drifted")
    if dict(receipt.get("source_binding") or {}) != dict(source_binding):
        raise ActionCriticTrainingError("training receipt source binding mismatch")
    if str(receipt.get("checkpoint_sha256") or "") != checkpoint_sha256:
        raise ActionCriticTrainingError("training receipt checkpoint binding mismatch")
    return {"provided": True, "path": str(path), "sha256": actual_sha}


def validate(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    if args.batch_size < 1 or args.max_validation_programs < 0:
        raise ActionCriticTrainingError("batch size or validation program limit is invalid")
    dataset, targets, split_days, source_binding = open_sealed_inputs(args)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint_sha = _require_sha256(args.checkpoint_sha256, label="checkpoint SHA-256")
    checkpoint = _read_checkpoint(checkpoint_path, expected_sha256=checkpoint_sha)
    sidecar = _require_sidecar_module()
    strict_schema = str(getattr(sidecar, "ACTION_CRITIC_SIDECAR_CHECKPOINT_SCHEMA", ""))
    strict = _strict_sidecar_checkpoint_available() and checkpoint.get("schema") == strict_schema
    if strict:
        metadata = _mapping(checkpoint.get("metadata"), label="checkpoint metadata")
        training_state = _mapping(checkpoint.get("training_state"), label="checkpoint training state")
        checkpoint_source = metadata.get("source_binding")
        checkpoint_split_days = training_state.get("split_days")
        config_payload = checkpoint.get("config")
    else:
        checkpoint_source = checkpoint.get("source_binding")
        checkpoint_split_days = checkpoint.get("split_days")
        config_payload = checkpoint.get("model_config")
    if dict(checkpoint_source or {}) != dict(source_binding):
        raise ActionCriticTrainingError("checkpoint source binding mismatch")
    if dict(checkpoint_split_days or {}) != dict(split_days):
        raise ActionCriticTrainingError("checkpoint split-day binding mismatch")
    model_config = _mapping(config_payload, label="checkpoint model config")
    model, rebuilt_config = build_critic(saved_config=model_config)
    try:
        if strict:
            sidecar.restore_action_critic_checkpoint(model, checkpoint)
        else:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except (KeyError, RuntimeError, ValueError) as exc:
        raise ActionCriticTrainingError("checkpoint model state is incompatible") from exc
    device = resolve_device(args.device)
    model.to(device=device, dtype=torch.float32)
    validation_metrics, _ = _run_batches(
        model,
        iter_complete_action_examples(
            dataset,
            targets,
            split="validation",
            max_programs=int(args.max_validation_programs),
        ),
        device=device,
        batch_size=int(args.batch_size),
        optimizer=None,
        grad_clip=0.0,
    )
    validation = validation_metrics.summary()
    training_receipt = _check_optional_training_receipt(
        args,
        source_binding=source_binding,
        checkpoint_sha256=checkpoint_sha,
    )
    completion = {
        "schema": VALIDATION_SCHEMA,
        "status": "passed_offline_validation_only",
        "source_binding": source_binding,
        "split_days": split_days,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_model_config": dict(model_config),
        "rebuilt_model_config": rebuilt_config,
        "validation": validation,
        "validation_gates": _activation_metric_summary(
            validation,
            full_validation=not bool(args.max_validation_programs),
        ),
        "training_receipt": training_receipt,
        "device": str(device),
        "fp32": True,
        "optimization_performed": False,
        "evaluation_split_consumed": False,
        "runtime_or_policy_attachment": False,
        "policy_model_state_dict_changed": False,
        "existing_value_head_changed": False,
        "existing_action_q_head_changed": False,
        "replay_weights_changed": False,
        "search_rtp_mcts_or_simulator_branching_used": False,
        "activation_eligible": False,
        "elapsed_seconds": time.time() - started,
        "completed_at_unix_seconds": time.time(),
    }
    output = Path(args.output_json).expanduser().resolve()
    receipt_sha = atomic_write_json(output, completion)
    result = {
        "receipt_path": str(output),
        "receipt_sha256": receipt_sha,
        "checkpoint_sha256": checkpoint_sha,
        "validation": validation,
    }
    print(json.dumps({"phase": "complete", **result}, sort_keys=True), flush=True)
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay-manifest", type=Path, required=True)
    parser.add_argument("--overlay-manifest-sha256", required=True)
    parser.add_argument("--base-pack-root", type=Path, required=True)
    parser.add_argument("--base-completion-sha256", required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--target-manifest-sha256", required=True)
    parser.add_argument("--target-view", type=Path, default=None)
    parser.add_argument("--target-view-sha256", default="")
    parser.add_argument("--contract", type=Path, default=ROOT / "goals/alakazam-elmo-rule-derivative/contract.json")
    parser.add_argument("--contract-sha256", default="")
    parser.add_argument("--training-view", type=Path, default=None)
    parser.add_argument("--training-view-sha256", default="")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--training-receipt", type=Path, default=None)
    parser.add_argument("--training-receipt-sha256", default="")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--max-validation-programs", type=int, default=0)
    parser.add_argument("--test-allow-noncanonical-split", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--test-skip-input-shard-sha256", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        validate(args)
    except (ActionCriticTrainingError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
