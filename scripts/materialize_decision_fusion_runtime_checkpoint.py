#!/usr/bin/env python3
"""Create a serving-enabled child after the complete fusion audit passes."""

from __future__ import annotations

import argparse
import copy
import hashlib
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

from poke_bot import checkpoint  # noqa: E402
from poke_bot.model import (  # noqa: E402
    DECISION_FUSION_REQUIRED_HEADS,
    DECISION_FUSION_SCHEMA,
)
from poke_bot.train import load_model_from_checkpoint  # noqa: E402


SCHEMA = "poke_bot.causal_decision_fusion_runtime_materialization/v1"
VALIDATION_SCHEMA = "poke_bot.causal_decision_fusion_activation_validation/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _assert_nested_bit_identical(left: Any, right: Any, *, path: str) -> None:
    """Compare optimizer/checkpoint structures without tensor truth coercion."""

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise RuntimeError(f"{path} changed type during runtime activation")
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        return
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            raise RuntimeError(f"{path} changed type during runtime activation")
        if set(left) != set(right):
            raise RuntimeError(f"{path} changed keys during runtime activation")
        for key in left:
            _assert_nested_bit_identical(
                left[key], right[key], path=f"{path}.{key}"
            )
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if (
            not isinstance(left, (list, tuple))
            or not isinstance(right, (list, tuple))
            or type(left) is not type(right)
            or len(left) != len(right)
        ):
            raise RuntimeError(f"{path} changed sequence during runtime activation")
        for index, (left_value, right_value) in enumerate(zip(left, right)):
            _assert_nested_bit_identical(
                left_value, right_value, path=f"{path}[{index}]"
            )
        return
    if left != right:
        raise RuntimeError(f"{path} changed during runtime activation")


def materialize(
    *,
    trained: Path,
    validation_receipt: Path,
    output: Path,
    receipt: Path,
) -> dict[str, Any]:
    trained = trained.expanduser().resolve()
    validation_receipt = validation_receipt.expanduser().resolve()
    output = output.expanduser().resolve()
    receipt = receipt.expanduser().resolve()
    if output.exists() or receipt.exists():
        raise FileExistsError("runtime fusion outputs are immutable")
    trained_digest = checkpoint.checkpoint_digest(trained)
    validation = json.loads(validation_receipt.read_text(encoding="utf-8"))
    checks = dict(validation.get("checks") or {})
    if not (
        validation.get("schema") == VALIDATION_SCHEMA
        and validation.get("checkpoint_digest") == trained_digest
        and validation.get("passed") is True
        and checks.get("trained_nonzero_fusion") is True
        and checks.get("every_required_head_influence_nonzero") is True
        and checks.get("deterministic_local_remote_logit_parity") is True
        and checks.get("causal_information_audit") is True
        and checks.get("rollout_throughput_and_memory") is True
    ):
        raise RuntimeError("fusion activation validation is incomplete")

    checkpoint.assert_trusted_policy_checkpoint(trained)
    payload = checkpoint.load_checkpoint(trained, map_location="cpu")
    source_config = dict(payload.get("model_config") or {})
    if not (
        source_config.get("decision_fusion_enabled") is True
        and source_config.get("decision_fusion_runtime_enabled") is False
    ):
        raise RuntimeError("trained checkpoint is not a fusion warmup learner")
    source_state = dict(payload.get("model_state_dict") or {})
    final_weight = source_state.get("decision_fusion.residual.2.weight")
    if final_weight is None or not bool(torch.count_nonzero(final_weight).item()):
        raise RuntimeError("trained checkpoint has a zero fusion residual")

    runtime_payload = copy.deepcopy(payload)
    runtime_config = dict(source_config)
    runtime_config["decision_fusion_runtime_enabled"] = True
    runtime_payload["model_config"] = runtime_config
    extra = dict(runtime_payload.get("extra") or {})
    extra["decision_fusion_runtime_activation"] = {
        "schema": SCHEMA,
        "source_checkpoint": str(trained),
        "source_checkpoint_digest": trained_digest,
        "validation_receipt": str(validation_receipt),
        "validation_receipt_digest": _sha256(validation_receipt),
        "runtime_enabled": True,
        "required_heads": list(DECISION_FUSION_REQUIRED_HEADS),
    }
    runtime_payload["extra"] = extra
    provenance = dict(runtime_payload.get("provenance") or {})
    fusion = dict(provenance.get("decision_fusion") or {})
    fusion["runtime_enabled"] = True
    provenance["decision_fusion"] = fusion
    runtime_payload["provenance"] = provenance
    checkpoint.immutable_torch_save(runtime_payload, output)

    loaded = load_model_from_checkpoint(output, device=torch.device("cpu"))
    if not (
        loaded.decision_fusion_enabled
        and loaded.decision_fusion_runtime_enabled
        and loaded.decision_fusion_inventory()["schema"]
        == DECISION_FUSION_SCHEMA
        and loaded.decision_fusion_inventory()["required_heads"]
        == list(DECISION_FUSION_REQUIRED_HEADS)
    ):
        raise RuntimeError("runtime-enabled checkpoint failed reconstruction")
    output_payload = checkpoint.load_checkpoint(output, map_location="cpu")
    for key, value in source_state.items():
        torch.testing.assert_close(
            value, output_payload["model_state_dict"][key], rtol=0, atol=0
        )
    _assert_nested_bit_identical(
        payload.get("optimizer_state_dict"),
        output_payload.get("optimizer_state_dict"),
        path="optimizer_state_dict",
    )

    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(trained),
        "source_checkpoint_digest": trained_digest,
        "runtime_checkpoint": str(output),
        "runtime_checkpoint_digest": checkpoint.checkpoint_digest(output),
        "validation_receipt": str(validation_receipt),
        "validation_receipt_digest": _sha256(validation_receipt),
        "decision_fusion_schema": DECISION_FUSION_SCHEMA,
        "required_heads": list(DECISION_FUSION_REQUIRED_HEADS),
        "runtime_enabled": True,
        "model_tensors_bit_identical_to_validated_source": True,
        "optimizer_state_bit_identical_to_validated_source": True,
    }
    _exclusive_json(receipt, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--validation-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(**vars(args)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
