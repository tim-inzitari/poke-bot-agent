"""Unit coverage for r298 frozen-derivative materialization guards."""

from __future__ import annotations

import copy
import importlib.util

import pytest

from poke_bot.alakazam_rule_derivative_training_r298 import (
    REQUIRED_FROZEN_COMPONENTS,
    R298_CHECKPOINT_INSPECTION_SCHEMA,
    RuleDerivativeTrainingError,
    immutable_artifact_identity,
    build_frozen_tensor_audit,
    inspect_checkpoint_state_dict_create_only,
)


def _checkpoint(byte: str) -> dict[str, object]:
    return {"sha256": "sha256:" + byte * 64, "size_bytes": 1}


def _component_map() -> dict[str, list[str]]:
    return {component: [f"frozen.{index}"] for index, component in enumerate(REQUIRED_FROZEN_COMPONENTS)}


def _parent_state() -> dict[str, bytes]:
    return {f"frozen.{index}": bytes([index]) for index, _component in enumerate(REQUIRED_FROZEN_COMPONENTS)}


def test_frozen_tensor_audit_requires_every_parent_tensor_and_allows_only_declared_sidecar() -> None:
    parent = _parent_state()
    candidate = {**parent, "r298_sidecar.final.weight": b"new"}
    audit = build_frozen_tensor_audit(
        baseline_state_dict=parent,
        candidate_state_dict=candidate,
        baseline_checkpoint=_checkpoint("a"),
        candidate_checkpoint=_checkpoint("b"),
        frozen_component_tensor_names=_component_map(),
        trainable_new_tensor_names=["r298_sidecar.final.weight"],
    )
    assert audit["inventory_complete"] is True
    assert audit["changed_protected_tensor_names"] == []
    assert audit["candidate_new_trainable_tensor_names"] == ["r298_sidecar.final.weight"]


def test_frozen_tensor_audit_rejects_any_parent_bit_change() -> None:
    parent = _parent_state()
    candidate = copy.deepcopy(parent)
    candidate["frozen.0"] = b"changed"
    with pytest.raises(RuleDerivativeTrainingError, match="modifies protected tensors"):
        build_frozen_tensor_audit(
            baseline_state_dict=parent,
            candidate_state_dict=candidate,
            baseline_checkpoint=_checkpoint("a"),
            candidate_checkpoint=_checkpoint("b"),
            frozen_component_tensor_names=_component_map(),
            trainable_new_tensor_names=[],
        )


def test_safe_checkpoint_inspection_fails_closed_without_torch(tmp_path) -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("this assertion covers the CPU-only audit-host fallback")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not a checkpoint")
    # The local static-analysis environment intentionally has no torch.  The
    # inspection must not fall back to unsafe pickle loading merely to pass.
    with pytest.raises(RuleDerivativeTrainingError, match="safe checkpoint tensor inspection requires torch"):
        inspect_checkpoint_state_dict_create_only(
            checkpoint,
            trusted_roots=[tmp_path],
            checkpoint_role="candidate",
        )


def test_candidate_checkpoint_artifact_cannot_escape_declared_trusted_root(tmp_path) -> None:
    trusted_root = tmp_path / "trusted"
    trusted_root.mkdir()
    outside_checkpoint = tmp_path / "outside.pt"
    outside_checkpoint.write_bytes(b"immutable-looking-but-foreign")

    with pytest.raises(RuleDerivativeTrainingError, match="outside the declared trusted artifact roots"):
        immutable_artifact_identity(
            outside_checkpoint,
            trusted_roots=[trusted_root],
            field="candidate checkpoint",
        )


def test_reopened_checkpoint_identity_detects_swapped_bytes(tmp_path) -> None:
    trusted_root = tmp_path / "trusted"
    trusted_root.mkdir()
    checkpoint = trusted_root / "candidate.pt"
    checkpoint.write_bytes(b"first checkpoint")
    first = immutable_artifact_identity(
        checkpoint, trusted_roots=[trusted_root], field="candidate checkpoint"
    )
    checkpoint.write_bytes(b"second checkpoint")
    second = immutable_artifact_identity(
        checkpoint, trusted_roots=[trusted_root], field="candidate checkpoint"
    )

    assert first["sha256"] != second["sha256"]
