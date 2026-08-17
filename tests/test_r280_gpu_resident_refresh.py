from __future__ import annotations

import json
from pathlib import Path

import pytest

from poke_bot.r280_gpu_resident_refresh import (
    REFRESH_RECEIPT_SCHEMA,
    _file_identity,
    _semantic_digest,
    validate_refresh_receipt,
)


def test_r280_refresh_receipt_binds_parent_child_pack_and_gpu_path(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent.pt"
    child = tmp_path / "child.pt"
    receipt_path = tmp_path / "before_iter_00005.json"
    parent.write_bytes(b"parent")
    child.write_bytes(b"child")
    pack_sha256 = "sha256:" + "a" * 64
    receipt = {
        "schema": REFRESH_RECEIPT_SCHEMA,
        "status": "passed",
        "before_iteration": 5,
        "epochs": 5,
        "parent_digest": _file_identity(parent)["sha256"],
        "checkpoint_digest": _file_identity(child)["sha256"],
        "parent": _file_identity(parent),
        "checkpoint": _file_identity(child),
        "pack": {"sha256": pack_sha256},
        "pack_receipt": {},
        "counts": {"games": 26_704, "decisions": 2_040_911},
        "loss_weights": {},
        "rl_iteration_before_after": [4, 4],
        "gpu_residency": {
            "full_numeric_pack_resident": True,
            "device_side_batch_gather": True,
            "host_batch_streaming_used": False,
            "resident_python_objects_used": False,
        },
        "training_result": {},
        "receipt_sha256": None,
    }
    receipt["receipt_sha256"] = _semantic_digest(receipt)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")

    assert validate_refresh_receipt(
        receipt_path,
        expected_parent_sha256=receipt["parent"]["sha256"],
        expected_pack_sha256=pack_sha256,
        expected_before_iteration=5,
    )["checkpoint"] == receipt["checkpoint"]

    receipt["gpu_residency"]["host_batch_streaming_used"] = True
    receipt["receipt_sha256"] = _semantic_digest(receipt)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid"):
        validate_refresh_receipt(
            receipt_path,
            expected_parent_sha256=receipt["parent"]["sha256"],
            expected_pack_sha256=pack_sha256,
            expected_before_iteration=5,
        )
