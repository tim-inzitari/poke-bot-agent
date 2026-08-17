from __future__ import annotations

import json
from pathlib import Path

import pytest

from poke_bot import r274_bootstrap_handoff as handoff


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_full_manifest_and_shadow_overlay_are_exact(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    rows = []
    remaining = handoff.EXPECTED_EXPERT_DECISIONS
    for index, date in enumerate(handoff.EXPECTED_DATES):
        count = remaining if index == 0 else 0
        remaining -= count
        rows.append(
            {
                "date": date,
                "matching_decisions": count,
                "source_archive_validated": True,
                "source_feature_validated": True,
            }
        )
    _write_json(manifest, {"source_days": rows})
    observed = handoff.validate_full_expert_manifest(manifest)
    assert observed["days"] == 20
    assert observed["decisions"] == 2_040_911

    overlay = tmp_path / "overlay.json"
    overlay_rows = [
        {
            "episode_id": f"ep-{index}",
            "seat": index % 2,
            "env_step": index,
            "observation_fingerprint": f"sha256:{index:064x}",
        }
        for index in range(1_024)
    ]
    _write_json(
        overlay,
        {
            "mode": "shadow_only",
            "planner_dispatch_authority": False,
            "roots": len(overlay_rows),
            "rows": overlay_rows,
        },
    )
    assert handoff.validate_tactical_overlay(overlay)["roots"] == 1_024

    rows[0]["matching_decisions"] -= 1
    _write_json(manifest, {"source_days": rows})
    with pytest.raises(handoff.R274BootstrapHandoffError, match="2,040,911"):
        handoff.validate_full_expert_manifest(manifest)


def test_handoff_receipt_is_checksum_and_checkpoint_bound(tmp_path: Path) -> None:
    checkpoint = tmp_path / "activated.pt"
    checkpoint.write_bytes(b"activated checkpoint")
    checkpoint_identity = handoff.file_identity(checkpoint)
    receipt = {
        "schema": handoff.TACTICAL_ACTIVATION_SCHEMA,
        "status": "passed_ready_for_rl_update_0",
        "bootstrap_submission_receipt": {"path": "/receipt", "sha256": "sha256:" + "1" * 64, "size_bytes": 1},
        "bootstrap_submission_checkpoint": {"path": "/submission", "sha256": "sha256:" + "2" * 64, "size_bytes": 2},
        "bootstrap_upload_receipt": {"path": "/upload", "sha256": "sha256:" + "3" * 64, "size_bytes": 3},
        "bootstrap_remote_submission_id": 123,
        "activated_checkpoint": checkpoint_identity,
        "runtime_gates": {
            **{name: True for name in handoff.RUNTIME_GATE_FIELDS},
            **{name: True for name in handoff.TACTICAL_ROUTE_FIELDS},
        },
        "impact": {},
        "direct_policy_only": True,
        "rtp_enabled": False,
        "planner_dispatch_authority": False,
    }
    receipt["receipt_sha256"] = handoff.receipt_digest(receipt)
    receipt_path = tmp_path / "handoff.json"
    _write_json(receipt_path, receipt)
    validated = handoff.validate_handoff_receipt(
        receipt_path, expected_initial_checkpoint=checkpoint
    )
    assert validated["activated_checkpoint"] == checkpoint_identity

    checkpoint.write_bytes(b"tampered")
    with pytest.raises(handoff.R274BootstrapHandoffError, match="identity drifted"):
        handoff.validate_handoff_receipt(receipt_path)


def test_r280_gpu_result_is_checksum_and_output_bound(tmp_path: Path) -> None:
    output = tmp_path / "r280-bootstrap.pt"
    output.write_bytes(b"gpu bootstrap")
    result = {
        "schema": handoff.R280_GPU_BOOTSTRAP_SCHEMA,
        "owner_revision": 280,
        "status": "completed",
        "epochs_completed": 25,
        "pack": {
            "path": "/sealed/alakazam-expert-r279.pt",
            "sha256": "sha256:" + "1" * 64,
            "size_bytes": 5_725_073_070,
        },
        "output": handoff.file_identity(output),
        "counts": {
            "games": 26_704,
            "decisions": 2_040_911,
            "samples": 2_311_725,
            "options": 18_162_729,
        },
        "gpu_residency": {
            "device": "cuda:1",
            "pack_tensor_bytes": 5_725_073_070,
            "free_before_bytes": 1,
            "free_after_pack_bytes": 1,
            "full_numeric_pack_resident": True,
            "device_side_batch_gather": True,
            "pinned_cpu_fallback_used": False,
        },
        "training_result": {},
        "elapsed_seconds": 1.0,
    }
    result["receipt_sha256"] = handoff.receipt_digest(result)
    receipt = tmp_path / "r280-result.json"
    _write_json(receipt, result)

    validated = handoff.validate_r280_gpu_bootstrap_result(
        receipt, expected_bootstrap_checkpoint=output
    )
    assert validated["counts"]["decisions"] == 2_040_911

    result["counts"]["decisions"] -= 1
    result["receipt_sha256"] = handoff.receipt_digest(result)
    _write_json(receipt, result)
    with pytest.raises(handoff.R274BootstrapHandoffError, match="invalid"):
        handoff.validate_r280_gpu_bootstrap_result(
            receipt, expected_bootstrap_checkpoint=output
        )
