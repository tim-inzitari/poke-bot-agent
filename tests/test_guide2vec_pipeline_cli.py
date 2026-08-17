"""Focused coverage for the inert r226 Guide2Vec pipeline command."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from poke_bot.guide2vec_contracts import (
    CAUSAL_SPLIT_SCHEMA,
    COMPATIBILITY_FIELDS,
    DIGEST_TENSOR_LAYOUT,
    FROZEN_LATENT_STORE_SCHEMA,
    FROZEN_LATENT_TENSORS,
    GUIDE_LABEL_FIELDS,
    GUIDE_LABEL_OVERLAY_SCHEMA,
    INERT_AUTHORITY_FIELDS,
    STAGE_INDEX_SCHEMA,
    STAGE_JOIN_KEY_FIELDS,
    TRAINING_INPUT_FIELDS,
    TRAINING_INPUT_IDENTITY_FIELDS,
    TRAINING_MANIFEST_SCHEMA,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/guide2vec_pipeline.py"
TEMPLATE = ROOT / "config/guide2vec/dormant-training-manifest-v1.json"


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: object) -> str:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _false_authority() -> dict[str, bool]:
    return {field: False for field in INERT_AUTHORITY_FIELDS}


def _build_ready_manifest(tmp_path: Path) -> Path:
    def content_ref(name: str) -> dict[str, str]:
        path = tmp_path / name
        path.write_bytes(f"guide2vec CLI fixture: {name}\n".encode())
        return {
            "path": name,
            "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    stage_path = tmp_path / "stage.json"
    stage_digest = _write_json(
        stage_path,
        {
            "schema": STAGE_INDEX_SCHEMA,
            "source_identity_sha256": _digest("stage-source"),
            "legal_option_key_sha256": _digest("legal-options"),
            "stage_join_key_fields": list(STAGE_JOIN_KEY_FIELDS),
            "digest_tensor_layout": dict(DIGEST_TENSOR_LAYOUT),
            "stage_count": 3,
            "stage_rows": content_ref("stage-rows.pt"),
            "authority": _false_authority(),
        },
    )
    compatibility = {
        "frozen_base_identity_sha256": _digest("base"),
        "latent_abi_sha256": _digest("latent-abi"),
        "adapter_runtime_context_sha256": _digest("adapter"),
        "extractor_code_sha256": _digest("extractor"),
        "dtype_policy_sha256": _digest("dtype"),
        "stage_index_sha256": stage_digest,
        "legal_option_key_sha256": _digest("legal-options"),
    }
    stage_ref = {"path": "stage.json", "sha256": stage_digest}

    latent_path = tmp_path / "latent.json"
    latent_digest = _write_json(
        latent_path,
        {
            "schema": FROZEN_LATENT_STORE_SCHEMA,
            "compatibility": compatibility,
            "stage_join_key_fields": list(STAGE_JOIN_KEY_FIELDS),
            "digest_tensor_layout": dict(DIGEST_TENSOR_LAYOUT),
            "stage_index_manifest": stage_ref,
            "stage_count": 3,
            "latent_payload": content_ref("latents.pt"),
            "tensors": list(FROZEN_LATENT_TENSORS),
            "guide_labels_embedded": False,
            "authority": _false_authority(),
        },
    )
    overlay_path = tmp_path / "overlay.json"
    overlay_digest = _write_json(
        overlay_path,
        {
            "schema": GUIDE_LABEL_OVERLAY_SCHEMA,
            "compatibility": compatibility,
            "stage_join_key_fields": list(STAGE_JOIN_KEY_FIELDS),
            "digest_tensor_layout": dict(DIGEST_TENSOR_LAYOUT),
            "stage_index_manifest": stage_ref,
            "stage_count": 3,
            "label_payload": content_ref("labels.pt"),
            "label_fields": list(GUIDE_LABEL_FIELDS),
            "guide_contract": content_ref("guide-contract.json"),
            "teacher_semantics": content_ref("teacher-semantics.json"),
            "teacher_dependencies": [content_ref("teacher-dependency.json")],
            "authority": _false_authority(),
        },
    )
    split_path = tmp_path / "split.json"
    split_digest = _write_json(
        split_path,
        {
            "schema": CAUSAL_SPLIT_SCHEMA,
            "compatibility": compatibility,
            "stage_join_key_fields": list(STAGE_JOIN_KEY_FIELDS),
            "digest_tensor_layout": dict(DIGEST_TENSOR_LAYOUT),
            "stage_index_manifest": stage_ref,
            "stage_count": 3,
            "partition_payload": content_ref("partitions.pt"),
            "partition_guards": {
                "whole_episode_disjoint": True,
                "whole_source_day_disjoint": True,
                "heldout_sealed_until_candidate_selection": True,
            },
            "authority": _false_authority(),
        },
    )
    manifest_path = tmp_path / "training.json"
    _write_json(
        manifest_path,
        {
            "schema": TRAINING_MANIFEST_SCHEMA,
            "status": "ready",
            "compatibility": compatibility,
            "stage_join_key_fields": list(STAGE_JOIN_KEY_FIELDS),
            "digest_tensor_layout": dict(DIGEST_TENSOR_LAYOUT),
            "stage_count": 3,
            "inputs": {
                "stage_index_manifest": stage_ref,
                "frozen_latent_store_manifest": {
                    "path": "latent.json",
                    "sha256": latent_digest,
                },
                "guide_label_overlay_manifest": {
                    "path": "overlay.json",
                    "sha256": overlay_digest,
                },
                "causal_split_manifest": {"path": "split.json", "sha256": split_digest},
            },
            "input_identities": {
                "stage_index_sha256": stage_digest,
                "frozen_latent_store_sha256": latent_digest,
                "guide_label_overlay_sha256": overlay_digest,
                "causal_split_sha256": split_digest,
            },
            "authority": _false_authority(),
            "activation_receipt": None,
        },
    )
    return manifest_path


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=False,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


def test_dormant_template_checks_without_training_or_runtime_authority() -> None:
    result = _run("check", "--manifest", str(TEMPLATE))

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["command"] == "check"
    assert payload["manifest_status"] == "dormant"
    assert payload["content_refs_sealed"] is False
    assert payload["split_semantics_proven"] is False
    assert payload["data_ready"] is False
    assert payload["training_authorized_by_manifest"] is False
    assert payload["separate_activation_receipt_required"] is True
    assert all(value is False for value in payload["authority"].values())


def test_preflight_rejects_dormant_manifest_before_receipt_is_read(tmp_path: Path) -> None:
    missing_receipt = tmp_path / "must-not-be-opened.json"
    result = _run(
        "preflight",
        "--manifest",
        str(TEMPLATE),
        "--activation-receipt",
        str(missing_receipt),
    )

    assert result.returncode == 2
    payload = json.loads(result.stderr)
    assert payload["command"] == "preflight"
    assert payload["training_executed"] is False
    assert all(value is False for value in payload["authority"].values())
    assert "semantic split proof" in payload["error"]
    assert not missing_receipt.exists()


def test_content_sealed_manifest_is_reported_but_cannot_preflight(tmp_path: Path) -> None:
    manifest = _build_ready_manifest(tmp_path)
    check = _run("check", "--manifest", str(manifest))
    assert check.returncode == 0, check.stderr
    checked = json.loads(check.stdout)
    assert checked["status"] == "manifest_content_sealed_but_not_semantically_ready"
    assert checked["content_refs_sealed"] is True
    assert checked["split_semantics_proven"] is False
    assert checked["data_ready"] is False
    assert checked["blocking_reasons"]

    missing_receipt = tmp_path / "must-not-be-opened.json"
    preflight = _run(
        "preflight",
        "--manifest",
        str(manifest),
        "--activation-receipt",
        str(missing_receipt),
    )
    assert preflight.returncode == 2
    rejected = json.loads(preflight.stderr)
    assert rejected["training_executed"] is False
    assert all(value is False for value in rejected["authority"].values())
    assert "semantic split proof" in rejected["error"]
    assert not missing_receipt.exists()


def test_cli_contains_no_trainer_or_execution_import() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "from poke_bot.guide2vec_training" not in source
    assert "import poke_bot.guide2vec_training" not in source
    assert "train_from_joined_chunks" not in source
    assert 'add_parser("train"' not in source
    assert '"preflight"' in source


def test_template_has_exact_dormant_null_bindings() -> None:
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    assert template["schema"] == TRAINING_MANIFEST_SCHEMA
    assert template["status"] == "dormant"
    assert tuple(template["compatibility"]) == COMPATIBILITY_FIELDS
    assert all(template["compatibility"][field] is None for field in COMPATIBILITY_FIELDS)
    assert tuple(template["inputs"]) == TRAINING_INPUT_FIELDS
    assert all(template["inputs"][field] is None for field in TRAINING_INPUT_FIELDS)
    assert tuple(template["input_identities"]) == TRAINING_INPUT_IDENTITY_FIELDS
    assert all(
        template["input_identities"][field] is None
        for field in TRAINING_INPUT_IDENTITY_FIELDS
    )
    assert template["activation_receipt"] is None
    assert all(value is False for value in template["authority"].values())
