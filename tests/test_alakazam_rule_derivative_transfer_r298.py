"""Regression tests for the r298 quarantined-shard transfer fence."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from poke_bot.alakazam_rule_derivative_training_r298 import (
    R298_CONTRACT_PATH,
    R298_CONTRACT_SHA256,
    R298_GOAL_SHA256,
    R298_REFEATURE_MANIFEST_SCHEMA,
    R298_REFEATURE_RECORD_SCHEMA,
)
from poke_bot.alakazam_rule_derivative_transfer_r298 import (
    INZI_QUARANTINE_DIRECTORY,
    R298_TRANSFER_PLAN_SCHEMA,
    RuleDerivativeTransferError,
    validate_refeature_manifest_for_transfer,
    validate_refeatured_shard,
    validate_transfer_plan,
)


def _sha(byte: str) -> str:
    return "sha256:" + byte * 64


def _manifest() -> dict:
    return {
        "schema": R298_REFEATURE_MANIFEST_SCHEMA,
        "status": "complete_30_day_create_only_refeaturization_pending_census_gate",
        "goal_contract_path": R298_CONTRACT_PATH,
        "goal_contract_sha256": R298_CONTRACT_SHA256,
        "goal_gateway_sha256": R298_GOAL_SHA256,
        "legacy_feature_source_sha256": "sha256:ce0eb08ca74e337fe6ee1eaeb678eb42bd7f413bbfab04f8a228c3cfd3ce3db5",
        "validated_30_day_raw_manifest_sha256": _sha("a"),
        "frozen_schema_manifest_sha256": _sha("b"),
        "zero_bypass_receipt_sha256": _sha("c"),
        "feature_schema_sha256": _sha("d"),
        "target_schema_sha256": _sha("e"),
        "checklist_provenance_schema_sha256": _sha("f"),
        "utc_day_partitions": [
            *[f"2026-07-{day:02d}" for day in range(13, 32)],
            *[f"2026-08-{day:02d}" for day in range(1, 12)],
        ],
        "all_physical_shards_at_or_below_96_mib": True,
        "physical_shards": [],
        "physical_shard_count": 0,
    }


def _record(bindings: dict, *, schema: str = R298_REFEATURE_RECORD_SCHEMA) -> dict:
    return {
        "schema": schema,
        "materialization_bindings": bindings,
        "target_only_fields_never_policy_features": True,
    }


def test_transfer_rejects_manifest_without_reopened_evidence(tmp_path: Path) -> None:
    with pytest.raises(RuleDerivativeTransferError, match="reopened raw/schema/zero-bypass evidence"):
        validate_refeature_manifest_for_transfer(tmp_path, _manifest())


def test_transfer_rejects_arbitrary_destination_before_reopening_evidence() -> None:
    plan = {
        "schema": R298_TRANSFER_PLAN_SCHEMA,
        "status": "planned_no_inzi_bytes_mutated",
        "goal_contract_path": R298_CONTRACT_PATH,
        "goal_contract_sha256": R298_CONTRACT_SHA256,
        "goal_gateway_sha256": R298_GOAL_SHA256,
        "goal_revision": 4,
        "destination_host": "somewhere-else",
        "destination_directory": INZI_QUARANTINE_DIRECTORY,
    }
    with pytest.raises(RuleDerivativeTransferError, match="destination is not the quarantined directory"):
        validate_transfer_plan(plan)


def test_shard_validator_scans_and_rejects_a_late_bad_record(tmp_path: Path) -> None:
    manifest = _manifest()
    bindings = {
        "validated_30_day_raw_manifest_sha256": manifest["validated_30_day_raw_manifest_sha256"],
        "frozen_schema_manifest_sha256": manifest["frozen_schema_manifest_sha256"],
        "feature_schema_sha256": manifest["feature_schema_sha256"],
        "target_schema_sha256": manifest["target_schema_sha256"],
        "checklist_provenance_schema_sha256": manifest[
            "checklist_provenance_schema_sha256"
        ],
    }
    good = json.dumps(_record(bindings), sort_keys=True).encode("utf-8") + b"\n"
    late_bad = json.dumps(_record(bindings, schema="foreign/schema")).encode("utf-8") + b"\n"
    payload = good + late_bad
    relative = "day=2026-07-13/records/late-bad.jsonl.gz"
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    with gzip.open(target, "wb") as stream:
        stream.write(payload)
    shard = {
        "relative_path": relative,
        "sha256": "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest(),
        "size_bytes": target.stat().st_size,
        "record_count": 2,
        "uncompressed_jsonl_bytes": len(payload),
    }
    with pytest.raises(RuleDerivativeTransferError, match="record schema drifted"):
        validate_refeatured_shard(tmp_path, shard, manifest=manifest)

