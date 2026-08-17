from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts.prepare_marnie_latest20_runtime_migration import (
    EXPECTED_DATES,
    MIGRATION_REASON,
    REGISTRATION_SCHEMA,
    REGISTRY_SCHEMA,
    TARGET_DIGEST,
    TARGET_SCHEMA,
    prepare,
    sha256,
    sync_receipt_ready,
)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _fixture(tmp_path: Path) -> argparse.Namespace:
    destination = tmp_path / "expert-latest20-r109"
    specialist = destination / "marnie-s-grimmsnarl-ex"
    manifest = specialist / "manifest.json"
    _write(
        manifest,
        {
            "totals": {"decisions_kept": 123_456},
            "expanded_strategic_targets": {
                "schema": TARGET_SCHEMA,
                "digest": TARGET_DIGEST,
                "decisions": 123_456,
            },
        },
    )
    pointer = specialist / "PROTECTED_EXPERT_CORPUS.json"
    _write(
        pointer,
        {
            "schema": "poke_bot.pinned_expert_corpus/v1",
            "protected": True,
            "manifest": "manifest.json",
            "manifest_sha256": sha256(manifest),
        },
    )
    current = tmp_path / "staged-current"
    current.symlink_to(destination, target_is_directory=True)
    sync = tmp_path / "sync.json"
    _write(
        sync,
        {
            "schema": "poke_bot.latest20_specialist_sync/v1",
            "status": "ready",
            "dates": EXPECTED_DATES,
            "specialist_count": 18,
            "source_bytes": 100,
            "copied_bytes": 100,
            "destination": str(destination),
            "current_pointer": str(current),
            "expanded_target_schema": TARGET_SCHEMA,
            "expanded_target_digest": TARGET_DIGEST,
        },
    )
    source_registry = tmp_path / "source-registry.json"
    _write(
        source_registry,
        {
            "schema": REGISTRY_SCHEMA,
            "common_trainer_args": [
                "--allow-clean-boundary-design-migration",
                "--boundary-design-migration-reason",
                "old-reason",
                "--resume",
                "auto",
            ],
            "specialists": {
                "marnie-s-grimmsnarl-ex": {
                    "status": "ready",
                    "expert_manifest": "/old/PROTECTED_EXPERT_CORPUS.json",
                    "expert_manifest_sha256": "old",
                }
            },
        },
    )
    source_registration = tmp_path / "source-registration.json"
    _write(
        source_registration,
        {
            "schema": REGISTRATION_SCHEMA,
            "status": "registered_ready_for_managed_rl",
            "runtime_registry": str(source_registry),
            "runtime_registry_sha256": sha256(source_registry),
            "checkpoint": "/immutable/model.pt",
            "checkpoint_sha256": "sha256:model",
        },
    )
    return argparse.Namespace(
        sync_receipt=sync,
        source_registry=source_registry,
        source_registration=source_registration,
        output_registry=tmp_path / "candidate-registry.json",
        output_registration=tmp_path / "candidate-registration.json",
        stage_receipt=tmp_path / "stage.json",
        minimum_decisions=100_000,
    )


def test_prepare_stages_versioned_identity_without_mutating_sources(
    tmp_path: Path,
) -> None:
    args = _fixture(tmp_path)
    source_registry_body = args.source_registry.read_bytes()
    source_registration_body = args.source_registration.read_bytes()

    receipt = prepare(args)

    assert receipt["status"] == "ready_for_clean_rehearsal_boundary"
    assert receipt["selector_modified"] is False
    assert receipt["managed_units_modified"] is False
    assert args.source_registry.read_bytes() == source_registry_body
    assert args.source_registration.read_bytes() == source_registration_body
    candidate = json.loads(args.output_registry.read_text())
    reason_index = candidate["common_trainer_args"].index(
        "--boundary-design-migration-reason"
    )
    assert candidate["common_trainer_args"][reason_index + 1] == MIGRATION_REASON
    row = candidate["specialists"]["marnie-s-grimmsnarl-ex"]
    assert row["expert_manifest"].endswith("PROTECTED_EXPERT_CORPUS.json")
    assert row["expert_manifest_sha256"] == sha256(
        Path(row["expert_manifest"])
    ).removeprefix("sha256:")
    registration = json.loads(args.output_registration.read_text())
    assert registration["runtime_registry"] == str(args.output_registry)
    assert registration["runtime_registry_sha256"] == sha256(
        args.output_registry
    )
    assert prepare(args) == receipt


def test_prepare_rejects_wrong_window(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    sync = json.loads(args.sync_receipt.read_text())
    sync["dates"] = sync["dates"][:-1]
    _write(args.sync_receipt, sync)

    with pytest.raises(RuntimeError, match="not activation-ready"):
        prepare(args)


def test_sync_receipt_waiter_requires_ready_status(tmp_path: Path) -> None:
    receipt = tmp_path / "sync.json"
    assert not sync_receipt_ready(receipt)
    receipt.write_text('{"status":"syncing"}\n', encoding="utf-8")
    assert not sync_receipt_ready(receipt)
    receipt.write_text('{"status":"ready"}\n', encoding="utf-8")
    assert sync_receipt_ready(receipt)


def test_managed_stage_unit_is_non_authoritative() -> None:
    unit = Path(
        "deploy/systemd/"
        "pokebot-marnie-latest20-runtime-stage-r109.service"
    ).read_text()

    assert "prepare_marnie_latest20_runtime_migration.py" in unit
    assert "--wait-seconds 21600" in unit
    assert "--minimum-decisions 100000" in unit
    assert "systemctl" not in unit
    assert (
        "After=pokebot-expert-latest20-v6-strategic-r109-stage-sync-11m.service"
        in unit
    )
    assert "Wants=pokebot-expert-latest20" not in unit
    assert (
        "After=pokebot-expert-latest20-v6-strategic-r109-stage-sync.service"
        not in unit
    )
    assert "stage-sync-8m.service" not in unit
    assert "specialist_runtime_registry_h10_r109_latest20_20260714_20260802.json" in unit
    assert "final-format-marnie-r104-h10-runtime-registration-r109-latest20.json" in unit
