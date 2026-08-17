from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import scripts.apply_marnie_archetype_family_rollback as rollback_module
import scripts.resolve_marnie_archetype_family_monitor as resolver


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _args(tmp_path: Path, *, rollback_required: bool) -> argparse.Namespace:
    migration = _write(tmp_path / "migration.json", {"migration": True})
    monitor = _write(
        tmp_path / "monitor.json",
        {
            "rollback_required": rollback_required,
            "migration": {
                "path": str(migration.resolve()),
                "sha256": resolver.sha256(migration),
            },
        },
    )
    rollback_request = tmp_path / "rollback-request.json"
    if rollback_required:
        _write(rollback_request, {"rollback": True})
    zero_guide_row = {
        "guide_loss_weight": 0.0,
        "guide_retired": True,
        "guide_target_generation_required": False,
        "guide_conditioned_losses_enabled": False,
        "guide_action_influence": False,
    }
    merged = _write(
        tmp_path / "merged-family-guide-shadow.json",
        {"specialists": {"marnie-s-grimmsnarl-ex": dict(zero_guide_row)}},
    )
    fallback = _write(
        tmp_path / "guide-retired-parent.json",
        {"specialists": {"marnie-s-grimmsnarl-ex": dict(zero_guide_row)}},
    )
    guide_authority = _write(
        tmp_path / "guide-shadow-runtime.json",
        {
            "schema": resolver.GUIDE_SHADOW_SCHEMA,
            "status": "active_next_start_overlay",
            "owner_revision": 142,
            "merged_registry": {
                "path": str(merged.resolve()),
                "sha256": resolver.sha256(merged),
            },
            "guide_retired_registry": {
                "path": str(fallback.resolve()),
                "sha256": resolver.sha256(fallback),
            },
        },
    )
    python_executable = tmp_path / "python"
    launcher = tmp_path / "launcher.py"
    python_executable.write_text("python\n", encoding="utf-8")
    launcher.write_text("launcher\n", encoding="utf-8")
    return argparse.Namespace(
        service="trainer.service",
        monitor=monitor,
        migration=migration,
        rollback_request=rollback_request,
        loop_state=tmp_path / "loop.json",
        rollback_registry=tmp_path / "rollback-registry.json",
        environment_drop_in=tmp_path / "family.conf",
        python_executable=python_executable,
        launcher=launcher,
        receipt=tmp_path / "rollback-receipt.json",
        resolution_receipt=tmp_path / "resolution.json",
        guide_shadow_runtime_receipt=guide_authority,
        guide_shadow_overlay=tmp_path / "guide-shadow.conf",
    )


def test_passed_monitor_authorizes_managed_resume_without_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, rollback_required=False)
    monkeypatch.setattr(resolver, "_trainer_state", lambda _service: "failed")
    monkeypatch.setattr(
        resolver, "validate_post_activation_monitor", lambda value: value
    )
    result = resolver.resolve(args)
    assert result["status"] == "monitor_passed_resume_family_design"
    assert result["resume_authorized"] is True
    assert result["rollback_receipt"] is None
    assert result["guide_shadow_runtime"]["family_rollback_applied"] is False
    assert "merged-family-guide-shadow.json" in args.guide_shadow_overlay.read_text()
    assert resolver.resolve(args) == result


def test_failed_monitor_applies_rollback_before_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, rollback_required=True)
    monkeypatch.setattr(resolver, "_trainer_state", lambda _service: "failed")
    monkeypatch.setattr(
        resolver, "validate_post_activation_monitor", lambda value: value
    )

    def fake_apply(namespace: argparse.Namespace) -> dict[str, object]:
        _write(namespace.receipt, {"status": "rolled_back_at_clean_boundary"})
        return {"status": "rolled_back_at_clean_boundary"}

    monkeypatch.setattr(resolver, "apply_rollback", fake_apply)
    result = resolver.resolve(args)
    assert result["status"] == "rollback_applied_resume_parent"
    assert result["resume_authorized"] is True
    assert result["rollback_receipt"]["sha256"] == resolver.sha256(args.receipt)
    assert result["guide_shadow_runtime"]["family_rollback_applied"] is True
    assert "guide-retired-parent.json" in args.guide_shadow_overlay.read_text()


def test_family_rollback_restores_every_checkpoint_authority_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent_checkpoint = tmp_path / "parent.pt"
    parent_checkpoint.write_bytes(b"parent")
    parent = {
        "path": str(parent_checkpoint.resolve()),
        "sha256": rollback_module.sha256(parent_checkpoint),
    }
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_bytes(b"candidate")
    candidate = {
        "path": str(candidate_checkpoint.resolve()),
        "digest": rollback_module.sha256(candidate_checkpoint),
    }
    registry = _write(
        tmp_path / "before-registry.json",
        {
            "common_trainer_args": [
                "--boundary-design-migration-reason",
                "family_active",
            ]
        },
    )
    selector = _write(tmp_path / "selector.env", {"selector": True})
    pause = _write(
        tmp_path / "pause.json",
        {
            "schema": rollback_module.PAUSE_SCHEMA,
            "next_collection_started": False,
            "restart_prevent_status": 78,
            "learner_sha256": candidate["digest"],
        },
    )
    monitor = _write(tmp_path / "monitor.json", {"rollback_required": True})
    migration = tmp_path / "migration.json"
    drop_in = tmp_path / "family.conf"
    drop_in.write_text("[Service]\nEnvironment=X=1\n", encoding="utf-8")
    _write(
        migration,
        {"environment_drop_in_sha256": rollback_module.sha256(drop_in)},
    )
    request = _write(
        tmp_path / "rollback-request.json",
        {
            "pause": {
                "path": str(pause.resolve()),
                "sha256": rollback_module.sha256(pause),
            },
            "restore_registry": {
                "path": str(registry.resolve()),
                "sha256": rollback_module.sha256(registry),
            },
            "restore_selector": {
                "path": str(selector.resolve()),
                "sha256": rollback_module.sha256(selector),
            },
            "restore_checkpoint": parent,
        },
    )
    loop_state = _write(
        tmp_path / "loop.json",
        {
            "learner": candidate,
            "champion": candidate,
            "heldout_champion": candidate,
            "next_iteration": 11,
        },
    )
    python_executable = tmp_path / "python"
    python_executable.write_text("python\n", encoding="utf-8")
    launcher = tmp_path / "launcher.py"
    launcher.write_text("launcher\n", encoding="utf-8")
    args = argparse.Namespace(
        service="trainer.service",
        monitor=monitor,
        migration=migration,
        rollback_request=request,
        loop_state=loop_state,
        rollback_registry=tmp_path / "rollback-registry.json",
        environment_drop_in=drop_in,
        python_executable=python_executable,
        launcher=launcher,
        receipt=tmp_path / "rollback-receipt.json",
    )
    monkeypatch.setattr(
        rollback_module, "validate_post_activation_monitor", lambda value: value
    )
    monkeypatch.setattr(
        rollback_module, "validate_rollback_request", lambda *a, **k: {}
    )
    monkeypatch.setattr(rollback_module, "_service_state", lambda _s: "failed")
    receipt = rollback_module.apply(args)
    updated = json.loads(loop_state.read_text(encoding="utf-8"))
    expected = {"path": parent["path"], "digest": parent["sha256"]}
    assert updated["learner"] == expected
    assert updated["champion"] == expected
    assert updated["heldout_champion"] == expected
    assert receipt["restored_authority_fields"] == [
        "learner",
        "champion",
        "heldout_champion",
    ]
