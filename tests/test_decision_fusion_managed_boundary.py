from __future__ import annotations

from pathlib import Path

import pytest

from scripts.apply_decision_fusion_warmup_managed_boundary import (
    _allocate_artifact_dir,
    _assert_runtime_registry_root,
    _selector_for_root,
)


ROOT = Path(__file__).resolve().parents[1]


def test_selector_swap_changes_only_runtime_and_pythonpath() -> None:
    old = Path("/runtime/old")
    new = Path("/runtime/new")
    source = (
        "POKEBOT_ACTIVE_SPECIALIST=dudunsparce\n"
        "POKEBOT_SPECIALIST_RUNTIME_ROOT=/runtime/old\n"
        "PYTHONPATH=/runtime/old\n"
        "UNRELATED=preserved\n"
    )
    updated = _selector_for_root(source, old, new)
    assert "POKEBOT_SPECIALIST_RUNTIME_ROOT=/runtime/new" in updated
    assert "PYTHONPATH=/runtime/new" in updated
    assert "UNRELATED=preserved" in updated
    assert "/runtime/old" not in updated
    assert "POKEBOT_DECISION_FUSION_ENABLED=1" in updated
    assert "POKEBOT_DECISION_FUSION_RUNTIME_ENABLED=0" in updated
    assert (
        "PURE_RL_BOUNDARY_MIGRATION_REASON_OVERRIDE="
        "receipt_backed_decision_fusion_warmup_v1"
    ) in updated


def test_selector_swap_fails_closed_on_layering_or_drift() -> None:
    with pytest.raises(RuntimeError, match="exactly"):
        _selector_for_root(
            "POKEBOT_SPECIALIST_RUNTIME_ROOT=/runtime/old\n",
            Path("/runtime/old"),
            Path("/runtime/new"),
        )


def test_boundary_artifacts_preserve_failed_attempts(tmp_path: Path) -> None:
    digest = "sha256:" + ("0" * 52) + "0123456789ab"
    first = _allocate_artifact_dir(
        tmp_path, after_iteration=12, parent_digest=digest
    )
    (first / "failure-evidence.json").write_text("{}\n", encoding="utf-8")
    second = _allocate_artifact_dir(
        tmp_path, after_iteration=12, parent_digest=digest
    )

    assert first.name == "after_iter_00012-0123456789ab"
    assert second.name == "after_iter_00012-0123456789ab-attempt-0002"
    assert (first / "failure-evidence.json").is_file()


def test_boundary_rejects_registry_delegation_outside_staged_root(
    tmp_path: Path,
) -> None:
    staged = tmp_path / "staged"
    (staged / "ops").mkdir(parents=True)
    (staged / "ops/specialist_runtime_registry_v1.json").write_text(
        '{"runtime_root": "/different/runtime"}\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="delegates outside"):
        _assert_runtime_registry_root(staged)

    (staged / "ops/specialist_runtime_registry_v1.json").write_text(
        '{"runtime_root": "' + str(staged) + '"}\n',
        encoding="utf-8",
    )
    _assert_runtime_registry_root(staged)


def test_boundary_uses_only_declared_managed_lifecycle_controls() -> None:
    script = (
        ROOT / "scripts/apply_decision_fusion_warmup_managed_boundary.py"
    ).read_text(encoding="utf-8")
    unit = (
        ROOT
        / "deploy/systemd/pokebot-decision-fusion-warmup-boundary.service"
    ).read_text(encoding="utf-8")
    for forbidden in ("pkill", "killall", "os.kill(", "SIGKILL", "SIGTERM"):
        assert forbidden not in script
        assert forbidden not in unit
    assert '["systemctl", "--user", "stop", args.unit]' in script
    assert '"launchctl", "kickstart", "-k"' in script
    assert "docker compose" in script
    assert "RefuseManualStop=no" in script
    assert "stop_protection_restored=True" in script
    assert "staged Python package digest mismatch" in script
    assert "Elmo staged Python package digest mismatch" in script
    assert '_wait_endpoint(args.old_runtime_root, "bert.local:8766")' in script
    assert '_wait_endpoint(args.old_runtime_root, "192.168.1.143:8765")' in script
    assert "--maintenance-lock" in unit
    assert "--after-iteration 14" in unit
    assert "--wait-for-completed-collection-iteration 15" in unit
    assert "consume the start-limit budget" in script
    managed_start = script.index(
        '_run(["systemctl", "--user", "start", args.unit], timeout=90)'
    )
    first_lock_release = script.index(
        "args.maintenance_lock.unlink(missing_ok=True)"
    )
    assert managed_start < first_lock_release
    assert script.index(
        'raise RuntimeError("fusion warmup trainer failed stability check")'
    ) < first_lock_release
