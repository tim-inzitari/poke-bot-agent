from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/materialize_alakazam_action_critic_targets_elmo.py"


def _controller_module():
    spec = importlib.util.spec_from_file_location("target_materialization_controller", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _overlay_manifest(controller, *, wrong_split: bool = False) -> dict[str, object]:
    remaining = controller.EXPECTED_TARGET_ROW_COUNT - 19 * 100
    shards = []
    for index, day in enumerate(controller.WINDOW_DAYS):
        shards.append(
            {
                "utc_day": day,
                "split": (
                    "evaluation"
                    if wrong_split and day == controller.WINDOW_DAYS[0]
                    else controller.SPLIT_BY_DAY[day]
                ),
                "path": f"objects/{day}.rtp-overlay.jsonl",
                "sha256": f"sha256:{index + 1:064x}",
                "size_bytes": 10_000 + index,
                "complete_action_programs": remaining if index == 0 else 100,
            }
        )
    return {
        "schema": controller.OVERLAY_MANIFEST_SCHEMA,
        "overlay_shards": shards,
    }


def _snapshot_and_config(controller, *, execute: bool = False):
    snapshot = controller.build_source_snapshot(
        ROOT / "goals/alakazam-elmo-rule-derivative/contract.json"
    )
    config = controller.build_remote_config(
        execute=execute,
        snapshot=snapshot,
        overlay_root=controller.DEFAULT_OVERLAY_ROOT,
        overlay_manifest=controller.DEFAULT_OVERLAY_MANIFEST,
        expected_overlay_manifest_sha256=controller.DEFAULT_OVERLAY_MANIFEST_SHA256,
        base_pack_completion=controller.DEFAULT_BASE_PACK_COMPLETION,
        expected_base_pack_completion_sha256=controller.DEFAULT_BASE_PACK_COMPLETION_SHA256,
        raw_episode_root=controller.DEFAULT_RAW_EPISODE_ROOT,
        snapshot_parent=controller.DEFAULT_SNAPSHOT_PARENT,
        private_stage_parent=controller.DEFAULT_PRIVATE_STAGE_PARENT,
        target_root=controller.DEFAULT_TARGET_ROOT,
        image_sha256=controller.ELMO_IMAGE_SHA256,
        stage_nonce="00000000-0000-0000-0000-000000000001",
    )
    return snapshot, config


def test_canonical_overlay_inventory_requires_exact_window_splits_and_row_total() -> None:
    controller = _controller_module()
    days = controller.parse_overlay_days(_overlay_manifest(controller))

    assert [day.utc_day for day in days] == list(controller.WINDOW_DAYS)
    assert [day.split for day in days] == [
        controller.SPLIT_BY_DAY[day] for day in controller.WINDOW_DAYS
    ]
    assert sum(day.complete_action_programs for day in days) == 2_081_530
    assert all(day.raw_episode_filename.endswith(f"{day.utc_day}.zip") for day in days)

    with pytest.raises(controller.TargetMaterializationError, match="split drifted"):
        controller.parse_overlay_days(_overlay_manifest(controller, wrong_split=True))


def test_four_lane_allocation_is_unique_complete_and_five_days_per_lane() -> None:
    controller = _controller_module()
    lanes = controller.allocate_lanes(controller.parse_overlay_days(_overlay_manifest(controller)))

    assert len(lanes) == 4
    assert all(len(lane) == 5 for lane in lanes)
    assert sorted(day.utc_day for lane in lanes for day in lane) == list(controller.WINDOW_DAYS)


def test_minimal_snapshot_binds_current_contract_and_embedded_r21_authority() -> None:
    controller = _controller_module()
    snapshot = controller.build_source_snapshot(
        ROOT / "goals/alakazam-elmo-rule-derivative/contract.json"
    )

    contract = snapshot.manifest["canonical_goal_contract"]
    assert contract["canonical_goal_revision"] >= 21
    assert contract["target_owner_goal_revision"] == 21
    assert contract["path"] == "goals/alakazam-elmo-rule-derivative/contract.json"
    assert contract["sha256"] == controller.sha256_file(
        ROOT / "goals/alakazam-elmo-rule-derivative/contract.json"
    )
    assert [member["path"] for member in snapshot.manifest["members"]] == [
        "goals/alakazam-elmo-rule-derivative/contract.json",
        "poke_bot/__init__.py",
        "poke_bot/action_critic_targets.py",
        "scripts/build_alakazam_action_critic_targets.py",
    ]
    assert snapshot.manifest["minimal_member_count"] == 4
    assert snapshot.manifest_sha256.startswith("sha256:")


def test_rendered_remote_controller_is_dry_run_ready_and_enforces_isolation() -> None:
    controller = _controller_module()
    snapshot, config = _snapshot_and_config(controller)
    program = controller.render_remote_program(config)

    compile(program, "remote_action_critic_target_controller.py", "exec")
    assert config["execute"] is False
    assert config["source_snapshot"]["manifest_sha256"] == snapshot.manifest_sha256
    assert config["publication"]["atomic_no_clobber_publish"] is True
    assert config["publication"]["cleanup_retry_or_overwrite_allowed"] is False
    assert config["lane_count_exact"] == 4
    assert "renameat2(RENAME_NOREPLACE)" in program
    assert '["sudo", "-n", "docker", "image", "inspect"' in program
    assert '"sudo", "-n", "docker", "run", "--name", name' in program
    assert '"--pull", "never"' in program
    assert '"--network", "none"' in program
    assert '"--read-only"' in program
    assert '"--user", container["user"]' in program
    assert '"--cpus", str(container["cpus"])' in program
    assert '"--memory", container["memory"]' in program
    assert '"--entrypoint", "/usr/bin/ionice"' in program
    assert '"-c", str(container["low_priority_ionice_class"])' in program
    assert '"/usr/bin/nice", "-n", str(container["low_priority_nice"])' in program
    assert "planned_new(snapshot_root, \"source snapshot root\", allow_create_parent=True)" in program
    assert "ensure_create_only_parent(root.parent, \"source snapshot parent\")" in program
    assert "--rm" not in program
    assert "shutil.rmtree" not in program
    assert "os.unlink" not in program


def test_remote_config_refuses_output_outside_the_elmo_artifact_root() -> None:
    controller = _controller_module()
    snapshot, _config = _snapshot_and_config(controller)

    with pytest.raises(controller.TargetMaterializationError, match="escaped the Elmo artifact root"):
        controller.build_remote_config(
            execute=False,
            snapshot=snapshot,
            overlay_root=controller.DEFAULT_OVERLAY_ROOT,
            overlay_manifest=controller.DEFAULT_OVERLAY_MANIFEST,
            expected_overlay_manifest_sha256=controller.DEFAULT_OVERLAY_MANIFEST_SHA256,
            base_pack_completion=controller.DEFAULT_BASE_PACK_COMPLETION,
            expected_base_pack_completion_sha256=controller.DEFAULT_BASE_PACK_COMPLETION_SHA256,
            raw_episode_root=controller.DEFAULT_RAW_EPISODE_ROOT,
            snapshot_parent=controller.DEFAULT_SNAPSHOT_PARENT,
            private_stage_parent=controller.DEFAULT_PRIVATE_STAGE_PARENT,
            target_root="/tmp/not-an-elmo-artifact-root",
            image_sha256=controller.ELMO_IMAGE_SHA256,
            stage_nonce="00000000-0000-0000-0000-000000000001",
        )


def test_cli_reports_the_exact_remote_stderr_without_fabricating_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _controller_module()
    expected_stderr = "sudo: a password is required\n"

    def refused(_host: str, _program: str) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["ssh", "elmo", "python3 -"],
            returncode=1,
            stdout="",
            stderr=expected_stderr,
        )

    monkeypatch.setattr(controller, "run_remote_program", refused)
    with pytest.raises(controller.TargetMaterializationError, match="sudo: a password is required"):
        controller.main(
            ["--stage-nonce", "00000000-0000-0000-0000-000000000001"]
        )
