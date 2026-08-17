"""Tests for the immutable, no-clobber r197 source snapshot helper."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "stage_alakazam_rtp_r197_source_snapshot.py"


@pytest.fixture(scope="module")
def snapshot_module():
    spec = importlib.util.spec_from_file_location("r197_source_snapshot", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _assembled_root(tmp_path: Path, module) -> Path:
    source = tmp_path / "assembled-r197"
    source.mkdir()
    for relative in module.REQUIRED_RELATIVE_FILES:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == module.UNIT_TEMPLATE_RELATIVE:
            shutil.copy2(ROOT / relative, path)
        elif relative.name == "stage_alakazam_rtp_r197_source_snapshot.py":
            shutil.copy2(SCRIPT, path)
        else:
            path.write_text(f"# fixture for {relative}\n", encoding="utf-8")
    (source / "poke_bot" / "payload").mkdir()
    (source / "poke_bot" / "payload" / "bound.txt").write_text(
        "bound bytes\n", encoding="utf-8"
    )
    (source / "poke_bot" / "payload" / "bound-link").symlink_to("bound.txt")
    return source


@pytest.mark.unit
def test_publish_binds_files_symlinks_and_rendered_unit(tmp_path: Path, snapshot_module) -> None:
    source = _assembled_root(tmp_path, snapshot_module)
    deployments = tmp_path / "deployments"
    deployments.mkdir()

    target, manifest, _ = snapshot_module.build_plan(source, deployments)
    result = snapshot_module.publish(source, deployments)

    assert result["status"] == "published"
    assert Path(result["published_root"]) == target
    assert target.name == "alakazam-rtp-r197-src-" + manifest["source_tree_sha256"][7:19]
    persisted = snapshot_module.validate_published_root(target)
    assert persisted["status"] == "valid"
    entries = {entry["path"]: entry for entry in manifest["source_entries"]}
    assert entries["poke_bot/payload/bound-link"]["type"] == "symlink"
    assert entries["poke_bot/payload/bound-link"]["target"] == "bound.txt"
    rendered = (target / snapshot_module.RENDERED_UNIT_RELATIVE).read_text(encoding="utf-8")
    assert snapshot_module.TEMPLATE_SOURCE_ROOT not in rendered
    assert f"WorkingDirectory={target}" in rendered
    assert f"Environment=CG_LIB_PATH={target}/kaggle/input/cg-lib" in rendered
    assert f"ConditionPathExists={target}/{snapshot_module.MANIFEST_NAME}" in rendered
    assert f"verify --published-root {target}" in rendered


@pytest.mark.unit
def test_second_publish_is_verified_and_does_not_clobber(tmp_path: Path, snapshot_module) -> None:
    source = _assembled_root(tmp_path, snapshot_module)
    deployments = tmp_path / "deployments"
    deployments.mkdir()

    first = snapshot_module.publish(source, deployments)
    target = Path(first["published_root"])
    manifest_before = (target / snapshot_module.MANIFEST_NAME).read_bytes()
    second = snapshot_module.publish(source, deployments)

    assert second["status"] == "already_published"
    assert (target / snapshot_module.MANIFEST_NAME).read_bytes() == manifest_before


@pytest.mark.unit
def test_rejects_external_symlink_before_publish(tmp_path: Path, snapshot_module) -> None:
    source = _assembled_root(tmp_path, snapshot_module)
    deployments = tmp_path / "deployments"
    deployments.mkdir()
    (source / "poke_bot" / "payload" / "outside").symlink_to("/etc/hosts")

    with pytest.raises(snapshot_module.SnapshotError, match="relative target"):
        snapshot_module.build_plan(source, deployments)


@pytest.mark.unit
@pytest.mark.parametrize(
    "name",
    [
        ".private",
        ".vscode",
        ".codex-hidden-deploy",
        ".staging",
        "overlays",
        "state-sync",
        "._GOAL.md",
        "config",
    ],
)
def test_rejects_every_uncurated_top_level_entry(
    tmp_path: Path, snapshot_module, name: str
) -> None:
    source = _assembled_root(tmp_path, snapshot_module)
    deployments = tmp_path / "deployments"
    deployments.mkdir()
    extra = source / name
    if name.startswith("._"):
        extra.write_text("appledouble fixture\n", encoding="utf-8")
    else:
        extra.mkdir()

    with pytest.raises(snapshot_module.SnapshotError, match="unexpected top-level"):
        snapshot_module.build_plan(source, deployments)


@pytest.mark.unit
def test_rejects_uncurated_script_and_package_debris(
    tmp_path: Path, snapshot_module
) -> None:
    source = _assembled_root(tmp_path, snapshot_module)
    deployments = tmp_path / "deployments"
    deployments.mkdir()
    (source / "scripts" / "foreign.py").write_text("# unrelated\n", encoding="utf-8")

    with pytest.raises(snapshot_module.SnapshotError, match="uncurated entries under scripts"):
        snapshot_module.build_plan(source, deployments)

    (source / "scripts" / "foreign.py").unlink()
    (source / "poke_bot" / ".private").mkdir()
    with pytest.raises(snapshot_module.SnapshotError, match="disallowed package debris"):
        snapshot_module.build_plan(source, deployments)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("relative", "expected"),
    [
        (
            Path("scripts/extract_verified_specialist_records.py"),
            "required r197 source file is missing",
        ),
        (
            Path("kaggle/input/cg-lib/cg/libcg.so"),
            "required r197 source file is missing",
        ),
    ],
)
def test_rejects_missing_runtime_closure_file(
    tmp_path: Path, snapshot_module, relative: Path, expected: str
) -> None:
    source = _assembled_root(tmp_path, snapshot_module)
    deployments = tmp_path / "deployments"
    deployments.mkdir()
    (source / relative).unlink()

    with pytest.raises(snapshot_module.SnapshotError, match=expected):
        snapshot_module.build_plan(source, deployments)


@pytest.mark.unit
def test_rejects_unlisted_cg_runtime_file(tmp_path: Path, snapshot_module) -> None:
    source = _assembled_root(tmp_path, snapshot_module)
    deployments = tmp_path / "deployments"
    deployments.mkdir()
    (source / snapshot_module.CG_RUNTIME_RELATIVE / "foreign.py").write_text(
        "# not part of the curated runtime\n", encoding="utf-8"
    )

    with pytest.raises(
        snapshot_module.SnapshotError,
        match="uncurated entries under kaggle/input/cg-lib/cg",
    ):
        snapshot_module.build_plan(source, deployments)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("unsafe_directive", "section_marker", "expected"),
    [
        (
            "Conflicts=pokebot-final-format-alakazam-rtp-r175-rl.service",
            "[Service]",
            "forbidden",
        ),
        (
            "BindsTo=pokebot-final-format-alakazam-rtp-r175-rl.service",
            "[Service]",
            "forbidden",
        ),
        (
            "PartOf=pokebot-final-format-alakazam-rtp-r175-rl.service",
            "[Service]",
            "forbidden",
        ),
        (
            "Wants=pokebot-final-format-alakazam-rtp-r175-rl.service",
            "[Service]",
            "exactly allowlisted",
        ),
        (
            "Upholds=pokebot-final-format-alakazam-rtp-r175-rl.service",
            "[Service]",
            "exactly allowlisted",
        ),
        (
            "OnSuccess=pokebot-final-format-alakazam-rtp-r175-rl.service",
            "[Service]",
            "exactly allowlisted",
        ),
        (
            "ConsistsOf=pokebot-final-format-alakazam-rtp-r175-rl.service",
            "[Service]",
            "exactly allowlisted",
        ),
        (
            "PropagatesStopTo=pokebot-final-format-alakazam-rtp-r175-rl.service",
            "[Service]",
            "exactly allowlisted",
        ),
        (
            "PropagatesReloadTo=pokebot-final-format-alakazam-rtp-r175-rl.service",
            "[Service]",
            "exactly allowlisted",
        ),
        (
            "StopPropagatedFrom=pokebot-final-format-alakazam-rtp-r175-rl.service",
            "[Service]",
            "exactly allowlisted",
        ),
        (
            "JoinsNamespaceOf=pokebot-final-format-alakazam-rtp-r175-rl.service",
            "[Service]",
            "exactly allowlisted",
        ),
        (
            "RequiresMountsFor=/home/pokebot/poke-bot-agent",
            "[Service]",
            "exactly allowlisted",
        ),
        (
            "WantsMountsFor=/home/pokebot/poke-bot-agent",
            "[Service]",
            "exactly allowlisted",
        ),
        (
            "EnvironmentFile=/home/pokebot/poke-bot-agent/.private/r197.env",
            "[Install]",
            "forbidden",
        ),
        (
            "ExecStartPost=/usr/bin/systemctl stop pokebot-final-format-alakazam-rtp-r175-rl.service",
            "Restart=no",
            "forbidden",
        ),
    ],
)
def test_rejects_template_service_control_or_relationships(
    tmp_path: Path,
    snapshot_module,
    unsafe_directive: str,
    section_marker: str,
    expected: str,
) -> None:
    source = _assembled_root(tmp_path, snapshot_module)
    deployments = tmp_path / "deployments"
    deployments.mkdir()
    template = source / snapshot_module.UNIT_TEMPLATE_RELATIVE
    template.write_text(
        template.read_text(encoding="utf-8").replace(
            section_marker, f"{unsafe_directive}\n{section_marker}", 1
        ),
        encoding="utf-8",
    )

    with pytest.raises(snapshot_module.SnapshotError, match=expected):
        snapshot_module.build_plan(source, deployments)


@pytest.mark.unit
def test_rejects_template_install_target_override(
    tmp_path: Path, snapshot_module
) -> None:
    source = _assembled_root(tmp_path, snapshot_module)
    deployments = tmp_path / "deployments"
    deployments.mkdir()
    template = source / snapshot_module.UNIT_TEMPLATE_RELATIVE
    template.write_text(
        template.read_text(encoding="utf-8").replace(
            "WantedBy=default.target",
            "WantedBy=pokebot-final-format-alakazam-rtp-r175-rl.service",
        ),
        encoding="utf-8",
    )

    with pytest.raises(snapshot_module.SnapshotError, match="WantedBy=default.target"):
        snapshot_module.build_plan(source, deployments)


@pytest.mark.unit
def test_rejects_template_environment_override(
    tmp_path: Path, snapshot_module
) -> None:
    source = _assembled_root(tmp_path, snapshot_module)
    deployments = tmp_path / "deployments"
    deployments.mkdir()
    template = source / snapshot_module.UNIT_TEMPLATE_RELATIVE
    template.write_text(
        template.read_text(encoding="utf-8")
        .replace(
            "Restart=no",
            "Environment=CUDA_VISIBLE_DEVICES=0\nRestart=no",
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(snapshot_module.SnapshotError, match="exact isolated shadow binding"):
        snapshot_module.build_plan(source, deployments)


@pytest.mark.unit
def test_rejects_restart_or_start_command_that_is_not_the_stage(
    tmp_path: Path, snapshot_module
) -> None:
    source = _assembled_root(tmp_path, snapshot_module)
    deployments = tmp_path / "deployments"
    deployments.mkdir()
    template = source / snapshot_module.UNIT_TEMPLATE_RELATIVE
    contents = template.read_text(encoding="utf-8").replace(
        "Restart=no", "Restart=always"
    )
    template.write_text(contents, encoding="utf-8")

    with pytest.raises(snapshot_module.SnapshotError, match="exactly Restart=no"):
        snapshot_module.build_plan(source, deployments)

    template.write_text(
        contents.replace("Restart=always", "Restart=no").replace(
            f"ExecStart={snapshot_module.PYTHON} -u scripts/stage_alakazam_rtp_r197.py",
            "ExecStart=/bin/true",
        ),
        encoding="utf-8",
    )
    with pytest.raises(snapshot_module.SnapshotError, match="checksum-bound r197 stage"):
        snapshot_module.build_plan(source, deployments)


@pytest.mark.unit
def test_rejects_template_output_root_override(
    tmp_path: Path, snapshot_module
) -> None:
    source = _assembled_root(tmp_path, snapshot_module)
    deployments = tmp_path / "deployments"
    deployments.mkdir()
    template = source / snapshot_module.UNIT_TEMPLATE_RELATIVE
    template.write_text(
        template.read_text(encoding="utf-8").replace(
            "--heldout-fraction 0.20",
            "--heldout-fraction 0.20 --output-root /home/pokebot/poke-bot-agent/outputs/pure_rl",
        ),
        encoding="utf-8",
    )

    with pytest.raises(snapshot_module.SnapshotError, match="exact checksum-bound r197 stage"):
        snapshot_module.build_plan(source, deployments)
