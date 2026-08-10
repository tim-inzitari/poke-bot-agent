"""Focused contract tests for the isolated Alakazam RTP r197 stage."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "stage_alakazam_rtp_r197.py"


@pytest.fixture(scope="module")
def r197_module():
    pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("stage_alakazam_rtp_r197", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_stage_source_has_only_complete_action_pipeline() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "sys.dont_write_bytecode = True" in source
    assert "plan_r197_complete_action_selection" in source
    assert "materialize_r197_complete_action_corpus" in source
    assert "EpisodeGroupedFeatureManifest" not in source
    assert "resolve_expert_manifest" not in source
    assert "COMPACT_MODE_TEMPORAL_EXPERT" not in source
    assert "_encode_partition" not in source
    assert "save_rtp_checkpoint" not in source
    assert "--dry-run" in source


@pytest.mark.unit
def test_shadow_unit_is_uuid_pinned_and_never_conflicts_r175() -> None:
    unit = (
        ROOT / "deploy/systemd/pokebot-alakazam-rtp-r197-shadow.service"
    ).read_text(encoding="utf-8")

    assert "WorkingDirectory=/home/inzi/poke-bot-agent-deployments/final-format-alakazam-rtp-r197-shadow-v1" in unit
    assert "CUDA_VISIBLE_DEVICES=GPU-79cf504f-6573-0b8c-c90e-eb567b7bcfa6" in unit
    assert "--device cuda:0" in unit
    assert "--max-train-games 512" in unit
    assert "--max-heldout-games 128" in unit
    assert not any(line.startswith("Conflicts=") for line in unit.splitlines())
    assert "systemctl" not in unit


@pytest.mark.unit
def test_r197_binds_the_owner_256_pass_budget(r197_module) -> None:
    config = r197_module._r197_planner_config(d_model=96)

    assert config.max_neural_passes == 256
    assert config.num_plan_candidates == 4
    assert config.max_recursion_depth == 2
    assert r197_module.MAX_ACTION_COMBOS == 1024
    assert r197_module.FUTURE_ABSOLUTE_MAX_NEURAL_PASSES == 256


@pytest.mark.unit
def test_r197_recursive_probe_completes_without_fallback(r197_module) -> None:
    probe = r197_module._recursive_budget_probe(d_model=96)

    assert probe["mode"] == "recursive_plan"
    assert probe["neural_passes"] == 6
    assert probe["forced_replan_mode"] == "recursive_plan"
    assert probe["forced_replan_neural_passes"] == 5
    assert probe["headroom"] == 256 - probe["neural_passes"]


@pytest.mark.unit
def test_r197_candidate_identity_changes_with_training_contract(r197_module) -> None:
    base = {"schema": r197_module.SCHEMA, "training": {"epochs": 4}}
    changed = {"schema": r197_module.SCHEMA, "training": {"epochs": 5}}

    base_id, base_digest = r197_module._candidate_identity(base)
    changed_id, changed_digest = r197_module._candidate_identity(changed)

    assert base_id.startswith("r197-")
    assert base_digest.startswith("sha256:")
    assert (base_id, base_digest) != (changed_id, changed_digest)


@pytest.mark.unit
def test_sidecar_config_digest_matches_promotion_compact_json(r197_module) -> None:
    config = {"default_subgoals": ["a", "b"], "d_model": 96}
    compact = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    assert r197_module._checkpoint_config_sha256(config) == (
        "sha256:" + hashlib.sha256(compact).hexdigest()
    )
    assert r197_module._checkpoint_config_sha256(config) != (
        "sha256:" + hashlib.sha256(compact + b"\n").hexdigest()
    )


@pytest.mark.unit
def test_r175_terminal_guard_accepts_failed_pid_zero_with_current_receipts(
    r197_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = tmp_path / "registry.json"
    completion = tmp_path / "completion.json"
    registry.write_text("registry", encoding="utf-8")
    completion.write_text("completion", encoding="utf-8")
    registry_sha256 = "sha256:" + "a" * 64
    completion_sha256 = "sha256:" + "b" * 64
    digests = {
        registry: registry_sha256,
        completion: completion_sha256,
    }
    monkeypatch.setattr(r197_module, "_sha256", lambda path: digests[Path(path)])
    monkeypatch.setattr(
        r197_module,
        "_r175_unit_state",
        lambda unit: {"unit": unit, "active_state": "failed", "main_pid": 0},
    )

    boundary = r197_module._verify_r175_terminal_boundary(
        {
            "production_boundary": {
                "guarded_services": ["r175-rl.service", "r175-orchestrator.service"],
                "terminal_registry": str(registry),
                "terminal_registry_sha256": registry_sha256,
                "terminal_completion_receipt": str(completion),
                "terminal_completion_receipt_sha256": completion_sha256,
            }
        }
    )

    assert boundary["r175_restart_or_preemption_performed"] is False
    assert all(row["main_pid"] == 0 for row in boundary["services"])


@pytest.mark.unit
def test_r175_terminal_guard_rejects_active_service(
    r197_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        r197_module,
        "_r175_unit_state",
        lambda unit: {"unit": unit, "active_state": "active", "main_pid": 1},
    )

    with pytest.raises(RuntimeError, match="refuses to overlap r175"):
        r197_module._verify_r175_terminal_boundary(
            {
                "production_boundary": {
                    "guarded_services": [
                        "r175-rl.service",
                        "r175-orchestrator.service",
                    ]
                }
            }
        )


@pytest.mark.unit
def test_blackwell_preflight_rejects_any_non_uuid_cuda_mask(
    r197_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

    with pytest.raises(RuntimeError, match="exact Blackwell UUID"):
        r197_module._verify_blackwell_device("cuda:0")


@pytest.mark.unit
def test_r197_rejects_an_output_root_override(r197_module, tmp_path: Path) -> None:
    args = r197_module._parser().parse_args(
        ["--check", "--output-root", str(tmp_path / "legacy-r195")]
    )

    with pytest.raises(RuntimeError, match="output_root must be the dedicated"):
        r197_module._validate_args(args)


@pytest.mark.unit
def test_r197_output_root_rejects_symlinked_root_or_ancestor(
    r197_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    legacy = tmp_path / "legacy-r195"
    legacy.mkdir()
    root = tmp_path / "alakazam-r197-shadow"
    root.symlink_to(legacy, target_is_directory=True)
    monkeypatch.setattr(r197_module, "DEFAULT_OUTPUT_ROOT", root)

    with pytest.raises(RuntimeError, match="forbidden symlink"):
        r197_module._safe_output_root(root)

    root.unlink()
    symlinked_parent = tmp_path / "output-parent"
    symlinked_parent.symlink_to(legacy, target_is_directory=True)
    nested_root = symlinked_parent / "alakazam-r197-shadow"
    monkeypatch.setattr(r197_module, "DEFAULT_OUTPUT_ROOT", nested_root)

    with pytest.raises(RuntimeError, match="forbidden symlink"):
        r197_module._safe_output_root(nested_root)


@pytest.mark.unit
def test_r197_output_root_rejects_nonphysical_component(
    r197_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    non_directory = tmp_path / "not-a-directory"
    non_directory.write_text("not a directory", encoding="utf-8")
    root = non_directory / "alakazam-r197-shadow"
    monkeypatch.setattr(r197_module, "DEFAULT_OUTPUT_ROOT", root)

    with pytest.raises(RuntimeError, match="non-directory component"):
        r197_module._safe_output_root(root)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("factory", "label"),
    [
        (lambda module, root: module._corpus_dir(root, "r197-corpus"), "corpus"),
        (lambda module, root: module._candidate_dir(root, "r197-candidate"), "candidate"),
    ],
)
def test_r197_output_children_reject_symlinks(
    r197_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    factory,
    label: str,
) -> None:
    root = tmp_path / "alakazam-r197-shadow"
    root.mkdir()
    monkeypatch.setattr(r197_module, "DEFAULT_OUTPUT_ROOT", root)
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    parent_name = "complete-action-corpus" if label == "corpus" else "candidates"
    child_name = "r197-corpus" if label == "corpus" else "r197-candidate"
    child_parent = root / parent_name
    child_parent.mkdir()
    (child_parent / child_name).symlink_to(legacy, target_is_directory=True)

    with pytest.raises(RuntimeError, match="forbidden symlink"):
        factory(r197_module, root)


@pytest.mark.unit
def test_r197_can_create_only_a_missing_physical_output_root(
    r197_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "alakazam-r197-shadow"
    monkeypatch.setattr(r197_module, "DEFAULT_OUTPUT_ROOT", root)

    assert r197_module._safe_output_root(root) == root
    created = r197_module._ensure_physical_directory(root, label="output root")

    assert created == root
    assert root.is_dir()
    assert not root.is_symlink()


@pytest.mark.unit
def test_r197_run_requires_an_immutable_source_snapshot(
    r197_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(r197_module.SOURCE_SNAPSHOT_ROOT_ENV, raising=False)
    monkeypatch.delenv(r197_module.SOURCE_SNAPSHOT_TREE_ENV, raising=False)

    with pytest.raises(RuntimeError, match="requires a rendered immutable source-snapshot"):
        r197_module._source_snapshot_binding(require=True)


@pytest.mark.unit
def test_r197_source_snapshot_binding_enters_candidate_contract(
    r197_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tree_sha256 = "sha256:" + "a" * 64
    snapshot = tmp_path / ("alakazam-rtp-r197-src-" + "a" * 12)
    unit_path = snapshot / r197_module.SOURCE_SNAPSHOT_UNIT_RELATIVE
    unit_path.parent.mkdir(parents=True)
    unit_path.write_text(
        "\n".join(
            (
                "[Service]",
                f"Environment={r197_module.SOURCE_SNAPSHOT_ROOT_ENV}={snapshot}",
                f"Environment={r197_module.SOURCE_SNAPSHOT_TREE_ENV}={tree_sha256}",
                "",
            )
        ),
        encoding="utf-8",
    )
    manifest_path = snapshot / r197_module.SOURCE_SNAPSHOT_MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(
            {
                "schema": r197_module.SOURCE_SNAPSHOT_SCHEMA,
                "source_tree_sha256": tree_sha256,
                "rendered_unit": {
                    "path": str(r197_module.SOURCE_SNAPSHOT_UNIT_RELATIVE),
                    "sha256": r197_module._sha256(unit_path),
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(r197_module, "ROOT", snapshot)
    monkeypatch.setenv(r197_module.SOURCE_SNAPSHOT_ROOT_ENV, str(snapshot))
    monkeypatch.setenv(r197_module.SOURCE_SNAPSHOT_TREE_ENV, tree_sha256)
    from scripts import stage_alakazam_rtp_r197_source_snapshot as snapshot_helper

    monkeypatch.setattr(
        snapshot_helper,
        "validate_published_root",
        lambda root: {
            "status": "valid",
            "published_root": str(snapshot.resolve()),
            "source_tree_sha256": tree_sha256,
            "manifest_sha256": r197_module._sha256(manifest_path),
            "rendered_unit_sha256": r197_module._sha256(unit_path),
        },
    )

    binding = r197_module._source_snapshot_binding(require=True)

    assert binding["status"] == "bound"
    assert binding["source_tree_sha256"] == tree_sha256
    assert binding["manifest_sha256"] == r197_module._sha256(manifest_path)
    assert binding["rendered_unit_sha256"] == r197_module._sha256(unit_path)
    assert binding["snapshot_verification_status"] == "valid"
    assert binding["managed_candidate_run_allowed"] is True
