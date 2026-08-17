from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "apply_teal_guide_weight_at_boundary",
    ROOT / "scripts/apply_teal_guide_weight_at_boundary.py",
)
assert SPEC is not None and SPEC.loader is not None
BOUNDARY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BOUNDARY
SPEC.loader.exec_module(BOUNDARY)
exact_boundary = BOUNDARY.exact_boundary
set_selector_value = BOUNDARY.set_selector_value
validate_registry_pair = BOUNDARY.validate_registry_pair
create_staged_registry = BOUNDARY.create_staged_registry
validate_current_deck_guide_log_naming_migration = (
    BOUNDARY.validate_current_deck_guide_log_naming_migration
)
wait_migration = BOUNDARY.wait_migration


def test_registry_pair_changes_only_teal_guide_weight(tmp_path: Path) -> None:
    current = {
        "specialists": {
            "teal-mask-ogerpon-ex": {
                "guide_loss_weight": 0.05,
                "guide_id": "teal-mask-ogerpon-ex",
            },
            "archaludon-ex": {"guide_loss_weight": 0.05},
        }
    }
    staged = json.loads(json.dumps(current))
    staged["specialists"]["teal-mask-ogerpon-ex"]["guide_loss_weight"] = 0.25
    current_path = tmp_path / "current.json"
    staged_path = tmp_path / "staged.json"
    current_path.write_text(json.dumps(current), encoding="utf-8")
    staged_path.write_text(json.dumps(staged), encoding="utf-8")

    validate_registry_pair(
        current_path,
        staged_path,
        old_weight=0.05,
        new_weight=0.25,
    )

    staged["specialists"]["archaludon-ex"]["guide_loss_weight"] = 0.25
    staged_path.write_text(json.dumps(staged), encoding="utf-8")
    with pytest.raises(RuntimeError, match="non-Teal"):
        validate_registry_pair(
            current_path,
            staged_path,
            old_weight=0.05,
            new_weight=0.25,
        )


def test_staged_registry_is_generated_as_one_field_change(tmp_path: Path) -> None:
    current = {
        "version": 1,
        "specialists": {
            "teal-mask-ogerpon-ex": {
                "guide_loss_weight": 0.05,
                "guide_id": "teal-mask-ogerpon-ex",
            },
            "archaludon-ex": {"guide_loss_weight": 0.05},
        },
    }
    current_path = tmp_path / "current.json"
    staged_path = tmp_path / "staged.json"
    current_path.write_text(json.dumps(current), encoding="utf-8")

    create_staged_registry(
        current_path,
        staged_path,
        old_weight=0.05,
        new_weight=0.25,
    )
    validate_registry_pair(
        current_path,
        staged_path,
        old_weight=0.05,
        new_weight=0.25,
    )
    staged = json.loads(staged_path.read_text(encoding="utf-8"))
    assert staged["version"] == 1
    assert (
        staged["specialists"]["teal-mask-ogerpon-ex"]["guide_loss_weight"]
        == 0.25
    )
    assert staged["specialists"]["archaludon-ex"]["guide_loss_weight"] == 0.05


def test_selector_mutation_requires_exactly_one_key() -> None:
    source = "A=1\nPURE_RL_ALLOW_CLEAN_BOUNDARY_DESIGN_MIGRATION=0\n"
    assert (
        set_selector_value(
            source,
            "PURE_RL_ALLOW_CLEAN_BOUNDARY_DESIGN_MIGRATION",
            "1",
        )
        == "A=1\nPURE_RL_ALLOW_CLEAN_BOUNDARY_DESIGN_MIGRATION=1\n"
    )
    with pytest.raises(RuntimeError, match="exactly one"):
        set_selector_value("A=1\n", "MISSING", "value")


def test_training_module_migration_changes_only_generic_guide_wording(
    tmp_path: Path,
) -> None:
    current = tmp_path / "current.py"
    staged = tmp_path / "staged.py"
    current.write_text(
        'raise ValueError(\n'
        '    "nonzero Alakazam guide loss has no usable guide rows; "\n'
        '    "verify POKEBOT_ALAKAZAM_GUIDE_TARGETS=1 and scorer coverage"\n'
        ')\n'
        'print(\n'
        '    f"[rl-train] Alakazam guide rows={guide_rows} "\n'
        ')\n',
        encoding="utf-8",
    )
    staged.write_text(
        current.read_text(encoding="utf-8")
        .replace(
            '"nonzero Alakazam guide loss has no usable guide rows; "',
            '"nonzero current-deck guide loss has no usable guide rows; "',
        )
        .replace(
            '"verify POKEBOT_ALAKAZAM_GUIDE_TARGETS=1 and scorer coverage"',
            '"verify the selected current-deck guide target switch and "\n'
            '                "scorer coverage"',
        )
        .replace(
            'f"[rl-train] Alakazam guide rows={guide_rows} "',
            'f"[rl-train] current-deck guide rows={guide_rows} "',
        ),
        encoding="utf-8",
    )
    validate_current_deck_guide_log_naming_migration(current, staged)

    staged.write_text(
        staged.read_text(encoding="utf-8") + "UNRELATED = True\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="beyond"):
        validate_current_deck_guide_log_naming_migration(current, staged)


def test_boundary_requires_commit_loop_and_checkpoint_identity(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "commits").mkdir(parents=True)
    checkpoint = run_dir / "iter5.pt"
    checkpoint.write_bytes(b"teal-iter5")
    sha256 = BOUNDARY.sha256

    payload = {
        "last_completed_iteration": 5,
        "next_iteration": 6,
        "learner": {"path": str(checkpoint), "digest": sha256(checkpoint)},
    }
    (run_dir / "commits/iter_00005.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    (run_dir / "loop_state.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    proof = exact_boundary(run_dir, 5)
    assert proof is not None
    assert proof[3] == sha256(checkpoint)

    checkpoint.write_bytes(b"mutated")
    with pytest.raises(RuntimeError, match="does not verify"):
        exact_boundary(run_dir, 5)


def test_controller_uses_only_managed_service_control() -> None:
    source = (
        ROOT / "scripts/apply_teal_guide_weight_at_boundary.py"
    ).read_text(encoding="utf-8")
    for forbidden in ("pkill", "killall", "os.kill(", "SIGKILL", "SIGTERM"):
        assert forbidden not in source
    assert 'systemctl("stop", args.unit)' in source
    assert 'systemctl("start", args.unit)' in source
    assert "RefuseManualStop=no" in source
    assert "stop_protection_restored=True" in source
    assert source.index("proof = exact_boundary") < source.index(
        "install_stop_window(args.unit, dropin)"
    )


def test_migration_accepts_authorized_trainer_source_identity_change(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "design_migrations"
    directory.mkdir()
    receipt = {
        "reason": BOUNDARY.MIGRATION_REASON,
        "boundary_next_iteration": 6,
        "changed_paths": [
            "learner.alakazam_guide_loss_weight",
            "learner.current_deck_guide_loss_weight",
            "expert_rehearsal.loss_weights.alakazam_guide",
            "source.source_tree_sha256",
        ],
    }
    target = directory / "migration_0006.json"
    target.write_text(json.dumps(receipt), encoding="utf-8")
    assert wait_migration(
        tmp_path, boundary_next_iteration=6, timeout_seconds=0.1
    ) == target.resolve()
