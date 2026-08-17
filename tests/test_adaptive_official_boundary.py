from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ops.apply_adaptive_official_at_boundary import (
    migration_unit,
    validate_staged,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unit(root: Path) -> str:
    return f"""[Service]
WorkingDirectory={root}
Environment=PURE_RL_DECISION_CONTEXT=history
Environment=PURE_RL_TEMPORAL_LAYERS=1
Environment=PURE_RL_MAX_CONTEXT=320
Environment=PURE_RL_OFFICIAL_ADAPTIVE_TARGETING=1
Environment=PURE_RL_OFFICIAL_ADAPTIVE_MIN_SHARE=0.05
Environment=PURE_RL_OFFICIAL_ADAPTIVE_GAP_POWER=2.0
ExecStart=python train --games-per-iter 8192 --train-max-decisions-per-batch 8192 --heldout-games 1000 --heldout-per-opponent-floor 0.50 --resume auto
"""


def test_migration_unit_grants_authority_once() -> None:
    stable = "ExecStart=python train --resume auto\n"
    migrated = migration_unit(stable, "adaptive-official-v1")
    assert migrated.count("--allow-clean-boundary-design-migration") == 1
    assert migrated.count("--boundary-design-migration-reason adaptive-official-v1") == 1
    assert "--allow-clean-boundary-design-migration" not in stable
    with pytest.raises(ValueError):
        migration_unit(migrated, "second-use")


def test_validate_staged_binds_digest_contract_and_no_persistent_authority(
    tmp_path: Path,
) -> None:
    source = tmp_path / "scripts/train_pure_rl.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "\n".join(
            [
                "def _latest_official_heldout_win_rates(): pass",
                "def _adaptive_official_target_weights(): pass",
                'A = "latest_exact_heldout_gap_v1"',
                'B = "adaptive_exact_heldout_gap_v1"',
                'C = "collection.official_targeting"',
            ]
        ),
        encoding="utf-8",
    )
    unit = tmp_path / "adaptive.service"
    unit.write_text(_unit(tmp_path), encoding="utf-8")
    validated = validate_staged(
        staged_root=tmp_path,
        staged_unit=unit,
        expected_source_sha256=_sha(source),
        expected_unit_sha256=_sha(unit),
    )
    assert f"WorkingDirectory={tmp_path}" in validated
    unit.write_text(unit.read_text() + "# mutation\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unit digest changed"):
        validate_staged(
            staged_root=tmp_path,
            staged_unit=unit,
            expected_source_sha256=_sha(source),
            expected_unit_sha256="0" * 64,
        )
