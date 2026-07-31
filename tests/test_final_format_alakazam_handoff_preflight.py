from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest
import yaml

from scripts.validate_final_format_alakazam_handoff import (
    _canonical_json_sha256,
    sha256,
    validate_preferred_parent_metadata,
    validate_release_boundary,
    validate_static,
)


ROOT = Path(__file__).resolve().parents[1]


def test_static_handoff_preflight_performs_no_model_or_replay_work() -> None:
    result = validate_static(
        root=ROOT,
        static_contract_path=(
            ROOT
            / "ops/final_submission/final_format_alakazam_static_preparation_v1.json"
        ),
        validation_receipt_path=(
            ROOT / "state/final_format_alakazam_static_preparation_validation_v11.json"
        ),
        cycle_contract_path=ROOT / "ops/specialist_cycle_handoff_v1.json",
        state_path=ROOT / "state/specialists.yaml",
    )
    assert result["checked_artifact_count"] == 10
    assert result["refresh_order"] == ["alakazam", "marnie-s-grimmsnarl-ex"]
    assert result["completed_refresh_specialist_ids"] == []
    assert result["model_or_replay_computation_performed"] is False


def test_release_boundary_rejects_active_slowking_trainer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "scripts.validate_final_format_alakazam_handoff.subprocess.run",
        lambda argv, check: subprocess.CompletedProcess(argv, 0),
    )
    with pytest.raises(RuntimeError, match="still active"):
        validate_release_boundary(
            state_path=ROOT / "state/specialists.yaml",
            frozen_registry_path=ROOT / "ops/frozen_specialist_registry_v1.json",
            failed_experiment_receipt_path=(
                ROOT / "state/slowking_failed_experiment_transition_r79.json"
            ),
            training_service="pokebot-pure-rl-trevenant-staged.service",
        )


def test_release_boundary_accepts_14_frozen_plus_slowking_failed_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "scripts.validate_final_format_alakazam_handoff.subprocess.run",
        lambda argv, check: subprocess.CompletedProcess(argv, 3),
    )
    state = yaml.safe_load((ROOT / "state/specialists.yaml").read_text())
    state_path = tmp_path / "specialists.yaml"
    state_path.write_text(yaml.safe_dump(state), encoding="utf-8")
    (tmp_path / "matchup_adapter_roster.json").write_text(
        (ROOT / "state/matchup_adapter_roster.json").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    registry = json.loads(
        (ROOT / "ops/frozen_specialist_registry_v1.json").read_text()
    )
    registry_path = tmp_path / "frozen.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    result = validate_release_boundary(
        state_path=state_path,
        frozen_registry_path=registry_path,
        failed_experiment_receipt_path=(
            ROOT / "state/slowking_failed_experiment_transition_r79.json"
        ),
        training_service="pokebot-pure-rl-trevenant-staged.service",
    )
    assert result["required_specialist_count"] == 15
    assert result["frozen_specialist_count"] == 14
    assert result["terminal_failed_experiment_exception_count"] == 1
    assert result["slowking_terminal_disposition"] == "failed_experiment"


def test_preferred_parent_metadata_binds_config_without_checkpoint_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_run = tmp_path / "source-run"
    source_run.mkdir()
    profile = {
        "d_model": 96,
        "n_heads": 8,
        "spatial_layers": 4,
        "temporal_layers": 1,
    }
    source_manifest = {
        "run_name": "pure_rl_alakazam_temporal1_8k_teacher_v16_20260721",
        "model_profile": profile,
        "design_contract": {"learner": {"profile": profile}},
    }
    (source_run / "manifest.json").write_text(
        json.dumps(source_manifest), encoding="utf-8"
    )
    frozen_manifest = {
        "schema": "poke_bot.frozen_model/v1",
        "immutable": True,
        "family": "alakazam-owner-accepted-iter39-v1",
        "checkpoint_digest": (
            "sha256:270b5156781b0a95f703abe3e8fe13866"
            "d2fbb4c85a8f32534f99af74aece2ea"
        ),
        "provenance": {"source_run": str(source_run)},
    }
    frozen_path = tmp_path / "manifest.json"
    frozen_path.write_text(json.dumps(frozen_manifest), encoding="utf-8")
    monkeypatch.setattr(
        "scripts.validate_final_format_alakazam_handoff."
        "EXPECTED_PARENT_MANIFEST_SHA256",
        sha256(frozen_path),
    )
    monkeypatch.setattr(
        "scripts.validate_final_format_alakazam_handoff."
        "EXPECTED_SOURCE_RUN_MANIFEST_SHA256",
        sha256(source_run / "manifest.json"),
    )
    monkeypatch.setattr(
        "scripts.validate_final_format_alakazam_handoff."
        "EXPECTED_PARENT_MODEL_CONFIG_SHA256",
        _canonical_json_sha256(profile),
    )

    result = validate_preferred_parent_metadata(frozen_path)

    assert result["model_config_sha256"] == _canonical_json_sha256(profile)
    assert result["model_config_projection_schema"] == (
        "poke_bot.parent_model_profile_projection/v1"
    )
    assert result["checkpoint_loaded"] is False
    assert result["model_instantiated"] is False


def test_preferred_parent_metadata_rejects_profile_disagreement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_run = tmp_path / "source-run"
    source_run.mkdir()
    (source_run / "manifest.json").write_text(
        json.dumps(
            {
                "run_name": (
                    "pure_rl_alakazam_temporal1_8k_teacher_v16_20260721"
                ),
                "model_profile": {"d_model": 96},
                "design_contract": {
                    "learner": {"profile": {"d_model": 64}}
                },
            }
        ),
        encoding="utf-8",
    )
    frozen_path = tmp_path / "manifest.json"
    frozen_path.write_text(
        json.dumps(
            {
                "schema": "poke_bot.frozen_model/v1",
                "immutable": True,
                "family": "alakazam-owner-accepted-iter39-v1",
                "checkpoint_digest": (
                    "sha256:270b5156781b0a95f703abe3e8fe13866"
                    "d2fbb4c85a8f32534f99af74aece2ea"
                ),
                "provenance": {"source_run": str(source_run)},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "scripts.validate_final_format_alakazam_handoff."
        "EXPECTED_PARENT_MANIFEST_SHA256",
        sha256(frozen_path),
    )
    monkeypatch.setattr(
        "scripts.validate_final_format_alakazam_handoff."
        "EXPECTED_SOURCE_RUN_MANIFEST_SHA256",
        sha256(source_run / "manifest.json"),
    )

    with pytest.raises(RuntimeError, match="source model profile changed"):
        validate_preferred_parent_metadata(frozen_path)
