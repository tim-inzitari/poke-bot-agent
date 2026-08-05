from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from scripts.register_post_marnie_crustle_h10 import (
    MARNIE_ONLY_TOP_LEVEL_KEYS,
    prepare_source_registry,
)
from scripts.validate_post_marnie_crustle_handoff_r113 import validate_corpus_import


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_marnie_completion_cli_bootstraps_canonical_import_root(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/complete_final_format_marnie_refresh.py"),
            "--help",
        ],
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin"},
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Register completed Marnie H10" in completed.stdout


def test_crustle_v2_handoff_verifies_imported_artifact_digests(tmp_path: Path) -> None:
    counts = {"records": 16_639, "decisions": 100_000, "guide_rows": 2_000}
    common = {
        "specialist_id": "crustle",
        "guide_version": "crustle-north-star-v2",
        **counts,
    }
    artifacts = {
        "ready": tmp_path / "CURRENT_DECK_GUIDE_CORPUS_READY.json",
        "protected": tmp_path / "PROTECTED_EXPERT_CORPUS.json",
        "manifest": tmp_path / "manifest.json",
        "finalization": tmp_path / "CRUSTLE_GUIDE_CORPUS_VALIDATED.json",
    }
    _write_json(
        artifacts["ready"],
        {"schema": "poke_bot.current_deck_guide_corpus_ready/v1", "status": "ready", **common},
    )
    _write_json(artifacts["protected"], {"schema": "test/protected", **common})
    _write_json(artifacts["manifest"], {"schema": "test/manifest", **common})
    _write_json(
        artifacts["finalization"],
        {
            "schema": "poke_bot.crustle_guide_corpus_validation/v1",
            "status": "ready_checksum_validated",
            **common,
        },
    )
    receipt_path = tmp_path / "import.json"
    _write_json(
        receipt_path,
        {
            "schema": "poke_bot.prestaged_specialist_corpus_import/v1",
            "status": "ready",
            "active_training_modified": False,
            "destination": str(tmp_path),
            "finalization_receipt_name": artifacts["finalization"].name,
            "ready_receipt_sha256": _sha256(artifacts["ready"]),
            "protected_pointer_sha256": _sha256(artifacts["protected"]),
            "manifest_sha256": _sha256(artifacts["manifest"]),
            "finalization_receipt_sha256": _sha256(artifacts["finalization"]),
            **common,
        },
    )

    validated = validate_corpus_import(receipt_path)
    assert validated["status"] == "ready_checksum_validated_imported"
    assert validated["records"] == 16_639

    artifacts["manifest"].write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest sha256"):
        validate_corpus_import(receipt_path)


def test_crustle_chain_is_exactly_after_marnie_and_before_population() -> None:
    state = yaml.safe_load((ROOT / "state/specialists.yaml").read_text())
    phase = state["post_fleet_refresh"]
    assert phase["ordered_specialist_ids"] == [
        "alakazam",
        "marnie-s-grimmsnarl-ex",
        "crustle",
    ]
    crustle = next(row for row in state["specialists"] if row["id"] == "crustle")
    assert crustle["required_specialist"] is False
    assert crustle["post_fleet_specialist_required"] is True
    assert crustle["model_contract"]["capacity_profile"] == "H10-I/v1"
    assert crustle["model_contract"]["decision_fusion_schema"] == (
        "poke_bot.causal_decision_fusion/v3"
    )
    assert crustle["public_practice_gate_opponent"]["inference_only"] is True


def test_crustle_v2_corpus_is_fail_closed_before_checksum_reaudit() -> None:
    guide = yaml.safe_load((ROOT / "config/deck_guides/crustle.yaml").read_text())
    state = yaml.safe_load((ROOT / "state/specialists.yaml").read_text())
    crustle = next(row for row in state["specialists"] if row["id"] == "crustle")
    rebind = crustle["heads"]["current_deck_guide"][
        "public_guide_corpus_pipeline"
    ]["v2_rebind"]
    assert guide["guide_version"] == "crustle-north-star-v2"
    assert rebind["guide_version"] == "crustle-north-star-v2"
    assert rebind["training_authority_before_validation"] is False
    assert rebind["output_root"].endswith("crustle-guide-corpus-family-full33-v2")


def test_managed_units_preserve_96_workers_and_terminal_chain() -> None:
    units = ROOT / "deploy/systemd"
    marnie_completion = (
        units / "pokebot-final-format-marnie-r104-completion.service"
    ).read_text()
    latest20_override = (
        units
        / "pokebot-final-format-marnie-r104-h10-rl.service.d"
        / "zz-latest20-r109.conf"
    ).read_text()
    bootstrap = (units / "pokebot-final-format-crustle-r113-h10-bootstrap.service").read_text()
    register = (units / "pokebot-final-format-crustle-r113-h10-register.service").read_text()
    completion_rebind = (
        units
        / "pokebot-final-format-marnie-r104-completion.service.d"
        / "zzzzzzz-crustle-handoff-r143.conf"
    ).read_text()
    trainer = (units / "pokebot-final-format-crustle-r113-h10-rl.service").read_text()
    completion = (units / "pokebot-final-format-crustle-r113-completion.service").read_text()
    capacity = (units / "pokebot-post-refresh-capacity-boundary-r104.service").read_text()
    assert "--next-service pokebot-final-format-crustle-r113-h10-bootstrap.service" in marnie_completion
    assert "--next-service pokebot-final-format-crustle-r113-h10-bootstrap.service" in latest20_override
    assert "--next-service pokebot-post-refresh-capacity-boundary-r104.service" not in latest20_override
    assert "ConditionPathExists=/home/inzi/poke-bot-agent/outputs/state/final-format-marnie-r104-h10-completion-v1.json" in bootstrap
    assert "OnSuccess=pokebot-final-format-crustle-r113-h10-register.service" in bootstrap
    assert "OnSuccess=pokebot-final-format-crustle-r113-h10-rl.service" in register
    assert "--marnie-completion" in register
    assert "--source-registry" not in register
    assert "--runtime-registry-service pokebot-final-format-marnie-r104-h10-rl.service" in completion_rebind
    assert "complete_final_format_marnie_refresh.py --handler-state" in completion_rebind
    assert "launch_active_specialist.py --registry" not in completion_rebind
    assert "PURE_RL_SIM_WORKERS=96" in trainer
    assert "PURE_RL_GAMES_IN_FLIGHT=96" in trainer
    assert "POKEBOT_LIVE_POOL_MAX_WORKERS=96" in trainer
    assert "OnSuccess=pokebot-final-format-crustle-r113-h10-gate-handler.service" in trainer
    assert "pokebot-post-refresh-capacity-boundary-r104.service" in completion
    assert "--crustle-completion" in capacity


def test_crustle_registration_binds_then_current_marnie_registry() -> None:
    source = (ROOT / "scripts/register_post_marnie_crustle_h10.py").read_text()
    assert 'completion.get("runtime_registry")' in source
    assert 'sha256(source_path) != completion.get("runtime_registry_sha256")' in source
    assert 'marnie.get("guide_loss_weight")' in source
    assert "MARNIE_ONLY_TOP_LEVEL_KEYS" in source
    assert 'source["post_marnie_runtime_rebind"]' in source


def test_crustle_source_derivative_removes_marnie_only_runtime_fields(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "current-marnie.json"
    source = {
        "schema": "poke_bot.specialist_runtime_registry/v1",
        "runtime_root": str(tmp_path),
        "common_trainer_args": [
            "--boundary-design-migration-reason",
            "owner_ceiling_marnie_family_activation_r139",
        ],
        "specialists": {
            "marnie-s-grimmsnarl-ex": {
                "status": "ready",
                "guide_loss_weight": 0.0,
                "decision_fusion": {
                    "schema": "poke_bot.causal_decision_fusion/v3",
                    "runtime_enabled": True,
                },
            }
        },
        **{key: {"marnie_only": True} for key in MARNIE_ONLY_TOP_LEVEL_KEYS},
    }
    _write_json(source_path, source)
    completion = tmp_path / "completion.json"
    _write_json(
        completion,
        {
            "schema": "poke_bot.post_fleet_specialist_refresh_completion/v1",
            "specialist_id": "marnie-s-grimmsnarl-ex",
            "frozen": True,
            "registered": True,
            "runtime_registry": str(source_path),
            "runtime_registry_sha256": _sha256(source_path),
            "refresh_checkpoint_checksum": "sha256:" + "1" * 64,
        },
    )
    frozen = tmp_path / "frozen.json"
    gate = tmp_path / "gate.json"
    _write_json(frozen, {"schema": "test/frozen"})
    _write_json(gate, {"schema": "test/gate"})
    stage = tmp_path / "stage.json"
    _write_json(
        stage,
        {
            "schema": "poke_bot.crustle_marnie_expert_practice_stage/v1",
            "status": "ready_for_post_marnie_crustle_bootstrap",
            "marnie_completion_sha256": _sha256(completion),
            "checkpoint_digest": "sha256:" + "1" * 64,
            "frozen_registry": str(frozen),
            "frozen_registry_relative": "frozen.json",
            "frozen_registry_sha256": _sha256(frozen),
            "active_gate": str(gate),
            "active_gate_relative": "gate.json",
            "active_gate_sha256": _sha256(gate),
            "active_gate_id": "test-crustle-r160",
        },
    )
    prepared, resolved = prepare_source_registry(completion, stage)
    assert resolved == source_path.resolve()
    assert not (MARNIE_ONLY_TOP_LEVEL_KEYS & prepared.keys())
    assert prepared["common_trainer_args"][-1] == (
        "post_marnie_crustle_h10_registration_r143"
    )
    assert prepared["post_marnie_runtime_rebind"]["source_registry_sha256"] == (
        _sha256(source_path)
    )
    assert prepared["frozen_specialist_registry"] == "frozen.json"
    assert prepared["active_gate_contract"] == "gate.json"


def test_crustle_revision164_keeps_weighted_guide_active_for_all_epochs_and_rl() -> None:
    bootstrap = (ROOT / "scripts/run_post_marnie_crustle_h10_bootstrap.py").read_text()
    common = (ROOT / "scripts/run_starmie_expert_bootstrap.py").read_text()
    register = (ROOT / "scripts/register_post_marnie_crustle_h10.py").read_text()
    runtime_register = (ROOT / "scripts/register_next_specialist_runtime.py").read_text()
    unit = (
        ROOT
        / "deploy/systemd/pokebot-final-format-crustle-r113-h10-bootstrap.service"
    ).read_text()
    launcher = (ROOT / "scripts/launch_active_specialist.py").read_text()
    assert '"--epochs", "35"' in bootstrap
    assert '"--owner-crustle-guide-active-epochs", "35"' in bootstrap
    assert '"--owner-crustle-guide-free-refresh-epochs", "0"' in bootstrap
    assert '"guide_free_expert_refresh_epochs": []' in common
    assert '"expert_pilot_importance_epochs": [1, 35]' in common
    # The 35-epoch Crustle extension retains the final all-head plan after the
    # canonical 25-epoch head-ramp, rather than falling back to zero weights.
    assert "min(epoch, 25) if crustle_extended_refresh else epoch" in common
    assert "set(plan.enabled_heads) == set(EXPANDED_HEAD_IDS)" in common
    assert '"training_game_sampling_weights": pilot_weights' in common
    assert '"--pilot-importance-index"' in bootstrap
    assert "crustle-r161-expert-pilot-importance.json" in unit
    assert "materialize_expert_pilot_importance.py targets" in unit
    assert "ensure_expert_pilot_map.py" in unit
    assert "materialize_expert_pilot_importance.py finalize" in unit
    assert "or int(ready.get(\"epochs_completed\") or 0) != 35" in register
    assert "guide_loss_weight=0.05" in register
    assert "guide_retired_after_bootstrap=False" in register
    assert "_persistent_crustle_guide_policy" in runtime_register
    assert '"automatic_decay_allowed": False' in runtime_register
    assert "stage_crustle_h10_marnie_opponent_r160.py" in unit
    assert "completed Marnie H10 is absent from Crustle practice/holdout" in launcher


def test_marnie_iteration20_boundary_cannot_terminate_at_iteration5() -> None:
    stage = (ROOT / "scripts/stage_marnie_iteration20_r113.py").read_text()
    watcher = (
        ROOT
        / "deploy/systemd/pokebot-marnie-opponent-tiers-r111-activate.service"
    ).read_text()
    trainer = (
        ROOT
        / "deploy/systemd/pokebot-final-format-marnie-r104-h10-rl.service"
    ).read_text()
    assert 'candidate["minimum_terminal_iteration"] = 20' in stage
    assert (
        'candidate["specialists"][SPECIALIST_ID]'
        '["minimum_terminal_iteration"] = 20'
    ) in stage
    assert "iteration20-stage-r113-v3.json" in watcher
    assert "registry_h10_r113_iter20_v3.json" in watcher
    assert 'isolated["self_play_games_per_iteration"] = 1024' in stage
    assert 'isolated["public_opponent_games_per_iteration"] = 7168' in stage
    assert 'isolated["premium_skill_weighted_win_rate"] = 0.80' in stage
    assert 'isolated["premium_skill_weighted_confidence_lower"] = 0.50' in stage
    assert "--next-service pokebot-final-format-crustle-r113-h10-bootstrap.service" in trainer


def test_bootstrap_retains_parent_h10_combo_route_without_slowking_authority() -> None:
    source = (ROOT / "scripts/run_post_marnie_crustle_h10_bootstrap.py").read_text()
    assert '"--allow-h10-specialist-parent"' in source
    assert '"--retain-inherited-h10-combo-state-head"' in source
    assert "include_combo_state=True" in source
    assert "--combo-state-implementation-receipt" not in source
