from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts.stage_marnie_family_guide_shadow_runtime_r142 import (
    merge_registries,
    stage,
    validate_non_guide_identity,
)


def _family() -> dict:
    return {
        "version": 5,
        "family_manifest": "/family/manifest.json",
        "selected_loss_vector": "/family/vector.json",
        "specialists": {
            "marnie-s-grimmsnarl-ex": {
                "guide_loss_weight": 0.05,
                "guide_training_mode": "strategic_directional_v2",
                "decision_fusion": {"schema": "poke_bot.causal_decision_fusion/v3"},
                "latent_policy": {
                    "schema": "poke_bot.action_conditioned_latent_lookahead/v1",
                    "action_authority_enabled": True,
                },
                "archetype_family": {"enabled": True},
            }
        },
    }


def _retired() -> dict:
    return {
        "version": 4,
        "specialists": {
            "marnie-s-grimmsnarl-ex": {
                "guide_loss_weight": 0.0,
                "guide_retired": True,
                "guide_retirement_revision": 140,
                "guide_target_generation_required": False,
                "guide_conditioned_losses_enabled": False,
                "guide_action_influence": False,
                "guide_historical_artifacts": "audit_only",
            }
        },
    }


def test_merge_preserves_family_and_removes_every_guide_authority() -> None:
    family = _family()
    merged = merge_registries(family, _retired())
    validate_non_guide_identity(family, merged)
    row = merged["specialists"]["marnie-s-grimmsnarl-ex"]
    assert row["archetype_family"] == {"enabled": True}
    assert row["latent_policy"]["action_authority_enabled"] is True
    assert row["guide_loss_weight"] == 0.0
    assert row["guide_shadow_only"] is True
    assert row["guide_shadow_blocking"] is False
    assert row["guide_shadow_runtime_authority"] is False


def test_merge_rejects_noncanonical_parents() -> None:
    family = _family()
    family["specialists"]["marnie-s-grimmsnarl-ex"]["guide_loss_weight"] = 0.04
    with pytest.raises(RuntimeError, match="not mergeable"):
        merge_registries(family, _retired())


def test_stage_materializes_immutable_registry_and_next_start_overlay(
    tmp_path: Path,
) -> None:
    family = tmp_path / "family.json"
    retired = tmp_path / "retired.json"
    family.write_text(json.dumps(_family()) + "\n", encoding="utf-8")
    retired.write_text(json.dumps(_retired()) + "\n", encoding="utf-8")
    python = tmp_path / "python"
    launcher = tmp_path / "launcher.py"
    python.write_text("python\n", encoding="utf-8")
    launcher.write_text("launcher\n", encoding="utf-8")
    args = argparse.Namespace(
        family_registry=family,
        guide_retired_registry=retired,
        output_registry=tmp_path / "merged.json",
        environment_drop_in=tmp_path / "guide-shadow.conf",
        python_executable=python,
        launcher=launcher,
        receipt=tmp_path / "receipt.json",
    )
    receipt = stage(args)
    assert receipt["status"] == "active_next_start_overlay"
    assert receipt["proof"]["guide_weight"] == 0.0
    body = args.environment_drop_in.read_text(encoding="utf-8")
    assert f"--registry {args.output_registry.resolve()}" in body
    assert "--check" in body
    assert stage(args) == receipt
