"""Focused offline-only tests for the Prize-plan-v2 trainer lane."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import runpy

import pytest

torch = pytest.importorskip("torch")

from poke_bot.recursive_turn_planner.recent20_overlay import sha256_file
from scripts import train_alakazam_prize_plan_v2 as trainer


def _digest(seed: str) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _identity() -> dict[str, object]:
    return {
        "utc_day": "2026-07-23",
        "source_archive_sha256": _digest("archive"),
        "source_member": "episode-000.json",
        "episode_id": "episode-000",
        "acting_seat": 0,
        "env_step": 11,
        "program_identity": "program-000",
    }


def _target_row() -> dict[str, object]:
    row: dict[str, object] = {
        "schema": trainer.TARGET_ROW_SCHEMA,
        "owner_goal_revision": 23,
        "goal_contract_goal_revision": 24,
        **_identity(),
        "split": "train",
        "causal_segments": [],
        "target_only": True,
        "hidden_information_fields_present": False,
        "terminal_z_present": False,
        "raw_target_provenance": {},
    }
    returns: dict[str, object] = {}
    for horizon, raw in ((1, 0.25), (3, -0.5), (6, 0.75), (12, -1.0)):
        returns[f"h{horizon}"] = {
            "h": horizon,
            "gamma": 1.0,
            "required_segment_count": horizon,
            "segment_count": horizon,
            "mask": True,
            "unavailable_reason": None,
            "raw_return_value": raw,
            "model_target_value": raw / 2.0,
        }
    row["prize_plan_returns"] = returns
    return row


def _sample() -> dict[str, object]:
    program = {
        **_identity(),
        "stages": [
            {
                "factorized_stage": 0,
                "ordered_legal_action_programs": [[1, 2], [3]],
                "valid_option_mask": [True, True],
                "selected_option_index": 0,
                "selected_action_program": [1, 2],
            }
        ],
    }
    return {
        "public_information_only": True,
        "base_feature_width": 40,
        "program": program,
        "base_option_features_by_stage": [[[-0.25] * 40, [0.75] * 40]],
    }


def _row() -> trainer.CompleteActionExample:
    return trainer.example_from_join(_sample(), _target_row())


def test_current_wrapper_accepts_embedded_r23_authority() -> None:
    contract = trainer.ROOT / "goals/alakazam-elmo-rule-derivative/contract.json"
    _, digest, revision = trainer.load_bound_contract(
        contract, expected_sha256=sha256_file(contract), production=True
    )
    assert digest == sha256_file(contract)
    assert revision >= 23


def test_target_validator_checks_analytic_transform_and_never_accepts_terminal_label() -> None:
    target_set = object.__new__(trainer.PrizePlanTargetSet)
    target_set.gamma = 1.0
    target_set.contract_goal_revision = 24
    valid = _target_row()
    target_set.validate_row(valid, day="2026-07-23", split="train")

    wrong_transform = copy.deepcopy(valid)
    wrong_transform["prize_plan_returns"]["h3"]["model_target_value"] = 0.0
    with pytest.raises(trainer.PrizePlanV2TrainingError, match=r"raw/\(1\+gamma\*\*h\)"):
        target_set.validate_row(wrong_transform, day="2026-07-23", split="train")

    terminal = copy.deepcopy(valid)
    terminal["terminal_z"] = 1.0
    with pytest.raises(trainer.PrizePlanV2TrainingError, match="forbidden terminal"):
        target_set.validate_row(terminal, day="2026-07-23", split="train")


def test_exact_seven_field_join_keeps_only_model_target_values() -> None:
    example = trainer.example_from_join(_sample(), _target_row())
    assert example.identity == tuple(_identity()[field] for field in trainer.IDENTITY_FIELDS)
    assert example.plan_targets == pytest.approx((0.125, -0.25, 0.375, -0.5))
    assert not hasattr(example, "raw_return_value")
    assert not hasattr(example, "terminal_z")

    mismatched = _target_row()
    mismatched["source_member"] = "different-member.json"
    with pytest.raises(trainer.PrizePlanV2TrainingError, match="seven-field join"):
        trainer.example_from_join(_sample(), mismatched)


def test_train_only_h3_scale_support_is_unclipped_and_actor_dormant() -> None:
    first = _row()
    second = trainer.CompleteActionExample(
        **{
            **first.__dict__,
            "identity": ("2026-07-24", *first.identity[1:]),
            "utc_day": "2026-07-24",
            "plan_targets": (0.0, 0.5, 0.0, 0.0),
            "plan_masks": (False, True, False, False),
        }
    )
    source = {"sealed": "fixture", "target_consumption": "model_target_value_only"}
    model, _ = trainer._build_model(hidden_width=8)
    artifact = trainer.fit_h3_scale_support(
        model,
        [first, second],
        device=torch.device("cpu"),
        batch_size=2,
        source_binding=source,
        sidecar_checkpoint={"path": "/sealed/fixture-sidecar.pt", "sha256": _digest("sidecar")},
    )
    assert artifact["fit_split"] == "train"
    assert artifact["unclipped"] is True
    assert artifact["actor_activation"] is False
    assert artifact["actor_integration_required_for_critic_loss"] is False
    assert artifact["raw_return_value_used_as_fit_target"] is False
    assert artifact["model_target_value_used_as_scale_target"] is False
    assert artifact["frozen_sidecar_checkpoint"]["sha256"] == _digest("sidecar")
    assert "Q_plan_3(s,a)-V_plan_3(s)" in artifact["action_advantage_definition"]
    assert artifact["future_actor_integration_eligible"] is False
    assert artifact["support_diagnostic"]["h3_labeled_complete_actions"] == 2
    assert artifact["artifact_sha256"] == trainer._json_sha(
        {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    )


def test_sidecar_batch_and_loss_use_chosen_complete_actions() -> None:
    row = _row()
    model, _config = trainer._build_model(hidden_width=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    metrics, steps = trainer.run_batches(
        model,
        [row, row],
        device=torch.device("cpu"),
        batch_size=2,
        optimizer=optimizer,
        grad_clip=1.0,
    )
    assert steps == 1
    assert metrics.summary()["complete_actions"] == 2
    assert metrics.summary()["coverage"]["plan_horizons"]["3"]["available"] == 2


def test_target_view_requires_exact_transfer_graph_and_production_elmo_source(
    tmp_path: Path,
) -> None:
    """The trainer derives portable target paths only through the receipt graph."""

    helpers = runpy.run_path(
        str(trainer.ROOT / "tests/test_transfer_alakazam_prize_plan_v2_targets.py")
    )
    fixture = helpers["_fixture"](tmp_path)
    plan, plan_sha = helpers["_plan"](fixture)
    transfer = helpers["transfer"]
    result = transfer.execute_prize_plan_v2_transfer_plan(
        plan,
        source=transfer.LocalSourceReader(),
        plan_sha256=plan_sha,
        free_bytes=lambda _path: 10**12,
        test_allow_non_elmo_source=True,
    )
    resolved = trainer.resolve_target_view(
        pointer=result["target_view_path"],
        expected_sha256=result["target_view_sha256"],
        contract_sha256=fixture["contract_sha"],
        contract_goal_revision=24,
        overlay_manifest_sha256=plan["source"]["complete_action_overlay_manifest_sha256"],
        production=False,
    )
    assert resolved["plan_sha256"] == plan_sha
    assert resolved["completion_sha256"].startswith("sha256:")
    assert resolved["entry_count"] == len(plan["entries"])
    assert resolved["private_partials_not_training_eligible"] is True

    # The hermetic fixture intentionally marks a non-Elmo source.  Production
    # resolution must reject it even though all target bytes/receipts validate.
    with pytest.raises(trainer.PrizePlanV2TrainingError, match="Elmo source plan"):
        trainer.resolve_target_view(
            pointer=result["target_view_path"],
            expected_sha256=result["target_view_sha256"],
            contract_sha256=fixture["contract_sha"],
            contract_goal_revision=24,
            overlay_manifest_sha256=plan["source"]["complete_action_overlay_manifest_sha256"],
            production=True,
        )
