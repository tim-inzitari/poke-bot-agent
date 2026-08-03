from __future__ import annotations

import inspect
import hashlib
from pathlib import Path

import pytest
import torch

from poke_bot.train import BatchMetrics, _merge_metrics, supervised_rehearsal_step
from scripts.run_final_format_marnie_h10_bootstrap import (
    _combo_rows,
    _directional_rows,
)
from scripts.register_final_format_marnie_h10_rl import (
    _materialize_adapter_authorization,
    _route_reliability_telemetry,
    _validate_runtime_assets,
    _validate_selected_bootstrap_training,
)
from poke_bot.matchup_adapter_activation import (
    validate_adapter_training_authorization,
)
from poke_bot.strategic_schedule import EXPANDED_HEAD_IDS


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_marnie_adapter_authorization_binds_inherited_training_evidence(
    tmp_path: Path,
) -> None:
    expert = tmp_path / "expert.pt"
    router = tmp_path / "router.pt"
    expert.write_bytes(b"25-epoch-marnie")
    router.write_bytes(b"router-v6-marnie")

    authorization = _materialize_adapter_authorization(
        state_root=tmp_path / "state",
        router_checkpoint=router,
        router_checkpoint_digest=_digest(router),
        expert_checkpoint=expert,
        expert_checkpoint_digest=_digest(expert),
    )
    proof = validate_adapter_training_authorization(
        authorization,
        parent_checkpoint=router,
        permit_post_boundary_use=True,
    )

    assert proof.parent_checkpoint == router.resolve()
    assert proof.completed_iteration == -1


def test_marnie_directional_bootstrap_requires_all_five_guided_routes() -> None:
    metrics = {
        "guide_curriculum_head_metrics": {
            "directional_route_ranking": {
                "eligible_rows": 17,
                "heads": [
                    "action_q",
                    "action_resource",
                    "action_utility",
                    "setup_board_outcome",
                    "combo_state",
                ],
            }
        }
    }
    assert _directional_rows(metrics) == 17


def test_marnie_directional_bootstrap_rejects_missing_combo_route() -> None:
    metrics = {
        "guide_curriculum_head_metrics": {
            "directional_route_ranking": {
                "eligible_rows": 17,
                "heads": [
                    "action_q",
                    "action_resource",
                    "action_utility",
                    "setup_board_outcome",
                ],
            }
        }
    }
    with pytest.raises(RuntimeError, match="omitted a required Marnie route"):
        _directional_rows(metrics)


def test_directional_route_metrics_survive_multi_batch_merge() -> None:
    merged = _merge_metrics(
        [
            BatchMetrics(
                n_decisions=2,
                guide_curriculum_head_metrics={
                    "total": {"action_q": 2},
                    "labeled": {"action_q": 2},
                    "losses": {"action_q": 0.25},
                    "directional_route_ranking": {
                        "eligible_rows": 2,
                        "heads": {
                            "action_q": 0.5,
                            "action_resource": 0.6,
                            "action_utility": 0.7,
                            "setup_board_outcome": 0.8,
                            "combo_state": 0.9,
                        },
                        "margin": 0.1,
                    },
                },
            ),
            BatchMetrics(
                n_decisions=3,
                guide_curriculum_head_metrics={
                    "total": {"action_q": 3},
                    "labeled": {"action_q": 3},
                    "losses": {"action_q": 0.5},
                    "directional_route_ranking": {
                        "eligible_rows": 3,
                        "heads": {
                            "action_q": 1.0,
                            "action_resource": 1.1,
                            "action_utility": 1.2,
                            "setup_board_outcome": 1.3,
                            "combo_state": 1.4,
                        },
                        "margin": 0.1,
                    },
                },
            ),
        ]
    )
    directional = merged.guide_curriculum_head_metrics[
        "directional_route_ranking"
    ]
    assert directional["eligible_rows"] == 5
    assert set(directional["heads"]) == {
        "action_q",
        "action_resource",
        "action_utility",
        "setup_board_outcome",
        "combo_state",
    }
    assert directional["heads"]["action_q"] == pytest.approx(0.8)


def test_marnie_combo_rows_preserve_valid_all_masked_corpus() -> None:
    assert _combo_rows(
        {"combo_state_metrics": {"total_rows": 101, "eligible_rows": 0}}
    ) == 0


def test_marnie_combo_rows_reject_invalid_masking_telemetry() -> None:
    with pytest.raises(RuntimeError, match="masking telemetry is invalid"):
        _combo_rows(
            {"combo_state_metrics": {"total_rows": 3, "eligible_rows": 4}}
        )


def test_marnie_bootstrap_can_reset_inherited_core_head_history() -> None:
    parameter = inspect.signature(supervised_rehearsal_step).parameters[
        "reset_expanded_training_history"
    ]
    assert parameter.default is False

    source = inspect.getsource(
        __import__(
            "scripts.run_final_format_marnie_h10_bootstrap",
            fromlist=["main"],
        ).main
    )
    assert "reset_expanded_training_history=(epoch == 1)" in source


def _selected_payload(epoch: int = 16) -> dict:
    heads = {
        name: {
            "present": True,
            "gradient_enabled": True,
            "trained": True,
            "trained_this_epoch": True,
            "train_labeled_rows": 10,
            "validation_labeled_rows": 2,
        }
        for name in EXPANDED_HEAD_IDS
    }
    return {
        "extra": {
            "final_format_marnie_h10_bootstrap": {
                "schema": "poke_bot.final_format_marnie_h10_bootstrap_epoch/v1",
                "epoch": epoch,
                "guide_mode": "strategic_directional_v2",
                "guide_weight": 0.05,
                "decision_fusion_schema": "poke_bot.causal_decision_fusion/v3",
            },
            "expanded_head_training": {
                "schema": "poke_bot.expanded_head_training/v1",
                "epoch": epoch,
                "epochs_total": 25,
                "gradient_enabled_heads": list(EXPANDED_HEAD_IDS),
                "trained_this_epoch": list(EXPANDED_HEAD_IDS),
                "heads": heads,
            },
        }
    }


def test_marnie_registration_accepts_only_full_head_selection_epochs() -> None:
    _validate_selected_bootstrap_training(_selected_payload())

    incomplete = _selected_payload(epoch=15)
    with pytest.raises(RuntimeError, match="full-head epoch evidence"):
        _validate_selected_bootstrap_training(incomplete)


def test_marnie_registration_rejects_missing_per_head_training_evidence() -> None:
    incomplete = _selected_payload()
    incomplete["extra"]["expanded_head_training"]["heads"][
        "remaining_turns"
    ]["validation_labeled_rows"] = 0
    with pytest.raises(RuntimeError, match="remaining_turns"):
        _validate_selected_bootstrap_training(incomplete)


def test_marnie_registration_records_exact_effective_route_reliabilities() -> None:
    payload = {
        "model_state_dict": {
            "decision_fusion.dedicated_route_log_reliability." + name:
            torch.tensor(0.0)
            for name in (
                "value",
                "archetype",
                "opponent_hand",
                "opponent_remainder",
                "lethal_threat",
                "prize_race",
                "action_q",
                "action_type",
                "action_target",
                "action_resource",
                "action_utility",
                "tactical_outcomes",
                "opponent_response",
                "resource_forecast",
                "game_phase",
                "outcome_distribution",
                "remaining_turns",
                "setup_board_outcome",
                "combo_state",
            )
        }
    }
    telemetry = _route_reliability_telemetry(payload)
    assert telemetry["value"] == pytest.approx(1.0)
    assert telemetry["action_type"] == pytest.approx(0.25)
    assert len(telemetry) == 19


def test_marnie_registration_rejects_nonfinite_route_reliability() -> None:
    payload = {
        "model_state_dict": {
            "decision_fusion.dedicated_route_log_reliability." + name:
            torch.tensor(float("nan") if name == "action_q" else 0.0)
            for name in (
                "value",
                "archetype",
                "opponent_hand",
                "opponent_remainder",
                "lethal_threat",
                "prize_race",
                "action_q",
                "action_type",
                "action_target",
                "action_resource",
                "action_utility",
                "tactical_outcomes",
                "opponent_response",
                "resource_forecast",
                "game_phase",
                "outcome_distribution",
                "remaining_turns",
                "setup_board_outcome",
                "combo_state",
            )
        }
    }
    with pytest.raises(RuntimeError, match="not finite: action_q"):
        _route_reliability_telemetry(payload)


def test_marnie_registration_requires_checksum_bound_ladder_mix(
    tmp_path, monkeypatch
) -> None:
    deployment = tmp_path / "deployment"
    import scripts.register_final_format_marnie_h10_rl as registration

    expected = {}
    for relative in (
        "submission/main.py",
        "submission/search_config.json",
        "data/training_mixes/top_ladder.v1.json",
        "data/training_mixes/top_ladder_representatives.v1.json",
        "data/training_mixes/specialist_representatives.v1.json",
    ):
        path = deployment / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'{{"schema":"{relative}"}}\n', encoding="utf-8")
        expected[relative] = registration._sha256(path)
    monkeypatch.setattr(registration, "RUNTIME_ASSET_SHA256", expected)
    assert _validate_runtime_assets(deployment) == expected
    mix = deployment / "data/training_mixes/top_ladder.v1.json"
    mix.write_text('{"schema":"changed"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="runtime training-mix asset"):
        _validate_runtime_assets(deployment)
