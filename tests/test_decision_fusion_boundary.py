from __future__ import annotations

import json
from pathlib import Path

import torch

from poke_bot import checkpoint, config
from poke_bot.model import build_model
from poke_bot.train import load_model_from_checkpoint
from scripts.apply_decision_fusion_at_boundary import apply_boundary
from scripts.apply_decision_fusion_runtime_at_boundary import (
    apply_boundary as apply_runtime_boundary,
)
from scripts.build_decision_fusion_activation_validation import (
    AUDIT_SCHEMA,
    _numerical_parity,
    build as build_activation_validation,
)
from scripts.materialize_decision_fusion_checkpoint import materialize
from scripts.materialize_decision_fusion_runtime_checkpoint import (
    materialize as materialize_runtime,
)


def _parent(path: Path) -> Path:
    cfg = config.ModelConfig(
        d_model=16,
        spatial_layers=1,
        temporal_layers=1,
        option_decoder_layers=1,
        n_heads=4,
        ff_dim=32,
        max_context=8,
        temporal_pos="rope",
        decision_context="history",
        kv_cache=True,
        expanded_heads_enabled=True,
        dropout=0.0,
    )
    model = build_model(
        cfg,
        aux_archetype_classes=4,
        encoder_vocab=64,
        decoder_vocab=64,
        belief_card_vocab=32,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad]
    )
    optimizer.zero_grad(set_to_none=True)
    sum(parameter.sum() for parameter in model.parameters() if parameter.requires_grad).backward()
    optimizer.step()
    payload = checkpoint.build_checkpoint(
        model=model,
        optimizer=optimizer,
        model_config=cfg,
        step=12,
        epoch=7,
        rl_iteration=4,
        archetype_id="dudunsparce",
        model_id="pure_rl_dudunsparce_test.iter00004",
        extra={
            "pure_rl": True,
            "expanded_head_training": {
                "schema": "poke_bot.expanded_head_training/v1",
                "loss_weights": {},
            },
        },
    )
    return checkpoint.atomic_torch_save(payload, path)


def test_materialize_and_register_zero_safe_fusion_child(tmp_path: Path) -> None:
    parent = _parent(tmp_path / "parent.pt")
    migrated = tmp_path / "fusion-warmup.pt"
    materialization_receipt = tmp_path / "materialization.json"
    report = materialize(
        parent=parent,
        output=migrated,
        receipt=materialization_receipt,
        fusion_width=8,
    )
    assert report["proof"]["legacy_tensors_bit_identical"] is True
    assert report["decision_fusion"]["runtime_enabled"] is False

    parent_payload = checkpoint.load_checkpoint(parent, map_location="cpu")
    migrated_payload = checkpoint.load_checkpoint(migrated, map_location="cpu")
    for key, value in parent_payload["model_state_dict"].items():
        torch.testing.assert_close(
            value, migrated_payload["model_state_dict"][key], rtol=0, atol=0
        )
    loaded = load_model_from_checkpoint(migrated, device=torch.device("cpu"))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in loaded.parameters() if parameter.requires_grad]
    )
    optimizer.load_state_dict(migrated_payload["optimizer_state_dict"])

    parent_digest = checkpoint.checkpoint_digest(parent)
    run_dir = tmp_path / "run"
    (run_dir / "commits").mkdir(parents=True)
    state = {
        "version": 1,
        "run_name": "pure_rl_dudunsparce_test",
        "mode": "specialist",
        "last_completed_iteration": 4,
        "next_iteration": 5,
        "learner": {"path": str(parent), "digest": parent_digest},
        "champion": {"path": str(parent), "digest": parent_digest},
        "heldout_champion": {"path": str(parent), "digest": parent_digest},
    }
    (run_dir / "loop_state.json").write_text(json.dumps(state))
    (run_dir / "commits" / "iter_00004.json").write_text(json.dumps(state))
    activation_receipt = tmp_path / "activation.json"
    result = apply_boundary(
        run_dir=run_dir,
        parent=parent,
        migrated=migrated,
        materialization_receipt=materialization_receipt,
        activation_receipt=activation_receipt,
        expected_last_iteration=4,
    )
    assert result["decision_fusion"]["training_enabled"] is True
    updated = json.loads((run_dir / "loop_state.json").read_text())
    assert updated["learner"]["digest"] == checkpoint.checkpoint_digest(migrated)
    assert updated["champion"] == state["champion"]
    assert updated["heldout_champion"] == state["heldout_champion"]
    assert json.loads(
        (run_dir / "commits" / "iter_00004.json").read_text()
    ) == state


def _audit(
    path: Path,
    *,
    checkpoint_digest: str,
    host: str,
    regression: float = 0.0,
) -> Path:
    influences = {
        name: 0.001 for name in (
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
        )
    }
    payload = {
        "schema": AUDIT_SCHEMA,
        "host": host,
        "checkpoint_digest": checkpoint_digest,
        "device": "cuda",
        "deterministic_signature": {
            "sha256": "sha256:" + "a" * 64,
            "shape": [2, 2],
            "values": [0.1, 0.9, 0.8, 0.2],
            "repeat_bit_exact": True,
        },
        "influence": {
            "required_head_count": len(influences),
            "every_required_head_nonzero": True,
            "per_head_max_abs_ablation_delta": influences,
        },
        "causal_contract": {
            "training_labels_enter_policy_observation": False,
            "hidden_or_future_information_enter_policy_observation": False,
            "matchup_adapter_route_handled_upstream": True,
            "absent_deck_guide_exact_bypass": True,
        },
        "performance": {
            "measured_regression_percent": regression,
            "fused_decisions_per_second": 100.0,
            "oom": False,
            "additional_peak_allocated_bytes": 1024,
        },
    }
    path.write_text(json.dumps(payload))
    return path


def test_validate_materialize_and_register_runtime_fusion_child(
    tmp_path: Path,
) -> None:
    parent = _parent(tmp_path / "parent.pt")
    warmup = tmp_path / "fusion-warmup.pt"
    warmup_receipt = tmp_path / "warmup.json"
    materialize(
        parent=parent,
        output=warmup,
        receipt=warmup_receipt,
        fusion_width=8,
    )
    trained_payload = checkpoint.load_checkpoint(warmup, map_location="cpu")
    trained_payload["model_state_dict"][
        "decision_fusion.residual.2.weight"
    ].fill_(0.01)
    trained = checkpoint.immutable_torch_save(
        trained_payload, tmp_path / "fusion-trained.pt"
    )
    trained_digest = checkpoint.checkpoint_digest(trained)

    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(
        """
specialist_training:
  decision_fusion:
    activation:
      performance_acceptance:
        minimum_fused_decisions_per_second: 50
        maximum_additional_peak_allocated_bytes: 2048
        oom_allowed: false
""".lstrip()
    )
    local_audit = _audit(
        tmp_path / "local.json",
        checkpoint_digest=trained_digest,
        host="inzi",
    )
    remote_audit = _audit(
        tmp_path / "remote.json",
        checkpoint_digest=trained_digest,
        host="elmo",
    )
    validation_path = tmp_path / "activation-validation.json"
    validation = build_activation_validation(
        checkpoint_path=trained,
        parity_audits=[local_audit, remote_audit],
        performance_audit=local_audit,
        protocol_path=protocol,
        output=validation_path,
    )
    assert validation["checks"]["every_required_head_influence_nonzero"] is True

    runtime = tmp_path / "fusion-runtime.pt"
    runtime_receipt = tmp_path / "runtime-materialization.json"
    report = materialize_runtime(
        trained=trained,
        validation_receipt=validation_path,
        output=runtime,
        receipt=runtime_receipt,
    )
    assert report["runtime_enabled"] is True

    run_dir = tmp_path / "run"
    (run_dir / "commits").mkdir(parents=True)
    state = {
        "last_completed_iteration": 5,
        "next_iteration": 6,
        "learner": {"path": str(trained), "digest": trained_digest},
        "champion": {"path": str(parent), "digest": checkpoint.checkpoint_digest(parent)},
        "heldout_champion": {
            "path": str(parent),
            "digest": checkpoint.checkpoint_digest(parent),
        },
    }
    (run_dir / "loop_state.json").write_text(json.dumps(state))
    (run_dir / "commits" / "iter_00005.json").write_text(json.dumps(state))
    activation_path = tmp_path / "runtime-activation.json"
    result = apply_runtime_boundary(
        run_dir=run_dir,
        trained=trained,
        runtime_checkpoint=runtime,
        validation_receipt=validation_path,
        materialization_receipt=runtime_receipt,
        activation_receipt=activation_path,
        expected_last_iteration=5,
    )
    assert result["decision_fusion"]["runtime_enabled"] is True
    updated = json.loads((run_dir / "loop_state.json").read_text())
    assert updated["learner"]["digest"] == checkpoint.checkpoint_digest(runtime)
    assert updated["champion"] == state["champion"]
    assert updated["heldout_champion"] == state["heldout_champion"]
    assert json.loads(
        (run_dir / "commits" / "iter_00005.json").read_text()
    ) == state


def test_cross_cpu_parity_allows_float32_noise_but_requires_same_decisions(
    tmp_path: Path,
) -> None:
    left = {
        "deterministic_signature": {
            "shape": [2, 2],
            "values": [0.1, 0.9, 0.8, 0.2],
        }
    }
    right = {
        "deterministic_signature": {
            "shape": [2, 2],
            "values": [0.10000006, 0.9, 0.8, 0.19999994],
        }
    }
    result = _numerical_parity(
        [(tmp_path / "left.json", left), (tmp_path / "right.json", right)]
    )
    assert result["maximum_absolute_logit_delta"] < 1e-6
    assert result["greedy_decisions_exact"] is True
