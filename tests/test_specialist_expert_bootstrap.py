from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from poke_bot import archetypes, checkpoint, config
from poke_bot.model import CausalDecisionFusion, build_model
from poke_bot.train import expand_aux_head_to_current_registry, load_model_from_checkpoint
from scripts import run_starmie_expert_bootstrap as bootstrap


ROOT = Path(__file__).resolve().parents[1]


def _argv(tmp_path: Path, *, epochs: int) -> list[str]:
    return [
        "--expert-corpus",
        str(tmp_path / "corpus.json"),
        "--archetype",
        "hops-trevenant",
        "--core-family",
        str(tmp_path / "core"),
        "--registry-root",
        str(tmp_path / "registry"),
        "--ready",
        str(tmp_path / "ready.json"),
        "--run-name",
        "trevenant-bootstrap-test",
        "--run-dir",
        str(tmp_path / "run"),
        "--epochs",
        str(epochs),
        "--cpu-pack-root",
        str(tmp_path / "cpu-pack"),
    ]


def test_specialist_bootstrap_rejects_non_exact_epoch_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly 25 epochs"):
        bootstrap.main(_argv(tmp_path, epochs=24))


def test_diagnostic_patience_cannot_end_bootstrap_early() -> None:
    source = (ROOT / "scripts/run_starmie_expert_bootstrap.py").read_text(
        encoding="utf-8"
    )
    assert "if bad_epochs >= int(args.patience)" not in source
    assert "for epoch in range(start_epoch, int(args.epochs) + 1)" in source


def test_bootstrap_guide_receipt_rebind_is_digest_only() -> None:
    old = {
        "schema": "poke_bot.current_deck_guide_handoff/v1",
        "specialist_id": "slowking",
        "strategic_curriculum": {
            "curriculum_spec_sha256": "spec",
            "head_role_map_sha256": "roles",
            "validation_receipt_sha256": "old",
        },
    }
    new = json.loads(json.dumps(old))
    new["strategic_curriculum"]["validation_receipt_sha256"] = "new"
    assert bootstrap.validation_or_evidence_only_rebind(old, new)

    changed = json.loads(json.dumps(new))
    changed["guide_version"] = "changed"
    assert not bootstrap.validation_or_evidence_only_rebind(old, changed)


def test_specialist_bootstrap_materializes_all_head_targets() -> None:
    source = (ROOT / "scripts/run_starmie_expert_bootstrap.py").read_text(
        encoding="utf-8"
    )
    assert "belief_card_vocab_from_state" in source
    assert "belief_card_vocab=belief_card_vocab" in source
    assert "if not corpus.has_exact_targets" in source


def test_rehearsal_expands_historical_archetype_head_before_training() -> None:
    source = (ROOT / "poke_bot/train.py").read_text(encoding="utf-8")
    load = source.index("model = load_model_from_checkpoint(base_path, device=device)")
    expand = source.index(
        "aux_head_expanded = expand_aux_head_to_current_registry(model)",
        load,
    )
    optimizer = source.index("optimizer = torch.optim.AdamW(", expand)
    assert load < expand < optimizer
    assert "warm_started_heads_before" in source[optimizer:]
    assert "warm_started_expanded_before" in source[optimizer:]
    assert "warm_started_fusion_before" in source[optimizer:]
    assert "or aux_head_expanded" in source[optimizer:]


def test_pre_teal_auxiliary_registry_is_supported_for_h10_warm_starts() -> None:
    assert len(archetypes.PRE_TEAL_AUX_ARCHETYPE_IDS) + 1 == 25
    assert archetypes.PRE_TEAL_AUX_ARCHETYPE_IDS[-2:] == (
        "thwackey",
        "team-rockets-spidops",
    )
    model = torch.nn.Module()
    model.aux_head = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.ReLU(),
        torch.nn.Identity(),
        torch.nn.Linear(4, 25),
    )
    model.decision_fusion = CausalDecisionFusion(
        d_model=4,
        width=3,
        archetype_classes=25,
        belief_card_vocab=2,
        dedicated_routes_enabled=True,
        typed_output_centered_routes=True,
    )
    old_weight = model.aux_head[-1].weight.detach().clone()
    old_bias = model.aux_head[-1].bias.detach().clone()
    old_projection = (
        model.decision_fusion.state_projections["archetype"].weight.detach().clone()
    )
    old_route = model.decision_fusion.dedicated_routes[
        "archetype"
    ].network[0].weight.detach().clone()

    assert expand_aux_head_to_current_registry(model) is True
    assert model.aux_head[-1].out_features == len(archetypes.archetype_ids()) + 1
    projection = model.decision_fusion.state_projections["archetype"]
    route = model.decision_fusion.dedicated_routes["archetype"]
    assert projection.in_features == len(archetypes.archetype_ids()) + 1
    assert route.head_dim == len(archetypes.archetype_ids()) + 1
    assert route.network[0].in_features == 4 + len(archetypes.archetype_ids()) + 1
    for old_index, name in enumerate(archetypes.PRE_TEAL_AUX_ARCHETYPE_IDS):
        new_index = archetypes.archetype_ids().index(name)
        assert torch.equal(model.aux_head[-1].weight[new_index], old_weight[old_index])
        assert torch.equal(model.aux_head[-1].bias[new_index], old_bias[old_index])
        assert torch.equal(projection.weight[:, new_index], old_projection[:, old_index])
        assert torch.equal(route.network[0].weight[:, 4 + new_index], old_route[:, 4 + old_index])
    assert torch.equal(model.aux_head[-1].weight[-1], old_weight[-1])
    assert torch.equal(model.aux_head[-1].bias[-1], old_bias[-1])
    assert torch.equal(projection.weight[:, -1], old_projection[:, -1])
    assert torch.equal(route.network[0].weight[:, -1], old_route[:, -1])


def test_generic_entrypoint_uses_the_audited_bootstrap() -> None:
    source = (ROOT / "scripts/run_specialist_expert_bootstrap.py").read_text(
        encoding="utf-8"
    )
    assert "from scripts.run_starmie_expert_bootstrap import main" in source


def test_specialist_hot_start_append_expands_archetype_head(
    tmp_path: Path,
) -> None:
    old_ids = list(archetypes.PINNED_CORE_AUX_ARCHETYPE_IDS)
    target_ids = list(archetypes.archetype_ids())
    width = 4
    old_weight = torch.arange(
        (len(old_ids) + 1) * width, dtype=torch.float32
    ).reshape(len(old_ids) + 1, width)
    old_bias = torch.arange(len(old_ids) + 1, dtype=torch.float32)
    core = tmp_path / "core.pt"
    checkpoint.atomic_torch_save(
        {
            "model_state_dict": {
                "aux_head.3.weight": old_weight,
                "aux_head.3.bias": old_bias,
            },
            "optimizer_state_dict": {"must": "not survive"},
            "step": 99,
            "epoch": 7,
            "rl_iteration": 5,
        },
        core,
    )

    hot_start, hot_digest, expansion = (
        bootstrap._specialist_hot_start_from_core(
            core,
            run_dir=tmp_path / "run",
            archetype="hops-trevenant",
        )
    )
    payload = checkpoint.load_checkpoint(hot_start, map_location="cpu")
    state = payload["model_state_dict"]

    assert state["aux_head.3.weight"].shape == (len(target_ids) + 1, width)
    for old_index, name in enumerate(old_ids):
        new_index = target_ids.index(name)
        assert torch.equal(
            state["aux_head.3.weight"][new_index], old_weight[old_index]
        )
        assert torch.equal(
            state["aux_head.3.bias"][new_index], old_bias[old_index]
        )
    assert torch.equal(state["aux_head.3.weight"][-1], old_weight[-1])
    assert torch.equal(state["aux_head.3.bias"][-1], old_bias[-1])
    assert expansion["newly_initialized_rows"] == [
        name for name in target_ids if name not in old_ids
    ]
    assert expansion["unknown_row_moved_to_final"] is True
    assert payload["step"] == payload["epoch"] == payload["rl_iteration"] == 0
    assert "optimizer_state_dict" not in payload
    assert hot_digest == checkpoint.checkpoint_digest(hot_start)

    repeated_path, repeated_digest, repeated_expansion = (
        bootstrap._specialist_hot_start_from_core(
            core,
            run_dir=tmp_path / "run",
            archetype="hops-trevenant",
        )
    )
    assert repeated_path == hot_start
    assert repeated_digest == hot_digest
    assert repeated_expansion == expansion


def test_specialist_hot_start_expands_cumulative_v4_archetype_order(
    tmp_path: Path,
) -> None:
    old_ids = list(archetypes.CUMULATIVE_V4_AUX_ARCHETYPE_IDS)
    target_ids = list(archetypes.archetype_ids())
    width = 3
    old_weight = torch.arange(
        (len(old_ids) + 1) * width, dtype=torch.float32
    ).reshape(len(old_ids) + 1, width)
    old_bias = torch.arange(len(old_ids) + 1, dtype=torch.float32)
    core = tmp_path / "cumulative-v4.pt"
    checkpoint.atomic_torch_save(
        {
            "model_state_dict": {
                "aux_head.3.weight": old_weight,
                "aux_head.3.bias": old_bias,
            },
            "extra": {
                "matchup_adapter_config": {"expert_ids": old_ids},
            },
        },
        core,
    )

    hot_start, _, expansion = bootstrap._specialist_hot_start_from_core(
        core,
        run_dir=tmp_path / "run",
        archetype="dragapult-dusknoir",
    )
    state = checkpoint.load_checkpoint(
        hot_start, map_location="cpu"
    )["model_state_dict"]

    for old_index, name in enumerate(old_ids):
        new_index = target_ids.index(name)
        assert torch.equal(
            state["aux_head.3.weight"][new_index], old_weight[old_index]
        )
        assert torch.equal(
            state["aux_head.3.bias"][new_index], old_bias[old_index]
        )
    assert torch.equal(state["aux_head.3.weight"][-1], old_weight[-1])
    assert torch.equal(state["aux_head.3.bias"][-1], old_bias[-1])
    assert expansion["source_archetype_ids"] == old_ids
    assert expansion["unknown_row_moved_to_final"] is True


def test_specialist_hot_start_uses_checkpoint_recorded_row_identity(
    tmp_path: Path,
) -> None:
    target_ids = list(archetypes.archetype_ids())
    old_ids = target_ids[:-1]
    width = 3
    old_weight = torch.arange(
        (len(old_ids) + 1) * width, dtype=torch.float32
    ).reshape(len(old_ids) + 1, width)
    old_bias = torch.arange(len(old_ids) + 1, dtype=torch.float32)
    old_fusion = torch.arange(
        2 * (len(old_ids) + 1), dtype=torch.float32
    ).reshape(2, len(old_ids) + 1)
    core = tmp_path / "recorded-row-core.pt"
    checkpoint.atomic_torch_save(
        {
            "model_state_dict": {
                "aux_head.3.weight": old_weight,
                "aux_head.3.bias": old_bias,
                "decision_fusion.state_projections.archetype.weight": (
                    old_fusion
                ),
            },
            "extra": {
                "specialist_aux_archetype_head_expansion": {
                    "schema": bootstrap.SPECIALIST_AUX_EXPANSION_SCHEMA,
                    "target_archetype_ids": old_ids,
                    "target_classes": len(old_ids) + 1,
                }
            },
        },
        core,
    )

    hot_start, _, expansion = bootstrap._specialist_hot_start_from_core(
        core,
        run_dir=tmp_path / "run",
        archetype="hammer-pult",
    )
    state = checkpoint.load_checkpoint(
        hot_start, map_location="cpu"
    )["model_state_dict"]

    assert expansion["source_archetype_ids"] == old_ids
    assert expansion["newly_initialized_rows"] == [target_ids[-1]]
    for old_index, name in enumerate(old_ids):
        new_index = target_ids.index(name)
        assert torch.equal(
            state["aux_head.3.weight"][new_index], old_weight[old_index]
        )
        assert torch.equal(
            state["aux_head.3.bias"][new_index], old_bias[old_index]
        )
    assert torch.equal(state["aux_head.3.weight"][-1], old_weight[-1])
    assert torch.equal(state["aux_head.3.bias"][-1], old_bias[-1])
    fusion = state[
        "decision_fusion.state_projections.archetype.weight"
    ]
    for old_index, name in enumerate(old_ids):
        new_index = target_ids.index(name)
        assert torch.equal(fusion[:, new_index], old_fusion[:, old_index])
    assert torch.equal(fusion[:, -1], old_fusion[:, -1])
    assert torch.equal(fusion[:, target_ids.index(target_ids[-1])], torch.zeros(2))
    fusion_expansion = expansion["decision_fusion_archetype_projection"]
    assert fusion_expansion["new_columns_zero_initialized"] == [
        target_ids[-1]
    ]
    assert fusion_expansion["all_inherited_columns_byte_identical"] is True


def test_v6_hot_start_opts_in_without_changing_inherited_tensors(
    tmp_path: Path,
) -> None:
    target_classes = len(archetypes.archetype_ids()) + 1
    core = tmp_path / "core-v5.pt"
    original = {
        "aux_head.3.weight": torch.arange(
            target_classes * 4, dtype=torch.float32
        ).reshape(target_classes, 4),
        "aux_head.3.bias": torch.arange(
            target_classes, dtype=torch.float32
        ),
        "shared.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
    }
    checkpoint.atomic_torch_save(
        {
            "model_state_dict": original,
            "model_config": {"expanded_heads_enabled": False},
            "optimizer_state_dict": {"legacy": True},
        },
        core,
    )
    _raw, identity = bootstrap.load_expanded_head_contract()

    hot_start, _, expansion = bootstrap._specialist_hot_start_from_core(
        core,
        run_dir=tmp_path / "run",
        archetype="dragapult-dusknoir",
        enable_expanded_heads=True,
        expanded_identity=identity,
    )

    payload = checkpoint.load_checkpoint(hot_start, map_location="cpu")
    assert payload["model_config"]["expanded_heads_enabled"] is True
    assert set(payload["model_state_dict"]) == set(original)
    for key, value in original.items():
        assert torch.equal(payload["model_state_dict"][key], value)
    migration = payload["extra"]["expanded_head_migration"]
    assert migration["schema"] == "poke_bot.expanded_head_migration/v1"
    assert migration["source_checkpoint_digest"] == checkpoint.checkpoint_digest(
        core
    )
    assert migration["target_schema_digest"] == (
        identity["target_schema_digest"]
    )
    assert migration["schedule_digest"] == identity["schedule_digest"]
    assert migration["runtime_enabled_heads"] == []
    assert migration["append_expanded_tensor_keys"] == []
    assert expansion["expanded_head_migration"] == migration
    assert "optimizer_state_dict" not in payload


def test_revision56_hot_start_enables_setup_and_distinct_routes_zero_safely(
    tmp_path: Path,
) -> None:
    target_classes = len(archetypes.archetype_ids()) + 1
    core = tmp_path / "core-v6-fused.pt"
    original = {
        "aux_head.3.weight": torch.zeros(target_classes, 4),
        "aux_head.3.bias": torch.zeros(target_classes),
        "shared.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
    }
    checkpoint.atomic_torch_save(
        {
            "model_state_dict": original,
            "model_config": {
                "expanded_heads_enabled": False,
                "decision_fusion_enabled": False,
            },
        },
        core,
    )
    _raw, identity = bootstrap.load_expanded_head_contract()

    hot_start, _, expansion = bootstrap._specialist_hot_start_from_core(
        core,
        run_dir=tmp_path / "run",
        archetype="archaludon-ex",
        enable_expanded_heads=True,
        expanded_identity=identity,
        enable_decision_fusion=True,
        enable_strategic_curriculum=True,
    )

    payload = checkpoint.load_checkpoint(hot_start, map_location="cpu")
    model_config = payload["model_config"]
    assert model_config["setup_board_outcome_head_enabled"] is True
    assert model_config["decision_fusion_dedicated_routes_enabled"] is True
    assert (
        model_config["decision_fusion_dedicated_routes_runtime_enabled"]
        is True
    )
    assert set(payload["model_state_dict"]) == set(original)
    for key, value in original.items():
        assert torch.equal(payload["model_state_dict"][key], value)
    migration = expansion["decision_fusion_migration"]
    assert migration["target_schema"] == "poke_bot.causal_decision_fusion/v2"
    assert migration["zero_safe_initialization"] is True
    assert migration["one_option_conditioned_route_per_learned_head"] is True


def test_slowking_hot_start_migrates_complete_fusion_v1_additively(
    tmp_path: Path,
) -> None:
    target_classes = len(archetypes.archetype_ids()) + 1
    cfg = config.ModelConfig(
        d_model=16,
        spatial_layers=1,
        temporal_layers=1,
        option_decoder_layers=1,
        n_heads=4,
        ff_dim=32,
        max_context=8,
        expanded_heads_enabled=True,
        decision_fusion_enabled=True,
        decision_fusion_runtime_enabled=True,
        decision_fusion_width=8,
        dropout=0.0,
    )
    source = build_model(
        cfg,
        aux_archetype_classes=target_classes,
        encoder_vocab=64,
        decoder_vocab=64,
        belief_card_vocab=32,
    )
    core = checkpoint.atomic_torch_save(
        checkpoint.build_checkpoint(
            model=source,
            model_config=cfg,
        ),
        tmp_path / "fusion-v1-core.pt",
    )
    _raw, identity = bootstrap.load_expanded_head_contract()

    hot_start, _, expansion = bootstrap._specialist_hot_start_from_core(
        core,
        run_dir=tmp_path / "run",
        archetype="slowking",
        enable_expanded_heads=True,
        expanded_identity=identity,
        enable_decision_fusion=True,
        enable_strategic_curriculum=True,
        enable_combo_state_head=True,
    )

    assert hot_start.name.endswith("fusion-v2-combo-v1.pt")
    migration = expansion["decision_fusion_migration"]
    assert (
        migration["schema"]
        == "poke_bot.causal_decision_fusion_v2_migration/v1"
    )
    assert migration["source_schema"] == "poke_bot.causal_decision_fusion/v1"
    assert migration["target_schema"] == "poke_bot.causal_decision_fusion/v2"
    assert migration["inherited_fusion_tensor_count"] == 30
    assert migration["new_auxiliary_head_names"] == [
        "setup_board_outcome_head",
        "combo_state_head",
    ]
    migrated = load_model_from_checkpoint(
        hot_start,
        device=torch.device("cpu"),
    )
    for key, value in source.state_dict().items():
        torch.testing.assert_close(
            value,
            migrated.state_dict()[key],
            rtol=0,
            atol=0,
        )


def test_expanded_manifest_coverage_allows_masks_but_not_zero_labels(
    tmp_path: Path,
) -> None:
    _, identity = bootstrap.load_expanded_head_contract()
    rows = {
        head_id: {
            "labeled_rows": 1,
            "masked_rows": 9,
            "total_rows": 10,
        }
        for head_id in bootstrap.EXPANDED_HEAD_IDS
    }
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "expanded_strategic_targets": {
                    "schema": identity["target_schema"],
                    "digest": identity["target_schema_digest"],
                    "decisions": 10,
                    "head_coverage": rows,
                }
            }
        ),
        encoding="utf-8",
    )
    result = bootstrap._manifest_expanded_targets(path, decisions=10)
    assert result["head_coverage"]["remaining_turns"]["labeled_rows"] == 1

    rows["remaining_turns"]["labeled_rows"] = 0
    rows["remaining_turns"]["masked_rows"] = 10
    path.write_text(
        json.dumps(
            {
                "expanded_strategic_targets": {
                    "schema": identity["target_schema"],
                    "digest": identity["target_schema_digest"],
                    "decisions": 10,
                    "head_coverage": rows,
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="zero labeled rows"):
        bootstrap._manifest_expanded_targets(path, decisions=10)


def test_canonical_expanded_handoff_pins_exact_cumulative_schedule() -> None:
    contract = bootstrap.expanded_handoff_training_contract()
    schedule = contract["schedule"]

    assert contract["schema"] == "poke_bot.expanded_head_training/v1"
    assert contract["target_schema_digest"].startswith("sha256:")
    assert contract["schedule_digest"].startswith("sha256:")
    assert contract["runtime_enabled_heads"] == []
    assert schedule["total_epochs"] == 25
    assert schedule["stages"][0]["epochs"] == [1, 5]
    assert schedule["stages"][-1]["epochs"] == [21, 25]
    assert set(schedule["stages"][-1]["enabled_heads"]) == set(
        bootstrap.EXPANDED_HEAD_IDS
    )


def test_v6_hot_start_preserves_head_tensors_but_resets_specialist_telemetry(
    tmp_path: Path,
) -> None:
    target_classes = len(archetypes.archetype_ids()) + 1
    state = {
        "aux_head.3.weight": torch.zeros(target_classes, 2),
        "aux_head.3.bias": torch.zeros(target_classes),
    }
    for index, head_id in enumerate(bootstrap.EXPANDED_HEAD_IDS):
        state[f"{head_id}_head.weight"] = torch.full(
            (1, 2), float(index)
        )
        state[f"{head_id}_head.bias"] = torch.full((1,), float(index))
    core = tmp_path / "core-v6.pt"
    checkpoint.atomic_torch_save(
        {
            "model_state_dict": state,
            "model_config": {"expanded_heads_enabled": True},
            "extra": {
                "expanded_head_training": {
                    "schema": "poke_bot.expanded_head_training/v1",
                    "trained_heads": list(bootstrap.EXPANDED_HEAD_IDS),
                }
            },
        },
        core,
    )
    _, identity = bootstrap.load_expanded_head_contract()

    hot_start, _, _ = bootstrap._specialist_hot_start_from_core(
        core,
        run_dir=tmp_path / "run",
        archetype="dudunsparce",
        enable_expanded_heads=True,
        expanded_identity=identity,
    )
    payload = checkpoint.load_checkpoint(hot_start, map_location="cpu")

    assert "expanded_head_training" not in payload["extra"]
    migration = payload["extra"]["expanded_head_migration"]
    assert migration["source_expanded_heads_enabled"] is True
    assert migration["source_expanded_tensor_count"] == (
        2 * len(bootstrap.EXPANDED_HEAD_IDS)
    )
    assert migration["specialist_training_metadata_reset"] is True
    for key, value in state.items():
        assert torch.equal(payload["model_state_dict"][key], value)


def test_h10_successor_reuses_exact_parent_without_remigration(
    tmp_path: Path,
) -> None:
    state = {
        f"decision_fusion.dedicated_routes.route_{index}.weight": torch.ones(1)
        for index in range(19)
    }
    parent = tmp_path / "marnie-h10.pt"
    checkpoint.atomic_torch_save(
        {
            "model_state_dict": state,
            "model_config": {
                "spatial_layers": 7,
                "temporal_layers": 3,
                "option_decoder_layers": 7,
                "ff_dim": 2496,
                "h10_head_residual_width": 512,
                "h10_capacity_enabled": True,
                "expanded_heads_enabled": True,
                "decision_fusion_enabled": True,
                "decision_fusion_runtime_enabled": True,
                "decision_fusion_dedicated_routes_enabled": True,
                "decision_fusion_typed_output_centered_routes_enabled": True,
            },
            "archetype_id": "marnie-s-grimmsnarl-ex",
        },
        parent,
    )

    hot_start, digest, evidence = bootstrap._specialist_hot_start_from_core(
        parent,
        run_dir=tmp_path / "crustle",
        archetype="crustle",
        enable_expanded_heads=True,
        enable_decision_fusion=True,
        enable_strategic_curriculum=True,
        enable_combo_state_head=True,
        allow_h10_specialist_parent=True,
    )

    assert hot_start == parent
    assert digest == checkpoint.checkpoint_digest(parent)
    assert evidence["status"] == "h10_parent_reused_zero_safe"
    assert evidence["all_parent_tensors_reused_byte_exact"] is True
    assert len(evidence["dedicated_fusion_route_names"]) == 19


def test_v6_hot_start_loader_materializes_only_missing_expanded_heads(
    tmp_path: Path,
) -> None:
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
        dense_card2vec=False,
        dropout=0.0,
        expanded_heads_enabled=False,
    )
    source_model = build_model(
        cfg,
        device=torch.device("cpu"),
        aux_archetype_classes=len(archetypes.archetype_ids()) + 1,
        encoder_vocab=64,
        decoder_vocab=64,
        belief_card_vocab=64,
    )
    source_payload = checkpoint.build_checkpoint(
        model=source_model,
        model_config=cfg,
        archetype_id="unknown",
    )
    core = checkpoint.atomic_torch_save(
        source_payload, tmp_path / "real-core-v5.pt"
    )
    _, identity = bootstrap.load_expanded_head_contract()
    hot_start, _, _ = bootstrap._specialist_hot_start_from_core(
        core,
        run_dir=tmp_path / "run",
        archetype="dragapult-dusknoir",
        enable_expanded_heads=True,
        expanded_identity=identity,
    )

    loaded = load_model_from_checkpoint(
        hot_start,
        device=torch.device("cpu"),
    )
    loaded_state = loaded.state_dict()

    assert loaded.expanded_heads_enabled is True
    assert set(loaded.warm_started_expanded_heads) == {
        f"{head_id}_head" for head_id in bootstrap.EXPANDED_HEAD_IDS
    }
    for key, source_tensor in source_payload["model_state_dict"].items():
        assert bootstrap._tensor_bytes_equal(
            loaded_state[key], source_tensor
        )
