from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from poke_bot import archetypes, checkpoint, config
from poke_bot.matchup_adapters import MatchupAdapterBank
from poke_bot.model import build_model
from scripts import run_multi_teacher_core_refresh as refresh
from scripts.run_post_starmie_core_handoff import _core_refresh_command
from scripts.run_starmie_expert_bootstrap import (
    decision_fusion_handoff_contract,
    expanded_handoff_training_contract,
)


def _checkpoint(
    path: Path,
    value: float,
    *,
    archetype: str,
    aux_rows: int | None = None,
    aux_ids: list[str] | None = None,
) -> Path:
    torch.manual_seed(17)
    bank = MatchupAdapterBank(enabled=False)
    with torch.no_grad():
        bank.experts[0].up.bias.fill_(value)
    state = {
        "encoder.weight": torch.full((3, 4), value),
        "policy_head.weight": torch.full((2, 4), value * 2),
        "integer_buffer": torch.tensor([7], dtype=torch.int64),
        **{
            f"matchup_adapter_bank.{name}": tensor.detach().clone()
            for name, tensor in bank.state_dict().items()
        },
    }
    if aux_rows is not None:
        state["aux_head.3.weight"] = torch.full((aux_rows, 4), value)
        state["aux_head.3.bias"] = torch.full((aux_rows,), value)
    checkpoint.immutable_torch_save(
        {
            "model_state_dict": state,
            "model_config": {"matchup_adapters_enabled": False},
            "archetype_id": archetype,
            "optimizer_state_dict": {"state": {"must": "be removed"}},
            "extra": {
                "specialist_only": True,
                **(
                    {
                        "specialist_aux_archetype_head_expansion": {
                            "target_archetype_ids": aux_ids,
                        }
                    }
                    if aux_ids is not None
                    else {}
                ),
            },
        },
        path,
    )
    return path


def test_additive_head_architecture_differences_retain_target_shape() -> None:
    target = {
        "encoder.weight": torch.full((2, 2), 3.0),
        "action_q_head.weight": torch.full((1, 2), 5.0),
    }
    legacy = {
        "model_state_dict": {
            "encoder.weight": torch.full((2, 2), 7.0),
            "decision_fusion.residual.0.weight": torch.full((2, 2), 9.0),
        }
    }

    aligned, adaptations = refresh._align_teacher_state(
        legacy,
        target_state=target,
        target_aux_ids=None,
    )

    assert set(aligned) == set(target)
    assert torch.equal(
        aligned["encoder.weight"],
        legacy["model_state_dict"]["encoder.weight"],
    )
    assert torch.equal(
        aligned["action_q_head.weight"],
        target["action_q_head.weight"],
    )
    assert {
        (row["tensor"], row["behavior"]) for row in adaptations
    } == {
        (
            "action_q_head.weight",
            "retain_initialization_for_absent_additive_head",
        ),
        (
            "decision_fusion.residual.0.weight",
            "omit_additive_head_absent_from_initialization_architecture",
        ),
    }


def test_multi_teacher_initialization_is_exact_mean_with_zero_adapters(
    tmp_path: Path,
    monkeypatch,
) -> None:
    initialization = _checkpoint(
        tmp_path / "core-v1.pt", 9.0, archetype="unknown"
    )
    teacher_a = _checkpoint(
        tmp_path / "alakazam.pt", 2.0, archetype="alakazam"
    )
    teacher_b = _checkpoint(
        tmp_path / "trevenant.pt", 6.0, archetype="hops-trevenant"
    )
    families = {
        "core": initialization,
        "alakazam": teacher_a,
        "trevenant": teacher_b,
    }

    def frozen(path: Path) -> dict[str, str]:
        model = families[Path(path).name]
        return {
            "family": Path(path).name,
            "model_path": str(model),
            "checkpoint_digest": checkpoint.checkpoint_digest(model),
        }

    monkeypatch.setattr(refresh, "verify_frozen_model", frozen)
    monkeypatch.setattr(refresh, "_architecture", lambda _path: {})
    output = tmp_path / "multi-teacher.pt"

    result, teachers = refresh.materialize_initialization(
        initialization_family=Path("core"),
        teacher_families=[Path("alakazam"), Path("trevenant")],
        output=output,
    )
    saved = checkpoint.load_checkpoint(output, map_location="cpu")

    assert result["teacher_count"] == 2
    assert [row["family"] for row in teachers] == ["alakazam", "trevenant"]
    assert torch.equal(
        saved["model_state_dict"]["encoder.weight"],
        torch.full((3, 4), 4.0),
    )
    assert torch.equal(
        saved["model_state_dict"]["policy_head.weight"],
        torch.full((2, 4), 8.0),
    )
    assert torch.equal(
        saved["model_state_dict"]["integer_buffer"],
        torch.tensor([7], dtype=torch.int64),
    )
    assert "optimizer_state_dict" not in saved
    assert saved["archetype_id"] == "unknown"
    assert all(
        int(value.count_nonzero().item()) == 0
        for name, value in saved["model_state_dict"].items()
        if name.startswith("matchup_adapter_bank.")
        and (name.endswith(".up.weight") or name.endswith(".up.bias"))
    )
    record = saved["extra"]["multi_teacher_core_initialization"]
    assert record["averaging"] == "equal_weight_parameter_space_mean"
    assert [row["weight"] for row in record["teachers"]] == [0.5, 0.5]

    recovered, recovered_teachers = refresh.materialize_initialization(
        initialization_family=Path("core"),
        teacher_families=[Path("alakazam"), Path("trevenant")],
        output=output,
    )
    assert recovered == result
    assert recovered_teachers == teachers


def test_fused_core_initialization_averages_only_compatible_fused_teachers(
    tmp_path: Path,
) -> None:
    def save_model(
        name: str,
        *,
        fusion_enabled: bool,
        fusion_value: float,
    ) -> Path:
        model_cfg = config.ModelConfig(
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
            expanded_heads_enabled=True,
            decision_fusion_enabled=fusion_enabled,
            decision_fusion_runtime_enabled=fusion_enabled,
        )
        model = build_model(
            model_cfg,
            device=torch.device("cpu"),
            aux_archetype_classes=len(archetypes.archetype_ids()) + 1,
            encoder_vocab=64,
            decoder_vocab=64,
            belief_card_vocab=64,
        )
        if fusion_enabled:
            with torch.no_grad():
                for key, value in model.state_dict().items():
                    if key.startswith("decision_fusion.") and (
                        value.is_floating_point() or value.is_complex()
                    ):
                        value.fill_(fusion_value)
        path = tmp_path / f"{name}.pt"
        checkpoint.immutable_torch_save(
            checkpoint.build_checkpoint(
                model=model,
                model_config=model_cfg,
                archetype_id=name,
                model_id=name,
            ),
            path,
        )
        return path

    source = save_model(
        "source-fused-parent",
        fusion_enabled=True,
        fusion_value=9.0,
    )
    fused_a = save_model(
        "fused-a",
        fusion_enabled=True,
        fusion_value=2.0,
    )
    fused_b = save_model(
        "fused-b",
        fusion_enabled=True,
        fusion_value=6.0,
    )
    legacy = save_model(
        "legacy-flat",
        fusion_enabled=False,
        fusion_value=0.0,
    )
    teachers = [
        {
            "family": name,
            "checkpoint": str(path),
            "checkpoint_digest": checkpoint.checkpoint_digest(path),
        }
        for name, path in (
            ("fused-a", fused_a),
            ("fused-b", fused_b),
            ("legacy-flat", legacy),
        )
    ]
    source_payload = checkpoint.load_checkpoint(source, map_location="cpu")
    source_shared = source_payload["model_state_dict"][
        "spatial_encoder.layers.0.self_attn.in_proj_weight"
    ].clone()
    output = tmp_path / "fused-teacher-mean.pt"

    result = refresh.materialize_fusion_teacher_initialization(
        source_parent=source,
        teachers=teachers,
        output=output,
    )
    saved = checkpoint.load_checkpoint(output, map_location="cpu")

    assert result["schema"] == refresh.FUSION_TEACHER_INITIALIZATION_SCHEMA
    assert [row["family"] for row in result["contributors"]] == [
        "fused-a",
        "fused-b",
    ]
    assert [row["weight"] for row in result["contributors"]] == [0.5, 0.5]
    assert result["excluded_teachers"] == [
        {
            "family": "legacy-flat",
            "checkpoint_digest": checkpoint.checkpoint_digest(legacy),
            "reason": "decision_fusion_not_enabled",
        }
    ]
    assert torch.equal(
        saved["model_state_dict"][
            "spatial_encoder.layers.0.self_attn.in_proj_weight"
        ],
        source_shared,
    )
    for name, value in saved["model_state_dict"].items():
        if name.startswith("decision_fusion.") and value.is_floating_point():
            assert torch.equal(value, torch.full_like(value, 4.0)), name

    recovered = refresh.materialize_fusion_teacher_initialization(
        source_parent=source,
        teachers=teachers,
        output=output,
    )
    assert recovered == result


def test_legacy_auxiliary_rows_are_aligned_without_losing_teacher(
    tmp_path: Path,
    monkeypatch,
) -> None:
    initialization = _checkpoint(
        tmp_path / "core-v2.pt",
        9.0,
        archetype="unknown",
        aux_rows=len(archetypes.archetype_ids()) + 1,
    )
    legacy = _checkpoint(
        tmp_path / "alakazam.pt",
        2.0,
        archetype="alakazam",
        aux_rows=len(archetypes.PINNED_CORE_AUX_ARCHETYPE_IDS) + 1,
    )
    current = _checkpoint(
        tmp_path / "lucario.pt",
        6.0,
        archetype="lucario",
        aux_rows=len(archetypes.archetype_ids()) + 1,
    )
    families = {"core": initialization, "alakazam": legacy, "lucario": current}

    def frozen(path: Path) -> dict[str, str]:
        model = families[Path(path).name]
        return {
            "family": Path(path).name,
            "model_path": str(model),
            "checkpoint_digest": checkpoint.checkpoint_digest(model),
        }

    monkeypatch.setattr(refresh, "verify_frozen_model", frozen)
    monkeypatch.setattr(refresh, "_architecture", lambda _path: {})
    output = tmp_path / "aligned.pt"
    refresh.materialize_initialization(
        initialization_family=Path("core"),
        teacher_families=[Path("alakazam"), Path("lucario")],
        output=output,
    )
    saved = checkpoint.load_checkpoint(output, map_location="cpu")
    weight = saved["model_state_dict"]["aux_head.3.weight"]
    current_ids = list(archetypes.archetype_ids())
    pinned_ids = list(archetypes.PINNED_CORE_AUX_ARCHETYPE_IDS)
    for index, archetype_id in enumerate(current_ids):
        expected = 4.0 if archetype_id in pinned_ids else 7.5
        assert torch.equal(weight[index], torch.full((4,), expected))
    assert torch.equal(weight[-1], torch.full((4,), 4.0))
    alignment = saved["extra"]["multi_teacher_core_initialization"][
        "architecture_alignment"
    ]
    assert len(alignment) == 1
    assert alignment[0]["tensors"][0]["retained_initialization_rows"] == [
        archetype_id
        for archetype_id in current_ids
        if archetype_id not in pinned_ids
    ]


def test_mixed_historical_auxiliary_layouts_align_by_semantic_row(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target_ids = list(archetypes.CUMULATIVE_V4_AUX_ARCHETYPE_IDS)
    current_ids = list(archetypes.archetype_ids())
    initialization = _checkpoint(
        tmp_path / "core-v4.pt",
        9.0,
        archetype="unknown",
        aux_rows=len(target_ids) + 1,
        aux_ids=target_ids,
    )
    legacy = _checkpoint(
        tmp_path / "alakazam.pt",
        2.0,
        archetype="alakazam",
        aux_rows=len(archetypes.PINNED_CORE_AUX_ARCHETYPE_IDS) + 1,
    )
    current = _checkpoint(
        tmp_path / "dragapult.pt",
        6.0,
        archetype="dragapult-dusknoir",
        aux_rows=len(current_ids) + 1,
        aux_ids=current_ids,
    )
    families = {
        "core": initialization,
        "alakazam": legacy,
        "dragapult": current,
    }

    def frozen(path: Path) -> dict[str, str]:
        model = families[Path(path).name]
        return {
            "family": Path(path).name,
            "model_path": str(model),
            "checkpoint_digest": checkpoint.checkpoint_digest(model),
        }

    monkeypatch.setattr(refresh, "verify_frozen_model", frozen)
    monkeypatch.setattr(refresh, "_architecture", lambda _path: {})
    output = tmp_path / "mixed.pt"
    refresh.materialize_initialization(
        initialization_family=Path("core"),
        teacher_families=[Path("alakazam"), Path("dragapult")],
        output=output,
    )
    saved = checkpoint.load_checkpoint(output, map_location="cpu")
    weight = saved["model_state_dict"]["aux_head.3.weight"]
    pinned = set(archetypes.PINNED_CORE_AUX_ARCHETYPE_IDS)
    for index, archetype_id in enumerate(target_ids):
        expected = 4.0 if archetype_id in pinned else 7.5
        assert torch.equal(weight[index], torch.full((4,), expected))
    assert torch.equal(weight[-1], torch.full((4,), 4.0))
    alignment = saved["extra"]["multi_teacher_core_initialization"][
        "architecture_alignment"
    ]
    dragapult_alignment = next(
        row for row in alignment if row["checkpoint_digest"]
        == checkpoint.checkpoint_digest(current)
    )
    assert dragapult_alignment["tensors"][0]["omitted_source_rows"] == [
        archetype_id
        for archetype_id in current_ids
        if archetype_id not in target_ids
    ]


def test_expanded_core_refresh_command_is_checksum_pinned() -> None:
    expanded = expanded_handoff_training_contract()
    fusion = decision_fusion_handoff_contract()
    contract = {
        "core_refresh": {
            "initialization": {"checkpoint": "/models/core/model.pt"},
            "teachers": [
                {"checkpoint": "/models/specialist-a/model.pt"},
                {"checkpoint": "/models/specialist-b/model.pt"},
            ],
            "balanced_corpus": {"pointer": "/corpus/PROTECTED.json"},
            "family": "/registry/cumulative-v6",
            "ready_receipt": "/state/cumulative-v6-ready.json",
            "run_name": "cumulative_v6",
            "run_dir": "/runs/cumulative-v6",
            "max_epochs": 25,
            "early_stop_patience": 5,
            "early_stop_min_delta": 1e-4,
            "minimum_decisions": 500_000,
            "requested_decisions_per_batch": 12_288,
            "cpu_pack_root": "/packs/cumulative-v6",
                "expanded_heads": expanded,
                "decision_fusion": fusion,
            }
        }
    command = _core_refresh_command(
        contract=contract,
        runtime={"python": "/python", "registry_root": "/registry"},
    )

    assert "--expanded-heads" in command
    assert "--decision-fusion" in command
    assert command[
        command.index("--expected-expanded-schedule-digest") + 1
    ] == expanded["schedule_digest"]
    assert command[
        command.index("--expected-expanded-target-digest") + 1
    ] == expanded["target_schema_digest"]


def test_core_refresh_command_enables_bounded_teacher_behavior_repair() -> None:
    expanded = expanded_handoff_training_contract()
    fusion = decision_fusion_handoff_contract()
    contract = {
        "core_refresh": {
            "initialization": {"checkpoint": "/models/core/model.pt"},
            "teachers": [
                {"checkpoint": "/models/specialist-a/model.pt"},
                {"checkpoint": "/models/specialist-b/model.pt"},
            ],
            "balanced_corpus": {"pointer": "/corpus/PROTECTED.json"},
            "family": "/registry/cumulative-v6",
            "ready_receipt": "/state/cumulative-v6-ready.json",
            "run_name": "cumulative_v6",
            "run_dir": "/runs/cumulative-v6",
            "max_epochs": 25,
            "early_stop_patience": 5,
            "early_stop_min_delta": 1e-4,
            "minimum_decisions": 500_000,
            "requested_decisions_per_batch": 12_288,
            "cpu_pack_root": "/packs/cumulative-v6",
            "expanded_heads": expanded,
            "decision_fusion": fusion,
            "teacher_behavior_distillation": {
                "schema": "poke_bot.teacher_behavior_distillation/v1",
                "enabled": True,
                "target": (
                    "matching_archetype_frozen_teacher_greedy_action"
                ),
                "causal_inputs_only": True,
                "loss_weight": 0.5,
            },
        }
    }
    command = _core_refresh_command(
        contract=contract,
        runtime={"python": "/python", "registry_root": "/registry"},
    )
    assert "--teacher-behavior-distillation" in command
    assert command[command.index("--teacher-policy-weight") + 1] == "0.5"


def test_teacher_behavior_uses_checksum_bound_inference_derivative(
    tmp_path: Path,
) -> None:
    source_family = tmp_path / "alakazam-owner-accepted"
    source_family.mkdir()
    source = source_family / "model.pt"
    source.write_bytes(b"immutable-passing-checkpoint")
    source_digest = checkpoint.checkpoint_digest(source)
    derivative_family = tmp_path / "alakazam-owner-accepted-roster18-v5"
    derivative_family.mkdir()
    derivative = derivative_family / "model.pt"
    derivative.write_bytes(b"inference-compatible-derivative")
    derivative_digest = checkpoint.checkpoint_digest(derivative)
    manifest = derivative_family / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": refresh.FROZEN_V5_DERIVATIVE_SCHEMA,
                "specialist_id": "alakazam",
                "source_passing_checkpoint": str(source),
                "source_passing_checkpoint_digest": source_digest,
                "derived_checkpoint": str(derivative),
                "derived_checkpoint_digest": derivative_digest,
                "retained_rows_byte_identical": True,
                "inference_only": True,
            }
        ),
        encoding="utf-8",
    )

    identity = refresh._teacher_behavior_inference_identity(
        {
            "checkpoint": str(source),
            "checkpoint_digest": source_digest,
            "archetype_id": "alakazam",
        }
    )

    assert identity["source_checkpoint"] == str(source.resolve())
    assert identity["source_checkpoint_digest"] == source_digest
    assert identity["inference_checkpoint"] == str(derivative.resolve())
    assert identity["inference_checkpoint_digest"] == derivative_digest
    assert identity["derivative_manifest"] == str(manifest.resolve())
    assert identity["derivative_schema"] == refresh.FROZEN_V5_DERIVATIVE_SCHEMA


def test_teacher_behavior_rejects_derivative_bound_to_other_source(
    tmp_path: Path,
) -> None:
    source_family = tmp_path / "teacher"
    source_family.mkdir()
    source = source_family / "model.pt"
    source.write_bytes(b"source")
    derivative_family = tmp_path / "teacher-roster18-v5"
    derivative_family.mkdir()
    derivative = derivative_family / "model.pt"
    derivative.write_bytes(b"derived")
    (derivative_family / "manifest.json").write_text(
        json.dumps(
            {
                "schema": refresh.FROZEN_V5_DERIVATIVE_SCHEMA,
                "specialist_id": "alakazam",
                "source_passing_checkpoint": str(source),
                "source_passing_checkpoint_digest": "sha256:" + "0" * 64,
                "derived_checkpoint": str(derivative),
                "derived_checkpoint_digest": checkpoint.checkpoint_digest(
                    derivative
                ),
                "retained_rows_byte_identical": True,
                "inference_only": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError, match="inference derivative identity is invalid"
    ):
        refresh._teacher_behavior_inference_identity(
            {
                "checkpoint": str(source),
                "checkpoint_digest": checkpoint.checkpoint_digest(source),
                "archetype_id": "alakazam",
            }
        )
