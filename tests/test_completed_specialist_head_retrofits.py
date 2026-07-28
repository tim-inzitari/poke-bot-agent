from __future__ import annotations

from pathlib import Path

import torch

from poke_bot import checkpoint, config
from poke_bot.model import EXPANDED_HEAD_KEY_PREFIXES, build_model
from poke_bot.strategic_schedule import EXPANDED_HEAD_IDS
from scripts.materialize_completed_specialist_head_retrofits import (
    _core_identity,
    materialize_derivative,
)
from scripts.run_starmie_expert_bootstrap import load_expanded_head_contract


def _cfg(*, expanded: bool) -> config.ModelConfig:
    return config.ModelConfig(
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
        expanded_heads_enabled=expanded,
        dropout=0.0,
    )


def _model(*, expanded: bool):
    return build_model(
        _cfg(expanded=expanded),
        aux_archetype_classes=4,
        encoder_vocab=64,
        decoder_vocab=64,
        belief_card_vocab=32,
    )


def test_materialize_derivative_preserves_source_and_copies_core_heads(
    tmp_path: Path,
) -> None:
    _, identity = load_expanded_head_contract()
    source_model = _model(expanded=False)
    source_path = tmp_path / "source.pt"
    source_payload = checkpoint.build_checkpoint(
        model=source_model,
        model_config=source_model.cfg,
        archetype_id="dudunsparce",
        model_id="dudunsparce.passing",
    )
    checkpoint.atomic_torch_save(source_payload, source_path)
    source_digest = checkpoint.checkpoint_digest(source_path)

    core_model = _model(expanded=True)
    core_path = tmp_path / "core.pt"
    core_payload = checkpoint.build_checkpoint(
        model=core_model,
        model_config=core_model.cfg,
        archetype_id="unknown",
        model_id="core.expanded",
        extra={
            "expanded_head_training": {
                "architecture_present_heads": list(EXPANDED_HEAD_IDS),
                "trained_heads": list(EXPANDED_HEAD_IDS),
                "runtime_enabled_heads": [],
                "target_schema_digest": identity["target_schema_digest"],
                "schedule_digest": identity["schedule_digest"],
            }
        },
    )
    checkpoint.atomic_torch_save(core_payload, core_path)
    core = _core_identity(
        core_path,
        protocol=Path(identity["canonical_config"]),
    )

    output = tmp_path / "retrofit" / "model.pt"
    manifest = materialize_derivative(
        specialist_id="dudunsparce",
        source_checkpoint=source_path,
        source_passing_checkpoint_digest=source_digest,
        core=core,
        output_checkpoint=output,
    )

    assert checkpoint.checkpoint_digest(source_path) == source_digest
    result = checkpoint.load_checkpoint(output, map_location="cpu")
    assert result["model_config"]["expanded_heads_enabled"] is True
    assert result["extra"]["runtime_enabled_heads"] == []
    retrofit = result["extra"]["completed_specialist_head_retrofit"]
    assert retrofit["status"] == "dormant_untrained"
    assert retrofit["runtime_enabled_heads"] == []
    assert retrofit["specialist_trained_heads"] == []
    assert retrofit["serving_eligible"] is False
    assert "optimizer_state_dict" not in result

    source_state = source_payload["model_state_dict"]
    core_state = core_payload["model_state_dict"]
    result_state = result["model_state_dict"]
    for name, tensor in source_state.items():
        torch.testing.assert_close(tensor, result_state[name], rtol=0, atol=0)
    core_heads = {
        name: tensor
        for name, tensor in core_state.items()
        if name.startswith(EXPANDED_HEAD_KEY_PREFIXES)
    }
    assert len(core_heads) == 22
    for name, tensor in core_heads.items():
        torch.testing.assert_close(tensor, result_state[name], rtol=0, atol=0)
    assert manifest["strict_model_load_passed"] is True
    assert manifest["source_checkpoint_unchanged"] is True
