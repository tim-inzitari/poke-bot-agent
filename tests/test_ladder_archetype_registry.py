from __future__ import annotations

import json
from pathlib import Path

import torch

from poke_bot import archetypes, config
from poke_bot.model import build_model
from scripts.train_privileged_belief_shards import _expand_aux_head


def test_every_pinned_core_ladder_deck_has_an_auxiliary_class() -> None:
    payload = json.loads(
        Path("data/training_mixes/top_ladder.v1.json").read_text(encoding="utf-8")
    )
    mix_ids = [str(row["deck_id"]) for row in payload["decks"]]
    assert mix_ids == list(archetypes.CORE_LADDER_ARCHETYPE_IDS)
    assert set(mix_ids).issubset(archetypes.archetype_ids())
    assert tuple(archetypes.archetype_ids()[:5]) == archetypes.LEGACY_AUX_ARCHETYPE_IDS


def test_aux_expansion_preserves_legacy_rows_and_moves_unknown() -> None:
    cfg = config.ModelConfig(
        d_model=16,
        spatial_layers=1,
        temporal_layers=0,
        option_decoder_layers=1,
        n_heads=4,
        ff_dim=32,
        max_context=8,
        decision_context="stateless",
        dropout=0.0,
    )
    legacy_classes = len(archetypes.LEGACY_AUX_ARCHETYPE_IDS) + 1
    model = build_model(cfg, aux_archetype_classes=legacy_classes)
    old_weight = model.aux_head[-1].weight.detach().clone()
    old_bias = model.aux_head[-1].bias.detach().clone()
    assert _expand_aux_head(model)
    new = model.aux_head[-1]
    assert new.out_features == len(archetypes.archetype_ids()) + 1
    for old_i, name in enumerate(archetypes.LEGACY_AUX_ARCHETYPE_IDS):
        new_i = archetypes.archetype_ids().index(name)
        assert torch.equal(new.weight[new_i], old_weight[old_i])
        assert torch.equal(new.bias[new_i], old_bias[old_i])
    assert torch.equal(new.weight[-1], old_weight[-1])
    assert torch.equal(new.bias[-1], old_bias[-1])
