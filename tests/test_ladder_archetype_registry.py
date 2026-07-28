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
    assert archetypes.archetype_ids()[-5:] == [
        "dudunsparce",
        "hops-trevenant",
        "walrein",
        "thwackey",
        "team-rockets-spidops",
    ]


def test_post_snapshot_archetypes_are_additive_and_signature_distinct() -> None:
    assert archetypes.classify_deck([879] + [1] * 59) == "hops-trevenant"
    assert archetypes.classify_deck([943] + [1] * 59) == "walrein"
    assert archetypes.classify_deck([306] + [1] * 59) == "dudunsparce"
    assert (
        archetypes.classify_deck([306, 646, 647, 648] + [1] * 56)
        == "marnie-s-grimmsnarl-ex"
    )
    assert (
        archetypes.classify_deck([400] * 4 + [401] * 4 + [431] + [1] * 51)
        == "team-rockets-spidops"
    )
    # The shared Spidops engine must not steal Rocket's Mewtwo lists.
    assert (
        archetypes.classify_deck([400] * 4 + [401] * 4 + [431] * 2 + [1] * 50)
        == "unknown"
    )
    assert (
        archetypes.classify_deck(
            [89] * 4 + [90] * 4 + [93] * 4 + [1245] * 4 + [1] * 44
        )
        == "thwackey"
    )
    assert (
        archetypes.classify_deck(
            [89] * 2 + [90] * 2 + [93] * 2 + [1245] * 2 + [1] * 52
        )
        == "unknown"
    )
    # Dragapult+Dudunsparce remains its older, separate family.
    assert (
        archetypes.classify_deck([119, 306] + [1] * 58)
        == "dragapult-dudunsparce"
    )


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


def test_aux_expansion_preserves_current_core_rows_and_moves_unknown() -> None:
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
    old_ids = list(archetypes.PINNED_CORE_AUX_ARCHETYPE_IDS)
    model = build_model(cfg, aux_archetype_classes=len(old_ids) + 1)
    old_weight = model.aux_head[-1].weight.detach().clone()
    old_bias = model.aux_head[-1].bias.detach().clone()
    assert _expand_aux_head(model)
    new = model.aux_head[-1]
    for old_i, name in enumerate(old_ids):
        new_i = archetypes.archetype_ids().index(name)
        assert torch.equal(new.weight[new_i], old_weight[old_i])
        assert torch.equal(new.bias[new_i], old_bias[old_i])
    assert torch.equal(new.weight[-1], old_weight[-1])
    assert torch.equal(new.bias[-1], old_bias[-1])
