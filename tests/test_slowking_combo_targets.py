from __future__ import annotations

import copy

import pytest
import torch

from poke_bot import archetypes, config, features
from poke_bot.combo_state import (
    COMBO_STATE_KEY,
    COMBO_STATE_TARGET_SCHEMA,
    validate_combo_state_labels,
)
from poke_bot.dataset import DecisionSample, GameSequence, PolicyStage
from poke_bot.device_corpus import DeviceResidentBootstrapCorpus
from poke_bot.model import build_model
from poke_bot.slowking_combo_targets import (
    SLOWKING_DECK,
    attach_slowking_combo_state_labels,
    build_combo_state_target,
)
from poke_bot.train import device_temporal_batch_losses
from scripts.train_pure_rl import _record_to_compact_game


def _deck() -> list[int]:
    return [
        card
        for card, count in SLOWKING_DECK.items()
        for _ in range(count)
    ]


def _observation(*, context: int, effect: int, options: list[dict]) -> dict:
    return {
        "current": {
            "yourIndex": 0,
            "players": [
                {
                    "hand": [{"id": 1248}, {"id": 19}],
                    "active": {"card": {"id": 163}},
                    "bench": [{"card": {"id": 162}}],
                    "discard": [{"id": 1146}],
                },
                {"hand": None, "active": None, "bench": []},
            ],
        },
        "select": {
            "context": context,
            "effect": effect,
            "option": options,
        },
    }


def test_academy_and_cipher_targets_are_exact_and_causal() -> None:
    academy = build_combo_state_target(
        _observation(
            context=9,
            effect=1248,
            options=[{"id": 144}, {"id": 115}],
        ),
        [1],
    )
    assert academy["schema"] == COMBO_STATE_TARGET_SCHEMA
    assert academy["top_deck"] == {"target": 0, "mask": True}
    assert academy["seek_source"] == {"target": 0, "mask": True}

    cipher = build_combo_state_target(
        _observation(
            context=9,
            effect=1188,
            options=[{"id": 224}, {"id": 144}],
        ),
        [0, 1],
    )
    # Cipher puts the demonstrated ordered pair on top sequentially, so the
    # final selected card is the exposed top card.
    assert cipher["top_deck"] == {"target": 1, "mask": True}
    assert cipher["seek_source"] == {"target": 1, "mask": True}


def test_replay_area_index_options_resolve_from_visible_masked_zones() -> None:
    academy_obs = _observation(
        context=9,
        effect=1248,
        options=[
            {"area": 2, "index": 0, "playerIndex": 0, "type": 3},
            {"area": 2, "index": 1, "playerIndex": 0, "type": 3},
        ],
    )
    academy_obs["current"]["players"][0]["hand"] = [
        {"id": 115},
        {"id": 144},
    ]
    academy = build_combo_state_target(academy_obs, [0])
    assert academy["top_deck"] == {"target": 0, "mask": True}

    cipher_obs = _observation(
        context=9,
        effect=1188,
        options=[
            {"area": 1, "index": 0, "playerIndex": 0, "type": 3},
            {"area": 1, "index": 1, "playerIndex": 0, "type": 3},
        ],
    )
    cipher_obs["select"]["deck"] = [{"id": 224}, {"id": 144}]
    cipher = build_combo_state_target(cipher_obs, [0, 1])
    assert cipher["top_deck"] == {"target": 1, "mask": True}


def test_unknown_facts_are_masked_and_suffix_cannot_change_target() -> None:
    observation = _observation(context=0, effect=0, options=[{"id": 1227}])
    before = build_combo_state_target(observation, [0])
    changed_future = copy.deepcopy(observation)
    changed_future["noncausal_future_suffix"] = {
        "deck_order": [115, 144, 224],
        "winner": 1,
    }
    after = build_combo_state_target(changed_future, [0])
    assert before == after
    assert before["top_deck"] == {"target": None, "mask": False}
    assert before["seek_source"] == {"target": None, "mask": False}


def test_exact_deck_attachment_rejects_substitutions() -> None:
    step = {
        "observation": _observation(
            context=35,
            effect=163,
            options=[{"id": 115}, {"id": 144}],
        ),
        "action": [0],
        "aux_labels": {},
    }
    coverage = attach_slowking_combo_state_labels([step], deck=_deck())
    assert coverage["decisions"] == 1
    assert coverage["seek_source"] == 1
    validate_combo_state_labels(step["aux_labels"][COMBO_STATE_KEY])

    wrong = _deck()
    wrong[-1] = 9999
    with pytest.raises(ValueError, match="exact Slowking deck"):
        attach_slowking_combo_state_labels([copy.deepcopy(step)], deck=wrong)


def test_live_slowking_rl_compaction_attaches_combo_targets() -> None:
    record = {
        "episode_id": "slowking-live-rl",
        "seat": 0,
        "archetype": "slowking",
        "opp_archetype": "unknown",
        "deck": _deck(),
        "value": 1.0,
        "steps": [
            {
                "env_step": 1,
                "observation": _observation(
                    context=35,
                    effect=163,
                    options=[{"id": 115}, {"id": 144}],
                ),
                "action": [0],
                "aux_labels": {},
            }
        ],
    }

    compact = _record_to_compact_game(record)

    assert compact is not None
    assert compact.target_provenance["slowking_combo_state_targets"][
        "decisions"
    ] == 1
    validate_combo_state_labels(
        compact.decisions[0].aux_labels[COMBO_STATE_KEY]
    )


def test_masked_values_cannot_silently_become_negative_examples() -> None:
    target = build_combo_state_target(
        _observation(context=0, effect=0, options=[{"id": 1227}]),
        [0],
    )
    target["vector"]["target"][0] = 0.0
    with pytest.raises(ValueError, match="must be null"):
        validate_combo_state_labels(target)


def _sparse(words: int, offset: int = 0) -> features.SparseVector:
    result = features.SparseVector()
    for word in range(words):
        result.word_start()
        result.add((offset + word) % 32, 1.0)
    return result


def test_resident_temporal_loss_trains_combo_head_and_its_action_route() -> None:
    options = _sparse(2, 4)
    target = build_combo_state_target(
        _observation(
            context=9,
            effect=1248,
            options=[{"id": 144}, {"id": 115}],
        ),
        [1],
    )
    decision = DecisionSample(
        board=_sparse(features.NUM_BOARD_TOKENS),
        options=options,
        action=[1],
        action_combo_index=1,
        action_combos=[[0], [1]],
        env_step=1,
        aux_labels={COMBO_STATE_KEY: target},
        action_token=_sparse(1, 12),
        policy_stages=[
            PolicyStage(
                options=options,
                action_combos=[[0], [1]],
                target_index=1,
                guide_confidence=0.5,
                select_context=9,
            )
        ],
    )
    game = GameSequence(
        episode_id="slowking-combo-gradient",
        seat=0,
        archetype="slowking",
        opp_archetype="unknown",
        deck=_deck(),
        value=1.0,
        decisions=[decision],
    )
    corpus = DeviceResidentBootstrapCorpus.from_splits(
        [game], [], device=torch.device("cpu")
    )
    model = build_model(
        config.ModelConfig(
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
            dropout=0.0,
            expanded_heads_enabled=True,
            combo_state_head_enabled=True,
            decision_fusion_enabled=True,
            decision_fusion_runtime_enabled=True,
            decision_fusion_dedicated_routes_enabled=True,
            decision_fusion_dedicated_routes_runtime_enabled=True,
        ),
        device=torch.device("cpu"),
        aux_archetype_classes=len(archetypes.archetype_ids()) + 1,
        encoder_vocab=64,
        decoder_vocab=64,
        belief_card_vocab=64,
    )
    loss, metrics = device_temporal_batch_losses(
        model,
        corpus,
        torch.tensor([0]),
        combo_state_loss_weight=0.025,
    )
    loss.backward()
    combo_gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.combo_state_head.parameters()
        if parameter.grad is not None
    )
    route_gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.decision_fusion.dedicated_routes[
            "combo_state"
        ].parameters()
        if parameter.grad is not None
    )
    assert metrics.combo_state_metrics["eligible_rows"] == 1
    assert metrics.combo_state_loss > 0.0
    assert combo_gradient > 0.0
    assert route_gradient > 0.0
