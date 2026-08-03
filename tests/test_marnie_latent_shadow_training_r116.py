from dataclasses import replace

import torch

from poke_bot import config, features
from poke_bot.dataset import DecisionSample, GameSequence, PolicyStage
from poke_bot.model import TemporalCabtTransformer
from poke_bot.pure_rl.latent_lookahead_shadow import (
    freeze_for_latent_shadow,
    latent_shadow_losses,
)


def _sparse(words: int, offset: int = 0) -> features.SparseVector:
    row = features.SparseVector()
    for index in range(words):
        row.word_start()
        row.add((offset + index) % 32, 1.0)
    return row


def _game() -> GameSequence:
    decisions = []
    for index in range(3):
        options = _sparse(3, 10 + index)
        decisions.append(
            DecisionSample(
                board=_sparse(features.NUM_BOARD_TOKENS, index),
                options=options,
                action=[index % 3],
                action_combo_index=index % 3,
                action_combos=[[0], [1], [2]],
                env_step=index,
                action_token=_sparse(1, 20 + index),
                policy_stages=[
                    PolicyStage(
                        options=options,
                        action_combos=[[0], [1], [2]],
                        target_index=index % 3,
                    )
                ],
            )
        )
    return GameSequence(
        episode_id="latent-shadow",
        seat=0,
        archetype="marnie-s-grimmsnarl-ex",
        opp_archetype="crustle",
        deck=[1] * 60,
        value=1.0,
        decisions=decisions,
        policy_targets=None,
        factorized_policy_targets=None,
        target_provenance={"pure_rl": True, "soft_policy_targets": False},
    )


def test_shadow_update_changes_only_latent_tensors() -> None:
    cfg = replace(
        config.ModelConfig(),
        d_model=32,
        n_heads=4,
        spatial_layers=1,
        temporal_layers=1,
        option_decoder_layers=1,
        ff_dim=64,
        max_context=8,
        dropout=0.0,
        latent_lookahead_enabled=True,
        latent_lookahead_action_authority_enabled=False,
        latent_lookahead_width=32,
    )
    model = TemporalCabtTransformer(cfg, aux_archetype_classes=4)
    params = freeze_for_latent_shadow(model)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    loss, metrics = latent_shadow_losses(model, [_game()])
    loss.backward()
    assert metrics.next_state_rows == 2
    assert all(parameter.grad is not None for parameter in params)
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    optimizer.step()
    after = model.state_dict()
    assert any(
        not torch.equal(before[key], after[key])
        for key in before if key.startswith("latent_lookahead.")
    )
    assert all(
        torch.equal(before[key], after[key])
        for key in before if not key.startswith("latent_lookahead.")
    )
    assert model.latent_lookahead_action_authority_enabled is False
