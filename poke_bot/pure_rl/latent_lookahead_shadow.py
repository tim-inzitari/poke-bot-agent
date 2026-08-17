"""Training-only revision-116 latent-lookahead shadow objective.

The protected H10 parent remains frozen.  Only the separately versioned
``latent_lookahead`` module receives gradients, and its policy aid remains
outside the authoritative action path until the external gate suite passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from poke_bot.dataset import GameSequence, PolicyStage
from poke_bot.model import ActionConditionedLatentLookahead, TemporalCabtTransformer


@dataclass(frozen=True)
class LatentShadowMetrics:
    loss: float
    next_state_loss: float
    contrastive_loss: float
    continuation_loss: float
    policy_aid_loss: float
    unselected_aid_regularizer: float
    games: int
    decisions: int
    option_rows: int
    next_state_rows: int


def freeze_for_latent_shadow(model: TemporalCabtTransformer) -> list[torch.nn.Parameter]:
    """Freeze the protected policy and return only latent trainable tensors."""

    module = model.latent_lookahead
    if not isinstance(module, ActionConditionedLatentLookahead):
        raise ValueError("latent-lookahead architecture is required")
    if model.latent_lookahead_action_authority_enabled:
        raise ValueError("shadow training requires latent action authority off")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    params = list(module.parameters())
    for parameter in params:
        parameter.requires_grad_(True)
    model.eval()
    module.train()
    return params


def latent_shadow_losses(
    model: TemporalCabtTransformer,
    sequences: Sequence[GameSequence],
    *,
    next_state_weight: float = 1.0,
    contrastive_weight: float = 0.25,
    continuation_weight: float = 0.50,
    policy_aid_weight: float = 0.25,
    unselected_aid_weight: float = 0.01,
    contrastive_margin: float = 0.10,
) -> tuple[torch.Tensor, LatentShadowMetrics]:
    """Train a single-pass latent model from observed, training-eligible play.

    The selected option predicts the next causally observed state and terminal
    return.  Other legal options act only as contrastive negatives and receive
    a small zero-aid regularizer; they are never assigned invented outcomes.
    The policy-aid target is the bounded observed return advantage over the
    frozen parent's value estimate, so this is outcome learning rather than
    imitation of a guide or behavior distribution.
    """

    games = [sequence for sequence in sequences if sequence.decisions]
    if not games:
        device = next(model.parameters()).device
        zero = torch.zeros((), device=device, requires_grad=True)
        return zero, LatentShadowMetrics(*(0.0,) * 6, 0, 0, 0, 0)
    module = model.latent_lookahead
    if not isinstance(module, ActionConditionedLatentLookahead):
        raise ValueError("latent-lookahead architecture is required")
    if model.latent_lookahead_action_authority_enabled:
        raise ValueError("shadow loss cannot run with action authority")
    device = next(model.parameters()).device

    all_boards = [decision.board for game in games for decision in game.decisions]
    lengths = [len(game.decisions) for game in games]
    previous_actions = [
        action
        for game in games
        for action in ([None] + [d.action_token for d in game.decisions[:-1]])
    ]
    with torch.no_grad():
        spatial = model.encode_board(all_boards)
        cls = model.pool_cls(spatial) + float(model.cfg.history_action_scale) * (
            model.encode_previous_actions(previous_actions)
        )
        state_parts: list[torch.Tensor] = []
        cursor = 0
        for length in lengths:
            encoded, _ = model.temporal_encode(
                cls[cursor : cursor + length].unsqueeze(0),
                append=False,
                return_all=True,
            )
            state_parts.append(encoded.squeeze(0))
            cursor += length
        states = torch.cat(state_parts, dim=0)

        stage_options = []
        stage_counts: list[int] = []
        selected_indices: list[int] = []
        state_rows: list[int] = []
        next_state_rows: list[int] = []
        returns: list[float] = []
        cursor = 0
        for game, length in zip(games, lengths):
            for local_index, decision in enumerate(game.decisions):
                stages = list(decision.policy_stages)
                if not stages:
                    stages = [
                        PolicyStage(
                            options=decision.options,
                            action_combos=decision.action_combos,
                            target_index=decision.action_combo_index,
                        )
                    ]
                for stage in stages:
                    count = int(stage.options.num_words)
                    selected = int(stage.target_index)
                    if count < 2 or selected < 0 or selected >= count:
                        continue
                    stage_options.append(stage.options)
                    stage_counts.append(count)
                    selected_indices.append(selected)
                    state_rows.append(cursor + local_index)
                    next_state_rows.append(
                        cursor + local_index + 1 if local_index + 1 < length else -1
                    )
                    returns.append(float(game.value))
            cursor += length
        if not stage_options:
            zero = states.sum() * 0.0
            return zero, LatentShadowMetrics(*(0.0,) * 6, len(games), sum(lengths), 0, 0)
        state_index = torch.tensor(state_rows, device=device, dtype=torch.long)
        stage_state = states.index_select(0, state_index)
        stage_spatial = spatial.index_select(0, state_index)
        decoded = model.decode_options(
            stage_options,
            stage_spatial,
            stage_state,
            n_options=stage_counts,
            return_hidden=True,
            decision_fusion_state_vec=stage_state,
        )
        if not isinstance(decoded, tuple):
            raise AssertionError("latent shadow decoder did not return option states")
        _base_logits, option_hidden = decoded
        parent_values = torch.tanh(model.value_head(stage_state)).squeeze(-1)

    outputs = module(option_hidden.detach(), stage_state.detach())
    predicted_next = outputs["predicted_next_state_latent"]
    continuation = outputs["continuation_value"]
    aid = outputs["policy_aid"]
    selected = torch.tensor(selected_indices, device=device, dtype=torch.long)
    rows = torch.arange(selected.numel(), device=device)
    selected_next = predicted_next[rows, selected]
    selected_continuation = continuation[rows, selected]
    selected_aid = aid[rows, selected]
    return_target = torch.tensor(returns, device=device, dtype=stage_state.dtype)

    valid_next = torch.tensor(
        [row >= 0 for row in next_state_rows], device=device, dtype=torch.bool
    )
    next_loss = selected_next.sum() * 0.0
    contrastive = selected_next.sum() * 0.0
    if bool(valid_next.any()):
        next_index = torch.tensor(
            [max(0, row) for row in next_state_rows], device=device, dtype=torch.long
        )
        target_next = states.index_select(0, next_index).detach()
        next_loss = F.smooth_l1_loss(
            selected_next[valid_next], target_next[valid_next]
        )
        distances = F.smooth_l1_loss(
            predicted_next,
            target_next.unsqueeze(1).expand_as(predicted_next),
            reduction="none",
        ).mean(dim=-1)
        option_mask = (
            torch.arange(aid.size(1), device=device).unsqueeze(0)
            < torch.tensor(stage_counts, device=device).unsqueeze(1)
        )
        negative_mask = option_mask.clone()
        negative_mask[rows, selected] = False
        negative_distance = distances.masked_fill(
            ~negative_mask, float("inf")
        ).amin(dim=1)
        selected_distance = distances[rows, selected]
        contrastive = F.relu(
            float(contrastive_margin)
            + selected_distance[valid_next]
            - negative_distance[valid_next]
        ).mean()
    continuation_loss = F.smooth_l1_loss(
        selected_continuation, return_target
    )
    advantage = (return_target - parent_values.detach()).clamp(
        -module.policy_aid_cap, module.policy_aid_cap
    )
    policy_aid_loss = F.smooth_l1_loss(selected_aid, advantage)
    option_mask = (
        torch.arange(aid.size(1), device=device).unsqueeze(0)
        < torch.tensor(stage_counts, device=device).unsqueeze(1)
    )
    unselected_mask = option_mask.clone()
    unselected_mask[rows, selected] = False
    unselected_regularizer = aid[unselected_mask].square().mean()
    total = (
        float(next_state_weight) * next_loss
        + float(contrastive_weight) * contrastive
        + float(continuation_weight) * continuation_loss
        + float(policy_aid_weight) * policy_aid_loss
        + float(unselected_aid_weight) * unselected_regularizer
    )
    metrics = LatentShadowMetrics(
        loss=float(total.detach().item()),
        next_state_loss=float(next_loss.detach().item()),
        contrastive_loss=float(contrastive.detach().item()),
        continuation_loss=float(continuation_loss.detach().item()),
        policy_aid_loss=float(policy_aid_loss.detach().item()),
        unselected_aid_regularizer=float(unselected_regularizer.detach().item()),
        games=len(games),
        decisions=sum(lengths),
        option_rows=len(stage_options),
        next_state_rows=int(valid_next.sum().item()),
    )
    return total, metrics
