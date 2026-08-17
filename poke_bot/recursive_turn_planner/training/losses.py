"""Supervised losses for Recursive Turn Planner multi-turn training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class RTPLossBundle:
    total: Tensor
    action: Tensor
    ranking: Tensor
    complexity: Tensor
    dynamics: Tensor
    value: Tensor
    calibration: Tensor
    candidate_return: Tensor
    candidate_ranking: Tensor
    candidate_calibration: Tensor
    root_plan: Tensor
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach().item()),
            "action": float(self.action.detach().item()),
            "ranking": float(self.ranking.detach().item()),
            "complexity": float(self.complexity.detach().item()),
            "dynamics": float(self.dynamics.detach().item()),
            "value": float(self.value.detach().item()),
            "calibration": float(self.calibration.detach().item()),
            "candidate_return": float(self.candidate_return.detach().item()),
            "candidate_ranking": float(self.candidate_ranking.detach().item()),
            "candidate_calibration": float(
                self.candidate_calibration.detach().item()
            ),
            "root_plan": float(self.root_plan.detach().item()),
        }


def compute_rtp_losses(
    *,
    action_scores: Tensor,
    chosen_action_index: int,
    complexity_logit: Tensor,
    should_recurse: Optional[bool] = None,
    predicted_next_latent: Optional[Tensor] = None,
    target_next_latent: Optional[Tensor] = None,
    root_plan_logits: Optional[Tensor] = None,
    root_plan_target: Optional[int] = None,
    chosen_value_prediction: Optional[Tensor] = None,
    chosen_uncertainty: Optional[Tensor] = None,
    game_value: Optional[float] = None,
    candidate_return_predictions: Optional[Tensor] = None,
    candidate_return_targets: Optional[Tensor] = None,
    candidate_return_mask: Optional[Tensor] = None,
    candidate_ranking_scores: Optional[Tensor] = None,
    candidate_ranking_targets: Optional[Tensor] = None,
    candidate_ranking_mask: Optional[Tensor] = None,
    candidate_uncertainty_predictions: Optional[Tensor] = None,
    candidate_calibration_targets: Optional[Tensor] = None,
    candidate_calibration_mask: Optional[Tensor] = None,
    action_weight: float = 1.0,
    ranking_weight: float = 0.10,
    complexity_weight: float = 0.25,
    dynamics_weight: float = 0.5,
    value_weight: float = 0.25,
    calibration_weight: float = 0.10,
    candidate_return_weight: float = 0.25,
    candidate_ranking_weight: float = 0.10,
    candidate_calibration_weight: float = 0.10,
    root_plan_weight: float = 0.15,
) -> RTPLossBundle:
    """Compute losses from observed selected-action evidence only.

    The compact training shards record the action that was taken, not a
    counterfactual outcome for every legal action.  Accordingly, the ranking
    term is a demonstrated-choice preference and value/calibration terms apply
    *only* to the selected complete action when a terminal outcome is known.
    Invalid or unavailable targets are masked rather than remapped to action
    zero or treated as a negative target for an unchosen action.
    """
    if action_scores.dim() == 1:
        action_scores = action_scores.unsqueeze(0)
    if complexity_logit.dim() == 0:
        complexity_logit = complexity_logit.unsqueeze(0)
    elif complexity_logit.dim() == 2 and complexity_logit.size(-1) == 1:
        complexity_logit = complexity_logit.squeeze(-1)

    if action_scores.size(-1) <= 0:
        raise ValueError("RTP action scores require at least one legal action")

    device = action_scores.device
    valid_choice = 0 <= int(chosen_action_index) < int(action_scores.size(-1))
    zero = action_scores.new_zeros(())

    def masked_evaluator_targets(
        prediction: Optional[Tensor],
        target: Optional[Tensor],
        mask: Optional[Tensor],
        *,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
    ) -> tuple[Optional[Tensor], Optional[Tensor], int]:
        """Return only exact aligned evaluator targets, or an empty mask.

        A missing, malformed, or non-finite optional target is unavailable;
        it never gets coerced into an outcome for a legal but unchosen action.
        """
        if prediction is None or target is None:
            return None, None, 0
        pred = prediction.reshape(-1)
        target_t = target.to(device=device, dtype=pred.dtype).reshape(-1)
        if pred.numel() != target_t.numel() or pred.numel() == 0:
            return None, None, 0
        # A corrupt prediction must not turn a masked evaluator row into a
        # non-finite loss.  This is deliberately a mask, rather than a
        # replacement value, because evaluator targets are optional evidence.
        valid = torch.isfinite(pred) & torch.isfinite(target_t)
        if lower is not None:
            valid = valid & (target_t >= float(lower))
        if upper is not None:
            valid = valid & (target_t <= float(upper))
        if mask is not None:
            mask_t = mask.to(device=device).reshape(-1)
            if mask_t.numel() != pred.numel():
                return None, None, 0
            valid = valid & mask_t.to(dtype=torch.bool)
        count = int(valid.sum().item())
        if count == 0:
            return None, None, 0
        return pred[valid], target_t[valid], count
    if valid_choice:
        action_t = torch.tensor(
            [int(chosen_action_index)], device=device, dtype=torch.long
        )
        action_loss = F.cross_entropy(action_scores, action_t)
        if action_scores.size(-1) > 1:
            chosen_score = action_scores[0, int(chosen_action_index)]
            other_mask = torch.ones(
                action_scores.size(-1), dtype=torch.bool, device=device
            )
            other_mask[int(chosen_action_index)] = False
            # A behavioral preference target: this does not assert that any
            # unchosen legal action had a bad game outcome.
            ranking_loss = F.softplus(
                torch.logsumexp(action_scores[0, other_mask], dim=0)
                - chosen_score
            )
        else:
            ranking_loss = zero
    else:
        action_loss = zero
        ranking_loss = zero

    if should_recurse is None:
        complexity_loss = zero
    else:
        recurse_t = torch.tensor(
            [1.0 if bool(should_recurse) else 0.0],
            device=device,
            dtype=torch.float32,
        )
        complexity_loss = F.binary_cross_entropy_with_logits(
            complexity_logit.view(-1), recurse_t
        )

    if predicted_next_latent is not None and target_next_latent is not None:
        dynamics_loss = F.mse_loss(
            predicted_next_latent, target_next_latent.detach()
        )
    else:
        dynamics_loss = zero

    valid_terminal_value = False
    value_loss = zero
    calibration_loss = zero
    if (
        valid_choice
        and chosen_value_prediction is not None
        and game_value is not None
    ):
        target_value = float(game_value)
        if torch.isfinite(torch.tensor(target_value)) and -1.0 <= target_value <= 1.0:
            value_t = torch.tensor(
                [target_value], device=device, dtype=action_scores.dtype
            )
            value_pred = chosen_value_prediction.reshape(-1)[:1]
            if value_pred.numel() == 1:
                valid_terminal_value = True
                value_loss = F.smooth_l1_loss(value_pred, value_t)
                if chosen_uncertainty is not None:
                    uncertainty = chosen_uncertainty.reshape(-1)[:1]
                    if uncertainty.numel() == 1:
                        # ``uncertainty`` is bounded by the dynamics sigmoid;
                        # calibrate it to this selected action's observed
                        # terminal-value residual, never an unchosen action.
                        observed_error = (
                            (value_pred.detach() - value_t).abs().clamp(0.0, 1.0)
                        )
                        calibration_loss = F.mse_loss(uncertainty, observed_error)

    # These three candidate terms require an external evaluator that has
    # explicitly bound its targets to the complete runtime action list.  They
    # remain exact zero for ordinary behavior shards, which contain only the
    # action that was actually selected.
    return_pred, return_target, candidate_return_count = masked_evaluator_targets(
        candidate_return_predictions,
        candidate_return_targets,
        candidate_return_mask,
        lower=-1.0,
        upper=1.0,
    )
    candidate_return_loss = (
        F.smooth_l1_loss(return_pred, return_target)
        if return_pred is not None and return_target is not None
        else zero
    )

    rank_pred, rank_target, candidate_ranking_count = masked_evaluator_targets(
        candidate_ranking_scores,
        candidate_ranking_targets,
        candidate_ranking_mask,
    )
    candidate_ranking_loss = zero
    candidate_ranking_pairs = 0
    if rank_pred is not None and rank_target is not None and rank_pred.numel() >= 2:
        # Pairwise comparison is defined only where the trusted evaluator
        # distinguishes two candidates.  Ties remain unlabelled.
        target_delta = rank_target.unsqueeze(1) - rank_target.unsqueeze(0)
        upper_tri = torch.triu(
            torch.ones_like(target_delta, dtype=torch.bool), diagonal=1
        )
        comparable = upper_tri & (target_delta != 0)
        candidate_ranking_pairs = int(comparable.sum().item())
        if candidate_ranking_pairs:
            score_delta = rank_pred.unsqueeze(1) - rank_pred.unsqueeze(0)
            signs = torch.sign(target_delta[comparable])
            candidate_ranking_loss = F.softplus(
                -signs * score_delta[comparable]
            ).mean()

    calibration_pred, calibration_target, candidate_calibration_count = (
        masked_evaluator_targets(
            candidate_uncertainty_predictions,
            candidate_calibration_targets,
            candidate_calibration_mask,
            lower=0.0,
            upper=1.0,
        )
    )
    candidate_calibration_loss = (
        F.mse_loss(calibration_pred, calibration_target)
        if calibration_pred is not None and calibration_target is not None
        else zero
    )

    if (
        root_plan_logits is not None
        and root_plan_target is not None
        and int(root_plan_target) >= 0
    ):
        if root_plan_logits.dim() == 1:
            root_plan_logits = root_plan_logits.unsqueeze(0)
        if int(root_plan_target) < int(root_plan_logits.size(-1)):
            root_t = torch.tensor(
                [int(root_plan_target)],
                device=device,
                dtype=torch.long,
            )
            root_loss = F.cross_entropy(root_plan_logits, root_t)
        else:
            root_loss = zero
    else:
        root_loss = zero

    total = (
        float(action_weight) * action_loss
        + float(ranking_weight) * ranking_loss
        + float(complexity_weight) * complexity_loss
        + float(dynamics_weight) * dynamics_loss
        + float(value_weight) * value_loss
        + float(calibration_weight) * calibration_loss
        + float(candidate_return_weight) * candidate_return_loss
        + float(candidate_ranking_weight) * candidate_ranking_loss
        + float(candidate_calibration_weight) * candidate_calibration_loss
        + float(root_plan_weight) * root_loss
    )
    return RTPLossBundle(
        total=total,
        action=action_loss,
        ranking=ranking_loss,
        complexity=complexity_loss,
        dynamics=dynamics_loss,
        value=value_loss,
        calibration=calibration_loss,
        candidate_return=candidate_return_loss,
        candidate_ranking=candidate_ranking_loss,
        candidate_calibration=candidate_calibration_loss,
        root_plan=root_loss,
        metadata={
            "action_target_available": bool(valid_choice),
            "ranking_target_available": bool(valid_choice and action_scores.size(-1) > 1),
            "complexity_target_available": should_recurse is not None,
            "dynamics_target_available": (
                predicted_next_latent is not None and target_next_latent is not None
            ),
            "value_target_available": bool(valid_terminal_value),
            "calibration_target_available": bool(
                valid_terminal_value and chosen_uncertainty is not None
            ),
            "root_plan_target_available": bool(
                root_plan_logits is not None
                and root_plan_target is not None
                and int(root_plan_target) >= 0
                and int(root_plan_target) < int(root_plan_logits.size(-1))
            ),
            "candidate_return_target_available": candidate_return_count > 0,
            "candidate_return_target_count": candidate_return_count,
            "candidate_ranking_target_available": candidate_ranking_pairs > 0,
            "candidate_ranking_target_count": candidate_ranking_count,
            "candidate_ranking_pair_count": candidate_ranking_pairs,
            "candidate_calibration_target_available": (
                candidate_calibration_count > 0
            ),
            "candidate_calibration_target_count": candidate_calibration_count,
        },
    )
