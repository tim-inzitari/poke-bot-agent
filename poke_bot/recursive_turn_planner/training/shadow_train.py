"""Offline / shadow training for the Recursive Turn Planner.

Trains planner + dynamics heads while the CABT encoder stays frozen (when a
parent checkpoint is supplied). Supports a synthetic path for CI without shards.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from poke_bot.recursive_turn_planner.config import (
    RTPConfig,
    RTP_MAX_AUTHORIZED_NEURAL_PASSES,
)
from poke_bot.recursive_turn_planner.planner import RecursiveTurnPlanner
from poke_bot.recursive_turn_planner.profiles import get_profile
from poke_bot.recursive_turn_planner.training.checkpoint import save_rtp_checkpoint
from poke_bot.recursive_turn_planner.training.losses import compute_rtp_losses


@dataclass
class RTPDecisionBatch:
    """One shadow-training decision in runtime-shaped encoder feature space.

    ``legal_actions`` is complete ordered action support when
    ``action_space_source`` starts with ``"runtime_complete"``.  Legacy
    factorized stages remain explicitly labelled as such rather than being
    misrepresented as executable whole-turn actions.
    """

    state: Tensor  # [D]
    option_hidden: Tensor  # [N, D]
    legal_actions: list[list[int]]
    chosen_index: int
    should_recurse: Optional[bool] = None
    next_state: Optional[Tensor] = None
    root_plan_target: Optional[int] = None
    game_value: Optional[float] = None
    outcome_available: bool = False
    episode_id: str = ""
    #: Deterministic encoder window within an episode.  This is provenance
    #: only; ``episode_id`` remains the whole-game split key.
    sequence_window_id: str = ""
    action_space_source: str = "unknown"
    #: Optional evaluator-only targets, all aligned to ``legal_actions`` and
    #: only populated after an exact action-space/provenance check.
    candidate_return_targets: Optional[Tensor] = None
    candidate_return_mask: Optional[Tensor] = None
    candidate_ranking_targets: Optional[Tensor] = None
    candidate_ranking_mask: Optional[Tensor] = None
    candidate_calibration_targets: Optional[Tensor] = None
    candidate_calibration_mask: Optional[Tensor] = None
    candidate_target_provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class RTPTrainConfig:
    d_model: int = 96
    profile: str = "pure_rl"
    epochs: int = 2
    lr: float = 1e-3
    seed: int = 0
    action_weight: float = 1.0
    ranking_weight: float = 0.10
    complexity_weight: float = 0.25
    dynamics_weight: float = 0.5
    value_weight: float = 0.25
    calibration_weight: float = 0.10
    candidate_return_weight: float = 0.25
    candidate_ranking_weight: float = 0.10
    candidate_calibration_weight: float = 0.10
    root_plan_weight: float = 0.15
    complexity_option_threshold: int = 8
    complexity_entropy_threshold: float = 1.5
    num_plan_candidates: int = 4
    max_recursion_depth: int = 2
    #: Generic legacy default.  The r197 candidate must explicitly request its
    #: separately versioned profile and its 256-pass bound.
    max_neural_passes: int = 4
    device: str = "cpu"

    def __post_init__(self) -> None:
        """Keep the r197 candidate exact without changing legacy defaults."""
        if int(self.d_model) < 1:
            raise ValueError("d_model must be positive")
        if int(self.num_plan_candidates) < 1:
            raise ValueError("num_plan_candidates must be positive")
        if int(self.max_recursion_depth) < 0:
            raise ValueError("max_recursion_depth must be non-negative")
        if int(self.max_neural_passes) < 1:
            raise ValueError("max_neural_passes must be positive")
        if int(self.max_neural_passes) > RTP_MAX_AUTHORIZED_NEURAL_PASSES:
            raise ValueError(
                "max_neural_passes exceeds the global authorized ceiling "
                f"({RTP_MAX_AUTHORIZED_NEURAL_PASSES})"
            )
        if str(self.profile).strip().lower() == "pure_rl_r197":
            if int(self.d_model) != 96:
                raise ValueError("pure_rl_r197 requires d_model=96")
            if int(self.num_plan_candidates) != 4:
                raise ValueError("pure_rl_r197 requires exactly four plan candidates")
            if int(self.max_recursion_depth) != 2:
                raise ValueError("pure_rl_r197 requires max_recursion_depth=2")
            if int(self.max_neural_passes) != RTP_MAX_AUTHORIZED_NEURAL_PASSES:
                raise ValueError(
                    "pure_rl_r197 requires max_neural_passes="
                    f"{RTP_MAX_AUTHORIZED_NEURAL_PASSES}"
                )


@dataclass
class RTPTrainResult:
    checkpoint_path: str
    receipt_path: str
    metrics: dict[str, Any]
    heldout_metrics: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, float]] = field(default_factory=list)
    inventory: dict[str, Any] = field(default_factory=dict)


def required_recursive_passes(
    *,
    num_plan_candidates: int,
    max_recursion_depth: int,
) -> dict[str, int]:
    """Return the current planner's actual minimum pass requirements.

    At depth two each root candidate contains one refinable subgoal.  A normal
    path consumes one complexity pass, one root proposal pass, and one refine
    pass per candidate; ``force_recurse`` skips the first of those.  This is
    provenance only: it does not alter runtime budgets.
    """
    candidates = max(1, int(num_plan_candidates))
    depth = max(0, int(max_recursion_depth))
    # Keep this in lockstep with planner.required_recursive_passes(): one
    # skeleton subgoal expands once at depth two and each extra depth would
    # add one expansion per root candidate.
    refinements = candidates * max(0, depth - 1)
    return {
        "normal_recursive": 2 + refinements,
        "forced_recursive": 1 + refinements,
    }


def _stable_group_digest(values: Sequence[str]) -> str:
    payload = "\n".join(sorted(str(value) for value in values)).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _is_sha256_digest(value: object) -> bool:
    raw = str(value or "").strip().lower()
    if raw.startswith("sha256:"):
        raw = raw[len("sha256:") :]
    return len(raw) == 64 and all(char in "0123456789abcdef" for char in raw)


def trusted_candidate_targets_from_record(
    record: Mapping[str, Any],
    *,
    n_actions: int,
    action_space_fingerprint: str,
) -> dict[str, Any]:
    """Extract explicitly trusted, action-space-bound evaluator targets.

    Ordinary replay/corpus rows observe only one chosen action and therefore
    must not be expanded into pseudo-return labels for the unchosen actions.
    A future evaluator may provide all-candidate supervision, but only under
    this narrow schema: a truthy ``trusted`` marker, a receipt digest, and an
    exact fingerprint of the complete ordered action support are required.
    Any absence or mismatch returns all targets masked rather than guessing.
    """
    unavailable: dict[str, Any] = {
        "candidate_return_targets": None,
        "candidate_return_mask": None,
        "candidate_ranking_targets": None,
        "candidate_ranking_mask": None,
        "candidate_calibration_targets": None,
        "candidate_calibration_mask": None,
        "provenance": {
            "schema": "poke_bot.rtp_candidate_evaluator_binding/v1",
            "status": "not_supplied",
            "latent_lookahead_targets": "not_wired_future_input",
        },
    }
    raw = record.get("evaluator_targets")
    if raw is None:
        return unavailable
    if not isinstance(raw, Mapping):
        unavailable["provenance"]["status"] = "masked_malformed_evaluator_targets"
        return unavailable
    if (
        str(raw.get("schema") or "")
        != "poke_bot.rtp_complete_action_evaluator_targets/v1"
    ):
        unavailable["provenance"]["status"] = "masked_unrecognized_evaluator_schema"
        return unavailable
    if raw.get("trusted") is not True:
        unavailable["provenance"]["status"] = "masked_untrusted_evaluator_targets"
        return unavailable
    if str(raw.get("action_space_fingerprint") or "") != str(
        action_space_fingerprint
    ):
        unavailable["provenance"]["status"] = "masked_action_space_fingerprint_mismatch"
        return unavailable
    receipt_digest = str(raw.get("evaluator_receipt_sha256") or "")
    if not _is_sha256_digest(receipt_digest):
        unavailable["provenance"]["status"] = "masked_missing_evaluator_receipt_digest"
        return unavailable
    if int(n_actions) <= 0:
        unavailable["provenance"]["status"] = "masked_empty_action_space"
        return unavailable

    def parse(
        name: str,
        *,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
    ) -> tuple[Optional[Tensor], Optional[Tensor], dict[str, int]]:
        raw_values = raw.get(name)
        stats = {"provided": 0, "usable": 0, "masked": 0}
        if raw_values is None:
            return None, None, stats
        if not isinstance(raw_values, (list, tuple)) or len(raw_values) != int(n_actions):
            stats["masked"] = int(n_actions)
            return None, None, stats
        raw_mask = raw.get(f"{name}_mask")
        if raw_mask is not None and (
            not isinstance(raw_mask, (list, tuple))
            or len(raw_mask) != int(n_actions)
            or any(not isinstance(item, bool) for item in raw_mask)
        ):
            stats["masked"] = int(n_actions)
            return None, None, stats
        values: list[float] = []
        usable: list[bool] = []
        for index, item in enumerate(raw_values):
            try:
                numeric = float(item)
            except (TypeError, ValueError):
                numeric = float("nan")
            allowed = math.isfinite(numeric)
            if lower is not None:
                allowed = allowed and numeric >= float(lower)
            if upper is not None:
                allowed = allowed and numeric <= float(upper)
            if raw_mask is not None:
                allowed = allowed and raw_mask[index] is True
            values.append(numeric)
            usable.append(bool(allowed))
        stats["provided"] = int(n_actions)
        stats["usable"] = sum(usable)
        stats["masked"] = int(n_actions) - stats["usable"]
        return (
            torch.tensor(values, dtype=torch.float32),
            torch.tensor(usable, dtype=torch.bool),
            stats,
        )

    returns, returns_mask, returns_stats = parse(
        "candidate_return_targets", lower=-1.0, upper=1.0
    )
    ranking, ranking_mask, ranking_stats = parse("candidate_ranking_targets")
    calibration, calibration_mask, calibration_stats = parse(
        "candidate_calibration_targets", lower=0.0, upper=1.0
    )
    if not any(
        int(stats["usable"])
        for stats in (returns_stats, ranking_stats, calibration_stats)
    ):
        unavailable["provenance"]["status"] = "masked_no_usable_evaluator_targets"
        return unavailable
    unavailable.update(
        {
            "candidate_return_targets": returns,
            "candidate_return_mask": returns_mask,
            "candidate_ranking_targets": ranking,
            "candidate_ranking_mask": ranking_mask,
            "candidate_calibration_targets": calibration,
            "candidate_calibration_mask": calibration_mask,
            "provenance": {
                "schema": "poke_bot.rtp_candidate_evaluator_binding/v1",
                "status": "trusted_action_space_bound",
                "evaluator_receipt_sha256": receipt_digest.lower(),
                "action_space_fingerprint": str(action_space_fingerprint),
                "latent_lookahead_targets": str(
                    raw.get("latent_lookahead_targets") or "external_evaluator"
                ),
                "candidate_return": returns_stats,
                "candidate_ranking": ranking_stats,
                "candidate_calibration": calibration_stats,
            },
        }
    )
    return unavailable


def split_batches_by_game(
    batches: Sequence[RTPDecisionBatch],
    *,
    heldout_fraction: float = 0.20,
    seed: int = 0,
) -> tuple[list[RTPDecisionBatch], list[RTPDecisionBatch], dict[str, Any]]:
    """Split whole games deterministically, never individual decisions.

    Empty episode IDs cannot establish that two decisions are from the same
    game, so each is conservatively assigned its own anonymous group.  With a
    single game there is no honest heldout split; the returned provenance says
    so instead of leaking that game's later decisions into validation.
    """
    fraction = float(heldout_fraction)
    if not 0.0 <= fraction < 1.0:
        raise ValueError("heldout_fraction must be in [0, 1)")

    grouped: dict[str, list[RTPDecisionBatch]] = {}
    anonymous = 0
    for index, batch in enumerate(batches):
        raw = str(batch.episode_id or "").strip()
        if raw:
            key = raw
        else:
            key = f"__anonymous_batch_{index:09d}"
            anonymous += 1
        grouped.setdefault(key, []).append(batch)

    group_ids = sorted(grouped)
    if len(group_ids) < 2 or fraction <= 0.0:
        train_ids = group_ids
        heldout_ids: list[str] = []
    else:
        target = max(1, int(round(len(group_ids) * fraction)))
        target = min(target, len(group_ids) - 1)
        ranked = sorted(
            group_ids,
            key=lambda game_id: hashlib.sha256(
                f"{int(seed)}:{game_id}".encode("utf-8")
            ).hexdigest(),
        )
        heldout_ids = ranked[:target]
        heldout_set = set(heldout_ids)
        train_ids = [game_id for game_id in group_ids if game_id not in heldout_set]

    train_set = set(train_ids)
    train = [
        batch
        for index, batch in enumerate(batches)
        if (str(batch.episode_id or "").strip() or f"__anonymous_batch_{index:09d}")
        in train_set
    ]
    heldout_set = set(heldout_ids)
    heldout = [
        batch
        for index, batch in enumerate(batches)
        if (str(batch.episode_id or "").strip() or f"__anonymous_batch_{index:09d}")
        in heldout_set
    ]
    return train, heldout, {
        "schema": "poke_bot.recursive_turn_planner.game_heldout_split/v1",
        "seed": int(seed),
        "heldout_fraction_requested": fraction,
        "group_unit": "episode_id",
        "anonymous_batches_as_distinct_games": int(anonymous),
        "n_games": len(group_ids),
        "n_train_games": len(train_ids),
        "n_heldout_games": len(heldout_ids),
        "n_train_batches": len(train),
        "n_heldout_batches": len(heldout),
        "train_game_ids_digest": _stable_group_digest(train_ids),
        "heldout_game_ids_digest": _stable_group_digest(heldout_ids),
        "heldout_available": bool(heldout_ids),
    }


def heuristic_should_recurse(
    n_legal: int,
    policy_logits: Optional[Tensor],
    *,
    option_threshold: int = 8,
    entropy_threshold: float = 1.5,
) -> bool:
    if n_legal <= 1:
        return False
    by_options = n_legal >= option_threshold
    entropy = 0.0
    if policy_logits is not None and policy_logits.numel() > 0:
        probs = F.softmax(policy_logits.float(), dim=-1)
        entropy = float((-(probs * torch.log(probs.clamp_min(1e-12))).sum()).item())
    by_entropy = entropy >= entropy_threshold
    return bool(by_options or by_entropy)


def root_plan_target_for_choice(
    chosen_index: int,
    n_legal: int,
    n_candidates: int,
) -> int:
    if n_legal <= 0 or n_candidates <= 0:
        return 0
    stride = max(1, n_legal // n_candidates)
    # Invert the skeleton indexing used in RecursiveTurnPlanner._candidate_skeleton.
    best = 0
    best_dist = 10**9
    for cand in range(n_candidates):
        primary = min(cand * stride, n_legal - 1)
        dist = abs(primary - int(chosen_index))
        if dist < best_dist:
            best_dist = dist
            best = cand
    return best


def make_synthetic_batches(
    *,
    n_decisions: int = 64,
    d_model: int = 96,
    max_legal: int = 8,
    seed: int = 0,
    option_threshold: int = 8,
    entropy_threshold: float = 1.5,
) -> list[RTPDecisionBatch]:
    """Deterministic synthetic multi-turn feature batches for smoke training."""
    g = torch.Generator().manual_seed(int(seed))
    batches: list[RTPDecisionBatch] = []
    prev_state: Optional[Tensor] = None
    for i in range(n_decisions):
        n_legal = 2 + int(torch.randint(0, max(1, max_legal - 1), (1,), generator=g).item())
        state = torch.randn(d_model, generator=g)
        option_hidden = torch.randn(n_legal, d_model, generator=g)
        # Make chosen option slightly aligned with state for learnability.
        chosen = int(torch.randint(0, n_legal, (1,), generator=g).item())
        option_hidden[chosen] = option_hidden[chosen] + 0.35 * state
        legal = [[j] for j in range(n_legal)]
        fake_logits = option_hidden @ state
        should = heuristic_should_recurse(
            n_legal,
            fake_logits,
            option_threshold=option_threshold,
            entropy_threshold=entropy_threshold,
        )
        if prev_state is not None and batches:
            batches[-1].next_state = state.clone()
        batches.append(
            RTPDecisionBatch(
                state=state,
                option_hidden=option_hidden,
                legal_actions=legal,
                chosen_index=chosen,
                should_recurse=should,
                # Synthetic data has no observed root-plan candidate label.
                root_plan_target=None,
                game_value=float((-1.0) ** i),
                outcome_available=True,
                episode_id=f"synthetic-game-{i // 4:05d}",
                action_space_source="runtime_complete_synthetic",
            )
        )
        prev_state = state
    return batches


def _action_outputs_with_grad(
    planner: RecursiveTurnPlanner,
    state: Tensor,
    option_hidden: Tensor,
    legal_actions: Sequence[Sequence[int]],
) -> dict[str, Tensor]:
    """Score complete legal actions with gradients.

    ``LatentTransitionDynamics.score_actions`` is intentionally no-grad for
    serving.  Shadow training calls the module directly and retains selected
    value/uncertainty tensors for outcome and calibration supervision.
    """
    embeds, _src = planner.dynamics.resolve_action_embeds(
        legal_actions,
        option_hidden=option_hidden,
        device=state.device,
        dtype=state.dtype,
    )
    out = planner.dynamics(state, embeds)
    scores = out["value"] - planner.config.compute_cost_penalty * out["uncertainty"]
    # Tiny learned bias from subgoal_action_head on state (matches refine mix).
    bias = planner.subgoal_action_head(state.unsqueeze(0)).squeeze()
    scores = scores + 0.05 * bias
    return {
        "scores": scores,
        "next_latent": out["next_latent"],
        "value": out["value"],
        "uncertainty": out["uncertainty"],
    }


def _action_scores_with_grad(
    planner: RecursiveTurnPlanner,
    state: Tensor,
    option_hidden: Tensor,
    legal_actions: Sequence[Sequence[int]],
) -> tuple[Tensor, Tensor]:
    """Backward-compatible score helper used by focused smoke tests."""
    out = _action_outputs_with_grad(planner, state, option_hidden, legal_actions)
    return out["scores"], out["next_latent"]


def train_step(
    planner: RecursiveTurnPlanner,
    batch: RTPDecisionBatch,
    *,
    cfg: RTPTrainConfig,
) -> tuple[Tensor, dict[str, float]]:
    device = next(planner.parameters()).device
    state = batch.state.to(device)
    option_hidden = batch.option_hidden.to(device)
    outputs = _action_outputs_with_grad(planner, state, option_hidden, batch.legal_actions)
    scores = outputs["scores"]
    next_latents = outputs["next_latent"]
    values = outputs["value"]
    uncertainties = outputs["uncertainty"]
    chosen = int(batch.chosen_index)
    if 0 <= chosen < next_latents.size(0):
        pred_next = next_latents[chosen]
        chosen_value = values[chosen]
        chosen_uncertainty = uncertainties[chosen]
    else:
        # Missing/invalid action targets stay masked in compute_rtp_losses.
        pred_next = None
        chosen_value = None
        chosen_uncertainty = None
    target_next = batch.next_state.to(device) if batch.next_state is not None else None

    complexity_logit = planner.complexity_head(state.unsqueeze(0)).squeeze()
    root_logits = planner.root_plan_head(state.unsqueeze(0)).squeeze(0)

    bundle = compute_rtp_losses(
        action_scores=scores,
        chosen_action_index=chosen,
        complexity_logit=complexity_logit,
        should_recurse=batch.should_recurse,
        predicted_next_latent=(
            pred_next.unsqueeze(0)
            if target_next is not None and pred_next is not None
            else None
        ),
        target_next_latent=(
            target_next.unsqueeze(0)
            if target_next is not None and pred_next is not None
            else None
        ),
        root_plan_logits=root_logits,
        root_plan_target=batch.root_plan_target,
        chosen_value_prediction=chosen_value,
        chosen_uncertainty=chosen_uncertainty,
        game_value=(batch.game_value if batch.outcome_available else None),
        candidate_return_predictions=values,
        candidate_return_targets=(
            None
            if batch.candidate_return_targets is None
            else batch.candidate_return_targets.to(device)
        ),
        candidate_return_mask=(
            None
            if batch.candidate_return_mask is None
            else batch.candidate_return_mask.to(device)
        ),
        candidate_ranking_scores=scores,
        candidate_ranking_targets=(
            None
            if batch.candidate_ranking_targets is None
            else batch.candidate_ranking_targets.to(device)
        ),
        candidate_ranking_mask=(
            None
            if batch.candidate_ranking_mask is None
            else batch.candidate_ranking_mask.to(device)
        ),
        candidate_uncertainty_predictions=uncertainties,
        candidate_calibration_targets=(
            None
            if batch.candidate_calibration_targets is None
            else batch.candidate_calibration_targets.to(device)
        ),
        candidate_calibration_mask=(
            None
            if batch.candidate_calibration_mask is None
            else batch.candidate_calibration_mask.to(device)
        ),
        action_weight=cfg.action_weight,
        ranking_weight=cfg.ranking_weight,
        complexity_weight=cfg.complexity_weight,
        dynamics_weight=cfg.dynamics_weight,
        value_weight=cfg.value_weight,
        calibration_weight=cfg.calibration_weight,
        candidate_return_weight=cfg.candidate_return_weight,
        candidate_ranking_weight=cfg.candidate_ranking_weight,
        candidate_calibration_weight=cfg.candidate_calibration_weight,
        root_plan_weight=cfg.root_plan_weight,
    )
    metrics = bundle.as_dict()
    metrics.update(
        {
            f"{name}_available": float(bool(available))
            for name, available in bundle.metadata.items()
            if name.endswith("_available")
        }
    )
    metrics.update(
        {
            name: float(count)
            for name, count in bundle.metadata.items()
            if name.endswith("_count")
        }
    )
    return bundle.total, metrics


def _mean_metric_rows(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {
        key: sum(float(row.get(key, 0.0)) for row in rows) / float(len(rows))
        for key in keys
    }


@torch.no_grad()
def evaluate_rtp_shadow(
    planner: RecursiveTurnPlanner,
    batches: Sequence[RTPDecisionBatch],
    *,
    cfg: RTPTrainConfig,
) -> dict[str, Any]:
    """Evaluate only on heldout games, with selected-action outcome metrics.

    The score reports demonstrated-choice rank and selected-action terminal
    value/calibration.  It deliberately has no per-unchosen-action outcome
    metric because the shard does not observe those counterfactuals.
    """
    if not batches:
        return {
            "available": False,
            "n_batches": 0,
            "reason": "no_distinct_heldout_games",
        }

    was_training = planner.training
    planner.eval()
    rows: list[dict[str, float]] = []
    ranks: list[float] = []
    top1: list[float] = []
    value_errors: list[float] = []
    calibration_errors: list[float] = []
    outcome_count = 0
    for batch in batches:
        loss, row = train_step(planner, batch, cfg=cfg)
        row = {**row, "loss": float(loss.item())}
        rows.append(row)

        device = next(planner.parameters()).device
        state = batch.state.to(device)
        option_hidden = batch.option_hidden.to(device)
        outputs = _action_outputs_with_grad(
            planner, state, option_hidden, batch.legal_actions
        )
        scores = outputs["scores"]
        chosen = int(batch.chosen_index)
        if 0 <= chosen < scores.numel():
            chosen_score = scores[chosen]
            rank = 1 + int((scores > chosen_score).sum().item())
            ranks.append(float(rank))
            top1.append(1.0 if rank == 1 else 0.0)
            if (
                batch.outcome_available
                and batch.game_value is not None
                and math.isfinite(float(batch.game_value))
                and -1.0 <= float(batch.game_value) <= 1.0
            ):
                target = float(batch.game_value)
                value = float(outputs["value"][chosen].item())
                uncertainty = float(outputs["uncertainty"][chosen].item())
                error = abs(value - target)
                value_errors.append(error)
                calibration_errors.append(abs(uncertainty - min(1.0, error)))
                outcome_count += 1

    if was_training:
        planner.train()
    means = _mean_metric_rows(rows)
    metrics: dict[str, Any] = {
        "available": True,
        "n_batches": len(batches),
        "observed_choice_count": len(ranks),
        "observed_terminal_outcome_count": outcome_count,
        "mean_loss": means.get("loss", 0.0),
        "mean_action_loss": means.get("action", 0.0),
        "mean_behavioral_ranking_loss": means.get("ranking", 0.0),
        "mean_value_loss": means.get("value", 0.0),
        "mean_calibration_loss": means.get("calibration", 0.0),
        "mean_candidate_return_loss": means.get("candidate_return", 0.0),
        "mean_candidate_ranking_loss": means.get("candidate_ranking", 0.0),
        "mean_candidate_calibration_loss": means.get(
            "candidate_calibration", 0.0
        ),
        "mean_candidate_return_target_count": means.get(
            "candidate_return_target_count", 0.0
        ),
        "mean_candidate_ranking_pair_count": means.get(
            "candidate_ranking_pair_count", 0.0
        ),
        "mean_candidate_calibration_target_count": means.get(
            "candidate_calibration_target_count", 0.0
        ),
    }
    if ranks:
        metrics["chosen_top1_rate"] = sum(top1) / float(len(top1))
        metrics["chosen_mean_rank"] = sum(ranks) / float(len(ranks))
    if value_errors:
        metrics["selected_value_mae"] = sum(value_errors) / float(len(value_errors))
        metrics["selected_value_mse"] = sum(
            error * error for error in value_errors
        ) / float(len(value_errors))
        metrics["selected_uncertainty_calibration_mae"] = sum(
            calibration_errors
        ) / float(len(calibration_errors))
    return metrics


def train_rtp_shadow(
    batches: Sequence[RTPDecisionBatch],
    *,
    output_dir: Path | str,
    config: Optional[RTPTrainConfig] = None,
    planner: Optional[RecursiveTurnPlanner] = None,
    heldout_batches: Optional[Sequence[RTPDecisionBatch]] = None,
    provenance: Optional[Mapping[str, Any]] = None,
    parent_checkpoint_sha256: Optional[str] = None,
) -> RTPTrainResult:
    """Train an RTP sidecar and evaluate a pre-split heldout game set.

    This function never chooses a split itself: callers must provide a
    game-level heldout partition so training cannot leak decisions from one
    episode into validation.
    """
    cfg = config or RTPTrainConfig()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(cfg.seed))

    if planner is None:
        try:
            profile = get_profile(cfg.profile)
            if str(cfg.profile).strip().lower() == "pure_rl_r197" and (
                str(profile.name) != "pure_rl_r197"
                or int(profile.d_model) != 96
                or int(profile.num_plan_candidates) != 4
                or int(profile.max_recursion_depth) != 2
                or int(profile.max_neural_passes)
                != RTP_MAX_AUTHORIZED_NEURAL_PASSES
            ):
                raise ValueError(
                    "pure_rl_r197 profile is not the revision-198 "
                    "96/4/depth-2/256-pass contract"
                )
            rtp_cfg = profile.to_config(
                d_model=int(cfg.d_model),
                num_plan_candidates=int(cfg.num_plan_candidates),
                max_recursion_depth=int(cfg.max_recursion_depth),
                max_neural_passes=int(cfg.max_neural_passes),
                complexity_option_threshold=int(cfg.complexity_option_threshold),
                complexity_entropy_threshold=float(cfg.complexity_entropy_threshold),
            )
        except Exception:
            # A generic research caller may supply a custom sizing profile,
            # but the production r197 identity must never silently fall back
            # to an ad-hoc configuration if its registered profile is stale
            # or unavailable.
            if str(cfg.profile).strip().lower() == "pure_rl_r197":
                raise
            rtp_cfg = RTPConfig(
                sizing_profile=cfg.profile,
                d_model=int(cfg.d_model),
                dynamics_width=max(32, 2 * int(cfg.d_model)),
                num_plan_candidates=int(cfg.num_plan_candidates),
                max_recursion_depth=int(cfg.max_recursion_depth),
                max_neural_passes=int(cfg.max_neural_passes),
                complexity_option_threshold=int(cfg.complexity_option_threshold),
                complexity_entropy_threshold=float(cfg.complexity_entropy_threshold),
            )
        # Bind exact width from config when synthetic/custom.
        if int(rtp_cfg.d_model) != int(cfg.d_model):
            rtp_cfg = RTPConfig(
                sizing_profile=cfg.profile,
                d_model=int(cfg.d_model),
                dynamics_width=max(32, 2 * int(cfg.d_model)),
                num_plan_candidates=int(cfg.num_plan_candidates),
                max_recursion_depth=int(cfg.max_recursion_depth),
                max_neural_passes=int(cfg.max_neural_passes),
                complexity_option_threshold=int(cfg.complexity_option_threshold),
                complexity_entropy_threshold=float(cfg.complexity_entropy_threshold),
            )
        planner = RecursiveTurnPlanner(rtp_cfg)

    if str(cfg.profile).strip().lower() == "pure_rl_r197" and (
        str(planner.config.sizing_profile) != "pure_rl_r197"
        or int(planner.config.d_model) != 96
        or int(planner.config.num_plan_candidates) != 4
        or int(planner.config.max_recursion_depth) != 2
        or int(planner.config.max_neural_passes)
        != RTP_MAX_AUTHORIZED_NEURAL_PASSES
    ):
        raise ValueError(
            "pure_rl_r197 shadow training requires the registered revision-198 "
            "96/4/depth-2/256-pass planner"
        )

    device = torch.device(cfg.device)
    planner.to(device)
    planner.train()
    opt = torch.optim.AdamW(planner.parameters(), lr=float(cfg.lr))

    history: list[dict[str, float]] = []
    last: dict[str, Any] = {}
    for epoch in range(int(cfg.epochs)):
        total = 0.0
        n = 0
        for batch in batches:
            opt.zero_grad(set_to_none=True)
            loss, metrics = train_step(planner, batch, cfg=cfg)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("RTP shadow loss is non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(planner.parameters(), 1.0)
            opt.step()
            row = {
                "loss": float(loss.detach().item()),
                **metrics,
                "epoch": float(epoch),
            }
            history.append(row)
            total += row["loss"]
            n += 1
        last = {
            "epoch": float(epoch),
            "mean_loss": total / max(1, n),
            "n_steps": float(n),
            "n_train_batches": int(len(batches)),
        }

    heldout = list(heldout_batches or [])
    heldout_metrics = evaluate_rtp_shadow(planner, heldout, cfg=cfg)
    required_passes = required_recursive_passes(
        num_plan_candidates=int(planner.config.num_plan_candidates),
        max_recursion_depth=int(planner.config.max_recursion_depth),
    )
    last["heldout_available"] = bool(heldout_metrics.get("available", False))
    last["n_heldout_batches"] = int(len(heldout))
    if heldout_metrics.get("available"):
        last["heldout_mean_loss"] = float(heldout_metrics["mean_loss"])

    provenance_payload = dict(provenance or {})
    provenance_parent = str(provenance_payload.get("parent_digest") or "").strip()
    if parent_checkpoint_sha256 and provenance_parent and (
        str(parent_checkpoint_sha256).strip().lower() != provenance_parent.lower()
    ):
        raise ValueError(
            "parent_checkpoint_sha256 conflicts with provenance parent_digest"
        )
    bound_parent = parent_checkpoint_sha256 or provenance_parent or None

    ckpt_path = out / "rtp_shadow_planner.pt"
    receipt = save_rtp_checkpoint(
        planner,
        ckpt_path,
        metrics=last,
        parent_checkpoint_sha256=bound_parent,
        shadow_only=True,
        research_only=False,
        extra={
            "train_config": asdict(cfg),
            "n_batches": len(batches),
            "n_heldout_batches": len(heldout),
            "heldout_metrics": heldout_metrics,
            "provenance": provenance_payload,
            "required_recursive_passes": required_passes,
            "shadow_only": True,
            "generated_at_unix": time.time(),
        },
    )
    return RTPTrainResult(
        checkpoint_path=str(ckpt_path.resolve()),
        receipt_path=str(Path(ckpt_path.with_suffix(ckpt_path.suffix + ".receipt.json")).resolve()),
        metrics=last,
        heldout_metrics=heldout_metrics,
        history=history,
        inventory=dict(planner.inventory()),
    )


def encode_sequences_to_batches(
    model: Any,
    sequences: Sequence[Any],
    *,
    option_threshold: int = 8,
    entropy_threshold: float = 1.5,
    num_plan_candidates: int = 4,
    runtime_records: Optional[Mapping[tuple[Any, ...], Mapping[str, Any]]] = None,
    runtime_action_fingerprint: Optional[
        Callable[[Mapping[str, Any], Sequence[Sequence[int]]], str]
    ] = None,
    require_complete_ordered_actions: bool = False,
    max_runtime_action_combos: int = 256,
    return_provenance: bool = False,
) -> list[RTPDecisionBatch] | tuple[list[RTPDecisionBatch], dict[str, Any]]:
    """Encode training rows with runtime-equivalent whole-action support.

    ``runtime_records`` carries the source's causal observation and selected
    ordered action keyed by ``(episode_id, seat, env_step)``.  A two-part
    ``(episode_id, env_step)`` key is accepted for older callers.  When a
    record also provides ``legal_actions`` / ``selected_action_index`` / an
    action-space fingerprint (the r197 corpus path), all three are checked
    against a fresh runtime enumeration before any batch is emitted.  When
    present, this function uses
    this function uses :func:`features.enumerate_action_combos` plus the same
    option decoder shape that the runtime bridge uses.  This is the only path
    that honestly yields complete ordered action candidates and their matching
    option embeddings from the current source data.

    If a legacy source has only factorized stages, it is explicitly tagged as
    such and may be retained only when ``require_complete_ordered_actions`` is
    false.  No guessed full action space, root-plan index, or unchosen outcome
    target is manufactured from factorized prefixes.
    """
    from poke_bot.dataset import GameSequence, PolicyStage
    from poke_bot import features

    if int(max_runtime_action_combos) < 1:
        raise ValueError("max_runtime_action_combos must be positive")

    device = next(model.parameters()).device
    model.eval()
    batches: list[RTPDecisionBatch] = []
    accounting: dict[str, int] = {
        "sequences_seen": 0,
        "decisions_seen": 0,
        "runtime_complete_batches": 0,
        "factorized_fallback_batches": 0,
        "skipped_missing_runtime_record": 0,
        "skipped_runtime_action_space_too_large": 0,
        "skipped_runtime_action_invalid": 0,
        "skipped_runtime_decode_error": 0,
        "skipped_factorized": 0,
        "candidate_evaluator_bound_batches": 0,
        "candidate_evaluator_masked_batches": 0,
    }

    def terminal_value(sequence: Any, record: Optional[Mapping[str, Any]]) -> tuple[Optional[float], bool]:
        value = (
            record.get("game_value")
            if record is not None and "game_value" in record
            else getattr(sequence, "value", None)
        )
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None, False
        sequence_provenance = dict(getattr(sequence, "target_provenance", {}) or {})
        terminal_complete = not bool(sequence_provenance.get("terminal_policy_failure"))
        if record is not None:
            terminal_complete = bool(record.get("terminal_complete", terminal_complete))
            if "outcome_available" in record:
                terminal_complete = terminal_complete and bool(record["outcome_available"])
        if not terminal_complete or not math.isfinite(numeric) or not -1.0 <= numeric <= 1.0:
            return None, False
        return numeric, True

    def decode_runtime_options(
        *,
        observation: Mapping[str, Any],
        combos: Sequence[Sequence[int]],
        spatial_row: Tensor,
        state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        runtime_options = features.build_option_tokens(
            dict(observation), [list(combo) for combo in combos]
        )
        decoded = model.decode_options(
            [runtime_options],
            spatial_row,
            state.unsqueeze(0),
            return_hidden=True,
        )
        if not isinstance(decoded, tuple):
            raise RuntimeError("model.decode_options did not return option_hidden")
        logits, option_hidden = decoded
        return logits[0, : len(combos)], option_hidden[0, : len(combos)]

    with torch.no_grad():
        for sequence in sequences:
            if not isinstance(sequence, GameSequence) or not sequence.decisions:
                continue
            accounting["sequences_seen"] += 1
            decisions = list(sequence.decisions)
            boards = [d.board for d in decisions]
            previous_actions = [None] + [d.action_token for d in decisions[:-1]]
            spatial = model.encode_board(boards)
            cls = model.pool_cls(spatial) + float(model.cfg.history_action_scale) * (
                model.encode_previous_actions(previous_actions)
            )
            encoded, _ = model.temporal_encode(cls.unsqueeze(0), append=False, return_all=True)
            states = encoded.squeeze(0)
            for local_index, decision in enumerate(decisions):
                accounting["decisions_seen"] += 1
                episode_id = str(getattr(sequence, "episode_id", "") or "")
                record = None
                if runtime_records is not None:
                    record = runtime_records.get(
                        (
                            episode_id,
                            int(getattr(sequence, "seat", -1)),
                            int(decision.env_step),
                        )
                    ) or runtime_records.get((episode_id, int(decision.env_step)))
                outcome_value, outcome_available = terminal_value(sequence, record)

                if record is not None:
                    try:
                        observation = record.get("observation")
                        if not isinstance(observation, Mapping):
                            raise ValueError("runtime record observation missing")
                        declared_source = record.get("action_space_source")
                        if declared_source is not None and str(declared_source) != (
                            "runtime_complete_observation"
                        ):
                            raise ValueError(
                                "runtime record is not a complete-observation action space"
                            )
                        if record.get("unobserved_action_targets_present") is True:
                            raise ValueError(
                                "runtime record claims unobserved action targets"
                            )
                        combos = features.enumerate_action_combos(
                            dict(observation),
                            max_combos=int(max_runtime_action_combos),
                        )
                        legal_actions = [list(combo) for combo in combos]
                        declared_legal = record.get("legal_actions")
                        if declared_legal is not None:
                            if not isinstance(declared_legal, (list, tuple)):
                                raise ValueError("declared legal_actions is malformed")
                            try:
                                declared_normalized = [
                                    [int(value) for value in list(action)]
                                    for action in declared_legal
                                ]
                            except (TypeError, ValueError) as exc:
                                raise ValueError(
                                    "declared legal_actions is malformed"
                                ) from exc
                            if declared_normalized != legal_actions:
                                raise ValueError(
                                    "complete ordered legal_actions mismatch runtime enumeration"
                                )
                        source_action = record.get("action", decision.action)
                        selected_action = [int(value) for value in list(source_action)]
                        if selected_action != [int(value) for value in list(decision.action)]:
                            raise ValueError("runtime record action does not match sequence action")
                        declared_selected = record.get("selected_action_index")
                        if declared_selected is not None:
                            if isinstance(declared_selected, bool) or not isinstance(
                                declared_selected, int
                            ):
                                raise ValueError("selected_action_index is malformed")
                            selected = int(declared_selected)
                            if not 0 <= selected < len(legal_actions):
                                raise ValueError(
                                    "selected_action_index outside complete support"
                                )
                            if legal_actions[selected] != selected_action:
                                raise ValueError(
                                    "selected_action_index does not select recorded action"
                                )
                        else:
                            try:
                                selected = legal_actions.index(selected_action)
                            except ValueError as exc:
                                raise ValueError(
                                    "selected ordered action absent from complete support"
                                ) from exc
                        action_space_fingerprint = ""
                        declared_fingerprint = record.get("action_space_fingerprint")
                        if declared_fingerprint is not None:
                            if runtime_action_fingerprint is None:
                                raise ValueError(
                                    "cannot verify declared action_space_fingerprint"
                                )
                            computed_fingerprint = str(
                                runtime_action_fingerprint(observation, legal_actions)
                            )
                            if computed_fingerprint != str(declared_fingerprint):
                                raise ValueError(
                                    "action_space_fingerprint mismatch runtime enumeration"
                                )
                            action_space_fingerprint = computed_fingerprint
                        state = states[local_index]
                        spat = spatial[local_index : local_index + 1]
                        logits, option_hidden = decode_runtime_options(
                            observation=observation,
                            combos=legal_actions,
                            spatial_row=spat,
                            state=state,
                        )
                    except features.ActionSpaceTooLarge:
                        accounting["skipped_runtime_action_space_too_large"] += 1
                    except ValueError:
                        accounting["skipped_runtime_action_invalid"] += 1
                    except Exception:
                        accounting["skipped_runtime_decode_error"] += 1
                    else:
                        next_state = (
                            states[local_index + 1]
                            if local_index + 1 < len(decisions)
                            else None
                        )
                        root_target = record.get("root_plan_target")
                        if not isinstance(root_target, int) or not (
                            0 <= int(root_target) < int(num_plan_candidates)
                        ):
                            root_target = None
                        evaluator_targets = trusted_candidate_targets_from_record(
                            record,
                            n_actions=len(legal_actions),
                            action_space_fingerprint=action_space_fingerprint,
                        )
                        batches.append(
                            RTPDecisionBatch(
                                state=state.detach().cpu(),
                                option_hidden=option_hidden.detach().cpu(),
                                legal_actions=legal_actions,
                                chosen_index=int(selected),
                                # Raw shard actions do not record an observed
                                # value-of-planning label, so leave the gate
                                # target masked rather than recreate a
                                # complexity heuristic as ground truth.
                                should_recurse=None,
                                next_state=(
                                    None
                                    if next_state is None
                                    else next_state.detach().cpu()
                                ),
                                root_plan_target=root_target,
                                game_value=outcome_value,
                                outcome_available=outcome_available,
                                episode_id=episode_id,
                                sequence_window_id=str(
                                    record.get("sequence_window_id") or ""
                                ),
                                action_space_source="runtime_complete_observation",
                                candidate_return_targets=evaluator_targets[
                                    "candidate_return_targets"
                                ],
                                candidate_return_mask=evaluator_targets[
                                    "candidate_return_mask"
                                ],
                                candidate_ranking_targets=evaluator_targets[
                                    "candidate_ranking_targets"
                                ],
                                candidate_ranking_mask=evaluator_targets[
                                    "candidate_ranking_mask"
                                ],
                                candidate_calibration_targets=evaluator_targets[
                                    "candidate_calibration_targets"
                                ],
                                candidate_calibration_mask=evaluator_targets[
                                    "candidate_calibration_mask"
                                ],
                                candidate_target_provenance=dict(
                                    evaluator_targets["provenance"]
                                ),
                            )
                        )
                        accounting["runtime_complete_batches"] += 1
                        if evaluator_targets["provenance"].get("status") == (
                            "trusted_action_space_bound"
                        ):
                            accounting["candidate_evaluator_bound_batches"] += 1
                        else:
                            accounting["candidate_evaluator_masked_batches"] += 1
                        continue
                elif runtime_records is not None:
                    accounting["skipped_missing_runtime_record"] += 1

                if require_complete_ordered_actions:
                    accounting["skipped_factorized"] += 1
                    continue

                stages = list(decision.policy_stages) or [
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
                    state = states[local_index]
                    spat = spatial[local_index : local_index + 1]
                    decoded = model.decode_options(
                        [stage.options],
                        spat,
                        state.unsqueeze(0),
                        return_hidden=True,
                    )
                    if isinstance(decoded, tuple):
                        logits, option_hidden = decoded
                        logits = logits[0, :count]
                        option_hidden = option_hidden[0, :count]
                    else:
                        logits = decoded[0, :count]
                        option_hidden = torch.randn(
                            count, int(state.numel()), device=device
                        )
                    should = heuristic_should_recurse(
                        count,
                        logits,
                        option_threshold=option_threshold,
                        entropy_threshold=entropy_threshold,
                    )
                    next_state = (
                        states[local_index + 1]
                        if local_index + 1 < len(decisions)
                        else None
                    )
                    batches.append(
                        RTPDecisionBatch(
                            state=state.detach().cpu(),
                            option_hidden=option_hidden.detach().cpu()[:count],
                            legal_actions=[list(c) for c in stage.action_combos[:count]],
                            chosen_index=selected,
                            # A factorized stage is not a whole runtime action
                            # and has no observed value-of-planning target.
                            should_recurse=None,
                            next_state=None if next_state is None else next_state.detach().cpu(),
                            root_plan_target=None,
                            game_value=outcome_value,
                            outcome_available=outcome_available,
                            episode_id=episode_id,
                            action_space_source="factorized_stage_legacy",
                        )
                    )
                    accounting["factorized_fallback_batches"] += 1

    provenance = {
        "schema": "poke_bot.recursive_turn_planner.batch_encoding/v2",
        "require_complete_ordered_actions": bool(require_complete_ordered_actions),
        "max_runtime_action_combos": int(max_runtime_action_combos),
        "runtime_records_supplied": runtime_records is not None,
        "runtime_action_fingerprint_verifier_supplied": (
            runtime_action_fingerprint is not None
        ),
        **accounting,
    }
    if return_provenance:
        return batches, provenance
    return batches
