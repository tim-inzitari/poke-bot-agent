"""Offline / shadow training for the Recursive Turn Planner.

Trains planner + dynamics heads while the CABT encoder stays frozen (when a
parent checkpoint is supplied). Supports a synthetic path for CI without shards.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from poke_bot.recursive_turn_planner.config import RTPConfig
from poke_bot.recursive_turn_planner.planner import RecursiveTurnPlanner
from poke_bot.recursive_turn_planner.profiles import get_profile
from poke_bot.recursive_turn_planner.training.checkpoint import save_rtp_checkpoint
from poke_bot.recursive_turn_planner.training.losses import compute_rtp_losses


@dataclass
class RTPDecisionBatch:
    """One training decision in encoder feature space."""

    state: Tensor  # [D]
    option_hidden: Tensor  # [N, D]
    legal_actions: list[list[int]]
    chosen_index: int
    should_recurse: bool
    next_state: Optional[Tensor] = None
    root_plan_target: Optional[int] = None
    game_value: float = 0.0


@dataclass
class RTPTrainConfig:
    d_model: int = 96
    profile: str = "pure_rl"
    epochs: int = 2
    lr: float = 1e-3
    seed: int = 0
    action_weight: float = 1.0
    complexity_weight: float = 0.25
    dynamics_weight: float = 0.5
    root_plan_weight: float = 0.15
    complexity_option_threshold: int = 8
    complexity_entropy_threshold: float = 1.5
    device: str = "cpu"


@dataclass
class RTPTrainResult:
    checkpoint_path: str
    receipt_path: str
    metrics: dict[str, float]
    history: list[dict[str, float]] = field(default_factory=list)
    inventory: dict[str, Any] = field(default_factory=dict)


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
                root_plan_target=root_plan_target_for_choice(chosen, n_legal, 4),
                game_value=float((-1.0) ** i),
            )
        )
        prev_state = state
    return batches


def _action_scores_with_grad(
    planner: RecursiveTurnPlanner,
    state: Tensor,
    option_hidden: Tensor,
    legal_actions: Sequence[Sequence[int]],
) -> tuple[Tensor, Tensor]:
    """Score legal actions with gradients (dynamics.score_actions is no_grad)."""
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
    # Chosen-action next latent for dynamics supervision (index later).
    return scores, out["next_latent"]


def train_step(
    planner: RecursiveTurnPlanner,
    batch: RTPDecisionBatch,
    *,
    cfg: RTPTrainConfig,
) -> tuple[Tensor, dict[str, float]]:
    device = next(planner.parameters()).device
    state = batch.state.to(device)
    option_hidden = batch.option_hidden.to(device)
    scores, next_latents = _action_scores_with_grad(
        planner, state, option_hidden, batch.legal_actions
    )
    chosen = int(batch.chosen_index)
    if 0 <= chosen < next_latents.size(0):
        pred_next = next_latents[chosen]
    else:
        pred_next = next_latents[0]
    target_next = batch.next_state.to(device) if batch.next_state is not None else None

    complexity_logit = planner.complexity_head(state.unsqueeze(0)).squeeze()
    root_logits = planner.root_plan_head(state.unsqueeze(0)).squeeze(0)

    bundle = compute_rtp_losses(
        action_scores=scores,
        chosen_action_index=chosen,
        complexity_logit=complexity_logit,
        should_recurse=batch.should_recurse,
        predicted_next_latent=pred_next.unsqueeze(0) if target_next is not None else None,
        target_next_latent=target_next.unsqueeze(0) if target_next is not None else None,
        root_plan_logits=root_logits,
        root_plan_target=batch.root_plan_target,
        action_weight=cfg.action_weight,
        complexity_weight=cfg.complexity_weight,
        dynamics_weight=cfg.dynamics_weight,
        root_plan_weight=cfg.root_plan_weight,
    )
    return bundle.total, bundle.as_dict()


def train_rtp_shadow(
    batches: Sequence[RTPDecisionBatch],
    *,
    output_dir: Path | str,
    config: Optional[RTPTrainConfig] = None,
    planner: Optional[RecursiveTurnPlanner] = None,
) -> RTPTrainResult:
    """Train RTP heads on feature batches; write sidecar checkpoint."""
    cfg = config or RTPTrainConfig()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(cfg.seed))

    if planner is None:
        try:
            profile = get_profile(cfg.profile)
            rtp_cfg = profile.to_config()
        except Exception:
            rtp_cfg = RTPConfig(
                sizing_profile=cfg.profile,
                d_model=int(cfg.d_model),
                dynamics_width=max(32, 2 * int(cfg.d_model)),
            )
        # Bind exact width from config when synthetic/custom.
        if int(rtp_cfg.d_model) != int(cfg.d_model):
            rtp_cfg = RTPConfig(
                sizing_profile=cfg.profile,
                d_model=int(cfg.d_model),
                dynamics_width=max(32, 2 * int(cfg.d_model)),
                num_plan_candidates=rtp_cfg.num_plan_candidates,
                max_recursion_depth=rtp_cfg.max_recursion_depth,
                complexity_option_threshold=int(cfg.complexity_option_threshold),
                complexity_entropy_threshold=float(cfg.complexity_entropy_threshold),
            )
        planner = RecursiveTurnPlanner(rtp_cfg)

    device = torch.device(cfg.device)
    planner.to(device)
    planner.train()
    opt = torch.optim.AdamW(planner.parameters(), lr=float(cfg.lr))

    history: list[dict[str, float]] = []
    last: dict[str, float] = {}
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
            row = {"loss": float(loss.detach().item()), **metrics, "epoch": float(epoch)}
            history.append(row)
            total += row["loss"]
            n += 1
        last = {
            "epoch": float(epoch),
            "mean_loss": total / max(1, n),
            "n_steps": float(n),
        }

    ckpt_path = out / "rtp_shadow_planner.pt"
    receipt = save_rtp_checkpoint(
        planner,
        ckpt_path,
        metrics=last,
        extra={
            "train_config": asdict(cfg),
            "n_batches": len(batches),
            "generated_at_unix": time.time(),
        },
    )
    return RTPTrainResult(
        checkpoint_path=str(ckpt_path.resolve()),
        receipt_path=str(Path(ckpt_path.with_suffix(ckpt_path.suffix + ".receipt.json")).resolve()),
        metrics=last,
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
) -> list[RTPDecisionBatch]:
    """Encode GameSequences with a frozen TemporalCabtTransformer into RTP batches."""
    from poke_bot.dataset import GameSequence, PolicyStage

    device = next(model.parameters()).device
    model.eval()
    batches: list[RTPDecisionBatch] = []
    with torch.no_grad():
        for sequence in sequences:
            if not isinstance(sequence, GameSequence) or not sequence.decisions:
                continue
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
                            should_recurse=should,
                            next_state=None if next_state is None else next_state.detach().cpu(),
                            root_plan_target=root_plan_target_for_choice(
                                selected, count, num_plan_candidates
                            ),
                            game_value=float(getattr(sequence, "value", 0.0)),
                        )
                    )
    return batches
