"""Supervised bootstrap / policy-value training with realized histories.

Every decision is trained causally on the same acting-seat observation history
that trusted serving consumes incrementally.
"""

from __future__ import annotations

import copy
import gc
import math
import random
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from . import archetypes, checkpoint, config, device as device_mod, features
from .blackwell_heads import (
    BLACKWELL_STRATEGY_HEAD_PREFIXES,
    lethal_target_from_aux,
    masked_bce_logit,
    masked_smooth_l1,
    prize_race_target_from_aux,
)
from .dataset import BootstrapDataset, GameSequence, PolicyStage
from .device_corpus import DEFAULT_MIN_FREE_GIB, DeviceResidentBootstrapCorpus
from .model import TemporalCabtTransformer, build_model

# Distinct named belief + Blackwell strategy heads — warm-start may omit only
# these key prefixes (Scope A particle priors + Scope B Hammer heads).
BELIEF_AUX_HEAD_KEY_PREFIXES: tuple[str, ...] = (
    "opp_hand_head.",
    "opp_remainder_head.",
) + BLACKWELL_STRATEGY_HEAD_PREFIXES


@dataclass
class TrainConfig:
    """Bootstrap supervised training knobs."""

    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 20
    games_per_batch: int = 4
    max_decisions_per_batch: int = 256
    val_frac: float = 0.1
    #: Keep both acting-seat records from an episode in the same split.
    split_by_episode: bool = False
    early_stop_patience: int = 5
    value_loss_weight: float = 1.0
    aux_loss_weight: float = 0.1
    opp_hand_loss_weight: float = 0.2
    opp_remainder_loss_weight: float = 0.15
    #: Training-only soft strategy guide. This must remain zero for core and
    #: non-Alakazam runs; the specialist launcher owns the explicit opt-in.
    alakazam_guide_loss_weight: float = 0.0
    #: Scope B (Blackwell Hammer) only — keep 0.0 for core / generic bootstrap.
    lethal_threat_loss_weight: float = 0.0
    prize_race_loss_weight: float = 0.0
    grad_clip: float = 1.0
    amp: bool = True
    seed: int = 0
    log_every: int = 1
    #: When True, use AWR on selected actions (never CE to soft behavior π).
    pure_rl: bool = False
    awr_beta: float = 0.5
    awr_weight_max: float = 20.0
    #: Whitening advantages before exp(A/β) (pure_rl only).
    awr_normalize_advantages: bool = True
    #: Precompute V(s) once at the iteration boundary and reuse it for every
    #: AWR update. This makes the documented "stale critic" baseline real
    #: instead of allowing it to drift with batch order.
    awr_freeze_baseline: bool = False
    #: Subtract from policy loss: ``entropy_bonus * H(π)`` (pure_rl only).
    entropy_bonus: float = 0.01
    #: Temporal hot-start only: keep the new history state close to the copied
    #: stateless parent's normalized state during frozen-trunk calibration.
    history_identity_loss_weight: float = 0.0

    @classmethod
    def pure_rl_defaults(cls, **overrides: Any) -> "TrainConfig":
        """Single-model pure-RL knobs: AWR on, all aux/strategy weights off.

        Two measured passes over each fresh replay window are the production
        default. Callers may override this for controlled experiments.
        """
        cfg = cls(
            pure_rl=True,
            aux_loss_weight=0.0,
            opp_hand_loss_weight=0.0,
            opp_remainder_loss_weight=0.0,
            alakazam_guide_loss_weight=0.0,
            lethal_threat_loss_weight=0.0,
            prize_race_loss_weight=0.0,
            early_stop_patience=3,
            epochs=2,
            # Larger than the tiny default=4 so multi-k game shards finish under
            # the collection wall time without inflating optimizer-step count.
            games_per_batch=16,
            max_decisions_per_batch=2048,
            awr_freeze_baseline=True,
            awr_normalize_advantages=bool(
                getattr(config.PURE_RL, "normalize_advantages", True)
            ),
            entropy_bonus=float(getattr(config.PURE_RL, "entropy_bonus", 0.01)),
            awr_beta=float(getattr(config.PURE_RL, "awr_beta", 0.5)),
            awr_weight_max=float(getattr(config.PURE_RL, "awr_weight_max", 20.0)),
        )
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                raise TypeError(f"unknown TrainConfig field: {key}")
            setattr(cfg, key, value)
        return cfg


@dataclass
class BatchMetrics:
    policy_loss: float = 0.0
    value_loss: float = 0.0
    aux_loss: float = 0.0
    opp_hand_loss: float = 0.0
    opp_remainder_loss: float = 0.0
    alakazam_guide_loss: float = 0.0
    lethal_threat_loss: float = 0.0
    prize_race_loss: float = 0.0
    history_identity_loss: float = 0.0
    total_loss: float = 0.0
    policy_acc: float = 0.0
    policy_kl: float = 0.0
    target_value_mean: float = 0.0
    value_pred_mean: float = 0.0
    n_decisions: int = 0
    n_games: int = 0
    n_alakazam_guide_rows: int = 0
    n_archetype_rows: int = 0
    n_opp_hand_rows: int = 0
    n_opp_remainder_rows: int = 0
    n_lethal_threat_rows: int = 0
    n_prize_race_rows: int = 0
    # Backward-compatible pipeline signal consumed by pure_rl.aborts. It is
    # deliberately the raw mean absolute advantage, not a signed/whitened mean.
    mean_advantage: float = 0.0
    raw_advantage_mean: float = 0.0
    raw_advantage_std: float = 0.0
    raw_advantage_mean_abs: float = 0.0
    raw_advantage_mean_sq: float = 0.0
    normalized_advantage_mean: float = 0.0
    normalized_advantage_std: float = 0.0
    normalized_advantage_mean_abs: float = 0.0
    normalized_advantage_mean_sq: float = 0.0
    awr_weight_mean: float = 0.0
    awr_weight_sum: float = 0.0
    awr_weight_sq_sum: float = 0.0
    awr_weight_p50: float = 0.0
    awr_weight_p95: float = 0.0
    awr_weight_clip_frac: float = 0.0
    awr_effective_sample_size: float = 0.0
    awr_effective_sample_fraction: float = 0.0
    policy_selected_nll: float = 0.0


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0
    best_metric: float = float("inf")
    patience_left: int = 5
    history: list[dict[str, Any]] = field(default_factory=list)


def _archetype_label(name: str) -> Optional[int]:
    """Return a valid auxiliary class, or ``None`` for unknown baseline ids."""
    ids = list(archetypes.archetype_ids())
    return ids.index(name) if name in ids else None


def is_allowed_missing_belief_head_key(key: str) -> bool:
    """True iff ``key`` is an expected new belief-aux head parameter."""
    return any(key.startswith(prefix) for prefix in BELIEF_AUX_HEAD_KEY_PREFIXES)


def belief_head_names_from_state_keys(keys: Sequence[str]) -> tuple[str, ...]:
    """Map state-dict keys → distinct head module names (sorted unique)."""
    names: set[str] = set()
    for key in keys:
        if is_allowed_missing_belief_head_key(key):
            names.add(key.split(".", 1)[0])
    return tuple(sorted(names))


def belief_card_vocab_from_state(state: dict[str, Any]) -> int:
    """Resolve the belief-card vocabulary for old or current checkpoints.

    Legacy policy checkpoints legitimately predate the opponent hand and
    hidden-remainder heads. In that case the live card vocabulary used by
    ``load_model_from_checkpoint`` is authoritative. A partially upgraded
    checkpoint may provide either head, but any present tensor must be a valid
    linear weight and the two output widths must agree.
    """

    widths: dict[str, int] = {}
    for name in ("opp_hand_head", "opp_remainder_head"):
        weight = state.get(f"{name}.weight")
        if weight is None:
            continue
        if getattr(weight, "ndim", 0) != 2:
            raise ValueError(f"checkpoint {name} weight must be rank-2")
        width = int(weight.shape[0])
        if width <= 0:
            raise ValueError(f"checkpoint {name} vocabulary must be positive")
        widths[name] = width
    if len(set(widths.values())) > 1:
        raise ValueError("checkpoint opponent belief head vocabularies disagree")
    return next(iter(widths.values()), int(features.card_vocab_size()))


def masked_belief_card_bce(
    logits: torch.Tensor,
    multilabel: Optional[torch.Tensor],
) -> torch.Tensor:
    """BCE with logits; returns a zero scalar when labels are absent (masked)."""
    if multilabel is None:
        return logits.sum() * 0.0
    if multilabel.shape != logits.shape:
        raise ValueError(
            "belief card multilabel shape mismatch: "
            f"logits={tuple(logits.shape)} labels={tuple(multilabel.shape)}"
        )
    return F.binary_cross_entropy_with_logits(logits, multilabel)


def _card_ids_from_aux_field(value: Any) -> Optional[list[int]]:
    if value is None:
        return None
    ids: list[int] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and item.get("id") is not None:
                ids.append(int(item["id"]))
            elif isinstance(item, int):
                ids.append(int(item))
            elif isinstance(item, list):
                nested = _card_ids_from_aux_field(item)
                if nested:
                    ids.extend(nested)
    elif isinstance(value, dict) and value.get("id") is not None:
        ids.append(int(value["id"]))
    return ids


def belief_multihots_from_aux_labels(
    aux_labels: dict[str, Any],
    card_vocab: int,
    *,
    device: torch.device,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Build hand / remainder multi-hots from privileged remask aux labels.

    Remainder = hand ∪ deck-order ∪ privileged prize dump when present.
    Either tensor may be ``None`` so the corresponding BCE term masks cleanly.
    """
    hand_ids = _card_ids_from_aux_field(aux_labels.get("opp_hand"))
    deck_ids = _card_ids_from_aux_field(aux_labels.get("opp_deck_order"))
    prize_ids = _card_ids_from_aux_field(aux_labels.get("opp_prizes"))
    exact_remainder_ids = _card_ids_from_aux_field(
        aux_labels.get("opp_hidden_remainder")
    )
    hand_mh: Optional[torch.Tensor] = None
    if hand_ids is not None:
        hand_mh = torch.zeros(card_vocab, device=device)
        for card_id in hand_ids:
            if 0 <= int(card_id) < card_vocab:
                hand_mh[int(card_id)] = 1.0
    rem_ids: Optional[list[int]] = None
    if exact_remainder_ids is not None:
        rem_ids = list(exact_remainder_ids)
    elif hand_ids is not None or deck_ids is not None or prize_ids is not None:
        rem_ids = []
        for src in (hand_ids, deck_ids, prize_ids):
            if src:
                rem_ids.extend(src)
    rem_mh: Optional[torch.Tensor] = None
    if rem_ids is not None:
        rem_mh = torch.zeros(card_vocab, device=device)
        for card_id in rem_ids:
            if 0 <= int(card_id) < card_vocab:
                rem_mh[int(card_id)] = 1.0
    return hand_mh, rem_mh


def masked_alakazam_guide_ce(
    model_log_probs: torch.Tensor,
    guide_target_indices: Sequence[int],
    guide_confidences: Sequence[float],
    n_options: Sequence[int],
) -> tuple[torch.Tensor, int]:
    """Confidence-weighted CE for collapsed advisory guide targets."""
    if model_log_probs.dim() != 2:
        raise ValueError("guide CE expects [rows, max_options] model log-probs")
    rows = int(model_log_probs.size(0))
    if not (
        len(guide_target_indices) == len(guide_confidences) == len(n_options) == rows
    ):
        raise ValueError("guide rows do not align with policy rows")

    losses: list[torch.Tensor] = []
    for row, (raw_target, raw_confidence, raw_n) in enumerate(
        zip(guide_target_indices, guide_confidences, n_options)
    ):
        n = int(raw_n)
        target = int(raw_target)
        confidence = float(raw_confidence)
        if target < 0 or confidence <= 0.0:
            continue
        if n < 2 or n > int(model_log_probs.size(1)):
            continue
        if target >= n:
            raise ValueError(
                f"guide target is outside row {row}: target={target} options={n}"
            )
        bounded_confidence = min(1.0, max(0.0, confidence))
        losses.append(-model_log_probs[row, target] * bounded_confidence)

    if not losses:
        return model_log_probs.sum() * 0.0, 0
    return torch.stack(losses).mean(), len(losses)


def count_usable_alakazam_guide_rows(
    sequences: Sequence[GameSequence],
) -> int:
    """Count non-flat, comparable guide rows without running the model."""
    usable = 0
    for game in sequences:
        for decision in game.decisions:
            for stage in decision.policy_stages:
                target = int(getattr(stage, "guide_target_index", -1))
                confidence = float(getattr(stage, "guide_confidence", 0.0))
                n = int(stage.options.num_words)
                if target >= n:
                    raise ValueError(
                        "Alakazam guide target is not aligned to policy options"
                    )
                if n >= 2 and target >= 0 and confidence > 0.0:
                    usable += 1
    return usable


def sequence_losses(
    model: TemporalCabtTransformer,
    seq: GameSequence,
    *,
    value_weight: float = 1.0,
    aux_weight: float = 0.1,
    opp_hand_weight: float = 0.2,
    opp_remainder_weight: float = 0.15,
    alakazam_guide_weight: float = 0.0,
    lethal_threat_weight: float = 0.0,
    prize_race_weight: float = 0.0,
    pure_rl: bool = False,
    awr_beta: float = 0.5,
    awr_weight_max: float = 20.0,
) -> tuple[torch.Tensor, BatchMetrics]:
    """Causal history-conditioned losses for one :class:`GameSequence`."""
    return batch_losses(
        model,
        [seq],
        value_weight=value_weight,
        aux_weight=aux_weight,
        opp_hand_weight=opp_hand_weight,
        opp_remainder_weight=opp_remainder_weight,
        alakazam_guide_weight=alakazam_guide_weight,
        lethal_threat_weight=lethal_threat_weight,
        prize_race_weight=prize_race_weight,
        pure_rl=pure_rl,
        awr_beta=awr_beta,
        awr_weight_max=awr_weight_max,
    )


def batch_losses(
    model: TemporalCabtTransformer,
    seqs: Sequence[GameSequence],
    *,
    value_weight: float = 1.0,
    aux_weight: float = 0.1,
    opp_hand_weight: float = 0.2,
    opp_remainder_weight: float = 0.15,
    alakazam_guide_weight: float = 0.0,
    lethal_threat_weight: float = 0.0,
    prize_race_weight: float = 0.0,
    opp_hand_multihot: Optional[torch.Tensor] = None,
    opp_remainder_multihot: Optional[torch.Tensor] = None,
    pure_rl: bool = False,
    awr_beta: float = 0.5,
    awr_weight_max: float = 20.0,
    awr_normalize_advantages: bool = True,
    entropy_bonus: float = 0.0,
    awr_baseline_cache: Optional[dict[tuple[int, int, int], float]] = None,
    awr_capture_baseline: Optional[dict[tuple[int, int, int], float]] = None,
    prediction_sink: Optional[list[int]] = None,
    history_identity_weight: float = 0.0,
) -> tuple[torch.Tensor, BatchMetrics]:
    """Causal history forward over all valid decisions.

    Spatial boards are batched, then each game's temporal states are computed
    with a causal mask. The state used for decision ``t`` can see only
    observations ``<= t`` and is parity-tested against incremental KV serving.

    Belief card multilabel losses are attached with zero/masked defaults so
    late head add does not break the training loop when labels are absent.

    Alakazam guide scores are collapsed during featurization to a unique-best
    index and bounded confidence, then distilled with masked CE. The default
    weight is zero, so core and older feature shards preserve exact behavior.

    Scope B (``lethal_threat`` / ``prize_race``) losses are masked when labels
    are absent and default-weight 0 so core / generic trains ignore them.

    When ``pure_rl=True``, soft behavior policy targets are rejected and the
    policy term is advantage-weighted regression (AWR) on the selected action
    index only — never CE toward cloned ``history_policy`` soft targets.
    AWR can consume a value cache keyed by ``(id(game), decision, stage)``.
    ``rl_train_step`` precomputes that cache once per pure-RL iteration, making
    the baseline genuinely stale/frozen. Direct callers without a cache retain
    the legacy detached-online behavior. Optional per-batch whitening and a
    small entropy bonus stabilize high-SPS pure RL.
    """
    if float(alakazam_guide_weight) < 0.0:
        raise ValueError("Alakazam guide loss weight cannot be negative")
    device = next(model.parameters()).device
    games = [s for s in seqs if s.decisions]
    if not games:
        return torch.zeros((), device=device, requires_grad=True), BatchMetrics()

    all_boards = [d.board for g in games for d in g.decisions]
    spatial_all = model.encode_board(all_boards)
    valid_spatial: list[torch.Tensor] = []
    valid_states: list[torch.Tensor] = []
    valid_identity_states: list[torch.Tensor] = []
    valid_options = []
    valid_n: list[int] = []
    soft_targets: list[Optional[list[float]]] = []
    hard_idx: list[int] = []
    value_targets: list[float] = []
    awr_baseline_keys: list[tuple[int, int, int]] = []
    aux_rows: list[int] = []
    aux_labels: list[int] = []
    decision_aux: list[dict[str, Any]] = []
    guide_target_rows: list[int] = []
    guide_confidence_rows: list[float] = []
    use_alakazam_guide = float(alakazam_guide_weight) > 0.0
    spatial_offset = 0
    for g in games:
        val = float(g.value)
        pt = g.policy_targets
        factorized_pt = g.factorized_policy_targets
        length = len(g.decisions)
        game_spatial = spatial_all[spatial_offset : spatial_offset + length]
        spatial_offset += length
        if model.decision_context == "history":
            previous_actions = [None] + [
                decision.action_token for decision in g.decisions[:-1]
            ]
            cls = model.history_tokens(
                game_spatial, previous_actions
            ).unsqueeze(0)
            game_states, _ = model.temporal_encode(
                cls, append=False, return_all=True
            )
            game_states = game_states.squeeze(0)
            game_identity_states = model.temporal_norm(
                model.pool_cls(game_spatial)
            ).detach()
        else:
            cls = model.pool_cls(game_spatial).unsqueeze(1)
            game_states, _ = model.temporal_encode(
                cls, append=False, return_all=True
            )
            game_states = game_states.squeeze(1)
            game_identity_states = game_states.detach()
        last_valid_row: Optional[int] = None
        for t, d in enumerate(g.decisions):
            stages = d.policy_stages or [
                PolicyStage(
                    options=d.options,
                    action_combos=d.action_combos,
                    target_index=d.action_combo_index,
                )
            ]
            target_stages = (
                factorized_pt[t]
                if factorized_pt is not None
                and t < len(factorized_pt)
                and factorized_pt[t] is not None
                else None
            )
            for stage_i, stage in enumerate(stages):
                n_opt = stage.options.num_words
                if n_opt <= 0:
                    continue
                soft = None
                if target_stages is not None and stage_i < len(target_stages):
                    row = dict(target_stages[stage_i] or {})
                    recorded_combos = [
                        list(combo) for combo in (row.get("action_combos") or [])
                    ]
                    if recorded_combos and recorded_combos != stage.action_combos:
                        raise ValueError(
                            "factorized target/action candidate ordering mismatch"
                        )
                    cand = list(row.get("policy") or [])
                    if len(cand) != n_opt or sum(cand) <= 0:
                        raise ValueError("invalid factorized soft policy target")
                    soft = [float(x) for x in cand]
                    idx = int(row.get("selected_index", stage.target_index))
                elif (
                    not d.policy_stages
                    and pt is not None
                    and t < len(pt)
                    and pt[t] is not None
                ):
                    cand = list(pt[t][:n_opt])
                    if len(cand) != n_opt or sum(cand) <= 0:
                        continue
                    soft = cand
                    idx = int(max(range(n_opt), key=lambda j: cand[j]))
                else:
                    idx = int(stage.target_index)
                if idx < 0 or idx >= n_opt:
                    continue
                valid_spatial.append(game_spatial[t])
                valid_states.append(game_states[t])
                valid_identity_states.append(game_identity_states[t])
                valid_options.append(stage.options)
                valid_n.append(n_opt)
                soft_targets.append(soft)
                hard_idx.append(idx)
                value_targets.append(val)
                awr_baseline_keys.append((id(g), t, stage_i))
                decision_aux.append(dict(d.aux_labels or {}))
                if use_alakazam_guide:
                    guide_target_rows.append(
                        int(getattr(stage, "guide_target_index", -1))
                    )
                    guide_confidence_rows.append(
                        float(getattr(stage, "guide_confidence", 0.0))
                    )
                last_valid_row = len(valid_options) - 1
        label = _archetype_label(g.opp_archetype)
        if last_valid_row is not None and label is not None:
            aux_rows.append(last_valid_row)
            aux_labels.append(label)

    if not valid_options:
        return (
            torch.zeros((), device=device, requires_grad=True),
            BatchMetrics(n_games=len(games)),
        )

    state_all = torch.stack(valid_states, dim=0)
    identity_state_all = torch.stack(valid_identity_states, dim=0)
    current_spatial = torch.stack(valid_spatial, dim=0)
    logits_all = model.decode_options(
        valid_options,
        current_spatial,
        state_all,
        n_options=valid_n,
    )
    value_pred = torch.tanh(model.value_head(state_all)).squeeze(-1)
    belief = model.belief_aux_logits(state_all)
    aux_logits_all = belief["aux_logits"]
    opp_hand_logits_all = belief["opp_hand_logits"]
    opp_remainder_logits_all = belief["opp_remainder_logits"]
    lethal_logits_all = belief["lethal_threat_logits"]
    prize_race_pred_all = belief["prize_race_pred"]
    k = logits_all.size(0)
    max_n = logits_all.size(1)

    target_idx = torch.tensor(hard_idx, device=device, dtype=torch.long)
    v_target = torch.tensor(value_targets, device=device, dtype=value_pred.dtype)
    log_p = torch.nan_to_num(F.log_softmax(logits_all, dim=-1), neginf=0.0)
    selected_log_p = log_p[torch.arange(k, device=device), target_idx]
    policy_selected_nll = float((-selected_log_p.detach()).mean().item())

    if use_alakazam_guide:
        guide_loss, n_guide_rows = masked_alakazam_guide_ce(
            log_p,
            guide_target_rows,
            guide_confidence_rows,
            valid_n,
        )
    else:
        guide_loss, n_guide_rows = log_p.sum() * 0.0, 0

    awr_mean_adv = 0.0
    raw_adv_mean = 0.0
    raw_adv_std = 0.0
    raw_adv_mean_abs = 0.0
    raw_adv_mean_sq = 0.0
    norm_adv_mean = 0.0
    norm_adv_std = 0.0
    norm_adv_mean_abs = 0.0
    norm_adv_mean_sq = 0.0
    awr_w_mean = 0.0
    awr_w_sum = 0.0
    awr_w_sq_sum = 0.0
    awr_w_p50 = 0.0
    awr_w_p95 = 0.0
    awr_clip_frac = 0.0
    awr_ess = 0.0
    awr_ess_frac = 0.0

    if pure_rl:
        if any(soft is not None for soft in soft_targets):
            raise ValueError(
                "PURE_RL=1 forbids soft factorized_policy_targets as CE/"
                "behavior-clone training targets; store selected_index only"
            )
        if awr_capture_baseline is not None:
            awr_capture_baseline.update(
                {
                    key: float(value)
                    for key, value in zip(
                        awr_baseline_keys,
                        value_pred.detach().float().cpu().tolist(),
                    )
                }
            )
        if awr_baseline_cache is None:
            baseline_pred = value_pred.detach()
        else:
            missing = [key for key in awr_baseline_keys if key not in awr_baseline_cache]
            if missing:
                raise KeyError(
                    "frozen AWR baseline cache is missing "
                    f"{len(missing)} decision-stage row(s)"
                )
            baseline_pred = torch.tensor(
                [awr_baseline_cache[key] for key in awr_baseline_keys],
                device=device,
                dtype=value_pred.dtype,
            )

        # Capture raw diagnostics before whitening. The compatibility
        # ``mean_advantage`` signal intentionally uses mean(abs(A)) so symmetric
        # positive/negative learning signal cannot look dead by cancellation.
        raw_advantages = v_target - baseline_pred.detach()
        raw_stats = raw_advantages.detach().float()
        raw_adv_mean = float(raw_stats.mean().item())
        raw_adv_std = float(raw_stats.std(unbiased=False).item())
        raw_adv_mean_abs = float(raw_stats.abs().mean().item())
        raw_adv_mean_sq = float(raw_stats.square().mean().item())
        awr_mean_adv = raw_adv_mean_abs

        advantages = raw_advantages
        if awr_normalize_advantages and k > 1:
            adv_std = advantages.std(unbiased=False).clamp_min(1e-6)
            advantages = (advantages - advantages.mean()) / adv_std
        norm_stats = advantages.detach().float()
        norm_adv_mean = float(norm_stats.mean().item())
        norm_adv_std = float(norm_stats.std(unbiased=False).item())
        norm_adv_mean_abs = float(norm_stats.abs().mean().item())
        norm_adv_mean_sq = float(norm_stats.square().mean().item())
        beta = max(float(awr_beta), 1e-6)
        wmax = max(float(awr_weight_max), 1e-6)
        raw_w = torch.exp(advantages / beta)
        weights = torch.clamp(raw_w, max=wmax)
        p_loss = -(weights.detach() * selected_log_p).mean()
        if float(entropy_bonus) > 0.0:
            probs = torch.nan_to_num(log_p.exp(), nan=0.0)
            ent = -(probs * log_p).sum(dim=-1).mean()
            p_loss = p_loss - float(entropy_bonus) * ent
        policy_kl = torch.zeros((), device=device)
        awr_w_mean = float(weights.detach().float().mean().item())
        awr_w_sum = float(weights.detach().float().sum().item())
        awr_w_sq_sum = float(weights.detach().float().square().sum().item())
        awr_ess = min(
            float(k),
            (awr_w_sum * awr_w_sum) / max(awr_w_sq_sum, 1e-12),
        )
        awr_ess_frac = awr_ess / max(k, 1)
        sorted_w, _ = torch.sort(weights.detach().float())
        awr_w_p50 = float(sorted_w[int(0.50 * (k - 1))].item())
        awr_w_p95 = float(sorted_w[int(0.95 * (k - 1))].item())
        awr_clip_frac = float((raw_w >= wmax).float().mean().item())
    else:
        target_mat = torch.zeros(k, max_n, device=device, dtype=logits_all.dtype)
        target_mat[torch.arange(k, device=device), target_idx] = 1.0
        for r, soft in enumerate(soft_targets):
            if soft is None:
                continue
            row = torch.tensor(soft, device=device, dtype=logits_all.dtype)
            target_mat[r].zero_()
            target_mat[r, : row.numel()] = row / row.sum().clamp_min(1e-8)

        p_loss = -(target_mat * log_p).sum(dim=1).mean()
        target_log = torch.where(
            target_mat > 0,
            target_mat.clamp_min(1e-12).log(),
            torch.zeros_like(target_mat),
        )
        target_entropy = -(target_mat * target_log).sum(dim=1).mean()
        policy_kl = (p_loss - target_entropy).clamp_min(0.0)

    v_loss = F.smooth_l1_loss(value_pred, v_target)
    history_identity_loss = (
        F.mse_loss(state_all, identity_state_all)
        if float(history_identity_weight) > 0.0
        else state_all.sum() * 0.0
    )

    aux_loss = torch.zeros((), device=device)
    if aux_weight > 0 and aux_rows:
        row_idx = torch.tensor(aux_rows, device=device, dtype=torch.long)
        aux_logits = aux_logits_all.index_select(0, row_idx)
        labels = torch.tensor(
            aux_labels,
            device=device,
            dtype=torch.long,
        )
        if int(labels.max().item()) >= aux_logits.size(-1):
            raise ValueError(
                "checkpoint auxiliary head is incompatible with registered "
                f"archetype labels ({aux_logits.size(-1)} classes)"
            )
        aux_loss = F.cross_entropy(aux_logits, labels)

    # Masked card-head BCE from privileged aux labels (or explicit override).
    card_vocab = int(getattr(model, "belief_card_vocab", opp_hand_logits_all.size(-1)))
    n_hand_rows = 0
    n_remainder_rows = 0
    if opp_hand_multihot is None and opp_remainder_multihot is None:
        hand_rows: list[torch.Tensor] = []
        rem_rows: list[torch.Tensor] = []
        hand_idx: list[int] = []
        rem_idx: list[int] = []
        for i, aux in enumerate(decision_aux):
            hand_mh, rem_mh = belief_multihots_from_aux_labels(
                aux, card_vocab, device=device
            )
            if hand_mh is not None:
                hand_rows.append(hand_mh)
                hand_idx.append(i)
            if rem_mh is not None:
                rem_rows.append(rem_mh)
                rem_idx.append(i)
        if hand_rows:
            n_hand_rows = len(hand_rows)
            opp_hand_loss = F.binary_cross_entropy_with_logits(
                opp_hand_logits_all.index_select(
                    0, torch.tensor(hand_idx, device=device, dtype=torch.long)
                ),
                torch.stack(hand_rows, dim=0),
            )
        else:
            opp_hand_loss = masked_belief_card_bce(opp_hand_logits_all, None)
        if rem_rows:
            n_remainder_rows = len(rem_rows)
            opp_remainder_loss = F.binary_cross_entropy_with_logits(
                opp_remainder_logits_all.index_select(
                    0, torch.tensor(rem_idx, device=device, dtype=torch.long)
                ),
                torch.stack(rem_rows, dim=0),
            )
        else:
            opp_remainder_loss = masked_belief_card_bce(
                opp_remainder_logits_all, None
            )
    else:
        hand_labels = opp_hand_multihot
        rem_labels = opp_remainder_multihot
        if hand_labels is not None and hand_labels.dim() == 1:
            hand_labels = hand_labels.unsqueeze(0).expand(k, -1)
        if rem_labels is not None and rem_labels.dim() == 1:
            rem_labels = rem_labels.unsqueeze(0).expand(k, -1)
        opp_hand_loss = masked_belief_card_bce(opp_hand_logits_all, hand_labels)
        opp_remainder_loss = masked_belief_card_bce(
            opp_remainder_logits_all, rem_labels
        )
        n_hand_rows = k if hand_labels is not None else 0
        n_remainder_rows = k if rem_labels is not None else 0

    # Scope B — masked when labels absent; weights default 0 on core/generic.
    lethal_rows: list[float] = []
    lethal_idx: list[int] = []
    race_rows: list[torch.Tensor] = []
    race_idx: list[int] = []
    for i, aux in enumerate(decision_aux):
        lethal = lethal_target_from_aux(aux)
        if lethal is not None:
            lethal_rows.append(float(lethal))
            lethal_idx.append(i)
        race = prize_race_target_from_aux(aux, device=device)
        if race is not None:
            race_rows.append(race)
            race_idx.append(i)
    if lethal_rows:
        lethal_threat_loss = F.binary_cross_entropy_with_logits(
            lethal_logits_all.index_select(
                0, torch.tensor(lethal_idx, device=device, dtype=torch.long)
            ),
            torch.tensor(lethal_rows, device=device, dtype=lethal_logits_all.dtype),
        )
    else:
        lethal_threat_loss = masked_bce_logit(lethal_logits_all, None)
    if race_rows:
        prize_race_loss = F.smooth_l1_loss(
            prize_race_pred_all.index_select(
                0, torch.tensor(race_idx, device=device, dtype=torch.long)
            ),
            torch.stack(race_rows, dim=0).to(dtype=prize_race_pred_all.dtype),
        )
    else:
        prize_race_loss = masked_smooth_l1(prize_race_pred_all, None)

    total = (
        p_loss
        + value_weight * v_loss
        + aux_weight * aux_loss
        + float(alakazam_guide_weight) * guide_loss
        + float(opp_hand_weight) * opp_hand_loss
        + float(opp_remainder_weight) * opp_remainder_loss
        + float(lethal_threat_weight) * lethal_threat_loss
        + float(prize_race_weight) * prize_race_loss
        + float(history_identity_weight) * history_identity_loss
    )
    preds = logits_all.argmax(dim=1)
    if prediction_sink is not None:
        prediction_sink.extend(int(x) for x in preds.detach().cpu().tolist())
    correct = int((preds == target_idx).sum().item())
    metrics = BatchMetrics(
        policy_loss=float(p_loss.detach().item()),
        value_loss=float(v_loss.detach().item()),
        aux_loss=float(aux_loss.detach().item()),
        alakazam_guide_loss=float(guide_loss.detach().item()),
        opp_hand_loss=float(opp_hand_loss.detach().item()),
        opp_remainder_loss=float(opp_remainder_loss.detach().item()),
        lethal_threat_loss=float(lethal_threat_loss.detach().item()),
        prize_race_loss=float(prize_race_loss.detach().item()),
        history_identity_loss=float(history_identity_loss.detach().item()),
        total_loss=float(total.detach().item()),
        policy_acc=correct / max(k, 1),
        policy_kl=float(policy_kl.detach().item()) if torch.is_tensor(policy_kl) else float(policy_kl),
        target_value_mean=float(v_target.detach().float().mean().item()),
        value_pred_mean=float(value_pred.detach().float().mean().item()),
        n_decisions=k,
        n_games=len(games),
        n_alakazam_guide_rows=int(n_guide_rows),
        n_archetype_rows=len(aux_rows),
        n_opp_hand_rows=int(n_hand_rows),
        n_opp_remainder_rows=int(n_remainder_rows),
        n_lethal_threat_rows=len(lethal_rows),
        n_prize_race_rows=len(race_rows),
        mean_advantage=awr_mean_adv,
        raw_advantage_mean=raw_adv_mean,
        raw_advantage_std=raw_adv_std,
        raw_advantage_mean_abs=raw_adv_mean_abs,
        raw_advantage_mean_sq=raw_adv_mean_sq,
        normalized_advantage_mean=norm_adv_mean,
        normalized_advantage_std=norm_adv_std,
        normalized_advantage_mean_abs=norm_adv_mean_abs,
        normalized_advantage_mean_sq=norm_adv_mean_sq,
        awr_weight_mean=awr_w_mean,
        awr_weight_sum=awr_w_sum,
        awr_weight_sq_sum=awr_w_sq_sum,
        awr_weight_p50=awr_w_p50,
        awr_weight_p95=awr_w_p95,
        awr_weight_clip_frac=awr_clip_frac,
        awr_effective_sample_size=awr_ess,
        awr_effective_sample_fraction=awr_ess_frac,
        policy_selected_nll=policy_selected_nll,
    )
    return total, metrics


def _resident_hard_target_objective(
    model: TemporalCabtTransformer,
    logits: torch.Tensor,
    state: torch.Tensor,
    target_idx: torch.Tensor,
    v_target: torch.Tensor,
    *,
    value_weight: float,
    n_games: int = 0,
) -> tuple[torch.Tensor, BatchMetrics]:
    """Shared supervised objective for stateless and temporal resident paths."""
    k = int(target_idx.numel())
    if k <= 0:
        device = next(model.parameters()).device
        return torch.zeros((), device=device, requires_grad=True), BatchMetrics()
    value_pred = torch.tanh(model.value_head(state)).squeeze(-1)
    v_target = v_target.to(dtype=value_pred.dtype)
    log_p = torch.nan_to_num(F.log_softmax(logits, dim=-1), neginf=0.0)
    selected_log_p = log_p[
        torch.arange(k, device=logits.device), target_idx
    ]
    p_loss = -selected_log_p.mean()
    v_loss = F.smooth_l1_loss(value_pred, v_target)
    total = p_loss + float(value_weight) * v_loss
    preds = logits.argmax(dim=1)
    correct = int((preds == target_idx).sum().item())
    p_scalar = float(p_loss.detach().item())
    metrics = BatchMetrics(
        policy_loss=p_scalar,
        value_loss=float(v_loss.detach().item()),
        total_loss=float(total.detach().item()),
        policy_acc=correct / max(k, 1),
        policy_kl=max(p_scalar, 0.0),
        target_value_mean=float(v_target.detach().float().mean().item()),
        value_pred_mean=float(value_pred.detach().float().mean().item()),
        n_decisions=k,
        n_games=int(n_games),
        policy_selected_nll=p_scalar,
    )
    return total, metrics


def device_batch_losses(
    model: TemporalCabtTransformer,
    corpus: DeviceResidentBootstrapCorpus,
    sample_ids: torch.Tensor,
    *,
    value_weight: float = 1.0,
) -> tuple[torch.Tensor, BatchMetrics]:
    """Hard-target stateless loss with every input already on the device.

    This is deliberately narrow: the latest-ladder hot start has hard selected
    actions and all auxiliary heads disabled.  Keeping that contract explicit
    prevents the fast path from silently discarding a future target type.
    """
    if model.decision_context != "stateless":
        raise ValueError("device-resident bootstrap requires stateless context")
    board, options, counts, target_idx, v_target = corpus.batch(sample_ids)
    k = int(target_idx.numel())
    if k <= 0:
        device = next(model.parameters()).device
        return torch.zeros((), device=device, requires_grad=True), BatchMetrics()

    spatial = model.encode_board_packed(board, batch_size=k)
    cls = model.pool_cls(spatial).unsqueeze(1)
    states, _ = model.temporal_encode(cls, append=False, return_all=True)
    state = states.squeeze(1)
    logits = model.decode_options_packed(
        options,
        spatial,
        state,
        n_options=counts,
        batch_size=k,
    )
    return _resident_hard_target_objective(
        model,
        logits,
        state,
        target_idx,
        v_target,
        value_weight=value_weight,
    )


def device_temporal_batch_losses(
    model: TemporalCabtTransformer,
    corpus: DeviceResidentBootstrapCorpus,
    game_ids: torch.Tensor,
    *,
    value_weight: float = 1.0,
    aux_weight: float = 0.0,
    opp_hand_weight: float = 0.0,
    opp_remainder_weight: float = 0.0,
    lethal_threat_weight: float = 0.0,
    prize_race_weight: float = 0.0,
    alakazam_guide_weight: float = 0.0,
) -> tuple[torch.Tensor, BatchMetrics]:
    """Hard-target full-game loss with every resident temporal target.

    Policy/value behavior is the same as :func:`_resident_hard_target_objective`.
    Optional auxiliary targets are trained only on rows whose packed presence
    masks are valid.  In particular, an absent exact opponent hand is not
    interpreted as an observed empty hand.
    """
    if model.decision_context != "history":
        raise ValueError("temporal resident loss requires history context")
    for name, weight in (
        ("aux", aux_weight),
        ("opp_hand", opp_hand_weight),
        ("opp_remainder", opp_remainder_weight),
        ("lethal_threat", lethal_threat_weight),
        ("prize_race", prize_race_weight),
        ("alakazam_guide", alakazam_guide_weight),
    ):
        if float(weight) < 0.0:
            raise ValueError(f"resident temporal {name} weight cannot be negative")
    (
        board,
        previous_actions,
        options,
        counts,
        target_idx,
        v_target,
        game_lengths,
        sample_state_rows,
    ) = corpus.temporal_batch(game_ids)
    decisions = int(game_lengths.sum().item())
    samples = int(target_idx.numel())
    if decisions <= 0 or samples <= 0:
        device = next(model.parameters()).device
        return torch.zeros((), device=device, requires_grad=True), BatchMetrics()
    if int(game_lengths.max().item()) > int(model.max_context):
        raise ValueError(
            "resident temporal game exceeds checkpoint context: "
            f"game={int(game_lengths.max().item())} max={int(model.max_context)}"
        )

    spatial = model.encode_board_packed(board, batch_size=decisions)
    action_state = model.encode_previous_actions_packed(
        previous_actions, batch_size=decisions
    )
    cls = model.pool_cls(spatial) + float(model.cfg.history_action_scale) * action_state

    # Games of equal length can share one temporal call: the batch dimension is
    # isolated by attention, while avoiding one Python/model launch per game.
    lengths = [int(value) for value in game_lengths.cpu().tolist()]
    starts: list[int] = []
    cursor = 0
    by_length: dict[int, list[int]] = {}
    for length in lengths:
        starts.append(cursor)
        by_length.setdefault(length, []).append(cursor)
        cursor += length
    state_parts: list[torch.Tensor] = []
    row_parts: list[torch.Tensor] = []
    for length, group_starts in by_length.items():
        tokens = torch.stack(
            [cls[start : start + length] for start in group_starts], dim=0
        )
        encoded, _ = model.temporal_encode(
            tokens, append=False, return_all=True
        )
        state_parts.append(encoded.reshape(-1, model.d_model))
        row_parts.append(
            torch.cat(
                [
                    torch.arange(
                        start,
                        start + length,
                        device=corpus.device,
                        dtype=torch.long,
                    )
                    for start in group_starts
                ]
            )
        )
    grouped_states = torch.cat(state_parts, dim=0)
    grouped_rows = torch.cat(row_parts, dim=0)
    state_by_decision = grouped_states.index_select(
        0, torch.argsort(grouped_rows)
    )
    state = state_by_decision.index_select(0, sample_state_rows)
    sample_spatial = spatial.index_select(0, sample_state_rows)
    logits = model.decode_options_packed(
        options,
        sample_spatial,
        state,
        n_options=counts,
        batch_size=samples,
    )
    total, metrics = _resident_hard_target_objective(
        model,
        logits,
        state,
        target_idx,
        v_target,
        value_weight=value_weight,
        n_games=int(game_ids.numel()),
    )
    if not any(
        float(weight) > 0.0
        for weight in (
            aux_weight,
            opp_hand_weight,
            opp_remainder_weight,
            lethal_threat_weight,
            prize_race_weight,
            alakazam_guide_weight,
        )
    ):
        # Retain the historical policy/value-only path exactly, including RNG
        # consumption (``aux_head`` contains dropout on nonzero-dropout models).
        return total, metrics

    # Recreate the packed sample ids in the same caller-specified game order
    # used by ``temporal_batch``.  Exact targets are decision-indexed while
    # archetype and guide targets are sample-indexed, so both mappings matter.
    assert corpus.game_sample_offset is not None
    resident_game_ids = game_ids.reshape(-1).to(
        device=corpus.device, dtype=torch.long
    )
    sample_starts = corpus.game_sample_offset.index_select(
        0, resident_game_ids
    ).to(dtype=torch.long)
    sample_ends = corpus.game_sample_offset.index_select(
        0, resident_game_ids + 1
    ).to(dtype=torch.long)
    sample_ids, _sample_lengths = corpus._expand_ranges(  # noqa: SLF001
        sample_starts, sample_ends
    )
    if int(sample_ids.numel()) != samples:
        raise AssertionError(
            "resident temporal sample target mapping disagrees with policy rows"
        )
    board_ids = corpus.sample_board.index_select(0, sample_ids).to(
        dtype=torch.long
    )

    belief = model.belief_aux_logits(state)
    aux_loss = belief["aux_logits"].sum() * 0.0
    opp_hand_loss = belief["opp_hand_logits"].sum() * 0.0
    opp_remainder_loss = belief["opp_remainder_logits"].sum() * 0.0
    lethal_threat_loss = belief["lethal_threat_logits"].sum() * 0.0
    prize_race_loss = belief["prize_race_pred"].sum() * 0.0
    guide_log_p = torch.nan_to_num(
        F.log_softmax(logits, dim=-1), neginf=0.0
    )
    guide_loss = guide_log_p.sum() * 0.0

    n_archetype_rows = 0
    if corpus.sample_aux_class is not None:
        aux_target = corpus.sample_aux_class.index_select(0, sample_ids).to(
            dtype=torch.long
        )
        aux_mask = aux_target >= 0
        if bool(aux_mask.any()):
            aux_loss = F.cross_entropy(
                belief["aux_logits"][aux_mask], aux_target[aux_mask]
            )
            n_archetype_rows = int(aux_mask.sum().item())

    n_opp_hand_rows = 0
    hand_fields = (
        corpus.hand_index,
        corpus.hand_offset,
        corpus.hand_present,
    )
    if int(corpus.belief_card_vocab) > 0 and all(
        value is not None for value in hand_fields
    ):
        assert corpus.hand_index is not None
        assert corpus.hand_offset is not None
        assert corpus.hand_present is not None
        hand_target, hand_mask = corpus._card_multihot(  # noqa: SLF001
            board_ids,
            index=corpus.hand_index,
            offset=corpus.hand_offset,
            present=corpus.hand_present,
        )
        if bool(hand_mask.any()):
            opp_hand_loss = F.binary_cross_entropy_with_logits(
                belief["opp_hand_logits"][hand_mask], hand_target[hand_mask]
            )
            n_opp_hand_rows = int(hand_mask.sum().item())

    n_opp_remainder_rows = 0
    remainder_fields = (
        corpus.remainder_index,
        corpus.remainder_offset,
        corpus.remainder_present,
    )
    if int(corpus.belief_card_vocab) > 0 and all(
        value is not None for value in remainder_fields
    ):
        assert corpus.remainder_index is not None
        assert corpus.remainder_offset is not None
        assert corpus.remainder_present is not None
        remainder_target, remainder_mask = corpus._card_multihot(  # noqa: SLF001
            board_ids,
            index=corpus.remainder_index,
            offset=corpus.remainder_offset,
            present=corpus.remainder_present,
        )
        if bool(remainder_mask.any()):
            opp_remainder_loss = F.binary_cross_entropy_with_logits(
                belief["opp_remainder_logits"][remainder_mask],
                remainder_target[remainder_mask],
            )
            n_opp_remainder_rows = int(remainder_mask.sum().item())

    n_lethal_threat_rows = 0
    if corpus.lethal_target is not None:
        lethal_target = corpus.lethal_target.index_select(0, board_ids).to(
            dtype=belief["lethal_threat_logits"].dtype
        )
        lethal_mask = torch.isfinite(lethal_target)
        if bool(lethal_mask.any()):
            lethal_threat_loss = F.binary_cross_entropy_with_logits(
                belief["lethal_threat_logits"][lethal_mask],
                lethal_target[lethal_mask],
            )
            n_lethal_threat_rows = int(lethal_mask.sum().item())

    n_prize_race_rows = 0
    if corpus.prize_race_target is not None:
        prize_race_target = corpus.prize_race_target.index_select(
            0, board_ids
        ).to(dtype=belief["prize_race_pred"].dtype)
        prize_race_mask = torch.isfinite(prize_race_target).all(dim=1)
        if bool(prize_race_mask.any()):
            prize_race_loss = F.smooth_l1_loss(
                belief["prize_race_pred"][prize_race_mask],
                prize_race_target[prize_race_mask],
            )
            n_prize_race_rows = int(prize_race_mask.sum().item())

    n_guide_rows = 0
    if (
        corpus.guide_target_index is not None
        and corpus.guide_confidence is not None
    ):
        guide_target = corpus.guide_target_index.index_select(
            0, sample_ids
        ).to(dtype=torch.long)
        guide_confidence = corpus.guide_confidence.index_select(
            0, sample_ids
        ).to(dtype=guide_log_p.dtype)
        invalid_guide = (guide_target >= counts) & (guide_target >= 0)
        if bool(invalid_guide.any()):
            bad_row = int(torch.nonzero(invalid_guide, as_tuple=False)[0].item())
            raise ValueError(
                "resident Alakazam guide target is outside option row: "
                f"target={int(guide_target[bad_row].item())} "
                f"options={int(counts[bad_row].item())}"
            )
        guide_mask = (
            (guide_target >= 0)
            & (guide_confidence > 0.0)
            & (counts >= 2)
        )
        if bool(guide_mask.any()):
            guide_rows = torch.nonzero(guide_mask, as_tuple=False).flatten()
            bounded_confidence = guide_confidence.index_select(
                0, guide_rows
            ).clamp(0.0, 1.0)
            guide_loss = -(
                guide_log_p[
                    guide_rows,
                    guide_target.index_select(0, guide_rows),
                ]
                * bounded_confidence
            ).mean()
            n_guide_rows = int(guide_rows.numel())

    total = (
        total
        + float(aux_weight) * aux_loss
        + float(opp_hand_weight) * opp_hand_loss
        + float(opp_remainder_weight) * opp_remainder_loss
        + float(lethal_threat_weight) * lethal_threat_loss
        + float(prize_race_weight) * prize_race_loss
        + float(alakazam_guide_weight) * guide_loss
    )
    metrics.aux_loss = float(aux_loss.detach().item())
    metrics.opp_hand_loss = float(opp_hand_loss.detach().item())
    metrics.opp_remainder_loss = float(opp_remainder_loss.detach().item())
    metrics.lethal_threat_loss = float(lethal_threat_loss.detach().item())
    metrics.prize_race_loss = float(prize_race_loss.detach().item())
    metrics.alakazam_guide_loss = float(guide_loss.detach().item())
    metrics.total_loss = float(total.detach().item())
    metrics.n_archetype_rows = n_archetype_rows
    metrics.n_opp_hand_rows = n_opp_hand_rows
    metrics.n_opp_remainder_rows = n_opp_remainder_rows
    metrics.n_lethal_threat_rows = n_lethal_threat_rows
    metrics.n_prize_race_rows = n_prize_race_rows
    metrics.n_alakazam_guide_rows = n_guide_rows
    return total, metrics


def device_exact_value_predictions(
    model: TemporalCabtTransformer,
    corpus: DeviceResidentBootstrapCorpus,
    sample_ids: torch.Tensor,
) -> torch.Tensor:
    """Value predictions for an exact resident batch without option gathers."""
    if model.decision_context != "stateless":
        raise ValueError("device-resident exact replay requires stateless context")
    board, _board_ids = corpus.board_batch(sample_ids)
    k = int(sample_ids.numel())
    spatial = model.encode_board_packed(board, batch_size=k)
    cls = model.pool_cls(spatial).unsqueeze(1)
    states, _ = model.temporal_encode(cls, append=False, return_all=True)
    state = states.squeeze(1)
    return torch.tanh(model.value_head(state)).squeeze(-1)


def device_exact_batch_losses(
    model: TemporalCabtTransformer,
    corpus: DeviceResidentBootstrapCorpus,
    sample_ids: torch.Tensor,
    *,
    baseline_pred: Optional[torch.Tensor] = None,
    value_weight: float = 1.0,
    aux_weight: float = 0.1,
    opp_hand_weight: float = 0.4,
    opp_remainder_weight: float = 0.3,
    lethal_threat_weight: float = 0.1,
    prize_race_weight: float = 0.1,
    alakazam_guide_weight: float = 0.0,
    awr_beta: float = 0.5,
    awr_weight_max: float = 20.0,
    awr_normalize_advantages: bool = True,
    entropy_bonus: float = 0.01,
) -> tuple[torch.Tensor, BatchMetrics]:
    """Full AWR + exact-hidden losses with all source tensors on-device."""
    if model.decision_context != "stateless":
        raise ValueError("device-resident exact replay requires stateless context")
    if not corpus.has_exact_targets:
        raise ValueError("resident corpus lacks exact-hidden targets")
    ids = sample_ids.reshape(-1).to(device=corpus.device, dtype=torch.long)
    board, options, counts, target_idx, v_target = corpus.batch(ids)
    k = int(target_idx.numel())
    if k <= 0:
        device = next(model.parameters()).device
        return torch.zeros((), device=device, requires_grad=True), BatchMetrics()

    spatial = model.encode_board_packed(board, batch_size=k)
    cls = model.pool_cls(spatial).unsqueeze(1)
    states, _ = model.temporal_encode(cls, append=False, return_all=True)
    state = states.squeeze(1)
    logits = model.decode_options_packed(
        options,
        spatial,
        state,
        n_options=counts,
        batch_size=k,
    )
    value_pred = torch.tanh(model.value_head(state)).squeeze(-1)
    v_target = v_target.to(dtype=value_pred.dtype)
    log_p = torch.nan_to_num(F.log_softmax(logits, dim=-1), neginf=0.0)
    selected_log_p = log_p[
        torch.arange(k, device=logits.device), target_idx
    ]

    guide_loss = log_p.sum() * 0.0
    n_guide_rows = 0
    if float(alakazam_guide_weight) > 0.0:
        if not corpus.has_guide_targets:
            raise ValueError(
                "nonzero Alakazam guide weight requires resident guide targets"
            )
        assert corpus.guide_target_index is not None
        assert corpus.guide_confidence is not None
        guide_target = corpus.guide_target_index.index_select(0, ids).to(
            dtype=torch.long
        )
        guide_confidence = corpus.guide_confidence.index_select(0, ids).to(
            dtype=log_p.dtype
        )
        invalid = (guide_target >= counts) & (guide_target >= 0)
        if bool(invalid.any()):
            bad_row = int(torch.nonzero(invalid, as_tuple=False)[0].item())
            raise ValueError(
                "resident Alakazam guide target is outside option row: "
                f"target={int(guide_target[bad_row].item())} "
                f"options={int(counts[bad_row].item())}"
            )
        guide_mask = (
            (guide_target >= 0)
            & (guide_confidence > 0.0)
            & (counts >= 2)
        )
        if bool(guide_mask.any()):
            guide_rows = torch.nonzero(guide_mask, as_tuple=False).flatten()
            bounded_confidence = guide_confidence.index_select(
                0, guide_rows
            ).clamp(0.0, 1.0)
            guide_loss = -(
                log_p[
                    guide_rows,
                    guide_target.index_select(0, guide_rows),
                ]
                * bounded_confidence
            ).mean()
            n_guide_rows = int(guide_rows.numel())

    baseline = value_pred.detach() if baseline_pred is None else baseline_pred.detach()
    baseline = baseline.to(device=value_pred.device, dtype=value_pred.dtype).reshape(-1)
    if baseline.numel() != k:
        raise ValueError(f"AWR baseline rows {baseline.numel()} != batch rows {k}")
    raw_advantages = v_target - baseline
    raw_stats = raw_advantages.detach().float()
    advantages = raw_advantages
    if awr_normalize_advantages and k > 1:
        advantages = (advantages - advantages.mean()) / advantages.std(
            unbiased=False
        ).clamp_min(1e-6)
    beta = max(float(awr_beta), 1e-6)
    weight_max = max(float(awr_weight_max), 1e-6)
    raw_weights = torch.exp(advantages / beta)
    weights = torch.clamp(raw_weights, max=weight_max)
    policy_loss = -(weights.detach() * selected_log_p).mean()
    if float(entropy_bonus) > 0.0:
        probabilities = torch.nan_to_num(log_p.exp(), nan=0.0)
        entropy = -(probabilities * log_p).sum(dim=-1).mean()
        policy_loss = policy_loss - float(entropy_bonus) * entropy
    value_loss = F.smooth_l1_loss(value_pred, v_target)

    belief = model.belief_aux_logits(state)
    targets = corpus.exact_targets(ids)
    aux_labels = targets["aux_class"]
    aux_mask = aux_labels >= 0
    if bool(aux_mask.any()):
        aux_loss = F.cross_entropy(
            belief["aux_logits"][aux_mask], aux_labels[aux_mask]
        )
    else:
        aux_loss = belief["aux_logits"].sum() * 0.0
    hand_mask = targets["hand_mask"]
    if bool(hand_mask.any()):
        opp_hand_loss = F.binary_cross_entropy_with_logits(
            belief["opp_hand_logits"][hand_mask], targets["hand"][hand_mask]
        )
    else:
        opp_hand_loss = belief["opp_hand_logits"].sum() * 0.0
    remainder_mask = targets["remainder_mask"]
    if bool(remainder_mask.any()):
        opp_remainder_loss = F.binary_cross_entropy_with_logits(
            belief["opp_remainder_logits"][remainder_mask],
            targets["remainder"][remainder_mask],
        )
    else:
        opp_remainder_loss = belief["opp_remainder_logits"].sum() * 0.0
    lethal_mask = targets["lethal_mask"]
    if bool(lethal_mask.any()):
        lethal_threat_loss = F.binary_cross_entropy_with_logits(
            belief["lethal_threat_logits"][lethal_mask],
            targets["lethal"][lethal_mask].to(
                dtype=belief["lethal_threat_logits"].dtype
            ),
        )
    else:
        lethal_threat_loss = belief["lethal_threat_logits"].sum() * 0.0
    prize_mask = targets["prize_race_mask"]
    if bool(prize_mask.any()):
        prize_race_loss = F.smooth_l1_loss(
            belief["prize_race_pred"][prize_mask],
            targets["prize_race"][prize_mask].to(
                dtype=belief["prize_race_pred"].dtype
            ),
        )
    else:
        prize_race_loss = belief["prize_race_pred"].sum() * 0.0

    total = (
        policy_loss
        + float(value_weight) * value_loss
        + float(aux_weight) * aux_loss
        + float(opp_hand_weight) * opp_hand_loss
        + float(opp_remainder_weight) * opp_remainder_loss
        + float(lethal_threat_weight) * lethal_threat_loss
        + float(prize_race_weight) * prize_race_loss
        + float(alakazam_guide_weight) * guide_loss
    )
    predictions = logits.argmax(dim=1)
    weight_stats = weights.detach().float()
    weight_sum = float(weight_stats.sum().item())
    weight_sq_sum = float(weight_stats.square().sum().item())
    effective_samples = min(
        float(k), (weight_sum * weight_sum) / max(weight_sq_sum, 1e-12)
    )
    sorted_weights, _ = torch.sort(weight_stats)
    normalized_stats = advantages.detach().float()
    metrics = BatchMetrics(
        policy_loss=float(policy_loss.detach().item()),
        value_loss=float(value_loss.detach().item()),
        aux_loss=float(aux_loss.detach().item()),
        alakazam_guide_loss=float(guide_loss.detach().item()),
        opp_hand_loss=float(opp_hand_loss.detach().item()),
        opp_remainder_loss=float(opp_remainder_loss.detach().item()),
        lethal_threat_loss=float(lethal_threat_loss.detach().item()),
        prize_race_loss=float(prize_race_loss.detach().item()),
        total_loss=float(total.detach().item()),
        policy_acc=float((predictions == target_idx).float().mean().item()),
        target_value_mean=float(v_target.detach().float().mean().item()),
        value_pred_mean=float(value_pred.detach().float().mean().item()),
        n_decisions=k,
        n_alakazam_guide_rows=n_guide_rows,
        n_archetype_rows=int(aux_mask.sum().item()),
        n_opp_hand_rows=int(hand_mask.sum().item()),
        n_opp_remainder_rows=int(remainder_mask.sum().item()),
        n_lethal_threat_rows=int(lethal_mask.sum().item()),
        n_prize_race_rows=int(prize_mask.sum().item()),
        mean_advantage=float(raw_stats.abs().mean().item()),
        raw_advantage_mean=float(raw_stats.mean().item()),
        raw_advantage_std=float(raw_stats.std(unbiased=False).item()),
        raw_advantage_mean_abs=float(raw_stats.abs().mean().item()),
        raw_advantage_mean_sq=float(raw_stats.square().mean().item()),
        normalized_advantage_mean=float(normalized_stats.mean().item()),
        normalized_advantage_std=float(normalized_stats.std(unbiased=False).item()),
        normalized_advantage_mean_abs=float(normalized_stats.abs().mean().item()),
        normalized_advantage_mean_sq=float(normalized_stats.square().mean().item()),
        awr_weight_mean=float(weight_stats.mean().item()),
        awr_weight_sum=weight_sum,
        awr_weight_sq_sum=weight_sq_sum,
        awr_weight_p50=float(sorted_weights[int(0.50 * (k - 1))].item()),
        awr_weight_p95=float(sorted_weights[int(0.95 * (k - 1))].item()),
        awr_weight_clip_frac=float((raw_weights >= weight_max).float().mean().item()),
        awr_effective_sample_size=effective_samples,
        awr_effective_sample_fraction=effective_samples / max(k, 1),
        policy_selected_nll=float((-selected_log_p.detach()).mean().item()),
    )
    return total, metrics


def _merge_metrics(parts: Sequence[BatchMetrics]) -> BatchMetrics:
    if not parts:
        return BatchMetrics()
    nd = sum(p.n_decisions for p in parts)
    ng = sum(p.n_games for p in parts)
    if nd == 0:
        return BatchMetrics(n_games=ng)

    def wavg(attr: str) -> float:
        return sum(getattr(p, attr) * p.n_decisions for p in parts) / nd

    raw_mean = wavg("raw_advantage_mean")
    raw_mean_sq = wavg("raw_advantage_mean_sq")
    normalized_mean = wavg("normalized_advantage_mean")
    normalized_mean_sq = wavg("normalized_advantage_mean_sq")
    weight_sum = sum(p.awr_weight_sum for p in parts)
    weight_sq_sum = sum(p.awr_weight_sq_sum for p in parts)
    ess = min(float(nd), (weight_sum * weight_sum) / max(weight_sq_sum, 1e-12))
    # Non-AWR batches carry zero weight sums; avoid reporting a fictitious ESS.
    if weight_sum <= 0.0:
        ess = 0.0
    guide_rows = sum(int(p.n_alakazam_guide_rows) for p in parts)
    archetype_rows = sum(int(p.n_archetype_rows) for p in parts)
    hand_rows = sum(int(p.n_opp_hand_rows) for p in parts)
    remainder_rows = sum(int(p.n_opp_remainder_rows) for p in parts)
    lethal_rows = sum(int(p.n_lethal_threat_rows) for p in parts)
    prize_rows = sum(int(p.n_prize_race_rows) for p in parts)
    guide_loss = (
        sum(
            float(p.alakazam_guide_loss) * int(p.n_alakazam_guide_rows)
            for p in parts
        )
        / guide_rows
        if guide_rows > 0
        else 0.0
    )

    return BatchMetrics(
        policy_loss=wavg("policy_loss"),
        value_loss=wavg("value_loss"),
        aux_loss=wavg("aux_loss"),
        alakazam_guide_loss=guide_loss,
        opp_hand_loss=wavg("opp_hand_loss"),
        opp_remainder_loss=wavg("opp_remainder_loss"),
        lethal_threat_loss=wavg("lethal_threat_loss"),
        prize_race_loss=wavg("prize_race_loss"),
        history_identity_loss=wavg("history_identity_loss"),
        total_loss=wavg("total_loss"),
        policy_acc=wavg("policy_acc"),
        policy_kl=wavg("policy_kl"),
        target_value_mean=wavg("target_value_mean"),
        value_pred_mean=wavg("value_pred_mean"),
        n_decisions=nd,
        n_games=ng,
        n_alakazam_guide_rows=guide_rows,
        n_archetype_rows=archetype_rows,
        n_opp_hand_rows=hand_rows,
        n_opp_remainder_rows=remainder_rows,
        n_lethal_threat_rows=lethal_rows,
        n_prize_race_rows=prize_rows,
        mean_advantage=wavg("mean_advantage"),
        raw_advantage_mean=raw_mean,
        raw_advantage_std=math.sqrt(max(raw_mean_sq - raw_mean * raw_mean, 0.0)),
        raw_advantage_mean_abs=wavg("raw_advantage_mean_abs"),
        raw_advantage_mean_sq=raw_mean_sq,
        normalized_advantage_mean=normalized_mean,
        normalized_advantage_std=math.sqrt(
            max(normalized_mean_sq - normalized_mean * normalized_mean, 0.0)
        ),
        normalized_advantage_mean_abs=wavg("normalized_advantage_mean_abs"),
        normalized_advantage_mean_sq=normalized_mean_sq,
        awr_weight_mean=wavg("awr_weight_mean"),
        awr_weight_sum=weight_sum,
        awr_weight_sq_sum=weight_sq_sum,
        awr_weight_p50=wavg("awr_weight_p50"),
        awr_weight_p95=wavg("awr_weight_p95"),
        awr_weight_clip_frac=wavg("awr_weight_clip_frac"),
        awr_effective_sample_size=ess,
        awr_effective_sample_fraction=ess / nd,
        policy_selected_nll=wavg("policy_selected_nll"),
    )


def split_dataset(
    ds: BootstrapDataset,
    val_frac: float,
    seed: int,
    *,
    group_by_episode: bool = False,
) -> tuple[list[GameSequence], list[GameSequence]]:
    seqs = list(ds.sequences)
    rng = random.Random(seed)
    if group_by_episode and val_frac > 0 and len(seqs) > 1:
        grouped: dict[str, list[GameSequence]] = {}
        sequence_group: dict[int, str] = {}
        for index, seq in enumerate(seqs):
            episode_id = str(seq.episode_id or f"__missing_episode_{index}")
            grouped.setdefault(episode_id, []).append(seq)
            sequence_group[id(seq)] = episode_id
        group_ids = list(grouped)
        rng.shuffle(group_ids)
        if len(group_ids) <= 1:
            return seqs, []
        target = max(1, int(len(seqs) * val_frac))
        val_ids: set[str] = set()
        selected = 0
        # Always leave at least one whole episode in training.
        for episode_id in group_ids[:-1]:
            val_ids.add(episode_id)
            selected += len(grouped[episode_id])
            if selected >= target:
                break
        train = [seq for seq in seqs if sequence_group[id(seq)] not in val_ids]
        val = [seq for seq in seqs if sequence_group[id(seq)] in val_ids]
        return train, val
    rng.shuffle(seqs)
    if val_frac <= 0 or len(seqs) <= 1:
        return seqs, []
    n_val = max(1, int(len(seqs) * val_frac)) if len(seqs) > 1 else 0
    if n_val == 0:
        return seqs, []
    return seqs[n_val:], seqs[:n_val]


def cap_game_sequence_context(
    sequence: GameSequence,
    max_context: int,
) -> tuple[GameSequence, bool]:
    """Keep a causal game prefix within the temporal attention cap.

    The sequence remains owned by exactly one acting seat and one episode.
    A rare game longer than the measured 320-step window contributes its
    correctly conditioned prefix; no reset chunk or cross-game packing is
    introduced merely to retain the tail.
    """
    cap = int(max_context)
    if cap <= 0:
        raise ValueError("history max_context must be positive")
    if len(sequence.decisions) <= cap:
        return sequence, False

    def _prefix(value):
        return value[:cap] if value is not None else None

    return (
        replace(
            sequence,
            decisions=list(sequence.decisions[:cap]),
            policy_targets=_prefix(sequence.policy_targets),
            factorized_policy_targets=_prefix(
                sequence.factorized_policy_targets
            ),
        ),
        True,
    )


def cap_history_sequences(
    sequences: Sequence[GameSequence],
    max_context: int,
) -> tuple[list[GameSequence], int]:
    capped: list[GameSequence] = []
    truncated = 0
    for sequence in sequences:
        row, changed = cap_game_sequence_context(sequence, max_context)
        capped.append(row)
        truncated += int(changed)
    return capped, truncated


def _iter_game_batches(
    sequences: list[GameSequence],
    games_per_batch: int,
    max_decisions: int,
    shuffle: bool,
    seed: int,
    epoch: int,
) -> list[list[GameSequence]]:
    order = list(range(len(sequences)))
    if shuffle:
        rng = random.Random(seed + epoch * 10007)
        rng.shuffle(order)
    batches: list[list[GameSequence]] = []
    cur: list[GameSequence] = []
    cur_dec = 0
    for i in order:
        seq = sequences[i]
        n = len(seq)
        if cur and (
            len(cur) >= games_per_batch or cur_dec + n > max_decisions
        ):
            batches.append(cur)
            cur, cur_dec = [], 0
        cur.append(seq)
        cur_dec += n
    if cur:
        batches.append(cur)
    return batches


@torch.no_grad()
def evaluate(
    model: TemporalCabtTransformer,
    sequences: list[GameSequence],
    *,
    cfg: TrainConfig,
    desc: str = "val",
    awr_baseline_cache: Optional[dict[tuple[int, int, int], float]] = None,
) -> BatchMetrics:
    model.eval()
    parts: list[BatchMetrics] = []
    batches = _iter_game_batches(
        list(sequences),
        cfg.games_per_batch,
        cfg.max_decisions_per_batch,
        shuffle=False,
        seed=cfg.seed,
        epoch=0,
    )
    use_amp = device_mod.cuda_available()
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    for batch in tqdm(batches, desc=desc, leave=False, unit="batch"):
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            _, m = batch_losses(
                model,
                batch,
                value_weight=cfg.value_loss_weight,
                aux_weight=cfg.aux_loss_weight,
                opp_hand_weight=cfg.opp_hand_loss_weight,
                opp_remainder_weight=cfg.opp_remainder_loss_weight,
                lethal_threat_weight=cfg.lethal_threat_loss_weight,
                prize_race_weight=cfg.prize_race_loss_weight,
                alakazam_guide_weight=cfg.alakazam_guide_loss_weight,
                pure_rl=bool(cfg.pure_rl),
                awr_beta=float(cfg.awr_beta),
                awr_weight_max=float(cfg.awr_weight_max),
                awr_normalize_advantages=bool(cfg.awr_normalize_advantages),
                entropy_bonus=float(cfg.entropy_bonus),
                awr_baseline_cache=awr_baseline_cache,
                history_identity_weight=float(
                    cfg.history_identity_loss_weight
                ),
            )
        parts.append(m)
    return _merge_metrics(parts)


@torch.no_grad()
def evaluate_device_corpus(
    model: TemporalCabtTransformer,
    corpus: DeviceResidentBootstrapCorpus,
    *,
    cfg: TrainConfig,
    batch_size: int,
    desc: str = "val",
) -> BatchMetrics:
    """Evaluate the validation partition without moving inputs off device."""
    model.eval()
    parts: list[BatchMetrics] = []
    temporal = model.decision_context == "history"
    batches = (
        corpus.temporal_batches(
            train=False,
            batch_size=batch_size,
            shuffle=False,
            seed=cfg.seed,
            epoch=0,
        )
        if temporal
        else corpus.batches(
            train=False,
            batch_size=batch_size,
            shuffle=False,
            seed=cfg.seed,
            epoch=0,
        )
    )
    use_amp = bool(cfg.amp and corpus.device.type == "cuda")
    amp_dtype = (
        torch.bfloat16
        if (use_amp and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    for batch_ids in tqdm(batches, desc=desc, leave=False, unit="batch"):
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            if temporal:
                _, metrics = device_temporal_batch_losses(
                    model,
                    corpus,
                    batch_ids,
                    value_weight=cfg.value_loss_weight,
                    aux_weight=cfg.aux_loss_weight,
                    opp_hand_weight=cfg.opp_hand_loss_weight,
                    opp_remainder_weight=cfg.opp_remainder_loss_weight,
                    lethal_threat_weight=cfg.lethal_threat_loss_weight,
                    prize_race_weight=cfg.prize_race_loss_weight,
                    alakazam_guide_weight=cfg.alakazam_guide_loss_weight,
                )
            else:
                _, metrics = device_batch_losses(
                    model,
                    corpus,
                    batch_ids,
                    value_weight=cfg.value_loss_weight,
                )
        parts.append(metrics)
    return _merge_metrics(parts)


@torch.no_grad()
def _device_exact_value_cache(
    model: TemporalCabtTransformer,
    corpus: DeviceResidentBootstrapCorpus,
    *,
    cfg: TrainConfig,
    batch_size: int,
    desc: str,
) -> torch.Tensor:
    """Freeze V(s) for every resident train/validation sample on-device."""
    was_training = model.training
    model.eval()
    values = torch.empty(
        corpus.total_samples,
        device=corpus.device,
        dtype=torch.float32,
    )
    ids = torch.arange(
        corpus.total_samples, device=corpus.device, dtype=torch.long
    )
    batches = list(ids.split(max(1, int(batch_size))))
    use_amp = bool(cfg.amp and corpus.device.type == "cuda")
    amp_dtype = (
        torch.bfloat16
        if (use_amp and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    try:
        for sample_ids in tqdm(batches, desc=desc, leave=False, unit="batch"):
            with torch.amp.autocast(
                "cuda", enabled=use_amp, dtype=amp_dtype
            ):
                prediction = device_exact_value_predictions(
                    model, corpus, sample_ids
                )
            values.index_copy_(0, sample_ids, prediction.float())
    finally:
        model.train(was_training)
    return values


@torch.no_grad()
def _device_exact_policy_predictions(
    model: TemporalCabtTransformer,
    corpus: DeviceResidentBootstrapCorpus,
    *,
    cfg: TrainConfig,
    batch_size: int,
    desc: str,
) -> torch.Tensor:
    """Stable argmax policy rows for resident parent/candidate agreement."""
    was_training = model.training
    model.eval()
    predictions = torch.empty(
        corpus.total_samples,
        device=corpus.device,
        dtype=torch.int32,
    )
    ids = torch.arange(
        corpus.total_samples, device=corpus.device, dtype=torch.long
    )
    batches = list(ids.split(max(1, int(batch_size))))
    use_amp = bool(cfg.amp and corpus.device.type == "cuda")
    amp_dtype = (
        torch.bfloat16
        if (use_amp and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    try:
        for sample_ids in tqdm(batches, desc=desc, leave=False, unit="batch"):
            with torch.amp.autocast(
                "cuda", enabled=use_amp, dtype=amp_dtype
            ):
                board, options, counts, _target, _value = corpus.batch(
                    sample_ids
                )
                k = int(sample_ids.numel())
                spatial = model.encode_board_packed(board, batch_size=k)
                cls = model.pool_cls(spatial).unsqueeze(1)
                states, _ = model.temporal_encode(
                    cls, append=False, return_all=True
                )
                logits = model.decode_options_packed(
                    options,
                    spatial,
                    states.squeeze(1),
                    n_options=counts,
                    batch_size=k,
                )
            predictions.index_copy_(
                0, sample_ids, logits.argmax(dim=1).to(dtype=torch.int32)
            )
    finally:
        model.train(was_training)
    return predictions


@torch.no_grad()
def _evaluate_device_exact_corpus(
    model: TemporalCabtTransformer,
    corpus: DeviceResidentBootstrapCorpus,
    *,
    cfg: TrainConfig,
    batch_size: int,
    baseline: torch.Tensor,
    desc: str,
) -> BatchMetrics:
    """Evaluate exact resident validation rows with the frozen AWR baseline."""
    was_training = model.training
    model.eval()
    parts: list[BatchMetrics] = []
    batches = corpus.batches(
        train=False,
        batch_size=batch_size,
        shuffle=False,
        seed=cfg.seed,
        epoch=0,
    )
    use_amp = bool(cfg.amp and corpus.device.type == "cuda")
    amp_dtype = (
        torch.bfloat16
        if (use_amp and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    try:
        for sample_ids in tqdm(batches, desc=desc, leave=False, unit="batch"):
            with torch.amp.autocast(
                "cuda", enabled=use_amp, dtype=amp_dtype
            ):
                _, metrics = device_exact_batch_losses(
                    model,
                    corpus,
                    sample_ids,
                    baseline_pred=baseline.index_select(0, sample_ids),
                    value_weight=cfg.value_loss_weight,
                    aux_weight=cfg.aux_loss_weight,
                    opp_hand_weight=cfg.opp_hand_loss_weight,
                    opp_remainder_weight=cfg.opp_remainder_loss_weight,
                    lethal_threat_weight=cfg.lethal_threat_loss_weight,
                    prize_race_weight=cfg.prize_race_loss_weight,
                    alakazam_guide_weight=cfg.alakazam_guide_loss_weight,
                    awr_beta=cfg.awr_beta,
                    awr_weight_max=cfg.awr_weight_max,
                    awr_normalize_advantages=cfg.awr_normalize_advantages,
                    entropy_bonus=cfg.entropy_bonus,
                )
            parts.append(metrics)
    finally:
        model.train(was_training)
    return _merge_metrics(parts)


def _fit_device_batch_size(
    model: TemporalCabtTransformer,
    corpus: DeviceResidentBootstrapCorpus,
    *,
    requested: int,
    cfg: TrainConfig,
    use_amp: bool,
    amp_dtype: torch.dtype,
    min_step_free_gib: float = 2.0,
) -> int:
    """Prove corpus + forward/backward fit, reducing only after a real OOM."""
    size = min(max(1, int(requested)), corpus.train_samples)
    if size <= 0:
        raise ValueError("device corpus has no training samples")
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(corpus.device) if corpus.device.type == "cuda" else None
    try:
        while True:
            # Include the globally widest training row in the probe.  This
            # exercises the decoder's rectangular padding at its actual worst
            # width instead of approving an unusually easy first batch.
            temporal = model.decision_context == "history"
            if temporal:
                batch_ids = corpus.temporal_probe_batch(size)
            else:
                batch_ids = torch.arange(
                    size, device=corpus.device, dtype=torch.long
                )
                widest = torch.argmax(
                    corpus.n_options[: corpus.train_samples]
                ).to(dtype=torch.long)
                batch_ids[-1] = widest
            if corpus.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(corpus.device)
            model.train()
            model.zero_grad(set_to_none=True)
            oom = False
            try:
                with torch.amp.autocast(
                    "cuda", enabled=use_amp, dtype=amp_dtype
                ):
                    if temporal:
                        total, _ = device_temporal_batch_losses(
                            model,
                            corpus,
                            batch_ids,
                            value_weight=cfg.value_loss_weight,
                            aux_weight=cfg.aux_loss_weight,
                            opp_hand_weight=cfg.opp_hand_loss_weight,
                            opp_remainder_weight=cfg.opp_remainder_loss_weight,
                            lethal_threat_weight=cfg.lethal_threat_loss_weight,
                            prize_race_weight=cfg.prize_race_loss_weight,
                            alakazam_guide_weight=cfg.alakazam_guide_loss_weight,
                        )
                    elif cfg.pure_rl and corpus.has_exact_targets:
                        total, _ = device_exact_batch_losses(
                            model,
                            corpus,
                            batch_ids,
                            baseline_pred=torch.zeros(
                                size,
                                device=corpus.device,
                                dtype=torch.float32,
                            ),
                            value_weight=cfg.value_loss_weight,
                            aux_weight=cfg.aux_loss_weight,
                            opp_hand_weight=cfg.opp_hand_loss_weight,
                            opp_remainder_weight=cfg.opp_remainder_loss_weight,
                            lethal_threat_weight=cfg.lethal_threat_loss_weight,
                            prize_race_weight=cfg.prize_race_loss_weight,
                            alakazam_guide_weight=cfg.alakazam_guide_loss_weight,
                            awr_beta=cfg.awr_beta,
                            awr_weight_max=cfg.awr_weight_max,
                            awr_normalize_advantages=(
                                cfg.awr_normalize_advantages
                            ),
                            entropy_bonus=cfg.entropy_bonus,
                        )
                    else:
                        total, _ = device_batch_losses(
                            model,
                            corpus,
                            batch_ids,
                            value_weight=cfg.value_loss_weight,
                        )
                total.backward()
                if corpus.device.type == "cuda":
                    torch.cuda.synchronize(corpus.device)
            except RuntimeError as exc:
                if not config.is_cuda_oom(exc):
                    raise
                oom = True
            if not oom:
                model.zero_grad(set_to_none=True)
                if corpus.device.type == "cuda":
                    peak = torch.cuda.max_memory_allocated(corpus.device)
                    free, total_memory = torch.cuda.mem_get_info(corpus.device)
                    if free < int(float(min_step_free_gib) * 2**30):
                        oom = True
                        print(
                            f"[device-corpus] fit-test batch={size} rejected: "
                            f"post-step CUDA-free={free / 2**30:.2f} GiB < "
                            f"required {float(min_step_free_gib):.2f} GiB",
                            flush=True,
                        )
                if not oom:
                    if corpus.device.type == "cuda":
                        free, total_memory = torch.cuda.mem_get_info(corpus.device)
                        peak = torch.cuda.max_memory_allocated(corpus.device)
                        print(
                            f"[device-corpus] fit-test batch={size} passed "
                            f"peak-allocated={peak / 2**30:.2f} GiB "
                            f"CUDA-free={free / 2**30:.2f}/"
                            f"{total_memory / 2**30:.2f} GiB",
                            flush=True,
                        )
                    return size

            model.zero_grad(set_to_none=True)
            del batch_ids
            gc.collect()
            if corpus.device.type == "cuda":
                torch.cuda.empty_cache()
            reduced = size // 2
            if reduced < 128:
                raise MemoryError(
                    "device-resident corpus fits, but even a 128-sample "
                    "training batch exhausts activation memory"
                )
            print(
                f"[device-corpus] fit-test OOM at batch={size}; retrying "
                f"batch={reduced}",
                flush=True,
            )
            size = reduced
    finally:
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, corpus.device)
        model.zero_grad(set_to_none=True)
        if corpus.device.type == "cuda":
            torch.cuda.empty_cache()


def train_bootstrap(
    dataset: BootstrapDataset,
    *,
    run_name: str = "dragapult_bootstrap",
    archetype_id: str = "dragapult",
    train_cfg: Optional[TrainConfig] = None,
    resume: Union[str, bool, None] = "auto",
    device: Optional[torch.device] = None,
    model_cfg: Optional[config.ModelConfig] = None,
    init_checkpoint: Optional[Union[str, Path]] = None,
    checkpoint_extra: Optional[dict[str, Any]] = None,
    device_resident: bool = False,
    trainable_parameter_prefixes: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Run supervised BC/value training with early stopping + AMP + checkpoints."""
    cfg = train_cfg or TrainConfig()
    device = device or device_mod.training_device(
        prefer_name=config.HARDWARE.train_gpu_name, allow_cpu=False
    )
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    train_seqs, val_seqs = split_dataset(
        dataset,
        cfg.val_frac,
        cfg.seed,
        group_by_episode=bool(cfg.split_by_episode),
    )
    print(
        f"[train] device={device} games={len(dataset)} "
        f"train={len(train_seqs)} val={len(val_seqs)} decisions={dataset.n_decisions}",
        flush=True,
    )

    resume_path = checkpoint.resolve_resume_path(run_name, resume)
    init_path = Path(init_checkpoint).expanduser().resolve() if init_checkpoint else None
    if resume_path is not None and init_path is not None:
        raise ValueError("init_checkpoint cannot be combined with a resumed run")
    if init_path is not None and not init_path.is_file():
        raise FileNotFoundError(f"initial checkpoint not found: {init_path}")
    source_path = resume_path or init_path
    model = (
        load_model_from_checkpoint(source_path, device=device)
        if source_path is not None
        else build_model(model_cfg or config.MODEL, device=device)
    )
    if model_cfg is not None and getattr(model, "cfg", None) != model_cfg:
        raise ValueError("initial checkpoint model profile does not match model_cfg")
    if model.decision_context == "history":
        train_seqs, train_truncated = cap_history_sequences(
            train_seqs, model.max_context
        )
        val_seqs, val_truncated = cap_history_sequences(
            val_seqs, model.max_context
        )
        print(
            f"[train] game-bounded temporal context={model.max_context} "
            f"truncated_sequences={train_truncated + val_truncated}",
            flush=True,
        )
    trainable_prefixes = tuple(
        str(value) for value in (trainable_parameter_prefixes or ())
    )
    if trainable_prefixes:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(
                any(name.startswith(prefix) for prefix in trainable_prefixes)
            )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("training freeze policy left no trainable parameters")
    if trainable_prefixes:
        print(
            "[train] frozen-copy calibration trainable_prefixes="
            f"{list(trainable_prefixes)} params="
            f"{sum(parameter.numel() for parameter in trainable_parameters)}",
            flush=True,
        )
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    # Blackwell throughput: allow TF32 matmuls and prefer bf16 autocast.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    use_amp = bool(cfg.amp and device.type == "cuda")
    amp_dtype = (
        torch.bfloat16
        if (use_amp and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    # bf16 carries fp32 dynamic range → no GradScaler needed; only fp16 needs it.
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(use_amp and amp_dtype == torch.float16)
    )
    if use_amp:
        print(f"[train] AMP dtype={amp_dtype} scaler={scaler.is_enabled()}", flush=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.epochs, 1)
    )

    state = TrainState(patience_left=cfg.early_stop_patience)
    mgr = checkpoint.CheckpointManager(run_name)

    inherited_extra: dict[str, Any] = {}
    if resume_path is not None:
        print(f"[train] resuming from {resume_path}", flush=True)
        ckpt = checkpoint.load_checkpoint(resume_path, map_location=device)
        inherited_extra = dict(ckpt.get("extra") or {})
        meta = checkpoint.apply_checkpoint(
            ckpt, model=model, optimizer=optimizer, scaler=scaler, scheduler=scheduler
        )
        state.step = int(meta["step"])
        state.epoch = int(meta["epoch"])
        if meta.get("best_metric") is not None:
            state.best_metric = float(meta["best_metric"])
        es = meta.get("early_stop_state") or {}
        if "patience_left" in es:
            state.patience_left = int(es["patience_left"])
        state.history = list((meta.get("extra") or {}).get("history") or [])
    elif init_path is not None:
        print(f"[train] initializing weights from {init_path}", flush=True)
        seed_ckpt = checkpoint.load_checkpoint(init_path, map_location="cpu")
        seed_extra = dict(seed_ckpt.get("extra") or {})
        for key in ("pure_rl", "smoke", "model_profile"):
            if key in seed_extra:
                inherited_extra[key] = seed_extra[key]
        inherited_extra["initialized_from"] = str(init_path)
        inherited_extra["initialized_from_digest"] = checkpoint.checkpoint_digest(
            init_path
        )

    resident_corpus: Optional[DeviceResidentBootstrapCorpus] = None
    resident_batch_size = int(cfg.max_decisions_per_batch)
    if device_resident:
        if device.type != "cuda":
            raise ValueError("device-resident bootstrap requires a CUDA device")
        if model.decision_context != "stateless":
            raise ValueError(
                "device-resident bootstrap is valid only for a stateless model"
            )
        weighted_aux = {
            "aux": cfg.aux_loss_weight,
            "opp_hand": cfg.opp_hand_loss_weight,
            "opp_remainder": cfg.opp_remainder_loss_weight,
            "lethal_threat": cfg.lethal_threat_loss_weight,
            "prize_race": cfg.prize_race_loss_weight,
            "alakazam_guide": cfg.alakazam_guide_loss_weight,
        }
        nonzero_aux = {
            name: float(weight)
            for name, weight in weighted_aux.items()
            if float(weight) != 0.0
        }
        if nonzero_aux:
            raise ValueError(
                "device-resident bootstrap requires all auxiliary weights zero: "
                f"{nonzero_aux}"
            )
        if cfg.pure_rl:
            raise ValueError(
                "device-resident bootstrap is supervised hard-target training, "
                "not the AWR pure-RL update path"
            )
        resident_corpus = DeviceResidentBootstrapCorpus.from_splits(
            train_seqs,
            val_seqs,
            device=device,
        )
        # The GPU corpus is now authoritative.  Release every host-side
        # GameSequence reference so the training process cannot accumulate both
        # representations and repeat the earlier host-memory failure.
        dataset.sequences.clear()
        train_seqs.clear()
        val_seqs.clear()
        gc.collect()
        torch.cuda.empty_cache()
        print(
            "[device-corpus] host GameSequence corpus released; all bootstrap "
            "features/targets remain on Blackwell",
            flush=True,
        )
        resident_batch_size = _fit_device_batch_size(
            model,
            resident_corpus,
            requested=cfg.max_decisions_per_batch,
            cfg=cfg,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

    def build_ckpt() -> dict[str, Any]:
        extra = dict(inherited_extra)
        extra.update(checkpoint_extra or {})
        extra.update(
            {
                "history": state.history,
                "train_cfg": cfg.__dict__,
                "device_resident_bootstrap": bool(resident_corpus is not None),
                "device_resident_batch_size": resident_batch_size,
                "trainable_parameter_prefixes": list(trainable_prefixes),
                "trainable_parameter_count": int(
                    sum(parameter.numel() for parameter in trainable_parameters)
                ),
            }
        )
        return checkpoint.build_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler if use_amp else None,
            scheduler=scheduler,
            step=state.step,
            epoch=state.epoch,
            best_metric=state.best_metric,
            early_stop_state={
                "patience_left": state.patience_left,
                "best_metric": state.best_metric,
            },
            archetype_id=archetype_id,
            model_id=run_name,
            model_config=model.cfg,
            extra=extra,
        )

    mgr.install_signal_flush(build_ckpt)

    try:
        epoch_bar = tqdm(
            range(state.epoch, cfg.epochs),
            desc="epochs",
            initial=state.epoch,
            total=cfg.epochs,
            unit="ep",
        )
        for epoch in epoch_bar:
            state.epoch = epoch
            model.train()
            if resident_corpus is not None:
                batches = resident_corpus.batches(
                    train=True,
                    batch_size=resident_batch_size,
                    shuffle=True,
                    seed=cfg.seed,
                    epoch=epoch,
                )
            else:
                batches = _iter_game_batches(
                    train_seqs,
                    cfg.games_per_batch,
                    cfg.max_decisions_per_batch,
                    shuffle=True,
                    seed=cfg.seed,
                    epoch=epoch,
                )
            epoch_parts: list[BatchMetrics] = []
            batch_bar = tqdm(batches, desc=f"train ep{epoch}", leave=False, unit="batch")
            for batch in batch_bar:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    if resident_corpus is not None:
                        total, bm = device_batch_losses(
                            model,
                            resident_corpus,
                            batch,
                            value_weight=cfg.value_loss_weight,
                        )
                    else:
                        total, bm = batch_losses(
                            model,
                            batch,
                            value_weight=cfg.value_loss_weight,
                            aux_weight=cfg.aux_loss_weight,
                            opp_hand_weight=cfg.opp_hand_loss_weight,
                            opp_remainder_weight=cfg.opp_remainder_loss_weight,
                            lethal_threat_weight=cfg.lethal_threat_loss_weight,
                            prize_race_weight=cfg.prize_race_loss_weight,
                            alakazam_guide_weight=cfg.alakazam_guide_loss_weight,
                            history_identity_weight=float(
                                cfg.history_identity_loss_weight
                            ),
                        )
                if bm.n_decisions == 0:
                    continue

                scaler.scale(total).backward()
                if cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters, cfg.grad_clip
                    )
                scaler.step(optimizer)
                scaler.update()

                state.step += 1
                epoch_parts.append(bm)
                batch_bar.set_postfix(
                    loss=f"{bm.total_loss:.3f}",
                    p=f"{bm.policy_loss:.3f}",
                    v=f"{bm.value_loss:.3f}",
                    aux=(
                        "off"
                        if cfg.aux_loss_weight == 0
                        else f"{bm.aux_loss:.3f}/{bm.n_archetype_rows}"
                    ),
                    hand=(
                        "off"
                        if cfg.opp_hand_loss_weight == 0
                        else f"{bm.opp_hand_loss:.3f}/{bm.n_opp_hand_rows}"
                    ),
                    rem=(
                        "off"
                        if cfg.opp_remainder_loss_weight == 0
                        else f"{bm.opp_remainder_loss:.3f}/"
                        f"{bm.n_opp_remainder_rows}"
                    ),
                    lethal=(
                        "off"
                        if cfg.lethal_threat_loss_weight == 0
                        else f"{bm.lethal_threat_loss:.3f}/"
                        f"{bm.n_lethal_threat_rows}"
                    ),
                    prize=(
                        "off"
                        if cfg.prize_race_loss_weight == 0
                        else f"{bm.prize_race_loss:.3f}/"
                        f"{bm.n_prize_race_rows}"
                    ),
                    guide=(
                        "off"
                        if cfg.alakazam_guide_loss_weight == 0
                        else f"{bm.alakazam_guide_loss:.3f}/"
                        f"{bm.n_alakazam_guide_rows}"
                    ),
                    acc=f"{bm.policy_acc:.2%}",
                    step=state.step,
                )

                saved = mgr.maybe_save(state.step, build_ckpt)
                if saved:
                    tqdm.write(
                        f"[checkpoint] step={state.step} saved → "
                        + ", ".join(f"{k}={v.name}" for k, v in saved.items())
                    )

            train_m = _merge_metrics(epoch_parts)
            if resident_corpus is not None and resident_corpus.val_samples:
                val_m = evaluate_device_corpus(
                    model,
                    resident_corpus,
                    cfg=cfg,
                    batch_size=resident_batch_size,
                    desc=f"val ep{epoch}",
                )
                metric = val_m.total_loss
            elif val_seqs:
                val_m = evaluate(model, val_seqs, cfg=cfg, desc=f"val ep{epoch}")
                metric = val_m.total_loss
            else:
                val_m = train_m
                metric = train_m.total_loss

            scheduler.step()
            row = {
                "epoch": epoch,
                "step": state.step,
                "train": train_m.__dict__,
                "val": val_m.__dict__,
                "lr": optimizer.param_groups[0]["lr"],
                "t": time.time(),
            }
            state.history.append(row)

            is_best = metric < state.best_metric - 1e-5
            if is_best:
                state.best_metric = metric
                state.patience_left = cfg.early_stop_patience
                mgr.save(build_ckpt(), is_best=True)
                tqdm.write(
                    f"[checkpoint] NEW BEST epoch={epoch} val_loss={metric:.4f} "
                    f"val_acc={val_m.policy_acc:.2%}"
                )
            else:
                state.patience_left -= 1
                mgr.save(build_ckpt(), is_best=False)
                tqdm.write(
                    f"[train] epoch={epoch} train_loss={train_m.total_loss:.4f} "
                    f"val_loss={metric:.4f} val_acc={val_m.policy_acc:.2%} "
                    f"patience={state.patience_left}"
                )

            epoch_bar.set_postfix(
                val_loss=f"{metric:.4f}",
                val_acc=f"{val_m.policy_acc:.2%}",
                best=f"{state.best_metric:.4f}",
                pat=state.patience_left,
            )

            if state.patience_left <= 0:
                tqdm.write(
                    f"[early-stop] patience exhausted at epoch={epoch} "
                    f"best_val_loss={state.best_metric:.4f}"
                )
                break
    finally:
        mgr.uninstall_signal_flush()
        # Final flush.
        try:
            mgr.save(build_ckpt(), is_best=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[checkpoint] final save failed: {exc}", flush=True)

    best = checkpoint.best_path(run_name)
    latest = checkpoint.latest_path(run_name)
    return {
        "run_name": run_name,
        "best_metric": state.best_metric,
        "step": state.step,
        "epoch": state.epoch,
        "best_path": str(best) if best.is_file() else None,
        "latest_path": str(latest) if latest.is_file() else None,
        "history": state.history,
    }


def supervised_rehearsal_step(
    corpus: DeviceResidentBootstrapCorpus,
    *,
    base_ckpt: Union[str, Path],
    output_path: Union[str, Path],
    parent_digest: str,
    rehearsal_iteration: int,
    manifest_identity: dict[str, Any],
    epochs: int = 1,
    lr: float = 2e-5,
    requested_batch_size: int = 8192,
    seed: int = 0,
    corpus_split_seed: int = 0,
    device: Optional[torch.device] = None,
    aux_loss_weight: float = 0.0,
    opp_hand_loss_weight: float = 0.0,
    opp_remainder_loss_weight: float = 0.0,
    lethal_threat_loss_weight: float = 0.0,
    prize_race_loss_weight: float = 0.0,
    alakazam_guide_loss_weight: float = 0.0,
) -> dict[str, Any]:
    """Run a bounded, resumable expert-policy rehearsal on a resident corpus.

    The returned checkpoint is immutable and keeps the RL iteration number
    unchanged.  It restores the learner's optimizer moments when available,
    applies a deliberately small learning rate for the supervised pass, and
    saves the resulting optimizer state so the next AWR update can continue
    from the exact trained learner rather than restarting AdamW.
    """
    base_path = Path(base_ckpt).expanduser().resolve()
    out_path = Path(output_path).expanduser().resolve()
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    if out_path.exists():
        raise FileExistsError(out_path)
    if epochs <= 0:
        raise ValueError("rehearsal epochs must be positive")
    if lr <= 0.0:
        raise ValueError("rehearsal learning rate must be positive")
    for name, weight in (
        ("aux", aux_loss_weight),
        ("opp_hand", opp_hand_loss_weight),
        ("opp_remainder", opp_remainder_loss_weight),
        ("lethal_threat", lethal_threat_loss_weight),
        ("prize_race", prize_race_loss_weight),
        ("alakazam_guide", alakazam_guide_loss_weight),
    ):
        if float(weight) < 0.0:
            raise ValueError(f"rehearsal {name} loss weight cannot be negative")
    actual_parent_digest = checkpoint.checkpoint_digest(base_path)
    if str(parent_digest) != actual_parent_digest:
        raise ValueError(
            "parent_digest does not match rehearsal base checkpoint: "
            f"expected={parent_digest!r} actual={actual_parent_digest!r}"
        )

    device = device or corpus.device
    if corpus.device != device:
        raise ValueError(
            f"resident corpus device {corpus.device} != rehearsal device {device}"
        )
    cfg = TrainConfig(
        lr=float(lr),
        epochs=int(epochs),
        max_decisions_per_batch=int(requested_batch_size),
        val_frac=0.10,
        split_by_episode=True,
        early_stop_patience=max(1, int(epochs)),
        aux_loss_weight=float(aux_loss_weight),
        opp_hand_loss_weight=float(opp_hand_loss_weight),
        opp_remainder_loss_weight=float(opp_remainder_loss_weight),
        lethal_threat_loss_weight=float(lethal_threat_loss_weight),
        prize_race_loss_weight=float(prize_race_loss_weight),
        alakazam_guide_loss_weight=float(alakazam_guide_loss_weight),
        amp=device.type == "cuda",
        seed=int(seed),
    )
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    model = load_model_from_checkpoint(base_path, device=device)
    warm_started_heads_before = tuple(
        getattr(model, "warm_started_belief_heads", ()) or ()
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    use_amp = bool(cfg.amp and device.type == "cuda")
    amp_dtype = (
        torch.bfloat16
        if (use_amp and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(use_amp and amp_dtype == torch.float16)
    )
    base_payload = checkpoint.load_checkpoint(base_path, map_location="cpu")
    if warm_started_heads_before:
        # ``load_model_from_checkpoint`` already restored every pre-existing
        # tensor and deterministically initialized only allowed new heads. An
        # optimizer snapshot from the legacy architecture cannot contain those
        # parameters, so preserve counters/RNG and use fresh AdamW state.
        meta = checkpoint.apply_checkpoint(base_payload, restore_rng=True)
        optimizer_state_restored = False
    else:
        meta = checkpoint.apply_checkpoint(
            base_payload,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            restore_rng=True,
        )
        optimizer_state_restored = "optimizer_state_dict" in base_payload
    # Optimizer restore also restores its old param-group LR.  Rehearsal uses
    # the explicitly recorded conservative LR while preserving the moments.
    for group in optimizer.param_groups:
        group["lr"] = cfg.lr

    batch_size = _fit_device_batch_size(
        model,
        corpus,
        requested=cfg.max_decisions_per_batch,
        cfg=cfg,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )
    step = int(meta.get("step", 0))
    epoch0 = int(meta.get("epoch", 0))
    rl_iteration = int(meta.get("rl_iteration", 0))
    train_metrics = BatchMetrics()
    started = time.time()
    temporal_rehearsal = model.decision_context == "history"
    if temporal_rehearsal and not corpus.has_temporal_layout:
        raise ValueError(
            "history checkpoint requires a resident full-game rehearsal layout"
        )
    for epoch_offset in range(int(epochs)):
        model.train()
        parts: list[BatchMetrics] = []
        batches = (
            corpus.temporal_batches(
                train=True,
                batch_size=batch_size,
                shuffle=True,
                seed=cfg.seed,
                epoch=epoch_offset,
            )
            if temporal_rehearsal
            else corpus.batches(
                train=True,
                batch_size=batch_size,
                shuffle=True,
                seed=cfg.seed,
                epoch=epoch_offset,
            )
        )
        bar = tqdm(
            batches,
            desc=(
                f"expert rehearsal before iter{int(rehearsal_iteration)} "
                f"ep{epoch_offset + 1}/{int(epochs)}"
            ),
            unit="batch",
        )
        for batch_ids in bar:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda", enabled=use_amp, dtype=amp_dtype
            ):
                if temporal_rehearsal:
                    total, metrics = device_temporal_batch_losses(
                        model,
                        corpus,
                        batch_ids,
                        value_weight=cfg.value_loss_weight,
                        aux_weight=cfg.aux_loss_weight,
                        opp_hand_weight=cfg.opp_hand_loss_weight,
                        opp_remainder_weight=cfg.opp_remainder_loss_weight,
                        lethal_threat_weight=cfg.lethal_threat_loss_weight,
                        prize_race_weight=cfg.prize_race_loss_weight,
                        alakazam_guide_weight=cfg.alakazam_guide_loss_weight,
                    )
                else:
                    total, metrics = device_batch_losses(
                        model,
                        corpus,
                        batch_ids,
                        value_weight=cfg.value_loss_weight,
                    )
            scaler.scale(total).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            parts.append(metrics)
            bar.set_postfix(
                loss=f"{metrics.total_loss:.3f}",
                policy=f"{metrics.policy_loss:.3f}",
                value=f"{metrics.value_loss:.3f}",
                acc=f"{metrics.policy_acc:.2%}",
                step=step,
            )
        train_metrics = _merge_metrics(parts)

    val_metrics = (
        evaluate_device_corpus(
            model,
            corpus,
            cfg=cfg,
            batch_size=batch_size,
            desc=f"expert validation before iter{int(rehearsal_iteration)}",
        )
        if corpus.val_samples
        else train_metrics
    )
    warm_head_loss_weights = {
        "opp_hand_head": float(cfg.opp_hand_loss_weight),
        "opp_remainder_head": float(cfg.opp_remainder_loss_weight),
        "lethal_threat_head": float(cfg.lethal_threat_loss_weight),
        "prize_race_head": float(cfg.prize_race_loss_weight),
    }
    warm_started_heads_remaining = tuple(
        name
        for name in warm_started_heads_before
        if warm_head_loss_weights.get(name, 0.0) <= 0.0
    )
    # Every positive-weight exact target has trained its head, so subsequent
    # policy use must consume the learned outputs instead of the warm fallback.
    model.warm_started_belief_heads = warm_started_heads_remaining
    inherited_extra = dict(base_payload.get("extra") or {})
    rehearsal_record = {
        "schema": 1,
        "before_iteration": int(rehearsal_iteration),
        "parent_digest": actual_parent_digest,
        "manifest": dict(manifest_identity),
        "epochs": int(epochs),
        "learning_rate": float(lr),
        "batch_size": int(batch_size),
        "requested_batch_size": int(requested_batch_size),
        "train_metrics": train_metrics.__dict__,
        "validation_metrics": val_metrics.__dict__,
        "elapsed_sec": float(time.time() - started),
        "optimizer_state_restored": bool(optimizer_state_restored),
        "warm_started_belief_heads_before": list(warm_started_heads_before),
        "warm_started_belief_heads_remaining": list(
            warm_started_heads_remaining
        ),
        "decision_context": str(model.decision_context),
        "resident_temporal_layout": bool(corpus.has_temporal_layout),
        "corpus_split_seed": int(corpus_split_seed),
        "loss_weights": {
            "value": float(cfg.value_loss_weight),
            "archetype": float(cfg.aux_loss_weight),
            "opponent_hand": float(cfg.opp_hand_loss_weight),
            "opponent_hidden_remainder": float(cfg.opp_remainder_loss_weight),
            "lethal_threat": float(cfg.lethal_threat_loss_weight),
            "prize_race": float(cfg.prize_race_loss_weight),
            "alakazam_guide": float(cfg.alakazam_guide_loss_weight),
        },
    }
    inherited_extra.update(
        {
            "pure_rl": True,
            "expert_rehearsal": rehearsal_record,
            "parent_digest": actual_parent_digest,
        }
    )
    payload = checkpoint.build_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler if use_amp else None,
        step=step,
        epoch=epoch0 + int(epochs),
        rl_iteration=rl_iteration,
        best_metric=float(val_metrics.total_loss),
        early_stop_state={
            "patience_left": max(1, int(epochs)),
            "best_metric": float(val_metrics.total_loss),
        },
        archetype_id=str(base_payload.get("archetype_id") or "core"),
        model_id=str(base_payload.get("model_id") or "pure_rl") + ".expert",
        model_config=model.cfg,
        extra=inherited_extra,
    )
    saved = checkpoint.immutable_torch_save(payload, out_path)
    digest = checkpoint.checkpoint_digest(saved)
    return {
        "candidate_path": str(saved),
        "candidate_digest": digest,
        "parent_digest": actual_parent_digest,
        "step": step,
        "rl_iteration": rl_iteration,
        "optimizer_state_restored": bool(optimizer_state_restored),
        "batch_size": int(batch_size),
        "train_metrics": train_metrics.__dict__,
        "validation_metrics": val_metrics.__dict__,
        "rehearsal": rehearsal_record,
    }


def process_with_oom_splitting(
    items: Sequence[Any],
    process: Callable[[list[Any]], Any],
    *,
    is_oom: Callable[[BaseException], bool] = config.is_cuda_oom,
    on_split: Optional[Callable[[], None]] = None,
) -> list[Any]:
    """Process every item, recursively splitting an OOMing batch.

    Both halves are queued in original order. A single-item OOM is re-raised,
    because silently dropping it would corrupt the effective training set.
    """
    pending: list[list[Any]] = [list(items)]
    completed: list[Any] = []
    while pending:
        work = pending.pop(0)
        if not work:
            continue
        try:
            completed.append(process(work))
        except BaseException as exc:  # noqa: BLE001 - preserve CUDA exception type
            if not is_oom(exc) or len(work) <= 1:
                raise
            if on_split is not None:
                on_split()
            mid = len(work) // 2
            pending[0:0] = [work[:mid], work[mid:]]
    return completed


@torch.no_grad()
def _precompute_awr_baseline_cache(
    model: TemporalCabtTransformer,
    sequences: Sequence[GameSequence],
    *,
    cfg: TrainConfig,
    desc: Optional[str] = None,
) -> dict[tuple[int, int, int], float]:
    """Snapshot V(s) for every selected decision-stage before an RL update.

    Keys use the in-memory ``GameSequence`` identity plus decision/stage index;
    split/shuffled batches retain those objects, so the cache remains stable for
    the lifetime of one :func:`rl_train_step` without mutating replay records.
    """
    cache: dict[tuple[int, int, int], float] = {}
    if not sequences:
        return cache

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    use_amp = bool(cfg.amp and device.type == "cuda")
    amp_dtype = (
        torch.bfloat16
        if (use_amp and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    batches = _iter_game_batches(
        list(sequences),
        cfg.games_per_batch,
        cfg.max_decisions_per_batch,
        shuffle=False,
        seed=cfg.seed,
        epoch=0,
    )
    try:
        visible_batches = (
            tqdm(batches, desc=desc, leave=False, unit="batch")
            if desc
            else batches
        )
        for batch in visible_batches:
            def _capture(work: list[GameSequence]) -> BatchMetrics:
                with torch.amp.autocast(
                    "cuda", enabled=use_amp, dtype=amp_dtype
                ):
                    _, metrics = batch_losses(
                        model,
                        work,
                        value_weight=cfg.value_loss_weight,
                        aux_weight=cfg.aux_loss_weight,
                        opp_hand_weight=cfg.opp_hand_loss_weight,
                        opp_remainder_weight=cfg.opp_remainder_loss_weight,
                        lethal_threat_weight=cfg.lethal_threat_loss_weight,
                        prize_race_weight=cfg.prize_race_loss_weight,
                        alakazam_guide_weight=cfg.alakazam_guide_loss_weight,
                        pure_rl=True,
                        awr_beta=float(cfg.awr_beta),
                        awr_weight_max=float(cfg.awr_weight_max),
                        awr_normalize_advantages=bool(
                            cfg.awr_normalize_advantages
                        ),
                        entropy_bonus=float(cfg.entropy_bonus),
                        awr_capture_baseline=cache,
                    )
                return metrics

            process_with_oom_splitting(
                batch,
                _capture,
                on_split=(
                    torch.cuda.empty_cache if device.type == "cuda" else None
                ),
            )
    finally:
        model.train(was_training)
    return cache


@torch.no_grad()
def _policy_argmax_predictions(
    model: TemporalCabtTransformer,
    sequences: Sequence[GameSequence],
    *,
    cfg: TrainConfig,
    awr_baseline_cache: Optional[dict[tuple[int, int, int], float]] = None,
    desc: Optional[str] = None,
) -> list[int]:
    """Return deterministic policy argmaxes in stable decision-stage order."""
    if not sequences:
        return []

    predictions: list[int] = []
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    use_amp = bool(cfg.amp and device.type == "cuda")
    amp_dtype = (
        torch.bfloat16
        if (use_amp and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    batches = _iter_game_batches(
        list(sequences),
        cfg.games_per_batch,
        cfg.max_decisions_per_batch,
        shuffle=False,
        seed=cfg.seed,
        epoch=0,
    )
    try:
        visible_batches = (
            tqdm(batches, desc=desc, leave=False, unit="batch")
            if desc
            else batches
        )
        for batch in visible_batches:
            def _predict(work: list[GameSequence]) -> BatchMetrics:
                with torch.amp.autocast(
                    "cuda", enabled=use_amp, dtype=amp_dtype
                ):
                    _, metrics = batch_losses(
                        model,
                        work,
                        value_weight=cfg.value_loss_weight,
                        aux_weight=cfg.aux_loss_weight,
                        opp_hand_weight=cfg.opp_hand_loss_weight,
                        opp_remainder_weight=cfg.opp_remainder_loss_weight,
                        lethal_threat_weight=cfg.lethal_threat_loss_weight,
                        prize_race_weight=cfg.prize_race_loss_weight,
                        alakazam_guide_weight=cfg.alakazam_guide_loss_weight,
                        pure_rl=bool(cfg.pure_rl),
                        awr_beta=float(cfg.awr_beta),
                        awr_weight_max=float(cfg.awr_weight_max),
                        awr_normalize_advantages=bool(
                            cfg.awr_normalize_advantages
                        ),
                        entropy_bonus=float(cfg.entropy_bonus),
                        awr_baseline_cache=awr_baseline_cache,
                        prediction_sink=predictions,
                    )
                return metrics

            process_with_oom_splitting(
                batch,
                _predict,
                on_split=(
                    torch.cuda.empty_cache if device.type == "cuda" else None
                ),
            )
    finally:
        model.train(was_training)
    return predictions


def rl_train_step(
    dataset: BootstrapDataset,
    *,
    base_ckpt: Union[str, Path],
    out_run_name: str,
    archetype_id: str,
    epochs: int,
    device: Optional[torch.device] = None,
    cfg: Optional[TrainConfig] = None,
    seed: int = 0,
    output_path: Optional[Union[str, Path]] = None,
    parent_digest: Optional[str] = None,
    training_provenance: Optional[dict[str, Any]] = None,
    replace_existing: bool = False,
    device_resident: bool = False,
    device_resident_min_free_gib: float = DEFAULT_MIN_FREE_GIB,
) -> dict[str, Any]:
    """One immutable history-policy candidate fit for the RL loop.

    Unlike :func:`train_bootstrap`, each call is one RL iteration over a fresh
    replay window. Model, Adam/scaler state and global learner counters resume
    from ``base_ckpt``; early-stop selection remains local to the new window.
    Pure RL uses AWR on selected actions plus terminal value regression.  A
    stateless CUDA learner can pack the complete exact replay window once and
    keep the frozen baseline, shuffle order, features, and targets on-device;
    this removes repeated Python packing/H2D gaps between optimizer batches.

    Returns ``{"latest_path", "metrics"}``. Deterministic-ish via ``seed``.
    """
    actual_parent_digest = checkpoint.checkpoint_digest(base_ckpt)
    if parent_digest is not None and str(parent_digest) != actual_parent_digest:
        raise ValueError(
            "pure-RL parent_digest does not match base checkpoint bytes: "
            f"supplied={parent_digest!r} actual={actual_parent_digest!r} "
            f"path={Path(base_ckpt).expanduser().resolve()}"
        )
    # Always persist the verified identity, including for callers which did
    # not supply one.  A nullable lineage field is too easy to misinterpret.
    parent_digest = actual_parent_digest
    cfg = cfg or TrainConfig()
    device = device or device_mod.training_device(
        prefer_name=config.HARDWARE.train_gpu_name, allow_cpu=False
    )
    torch.manual_seed(seed)
    random.seed(seed)

    model = load_model_from_checkpoint(base_ckpt, device=device)
    if model.decision_context not in {"history", "stateless"}:
        raise ValueError(
            "trusted RL training requires a history or stateless checkpoint"
        )
    initial_state = {
        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
    }
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    use_amp = bool(cfg.amp and device.type == "cuda")
    amp_dtype = (
        torch.bfloat16
        if (use_amp and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    base_step = 0
    base_epoch = 0
    base_rl_iteration = 0
    optimizer_state_restored = False
    if cfg.pure_rl:
        learner_ckpt = checkpoint.load_checkpoint(base_ckpt, map_location=device)
        base_step = int(learner_ckpt.get("step", 0))
        base_epoch = int(learner_ckpt.get("epoch", 0))
        base_rl_iteration = int(learner_ckpt.get("rl_iteration", 0))
        if "optimizer_state_dict" in learner_ckpt:
            try:
                checkpoint.apply_checkpoint(
                    learner_ckpt,
                    optimizer=optimizer,
                    scaler=scaler if use_amp else None,
                    restore_rng=False,
                )
            except (KeyError, RuntimeError, ValueError) as exc:
                raise ValueError(
                    "pure-RL parent optimizer state is incompatible with the "
                    "loaded model; refusing a silent Adam reset"
                ) from exc
            optimizer_state_restored = True
            # The checkpoint supplies moments/counters; the current iteration's
            # explicit config remains authoritative for tunable hyperparameters.
            for group in optimizer.param_groups:
                group["lr"] = float(cfg.lr)
                group["weight_decay"] = float(cfg.weight_decay)

    # Keep only decision-bearing games, then reuse bootstrap's val split / ES.
    usable_sequences = [s for s in dataset.sequences if s.decisions]
    if model.decision_context == "history":
        usable_sequences, truncated_sequences = cap_history_sequences(
            usable_sequences, model.max_context
        )
        print(
            f"[rl-train] game-bounded temporal context={model.max_context} "
            f"truncated_sequences={truncated_sequences}",
            flush=True,
        )
    usable = BootstrapDataset(sequences=usable_sequences)
    train_seqs, val_seqs = split_dataset(usable, cfg.val_frac, seed)
    if not train_seqs:
        train_seqs, val_seqs = list(usable.sequences), []
    if not train_seqs:
        raise ValueError("RL training dataset has no usable decision sequences")
    if float(cfg.alakazam_guide_loss_weight) > 0.0:
        guide_rows = count_usable_alakazam_guide_rows(
            [*train_seqs, *val_seqs]
        )
        if guide_rows <= 0:
            raise ValueError(
                "nonzero Alakazam guide loss has no usable guide rows; "
                "verify POKEBOT_ALAKAZAM_GUIDE_TARGETS=1 and scorer coverage"
            )
        print(
            f"[rl-train] Alakazam guide rows={guide_rows} "
            f"weight={float(cfg.alakazam_guide_loss_weight):.4f}",
            flush=True,
        )
    resident_corpus: Optional[DeviceResidentBootstrapCorpus] = None
    resident_baseline: Optional[torch.Tensor] = None
    resident_batch_size = int(cfg.max_decisions_per_batch)
    if device_resident:
        if device.type != "cuda":
            raise ValueError("device-resident RL training requires CUDA")
        if model.decision_context != "stateless":
            raise ValueError(
                "device-resident RL training requires a stateless checkpoint"
            )
        if not cfg.pure_rl or not cfg.awr_freeze_baseline:
            raise ValueError(
                "device-resident RL requires pure RL with a frozen AWR baseline"
            )
        try:
            resident_corpus = DeviceResidentBootstrapCorpus.from_splits(
                train_seqs,
                val_seqs,
                device=device,
                min_free_gib=float(device_resident_min_free_gib),
                exact_card_vocab=int(model.belief_card_vocab),
            )
            resident_batch_size = _fit_device_batch_size(
                model,
                resident_corpus,
                requested=cfg.max_decisions_per_batch,
                cfg=cfg,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                min_step_free_gib=max(
                    2.0, min(4.0, float(device_resident_min_free_gib) / 2.0)
                ),
            )
        except (MemoryError, torch.OutOfMemoryError) as exc:
            resident_corpus = None
            gc.collect()
            torch.cuda.empty_cache()
            print(
                "[device-corpus-rl] capacity fallback to bounded CPU batches: "
                f"{exc}",
                flush=True,
            )
        except RuntimeError as exc:
            # Some CUDA/PyTorch combinations surface allocation failure as a
            # plain RuntimeError instead of torch.OutOfMemoryError.  Preserve
            # fail-closed behavior for every other runtime/data error.
            if not config.is_cuda_oom(exc):
                raise
            resident_corpus = None
            gc.collect()
            torch.cuda.empty_cache()
            print(
                "[device-corpus-rl] CUDA capacity fallback to bounded CPU "
                f"batches: {exc}",
                flush=True,
            )
        if resident_corpus is not None:
            # The GPU corpus is now authoritative. Keep no second copy of the
            # 25+ GiB replay window in Python objects while optimization runs.
            dataset.sequences.clear()
            usable.sequences.clear()
            train_seqs.clear()
            val_seqs.clear()
            gc.collect()
            torch.cuda.empty_cache()
            print(
                "[device-corpus-rl] ALL replay features/targets resident "
                f"bytes={resident_corpus.input_bytes} "
                f"samples={resident_corpus.total_samples} "
                f"batch={resident_batch_size}; host GameSequence corpus released",
                flush=True,
            )
    awr_baseline_cache: Optional[dict[tuple[int, int, int], float]] = None
    if resident_corpus is not None:
        resident_baseline = _device_exact_value_cache(
            model,
            resident_corpus,
            cfg=cfg,
            batch_size=resident_batch_size,
            desc="rl-prep baseline",
        )
    elif cfg.pure_rl and cfg.awr_freeze_baseline:
        awr_baseline_cache = _precompute_awr_baseline_cache(
            model,
            [*train_seqs, *val_seqs],
            cfg=cfg,
            desc="rl-prep baseline",
        )
    agreement_sequences = [*train_seqs, *val_seqs]
    if resident_corpus is not None:
        parent_predictions: Union[list[int], torch.Tensor] = (
            _device_exact_policy_predictions(
                model,
                resident_corpus,
                cfg=cfg,
                batch_size=resident_batch_size,
                desc="rl-agreement parent",
            )
        )
    else:
        parent_predictions = _policy_argmax_predictions(
            model,
            agreement_sequences,
            cfg=cfg,
            awr_baseline_cache=awr_baseline_cache,
            desc="rl-agreement parent",
        )
    patience = max(0, int(cfg.early_stop_patience))
    patience_left = patience
    best_metric = float("inf")
    best_state: Optional[dict[str, Any]] = None
    best_optimizer_state: Optional[dict[str, Any]] = None
    best_scaler_state: Optional[dict[str, Any]] = None
    best_step = base_step
    best_epoch_offset = 0
    best_train_m = BatchMetrics()
    best_val_m = BatchMetrics()
    best_validation_source = (
        "heldout"
        if (
            resident_corpus.val_samples
            if resident_corpus is not None
            else val_seqs
        )
        else "current_data"
    )
    last: BatchMetrics = BatchMetrics()
    last_epoch_optimizer_sps = 0.0
    stepped_epochs = 0
    step = base_step
    epoch_bar = tqdm(range(max(1, epochs)), desc="rl-epochs", leave=False, unit="ep")
    for epoch in epoch_bar:
        model.train()
        epoch_started = time.monotonic()
        epoch_samples = 0
        if resident_corpus is not None:
            batches: Sequence[Any] = resident_corpus.batches(
                train=True,
                batch_size=resident_batch_size,
                shuffle=True,
                seed=seed,
                epoch=epoch,
            )
        else:
            batches = _iter_game_batches(
                train_seqs,
                cfg.games_per_batch,
                cfg.max_decisions_per_batch,
                shuffle=True,
                seed=seed,
                epoch=epoch,
            )
        parts: list[BatchMetrics] = []
        batch_bar = tqdm(
            batches,
            desc=f"rl-train ep{epoch}",
            leave=False,
            unit="batch",
        )
        for batch in batch_bar:
            def _apply_update(total: torch.Tensor, bm: BatchMetrics) -> BatchMetrics:
                nonlocal step
                if bm.n_decisions == 0:
                    return bm
                if not torch.isfinite(total):
                    raise FloatingPointError(f"non-finite training loss: {total}")
                scaler.scale(total).backward()
                if cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                step += 1
                return bm

            def _train_chunk(work: list[GameSequence]) -> BatchMetrics:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    total, bm = batch_losses(
                        model,
                        work,
                        value_weight=cfg.value_loss_weight,
                        aux_weight=cfg.aux_loss_weight,
                        opp_hand_weight=cfg.opp_hand_loss_weight,
                        opp_remainder_weight=cfg.opp_remainder_loss_weight,
                        lethal_threat_weight=cfg.lethal_threat_loss_weight,
                        prize_race_weight=cfg.prize_race_loss_weight,
                        alakazam_guide_weight=cfg.alakazam_guide_loss_weight,
                        pure_rl=bool(cfg.pure_rl),
                        awr_beta=float(cfg.awr_beta),
                        awr_weight_max=float(cfg.awr_weight_max),
                        awr_normalize_advantages=bool(cfg.awr_normalize_advantages),
                        entropy_bonus=float(cfg.entropy_bonus),
                        awr_baseline_cache=awr_baseline_cache,
                    )
                return _apply_update(total, bm)

            def _clear_oom() -> None:
                optimizer.zero_grad(set_to_none=True)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            if resident_corpus is not None:
                assert resident_baseline is not None
                sample_ids = batch
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    "cuda", enabled=use_amp, dtype=amp_dtype
                ):
                    total, resident_metrics = device_exact_batch_losses(
                        model,
                        resident_corpus,
                        sample_ids,
                        baseline_pred=resident_baseline.index_select(
                            0, sample_ids
                        ),
                        value_weight=cfg.value_loss_weight,
                        aux_weight=cfg.aux_loss_weight,
                        opp_hand_weight=cfg.opp_hand_loss_weight,
                        opp_remainder_weight=cfg.opp_remainder_loss_weight,
                        lethal_threat_weight=cfg.lethal_threat_loss_weight,
                        prize_race_weight=cfg.prize_race_loss_weight,
                        alakazam_guide_weight=cfg.alakazam_guide_loss_weight,
                        awr_beta=cfg.awr_beta,
                        awr_weight_max=cfg.awr_weight_max,
                        awr_normalize_advantages=(
                            cfg.awr_normalize_advantages
                        ),
                        entropy_bonus=cfg.entropy_bonus,
                    )
                completed_parts = [_apply_update(total, resident_metrics)]
            else:
                completed_parts = process_with_oom_splitting(
                    batch,
                    _train_chunk,
                    on_split=_clear_oom,
                )
            for bm in completed_parts:
                if bm.n_decisions > 0:
                    parts.append(bm)
                    epoch_samples += int(bm.n_decisions)
            visible = next(
                (bm for bm in reversed(completed_parts) if bm.n_decisions > 0),
                None,
            )
            if visible is not None:
                batch_bar.set_postfix(
                    loss=f"{visible.total_loss:.3f}",
                    p=f"{visible.policy_loss:.3f}",
                    v=f"{visible.value_loss:.3f}",
                    aux=(
                        "off"
                        if cfg.aux_loss_weight == 0
                        else f"{visible.aux_loss:.3f}/"
                        f"{visible.n_archetype_rows}"
                    ),
                    hand=(
                        "off"
                        if cfg.opp_hand_loss_weight == 0
                        else f"{visible.opp_hand_loss:.3f}/"
                        f"{visible.n_opp_hand_rows}"
                    ),
                    rem=(
                        "off"
                        if cfg.opp_remainder_loss_weight == 0
                        else f"{visible.opp_remainder_loss:.3f}/"
                        f"{visible.n_opp_remainder_rows}"
                    ),
                    lethal=(
                        "off"
                        if cfg.lethal_threat_loss_weight == 0
                        else f"{visible.lethal_threat_loss:.3f}/"
                        f"{visible.n_lethal_threat_rows}"
                    ),
                    prize=(
                        "off"
                        if cfg.prize_race_loss_weight == 0
                        else f"{visible.prize_race_loss:.3f}/"
                        f"{visible.n_prize_race_rows}"
                    ),
                    guide=(
                        "off"
                        if cfg.alakazam_guide_loss_weight == 0
                        else f"{visible.alakazam_guide_loss:.3f}/"
                        f"{visible.n_alakazam_guide_rows}"
                    ),
                    acc=f"{visible.policy_acc:.2%}",
                    sps=f"{epoch_samples / max(time.monotonic() - epoch_started, 1e-6):.0f}",
                )
        last = _merge_metrics(parts)
        last_epoch_optimizer_sps = epoch_samples / max(
            time.monotonic() - epoch_started, 1e-6
        )
        if resident_corpus is not None and resident_corpus.val_samples:
            assert resident_baseline is not None
            val_m = _evaluate_device_exact_corpus(
                model,
                resident_corpus,
                cfg=cfg,
                batch_size=resident_batch_size,
                baseline=resident_baseline,
                desc=f"rl-val ep{epoch}",
            )
            metric = val_m.total_loss
        elif val_seqs:
            val_m = evaluate(
                model,
                val_seqs,
                cfg=cfg,
                desc=f"rl-val ep{epoch}",
                awr_baseline_cache=awr_baseline_cache,
            )
            metric = val_m.total_loss
        else:
            val_m = last
            metric = last.total_loss
        stepped_epochs = epoch + 1

        is_best = metric < best_metric - 1e-5
        if is_best:
            best_metric = metric
            best_train_m = last
            best_val_m = val_m
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            best_scaler_state = copy.deepcopy(scaler.state_dict()) if use_amp else None
            best_step = step
            best_epoch_offset = epoch + 1
            if patience > 0:
                patience_left = patience
            tqdm.write(
                f"[rl-train] NEW BEST epoch={epoch} "
                f"{'val' if val_seqs else 'train'}_loss={metric:.4f} "
                f"acc={val_m.policy_acc:.2%}"
            )
        elif patience > 0:
            patience_left -= 1
            tqdm.write(
                f"[rl-train] epoch={epoch} train_loss={last.total_loss:.4f} "
                f"{'val' if val_seqs else 'train'}_loss={metric:.4f} "
                f"acc={val_m.policy_acc:.2%} patience={patience_left}"
            )

        epoch_bar.set_postfix(
            loss=f"{last.total_loss:.3f}",
            p=f"{last.policy_loss:.3f}",
            v=f"{last.value_loss:.3f}",
            aux=(
                "off"
                if cfg.aux_loss_weight == 0
                else f"{last.aux_loss:.3f}/{last.n_archetype_rows}"
            ),
            hand=(
                "off"
                if cfg.opp_hand_loss_weight == 0
                else f"{last.opp_hand_loss:.3f}/{last.n_opp_hand_rows}"
            ),
            rem=(
                "off"
                if cfg.opp_remainder_loss_weight == 0
                else f"{last.opp_remainder_loss:.3f}/"
                f"{last.n_opp_remainder_rows}"
            ),
            lethal=(
                "off"
                if cfg.lethal_threat_loss_weight == 0
                else f"{last.lethal_threat_loss:.3f}/"
                f"{last.n_lethal_threat_rows}"
            ),
            prize=(
                "off"
                if cfg.prize_race_loss_weight == 0
                else f"{last.prize_race_loss:.3f}/"
                f"{last.n_prize_race_rows}"
            ),
            guide=(
                "off"
                if cfg.alakazam_guide_loss_weight == 0
                else f"{last.alakazam_guide_loss:.3f}/"
                f"{last.n_alakazam_guide_rows}"
            ),
            acc=f"{last.policy_acc:.2%}",
            best=f"{best_metric:.3f}",
            pat=patience_left if patience > 0 else "-",
        )

        if patience > 0 and patience_left <= 0:
            tqdm.write(
                f"[rl-early-stop] patience exhausted at epoch={epoch} "
                f"best_loss={best_metric:.4f}"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        if best_optimizer_state is not None:
            optimizer.load_state_dict(best_optimizer_state)
        if use_amp and best_scaler_state is not None:
            scaler.load_state_dict(best_scaler_state)
        step = best_step
        last = best_train_m
        validation_metrics = best_val_m
    else:
        validation_metrics = last

    if resident_corpus is not None:
        candidate_predictions: Union[list[int], torch.Tensor] = (
            _device_exact_policy_predictions(
                model,
                resident_corpus,
                cfg=cfg,
                batch_size=resident_batch_size,
                desc="rl-agreement candidate",
            )
        )
    else:
        candidate_predictions = _policy_argmax_predictions(
            model,
            agreement_sequences,
            cfg=cfg,
            awr_baseline_cache=awr_baseline_cache,
            desc="rl-agreement candidate",
        )
    if isinstance(parent_predictions, torch.Tensor):
        if not isinstance(candidate_predictions, torch.Tensor):
            raise RuntimeError("resident candidate agreement type mismatch")
        parent_rows = int(parent_predictions.numel())
        candidate_rows = int(candidate_predictions.numel())
        if parent_rows <= 0 or parent_rows != candidate_rows:
            raise RuntimeError(
                "cannot measure resident parent/candidate policy agreement: "
                f"parent_rows={parent_rows} candidate_rows={candidate_rows}"
            )
        policy_prev_agreement = float(
            (parent_predictions == candidate_predictions)
            .float()
            .mean()
            .item()
        )
    else:
        if isinstance(candidate_predictions, torch.Tensor):
            raise RuntimeError("host candidate agreement type mismatch")
        parent_rows = len(parent_predictions)
        candidate_rows = len(candidate_predictions)
        if parent_rows <= 0 or parent_rows != candidate_rows:
            raise RuntimeError(
                "cannot measure parent/candidate policy agreement: "
                f"parent_rows={parent_rows} candidate_rows={candidate_rows}"
            )
        policy_prev_agreement = sum(
            int(parent == candidate)
            for parent, candidate in zip(
                parent_predictions, candidate_predictions
            )
        ) / float(parent_rows)
    rl_metrics = dict(last.__dict__)
    rl_metrics["policy_prev_agreement"] = policy_prev_agreement
    rl_metrics["optimizer_samples_per_second"] = last_epoch_optimizer_sps

    delta_sq = 0.0
    base_sq = 0.0
    for name, value in model.state_dict().items():
        current = value.detach().cpu().float()
        base = initial_state[name].float()
        delta_sq += float(torch.sum((current - base) ** 2).item())
        base_sq += float(torch.sum(base ** 2).item())
        if not torch.isfinite(current).all():
            raise FloatingPointError(f"non-finite candidate parameter: {name}")
    update_norm = math.sqrt(delta_sq)
    relative_update_norm = update_norm / max(math.sqrt(base_sq), 1e-12)

    ckpt = checkpoint.build_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler if use_amp else None,
        step=step,
        epoch=int(base_epoch + (best_epoch_offset or stepped_epochs)),
        rl_iteration=int(base_rl_iteration + 1),
        best_metric=best_metric if best_state is not None else last.total_loss,
        early_stop_state={
            "patience_left": patience_left,
            "best_metric": best_metric if best_state is not None else last.total_loss,
        },
        archetype_id=archetype_id,
        model_id=out_run_name,
        model_config=model.cfg,
        extra={
            "pure_rl": bool(cfg.pure_rl),
            "param_count": int(sum(p.numel() for p in model.parameters())),
            "rl_metrics": rl_metrics,
            "validation_metrics": validation_metrics.__dict__,
            "validation_source": best_validation_source,
            "rl_epochs_ran": stepped_epochs,
            "rl_epochs_cap": int(epochs),
            "rl_selected_epoch_offset": int(best_epoch_offset or stepped_epochs),
            "global_step": int(step),
            "optimizer_parent_step": int(base_step),
            "optimizer_state_restored": optimizer_state_restored,
            "awr_baseline_mode": (
                "frozen_device_resident"
                if resident_baseline is not None
                else "frozen_precomputed"
                if awr_baseline_cache is not None
                else "detached_online"
            ),
            "awr_baseline_rows": (
                int(resident_baseline.numel())
                if resident_baseline is not None
                else len(awr_baseline_cache or {})
            ),
            "device_resident_rl": resident_corpus is not None,
            "device_resident_bytes": (
                int(resident_corpus.input_bytes)
                if resident_corpus is not None
                else 0
            ),
            "device_resident_batch_size": (
                int(resident_batch_size)
                if resident_corpus is not None
                else None
            ),
            "device_resident_build_seconds": (
                float(resident_corpus.build_seconds)
                if resident_corpus is not None
                else None
            ),
            "device_resident_samples": (
                int(resident_corpus.total_samples)
                if resident_corpus is not None
                else 0
            ),
            "training_contract": "causal_realized_history",
            "parent_digest": parent_digest,
            "training_provenance": dict(training_provenance or {}),
            "update_norm_l2": update_norm,
            "relative_update_norm_l2": relative_update_norm,
            "policy_prev_agreement": policy_prev_agreement,
            "policy_prev_agreement_rows": parent_rows,
        },
    )
    if output_path is not None:
        # Iteration candidates are lineage evidence. Only an explicit caller
        # override may replace one; pure-RL provenance is never implicit write
        # permission.
        if replace_existing:
            saved_path = checkpoint.atomic_torch_save(ckpt, output_path)
        else:
            saved_path = checkpoint.immutable_torch_save(ckpt, output_path)
    else:
        mgr = checkpoint.CheckpointManager(out_run_name)
        saved = mgr.save(ckpt, is_best=False)
        saved_path = saved.get("latest", checkpoint.latest_path(out_run_name))
    digest = checkpoint.checkpoint_digest(saved_path)
    return {
        "latest_path": str(saved_path),
        "candidate_path": str(saved_path),
        "candidate_digest": digest,
        "parent_digest": parent_digest,
        "metrics": rl_metrics,
        "validation_metrics": validation_metrics.__dict__,
        "validation_source": best_validation_source,
        "step": step,
        "parent_step": base_step,
        "optimizer_state_restored": optimizer_state_restored,
        "rl_iteration": base_rl_iteration + 1,
        "awr_baseline_mode": (
            "frozen_device_resident"
            if resident_baseline is not None
            else "frozen_precomputed"
            if awr_baseline_cache is not None
            else "detached_online"
        ),
        "device_resident_rl": resident_corpus is not None,
        "device_resident_bytes": (
            int(resident_corpus.input_bytes)
            if resident_corpus is not None
            else 0
        ),
        "device_resident_batch_size": (
            int(resident_batch_size) if resident_corpus is not None else None
        ),
        "device_resident_build_seconds": (
            float(resident_corpus.build_seconds)
            if resident_corpus is not None
            else None
        ),
        "device_resident_samples": (
            int(resident_corpus.total_samples)
            if resident_corpus is not None
            else 0
        ),
        "epochs_ran": stepped_epochs,
        "best_metric": best_metric if best_state is not None else last.total_loss,
        "update_norm_l2": update_norm,
        "relative_update_norm_l2": relative_update_norm,
        "policy_prev_agreement": policy_prev_agreement,
        "policy_prev_agreement_rows": parent_rows,
    }


def load_model_from_checkpoint(
    path: Union[str, Path],
    *,
    device: Optional[torch.device] = None,
) -> TemporalCabtTransformer:
    """Reconstruct architecture and load weights with belief-head warm-start.

    Trunk + existing heads load strictly. Only expected new belief/strategy
    head keys (``opp_hand_head.*``, ``opp_remainder_head.*``,
    ``lethal_threat_head.*``, ``prize_race_head.*``) may be missing; those
    heads stay randomly initialized and are recorded on
    ``model.warm_started_belief_heads`` (Scope A card heads trigger uniform
    particle fallback; Scope B strategy heads are root-gated separately).
    """
    device = device or device_mod.inference_device(allow_cpu=True)
    ckpt = checkpoint.load_checkpoint(path, map_location=device)
    snap = ckpt.get("model_config")
    if snap is None:
        cfg = config.MODEL
    elif not isinstance(snap, dict):
        raise ValueError(
            f"checkpoint {path} has invalid model_config type {type(snap).__name__}"
        )
    else:
        known = set(config.ModelConfig.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = sorted(set(snap) - known)
        if unknown:
            raise ValueError(
                f"checkpoint {path} has unsupported model_config fields: {unknown}"
            )
        try:
            cfg = config.ModelConfig(**snap)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"checkpoint {path} has incompatible model_config: {exc}"
            ) from exc

    state = ckpt.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint {path} is missing model_state_dict")
    dense = bool(getattr(cfg, "dense_card2vec", False))
    try:
        aux_classes = int(state["aux_head.3.weight"].shape[0])
        if dense or "card2vec.card_emb.weight" in state:
            # Factorized path: vocab sizes live on card2vec buffers / tables.
            if "card2vec.board_kind" in state:
                encoder_vocab = int(state["card2vec.board_kind"].shape[0])
            else:
                encoder_vocab = int(features.encoder_vocab_size())
            if "card2vec.option_kind" in state:
                decoder_vocab = int(state["card2vec.option_kind"].shape[0])
            else:
                decoder_vocab = int(features.decoder_vocab_size())
            # Ensure cfg matches dense modules even if snap omitted the flag.
            if not dense:
                cfg.dense_card2vec = True
                dense = True
        else:
            encoder_vocab = int(state["board_bag.weight"].shape[0])
            decoder_vocab = int(state["option_bag.weight"].shape[0])
    except (KeyError, AttributeError, IndexError, TypeError) as exc:
        raise ValueError(
            f"checkpoint {path} lacks architecture-defining tensor shapes"
        ) from exc

    belief_card_vocab = belief_card_vocab_from_state(state)

    model = build_model(
        cfg,
        device=device,
        aux_archetype_classes=aux_classes,
        encoder_vocab=encoder_vocab,
        decoder_vocab=decoder_vocab,
        belief_card_vocab=belief_card_vocab,
    )
    incompatible = model.load_state_dict(state, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []) or [])
    unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
    if unexpected:
        raise RuntimeError(
            f"checkpoint architecture/state incompatibility for {path}: "
            f"unexpected keys {unexpected}"
        )
    disallowed_missing = [
        key for key in missing if not is_allowed_missing_belief_head_key(key)
    ]
    if disallowed_missing:
        raise RuntimeError(
            f"checkpoint architecture/state incompatibility for {path}: "
            f"missing non-belief-head keys {disallowed_missing}"
        )
    warm = belief_head_names_from_state_keys(missing)
    model.warm_started_belief_heads = warm
    if warm:
        extra = dict(ckpt.get("extra") or {})
        extra["warm_start"] = True
        extra["warm_started_belief_heads"] = list(warm)
        extra["aux_heads_present"] = list(model.aux_heads_present)
        ckpt["extra"] = extra
    model.eval()
    return model
