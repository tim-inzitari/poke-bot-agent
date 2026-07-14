"""Blackwell Hammer specialist strategy heads (Scope B).

Scope contrast
--------------
* **Scope A (core / deck-agnostic, 3080 Ti / small):** belief particle priors
  only (``opp_hand_head``, ``opp_remainder_head``, wired ``aux_head``) plus
  exact own-prize belief. No prize-map / lethal / specialist strategy heads.
* **Scope B (Blackwell archetype-specific, Hammer-pult large on GPU1):** all of
  Scope A, plus ``lethal_threat_head`` and ``prize_race_head`` (prize-map /
  KO-path scaffold). Train/deploy when the run is the Blackwell Hammer
  specialist; core_kernel keeps loss weights at 0 and never requires these
  heads for a successful train step.

Heads read info-set ``state_vec`` only. They are **never** written into
``features.build_board_tokens``. At inference, optional root-only value bias
(see :func:`root_value_bias_from_lethal`); not forced onto tiny core_kernel.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F

# Distinct named Scope-B heads — warm-start may omit only these (plus Scope A).
BLACKWELL_STRATEGY_HEAD_PREFIXES: tuple[str, ...] = (
    "lethal_threat_head.",
    "prize_race_head.",
)

BLACKWELL_STRATEGY_HEAD_NAMES: tuple[str, ...] = (
    "lethal_threat_head",
    "prize_race_head",
)

# Post-hoc horizon (# later own decision frames) for "can take a prize soon".
DEFAULT_LETHAL_HORIZON = 8
# Standard PTCG prize count for normalization.
_PRIZE_NORM = 6.0


def blackwell_strategy_heads_enabled(
    *,
    primary_archetype: Optional[str] = None,
    gpu_profile: Optional[str] = None,
) -> bool:
    """Gate Scope-B *deploy/search* usage for the Hammer Blackwell specialist.

    Explicit ``POKEBOT_BLACKWELL_STRATEGY_HEADS=0/1`` always wins. Otherwise
    enable only when primary archetype is hammer-pult **and**
    ``POKEBOT_GPU_PROFILE`` / ``gpu_profile`` is Blackwell. Core / 3080ti
    profiles stay off even if ``POKEBOT_PRIMARY_ARCHETYPE=hammer-pult``.

    Training is gated separately via loss weights (0 on core_kernel /
    bootstrap; non-zero on round-robin Blackwell Hammer CLI defaults).
    """
    env = os.environ.get("POKEBOT_BLACKWELL_STRATEGY_HEADS")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")

    arch = (
        primary_archetype
        or os.environ.get("POKEBOT_PRIMARY_ARCHETYPE")
        or ""
    ).strip().lower()
    prof = (
        gpu_profile
        or os.environ.get("POKEBOT_GPU_PROFILE")
        or ""
    ).strip().lower()
    is_hammer = arch in ("hammer-pult", "hammer_pult", "hammer")
    is_blackwell = prof in ("blackwell", "5000", "pro5000", "rtx_pro_5000")
    return bool(is_hammer and is_blackwell)


def is_allowed_missing_blackwell_head_key(key: str) -> bool:
    return any(key.startswith(p) for p in BLACKWELL_STRATEGY_HEAD_PREFIXES)


def _player_prize_count(player: Any) -> Optional[int]:
    if not isinstance(player, dict):
        return None
    prizes = player.get("prize")
    if prizes is None:
        # Some dumps only expose a count.
        for key in ("prizeCount", "prize_count", "remainingPrizes"):
            if key in player and player[key] is not None:
                return int(player[key])
        return None
    if isinstance(prizes, list):
        return len(prizes)
    return None


def prize_counts_from_obs(obs: dict[str, Any]) -> tuple[Optional[int], Optional[int]]:
    """Return ``(own_prizes_remaining, opp_prizes_remaining)`` from public board."""
    current = obs.get("current") if isinstance(obs, dict) else None
    if not isinstance(current, dict):
        return None, None
    players = current.get("players") or []
    if len(players) < 2:
        return None, None
    your_index = int(current.get("yourIndex", 0))
    opp_index = 1 - your_index
    own_n = _player_prize_count(players[your_index])
    opp_n = _player_prize_count(players[opp_index])
    return own_n, opp_n


def lethal_threat_label_from_trajectory(
    own_prize_counts: Sequence[Optional[int]],
    index: int,
    *,
    horizon: int = DEFAULT_LETHAL_HORIZON,
) -> Optional[float]:
    """Post-hoc binary label: own prize count decreases within ``horizon`` steps.

    Honest construction note — this is **outcome supervision from the recorded
    self-play/sim trajectory**, not a public damage calculator. It approximates
    "a KO / prize-take was available along the line that was played" (KO-path
    / can-take-prizes-soon). Mask (``None``) when the current own count is
    unknown.
    """
    if index < 0 or index >= len(own_prize_counts):
        return None
    cur = own_prize_counts[index]
    if cur is None:
        return None
    end = min(len(own_prize_counts), index + 1 + max(0, int(horizon)))
    for j in range(index + 1, end):
        nxt = own_prize_counts[j]
        if nxt is not None and int(nxt) < int(cur):
            return 1.0
    return 0.0


def attach_blackwell_strategy_labels(
    steps: list[dict[str, Any]],
    *,
    horizon: int = DEFAULT_LETHAL_HORIZON,
) -> None:
    """Mutate each step's ``aux_labels`` with Scope-B targets when constructible.

    Writes:
    * ``prize_race`` — ``[own/6, opp/6]`` from **public** prize counts
    * ``lethal_threat`` — float 0/1 post-hoc prize-take within ``horizon``
      later own decision frames (privileged trajectory / sim dump)
    """
    if not steps:
        return
    own_counts: list[Optional[int]] = []
    opp_counts: list[Optional[int]] = []
    for step in steps:
        obs = step.get("observation") or {}
        own_n, opp_n = prize_counts_from_obs(obs)
        own_counts.append(own_n)
        opp_counts.append(opp_n)

    for i, step in enumerate(steps):
        aux = dict(step.get("aux_labels") or {})
        own_n = own_counts[i]
        opp_n = opp_counts[i]
        if own_n is not None and opp_n is not None:
            aux["prize_race"] = [
                float(own_n) / _PRIZE_NORM,
                float(opp_n) / _PRIZE_NORM,
            ]
        lethal = lethal_threat_label_from_trajectory(
            own_counts, i, horizon=horizon
        )
        if lethal is not None:
            aux["lethal_threat"] = float(lethal)
        step["aux_labels"] = aux


def lethal_target_from_aux(aux: dict[str, Any]) -> Optional[float]:
    raw = aux.get("lethal_threat")
    if raw is None:
        return None
    return float(raw)


def prize_race_target_from_aux(
    aux: dict[str, Any],
    *,
    device: torch.device,
) -> Optional[torch.Tensor]:
    raw = aux.get("prize_race")
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return torch.tensor(
            [float(raw[0]), float(raw[1])],
            device=device,
            dtype=torch.float32,
        )
    return None


def masked_bce_logit(
    logits: torch.Tensor,
    target: Optional[torch.Tensor],
) -> torch.Tensor:
    """BCE-with-logits; zero scalar when ``target`` is absent (masked)."""
    if target is None:
        return logits.sum() * 0.0
    if target.shape != logits.shape:
        raise ValueError(
            "lethal target shape mismatch: "
            f"logits={tuple(logits.shape)} target={tuple(target.shape)}"
        )
    return F.binary_cross_entropy_with_logits(logits, target)


def masked_smooth_l1(
    pred: torch.Tensor,
    target: Optional[torch.Tensor],
) -> torch.Tensor:
    """Smooth-L1; zero scalar when ``target`` is absent (masked)."""
    if target is None:
        return pred.sum() * 0.0
    if target.shape != pred.shape:
        raise ValueError(
            "prize_race target shape mismatch: "
            f"pred={tuple(pred.shape)} target={tuple(target.shape)}"
        )
    return F.smooth_l1_loss(pred, target)


def root_value_bias_from_lethal(
    lethal_logit: Optional[torch.Tensor],
    *,
    scale: float = 0.05,
) -> float:
    """Optional **root-only** value nudge from lethal head (not board features).

    Documented deploy choice for v1: lightly bias the root value toward wins
    when P(take prize soon) is high. Does not rewrite policy logits or bags.
    Returns 0 when disabled / missing.
    """
    if not blackwell_strategy_heads_enabled():
        return 0.0
    if lethal_logit is None:
        return 0.0
    if float(scale) == 0.0:
        return 0.0
    p = float(torch.sigmoid(lethal_logit.detach().float().reshape(-1)[0]).item())
    # Center around 0.5 so uncertain lethal ≈ no bias.
    return float(scale) * (2.0 * p - 1.0)
