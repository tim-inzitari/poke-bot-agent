"""Hammer-Pult soft priors (Phase 2 stubs).

These bias MCTS/policy priors toward the Hammer-Pult game plan described in the
PDF guide: Crushing Hammer energy denial, Munkidori damage shuffling, Budew item
lock, Unfair Stamp hand disruption, and closing with the Dragapult "Phantom
Dive" spread. They apply **only** to the ``hammer-pult`` archetype (other
variants get their own priors later).

This is intentionally a *stub*: :func:`prior_logit_bias` returns a per-action
additive logit bias, currently all zeros (neutral), with the plumbing and
card/option hooks in place so a later phase can fill in the actual bias values
without changing the call sites. Enable experimentation via ``ENABLED``.
"""

from __future__ import annotations

from typing import Optional

from . import archetypes, cg_env

#: Master switch. Left off so the foundation contributes no (unvalidated) bias.
ENABLED: bool = False

#: Card ids central to the Hammer-Pult plan (mirror archetypes.py signature).
KEY_DISRUPTION = {
    archetypes.CRUSHING_HAMMER,  # energy denial
    archetypes.UNFAIR_STAMP,     # hand disruption
    archetypes.BUDEW,            # item lock (Itchy Pollen)
    *archetypes.MUNKIDORI,       # damage shuffle
}


def applies(deck_card_ids) -> bool:
    """True only when the deck matches the Hammer-Pult signature."""
    return archetypes.is_hammer_signature(deck_card_ids)


def prior_logit_bias(
    obs,
    action_combos: list[list[int]],
    *,
    scale: float = 1.0,
) -> list[float]:
    """Return an additive logit bias per action combo (same length/order).

    Stub contract (stable for later phases):
      - length == len(action_combos)
      - 0.0 == neutral; positive encourages, negative discourages an action.
      - Only non-zero when :data:`ENABLED` is True.
    """
    bias = [0.0] * len(action_combos)
    if not ENABLED:
        return bias

    if isinstance(obs, dict):
        obs = cg_env.to_observation(obs)
    OptionType = cg_env.OptionType
    sel = obs.select
    if sel is None:
        return bias

    # --- Illustrative hooks (values are placeholders pending PDF calibration) ---
    for i, combo in enumerate(action_combos):
        for opt_idx in combo:
            o = sel.option[opt_idx]
            if o.type == OptionType.PLAY:
                # Encourage deploying disruption items when in hand.
                pass  # TODO(phase2): weight by Crushing Hammer / Unfair Stamp / Budew
            elif o.type == OptionType.ATTACK:
                # Encourage the Dragapult spread close.
                pass  # TODO(phase2): weight Phantom Dive by lethality
    return bias


def describe() -> str:
    return f"HammerHeuristics(enabled={ENABLED}, key_cards={sorted(KEY_DISRUPTION)})"
