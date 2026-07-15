"""Archetype-keyed soft-prior heuristics (SME-PDF conversion target).

Each archetype module exposes a stable surface so subject-matter expert
guides (large PDFs) can later be converted by AI models into bias tables
without changing BeliefMCTS call sites.

Required surface per module
---------------------------
* ``ENABLED: bool`` — off until human review of converted content
* ``applies(deck_card_ids) -> bool``
* ``prior_logit_bias(obs, action_combos, *, scale=1.0) -> list[float]``
* ``describe() -> str``

See ``docs/HEURISTIC_CONVERSION.md`` for the PDF → module conversion contract.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, Optional, Sequence

from . import archetypes, hammer_heuristics

#: Archetype id → heuristics module. Extend here when adding new SME modules.
_REGISTRY: dict[str, ModuleType] = {
    "hammer-pult": hammer_heuristics,
}


class _NeutralHeuristics:
    """No-op heuristics for unknown / unregistered archetypes."""

    ENABLED = False

    @staticmethod
    def applies(_deck_card_ids: Any) -> bool:
        return False

    @staticmethod
    def prior_logit_bias(
        _obs: Any,
        action_combos: list,
        *,
        scale: float = 1.0,
    ) -> list[float]:
        del scale
        return [0.0] * len(action_combos)

    @staticmethod
    def describe() -> str:
        return "NeutralHeuristics(enabled=False)"


_NEUTRAL = _NeutralHeuristics()


def register(archetype_id: str, module: ModuleType) -> None:
    """Register or replace an archetype heuristics module."""
    for name in ("ENABLED", "applies", "prior_logit_bias", "describe"):
        if not hasattr(module, name):
            raise ValueError(
                f"heuristics module for {archetype_id!r} missing {name}"
            )
    _REGISTRY[str(archetype_id)] = module


def resolve_heuristics(
    *,
    archetype_id: Optional[str] = None,
    deck_card_ids: Optional[Sequence[int]] = None,
) -> Any:
    """Return the heuristics module for an archetype or deck signature."""
    arch = archetype_id
    if arch is None and deck_card_ids is not None:
        arch = archetypes.classify_deck(deck_card_ids)
    if arch is None:
        return _NEUTRAL
    module = _REGISTRY.get(str(arch))
    if module is None:
        return _NEUTRAL
    if deck_card_ids is not None and hasattr(module, "applies"):
        try:
            if not module.applies(deck_card_ids):
                return _NEUTRAL
        except Exception:
            return _NEUTRAL
    return module


def apply_prior_logit_bias(
    priors: Sequence[float],
    bias: Sequence[float],
) -> list[float]:
    """Softmax(log(prior) + bias). Lengths must match; empty → []."""
    import math

    if len(priors) != len(bias):
        raise ValueError("prior/bias length mismatch")
    if not priors:
        return []
    logits = [
        math.log(max(float(p), 1e-12)) + float(b)
        for p, b in zip(priors, bias)
    ]
    peak = max(logits)
    exps = [math.exp(x - peak) for x in logits]
    total = sum(exps)
    if total <= 0:
        n = len(exps)
        return [1.0 / n] * n
    return [e / total for e in exps]


def prior_margin(priors: Sequence[float]) -> float:
    """Top − 2nd prior (1.0 when only one action; 0.0 when empty)."""
    if not priors:
        return 0.0
    ordered = sorted((float(p) for p in priors), reverse=True)
    if len(ordered) == 1:
        return 1.0
    return ordered[0] - ordered[1]


def describe_resolved(
    *,
    archetype_id: Optional[str] = None,
    deck_card_ids: Optional[Sequence[int]] = None,
) -> str:
    module = resolve_heuristics(
        archetype_id=archetype_id, deck_card_ids=deck_card_ids
    )
    arch = archetype_id
    if arch is None and deck_card_ids is not None:
        arch = archetypes.classify_deck(deck_card_ids)
    return f"{arch or 'unknown'}:{module.describe()}"
