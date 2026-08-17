"""Attach / zero heuristic surrogate channels for offline training only."""

from __future__ import annotations

from typing import Any

from poke_bot.slowking_reverse_engineered_policy import audit_decision

from .authority import RESEARCH_ONLY


def attach_heuristic_features(
    row: dict[str, Any],
    *,
    zero_channel: bool = False,
) -> dict[str, Any]:
    """Return a copy with frozen heuristic teacher metadata.

    ``zero_channel=True`` is the mandatory ablation: scores/rule ids cleared so
    the actor cannot depend on the surrogate at train or eval time.
    """
    out = dict(row)
    out["research_only"] = RESEARCH_ONLY
    if zero_channel:
        out["heuristic"] = {
            "stage_class": None,
            "scores": [0.0] * int(out.get("legal_action_count") or 0),
            "preferred_combo_index": None,
            "margin": 0.0,
            "combo_rule_ids": [],
            "abstained": True,
            "zeroed_ablation": True,
        }
        out["heuristic_abstained"] = True
        out["heuristic_channel_zeroed"] = True
        return out

    if out.get("heuristic") is not None and not out.get("heuristic_abstained"):
        out["heuristic_channel_zeroed"] = False
        return out

    obs = out.get("observation")
    legal = out.get("legal_action_combos") or []
    deck = out.get("deck") or []
    if not isinstance(obs, dict) or not legal:
        out["heuristic"] = None
        out["heuristic_abstained"] = True
        out["heuristic_channel_zeroed"] = False
        return out
    audit = audit_decision(obs, legal, deck=deck)
    if audit is None:
        out["heuristic"] = None
        out["heuristic_abstained"] = True
    else:
        out["heuristic"] = {
            "stage_class": audit.get("stage_class"),
            "scores": audit.get("scores"),
            "preferred_combo_index": audit.get("preferred_combo_index"),
            "margin": audit.get("margin"),
            "combo_rule_ids": audit.get("combo_rule_ids"),
            "abstained": False,
            "zeroed_ablation": False,
        }
        out["heuristic_abstained"] = False
    out["heuristic_channel_zeroed"] = False
    return out


def heuristic_mask(row: dict[str, Any]) -> list[float]:
    """Per-option confidence mask in [0, 1]; zeros when abstaining/ablated."""
    n = int(row.get("legal_action_count") or 0)
    if row.get("heuristic_channel_zeroed") or row.get("heuristic_abstained"):
        return [0.0] * n
    heuristic = row.get("heuristic") or {}
    scores = list(heuristic.get("scores") or [])
    if len(scores) != n or n == 0:
        return [0.0] * n
    preferred = heuristic.get("preferred_combo_index")
    mask = [0.0] * n
    if isinstance(preferred, int) and 0 <= preferred < n:
        mask[preferred] = 1.0
    return mask
