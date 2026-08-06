"""Paired evaluation gate — wins required; replay agreement is not enough."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from .authority import EVAL_GATE_SCHEMA, RESEARCH_ONLY


PlayFn = Callable[[dict[str, Any]], Sequence[int]]


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


@dataclass
class EvalGateConfig:
    min_games: int = 64
    min_win_rate: float = 0.55
    require_training_ineligible: bool = True
    # Replay agreement alone never promotes.
    max_agreement_only_promotion: bool = False


@dataclass
class EvalGateResult:
    passed: bool
    metrics: dict[str, Any]
    reasons: list[str]


def evaluate_paired_games(
    games: list[dict[str, Any]],
    *,
    config: Optional[EvalGateConfig] = None,
    action_agreement: Optional[float] = None,
) -> EvalGateResult:
    """Judge a fresh paired-game set.

    Each game dict needs: ``result`` in {win,loss,draw}, and ideally
    ``training_ineligible`` bool, ``seat_order``, ``opponent_id``.
    """
    cfg = config or EvalGateConfig()
    reasons: list[str] = []
    n = len(games)
    wins = sum(1 for g in games if g.get("result") == "win")
    losses = sum(1 for g in games if g.get("result") == "loss")
    draws = n - wins - losses
    wr = wins / n if n else 0.0
    lo, hi = wilson_interval(wins, n)

    if cfg.require_training_ineligible:
        ineligible = all(bool(g.get("training_ineligible", True)) for g in games)
        if not ineligible:
            reasons.append("games_not_marked_training_ineligible")

    if n < cfg.min_games:
        reasons.append(f"insufficient_games:{n}<{cfg.min_games}")
    if wr < cfg.min_win_rate:
        reasons.append(f"win_rate_below_threshold:{wr:.4f}<{cfg.min_win_rate}")

    # Explicitly reject agreement-only promotion attempts.
    if action_agreement is not None and not games:
        reasons.append("agreement_without_paired_games")
    if cfg.max_agreement_only_promotion is False and not games and action_agreement:
        reasons.append("replay_agreement_cannot_promote")

    passed = not reasons
    metrics = {
        "schema": EVAL_GATE_SCHEMA,
        "research_only": RESEARCH_ONLY,
        "training_authority": False,
        "serving_authority": False,
        "n_games": n,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wr,
        "wilson_95": [lo, hi],
        "action_agreement": action_agreement,
        "passed": passed,
        "reasons": reasons,
    }
    return EvalGateResult(passed=passed, metrics=metrics, reasons=reasons)


def write_eval_receipt(result: EvalGateResult, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.metrics, indent=2) + "\n", encoding="utf-8")
