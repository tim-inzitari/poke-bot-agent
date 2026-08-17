"""Explicit, package-local guide authority for bounded Kaggle experiments.

This module is inert unless an immutable ``guide_policy.json`` is packaged
beside ``main.py``.  It intentionally does not change ordinary training or
serving.  The experiment applies a small normalized guide bonus to model
log-probabilities and falls back exactly to the model when the guide is absent.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "poke_bot.submission_guide_decision_policy/v1"
_MODE = "guide_logit_bonus"


def _package_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "guide_policy.json"


def load_config(path: Path | None = None) -> dict[str, Any] | None:
    config_path = _package_config_path() if path is None else Path(path)
    if not config_path.is_file():
        return None
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise RuntimeError("packaged guide decision policy schema changed")
    mode = str(payload.get("mode") or "")
    if mode != _MODE:
        raise RuntimeError("packaged guide decision mode changed")
    if payload.get("guide_id") != "alakazam":
        raise RuntimeError("packaged guide identity changed")
    if payload.get("rtp_enabled") is not False:
        raise RuntimeError("guide experiment must remain NO RTP")
    weight = float(payload.get("guide_logit_weight") or 0.0)
    if not (0.0 < weight <= 0.5):
        raise RuntimeError("guide logit weight is outside its bounded range")
    return payload


def _runtime_scores(
    observation: dict[str, Any],
    candidates: Sequence[Sequence[int]],
    deck: Sequence[int],
) -> list[float] | None:
    from . import deck_guides

    names = (
        "POKEBOT_CURRENT_DECK_GUIDE",
        "POKEBOT_CURRENT_DECK_GUIDE_TARGETS",
        "POKEBOT_ALAKAZAM_GUIDE_TARGETS",
    )
    prior = {name: os.environ.get(name) for name in names}
    os.environ["POKEBOT_CURRENT_DECK_GUIDE"] = "alakazam"
    os.environ["POKEBOT_CURRENT_DECK_GUIDE_TARGETS"] = "1"
    os.environ["POKEBOT_ALAKAZAM_GUIDE_TARGETS"] = "1"
    try:
        return deck_guides.guide_scores(observation, candidates, deck=deck)
    finally:
        for name, value in prior.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def select_index(
    *,
    observation: dict[str, Any],
    candidates: Sequence[Sequence[int]],
    model_policy: Sequence[float],
    model_index: int,
    deck: Sequence[int],
    config: Mapping[str, Any] | None = None,
    scorer: Callable[
        [dict[str, Any], Sequence[Sequence[int]], Sequence[int]],
        list[float] | None,
    ]
    | None = None,
) -> tuple[int, dict[str, Any]]:
    """Return the exact stage choice and an audit record.

    Missing, malformed, non-finite, flat, or tied guide output always falls
    back to the model.  The weighted arm adds at most ``guide_logit_weight``
    to any option after min/max normalization; it never mutates model weights.
    """

    resolved = dict(config) if config is not None else load_config()
    base = {
        "schema": SCHEMA,
        "model_index": int(model_index),
        "guide_available": False,
        "guide_selected": False,
        "reason": "guide_policy_not_packaged",
    }
    if resolved is None:
        return int(model_index), base
    mode = str(resolved["mode"])
    base.update(
        {
            "mode": mode,
            "guide_id": "alakazam",
            "guide_logit_weight": float(resolved.get("guide_logit_weight") or 0.0),
        }
    )
    try:
        raw_scores = (scorer or _runtime_scores)(observation, candidates, deck)
    except (ImportError, RuntimeError, ValueError, TypeError, KeyError, IndexError):
        base["reason"] = "guide_scoring_failed"
        return int(model_index), base
    if raw_scores is None or len(raw_scores) != len(candidates):
        base["reason"] = "guide_unavailable"
        return int(model_index), base
    try:
        scores = [float(value) for value in raw_scores]
    except (TypeError, ValueError):
        base["reason"] = "guide_scores_invalid"
        return int(model_index), base
    if not scores or any(not math.isfinite(value) for value in scores):
        base["reason"] = "guide_scores_nonfinite"
        return int(model_index), base
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    margin = scores[order[0]] - scores[order[1]] if len(order) > 1 else math.inf
    minimum_margin = float(resolved.get("minimum_unique_margin") or 1e-8)
    if margin < minimum_margin:
        base["reason"] = "guide_not_unique"
        return int(model_index), base
    base.update(
        {
            "guide_available": True,
            "guide_index": int(order[0]),
            "guide_margin": margin,
            "guide_scores": scores,
        }
    )
    low, high = min(scores), max(scores)
    if high <= low:
        base["reason"] = "guide_scores_flat"
        return int(model_index), base
    if len(model_policy) != len(candidates):
        base["reason"] = "model_policy_width_mismatch"
        return int(model_index), base
    weight = float(resolved["guide_logit_weight"])
    normalized = [(value - low) / (high - low) for value in scores]
    adjusted = [
        math.log(max(float(probability), 1e-12)) + weight * guide_value
        for probability, guide_value in zip(model_policy, normalized, strict=True)
    ]
    selected = max(range(len(adjusted)), key=adjusted.__getitem__)
    base.update(
        {
            "guide_selected": selected != int(model_index),
            "selected_index": int(selected),
            "normalized_guide_scores": normalized,
            "adjusted_log_scores": adjusted,
            "reason": "bounded_guide_logit_bonus",
        }
    )
    return int(selected), base


__all__ = ["SCHEMA", "load_config", "select_index"]
