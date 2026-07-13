"""Competition submission agent: timed MCTS, fail-closed legal actions.

Hard constraints:
  - No ``__file__`` at import time (isolated tarball / Kaggle).
  - Deck from ``deck.csv`` next to ``main.py`` or ``/kaggle_simulations/agent/``.
  - Info-set only (features.assert_info_set inside MCTS).
  - Fail-closed: illegal selects → legal random fallback.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

# Package-local imports work when cg/ and poke_bot-equivalent code are packed
# flat: we vendor a minimal runtime under submission/ (model.pt + cg + helpers).

_AGENT_DIR_CANDIDATES = (
    Path.cwd(),
    Path("/kaggle_simulations/agent"),
)


def _agent_dir() -> Path:
    for d in _AGENT_DIR_CANDIDATES:
        if (d / "deck.csv").is_file():
            return d
    return Path.cwd()


def _read_deck() -> list[int]:
    path = _agent_dir() / "deck.csv"
    lines = path.read_text().splitlines()
    deck: list[int] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        deck.append(int(line.split(",")[0]))
        if len(deck) >= 60:
            break
    if len(deck) != 60:
        raise ValueError(f"deck.csv must have 60 cards, got {len(deck)}")
    return deck


# Lazy singletons (avoid heavy work / CUDA at import for isolated smoke).
_DECK: list[int] | None = None
_MODEL = None
_CLOCK = None
_RNG = random.Random(0)


def _ensure_runtime():
    global _DECK, _MODEL, _CLOCK
    if _DECK is None:
        _DECK = _read_deck()
    if _MODEL is None:
        # Prefer packed local modules (submission tree).
        agent_dir = str(_agent_dir())
        if agent_dir not in sys.path:
            sys.path.insert(0, agent_dir)
        import torch
        from poke_bot.checkpoint import load_checkpoint, apply_checkpoint
        from poke_bot.model import build_model
        from poke_bot.mcts import GameClock

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(device=device)
        ckpt_path = _agent_dir() / "model.pt"
        if ckpt_path.is_file():
            ckpt = load_checkpoint(ckpt_path, map_location=device)
            apply_checkpoint(ckpt, model=model, restore_rng=False)
        model.eval()
        _MODEL = model
        _CLOCK = GameClock()
    return _DECK, _MODEL, _CLOCK


def _fail_closed(obs_dict: dict, preferred: list[int]) -> list[int]:
    sel = obs_dict.get("select") if obs_dict else None
    if sel is None:
        return preferred
    n = len(sel.get("option") or [])
    if n <= 0:
        return []
    lo = int(sel.get("minCount", 0) or 0)
    hi = min(int(sel.get("maxCount", 0) or 0), n)
    lo = max(0, min(lo, hi))
    clean: list[int] = []
    for x in preferred:
        try:
            xi = int(x)
        except (TypeError, ValueError):
            continue
        if 0 <= xi < n and xi not in clean:
            clean.append(xi)
    if lo <= len(clean) <= hi and clean:
        return clean[:hi]
    if hi <= 0:
        return []
    k = _RNG.randint(lo, hi) if hi >= lo else hi
    return _RNG.sample(range(n), k) if k > 0 else []


def agent(obs_dict: dict) -> list[int]:
    """Kaggle entry point."""
    from cg.api import to_observation_class

    deck, model, clock = _ensure_runtime()
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return list(deck)

    try:
        from poke_bot.mcts import MCTS

        # Remaining budget tight → greedy.
        use_mcts = clock is not None and clock.remaining_s > 1.0
        if use_mcts:
            result = MCTS(model, deck).search(obs_dict, clock=clock)
            action = list(result.select)
        else:
            from poke_bot.agent import PolicyAgent

            action = PolicyAgent(model=model, deck=deck, use_mcts=False).greedy_select(
                obs_dict
            )
    except Exception:
        action = []
    return _fail_closed(obs_dict, action)
