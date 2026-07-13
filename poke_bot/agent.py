"""Timed MCTS / greedy policy agents for local play and submission parity.

Info-set only. Fail-closed: illegal / empty selects become a legal random
fallback (or empty list only when no options exist).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch

from . import cg_env, config, features
from .mcts import GameClock, MCTS, MCTSResult
from .model import TemporalCabtTransformer


def _fail_closed_legal(obs_dict: dict, preferred: list[int], rng: random.Random) -> list[int]:
    """Clamp ``preferred`` into a legal select; never invent illegal indices."""
    sel = obs_dict.get("select") if obs_dict else None
    if sel is None:
        return preferred  # deck return path
    options = sel.get("option") or []
    n = len(options)
    if n <= 0:
        return []
    lo = int(sel.get("minCount", 0) or 0)
    hi = min(int(sel.get("maxCount", 0) or 0), n)
    lo = max(0, min(lo, hi))
    # Deduplicate / filter preferred.
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
    # Fallback: sample a legal count.
    if hi <= 0:
        return []
    k = rng.randint(lo, hi) if hi >= lo else hi
    if k <= 0:
        return []
    return rng.sample(range(n), k)


@dataclass
class PolicyAgent:
    """NN agent: greedy argmax or timed MCTS."""

    model: TemporalCabtTransformer
    deck: list[int]
    use_mcts: bool = True
    device: Optional[torch.device] = None
    clock: Optional[GameClock] = None
    max_sims: Optional[int] = None
    move_time_s: Optional[float] = None
    rng: random.Random = field(default_factory=random.Random)
    last_result: Optional[MCTSResult] = None
    collect_targets: bool = False
    targets: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = next(self.model.parameters()).device
        self.model.eval()
        if self.clock is None and self.use_mcts:
            self.clock = GameClock()

    def reset_game(self) -> None:
        if self.use_mcts:
            self.clock = GameClock()
        self.targets.clear()
        self.last_result = None

    @torch.no_grad()
    def greedy_select(self, obs_dict: dict) -> list[int]:
        features.assert_info_set(obs_dict)
        out = self.model.forward_from_obs(
            obs_dict, self.deck, kv_cache=None, append_cache=False, assert_info=True
        )
        combos = out["action_combos"]
        if not combos:
            return []
        logits = out["policy_logits"][0, : len(combos)]
        idx = int(torch.argmax(logits).item())
        return list(combos[idx])

    def mcts_select(self, obs_dict: dict) -> list[int]:
        engine = MCTS(self.model, self.deck, device=self.device)
        result = engine.search(
            obs_dict,
            clock=self.clock,
            max_sims=self.max_sims,
            move_time_s=self.move_time_s,
        )
        self.last_result = result
        if self.collect_targets:
            self.targets.append(
                {
                    "action_combos": result.target.action_combos,
                    "policy": result.target.policy,
                    "value": result.target.value,
                    "visits": result.target.visits,
                    "diagnostics": result.target.diagnostics,
                }
            )
        return list(result.select)

    def __call__(self, obs_dict: dict) -> list[int]:
        """Competition-style agent entry: deck when select is None."""
        if obs_dict is None or obs_dict.get("select") is None:
            return list(self.deck)
        try:
            if self.use_mcts:
                action = self.mcts_select(obs_dict)
            else:
                action = self.greedy_select(obs_dict)
        except Exception:
            action = []
        return _fail_closed_legal(obs_dict, action, self.rng)


AgentFn = Callable[[dict], list[int]]


def play_game(
    agent0: AgentFn,
    agent1: AgentFn,
    deck0: list[int],
    deck1: list[int],
    *,
    max_steps: int = 4000,
) -> dict[str, Any]:
    """Play one local game; returns winner + length."""
    obs, _start = cg_env.battle_start(deck0, deck1)
    steps = 0
    try:
        while obs is not None and not cg_env.is_finished(obs) and steps < max_steps:
            cur = obs.get("current") or {}
            seat = int(cur.get("yourIndex", 0))
            agent = agent0 if seat == 0 else agent1
            select = agent(obs)
            obs = cg_env.battle_select(select)
            steps += 1
        winner = cg_env.result_winner(obs) if obs else 2
        if winner is None:
            winner = 2
        return {"winner": int(winner), "steps": steps}
    finally:
        try:
            cg_env.battle_finish()
        except Exception:
            pass
