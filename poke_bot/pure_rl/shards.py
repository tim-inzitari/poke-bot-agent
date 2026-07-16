"""Compact pure-RL shard IO: selected actions + masks, not soft behavior π."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


@dataclass
class CompactDecision:
    """One training decision without soft policy clone targets."""

    env_step: int
    selected_index: int
    n_options: int
    action: list[int]
    observation: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompactGame:
    episode_id: str
    seat: int
    archetype: str
    opp_archetype: str
    deck: list[int]
    value: float
    decisions: list[CompactDecision] = field(default_factory=list)
    source: str = "pure_rl"
    target_provenance: dict[str, Any] = field(default_factory=dict)


def compact_decision_from_step(
    step: dict[str, Any],
    *,
    selected_index: Optional[int] = None,
) -> CompactDecision:
    """Build a compact decision; drops soft ``policy`` rows if present."""
    action = [int(x) for x in (step.get("action") or [])]
    n_options = int(step.get("legal_action_count") or step.get("n_options") or 0)
    idx = selected_index
    if idx is None:
        raw = step.get("selected_index")
        if raw is not None:
            idx = int(raw)
        else:
            # Factorized stage row may carry selected_index.
            stages = step.get("factorized_policy_targets") or step.get("policy_stages")
            if isinstance(stages, list) and stages:
                first = stages[0] if not isinstance(stages[0], list) else (
                    stages[0][0] if stages[0] else {}
                )
                if isinstance(first, dict) and first.get("selected_index") is not None:
                    idx = int(first["selected_index"])
    if idx is None:
        idx = 0
    if n_options <= 0:
        n_options = max(idx + 1, 1)
    return CompactDecision(
        env_step=int(step.get("env_step") or step.get("step") or 0),
        selected_index=int(idx),
        n_options=int(n_options),
        action=action,
        observation=dict(step.get("observation") or {}),
    )


def game_to_jsonable(game: CompactGame) -> dict[str, Any]:
    return {
        "episode_id": game.episode_id,
        "seat": game.seat,
        "archetype": game.archetype,
        "opp_archetype": game.opp_archetype,
        "deck": list(game.deck),
        "value": float(game.value),
        "source": game.source,
        "target_provenance": {
            **dict(game.target_provenance),
            "pure_rl": True,
            "soft_policy_targets": False,
        },
        "decisions": [
            {
                "env_step": d.env_step,
                "selected_index": d.selected_index,
                "n_options": d.n_options,
                "action": list(d.action),
                "observation": d.observation,
            }
            for d in game.decisions
        ],
    }


class CompactShardWriter:
    """Append compact games to a JSONL shard (worker-safe via one writer)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._n_games = 0
        self._n_decisions = 0
        self._t0 = time.time()

    def write_game(self, game: CompactGame) -> None:
        line = json.dumps(game_to_jsonable(game), separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        self._n_games += 1
        self._n_decisions += len(game.decisions)

    def write_games(self, games: Iterable[CompactGame]) -> None:
        for game in games:
            self.write_game(game)

    @property
    def n_games(self) -> int:
        return self._n_games

    @property
    def n_decisions(self) -> int:
        return self._n_decisions

    def throughput(self) -> dict[str, float]:
        elapsed = max(time.time() - self._t0, 1e-6)
        return {
            "games_per_sec": self._n_games / elapsed,
            "decisions_per_sec": self._n_decisions / elapsed,
            "elapsed_sec": elapsed,
        }


def iter_shard_games(path: Path) -> Iterator[CompactGame]:
    path = Path(path)
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            decisions = [
                CompactDecision(
                    env_step=int(d.get("env_step") or 0),
                    selected_index=int(d.get("selected_index") or 0),
                    n_options=int(d.get("n_options") or 1),
                    action=[int(x) for x in (d.get("action") or [])],
                    observation=dict(d.get("observation") or {}),
                )
                for d in (raw.get("decisions") or [])
            ]
            yield CompactGame(
                episode_id=str(raw.get("episode_id") or ""),
                seat=int(raw.get("seat") or 0),
                archetype=str(raw.get("archetype") or ""),
                opp_archetype=str(raw.get("opp_archetype") or ""),
                deck=[int(x) for x in (raw.get("deck") or [])],
                value=float(raw.get("value") or 0.0),
                decisions=decisions,
                source=str(raw.get("source") or "pure_rl"),
                target_provenance=dict(raw.get("target_provenance") or {}),
            )
