"""Training-only detection and labeling for legal no-progress game loops."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any


STALL_TRAINING_RETURN = -1.0
DEFAULT_MAX_STAGNANT_TURNS = 64


def configured_max_stagnant_turns(
    job: dict[str, Any] | None = None,
) -> int | None:
    """Return the explicit training-only stall limit, or ``None`` when off.

    The clean-boundary drop-in owns activation through
    ``PURE_RL_NO_PROGRESS_MAX_TURNS``.  A controller-stamped job value may
    narrow that configured limit, but merely loading the newer source must
    preserve the historical disabled path exactly.
    """

    raw: Any = None
    if job is not None:
        raw = job.get("max_no_progress_turns")
    if raw is None or str(raw).strip() == "":
        raw = os.environ.get("PURE_RL_NO_PROGRESS_MAX_TURNS", "0")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _pokemon_signature(card: Any) -> tuple[Any, ...] | None:
    if not isinstance(card, dict):
        return None
    return (
        card.get("id"),
        card.get("hp"),
        card.get("maxHp"),
        tuple(sorted(str(value) for value in (card.get("energies") or []))),
        tuple(sorted(str(value) for value in (card.get("energyCards") or []))),
    )


def win_progress_signature(observation: dict[str, Any]) -> tuple[Any, ...]:
    """Track public prize/damage/KO/board-energy progress, not zone churn."""

    current = dict(observation.get("current") or {})
    rows: list[tuple[Any, ...]] = []
    for raw_player in current.get("players") or []:
        player = raw_player if isinstance(raw_player, dict) else {}
        active = tuple(
            value
            for value in (
                _pokemon_signature(card) for card in (player.get("active") or [])
            )
            if value is not None
        )
        bench = tuple(
            value
            for value in (
                _pokemon_signature(card) for card in (player.get("bench") or [])
            )
            if value is not None
        )
        rows.append(
            (
                len(player.get("prize") or []),
                active,
                bench,
                bool(player.get("poisoned")),
                bool(player.get("burned")),
                bool(player.get("asleep")),
                bool(player.get("paralyzed")),
                bool(player.get("confused")),
            )
        )
    return tuple(rows)


@dataclass
class NoProgressTracker:
    """Detect N complete turns without any public win-relevant change."""

    max_stagnant_turns: int = DEFAULT_MAX_STAGNANT_TURNS
    last_seen_turn: int | None = None
    last_progress_turn: int | None = None
    last_signature: tuple[Any, ...] | None = None

    def __post_init__(self) -> None:
        if int(self.max_stagnant_turns) < 1:
            raise ValueError("max_stagnant_turns must be positive")
        self.max_stagnant_turns = int(self.max_stagnant_turns)

    def observe(self, observation: dict[str, Any]) -> bool:
        current = dict(observation.get("current") or {})
        if int(current.get("result", -1)) != -1:
            return False
        turn = int(current.get("turn", 0) or 0)
        if turn == self.last_seen_turn:
            return False
        signature = win_progress_signature(observation)
        if signature != self.last_signature:
            self.last_signature = signature
            self.last_progress_turn = turn
        self.last_seen_turn = turn
        return bool(
            self.last_progress_turn is not None
            and turn - self.last_progress_turn >= self.max_stagnant_turns
        )

    @property
    def stagnant_turns(self) -> int:
        if self.last_seen_turn is None or self.last_progress_turn is None:
            return 0
        return max(0, self.last_seen_turn - self.last_progress_turn)
