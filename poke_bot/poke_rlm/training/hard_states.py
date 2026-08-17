"""Hard-state curriculum helpers for recursive-plan training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass
class HardStateExample:
    """One difficult turn retained for oversampling."""

    battle_id: str
    turn_index: int
    reason: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class HardStateCurriculum:
    """Keeps a bounded set of hard turns ranked by score."""

    capacity: int = 256
    examples: list[HardStateExample] = field(default_factory=list)

    def consider(
        self,
        *,
        battle_id: str,
        turn_index: int,
        reason: str,
        score: float,
        payload: dict[str, Any] | None = None,
    ) -> bool:
        """Insert if capacity allows or score beats the weakest retained example."""
        example = HardStateExample(
            battle_id=str(battle_id),
            turn_index=int(turn_index),
            reason=str(reason),
            score=float(score),
            payload=dict(payload or {}),
        )
        if len(self.examples) < self.capacity:
            self.examples.append(example)
            return True
        weakest_i = min(range(len(self.examples)), key=lambda i: self.examples[i].score)
        if example.score <= self.examples[weakest_i].score:
            return False
        self.examples[weakest_i] = example
        return True

    def sample_ids(self, n: int) -> list[tuple[str, int]]:
        """Return up to n (battle_id, turn_index) pairs by descending score."""
        ranked = sorted(self.examples, key=lambda e: e.score, reverse=True)
        return [(e.battle_id, e.turn_index) for e in ranked[: max(0, int(n))]]

    def extend(self, items: Iterable[HardStateExample]) -> None:
        for item in items:
            self.consider(
                battle_id=item.battle_id,
                turn_index=item.turn_index,
                reason=item.reason,
                score=item.score,
                payload=item.payload,
            )
