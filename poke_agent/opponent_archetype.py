from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from poke_agent.archetypes import ArchetypeRegistry, load_archetype_registry
from poke_agent.game_tracker import GameEventTracker


@dataclass
class OpponentArchetypeClassifier:
    """Runtime opponent deck archetype guess from visible plays + meta priors."""

    registry: ArchetypeRegistry
    visible_cards: Counter[int] = field(default_factory=Counter)
    last_slug: str = "unknown"
    last_confidence: float = 0.0

    @classmethod
    def from_root(cls, root) -> "OpponentArchetypeClassifier":
        from pathlib import Path

        return cls(load_archetype_registry(Path(root)))

    def reset(self) -> None:
        self.visible_cards.clear()
        self.last_slug = "unknown"
        self.last_confidence = 0.0

    def observe(self, obs: dict[str, Any], *, your_index: int) -> dict[str, float | str]:
        self._collect_visible_cards(obs, your_index=your_index)
        slug, confidence = self.registry._classify_visible(self.visible_cards)
        if slug == "unknown" or confidence <= 0.0:
            slug = self.last_slug
            confidence = self.last_confidence
        else:
            prior = self.registry.priors.get(slug, 0.0) / 100.0
            confidence = min(1.0, confidence * 0.85 + prior * 0.15)
        self.last_slug = slug
        self.last_confidence = confidence
        return {"archetype_slug": slug, "confidence": confidence, "opponent_index": 1 - int(your_index)}

    def _collect_visible_cards(self, obs: dict[str, Any], *, your_index: int) -> None:
        current = obs.get("current") or {}
        players = current.get("players") or [{}, {}]
        opponent_index = 1 - int(your_index)
        if opponent_index >= len(players):
            return
        opponent = players[opponent_index] or {}

        active = opponent.get("active")
        if isinstance(active, dict):
            card_id = active.get("cardId") or active.get("id")
            if card_id is not None:
                self.visible_cards[int(card_id)] += 1

        for bench_card in opponent.get("bench") or []:
            if isinstance(bench_card, dict):
                card_id = bench_card.get("cardId") or bench_card.get("id")
                if card_id is not None:
                    self.visible_cards[int(card_id)] += 1

        for log_entry in obs.get("logs") or []:
            if not isinstance(log_entry, dict):
                continue
            card_id = log_entry.get("cardId") or log_entry.get("card")
            player = log_entry.get("player")
            if card_id is not None and player == opponent_index:
                self.visible_cards[int(card_id)] += 1

        tracker = GameEventTracker()
        tracker.observe(obs)
