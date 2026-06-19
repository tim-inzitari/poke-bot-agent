from __future__ import annotations

from pathlib import Path
from typing import Any


SAMPLE_DECK = [
    119, 119, 119, 119, 120, 120, 120, 120, 121, 121, 121,
    305, 305, 66, 66, 112, 112, 235, 1071, 140, 1227,
    1227, 1227, 1227, 1182, 1182, 1182, 1198, 1198, 1240, 1213,
    1086, 1086, 1086, 1086, 1152, 1152, 1152, 1152, 1121, 1121,
    1121, 1121, 1120, 1120, 1120, 1120, 1097, 1097, 1080, 1260,
    1260, 2, 2, 2, 5, 5, 5, 7, 7,
]


def read_deck(config: dict[str, Any], root: Path) -> tuple[list[int], Path]:
    candidates = [
        config["agent_deck_path"],
        root / "decks/submission.csv",
        root / "submission/deck.csv",
        root / "deck.csv",
        Path("/kaggle_simulations/agent/deck.csv"),
    ]
    for path in candidates:
        if path.exists():
            deck = [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]
            if len(deck) != 60:
                raise ValueError(f"{path} must contain 60 card IDs")
            return deck, path
    return SAMPLE_DECK, Path("<sample>")
