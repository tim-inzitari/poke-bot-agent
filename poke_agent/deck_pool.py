from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

from poke_agent.deck import SAMPLE_DECK

_PLACEMENT_RE = re.compile(r"_(\d+)(?:st|nd|rd|th)_", re.IGNORECASE)


def parse_deck_placement(deck_name: str) -> int | None:
    """Parse event placement from deck filename, e.g. ..._1000th_archetype -> 1000."""
    match = _PLACEMENT_RE.search(deck_name)
    if match is None:
        return None
    return int(match.group(1))


def read_deck_file(path: Path) -> list[int]:
    values: list[int] = []
    for raw_token in path.read_text(encoding="utf-8").replace(",", "\n").splitlines():
        token = raw_token.strip()
        if token:
            values.append(int(token))
    if len(values) != 60:
        raise ValueError(f"{path} must contain exactly 60 card IDs, found {len(values)}")
    return values


def read_deck_pool(path: str | Path | None, *, root: Path | None = None) -> list[tuple[str, list[int]]]:
    """Load (name, deck) tuples from a directory of 60-card CSV/txt files."""
    if path is None:
        return []

    pool_path = Path(path)
    if not pool_path.is_absolute() and root is not None:
        pool_path = root / pool_path
    if not pool_path.exists():
        raise FileNotFoundError(f"deck pool directory not found: {pool_path}")

    files = sorted(
        candidate
        for candidate in pool_path.iterdir()
        if candidate.is_file() and candidate.suffix.lower() in {".csv", ".txt", ".deck"}
    )
    if not files:
        raise FileNotFoundError(f"no deck files found in {pool_path}")

    pool: list[tuple[str, list[int]]] = []
    for deck_file in files:
        try:
            pool.append((deck_file.stem, read_deck_file(deck_file)))
        except ValueError as exc:
            print(f"skipping invalid deck {deck_file.name}: {exc}")
    if not pool:
        raise ValueError(f"no valid 60-card decks in {pool_path}")
    return pool


def choose_matchup(
    episode: int,
    deck0_pool: list[tuple[str, list[int]]],
    deck1_pool: list[tuple[str, list[int]]],
    mode: str,
) -> tuple[str, list[int], str, list[int]]:
    if mode == "round-robin":
        i = episode % len(deck0_pool)
        j = (episode // len(deck0_pool)) % len(deck1_pool)
        name0, deck0 = deck0_pool[i]
        name1, deck1 = deck1_pool[j]
    else:
        name0, deck0 = random.choice(deck0_pool)
        name1, deck1 = random.choice(deck1_pool)
    return name0, deck0, name1, deck1


@dataclass(frozen=True)
class FieldMatchup:
    deck0_name: str
    deck0: list[int]
    deck1_name: str
    deck1: list[int]
    agent_seat: int
    field_name: str
    opponent_placement: int | None = None

    @property
    def agent_deck(self) -> list[int]:
        return self.deck0 if self.agent_seat == 0 else self.deck1

    def deck_for_seat(self, seat: int) -> list[int]:
        return self.deck0 if seat == 0 else self.deck1

    def name_for_seat(self, seat: int) -> str:
        return self.deck0_name if seat == 0 else self.deck1_name


def filter_pool_by_max_placement(
    pool: list[tuple[str, list[int]]],
    max_placement: int,
) -> list[tuple[str, list[int]]]:
    """Keep decks with parsed placement <= max_placement (lower rank = stronger)."""
    filtered: list[tuple[str, list[int]]] = []
    for name, deck in pool:
        placement = parse_deck_placement(name)
        if placement is not None and placement <= max_placement:
            filtered.append((name, deck))
    return filtered


def choose_agent_vs_field_matchup(
    episode: int,
    agent_name: str,
    agent_deck: list[int],
    field_pool: list[tuple[str, list[int]]],
    *,
    mode: str = "sample",
    alternate_agent_seat: bool = True,
) -> FieldMatchup:
    """Pair our agent deck vs a meta deck from the field pool."""
    if not field_pool:
        raise ValueError("field deck pool is empty")

    if mode == "round-robin":
        field_name, field_deck = field_pool[episode % len(field_pool)]
    else:
        field_name, field_deck = random.choice(field_pool)

    agent_seat = (episode % 2) if alternate_agent_seat else 0
    placement = parse_deck_placement(field_name)
    if agent_seat == 0:
        return FieldMatchup(agent_name, agent_deck, field_name, field_deck, 0, field_name, placement)
    return FieldMatchup(field_name, field_deck, agent_name, agent_deck, 1, field_name, placement)


def mirror_matchup(agent_name: str, agent_deck: list[int]) -> FieldMatchup:
    """Fallback when no field pool is configured."""
    placement = parse_deck_placement(agent_name)
    return FieldMatchup(agent_name, agent_deck, agent_name, list(agent_deck), 0, agent_name, placement)


def resolve_field_pool(
    field_dir: str | Path | None,
    *,
    root: Path,
    agent_name: str,
    agent_deck: list[int],
) -> tuple[list[tuple[str, list[int]]], bool]:
    """Return (field_pool, using_field). Field pool excludes exact duplicate of agent deck."""
    if field_dir is None:
        return [], False

    try:
        raw_pool = read_deck_pool(field_dir, root=root)
    except (FileNotFoundError, ValueError) as exc:
        print(f"field deck pool unavailable ({exc}); using mirror matchups")
        return [], False

    agent_key = tuple(agent_deck)
    field_pool = [(name, deck) for name, deck in raw_pool if tuple(deck) != agent_key]
    if not field_pool:
        print("field deck pool has no decks distinct from agent deck; using mirror matchups")
        return [], False
    return field_pool, True
