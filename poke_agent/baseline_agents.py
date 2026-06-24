from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from poke_agent.deck_pool import (
    competitive_deck_archetype_slug,
    deck_matches_archetype_patterns,
    parse_deck_placement,
    read_deck_file,
)


@dataclass(frozen=True)
class BaselineAgent:
    """Competition opponent: official main.py model or first-legal heuristic fallback."""

    agent_id: str
    name: str
    path: Path
    deck: list[int]
    act: Callable[[dict[str, Any]], list[int]]
    heuristic: bool = False


def default_baseline_manifest() -> list[dict[str, str]]:
    """Catalog from https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/discussion/708584"""
    return [
        {"id": "iono", "name": "Iono's Deck", "dir": "iono"},
        {"id": "dragapult-ex", "name": "Dragapult ex Deck", "dir": "dragapult-ex"},
        {"id": "mega-abomasnow-ex", "name": "Mega Abomasnow ex Deck", "dir": "mega-abomasnow-ex"},
        {"id": "mega-lucario-ex", "name": "Mega Lucario ex Deck", "dir": "mega-lucario-ex"},
    ]


def baseline_manifest_path(root: Path) -> Path:
    return root / "baselines" / "manifest.json"


def write_default_baseline_manifest(root: Path) -> Path:
    path = baseline_manifest_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        payload = {"agents": default_baseline_manifest()}
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_baseline_manifest(root: Path) -> list[dict[str, str]]:
    path = baseline_manifest_path(root)
    if not path.exists():
        write_default_baseline_manifest(root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    agents = payload.get("agents")
    if not isinstance(agents, list) or not agents:
        raise ValueError(f"invalid baseline manifest: {path}")
    return agents


def load_kaggle_submission_agent(
    agent_dir: Path,
    *,
    agent_id: str,
    name: str,
    cg_lib_path: str | None,
) -> BaselineAgent:
    """Load a competition submission folder (main.py + deck.csv) as an opponent."""
    agent_dir = agent_dir.resolve()
    main_path = agent_dir / "main.py"
    deck_path = agent_dir / "deck.csv"
    if not main_path.is_file():
        raise FileNotFoundError(f"baseline agent missing main.py: {main_path}")
    if not deck_path.is_file():
        raise FileNotFoundError(f"baseline agent missing deck.csv: {deck_path}")

    paths_added: list[str] = []
    if cg_lib_path:
        paths_added.append(cg_lib_path)
        sys.path.insert(0, cg_lib_path)
    paths_added.append(str(agent_dir))
    sys.path.insert(0, str(agent_dir))

    try:
        module_name = f"baseline_agent_{agent_dir.name.replace('-', '_')}"
        spec = importlib.util.spec_from_file_location(module_name, main_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"failed to load baseline module spec: {main_path}")
        module = importlib.util.module_from_spec(spec)
        old_cwd = os.getcwd()
        try:
            os.chdir(agent_dir)
            spec.loader.exec_module(module)
        finally:
            os.chdir(old_cwd)
        agent_fn = getattr(module, "agent", None)
        if not callable(agent_fn):
            raise ImportError(f"{main_path} must define callable agent(obs_dict)")
        deck = read_deck_file(deck_path)
        return BaselineAgent(
            agent_id=agent_id,
            name=name,
            path=agent_dir,
            deck=deck,
            act=agent_fn,
        )
    finally:
        for path in paths_added:
            while path in sys.path:
                sys.path.remove(path)


def load_baseline_agents(
    root: Path,
    *,
    baseline_dir: Path | str,
    cg_lib_path: str | None,
) -> list[BaselineAgent]:
    """Load all official baseline agents that are present on disk."""
    base = Path(baseline_dir)
    if not base.is_absolute():
        base = root / base

    agents: list[BaselineAgent] = []
    missing: list[str] = []
    for entry in load_baseline_manifest(root):
        agent_dir = base / entry["dir"]
        if not (agent_dir / "main.py").is_file():
            missing.append(f"{entry['name']} ({agent_dir})")
            continue
        agents.append(
            load_kaggle_submission_agent(
                agent_dir,
                agent_id=str(entry["id"]),
                name=str(entry["name"]),
                cg_lib_path=cg_lib_path,
            )
        )

    if not agents:
        raise FileNotFoundError(
            "no baseline agents found. Install official Kaggle sample submissions under "
            f"{base} — see baselines/README.md. Missing: {', '.join(missing)}"
        )
    if missing:
        print("baseline agents partial load; missing:", ", ".join(missing))
    print(
        "baseline agents loaded:",
        ", ".join(f"{agent.name}@{agent.path.name}" for agent in agents),
    )
    return agents


# High-performing competitive lists aligned with each official baseline archetype.
BASELINE_OUR_DECK_ARCHETYPE_PATTERNS: dict[str, list[str]] = {
    "iono": ["lopunny-dudunsparce"],
    "dragapult-ex": ["dragapult"],
    "mega-abomasnow-ex": ["mega-abomasnow"],
    "mega-lucario-ex": ["mega-lucario", "lucario-hariyama"],
}


def resolve_baseline_archetype_deck_pool(
    root: Path,
    *,
    high_performing_dir: str = "decks/competitive/high_performing",
    baseline_dir: str = "baselines/official",
    include_official: bool = True,
    top_decks_per_archetype: int = 3,
) -> list[tuple[str, list[int]]]:
    """Our-side decks: best high-performing lists per baseline archetype (+ official lists)."""
    hp_dir = Path(high_performing_dir)
    if not hp_dir.is_absolute():
        hp_dir = root / hp_dir
    if not hp_dir.is_dir():
        raise FileNotFoundError(f"high-performing deck directory not found: {hp_dir}")

    official_base = Path(baseline_dir)
    if not official_base.is_absolute():
        official_base = root / official_base

    pool: list[tuple[str, list[int]]] = []
    seen_decks: set[tuple[int, ...]] = set()

    def add_deck(name: str, deck: list[int]) -> None:
        key = tuple(deck)
        if key in seen_decks:
            return
        seen_decks.add(key)
        pool.append((name, deck))

    hp_files = sorted(
        path
        for path in hp_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".csv", ".txt", ".deck"}
    )

    for entry in load_baseline_manifest(root):
        agent_id = str(entry["id"])
        patterns = BASELINE_OUR_DECK_ARCHETYPE_PATTERNS.get(agent_id, [agent_id])
        matched: list[tuple[int, str, list[int]]] = []
        for deck_file in hp_files:
            slug = competitive_deck_archetype_slug(deck_file.stem)
            if not deck_matches_archetype_patterns(slug, patterns):
                continue
            try:
                deck = read_deck_file(deck_file)
            except ValueError as exc:
                print(f"skipping invalid deck {deck_file.name}: {exc}")
                continue
            placement = parse_deck_placement(deck_file.stem)
            rank = placement if placement is not None else 10_000
            matched.append((rank, deck_file.stem, deck))
        matched.sort(key=lambda item: item[0])
        limit = max(1, int(top_decks_per_archetype))
        for _, name, deck in matched[:limit]:
            add_deck(name, deck)

        if include_official:
            official_deck = official_base / str(entry["dir"]) / "deck.csv"
            if official_deck.is_file():
                try:
                    add_deck(f"official-{agent_id}", read_deck_file(official_deck))
                except ValueError as exc:
                    print(f"skipping official baseline deck {official_deck}: {exc}")

    if not pool:
        raise ValueError(
            "no baseline archetype decks found in high_performing pool; "
            f"checked {hp_dir} for patterns {BASELINE_OUR_DECK_ARCHETYPE_PATTERNS}"
        )
    return pool


def summarize_baseline_eval(reports: dict[str, dict[str, float]]) -> dict[str, float]:
    """Aggregate per-baseline win rates into one score."""
    if not reports:
        return {"games": 0.0, "wins": 0.0, "losses": 0.0, "draws": 0.0, "win_rate": 0.0}
    total_games = sum(float(stats["games"]) for stats in reports.values())
    total_wins = sum(float(stats["wins"]) for stats in reports.values())
    total_losses = sum(float(stats["losses"]) for stats in reports.values())
    total_draws = sum(float(stats["draws"]) for stats in reports.values())
    decided = total_wins + total_losses
    return {
        "games": total_games,
        "wins": total_wins,
        "losses": total_losses,
        "draws": total_draws,
        "win_rate": (total_wins / decided) if decided else 0.0,
    }
