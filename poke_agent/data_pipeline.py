from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def default_cabt_generation_workers(*, episodes: int) -> int:
    cpu_count = os.cpu_count() or 1
    reserve = 2 if cpu_count > 4 else 0
    return max(1, min(cpu_count - reserve, int(episodes)))


def default_self_play_workers(*, games: int) -> int:
    """Parallel CABT game workers for self-play collection/eval."""
    return default_cabt_generation_workers(episodes=games)


def episode_chunks(episodes: int, workers: int) -> list[tuple[int, int]]:
    chunk_size = max(1, math.ceil(episodes / workers))
    return [(start, min(episodes, start + chunk_size)) for start in range(0, episodes, chunk_size)]


def generate_multideck_rollouts(
    root: Path,
    *,
    episodes: int,
    out_path: Path,
    matchups: str = "weighted",
    workers: int | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    worker_count = workers or default_cabt_generation_workers(episodes=episodes)
    cmd = [
        sys.executable,
        "scripts/generate_cabt_data.py",
        "--episodes",
        str(episodes),
        "--matchups",
        matchups,
        "--workers",
        str(worker_count),
        "--out",
        str(out_path),
    ]
    print(f"generate multideck: {episodes} games, {worker_count} workers -> {out_path}")
    return subprocess.run(cmd, cwd=root, check=check)


def merge_training_rollouts(
    root: Path,
    *,
    sources: list[Path],
    out_path: Path,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    existing = [path for path in sources if path.exists()]
    if not existing:
        raise FileNotFoundError(f"no rollout sources found to merge: {sources}")
    cmd = [
        sys.executable,
        "scripts/merge_rollouts.py",
        *[str(path) for path in existing],
        "--out",
        str(out_path),
    ]
    print(f"merge rollouts: {len(existing)} sources -> {out_path}")
    return subprocess.run(cmd, cwd=root, check=check)


def scrape_ladder_replays(
    root: Path,
    *,
    teams: int = 10,
    episodes_per_sub: int = 20,
    out_dir: Path | None = None,
    check: bool = False,
) -> bool:
    """Download new top-of-ladder replays via the Kaggle CLI (incremental/deduped).

    Returns True if the scrape subprocess succeeded. On failure (no API token,
    offline, etc.) prints a warning and returns False so callers can fall back
    to replays already on disk or the local episodes index.
    """
    ladder_dir = out_dir or (root / "data/ladder-replays")
    cmd = [
        sys.executable,
        "scripts/scrape_ladder_replays.py",
        "--teams",
        str(teams),
        "--episodes-per-sub",
        str(episodes_per_sub),
        "--out",
        str(ladder_dir),
    ]
    print(f"scrape ladder: top {teams} teams, {episodes_per_sub} eps/sub -> {ladder_dir}")
    proc = subprocess.run(cmd, cwd=root, check=False)
    if proc.returncode != 0:
        print("WARN: ladder scrape failed or skipped (need Kaggle API + competition join); using local replays")
        return False
    return True


def convert_ladder_replays_to_rollouts(
    root: Path,
    *,
    out_path: Path,
    scrape_index_csv: Path | None = None,
    episodes_index: Path | None = None,
    top_percent: float = 100.0,
    max_episodes: int = 0,
) -> int:
    """Convert scraped / indexed replay JSON into rollout JSONL (top-of-ladder corpus).

    Returns the number of episodes converted.
    """
    import json

    from poke_agent.archetypes import load_archetype_registry
    from poke_agent.episodes_index import default_index_path, filter_top_percent, load_episode_pool
    from poke_agent.replay_import import convert_replay_file

    scrape_csv = scrape_index_csv or (root / "data/ladder-replays/index.csv")
    index_path = episodes_index or default_index_path(root)

    records = load_episode_pool(
        root,
        index_path=index_path if index_path.is_file() else None,
        scrape_index_csv=scrape_csv if scrape_csv.is_file() else None,
    )
    records = [r for r in records if r.replay_path is not None and r.replay_path.is_file()]
    if top_percent > 0 and top_percent < 100:
        records = filter_top_percent(records, top_percent)
    if max_episodes > 0:
        records = records[:max_episodes]

    if not records:
        print("WARN: no replay files found to convert (scrape first or add kaggle/input episodes)")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", encoding="utf-8")
        return 0

    registry = load_archetype_registry(root)
    all_rows: list[dict] = []
    total = len(records)
    for episode_index, record in enumerate(records):
        assert record.replay_path is not None
        rows = convert_replay_file(
            record.replay_path,
            episode=episode_index,
            registry=registry,
            root=root,
            source=record.source or "ladder-scrape",
        )
        all_rows.extend(rows)
        if (episode_index + 1) % 25 == 0 or episode_index + 1 == total:
            print(f"  replays -> rollouts: {episode_index + 1}/{total} episodes, {len(all_rows):,} rows")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    print(f"ladder rollouts: {total} episodes -> {len(all_rows):,} rows -> {out_path}")
    return total


def validate_bootstrap_training_data(
    config: dict[str, Any],
    merged_path: Path,
) -> dict[str, Any]:
    """Load merged JSONL and run bootstrap data-quality gates."""
    from poke_agent.dataset import load_jsonl
    from poke_agent.training_diversity import (
        assert_top_of_ladder_data,
        assert_training_matchup_diversity,
        top_of_ladder_stats,
        training_matchup_stats,
    )

    if not merged_path.is_file():
        raise FileNotFoundError(f"merged training file missing: {merged_path}")

    rows = load_jsonl(merged_path, workers=1)
    if not rows:
        raise ValueError(f"merged training file is empty: {merged_path}")

    ladder = top_of_ladder_stats(rows)
    matchups = training_matchup_stats(rows)

    if config.get("require_top_of_ladder_data", True):
        assert_top_of_ladder_data(
            rows,
            min_fraction=float(config.get("min_top_of_ladder_fraction", 0.0)),
        )

    if config.get("require_training_matchup_diversity", True):
        assert_training_matchup_diversity(
            rows,
            min_matchups=int(config.get("min_training_matchups", 2)),
            min_deck_slugs=int(config.get("min_training_deck_slugs", 2)),
            allow_single_matchup=False,
        )

    print(
        "bootstrap data OK:"
        f" {matchups['games']} games,"
        f" {matchups['unique_matchups']} matchups,"
        f" {matchups['unique_deck_slugs']} deck slugs,"
        f" ladder {ladder['ladder_games']}/{ladder['games']}"
        f" ({ladder['ladder_fraction']:.1%})"
    )
    return {"ladder": ladder, "matchups": matchups, "rows": len(rows), "path": merged_path}


def prepare_bootstrap_training_data(
    config: dict[str, Any],
    root: Path,
    *,
    simulator_available: bool,
    scrape: bool = True,
    scrape_teams: int = 10,
    scrape_episodes_per_sub: int = 20,
    generate_multideck: bool = True,
    episodes: int | None = None,
    workers: int | None = None,
    top_percent: float | None = None,
    validate: bool = True,
) -> dict[str, Path | None]:
    """Refresh ladder + multideck corpora and write the merged bootstrap JSONL.

    Typical call before ``scripts/train_agent.py``:
      1. incremental Kaggle ladder scrape (optional)
      2. convert replays + local episodes index -> scraped_rollouts.jsonl
      3. generate multideck CABT games (optional, needs cg-lib)
      4. merge -> training_rollouts_merged.jsonl
      5. validate ladder + matchup diversity gates
    """
    scraped_path = Path(config["scraped_rollout_path"])
    multideck_path = Path(config["multideck_rollout_path"])
    merged_path = Path(config["merged_rollout_path"])
    top_pct = float(top_percent if top_percent is not None else config.get("top_episode_percent", 100.0))

    if scrape:
        scrape_ladder_replays(
            root,
            teams=scrape_teams,
            episodes_per_sub=scrape_episodes_per_sub,
        )

    convert_ladder_replays_to_rollouts(
        root,
        out_path=scraped_path,
        episodes_index=Path(config["episodes_index_path"]),
        top_percent=top_pct,
    )

    generate_games = episodes
    if generate_games is None:
        games = config.get("dataset_games")
        generate_games = max(0, int(games)) if games is not None else 0

    if generate_multideck and generate_games > 0 and simulator_available:
        generate_multideck_rollouts(
            root,
            episodes=generate_games,
            out_path=multideck_path,
            workers=workers,
        )
    elif generate_multideck and generate_games > 0:
        print("skip multideck CABT generation: simulator (cg-lib) unavailable")

    merge_training_rollouts(
        root,
        sources=[scraped_path, multideck_path],
        out_path=merged_path,
        check=True,
    )

    if validate:
        validate_bootstrap_training_data(config, merged_path)

    from poke_agent.cabt_validation import resolve_training_data_path

    train_path = resolve_training_data_path(config)
    return {
        "scraped_path": scraped_path if scraped_path.exists() else None,
        "multideck_path": multideck_path if multideck_path.exists() else None,
        "merged_path": merged_path if merged_path.exists() else None,
        "training_path": train_path,
    }


def prepare_training_rollout_files(
    config: dict[str, Any],
    root: Path,
    *,
    simulator_available: bool,
    episodes: int | None = None,
    workers: int | None = None,
    merge: bool = True,
) -> dict[str, Path | None]:
    """Generate multideck CABT rollouts (parallel) and optionally merge sources."""
    generate_games = episodes
    if generate_games is None:
        games = config.get("dataset_games")
        generate_games = max(0, int(games)) if games is not None else 0

    multideck_path = Path(config["multideck_rollout_path"])
    scraped_path = Path(config["scraped_rollout_path"])
    merged_path = Path(config["merged_rollout_path"])

    if generate_games > 0 and simulator_available:
        generate_multideck_rollouts(
            root,
            episodes=generate_games,
            out_path=multideck_path,
            workers=workers,
        )
    elif generate_games > 0:
        print("skip CABT generation: simulator unavailable")

    if merge:
        merge_training_rollouts(
            root,
            sources=[scraped_path, multideck_path],
            out_path=merged_path,
            check=False,
        )

    from poke_agent.cabt_validation import resolve_training_data_path

    train_path = resolve_training_data_path(config)
    return {
        "multideck_path": multideck_path if multideck_path.exists() else None,
        "scraped_path": scraped_path if scraped_path.exists() else None,
        "merged_path": merged_path if merged_path.exists() else None,
        "training_path": train_path,
    }
