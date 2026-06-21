from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def default_cabt_generation_workers(*, episodes: int) -> int:
    cpu_count = os.cpu_count() or 1
    reserve = 2 if cpu_count > 4 else 0
    return max(1, min(cpu_count - reserve, int(episodes)))


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
