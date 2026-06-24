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


def default_replay_convert_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 2 if cpu_count > 4 else cpu_count)


# Per-worker globals so the (read-only) archetype registry is loaded once per
# process instead of pickled for every replay task.
_WORKER_REGISTRY: Any = None
_WORKER_ROOT: Path | None = None


def _init_replay_worker(root_str: str) -> None:
    global _WORKER_REGISTRY, _WORKER_ROOT
    from poke_agent.archetypes import load_archetype_registry

    _WORKER_ROOT = Path(root_str)
    _WORKER_REGISTRY = load_archetype_registry(_WORKER_ROOT)


def _convert_replay_task(task: tuple[str, int, str]) -> list[dict]:
    replay_path_str, episode_index, source = task
    from poke_agent.replay_import import convert_replay_file

    return convert_replay_file(
        Path(replay_path_str),
        episode=episode_index,
        registry=_WORKER_REGISTRY,
        root=_WORKER_ROOT,
        source=source,
    )


def convert_records_to_rollout_rows(
    root: Path,
    records: list,
    *,
    default_source: str,
    workers: int | None = None,
    progress_label: str = "replays -> rollouts",
) -> list[dict]:
    """Convert replay records to rollout rows, parallel across episodes when worth it.

    Each replay file is independent, so conversion is embarrassingly parallel.
    Episode indices are baked into the tasks and ``executor.map`` preserves order,
    so output is deterministic regardless of worker count.
    """
    from concurrent.futures import ProcessPoolExecutor

    from poke_agent.archetypes import load_archetype_registry
    from poke_agent.replay_import import convert_replay_file

    total = len(records)
    if total == 0:
        return []

    worker_count = default_replay_convert_workers() if workers is None else max(1, int(workers))
    all_rows: list[dict] = []

    if worker_count <= 1 or total < worker_count * 2:
        registry = load_archetype_registry(root)
        for index, record in enumerate(records):
            rows = convert_replay_file(
                Path(record.replay_path),
                episode=index,
                registry=registry,
                root=root,
                source=record.source or default_source,
            )
            all_rows.extend(rows)
            if (index + 1) % 25 == 0 or index + 1 == total:
                print(f"  {progress_label}: {index + 1}/{total} episodes, {len(all_rows):,} rows")
        return all_rows

    tasks = [
        (str(record.replay_path), index, record.source or default_source)
        for index, record in enumerate(records)
    ]
    chunksize = max(1, total // (worker_count * 4))
    done = 0
    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_init_replay_worker,
        initargs=(str(root),),
    ) as executor:
        for rows in executor.map(_convert_replay_task, tasks, chunksize=chunksize):
            all_rows.extend(rows)
            done += 1
            if done % 100 == 0 or done == total:
                print(
                    f"  {progress_label}: {done}/{total} episodes, "
                    f"{len(all_rows):,} rows ({worker_count} workers)"
                )
    return all_rows


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
    workers: int | None = None,
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
    if workers is not None:
        cmd += ["--workers", str(int(workers))]
    print(f"merge rollouts: {len(existing)} sources -> {out_path}")
    return subprocess.run(cmd, cwd=root, check=check)


def ensure_episodes_index_data(
    root: Path,
    *,
    index_path: Path,
    top_percent_days: float = 100.0,
    download: bool = True,
) -> list[str]:
    """Ensure manifest + daily episode JSON bundles from episodes-index are on disk.

    Uses the official Kaggle dataset
    https://www.kaggle.com/datasets/kaggle/pokemon-tcg-ai-battle-episodes-index
    (not live leaderboard scraping).
    """
    from poke_agent.episodes_index import (
        daily_slugs_for_top_games,
        list_local_episode_files,
        load_daily_manifest,
        local_daily_episodes_dir,
    )

    if download and not index_path.is_file():
        print(f"downloading episodes index -> {index_path.parent}")
        subprocess.run(
            ["bash", "scripts/download-episodes-index.sh"],
            cwd=root,
            check=True,
        )

    if not index_path.is_file():
        raise FileNotFoundError(
            f"episodes index manifest missing: {index_path}\n"
            "Run: bash scripts/download-episodes-index.sh"
        )

    manifest = load_daily_manifest(index_path)
    slugs = daily_slugs_for_top_games(manifest, top_percent=top_percent_days)
    if not slugs:
        raise ValueError(f"no daily slugs in episodes index manifest: {index_path}")

    for slug in slugs:
        directory = local_daily_episodes_dir(root, slug)
        if list_local_episode_files(directory):
            print(f"episodes index: {slug} already local ({len(list_local_episode_files(directory))} files)")
            continue
        if not download:
            print(f"WARN: missing daily bundle {directory} (pass --download-index to fetch)")
            continue
        directory.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading daily episodes bundle: kaggle/{slug}")
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                f"kaggle/{slug}",
                "-p",
                str(directory),
                "--unzip",
            ],
            cwd=root,
            check=True,
        )
        count = len(list_local_episode_files(directory))
        print(f"episodes index: {slug} -> {count} replay files")
        if count == 0:
            raise FileNotFoundError(
                f"downloaded {slug} but found no .json replays under {directory}"
            )

    return slugs


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


def convert_episodes_index_to_rollouts(
    root: Path,
    *,
    out_path: Path,
    episodes_index: Path,
    top_percent: float = 1.0,
    top_percent_days: float = 100.0,
    max_episodes: int = 0,
    daily_slugs: list[str] | None = None,
    workers: int | None = None,
) -> int:
    """Convert episodes-index replay JSON into rollout JSONL (top competition games)."""
    import json

    from poke_agent.episodes_index import filter_top_percent, load_episode_pool

    records = load_episode_pool(
        root,
        index_path=episodes_index,
        daily_slugs=daily_slugs,
        include_scrape=False,
        top_percent_days=top_percent_days,
    )
    records = [r for r in records if r.replay_path is not None and r.replay_path.is_file()]
    if top_percent > 0 and top_percent < 100:
        records = filter_top_percent(records, top_percent)
    if max_episodes > 0:
        records = records[:max_episodes]

    if not records:
        raise FileNotFoundError(
            "no episodes-index replay files found. Run:\n"
            "  bash scripts/download-episodes-index.sh\n"
            "  python scripts/prepare_training_data.py"
        )

    total = len(records)
    all_rows = convert_records_to_rollout_rows(
        root,
        records,
        default_source="pokemon-tcg-ai-battle-episodes",
        workers=workers,
        progress_label="episodes-index -> rollouts",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    print(f"episodes-index rollouts: {total} episodes -> {len(all_rows):,} rows -> {out_path}")
    return total


convert_ladder_replays_to_rollouts = convert_episodes_index_to_rollouts


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

    # Merged corpus can be multi-GB; parse it in parallel (load_jsonl auto-scales).
    rows = load_jsonl(merged_path, workers=config.get("tensor_build_workers"))
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
    scrape: bool = False,
    scrape_teams: int = 10,
    scrape_episodes_per_sub: int = 20,
    download_index: bool = True,
    generate_multideck: bool = True,
    episodes: int | None = None,
    workers: int | None = None,
    top_percent: float | None = None,
    validate: bool = True,
) -> dict[str, Path | None]:
    """Refresh episodes-index + multideck corpora and write merged bootstrap JSONL.

    Top competition games come from the official episodes-index dataset only
    (https://www.kaggle.com/datasets/kaggle/pokemon-tcg-ai-battle-episodes-index),
    not live leaderboard scraping.
    """
    scraped_path = Path(config["scraped_rollout_path"])
    multideck_path = Path(config["multideck_rollout_path"])
    merged_path = Path(config["merged_rollout_path"])
    index_path = Path(config["episodes_index_path"])
    top_pct = float(top_percent if top_percent is not None else config.get("top_episode_percent", 1.0))

    if scrape:
        scrape_ladder_replays(
            root,
            teams=scrape_teams,
            episodes_per_sub=scrape_episodes_per_sub,
        )

    daily_slugs = ensure_episodes_index_data(
        root,
        index_path=index_path,
        download=download_index,
    )

    convert_episodes_index_to_rollouts(
        root,
        out_path=scraped_path,
        episodes_index=index_path,
        top_percent=top_pct,
        daily_slugs=daily_slugs,
        workers=workers,
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
        workers=workers,
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
