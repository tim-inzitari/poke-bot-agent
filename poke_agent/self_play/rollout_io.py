"""Rollout JSONL + manifest I/O for self-play.

Extracted from the former monolithic self_play module. These helpers only touch the
filesystem and a duck-typed ``settings`` object (``output_path``, ``train_window_games``,
``trim_rollout_file``, ``games_per_iteration``), so they carry no simulator or
multiprocessing dependencies. ``settings`` is typed ``Any`` to avoid a circular import
with core.py, which owns ``SelfPlaySettings``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_jsonl(path: Path, rows: list[dict[str, Any]], *, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def count_rollout_games(path: Path) -> int:
    if not path.exists():
        return 0
    episodes: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        episodes.add(int(json.loads(line)["episode"]))
    return len(episodes)


def maybe_trim_rollout_file(settings: Any) -> None:
    if not settings.trim_rollout_file:
        return
    from poke_agent.dataset import trim_rollout_jsonl

    trim_rollout_jsonl(settings.output_path, settings.train_window_games)


def rollout_buffer_overwrites(settings: Any) -> bool:
    """True when each iteration should replace the JSONL with only this batch."""
    if not settings.trim_rollout_file:
        return False
    window = settings.train_window_games
    if window is None or int(window) <= 0:
        return False
    return int(settings.games_per_iteration) == int(window)


def write_rollout_buffer(
    settings: Any,
    rows: list[dict[str, Any]],
    *,
    iteration: int,
) -> tuple[bool, int]:
    """Persist collected rows; returns (overwrote_file, games_on_disk)."""
    if rollout_buffer_overwrites(settings):
        write_jsonl(settings.output_path, rows, append=False)
        kept_games = count_rollout_games(settings.output_path)
        return True, kept_games

    append = settings.output_path.exists() and iteration > 1
    write_jsonl(settings.output_path, rows, append=append)
    maybe_trim_rollout_file(settings)
    return append, count_rollout_games(settings.output_path)


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"iterations": [], "champion": None}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
