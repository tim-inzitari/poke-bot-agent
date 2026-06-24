from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from poke_agent.features import STRUCTURED_FEATURE_DIM, row_feature_vector, total_feature_dim


from poke_agent.archetypes import load_archetype_registry, slug_from_deck_name
from poke_agent.episodes_index import is_top_of_ladder_source


class TrainingDiversityError(ValueError):
    """Training data or loop violates multi-deck training requirements."""


FORBIDDEN_CHECKPOINT_KEYS = frozenset({
    "agent_deck",
    "deck_cards",
    "submission_deck",
    "deck_list",
    "deck0",
    "deck1",
})


def episode_start_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_episode: dict[int, dict[str, Any]] = {}
    for row in rows:
        episode_id = int(row["episode"])
        step = int(row["step"])
        current = by_episode.get(episode_id)
        if current is None or step < int(current["step"]):
            by_episode[episode_id] = row
    return [by_episode[key] for key in sorted(by_episode)]


def training_matchup_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    starts = episode_start_rows(rows)
    matchups: list[tuple[str | None, str | None]] = []
    deck_slugs: set[str] = set()
    mirror_games = 0
    for row in starts:
        deck0 = row.get("deck0")
        deck1 = row.get("deck1")
        matchups.append((deck0, deck1))
        if isinstance(deck0, str) and deck0:
            deck_slugs.add(deck0)
        if isinstance(deck1, str) and deck1:
            deck_slugs.add(deck1)
        if deck0 and deck1 and deck0 == deck1:
            mirror_games += 1

    matchup_counts = Counter(matchups)
    return {
        "games": len(starts),
        "unique_matchups": len(matchup_counts),
        "unique_deck_slugs": len(deck_slugs),
        "mirror_games": mirror_games,
        "mirror_only": len(starts) > 0 and mirror_games == len(starts),
        "top_matchups": matchup_counts.most_common(5),
        "deck_slugs": sorted(deck_slugs),
    }


def _normalize_deck_slug(slug: str | None, registry) -> str | None:
    if not slug:
        return None
    return slug_from_deck_name(str(slug), registry)


def submission_deck_slug(config: dict[str, Any]) -> str | None:
    path = config.get("agent_deck_path")
    if path is None:
        return None
    deck_path = Path(path)
    if not deck_path.exists():
        return None
    try:
        root = deck_path.parents[2] if len(deck_path.parents) >= 3 else deck_path.parent
        registry = load_archetype_registry(root)
        return slug_from_deck_name(deck_path.stem, registry)
    except Exception:
        return deck_path.stem


def assert_submission_deck_separate_from_training(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    data_path: Path | str | None = None,
) -> None:
    """Runtime submission deck is configured separately from rollout JSONL training."""
    agent_path = Path(config["agent_deck_path"])
    if data_path is not None and Path(data_path).resolve() == agent_path.resolve():
        raise TrainingDiversityError(
            f"Training data path {data_path} must not be the submission deck file {agent_path}."
        )

    slug = submission_deck_slug(config)
    if slug is None:
        return

    stats = training_matchup_stats(rows)
    if stats["games"] == 0:
        return

    deck_path = Path(config["agent_deck_path"])
    root = deck_path.parents[2] if len(deck_path.parents) >= 3 else deck_path.parent
    registry = load_archetype_registry(root)

    starts = episode_start_rows(rows)
    submission_mirror_games = sum(
        1
        for row in starts
        if _normalize_deck_slug(row.get("deck0"), registry) == slug
        and _normalize_deck_slug(row.get("deck1"), registry) == slug
    )
    if submission_mirror_games == stats["games"]:
        raise TrainingDiversityError(
            f"All {stats['games']} training games are mirror matchups of the submission deck "
            f"({slug!r}). Train on merged multi-deck JSONL; keep {agent_path.name} for Kaggle submit only."
        )


def assert_training_matchup_diversity(
    rows: list[dict[str, Any]],
    *,
    min_matchups: int = 2,
    min_deck_slugs: int = 2,
    allow_single_matchup: bool = False,
) -> dict[str, Any]:
    stats = training_matchup_stats(rows)
    if stats["games"] == 0:
        raise TrainingDiversityError("Training rows contain no episodes.")

    if allow_single_matchup and stats["games"] <= 1:
        return stats

    if stats["mirror_only"]:
        raise TrainingDiversityError(
            "Every training game is a mirror matchup (deck0 == deck1). "
            "Use scraped replays + weighted multi-deck CABT generation, not mirror self-play."
        )

    required_matchups = min(min_matchups, stats["games"])
    if stats["unique_matchups"] < required_matchups:
        raise TrainingDiversityError(
            f"Training uses only {stats['unique_matchups']} distinct matchup(s) across "
            f"{stats['games']} game(s); need at least {required_matchups}. "
            f"Top matchups: {stats['top_matchups']}"
        )

    required_slugs = min(min_deck_slugs, max(2, stats["games"]))
    if stats["unique_deck_slugs"] < required_slugs:
        raise TrainingDiversityError(
            f"Training uses only {stats['unique_deck_slugs']} deck slug(s) "
            f"({stats['deck_slugs']}); need at least {required_slugs} for a generic model."
        )

    return stats


def top_of_ladder_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-episode provenance breakdown of how many games are top-of-ladder."""
    starts = episode_start_rows(rows)
    total = len(starts)
    ladder = 0
    source_counts: Counter[str] = Counter()
    for row in starts:
        source = str(row.get("source", ""))
        source_counts[source] += 1
        if is_top_of_ladder_source(source):
            ladder += 1
    return {
        "games": total,
        "ladder_games": ladder,
        "ladder_fraction": (ladder / total) if total else 0.0,
        "source_counts": source_counts.most_common(),
    }


def assert_top_of_ladder_data(
    rows: list[dict[str, Any]],
    *,
    min_fraction: float = 0.0,
) -> dict[str, Any]:
    """Require the training corpus to include top-of-ladder replay games.

    The bootstrap model must learn from real top-of-leaderboard play (scraped
    replays / episodes index), not only synthetic CABT self-play. By default at
    least one ladder game is required; ``min_fraction`` optionally enforces a
    minimum share of episodes from ladder sources.
    """
    stats = top_of_ladder_stats(rows)
    if stats["games"] == 0:
        raise TrainingDiversityError("Training rows contain no episodes.")

    if stats["ladder_games"] == 0:
        raise TrainingDiversityError(
            "No competition replay games from the episodes-index dataset found in bootstrap "
            f"training data (sources: {stats['source_counts']}). "
            "Run: bash scripts/download-episodes-index.sh "
            "then python scripts/prepare_training_data.py. "
            "Set REQUIRE_TOP_OF_LADDER_DATA=0 to bypass."
        )

    fraction = float(stats["ladder_fraction"])
    if min_fraction > 0.0 and fraction < min_fraction:
        raise TrainingDiversityError(
            f"Episodes-index games are only {fraction:.1%} of training episodes "
            f"({stats['ladder_games']}/{stats['games']}); need >= {min_fraction:.1%}. "
            "Increase TOP_EPISODE_PERCENT or lower MIN_TOP_OF_LADDER_FRACTION."
        )
    return stats


def assert_deck_metadata_not_in_features(
    rows: list[dict[str, Any]],
    *,
    state_hash_dim: int,
    sample_size: int = 3,
) -> None:
    """deck0/deck1 JSONL metadata must not change model inputs."""
    if not rows:
        return
    sample = rows[:sample_size]
    for row in sample:
        if row.get("observation") is None:
            continue
        base = row_feature_vector(row, state_hash_dim=state_hash_dim)
        mutated = dict(row)
        mutated["deck0"] = "__fake_deck_a__"
        mutated["deck1"] = "__fake_deck_b__"
        changed = row_feature_vector(mutated, state_hash_dim=state_hash_dim)
        if not np.allclose(base, changed, rtol=0.0, atol=0.0):
            raise TrainingDiversityError(
                "Feature vectors depend on deck0/deck1 metadata fields. "
                "Training inputs must come from observations only."
            )


def assert_generic_model_inputs(
    model: Any,
    tensors: Any,
    config: dict[str, Any],
) -> None:
    expected = total_feature_dim(state_hash_dim=int(config["state_hash_dim"]))
    actual = int(tensors.x_seq.shape[-1])
    if actual != expected:
        raise TrainingDiversityError(
            f"Unexpected input dim {actual}; expected {expected} from board/hand features only."
        )
    if int(getattr(model, "input_dim", actual)) != expected:
        raise TrainingDiversityError("Model input_dim does not match generic feature layout.")

    for name in model.state_dict():
        lowered = name.lower()
        if "deck" in lowered and "dec" not in lowered:
            raise TrainingDiversityError(f"Model parameter {name!r} looks deck-specific.")


def assert_checkpoint_has_no_deck(checkpoint: dict[str, Any]) -> None:
    for key in FORBIDDEN_CHECKPOINT_KEYS:
        if key in checkpoint:
            raise TrainingDiversityError(f"Checkpoint must not store deck-specific key {key!r}.")


def assert_training_pipeline(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    tensors: Any,
    model: Any,
    *,
    data_path: Path | str | None = None,
    allow_single_matchup: bool | None = None,
) -> dict[str, Any]:
    """Triple-check: config separation, diverse matchups, generic model inputs."""
    if allow_single_matchup is None:
        allow_single_matchup = not config.get("require_training_matchup_diversity", True)

    assert_submission_deck_separate_from_training(config, rows, data_path=data_path)
    stats = assert_training_matchup_diversity(
        rows,
        min_matchups=int(config.get("min_training_matchups", 2)),
        min_deck_slugs=int(config.get("min_training_deck_slugs", 2)),
        allow_single_matchup=allow_single_matchup,
    )
    assert_deck_metadata_not_in_features(rows, state_hash_dim=int(config["state_hash_dim"]))
    assert_generic_model_inputs(model, tensors, config)

    slug = submission_deck_slug(config)
    slug_note = f" submission deck slug={slug!r};" if slug else ""
    print(
        "training diversity OK:"
        f" {stats['games']} games,"
        f" {stats['unique_matchups']} matchups,"
        f" {stats['unique_deck_slugs']} deck slugs.{slug_note}"
        " Model learns from board state; your deck is supplied at runtime (beam/submit)."
    )
    return stats
