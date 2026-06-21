from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Fields written by scripts/generate_cabt_data.py when playing real CABT games via
# cg.game.battle_start / battle_select — the same engine Kaggle uses for evaluation.
CABT_EVALUATION_ROW_FIELDS = frozenset({
    "episode",
    "step",
    "features",
    "next_features",
    "observation",
    "action",
    "next_observation",
    "terminal",
    "player",
    "value",
    "deck0",
    "deck1",
    "legal_action_count",
    "select_min_count",
    "select_max_count",
})

CABT_OBSERVATION_FIELDS = frozenset({"current", "select", "logs", "remainingOverageTime", "step"})


class CabtEvaluationDataError(ValueError):
    """Raised when rollout data is not from CABT evaluation games."""


def is_cabt_observation(observation: Any) -> bool:
    if not isinstance(observation, dict):
        return False
    current = observation.get("current")
    if not isinstance(current, dict):
        return False
    if "result" not in current or "yourIndex" not in current:
        return False
    players = current.get("players")
    if not isinstance(players, list) or len(players) < 2:
        return False
    return True


def is_cabt_evaluation_row(row: dict[str, Any]) -> bool:
    if not CABT_EVALUATION_ROW_FIELDS.issubset(row.keys()):
        return False
    if not isinstance(row["features"], list) or not row["features"]:
        return False
    if not isinstance(row["next_features"], list) or not row["next_features"]:
        return False
    if not isinstance(row["action"], list):
        return False
    if not is_cabt_observation(row["observation"]):
        return False
    if not is_cabt_observation(row["next_observation"]):
        return False
    return True


def assert_cabt_evaluation_rows(
    rows: list[dict[str, Any]],
    *,
    path: Path | str | None = None,
    min_rows: int = 1,
) -> None:
    label = f" at {path}" if path else ""
    if len(rows) < min_rows:
        raise CabtEvaluationDataError(
            f"Expected at least {min_rows} CABT evaluation rows{label}, found {len(rows)}."
        )

    for index, row in enumerate(rows):
        if not is_cabt_evaluation_row(row):
            missing = sorted(CABT_EVALUATION_ROW_FIELDS - set(row.keys()))
            detail = f"missing fields: {missing}" if missing else "invalid observation/action payload"
            raise CabtEvaluationDataError(
                f"Row {index}{label} is not a CABT evaluation game transition ({detail}). "
                "Generate data with scripts/generate_cabt_data.py (cg.game engine), not the "
                "compact inline notebook rollouts."
            )

    sample = rows[0]
    print(
        f"CABT evaluation data OK{label}: {len(rows)} rows, "
        f"matchup={sample['deck0']} vs {sample['deck1']}, "
        f"feature_dim={len(sample['features'])}"
    )


def resolve_cabt_eval_data_path(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if not path.exists():
            continue
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not is_cabt_evaluation_row(row):
                    raise CabtEvaluationDataError(
                        f"{path} line {line_number} is not CABT evaluation format. "
                        "Use rollout JSONL from scripts/generate_cabt_data.py."
                    )
                rows.append(row)
                if len(rows) >= 3:
                    break
        if rows:
            return path
    return None


def is_training_rollout_row(row: dict[str, Any]) -> bool:
    return (
        "episode" in row
        and "step" in row
        and isinstance(row.get("features"), list)
        and bool(row["features"])
        and "value" in row
    )


def assert_training_rollout_rows(
    rows: list[dict[str, Any]],
    *,
    path: Path | str | None = None,
    min_rows: int = 1,
) -> None:
    label = f" at {path}" if path else ""
    if len(rows) < min_rows:
        raise CabtEvaluationDataError(
            f"Expected at least {min_rows} rollout rows{label}, found {len(rows)}."
        )
    for index, row in enumerate(rows):
        if not is_training_rollout_row(row):
            raise CabtEvaluationDataError(
                f"Row {index}{label} is missing episode/step/features/value fields."
            )
    sample = rows[0]
    print(
        f"rollout data OK{label}: {len(rows)} rows, "
        f"feature_dim={len(sample['features'])}"
    )


def uses_generated_training_data(config: dict[str, Any], data_path: Path) -> bool:
    generated = config.get("generated_path")
    if generated is None:
        return False
    return Path(data_path).resolve() == Path(generated).resolve()


def resolve_training_data_path(config: dict[str, Any]) -> Path | None:
    """Pick rollout JSONL for training.

    Explicit ``training_data_path`` wins (self-play retraining). Otherwise prefer
    merged multi-deck corpus, then scraped/multideck sources. Stale mirror-only
    ``generated_path`` files are a last resort.
    """
    explicit = config.get("training_data_path")
    if explicit is not None:
        explicit_path = Path(explicit)
        if explicit_path.exists():
            return explicit_path

    merged_path = config.get("merged_rollout_path")
    merged = Path(merged_path) if merged_path is not None else None
    if merged is not None and merged.exists():
        return merged

    for key in ("scraped_rollout_path", "multideck_rollout_path"):
        candidate = config.get(key)
        if candidate is not None:
            path = Path(candidate)
            if path.exists():
                return path

    for source in config.get("training_rollout_sources", []):
        path = Path(source)
        if path.exists():
            return path

    generated = config.get("generated_path")
    generated_path = Path(generated) if generated is not None else None
    dataset_games = config.get("dataset_games")
    require_cabt_eval = config.get("require_cabt_eval_data", True)

    if dataset_games is not None and generated_path is not None and generated_path.exists():
        return generated_path

    candidates = list(config.get("data_candidates", []))
    if generated_path is not None and generated_path.exists():
        candidates = [generated_path, *[path for path in candidates if path != generated_path]]

    if require_cabt_eval:
        return resolve_cabt_eval_data_path(candidates)

    return next((path for path in candidates if path.exists()), None)
