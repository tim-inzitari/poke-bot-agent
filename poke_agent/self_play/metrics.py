"""Pure outcome + value-calibration metrics for self-play.

Extracted from the former monolithic self_play module. These functions only read
plain rollout-row dicts (no simulator, no multiprocessing state), so they live apart
from the collection/orchestration machinery in core.py.
"""

from __future__ import annotations

from typing import Any

import torch


def terminal_result(match_rows: list[dict[str, Any]]) -> int | None:
    if not match_rows:
        return None
    return int(next(row for row in reversed(match_rows) if row.get("terminal")).get("result", -1))


def record_seat_outcome(
    result: int,
    our_seat: int,
    *,
    wins: int,
    losses: int,
    draws: int,
) -> tuple[int, int, int]:
    if result == 2:
        return wins, losses, draws + 1
    if result == our_seat:
        return wins + 1, losses, draws
    if result >= 0:
        return wins, losses + 1, draws
    return wins, losses, draws


def summarize_results(results: list[int], *, seat_index: int) -> dict[str, float]:
    """Summarize game outcomes from one seat's perspective (0=player0 wins)."""
    wins = sum(1 for result in results if result == seat_index)
    losses = sum(1 for result in results if result >= 0 and result != 2 and result != seat_index)
    draws = sum(1 for result in results if result == 2)
    decided = wins + losses
    win_rate = (wins / decided) if decided else 0.0
    return {
        "games": float(len(results)),
        "wins": float(wins),
        "losses": float(losses),
        "draws": float(draws),
        "win_rate": win_rate,
    }


def value_calibration_metrics(rows: list[dict[str, Any]], *, seat: int) -> dict[str, float]:
    """Brier score and ECE from search/root value predictions vs game outcome."""
    by_episode: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_episode.setdefault(int(row["episode"]), []).append(row)

    preds: list[float] = []
    labels: list[float] = []
    for episode_rows in by_episode.values():
        episode_rows.sort(key=lambda row: int(row["step"]))
        outcome = terminal_result(episode_rows)
        if outcome is None:
            continue
        label = 1.0 if int(outcome) == int(seat) else -1.0
        for row in episode_rows:
            if int(row.get("player", -1)) != int(seat):
                continue
            if "search_value" not in row:
                continue
            preds.append(float(row["search_value"]))
            labels.append(label)

    if not preds:
        return {"brier": 0.0, "ece": 0.0, "samples": 0.0}

    pred_t = torch.tensor(preds, dtype=torch.float32)
    label_t = torch.tensor(labels, dtype=torch.float32)
    brier = float(((pred_t - label_t) ** 2).mean().item())

    # Expected calibration error with 10 equal-width bins on [-1, 1].
    bins = torch.linspace(-1.0, 1.0, steps=11)
    ece = 0.0
    total = float(len(preds))
    for start, end in zip(bins[:-1], bins[1:]):
        mask = (pred_t >= start) & (pred_t < end)
        if not bool(mask.any()):
            continue
        bin_pred = float(pred_t[mask].mean().item())
        bin_label = float(label_t[mask].mean().item())
        ece += float(mask.sum().item()) / total * abs(bin_pred - bin_label)

    return {"brier": brier, "ece": ece, "samples": float(len(preds))}


def calibration_metrics_from_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    seat0 = value_calibration_metrics(rows, seat=0)
    seat1 = value_calibration_metrics(rows, seat=1)
    samples = seat0["samples"] + seat1["samples"]
    if samples <= 0:
        return {"brier": 0.0, "ece": 0.0, "samples": 0.0}
    brier = (seat0["brier"] * seat0["samples"] + seat1["brier"] * seat1["samples"]) / samples
    ece = (seat0["ece"] * seat0["samples"] + seat1["ece"] * seat1["samples"]) / samples
    return {"brier": brier, "ece": ece, "samples": samples}
