#!/usr/bin/env python3
"""Measure high-confidence Alakazam guide agreement on Kaggle decisions."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import alakazam_heuristics, features  # noqa: E402


def load(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def episode_id(path: Path) -> int:
    match = re.search(r"episode-(\d+)-replay\.json$", path.name)
    if match is None:
        raise ValueError(path)
    return int(match.group(1))


def initial_deck(replay: dict[str, Any], own_index: int) -> list[int]:
    for agent in replay["steps"][0]:
        timeline = agent.get("visualize")
        if isinstance(timeline, list) and timeline:
            return [
                int(card["id"])
                for card in timeline[0]["current"]["players"][own_index]["deck"]
            ]
    raise RuntimeError("initial deck unavailable")


def guide_target(
    obs: dict[str, Any], combos: list[list[int]], deck: list[int]
) -> tuple[int | None, float]:
    values = alakazam_heuristics.guide_scores(
        obs, combos, deck=deck, force_enabled=True
    )
    if values is None or len(values) < 2:
        return None, 0.0
    scores = [float(value) for value in values]
    if not all(math.isfinite(value) for value in scores):
        return None, 0.0
    order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
    margin = scores[order[0]] - scores[order[1]]
    if margin <= 1e-8:
        return None, 0.0
    return int(order[0]), min(1.0, margin)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def one(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "guide_rows": len(group),
            "agreement": mean(row["agreement"] for row in group) if group else None,
            "mean_confidence": mean(row["confidence"] for row in group) if group else None,
            "contexts": dict(Counter(str(row["context"]) for row in group)),
        }

    groups: dict[str, list[dict[str, Any]]] = {
        "all": rows,
        "wins": [row for row in rows if row["win"]],
        "losses": [row for row in rows if not row["win"]],
    }
    for archetype in sorted({row["opponent_archetype"] for row in rows}):
        groups[f"matchup:{archetype}"] = [
            row for row in rows if row["opponent_archetype"] == archetype
        ]
        groups[f"matchup:{archetype}:losses"] = [
            row
            for row in rows
            if row["opponent_archetype"] == archetype and not row["win"]
        ]
    return {key: one(group) for key, group in groups.items() if group}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("replay_root", type=Path)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    episodes = {
        int(row["episode_id"]): row
        for row in load(args.analysis)["episodes"]
        if int(row["iteration"]) == args.iteration
    }
    rows: list[dict[str, Any]] = []
    total_stages = 0
    for path in sorted(
        (args.replay_root / f"iter_{args.iteration:05d}").glob("*-replay.json")
    ):
        meta = episodes[episode_id(path)]
        replay = load(path)
        own_index = int(meta["own_index"])
        deck = initial_deck(replay, own_index)
        steps = replay["steps"]
        for index in range(len(steps) - 1):
            record = steps[index][own_index]
            obs = record.get("observation") or {}
            if record.get("status") != "ACTIVE" or obs.get("select") is None:
                continue
            if int((obs.get("select") or {}).get("context", -1)) == 41:
                continue
            action = [
                int(value)
                for value in (steps[index + 1][own_index].get("action") or [])
            ]
            for stage_index, (combos, target) in enumerate(
                features.factorized_teacher_forcing_stages(obs, action)
            ):
                total_stages += 1
                normalized = [list(combo) for combo in combos]
                preferred, confidence = guide_target(obs, normalized, deck)
                if preferred is None:
                    continue
                rows.append(
                    {
                        "episode_id": int(meta["episode_id"]),
                        "iteration": int(meta["iteration"]),
                        "win": bool(meta["win"]),
                        "opponent_archetype": meta["opponent_archetype"],
                        "step_index": index,
                        "factorized_stage": stage_index,
                        "turn": int((obs.get("current") or {}).get("turn", 0) or 0),
                        "context": int((obs.get("select") or {}).get("context", -1)),
                        "target": int(target),
                        "guide_target": preferred,
                        "confidence": confidence,
                        "agreement": int(target) == preferred,
                    }
                )
    payload = {
        "schema": "poke_bot.kaggle_alakazam_guide_alignment/v1",
        "iteration": args.iteration,
        "episodes": len(episodes),
        "total_stages": total_stages,
        "guide_stages": len(rows),
        "summary": summarize(rows),
        "stages": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"episodes": len(episodes), "stages": total_stages, "guide_stages": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
