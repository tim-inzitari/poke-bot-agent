#!/usr/bin/env python3
"""Extract bounded checkpoint/gate evidence for Alakazam replay diagnosis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def gate_rows(metrics_root: Path, iteration: int) -> dict[str, Any]:
    payload = json.loads(
        (metrics_root / f"iter_{iteration:05d}.json").read_text(encoding="utf-8")
    )
    gate = payload["extra"]["active_gate_result"]
    opponents = gate.get("opponents", gate.get("matchups", []))
    selected = {}
    for row in opponents:
        opponent_id = str(row.get("opponent_id", ""))
        if any(
            token in opponent_id
            for token in (
                "specialist-alakazam-owner-accepted",
                "specialist-marnie-s-grimmsnarl",
                "lucifer19-battlecore",
            )
        ):
            selected[opponent_id] = {
                key: row.get(key)
                for key in ("games", "wins", "losses", "draws", "wr")
            }
    rating = gate.get("kaggle_rating_simulation") or {}
    return {
        "skill_weighted_win_rate": gate.get("skill_weighted_wr"),
        "projected_rating": rating.get("projected_rating"),
        "projected_rating_confidence_lower": rating.get("confidence_lower"),
        "opponents": selected,
    }


def checkpoint_rows(checkpoint_root: Path, iteration: int) -> dict[str, Any]:
    path = checkpoint_root / f"iter_{iteration:05d}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model_state_dict"]
    extra = payload["extra"]
    expanded = extra["expanded_head_training"]
    guide = extra["current_deck_guide_training"]
    route_final_norms = {}
    for key, value in state.items():
        if key.startswith("decision_fusion.dedicated_routes.") and key.endswith(
            "network.2.weight"
        ):
            route_final_norms[key.split(".")[2]] = float(value.float().norm().item())
    return {
        "path": str(path),
        "rl_iteration": payload.get("rl_iteration"),
        "param_count": extra.get("param_count"),
        "expanded_heads": {
            name: {
                "loss_weight": row["loss_weight"],
                "validation_loss": row["validation_loss"],
                "coverage": row["coverage"],
                "labeled_rows": row["labeled_rows"],
            }
            for name, row in expanded["heads"].items()
        },
        "guide": {
            "multiplier": guide["contract"]["guide_multiplier"],
            "direct_policy_cross_entropy": guide["contract"][
                "direct_policy_cross_entropy"
            ],
            "guide_preferred_action_consumed": guide["contract"][
                "guide_preferred_action_consumed"
            ],
            "validation_guide_conditioned_observed_loss": guide["validation"][
                "guide_conditioned_observed_loss"
            ],
            "validation_setup_board_outcome_weighted_loss": guide["validation"][
                "setup_board_outcome_weighted_loss"
            ],
            "validation_guide_rows": guide["validation"]["guide_rows"],
            "validation_setup_metrics": guide["validation"]["setup_metrics"],
        },
        "route_final_weight_norms": dict(sorted(route_final_norms.items())),
        "belief_validation": {
            key: extra["validation_metrics"].get(key)
            for key in (
                "opp_hand_loss",
                "n_opp_hand_rows",
                "lethal_threat_loss",
                "n_lethal_threat_rows",
                "prize_race_loss",
                "n_prize_race_rows",
            )
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint_iterations = (0, 9, 12)
    gate_iterations = (0, 4, 9, 12)
    result = {
        "schema": "poke_bot.alakazam_training_evidence/v1",
        "checkpoints": {
            str(iteration): checkpoint_rows(args.checkpoint_root, iteration)
            for iteration in checkpoint_iterations
        },
        "gates": {
            str(iteration): gate_rows(args.metrics_root, iteration)
            for iteration in gate_iterations
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
