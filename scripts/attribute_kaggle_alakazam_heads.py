#!/usr/bin/env python3
"""Instrument Alakazam Kaggle decisions through all learned fusion heads.

This is a diagnostic-only replay.  It rebuilds the realized temporal history
from the public Kaggle observations, restores the public matchup router, and
measures leave-one-head-out logit effects at every teacher-forced factorized
decision stage.  It does not write a checkpoint or alter a running service.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import alakazam_heuristics, features  # noqa: E402
from poke_bot.model import CausalDecisionFusion  # noqa: E402
from poke_bot.public_matchup_router import RuntimePublicMatchupRouter  # noqa: E402
from poke_bot.train import load_model_from_checkpoint  # noqa: E402


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
        visualize = agent.get("visualize")
        if isinstance(visualize, list) and visualize:
            cards = visualize[0]["current"]["players"][own_index]["deck"]
            return [int(card["id"]) for card in cards]
    raise RuntimeError("decoded initial deck is absent")


def sources(
    model: torch.nn.Module,
    state: torch.Tensor,
    option_hidden: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    belief = model.belief_aux_logits(state)
    expanded_state = model.expanded_state_logits(state)
    state_sources = {
        "value": torch.tanh(model.value_head(state)),
        "archetype": belief["aux_logits"],
        "opponent_hand": belief["opp_hand_logits"],
        "opponent_remainder": belief["opp_remainder_logits"],
        "lethal_threat": belief["lethal_threat_logits"],
        "prize_race": belief["prize_race_pred"],
        "tactical_outcomes": expanded_state["tactical_outcome"],
        "opponent_response": expanded_state["opponent_response"],
        "resource_forecast": expanded_state["resource_forecast"],
        "game_phase": expanded_state["game_phase"],
        "outcome_distribution": expanded_state["outcome_distribution"],
        "remaining_turns": expanded_state["remaining_turns"],
    }
    option_sources = model.expanded_option_logits(option_hidden)
    option_sources["setup_board_outcome"] = model.setup_board_outcome_logits(
        option_hidden
    )
    option_sources["combo_state"] = model.combo_state_logits(option_hidden)
    return state_sources, option_sources


def margin(logits: torch.Tensor, target: int) -> float:
    row = logits[0]
    if row.numel() <= 1:
        return 0.0
    others = torch.cat((row[:target], row[target + 1 :]))
    return float((row[target] - torch.max(others)).item())


def unique_guide_target(
    obs: dict[str, Any], combos: list[list[int]], deck: list[int]
) -> tuple[int | None, float]:
    raw = alakazam_heuristics.guide_scores(obs, combos, deck=deck)
    if raw is None or len(raw) != len(combos) or len(raw) < 2:
        return None, 0.0
    values = [float(value) for value in raw]
    if not all(math.isfinite(value) for value in values):
        return None, 0.0
    order = sorted(range(len(values)), key=values.__getitem__, reverse=True)
    gap = values[order[0]] - values[order[1]]
    return (order[0], min(1.0, max(0.0, gap))) if gap > 1e-8 else (None, 0.0)


def record_stage(
    *,
    model: torch.nn.Module,
    fusion: CausalDecisionFusion,
    obs: dict[str, Any],
    deck: list[int],
    combos: list[list[int]],
    target: int,
    state: torch.Tensor,
    spatial: torch.Tensor,
    policy_value_state: torch.Tensor,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    option_tokens = features.build_option_tokens(obs, combos)
    decoded, hidden = model.decode_options(
        option_tokens,
        spatial,
        policy_value_state,
        n_options=[len(combos)],
        return_hidden=True,
        decision_fusion_state_vec=state,
    )
    base = model.policy_head(hidden).squeeze(-1)
    state_sources, option_sources = sources(model, state, hidden)
    full = fusion(
        hidden,
        base,
        state_sources=state_sources,
        option_sources=option_sources,
        dedicated_routes_active=True,
    )
    if not torch.allclose(decoded, full, atol=1e-5, rtol=1e-5):
        raise RuntimeError("instrumented fusion does not match model decode")
    full_choice = int(torch.argmax(full[0]).item())
    guide_target, guide_confidence = unique_guide_target(obs, combos, deck)
    head_records: dict[str, Any] = {}
    # Preserve physical H10 inventory for checkpoint audit, but attribute only
    # routes active in this runtime (r175 keeps combo tensors resident while
    # explicitly disabling its action/guide route).
    for name in getattr(fusion, "active_required_heads", fusion.required_heads):
        changed_state = dict(state_sources)
        changed_option = dict(option_sources)
        if name in changed_state:
            changed_state[name] = torch.zeros_like(changed_state[name])
        else:
            changed_option[name] = torch.zeros_like(changed_option[name])
        ablated = fusion(
            hidden,
            base,
            state_sources=changed_state,
            option_sources=changed_option,
            dedicated_routes_active=True,
        )
        effect = full - ablated
        head_records[name] = {
            "mean_abs_logit_effect": float(torch.mean(torch.abs(effect)).item()),
            "max_abs_logit_effect": float(torch.max(torch.abs(effect)).item()),
            "target_margin_effect": margin(full, target) - margin(ablated, target),
            "changes_model_choice_when_ablated": int(torch.argmax(ablated[0]).item())
            != full_choice,
            "guide_relative_effect": None
            if guide_target is None or guide_target == target
            else float((effect[0, guide_target] - effect[0, target]).item()),
        }
    select = obs.get("select") or {}
    current = obs.get("current") or {}
    return {
        **metadata,
        "turn": int(current.get("turn", 0) or 0),
        "context": int(select.get("context", -1) or -1),
        "candidate_count": len(combos),
        "target_index": target,
        "model_choice": full_choice,
        "model_matches_realized_action": full_choice == target,
        "model_target_margin": margin(full, target),
        "guide_target": guide_target,
        "guide_confidence": guide_confidence,
        "guide_matches_realized_action": guide_target == target
        if guide_target is not None
        else None,
        "guide_matches_model": guide_target == full_choice
        if guide_target is not None
        else None,
        "heads": head_records,
    }


def analyze_episode(
    *,
    path: Path,
    meta: dict[str, Any],
    model: torch.nn.Module,
    tree_path: Path,
) -> list[dict[str, Any]]:
    replay = load(path)
    own_index = int(meta["own_index"])
    deck = initial_deck(replay, own_index)
    router = RuntimePublicMatchupRouter.from_path(tree_path)
    board_history: list[features.SparseVector] = []
    action_history: list[features.SparseVector | None] = []
    previous_action: features.SparseVector | None = None
    output: list[dict[str, Any]] = []
    fusion = model.decision_fusion
    if not isinstance(fusion, CausalDecisionFusion):
        raise RuntimeError("checkpoint lacks causal decision fusion")

    steps = replay["steps"]
    for index in range(len(steps) - 1):
        row = steps[index][own_index]
        obs = row.get("observation") or {}
        if row.get("status") != "ACTIVE" or obs.get("select") is None:
            continue
        action = [int(value) for value in (steps[index + 1][own_index].get("action") or [])]
        router.observe(obs, scope="game_root", depth=len(board_history))
        board = features.build_board_tokens(obs, deck)
        board_history.append(board)
        action_history.append(previous_action)
        board_history = board_history[-int(model.max_context) :]
        action_history = action_history[-int(model.max_context) :]
        raw_context = (obs.get("select") or {}).get("context")
        if int(raw_context) == 41:
            previous_action = features.build_option_tokens(obs, [action])
            continue
        stages = features.factorized_teacher_forcing_stages(obs, action)
        if not stages:
            previous_action = features.build_option_tokens(obs, [action])
            continue
        first_options = features.build_option_tokens(obs, stages[0][0])
        route = int(router.candidate_model_route)
        with torch.inference_mode():
            model_out = model.forward_history_batch(
                [board_history],
                [first_options],
                n_options=[len(stages[0][0])],
                previous_action_histories=[action_history],
                matchup_routes=[route],
            )
            state = model_out["state_vec"]
            spatial = model_out["spatial_memory"]
            policy_value_state = model.matchup_policy_value_state(state, [route])
            for stage_index, (combos, target) in enumerate(stages):
                output.append(
                    record_stage(
                        model=model,
                        fusion=fusion,
                        obs=obs,
                        deck=deck,
                        combos=[list(combo) for combo in combos],
                        target=int(target),
                        state=state,
                        spatial=spatial,
                        policy_value_state=policy_value_state,
                        metadata={
                            "episode_id": int(meta["episode_id"]),
                            "iteration": int(meta["iteration"]),
                            "win": bool(meta["win"]),
                            "opponent_archetype": meta.get("opponent_archetype"),
                            "route": route,
                            "step_index": index,
                            "factorized_stage": stage_index,
                        },
                    )
                )
        previous_action = features.build_option_tokens(obs, [action])
    return output


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def group_summary(group: list[dict[str, Any]]) -> dict[str, Any]:
        guide_rows = [row for row in group if row["guide_target"] is not None]
        head_summary = {}
        head_names = sorted(group[0]["heads"]) if group else []
        for name in head_names:
            records = [row["heads"][name] for row in group]
            guide_effects = [
                record["guide_relative_effect"]
                for record in records
                if record["guide_relative_effect"] is not None
            ]
            head_summary[name] = {
                "mean_abs_logit_effect": mean(record["mean_abs_logit_effect"] for record in records),
                "p95_max_abs_logit_effect": percentile(
                    [record["max_abs_logit_effect"] for record in records], 0.95
                ),
                "choice_flip_rate_when_ablated": mean(
                    record["changes_model_choice_when_ablated"] for record in records
                ),
                "mean_target_margin_effect": mean(
                    record["target_margin_effect"] for record in records
                ),
                "mean_guide_relative_effect": mean(guide_effects) if guide_effects else None,
            }
        return {
            "stages": len(group),
            "model_realized_action_agreement": mean(
                row["model_matches_realized_action"] for row in group
            ) if group else None,
            "guide_rows": len(guide_rows),
            "guide_realized_action_agreement": mean(
                row["guide_matches_realized_action"] for row in guide_rows
            ) if guide_rows else None,
            "guide_model_agreement": mean(
                row["guide_matches_model"] for row in guide_rows
            ) if guide_rows else None,
            "routes": dict(Counter(str(row["route"]) for row in group)),
            "heads": head_summary,
        }

    groups: dict[str, list[dict[str, Any]]] = {
        "all": rows,
        "wins": [row for row in rows if row["win"]],
        "losses": [row for row in rows if not row["win"]],
    }
    for archetype in sorted({str(row["opponent_archetype"]) for row in rows}):
        groups[f"matchup:{archetype}"] = [
            row for row in rows if row["opponent_archetype"] == archetype
        ]
        groups[f"matchup:{archetype}:losses"] = [
            row
            for row in rows
            if row["opponent_archetype"] == archetype and not row["win"]
        ]
    return {name: group_summary(group) for name, group in groups.items() if group}


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    location = (len(values) - 1) * q
    low = int(math.floor(location))
    high = int(math.ceil(location))
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (location - low)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("replay_root", type=Path)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--matchup-tree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(max(1, int(args.threads)))

    episode_analysis = load(args.analysis)["episodes"]
    meta = {
        int(row["episode_id"]): row
        for row in episode_analysis
        if int(row["iteration"]) == args.iteration
    }
    model = load_model_from_checkpoint(args.checkpoint, device=torch.device("cpu"))
    model.eval()
    rows: list[dict[str, Any]] = []
    for path in sorted(
        (args.replay_root / f"iter_{args.iteration:05d}").glob("*-replay.json")
    ):
        eid = episode_id(path)
        if eid in meta:
            rows.extend(
                analyze_episode(
                    path=path,
                    meta=meta[eid],
                    model=model,
                    tree_path=args.matchup_tree,
                )
            )
    payload = {
        "schema": "poke_bot.kaggle_alakazam_head_attribution/v1",
        "iteration": args.iteration,
        "checkpoint": str(args.checkpoint),
        "matchup_tree": str(args.matchup_tree),
        "episodes": len(meta),
        "summary": summarize(rows),
        "stages": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"episodes": len(meta), "stages": len(rows), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
