"""Orchestrate Slowking distill stages A→E (+ optional D) under research-only authority."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

from .authority import PIPELINE_SCHEMA, RESEARCH_ONLY, TRAINING_AUTHORITY
from .bc_stage import OptionConditionedClone, StageAConfig, agreement_by_stage, run_stage_a_bc
from .belief_search_backend import BeliefSearchBundle, resolve_stage_c_search_fn
from .corpus import filter_split, load_jsonl
from .critical_search import run_stage_c_search
from .day_split import build_day_split, default_val_dates_for_window, write_split
from .distill_search import load_search_receipts, run_stage_e_distill
from .eval_gate import EvalGateConfig, evaluate_paired_games, write_eval_receipt
from .iql import IQLConfig, run_stage_b_iql
from .promotion import PromotionRequest, evaluate_promotion
from .self_play import run_population_self_play


@dataclass
class PipelineConfig:
    out_dir: Path
    val_dates: list[str] = field(default_factory=list)
    stage_a: StageAConfig = field(default_factory=StageAConfig)
    stage_b: IQLConfig = field(default_factory=IQLConfig)
    run_search: bool = True
    run_distill: bool = True
    run_self_play: bool = True
    max_critical_search: int = 0
    self_play_games_per_opponent: int = 2
    search_bundle: Optional[BeliefSearchBundle] = None
    eval_games: list[dict[str, Any]] = field(default_factory=list)
    eval_gate: EvalGateConfig = field(default_factory=EvalGateConfig)


def run_pipeline(
    decisions_jsonl: Path,
    *,
    config: PipelineConfig,
) -> dict[str, Any]:
    """Run day-split → A → B → C → E → eval gate on a decisions JSONL."""
    rows = load_jsonl(decisions_jsonl)
    if not rows:
        raise ValueError(f"no decisions in {decisions_jsonl}")

    dates = sorted({str(r.get("source_date") or "") for r in rows})
    val_dates = list(config.val_dates) or default_val_dates_for_window(dates)
    # Build game-level rows for splitting.
    games = {}
    for row in rows:
        games[str(row.get("game_id"))] = {
            "game_id": row.get("game_id"),
            "source_date": row.get("source_date"),
            "episode_id": row.get("episode_id"),
            "seat": row.get("seat"),
            "result": row.get("result"),
        }
    split = build_day_split(games.values(), val_dates=val_dates, require_dates=True)
    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    split_path = out / "day_split.json"
    split_sha = write_split(split, split_path)

    train_rows = filter_split(rows, game_ids=set(split.train_game_ids))
    val_rows = filter_split(rows, game_ids=set(split.val_game_ids))
    if not train_rows:
        # If only one day exists, train on all but keep empty val.
        train_rows = rows

    stage_a = run_stage_a_bc(train_rows, out_dir=out / "stage_a", config=config.stage_a)
    actor = OptionConditionedClone(config.stage_a.d_model)
    ckpt = torch.load(stage_a.checkpoint_path, map_location="cpu", weights_only=False)
    actor.load_state_dict(ckpt["state_dict"])

    # Mandatory ablation metrics on val (or train if val empty).
    eval_rows = val_rows or train_rows
    covered = agreement_by_stage(eval_rows, actor)
    abl_cfg = StageAConfig(**{**config.stage_a.__dict__, "zero_heuristic_channel": True, "epochs": 1})
    abl = run_stage_a_bc(train_rows, out_dir=out / "stage_a_ablation", config=abl_cfg)

    stage_b = run_stage_b_iql(
        train_rows,
        actor=actor,
        out_dir=out / "stage_b",
        config=config.stage_b,
    )
    # Reload actor if AWR updated it.
    b_ckpt = torch.load(stage_b.checkpoint_path, map_location="cpu", weights_only=False)
    actor.load_state_dict(b_ckpt["actor"])

    stage_c: dict[str, Any] = {"skipped": True}
    stage_d: dict[str, Any] = {"skipped": True}
    stage_e: dict[str, Any] = {"skipped": True}
    actor_ckpt_for_runtime = stage_b.checkpoint_path
    if config.run_search:
        search_fn, search_backend = resolve_stage_c_search_fn(config.search_bundle)
        stage_c = run_stage_c_search(
            train_rows,
            out_dir=out / "stage_c",
            search_fn=search_fn,
            max_critical=config.max_critical_search,
        )
        stage_c["search_backend"] = search_backend
        if config.run_distill and int(stage_c.get("n_receipts") or 0) > 0:
            receipts = load_search_receipts(Path(stage_c["receipts_path"]))
            distilled = run_stage_e_distill(
                train_rows,
                receipts,
                actor=actor,
                out_dir=out / "stage_e",
            )
            stage_e = {
                "metrics": distilled.metrics,
                "checkpoint_path": distilled.checkpoint_path,
            }
            if distilled.checkpoint_path:
                actor_ckpt_for_runtime = distilled.checkpoint_path
                e_ckpt = torch.load(
                    distilled.checkpoint_path, map_location="cpu", weights_only=False
                )
                state = e_ckpt.get("state_dict") or e_ckpt.get("actor")
                if state is not None:
                    actor.load_state_dict(state)

    if config.run_self_play and actor_ckpt_for_runtime:
        sp = run_population_self_play(
            actor_checkpoint=actor_ckpt_for_runtime,
            output_dir=out / "stage_d",
            games_per_opponent=int(config.self_play_games_per_opponent),
            seed=int(config.stage_a.seed),
        )
        stage_d = {
            "win_rate": sp.win_rate,
            "mean_prize_delta": sp.mean_prize_delta,
            "n_games": len(sp.games),
            "receipt_path": sp.receipt_path,
            "trajectories_path": sp.trajectories_path,
            "checkpoint_path": sp.checkpoint_path,
        }

    gate = evaluate_paired_games(
        config.eval_games,
        config=config.eval_gate,
        action_agreement=(covered.get("all") or {}).get("agreement"),
    )
    write_eval_receipt(gate, out / "eval_gate.json")

    promo = evaluate_promotion(
        PromotionRequest(
            eval_gate_passed=bool(gate.passed),
            paired_win_delta=float(gate.metrics.get("win_rate") or 0.0),
            action_agreement=float(
                ((covered.get("all") or {}).get("agreement")) or 0.0
            ),
            stage_d_win_rate=float(stage_d.get("win_rate") or 0.0),
            actor_checkpoint=str(actor_ckpt_for_runtime or ""),
        )
    )

    summary = {
        "schema": PIPELINE_SCHEMA,
        "research_only": RESEARCH_ONLY,
        "training_authority": TRAINING_AUTHORITY,
        "serving_authority": False,
        "decisions_jsonl": str(decisions_jsonl),
        "day_split_path": str(split_path),
        "day_split_sha256": split_sha,
        "split": split.to_json(),
        "stage_a": {
            "metrics": stage_a.metrics,
            "checkpoint_path": stage_a.checkpoint_path,
            "val_agreement": covered,
            "ablation_metrics": abl.metrics,
        },
        "stage_b": {
            "metrics": stage_b.metrics,
            "checkpoint_path": stage_b.checkpoint_path,
            "advantage_weighting_enabled": stage_b.advantage_weighting_enabled,
        },
        "stage_c": stage_c,
        "stage_d": stage_d,
        "stage_e": stage_e,
        "eval_gate": gate.metrics,
        "promotion": promo.to_json(),
        "actor_checkpoint": actor_ckpt_for_runtime,
        "promoted": False,  # hard: this pipeline never self-promotes
    }
    (out / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
