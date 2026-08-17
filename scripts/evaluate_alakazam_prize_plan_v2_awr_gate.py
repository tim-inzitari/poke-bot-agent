#!/usr/bin/env python3
"""Compare exact legacy and frozen-H3 AWR weights on a sealed policy shard."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.feature_shards import iter_feature_shard  # noqa: E402
from poke_bot.frozen_prize_plan_advantage import (  # noqa: E402
    PortableStageAdvantage,
    bind_portable_stage_advantages,
)
from poke_bot.recursive_turn_planner.recent20_overlay import sha256_file  # noqa: E402
from poke_bot.train import (  # noqa: E402
    TrainConfig,
    _iter_game_batches,
    evaluate,
    load_model_from_checkpoint,
)
from scripts.materialize_alakazam_prize_plan_v2_h3_cache import (  # noqa: E402
    CACHE_SCHEMA,
    RECEIPT_SCHEMA as CACHE_RECEIPT_SCHEMA,
)


SCHEMA = "poke_bot.alakazam_prize_plan_v2_h3_awr_gate_receipt/v1"


class AWRGateError(ValueError):
    pass


def _json(path: Path, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise AWRGateError(f"{label} must be a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AWRGateError(f"{label} must contain an object")
    return value


def _metric(value: Any) -> dict[str, Any]:
    return {
        "decisions": int(value.n_decisions),
        "raw_advantage_mean": float(value.raw_advantage_mean),
        "raw_advantage_std": float(value.raw_advantage_std),
        "normalized_advantage_mean": float(value.normalized_advantage_mean),
        "normalized_advantage_std": float(value.normalized_advantage_std),
        "awr_weight_mean": float(value.awr_weight_mean),
        "awr_weight_sum": float(value.awr_weight_sum),
        "awr_weight_sq_sum": float(value.awr_weight_sq_sum),
        "awr_weight_p50": float(value.awr_weight_p50),
        "awr_weight_p95": float(value.awr_weight_p95),
        "awr_weight_max_observed": float(value.awr_weight_max_observed),
        "awr_weight_clip_frac": float(value.awr_weight_clip_frac),
        "awr_effective_sample_size": float(value.awr_effective_sample_size),
        "awr_effective_sample_fraction": float(value.awr_effective_sample_fraction),
    }


def _offline_h3_metric(
    sequences: list[Any],
    *,
    cfg: TrainConfig,
    baseline: dict[tuple[int, int, int], float],
    additive: dict[tuple[int, int, int], float],
) -> dict[str, Any]:
    """Recompute exact AWR diagnostics from the captured frozen value baseline.

    H3 changes only the scalar advantage.  Re-running the policy forward is
    therefore redundant and can introduce no additional evidence.  Preserve
    the exact game-batch partition, per-batch whitening, beta, and clipping
    used by ``batch_losses`` while aggregating the same sufficient statistics.
    """

    raw_values: list[torch.Tensor] = []
    normalized_values: list[torch.Tensor] = []
    weights_all: list[torch.Tensor] = []
    unclipped_at_max = 0
    decisions = 0
    batches = _iter_game_batches(
        list(sequences),
        cfg.games_per_batch,
        max(int(cfg.max_decisions_per_batch), int(cfg.agreement_max_decisions_per_batch)),
        shuffle=False,
        seed=cfg.seed,
        epoch=0,
    )
    seen: set[tuple[int, int, int]] = set()
    beta = max(float(cfg.awr_beta), 1e-6)
    wmax = max(float(cfg.awr_weight_max), 1e-6)
    for batch in batches:
        keys: list[tuple[int, int, int]] = []
        target: list[float] = []
        for game in batch:
            for decision_index, decision in enumerate(game.decisions):
                for stage_index, _stage in enumerate(decision.policy_stages):
                    key = (id(game), decision_index, stage_index)
                    keys.append(key)
                    target.append(float(game.value))
        if not keys:
            continue
        missing_baseline = [key for key in keys if key not in baseline]
        missing_additive = [key for key in keys if key not in additive]
        if missing_baseline or missing_additive:
            raise AWRGateError(
                "captured baseline/additive cache lacks exact batch coverage"
            )
        if any(key in seen for key in keys):
            raise AWRGateError("AWR batch planner repeated a decision-stage key")
        seen.update(keys)
        # ``batch_losses`` casts each input to the value head's FP32 dtype,
        # subtracts the baseline, then performs a separate additive update.
        # Preserve both dtype *and operation order* so whitening and weights
        # match the live trainer rather than Python's float64 arithmetic.
        target_tensor = torch.tensor(target, dtype=torch.float32)
        baseline_tensor = torch.tensor(
            [baseline[key] for key in keys], dtype=torch.float32
        )
        additive_tensor = torch.tensor(
            [additive[key] for key in keys], dtype=torch.float32
        )
        raw = target_tensor - baseline_tensor
        raw = raw + additive_tensor
        if not torch.isfinite(raw).all():
            raise AWRGateError("offline H3 raw advantage is non-finite")
        normalized = raw
        if cfg.awr_normalize_advantages and raw.numel() > 1:
            normalized = (raw - raw.mean()) / raw.std(unbiased=False).clamp_min(1e-6)
        unclipped = torch.exp(normalized / beta)
        weights = torch.clamp(unclipped, max=wmax)
        raw_values.append(raw)
        normalized_values.append(normalized)
        weights_all.append(weights)
        unclipped_at_max += int((unclipped >= wmax).sum().item())
        decisions += len(keys)
    if set(baseline) != seen or set(additive) != seen:
        raise AWRGateError("offline H3 computation has surplus cache rows")
    if not raw_values:
        raise AWRGateError("offline H3 computation has no decision-stage rows")
    raw = torch.cat(raw_values)
    normalized = torch.cat(normalized_values)
    weights = torch.sort(torch.cat(weights_all)).values
    weight_sum = float(weights.sum().item())
    weight_sq_sum = float(weights.square().sum().item())
    ess = min(float(decisions), weight_sum * weight_sum / max(weight_sq_sum, 1e-12))
    return {
        "decisions": decisions,
        "raw_advantage_mean": float(raw.mean().item()),
        "raw_advantage_std": float(raw.std(unbiased=False).item()),
        "normalized_advantage_mean": float(normalized.mean().item()),
        "normalized_advantage_std": float(normalized.std(unbiased=False).item()),
        "awr_weight_mean": float(weights.mean().item()),
        "awr_weight_sum": weight_sum,
        "awr_weight_sq_sum": weight_sq_sum,
        "awr_weight_p50": float(weights[int(0.50 * (decisions - 1))].item()),
        "awr_weight_p95": float(weights[int(0.95 * (decisions - 1))].item()),
        "awr_weight_max_observed": float(weights[-1].item()),
        "awr_weight_clip_frac": float(unclipped_at_max) / float(decisions),
        "awr_effective_sample_size": ess,
        "awr_effective_sample_fraction": ess / float(decisions),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if sha256_file(args.policy_checkpoint) != args.policy_checkpoint_sha256:
        raise AWRGateError("policy checkpoint digest mismatch")
    cache_receipt = _json(args.cache_receipt, "cache receipt")
    if (
        cache_receipt.get("schema") != CACHE_RECEIPT_SCHEMA
        or cache_receipt.get("cache_value_semantics") != "h3_additive_term_only"
        or cache_receipt.get("exact_legacy_baseline_computed_in_batch") is not True
        or cache_receipt.get("actor_activation") is not False
    ):
        raise AWRGateError("cache receipt is not an inactive H3-additive artifact")
    rows = [item for item in cache_receipt.get("day_shards") or () if item.get("utc_day") == args.utc_day]
    if len(rows) != 1:
        raise AWRGateError("cache receipt does not bind exactly one requested day shard")
    cache_shard = Path(args.cache_receipt).resolve().parent / str(rows[0]["path"])
    if sha256_file(cache_shard) != rows[0].get("sha256"):
        raise AWRGateError("cache day shard digest mismatch")
    records: list[PortableStageAdvantage] = []
    observed = 0
    with cache_shard.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                raise AWRGateError("cache shard contains blank rows")
            row = json.loads(raw)
            if row.get("schema") != CACHE_SCHEMA or row.get("utc_day") != args.utc_day:
                raise AWRGateError("cache row identity drifted")
            observed += 1
            for stage in range(int(row["stage_count"])):
                records.append(
                    PortableStageAdvantage(
                        str(row["policy_episode_id"]),
                        int(row["acting_seat"]),
                        int(row["env_step"]),
                        stage,
                        float(row["h3_additive_term"]),
                    )
                )
    if observed != int(rows[0]["rows"]):
        raise AWRGateError("cache row count mismatch")
    policy_shard = Path(args.policy_shard).expanduser().resolve()
    sequences = list(iter_feature_shard(policy_shard))
    cache = bind_portable_stage_advantages(sequences, records)
    device = torch.device(args.device)
    model = load_model_from_checkpoint(args.policy_checkpoint, device=device)
    cfg = TrainConfig.pure_rl_defaults(
        games_per_batch=int(args.games_per_batch),
        max_decisions_per_batch=int(args.max_decisions_per_batch),
        setup_board_outcome_loss_weight=0.0,
        combo_state_loss_weight=0.0,
        capture_awr_weight_distribution=True,
    )
    captured_baseline: dict[tuple[int, int, int], float] = {}
    legacy = evaluate(
        model,
        sequences,
        cfg=cfg,
        desc="legacy-awr-gate",
        awr_capture_baseline=captured_baseline,
        allow_masked_own_deck_ledger_rows=True,
    )
    legacy_metrics = _metric(legacy)
    h3_metrics = _offline_h3_metric(
        sequences,
        cfg=cfg,
        baseline=captured_baseline,
        additive=cache,
    )
    if legacy_metrics["decisions"] != h3_metrics["decisions"]:
        raise AWRGateError("legacy/H3 decision membership changed")
    result = {
        "schema": SCHEMA,
        "utc_day": args.utc_day,
        "policy_shard_sha256": sha256_file(policy_shard),
        "policy_checkpoint_sha256": args.policy_checkpoint_sha256,
        "cache_receipt_sha256": sha256_file(args.cache_receipt),
        "configuration": {
            "awr_beta": 0.5,
            "awr_weight_max": 20.0,
            "awr_normalize_advantages": True,
            "games_per_batch": cfg.games_per_batch,
            "max_decisions_per_batch": cfg.max_decisions_per_batch,
        },
        "legacy": legacy_metrics,
        "h3": h3_metrics,
        "comparison": {
            "ess_fraction_ratio_h3_over_legacy": h3_metrics["awr_effective_sample_fraction"] / legacy_metrics["awr_effective_sample_fraction"],
            "clip_fraction_absolute_delta": h3_metrics["awr_weight_clip_frac"] - legacy_metrics["awr_weight_clip_frac"],
            "same_replay_membership_and_weights": True,
            "single_policy_forward_with_exact_frozen_baseline_reuse": True,
        },
        "finite": all(
            math.isfinite(float(value))
            for arm in (legacy_metrics, h3_metrics)
            for value in arm.values()
        ),
        "actor_activation": False,
        "activation_eligible": False,
    }
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise AWRGateError("AWR gate receipt is create-only")
    data = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(fd, data); os.fsync(fd)
    finally:
        os.close(fd)
    result["receipt_sha256"] = sha256_file(output)
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--utc-day", required=True)
    p.add_argument("--policy-shard", type=Path, required=True)
    p.add_argument("--policy-checkpoint", type=Path, required=True)
    p.add_argument("--policy-checkpoint-sha256", required=True)
    p.add_argument("--cache-receipt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "mps"), default="mps")
    p.add_argument("--games-per-batch", type=int, default=16)
    p.add_argument("--max-decisions-per-batch", type=int, default=2048)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    try:
        print(json.dumps(run(parser().parse_args(argv)), sort_keys=True), flush=True)
    except (AWRGateError, OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
