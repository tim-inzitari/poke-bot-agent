#!/usr/bin/env python3
"""Calibrate only a new temporal block on Alakazam acting-seat sequences.

The copied spatial trunk, option decoder, policy/value heads, and every
auxiliary head stay byte-identical.  A short validation-gated expert pass
teaches the identity-initialized temporal block to consume within-game history
while a latent teacher anchor keeps it close to the source stateless state.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import checkpoint
from poke_bot.feature_shards import load_feature_manifest
from poke_bot.pure_rl.model_profile import (
    count_params,
    pure_rl_history_model_config,
    validate_param_budget,
)
from poke_bot.train import TrainConfig, load_model_from_checkpoint, train_bootstrap
from scripts.run_alakazam_expert_bootstrap import resolve_filtered_manifest


TEMPORAL_TRAINABLE_PREFIXES = ("temporal_blocks.", "temporal_norm.")


def _non_temporal_copy_proof(
    seed_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
) -> dict[str, Any]:
    seed_state = dict(seed_payload.get("model_state_dict") or {})
    candidate_state = dict(candidate_payload.get("model_state_dict") or {})
    changed: list[str] = []
    checked = 0
    for name, expected in seed_state.items():
        if name.startswith(TEMPORAL_TRAINABLE_PREFIXES):
            continue
        actual = candidate_state.get(name)
        if actual is None or not torch.equal(expected.cpu(), actual.cpu()):
            changed.append(name)
        checked += 1
    unexpected = sorted(set(candidate_state) - set(seed_state))
    if changed or unexpected:
        raise RuntimeError(
            "frozen temporal calibration changed copied parameters: "
            f"changed={changed[:12]} unexpected={unexpected[:12]}"
        )
    return {
        "checked_frozen_tensors": checked,
        "changed_frozen_tensors": 0,
        "trainable_prefixes": list(TEMPORAL_TRAINABLE_PREFIXES),
    }


def calibrate(
    *,
    seed: Path,
    expert_manifest: Path,
    output: Path,
    run_name: str,
    epochs: int,
    patience: int,
    max_context: int,
    games_per_batch: int,
    max_decisions_per_batch: int,
    min_decisions: int,
) -> dict[str, Any]:
    seed = seed.expanduser().resolve()
    expert_manifest = expert_manifest.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest_path, manifest = resolve_filtered_manifest(
        expert_manifest,
        min_decisions=int(min_decisions),
    )
    dataset = load_feature_manifest(manifest_path, verify_hashes=True)
    if any(str(sequence.archetype).lower() != "alakazam" for sequence in dataset.sequences):
        raise ValueError("temporal calibration corpus contains a non-Alakazam acting seat")

    history_cfg = pure_rl_history_model_config(max_context=int(max_context))
    train_cfg = TrainConfig(
        lr=5e-5,
        weight_decay=1e-4,
        epochs=int(epochs),
        games_per_batch=int(games_per_batch),
        max_decisions_per_batch=int(max_decisions_per_batch),
        val_frac=0.10,
        split_by_episode=True,
        early_stop_patience=int(patience),
        value_loss_weight=1.0,
        aux_loss_weight=0.05,
        opp_hand_loss_weight=0.05,
        opp_remainder_loss_weight=0.05,
        alakazam_guide_loss_weight=0.05,
        lethal_threat_loss_weight=0.025,
        prize_race_loss_weight=0.025,
        grad_clip=1.0,
        amp=True,
        seed=20260720,
        history_identity_loss_weight=1.0,
    )
    seed_payload = checkpoint.load_checkpoint(seed, map_location="cpu")
    result = train_bootstrap(
        dataset,
        run_name=run_name,
        archetype_id="alakazam",
        train_cfg=train_cfg,
        resume=False,
        model_cfg=history_cfg,
        init_checkpoint=seed,
        checkpoint_extra={
            "pure_rl": True,
            "temporal_expert_calibration": {
                "expert_manifest": str(manifest_path),
                "selection": dict(manifest.get("selection") or {}),
                "sequence_scope": "acting_seat_game_bounded_rolling_prefix",
                "max_context": int(max_context),
                "cross_game_attention": False,
                "identity_loss_weight": 1.0,
                "epochs_cap": int(epochs),
                "early_stop_patience": int(patience),
                "games_per_batch": int(games_per_batch),
                "max_decisions_per_batch": int(max_decisions_per_batch),
            },
        },
        device_resident=False,
        trainable_parameter_prefixes=TEMPORAL_TRAINABLE_PREFIXES,
    )
    best_path = Path(str(result.get("best_path") or "")).expanduser().resolve()
    if not best_path.is_file():
        raise RuntimeError("temporal calibration did not publish a best checkpoint")
    candidate_payload = checkpoint.load_checkpoint(best_path, map_location="cpu")
    copy_proof = _non_temporal_copy_proof(seed_payload, candidate_payload)

    # The temporal-only Adam state cannot be restored into the subsequent
    # all-parameter RL optimizer. Publish a deliberate weights-only boundary.
    for key in ("optimizer_state_dict", "scaler_state_dict", "scheduler_state_dict"):
        candidate_payload.pop(key, None)
    model = load_model_from_checkpoint(best_path, device=torch.device("cpu"))
    params = count_params(model)
    validate_param_budget(params)
    extra = dict(candidate_payload.get("extra") or {})
    extra.update(
        {
            "pure_rl": True,
            "param_count": params,
            "optimizer_state_reset": True,
            "temporal_expert_calibration_result": {
                "best_metric": result.get("best_metric"),
                "epochs_completed": len(result.get("history") or []),
                "expert_records": int((manifest.get("totals") or {}).get("records_kept", 0)),
                "expert_decisions": int((manifest.get("totals") or {}).get("decisions_kept", 0)),
                "copy_proof": copy_proof,
            },
        }
    )
    candidate_payload["extra"] = extra
    candidate_payload["rl_iteration"] = int(seed_payload.get("rl_iteration", 0))
    candidate_payload["model_id"] = output.stem
    checkpoint.immutable_torch_save(candidate_payload, output)
    checkpoint.assert_trusted_policy_checkpoint(output)
    reloaded = load_model_from_checkpoint(output, device=torch.device("cpu"))
    if not (
        reloaded.decision_context == "history"
        and len(reloaded.temporal_blocks) == 1
        and reloaded.max_context == int(max_context)
        and reloaded.kv_cache_enabled
    ):
        raise RuntimeError("calibrated temporal checkpoint failed contract reload")
    return {
        "output": str(output),
        "digest": checkpoint.checkpoint_digest(output),
        "params": params,
        "expert_manifest": str(manifest_path),
        "expert_records": int((manifest.get("totals") or {}).get("records_kept", 0)),
        "expert_decisions": int((manifest.get("totals") or {}).get("decisions_kept", 0)),
        "max_context": int(max_context),
        "games_per_batch": int(games_per_batch),
        "max_decisions_per_batch": int(max_decisions_per_batch),
        "best_metric": result.get("best_metric"),
        "epochs_completed": len(result.get("history") or []),
        "copy_proof": copy_proof,
        "optimizer_state_reset": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--max-context", type=int, default=320)
    parser.add_argument("--games-per-batch", type=int, default=128)
    parser.add_argument("--max-decisions-per-batch", type=int, default=2048)
    parser.add_argument("--min-decisions", type=int, default=100000)
    args = parser.parse_args(argv)
    report = calibrate(
        seed=args.seed,
        expert_manifest=args.expert_manifest,
        output=args.output,
        run_name=args.run_name,
        epochs=args.epochs,
        patience=args.patience,
        max_context=args.max_context,
        games_per_batch=args.games_per_batch,
        max_decisions_per_batch=args.max_decisions_per_batch,
        min_decisions=args.min_decisions,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
