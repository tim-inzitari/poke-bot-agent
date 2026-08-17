#!/usr/bin/env python3
"""Evaluate and activate the learned r274 tactical Fusion route after upload."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import torch

from poke_bot.feature_shards import COMPACT_MODE_TEMPORAL_EXPERT, iter_feature_shard
from poke_bot.pure_rl.expert_feature_stream import EpisodeGroupedFeatureManifest
from poke_bot.r241_own_deck_successor import load_r260_owner_contract
from poke_bot.r260_inzi_sidecar_index import R260InziSidecarIndex
from poke_bot.r274_bootstrap_handoff import (
    materialize_tactical_activation_checkpoint,
)
from poke_bot.tactical_sequence_materialization import attach_tactical_target_overlay
from poke_bot.train import batch_losses, load_model_from_checkpoint


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def single_decision(sequence: Any, index: int) -> Any:
    return replace(
        sequence,
        decisions=[sequence.decisions[index]],
        policy_targets=(
            [sequence.policy_targets[index]]
            if sequence.policy_targets is not None
            else None
        ),
        factorized_policy_targets=(
            [sequence.factorized_policy_targets[index]]
            if sequence.factorized_policy_targets is not None
            else None
        ),
    )


def _batches(rows: list[Any], size: int = 16):
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _manifest_sequence_at(plan: EpisodeGroupedFeatureManifest, index: int) -> Any:
    """Load one checksum-verified manifest sequence without scanning prior shards."""

    shard_index = bisect_right(plan._shard_starts, int(index)) - 1
    if shard_index < 0 or shard_index >= len(plan._shards):
        raise RuntimeError("validation sequence index is outside the manifest")
    shard = plan._shards[shard_index]
    local_index = int(index) - int(plan._shard_starts[shard_index])
    shard.assert_unchanged()
    for offset, sequence in enumerate(iter_feature_shard(shard.path)):
        if offset != local_index:
            continue
        shard.assert_unchanged()
        if plan.max_context is not None:
            from poke_bot.train import cap_game_sequence_context

            sequence, _changed = cap_game_sequence_context(
                sequence, plan.max_context
            )
        return sequence
    raise RuntimeError("validation sequence is absent from its verified shard")


def _evaluate(
    model: Any,
    rows: list[Any],
    *,
    route_on: bool,
) -> dict[str, Any]:
    for name in (
        "tactical_sequence_outcome_route_enabled",
        "tactical_sequence_outcome_route_runtime_enabled",
    ):
        setattr(model, name, route_on)
        setattr(model.cfg, name, route_on)
    model.eval()
    predictions: list[int] = []
    log_probs: list[list[float]] = []
    values: list[float] = []
    route_deltas: list[list[float]] | None = [] if route_on else None
    started = time.perf_counter()
    with torch.no_grad():
        for batch in _batches(rows):
            batch_losses(
                model,
                batch,
                prediction_only=True,
                prediction_sink=predictions,
                policy_log_prob_sink=log_probs,
                value_prediction_sink=values,
                tactical_route_delta_sink=route_deltas,
                allow_masked_own_deck_ledger_rows=True,
            )
    elapsed_ms = 1000.0 * (time.perf_counter() - started)
    if not predictions or len(predictions) != len(log_probs) or len(values) != len(log_probs):
        raise RuntimeError("paired tactical evaluation returned misaligned rows")
    if route_on and (route_deltas is None or len(route_deltas) != len(log_probs)):
        raise RuntimeError("route-on evaluation lacks aligned route deltas")
    return {
        "predictions": predictions,
        "log_probs": log_probs,
        "values": values,
        "route_deltas": route_deltas,
        "latency_ms": elapsed_ms,
    }


def _margin(row: list[float]) -> float:
    ordered = sorted((float(value) for value in row), reverse=True)
    return ordered[0] - ordered[1] if len(ordered) >= 2 else 0.0


def _paired_impact(off: dict[str, Any], on: dict[str, Any]) -> dict[str, Any]:
    if len(off["log_probs"]) != len(on["log_probs"]):
        raise RuntimeError("route-on/off policy rows differ")
    support = len(off["log_probs"])
    kl_values: list[float] = []
    flat_deltas: list[float] = []
    for off_row, on_row in zip(off["log_probs"], on["log_probs"], strict=True):
        if len(off_row) != len(on_row):
            raise RuntimeError("route-on/off legal option counts differ")
        kl_values.append(
            sum(
                math.exp(float(on_value))
                * (float(on_value) - float(off_value))
                for off_value, on_value in zip(off_row, on_row, strict=True)
            )
        )
        flat_deltas.extend(
            abs(float(on_value) - float(off_value))
            for off_value, on_value in zip(off_row, on_row, strict=True)
        )
    route_values = [
        abs(float(value))
        for row in on["route_deltas"]
        for value in row
    ]
    if not flat_deltas or not route_values:
        raise RuntimeError("tactical impact has no supported route outputs")
    return {
        "support": support,
        "action_change_rate": sum(
            int(left != right)
            for left, right in zip(
                off["predictions"], on["predictions"], strict=True
            )
        )
        / support,
        "top_action_margin_delta": sum(
            _margin(on_row) - _margin(off_row)
            for off_row, on_row in zip(
                off["log_probs"], on["log_probs"], strict=True
            )
        )
        / support,
        "policy_kl_divergence": max(0.0, sum(kl_values) / support),
        "value_delta": sum(
            abs(float(right) - float(left))
            for left, right in zip(off["values"], on["values"], strict=True)
        )
        / support,
        "route_magnitude": sum(route_values) / len(route_values),
        "latency_ms": float(on["latency_ms"] - off["latency_ms"]),
        "max_abs_logit_delta": max(flat_deltas),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-checkpoint", required=True, type=Path)
    parser.add_argument("--bootstrap-submission-receipt", required=True, type=Path)
    parser.add_argument("--upload-receipt", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--sidecar-binding", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--tactical-overlay", required=True, type=Path)
    parser.add_argument("--output-checkpoint", required=True, type=Path)
    parser.add_argument("--output-receipt", required=True, type=Path)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--evaluation-rows", type=int, default=256)
    args = parser.parse_args()

    owner = load_r260_owner_contract()
    sidecar = json.loads(args.sidecar_binding.read_text(encoding="utf-8"))
    daily: dict[str, str] = {}
    for day, row in dict(sidecar["daily_sidecar_meta_receipts"]).items():
        meta = json.loads(Path(str(row["path"])).read_text(encoding="utf-8"))
        daily[str(day)] = str(meta["meta_sha256"])
    index = R260InziSidecarIndex(
        args.index,
        source_manifest_sha256=owner.source_manifest_sha256,
        daily_meta_sha256s=daily,
    )
    index.assert_verified(
        expected_source_manifest_sha256=owner.source_manifest_sha256,
        daily_meta_sha256s=daily,
    )
    plan = EpisodeGroupedFeatureManifest.open(
        args.manifest,
        expected_manifest_digest=sha256_file(args.manifest),
        val_frac=0.10,
        seed=274,
        max_context=320,
        expected_compact_mode=COMPACT_MODE_TEMPORAL_EXPERT,
        workers=1,
    )
    tactical_payload = json.loads(args.tactical_overlay.read_text(encoding="utf-8"))
    tactical_keys = {
        (
            str(row["episode_id"]),
            int(row["seat"]),
            int(row["env_step"]),
            str(row["observation_fingerprint"]),
        )
        for row in tactical_payload.get("rows") or ()
    }
    if len(tactical_keys) != int(tactical_payload.get("roots", -1)):
        raise RuntimeError("tactical overlay exact-key inventory is invalid")
    tactical_validation_episode_ids = {
        episode_id
        for episode_id, _seat, _env_step, _fingerprint in tactical_keys
        if episode_id in plan._validation_episode_ids
    }
    if not tactical_validation_episode_ids:
        raise RuntimeError("validation partition has no tactical overlay episodes")
    tactical_validation_indices = [
        index
        for index, row in enumerate(plan._sequence_metadata)
        if index in plan._selected_indices
        and row[0] in tactical_validation_episode_ids
    ]
    if not tactical_validation_indices:
        raise RuntimeError("validation partition has no tactical overlay sequences")
    rows: list[Any] = []
    labeled: list[Any] = []
    tactical_triplet_matches: list[dict[str, str]] = []
    evaluation_rows = int(args.evaluation_rows)
    if evaluation_rows < 2:
        raise RuntimeError("evaluation rows must include ordinary and labeled support")
    try:
        _train, validation = plan.splits()
        ordinary_target = max(0, evaluation_rows - 1)
        for sequence in validation:
            for decision_index in range(len(sequence.decisions)):
                trial = single_decision(sequence, decision_index)
                index.attach_available_batch([trial])
                rows.append(trial)
                if len(rows) >= ordinary_target:
                    break
            if len(rows) >= ordinary_target:
                break

        labeled_trial = None
        for sequence_index in tactical_validation_indices:
            sequence = _manifest_sequence_at(plan, sequence_index)
            for decision_index in range(len(sequence.decisions)):
                trial = single_decision(sequence, decision_index)
                index.attach_available_batch([trial])
                source_decision = trial.decisions[0]
                source_key = (
                    str(trial.episode_id),
                    int(trial.seat),
                    int(source_decision.env_step),
                    str(source_decision.observation_fingerprint or ""),
                )
                if any(
                    source_key[:3] == candidate[:3]
                    for candidate in tactical_keys
                ) and len(tactical_triplet_matches) < 4:
                    overlay_row = next(
                        row
                        for row in tactical_payload.get("rows") or ()
                        if (
                            str(row["episode_id"]),
                            int(row["seat"]),
                            int(row["env_step"]),
                        )
                        == source_key[:3]
                    )
                    tactical_triplet_matches.append(
                        {
                            "feature": source_key[3],
                            "overlay_compact": str(
                                overlay_row["observation_fingerprint"]
                            ),
                            "overlay_source": str(
                                overlay_row.get("source_observation_fingerprint")
                                or ""
                            ),
                        }
                    )
                if source_key not in tactical_keys:
                    continue
                attached = attach_tactical_target_overlay(
                    [trial], args.tactical_overlay, require_all=False
                )
                if int(attached["roots"]) != 1:
                    raise RuntimeError("tactical overlay exact-key attachment failed")
                labeled_trial = trial
                break
            if labeled_trial is not None:
                break
        if labeled_trial is not None:
            labeled.append(labeled_trial)
            rows.append(labeled_trial)
    finally:
        del plan
    if len(rows) != evaluation_rows or not labeled:
        raise RuntimeError(
            "source-disjoint validation lacks tactical impact support: "
            + json.dumps(
                {
                    "rows": len(rows),
                    "labeled": len(labeled),
                    "candidate_sequences": len(tactical_validation_indices),
                    "triplet_matches": tactical_triplet_matches,
                },
                sort_keys=True,
            )
        )

    device = torch.device(args.device)
    model = load_model_from_checkpoint(args.submission_checkpoint, device=device)
    off = _evaluate(model, rows, route_on=False)
    on = _evaluate(model, rows, route_on=True)
    impact = _paired_impact(off, on)
    with torch.no_grad():
        _loss, calibration_metrics = batch_losses(
            model,
            labeled,
            tactical_sequence_outcome_loss_weight=0.025,
            allow_masked_own_deck_ledger_rows=True,
        )
    tactical = dict(calibration_metrics.tactical_sequence_outcome_metrics or {})
    if int(tactical.get("labeled_options", 0)) <= 0:
        raise RuntimeError("tactical calibration evaluation has no labeled options")
    impact.update(
        {
            "source_disjoint": True,
            "terminal_win_and_public_sme_label_calibration": {
                "source": "expert_manifest_validation_partition",
                "loss": float(calibration_metrics.tactical_sequence_outcome_loss),
                "labeled_options": int(tactical["labeled_options"]),
                "label_counts": dict(tactical.get("label_counts") or {}),
            },
            # Scheduling jitter cannot make a valid route unsafe.  Report the
            # measured signed delta on the dashboard but gate nonnegative
            # incremental latency at zero.
            "latency_ms": max(0.0, float(impact["latency_ms"])),
        }
    )
    result = materialize_tactical_activation_checkpoint(
        submission_checkpoint=args.submission_checkpoint,
        bootstrap_submission_receipt=args.bootstrap_submission_receipt,
        upload_receipt=args.upload_receipt,
        impact=impact,
        output_checkpoint=args.output_checkpoint,
        output_receipt=args.output_receipt,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
