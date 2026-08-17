#!/usr/bin/env python3
"""Precompute the frozen H3 additive cache for sealed policy sequences."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.feature_shards import iter_feature_shard  # noqa: E402
from poke_bot.prize_plan_v2_sidecar import (  # noqa: E402
    PRIZE_PLAN_V2_HORIZONS,
    load_prize_plan_v2_checkpoint,
)
from poke_bot.recursive_turn_planner.recent20_overlay import sha256_file  # noqa: E402
from scripts.build_alakazam_prize_plan_v2_c3 import C3_SCHEMA  # noqa: E402
from scripts.train_alakazam_prize_plan_v2 import (  # noqa: E402
    H3_SCALE_SUPPORT_SCHEMA,
    _public_action_signature,
    batched,
    canonical_bytes,
    critic_predictions,
    iter_complete_action_examples,
    open_sealed_inputs,
    resolve_device,
)


CACHE_SCHEMA = "poke_bot.alakazam_prize_plan_v2_h3_policy_additive_cache/v1"
RECEIPT_SCHEMA = "poke_bot.alakazam_prize_plan_v2_h3_policy_additive_cache_receipt/v1"
COEFFICIENT = 0.025


class H3CacheError(ValueError):
    pass


def _json(path: Path, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise H3CacheError(f"{label} must be a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise H3CacheError(f"{label} must contain an object")
    return value


def _canonical_sha(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def _atomic_json(path: Path, value: Any) -> str:
    if path.exists():
        raise H3CacheError(f"create-only output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("." + path.name + f".partial-{os.getpid()}")
    data = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.link(temp, path)
    temp.unlink()
    return sha256_file(path)


def _validate_cache_jsonl(path: Path, expected_rows: int) -> None:
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                raise H3CacheError(
                    f"cache shard contains a blank JSONL record at line {line_number}"
                )
            value = json.loads(raw)
            if not isinstance(value, Mapping) or value.get("schema") != CACHE_SCHEMA:
                raise H3CacheError("cache shard row schema drifted")
            rows += 1
    if rows != expected_rows:
        raise H3CacheError(
            f"cache shard row count mismatch: expected {expected_rows}, observed {rows}"
        )


def _policy_days(manifest: Mapping[str, Any], root: Path) -> dict[str, tuple[Path, bool]]:
    rows = manifest.get("shards")
    if manifest.get("compact_mode") != "temporal-expert-v1" or not isinstance(rows, list):
        raise H3CacheError("policy manifest is not temporal-expert-v1")
    result: dict[str, tuple[Path, bool]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise H3CacheError("policy manifest row is malformed")
        days = row.get("source_dates")
        if not isinstance(days, list) or len(days) != 1:
            raise H3CacheError("policy shard must bind exactly one UTC day")
        day = str(days[0])
        path = (root / str(row.get("path") or "")).resolve()
        if path.parent != root or day in result or sha256_file(path) != row.get("sha256"):
            raise H3CacheError("policy shard identity/path drifted")
        cache_only = bool(
            row.get("critic_cache_only") is True
            and row.get("expert_training_eligible") is False
            and isinstance(row.get("canonical_alignment_receipt_sha256"), str)
            and str(row.get("canonical_alignment_receipt_sha256")).startswith("sha256:")
        )
        result[day] = (path, cache_only)
    return result


def _policy_index(path: Path) -> dict[tuple[str, int, int], tuple[tuple[int, ...], tuple[int, ...]]]:
    result: dict[tuple[str, int, int], tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for sequence in iter_feature_shard(path):
        for decision in sequence.decisions:
            key = (str(sequence.episode_id), int(sequence.seat), int(decision.env_step))
            if key in result:
                raise H3CacheError("policy corpus repeats an episode/seat/step key")
            result[key] = (
                tuple(int(stage.target_index) for stage in decision.policy_stages),
                tuple(len(stage.options.offset) for stage in decision.policy_stages),
            )
    if not result:
        raise H3CacheError("policy shard has no decisions")
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-view", type=Path, required=True)
    p.add_argument("--training-view-sha256", required=True)
    p.add_argument("--target-view", type=Path, required=True)
    p.add_argument("--target-view-sha256", required=True)
    p.add_argument("--historical-contract", type=Path, required=True)
    p.add_argument("--historical-contract-sha256", required=True)
    p.add_argument("--policy-corpus-root", type=Path, required=True)
    p.add_argument("--policy-transfer-receipt", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--checkpoint-sha256", required=True)
    p.add_argument("--h3-scale-support", type=Path, required=True)
    p.add_argument("--h3-scale-support-sha256", required=True)
    p.add_argument("--c3", type=Path, required=True)
    p.add_argument("--c3-sha256", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "mps", "auto"), default="mps")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--only-day", default="")
    return p


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 1:
        raise H3CacheError("batch size must be positive")
    for path, expected, label in (
        (args.checkpoint, args.checkpoint_sha256, "checkpoint"),
        (args.h3_scale_support, args.h3_scale_support_sha256, "H3 scale"),
        (args.c3, args.c3_sha256, "c3"),
    ):
        if sha256_file(path) != expected:
            raise H3CacheError(f"{label} digest mismatch")
    scale = _json(args.h3_scale_support, "H3 scale")
    c3 = _json(args.c3, "c3")
    if (
        scale.get("schema") != H3_SCALE_SUPPORT_SCHEMA
        or scale.get("full_train_split_consumed") is not True
        or c3.get("schema") != C3_SCHEMA
        or c3.get("full_train_split_consumed") is not True
        or c3.get("h3_scale_support_sha256") != args.h3_scale_support_sha256
        or c3.get("critic_checkpoint_sha256") != args.checkpoint_sha256
    ):
        raise H3CacheError("scale/c3 frozen identity mismatch")
    denominator = float(scale.get("frozen_train_h3_predicted_advantage_rms_floor"))
    if not math.isfinite(denominator) or denominator <= 0:
        raise H3CacheError("H3 scale denominator is invalid")
    table = c3.get("support_table")
    if not isinstance(table, Mapping):
        raise H3CacheError("c3 support table is absent")

    policy_root = Path(args.policy_corpus_root).expanduser().resolve()
    transfer = _json(args.policy_transfer_receipt, "policy transfer receipt")
    if transfer.get("destination_root") != str(policy_root) or transfer.get("source_mutated") is not False:
        raise H3CacheError("policy transfer receipt/root mismatch")
    manifest_path = policy_root / "manifest.json"
    if sha256_file(manifest_path) != transfer.get("manifest_sha256"):
        raise H3CacheError("policy manifest is not transfer-receipt-bound")
    policy_days = _policy_days(_json(manifest_path, "policy manifest"), policy_root)

    sealed = SimpleNamespace(
        contract=args.historical_contract,
        contract_sha256=args.historical_contract_sha256,
        training_view=args.training_view,
        training_view_sha256=args.training_view_sha256,
        target_view=args.target_view,
        target_view_sha256=args.target_view_sha256,
        target_set_root=None,
        target_manifest=None,
        target_manifest_sha256=None,
        target_set_receipt=None,
        target_set_receipt_sha256=None,
        test_mode=False,
        test_allow_noncanonical_split=False,
        test_skip_input_shard_sha256=False,
    )
    dataset, targets, split_days, source_binding = open_sealed_inputs(sealed)
    device = resolve_device(args.device)
    loaded = load_prize_plan_v2_checkpoint(args.checkpoint, device=device)
    model = loaded.model.to(device=device, dtype=torch.float32).eval()
    h3 = PRIZE_PLAN_V2_HORIZONS.index(3)
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise H3CacheError("cache output directory is create-only")
    output.mkdir(parents=True)
    summaries: list[dict[str, Any]] = []
    try:
        for split in ("train", "validation", "evaluation"):
            current_day = ""
            index: dict[tuple[str, int, int], tuple[tuple[int, ...], tuple[int, ...]]] = {}
            consumed: set[tuple[str, int, int]] = set()
            matched_rows: list[Any] = []
            matched_keys: list[tuple[str, int, int]] = []
            handle = None
            temp = None
            sha = hashlib.sha256()
            count = 0
            masked = 0
            unseen = 0
            allow_unmatched_policy = False
            missing_policy_from_sealed_target = 0
            mismatched_policy_replaced_by_sealed_target = 0

            def finish_day() -> None:
                nonlocal handle, temp, count, masked, unseen
                if not current_day:
                    return
                extra_policy = len(set(index) - consumed)
                if extra_policy and not allow_unmatched_policy:
                    raise H3CacheError(
                        f"policy/critic complete-action coverage mismatch for {current_day}: "
                        f"{extra_policy} policy decisions unmatched"
                    )
                assert handle is not None and temp is not None
                handle.flush(); os.fsync(handle.fileno()); handle.close()
                digest = "sha256:" + sha.hexdigest()
                final = output / f"{digest[7:]}.{current_day}.h3-additive.jsonl"
                os.link(temp, final); temp.unlink()
                _validate_cache_jsonl(final, count)
                summaries.append({"utc_day": current_day, "split": split, "path": final.name, "sha256": digest, "rows": count, "h3_masked": masked, "unseen_c3": unseen, "policy_cache_only_unmatched_excluded": extra_policy, "policy_cache_only_missing_reconstructed_from_sealed_target": missing_policy_from_sealed_target, "policy_cache_only_mismatched_replaced_by_sealed_target": mismatched_policy_replaced_by_sealed_target})
                handle = None

            def emit_batch() -> None:
                nonlocal count, masked, unseen
                if not matched_rows:
                    return
                with torch.no_grad():
                    pred = critic_predictions(model, matched_rows, device=device).detach().cpu()
                for row, key, values in zip(matched_rows, matched_keys, pred.tolist(), strict=True):
                    signature = _public_action_signature(row)
                    support = table.get(signature)
                    confidence = float(support.get("c3")) if isinstance(support, Mapping) else 0.0
                    available = bool(row.plan_masks[h3])
                    scaled = (float(values[2 * h3 + 1]) - float(values[2 * h3])) / denominator
                    addend = COEFFICIENT * confidence * scaled if available else 0.0
                    if not math.isfinite(addend):
                        raise H3CacheError("materialized H3 addend is non-finite")
                    if not available: masked += 1
                    if confidence == 0.0: unseen += 1
                    record = {"schema": CACHE_SCHEMA, "utc_day": current_day, "split": split, "policy_episode_id": key[0], "acting_seat": key[1], "env_step": key[2], "stage_count": row.stage_count, "h3_mask": available, "c3": confidence, "scaled_h3_advantage": scaled, "h3_additive_term": addend, "source_member": row.identity[2], "critic_episode_id": row.identity[3], "program_identity": row.identity[6]}
                    raw = canonical_bytes(record)
                    assert handle is not None
                    handle.write(raw); sha.update(raw); count += 1; consumed.add(key)
                matched_rows.clear(); matched_keys.clear()

            for row in iter_complete_action_examples(dataset, targets, split=split):
                day = row.utc_day
                if args.only_day and day != args.only_day:
                    if current_day == args.only_day:
                        break
                    continue
                if day not in policy_days:
                    continue
                if day != current_day:
                    emit_batch(); finish_day()
                    policy_path, allow_unmatched_policy = policy_days[day]
                    current_day = day; index = _policy_index(policy_path); consumed = set()
                    count = masked = unseen = missing_policy_from_sealed_target = mismatched_policy_replaced_by_sealed_target = 0; sha = hashlib.sha256()
                    temp = output / f".{day}.partial-{os.getpid()}"
                    handle = temp.open("xb")
                key = (Path(str(row.identity[2])).stem, int(row.identity[4]), int(row.identity[5]))
                alignment = index.get(key)
                if alignment is None:
                    if allow_unmatched_policy:
                        missing_policy_from_sealed_target += 1
                    else:
                        # Historical canonical expert shards intentionally own
                        # the policy-intersection subset; target-only rows are
                        # not actor rows for those days.
                        continue
                elif alignment != (row.selected_option_indices, row.selected_legal_counts):
                    if not allow_unmatched_policy:
                        raise H3CacheError(f"selected option/legal alignment mismatch for {day} {key}")
                    mismatched_policy_replaced_by_sealed_target += 1
                matched_rows.append(row); matched_keys.append(key)
                if len(matched_rows) >= args.batch_size:
                    emit_batch()
            emit_batch(); finish_day()
        if not summaries:
            raise H3CacheError("no policy/critic cache rows materialized")
        receipt = {"schema": RECEIPT_SCHEMA, "cache_value_semantics": "h3_additive_term_only", "exact_legacy_baseline_computed_in_batch": True, "coefficient": COEFFICIENT, "h1_h6_h12_actor_coefficients": [0.0, 0.0, 0.0], "checkpoint_sha256": args.checkpoint_sha256, "h3_scale_support_sha256": args.h3_scale_support_sha256, "c3_sha256": args.c3_sha256, "policy_transfer_receipt_sha256": sha256_file(args.policy_transfer_receipt), "source_binding": source_binding, "split_days": split_days, "day_shards": summaries, "runtime_critic_calls": False, "actor_activation": False, "activation_eligible": False}
        receipt["artifact_sha256"] = _canonical_sha(receipt)
        receipt_sha = _atomic_json(output / "cache-receipt.json", receipt)
        return {"output_dir": str(output), "receipt_sha256": receipt_sha, "artifact_sha256": receipt["artifact_sha256"], "days": len(summaries), "rows": sum(item["rows"] for item in summaries)}
    except BaseException:
        # Private partials remain for diagnosis; no receipt means ineligible.
        raise


def main(argv: Sequence[str] | None = None) -> int:
    try:
        print(json.dumps(run(parser().parse_args(argv)), sort_keys=True), flush=True)
    except (H3CacheError, OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
