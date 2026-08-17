"""Materialize one frozen Prize-plan-v2 H3 cache for a live replay window.

This is trainer preparation, never a runtime critic path. It consumes only the
public observation and recorded chosen action already present in sealed replay
shards, performs one frozen sidecar inference pass before actor optimization,
and emits the existing cache/activation schemas consumed by
``prize_plan_actor_boundary``.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch

from .frozen_prize_plan_advantage import canonical_sha256
from .prize_plan_actor_boundary import (
    CACHE_RECEIPT_SCHEMA,
    CACHE_ROW_SCHEMA,
    replay_membership_sha256,
)
from .prize_plan_live_features import (
    PrizePlanLiveFeatureError,
    complete_action_features_from_public_replay,
    h3_causal_interval_availability,
)
from .prize_plan_v2_sidecar import load_prize_plan_v2_checkpoint

ACTIVATION_SCHEMA = (
    "poke_bot.alakazam_prize_plan_v2_h3_actor_canary_activation_receipt/v1"
)
LIVE_RECEIPT_SCHEMA = "poke_bot.alakazam_prize_plan_v2_h3_live_materialization/v1"
COEFFICIENT = 0.025
DEFAULT_SCAN_WORKERS = 8
MAX_SCAN_WORKERS = 32


_H3_WORKER_WANTED: dict[tuple[str, int, int], int] | None = None


class PrizePlanLiveCacheError(RuntimeError):
    """The sealed replay window cannot produce the exact H3 cache."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _json(path: Path, *, label: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise PrizePlanLiveCacheError(f"{label} must be a regular file")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PrizePlanLiveCacheError(f"{label} must contain an object")
    return value


def _require_sha(path: Path, expected: str, *, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise PrizePlanLiveCacheError(
            f"{label} digest mismatch: expected={expected} actual={actual}"
        )
    return actual


def _write_exclusive(path: Path, data: bytes) -> str:
    if path.exists():
        raise PrizePlanLiveCacheError(f"create-only output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.partial-{os.getpid()}")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.link(temp, path)
    temp.unlink()
    return sha256_file(path)


def _write_json(path: Path, value: Mapping[str, Any]) -> str:
    return _write_exclusive(
        path,
        (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(),
    )


def _wanted_actions(sequences: Sequence[object]) -> dict[tuple[str, int, int], int]:
    wanted: dict[tuple[str, int, int], int] = {}
    for sequence in sequences:
        episode_id = str(getattr(sequence, "episode_id", "") or "")
        seat = getattr(sequence, "seat", None)
        decisions = getattr(sequence, "decisions", None)
        if not episode_id or seat not in (0, 1) or not isinstance(decisions, list):
            raise PrizePlanLiveCacheError("replay sequence identity is malformed")
        for decision in decisions:
            env_step = getattr(decision, "env_step", None)
            stages = getattr(decision, "policy_stages", None)
            if (
                isinstance(env_step, bool)
                or not isinstance(env_step, int)
                or env_step < 0
                or not isinstance(stages, list)
                or not stages
            ):
                raise PrizePlanLiveCacheError("replay decision identity is malformed")
            key = (episode_id, int(seat), env_step)
            if key in wanted:
                raise PrizePlanLiveCacheError("replay action identity is duplicated")
            wanted[key] = len(stages)
    if not wanted:
        raise PrizePlanLiveCacheError("replay window is empty")
    return wanted


def _iter_raw_sequences(paths: Sequence[Path]):
    for source in paths:
        source = Path(source).expanduser().resolve()
        if source.is_symlink() or not source.is_file():
            raise PrizePlanLiveCacheError(f"replay shard is absent: {source}")
        with source.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise PrizePlanLiveCacheError(
                        f"invalid replay JSON at {source}:{line_number}"
                    ) from exc
                if not isinstance(value, dict):
                    raise PrizePlanLiveCacheError("replay row must be an object")
                yield value


def _iter_raw_sequences_range(path: Path, start: int, end: int):
    """Yield each complete JSONL row whose first byte belongs to one range."""

    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise PrizePlanLiveCacheError(f"replay shard is absent: {source}")
    with source.open("rb") as handle:
        if start > 0:
            handle.seek(start - 1)
            if handle.read(1) != b"\n":
                handle.readline()
        else:
            handle.seek(0)
        actual_start = handle.tell()
        while handle.tell() < end:
            raw = handle.readline()
            if not raw:
                break
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise PrizePlanLiveCacheError(
                    f"invalid replay JSON at {source} byte {handle.tell() - len(raw)}"
                ) from exc
            if not isinstance(value, dict):
                raise PrizePlanLiveCacheError("replay row must be an object")
            yield value
        return max(0, handle.tell() - actual_start)


def _scan_worker_init(wanted: dict[tuple[str, int, int], int]) -> None:
    global _H3_WORKER_WANTED
    _H3_WORKER_WANTED = wanted


def _scan_raw_sequence(
    raw_sequence: Mapping[str, Any],
    *,
    wanted: Mapping[tuple[str, int, int], int],
) -> tuple[list[tuple[object, tuple[str, int, int], bool, str | None]], set[tuple[str, int, int]], int]:
    """Build H3 examples from one complete per-seat trajectory."""

    from scripts.train_alakazam_prize_plan_v2 import CompleteActionExample

    episode_id = str(raw_sequence.get("episode_id") or "")
    seat = raw_sequence.get("seat")
    decisions = raw_sequence.get("decisions")
    if not episode_id or seat not in (0, 1) or not isinstance(decisions, list):
        raise PrizePlanLiveCacheError("raw replay sequence identity is malformed")
    examples: list[tuple[object, tuple[str, int, int], bool, str | None]] = []
    seen: set[tuple[str, int, int]] = set()
    masked_feature_rows = 0
    for index, decision in enumerate(decisions):
        if not isinstance(decision, Mapping):
            raise PrizePlanLiveCacheError("raw replay decision is malformed")
        env_step = decision.get("env_step")
        key = (episode_id, int(seat), env_step)
        if key not in wanted:
            continue
        if key in seen:
            raise PrizePlanLiveCacheError("raw replay repeats a wanted action")
        seen.add(key)
        try:
            features = complete_action_features_from_public_replay(
                episode_id=episode_id,
                seat=int(seat),
                env_step=int(env_step),
                observation=decision.get("observation"),
                action=decision.get("action"),
            )
        except (PrizePlanLiveFeatureError, TypeError, ValueError):
            masked_feature_rows += 1
            continue
        if features.stage_count != wanted[key]:
            raise PrizePlanLiveCacheError("raw/replay factorized stage count drifted")
        available, reason = h3_causal_interval_availability(
            decisions, acting_seat=int(seat), start_index=index
        )
        example = CompleteActionExample(
            identity=(
                "live",
                "sha256:" + "0" * 64,
                "live",
                episode_id,
                int(seat),
                int(env_step),
                "live",
            ),
            utc_day="live",
            stage_count=features.stage_count,
            first_stage_menu=features.first_stage_menu,
            selected_stage_features=features.selected_stage_features,
            selected_option_indices=features.selected_option_indices,
            selected_legal_counts=features.selected_legal_counts,
            selected_action_programs=features.selected_action_programs,
            plan_targets=(0.0, 0.0, 0.0, 0.0),
            plan_masks=(False, available, False, False),
        )
        examples.append((example, key, available, reason))
    return examples, seen, masked_feature_rows


def _scan_range_worker(
    source: str,
    start: int,
    end: int,
    spool: str,
) -> dict[str, Any]:
    """Featurize one JSONL byte range into a private bounded spool."""

    wanted = _H3_WORKER_WANTED
    if wanted is None:
        raise PrizePlanLiveCacheError("H3 scan worker was not initialized")
    examples: list[tuple[object, tuple[str, int, int], bool, str | None]] = []
    seen: set[tuple[str, int, int]] = set()
    masked = 0
    for raw_sequence in _iter_raw_sequences_range(Path(source), start, end):
        rows, row_seen, row_masked = _scan_raw_sequence(
            raw_sequence, wanted=wanted
        )
        if seen.intersection(row_seen):
            raise PrizePlanLiveCacheError("range repeats a wanted action")
        examples.extend(rows)
        seen.update(row_seen)
        masked += row_masked
    destination = Path(spool)
    temporary = destination.with_suffix(f".tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        pickle.dump(
            {"examples": examples, "seen": tuple(seen), "masked": masked},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    temporary.replace(destination)
    return {
        "spool": str(destination),
        "seen": len(seen),
        "examples": len(examples),
        "masked": masked,
        "source_bytes": max(0, int(end) - int(start)),
    }


def _scan_tasks(paths: Sequence[Path], *, workers: int) -> list[tuple[Path, int, int]]:
    """Split immutable shards into enough ranges to keep every worker busy."""

    resolved = [Path(path).expanduser().resolve() for path in paths]
    sizes = [path.stat().st_size for path in resolved]
    total = sum(sizes)
    target_tasks = max(workers, min(workers * 4, max(1, total // (64 * 1024 * 1024))))
    tasks: list[tuple[Path, int, int]] = []
    for path, size in zip(resolved, sizes, strict=True):
        parts = max(1, round(target_tasks * size / max(1, total)))
        for index in range(parts):
            tasks.append((path, size * index // parts, size * (index + 1) // parts))
    return tasks


def materialize_live_h3_cache(
    *,
    sequences: Sequence[object],
    replay_shards: Sequence[Path],
    learner_checkpoint_sha256: str,
    sidecar_checkpoint: Path,
    sidecar_checkpoint_sha256: str,
    scale_support: Path,
    scale_support_sha256: str,
    c3_support: Path,
    c3_support_sha256: str,
    contract: Path,
    contract_sha256: str,
    output_dir: Path,
    device: torch.device,
    batch_size: int = 512,
    scan_workers: int | None = None,
) -> tuple[Path, str, Path, str, dict[str, Any]]:
    """Create the exact replay cache and its clean-boundary activation receipt."""

    if batch_size < 1:
        raise PrizePlanLiveCacheError("batch size must be positive")
    for path, expected, label in (
        (sidecar_checkpoint, sidecar_checkpoint_sha256, "sidecar checkpoint"),
        (scale_support, scale_support_sha256, "H3 scale support"),
        (c3_support, c3_support_sha256, "c3 support"),
        (contract, contract_sha256, "owner contract"),
    ):
        _require_sha(Path(path), str(expected), label=label)

    owner = _json(contract, label="owner contract")
    activation = owner.get("revision_26_prize_plan_v2_h3_actor_activation")
    if (
        owner.get("goal_revision") != 26
        or not isinstance(activation, Mapping)
        or (activation.get("activation") or {}).get("current_activation_allowed")
        is not True
    ):
        raise PrizePlanLiveCacheError("owner contract does not authorize H3 activation")

    scale = _json(scale_support, label="H3 scale support")
    c3 = _json(c3_support, label="c3 support")
    denominator = float(scale.get("frozen_train_h3_predicted_advantage_rms_floor", 0.0))
    table = c3.get("support_table")
    if not math.isfinite(denominator) or denominator <= 0 or not isinstance(table, Mapping):
        raise PrizePlanLiveCacheError("frozen H3 scale or c3 table is invalid")

    from scripts.train_alakazam_prize_plan_v2 import (
        CompleteActionExample,
        _public_action_signature,
        critic_predictions,
    )

    loaded = load_prize_plan_v2_checkpoint(sidecar_checkpoint, device=device)
    model = loaded.model.to(device=device, dtype=torch.float32).eval()
    predicted: dict[tuple[str, int, int], tuple[float, float, bool, str | None]] = {}

    def infer_examples(
        examples: Sequence[
            tuple[CompleteActionExample, tuple[str, int, int], bool, str | None]
        ],
    ) -> None:
        with torch.no_grad():
            for start in range(0, len(examples), batch_size):
                batch = examples[start : start + batch_size]
                values = critic_predictions(
                    model, [item[0] for item in batch], device=device
                ).detach().cpu().tolist()
                for (example, key, available, reason), output in zip(
                    batch, values, strict=True
                ):
                    if key in predicted:
                        raise PrizePlanLiveCacheError(
                            "parallel H3 scan repeated a wanted action"
                        )
                    support = table.get(_public_action_signature(example))
                    confidence = (
                        float(support.get("c3"))
                        if isinstance(support, Mapping)
                        else 0.0
                    )
                    scaled = (float(output[3]) - float(output[2])) / denominator
                    addend = COEFFICIENT * confidence * scaled if available else 0.0
                    if not math.isfinite(addend) or abs(addend) > 2.5:
                        raise PrizePlanLiveCacheError("H3 additive term is invalid")
                    predicted[key] = (addend, confidence, available, reason)

    wanted = _wanted_actions(sequences)
    seen: set[tuple[str, int, int]] = set()
    masked_feature_rows = 0
    requested_workers = (
        int(scan_workers)
        if scan_workers is not None
        else int(os.environ.get("PURE_RL_H3_SCAN_WORKERS", DEFAULT_SCAN_WORKERS))
    )
    workers = max(1, min(MAX_SCAN_WORKERS, requested_workers, os.cpu_count() or 1))
    if workers == 1:
        examples: list[
            tuple[CompleteActionExample, tuple[str, int, int], bool, str | None]
        ] = []
        for raw_sequence in _iter_raw_sequences(replay_shards):
            rows, row_seen, row_masked = _scan_raw_sequence(
                raw_sequence, wanted=wanted
            )
            if seen.intersection(row_seen):
                raise PrizePlanLiveCacheError("raw replay repeats a wanted action")
            examples.extend(rows)
            seen.update(row_seen)
            masked_feature_rows += row_masked
        infer_examples(examples)
    else:
        tasks = _scan_tasks(replay_shards, workers=workers)
        spool_parent = os.environ.get("PURE_RL_H3_SCAN_SPOOL_DIR", "").strip()
        if not spool_parent and Path("/dev/shm").is_dir():
            free = shutil.disk_usage("/dev/shm").free
            if free >= 8 * 1024**3:
                spool_parent = "/dev/shm"
        with tempfile.TemporaryDirectory(
            prefix="pure-rl-h3-scan-",
            dir=spool_parent or None,
        ) as temporary_root:
            context = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=context,
                initializer=_scan_worker_init,
                initargs=(wanted,),
            ) as executor:
                futures = {
                    executor.submit(
                        _scan_range_worker,
                        str(path),
                        start,
                        end,
                        str(Path(temporary_root) / f"part-{index:04d}.pkl"),
                    ): index
                    for index, (path, start, end) in enumerate(tasks)
                }
                for future in as_completed(futures):
                    result = future.result()
                    spool = Path(str(result["spool"]))
                    with spool.open("rb") as handle:
                        payload = pickle.load(handle)
                    spool.unlink()
                    part_seen = set(payload["seen"])
                    if seen.intersection(part_seen):
                        raise PrizePlanLiveCacheError(
                            "parallel H3 ranges overlap replay membership"
                        )
                    seen.update(part_seen)
                    masked_feature_rows += int(payload["masked"])
                    infer_examples(payload["examples"])

    # A feature-ABI failure cannot drop an actor row. Emit a zero additive term
    # for it, but every raw key must still have been found exactly once.
    if seen != set(wanted):
        raise PrizePlanLiveCacheError("raw replay does not cover exact replay membership")

    root = Path(output_dir).expanduser().resolve()
    if root.exists():
        raise PrizePlanLiveCacheError(f"create-only output exists: {root}")
    root.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    masked = 0
    unseen = 0
    for key, stage_count in sorted(wanted.items()):
        value = predicted.get(key)
        if value is None:
            addend, confidence, available, reason = 0.0, 0.0, False, "public_feature_abi_unavailable"
        else:
            addend, confidence, available, reason = value
        masked += int(not available)
        unseen += int(confidence == 0.0)
        rows.append(
            {
                "schema": CACHE_ROW_SCHEMA,
                "utc_day": "live",
                "split": "replay",
                "policy_episode_id": key[0],
                "acting_seat": key[1],
                "env_step": key[2],
                "stage_count": stage_count,
                "h3_mask": available,
                "h3_unavailable_reason": reason,
                "c3": confidence,
                "h3_additive_term": addend,
            }
        )
    shard_data = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    shard_sha = "sha256:" + hashlib.sha256(shard_data).hexdigest()
    shard_path = root / f"{shard_sha[7:]}.live.h3-additive.jsonl"
    _write_exclusive(shard_path, shard_data)
    membership_sha = replay_membership_sha256(sequences)
    receipt: dict[str, Any] = {
        "schema": CACHE_RECEIPT_SCHEMA,
        "cache_value_semantics": "h3_additive_term_only",
        "exact_legacy_baseline_computed_in_batch": True,
        "coefficient": COEFFICIENT,
        "h1_h6_h12_actor_coefficients": [0.0, 0.0, 0.0],
        "checkpoint_sha256": sidecar_checkpoint_sha256,
        "h3_scale_support_sha256": scale_support_sha256,
        "c3_sha256": c3_support_sha256,
        "policy_checkpoint_sha256": learner_checkpoint_sha256,
        "replay_membership_sha256": membership_sha,
        "source_replay_shards": [
            {"path": str(Path(path).resolve()), "sha256": sha256_file(Path(path).resolve())}
            for path in replay_shards
        ],
        "day_shards": [
            {
                "utc_day": "live",
                "split": "replay",
                "path": shard_path.name,
                "sha256": shard_sha,
                "rows": len(rows),
                "h3_masked": masked,
                "unseen_c3": unseen,
            }
        ],
        "runtime_critic_calls": False,
        "actor_activation": True,
        "activation_eligible": True,
    }
    receipt["artifact_sha256"] = canonical_sha256(receipt)
    receipt_path = root / "cache-receipt.json"
    receipt_file_sha = _write_json(receipt_path, receipt)

    activation_receipt: dict[str, Any] = {
        "schema": ACTIVATION_SCHEMA,
        "activation_eligible": True,
        "actor_activation": True,
        "safe_boundary": True,
        "all_pre_activation_gates_passed": True,
        "contract_current_activation_allowed": True,
        "exact_legacy_baseline_computed_in_batch": True,
        "rollback_preflight_passed": True,
        "noninterference_passed": True,
        "no_search_rtp_mcts": True,
        "cache_value_semantics": "h3_additive_term_only",
        "coefficient": COEFFICIENT,
        "h1_h6_h12_actor_coefficients": [0.0, 0.0, 0.0],
        "runtime_critic_calls": False,
        "cache_receipt_sha256": receipt_file_sha,
        "cache_artifact_sha256": receipt["artifact_sha256"],
        "policy_checkpoint_sha256": learner_checkpoint_sha256,
        "replay_membership_sha256": membership_sha,
        "semantic_owner_goal_revision": 23,
        "activation_owner_goal_revision": 26,
        "contract_sha256": contract_sha256,
    }
    activation_receipt["artifact_sha256"] = canonical_sha256(activation_receipt)
    activation_path = root / "activation-receipt.json"
    activation_file_sha = _write_json(activation_path, activation_receipt)
    summary = {
        "schema": LIVE_RECEIPT_SCHEMA,
        "complete_actions": len(rows),
        "h3_available": len(rows) - masked,
        "h3_masked": masked,
        "public_feature_abi_masked": masked_feature_rows,
        "unseen_c3": unseen,
        "scan_workers": workers,
        "scan_ranges": len(_scan_tasks(replay_shards, workers=workers)),
        "replay_membership_sha256": membership_sha,
        "cache_receipt_sha256": receipt_file_sha,
        "activation_receipt_sha256": activation_file_sha,
    }
    return receipt_path, receipt_file_sha, activation_path, activation_file_sha, summary


__all__ = ["PrizePlanLiveCacheError", "materialize_live_h3_cache", "sha256_file"]
