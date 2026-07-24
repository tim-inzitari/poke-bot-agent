"""Fail-closed AWR shadow-study contracts.

This module deliberately does *not* know how to publish a checkpoint, reload a
leaf, edit a systemd unit, or advance a production ledger.  It only freezes
read-only inputs, runs candidates into an isolated output directory, and emits
immutable receipts.  Promotion and automatic application are structural
false values in every accepted manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from poke_bot.pure_rl.eval_public import OFFICIAL_BASELINE_IDS


SCHEMA = "pokebot.awr-shadow-study.v1"
BETA_VALUES = (0.5, 0.75, 1.0)
FIXED_LR = 3e-4
FIXED_WEIGHT_DECAY = 1e-4
FIXED_GRAD_CLIP = 1.0
FIXED_DECISION_CAP = 8192
FIXED_MAX_CONTEXT = 320
DEFAULT_AWR_WEIGHT_MAX = 20.0
EVALUATION_ALPHA = 0.05


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def immutable_json(path: str | Path, value: dict[str, Any]) -> Path:
    """Create one JSON artifact without a replace path, then make it read-only."""
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp"
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link creation has no overwrite semantics.  This is safer than a
        # preflight exists() followed by replace(), which has a race window.
        os.link(temporary, target)
        os.chmod(target, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise TypeError(f"JSON artifact is not an object: {path}")
    return value


def _file_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    before = resolved.stat()
    digest = file_digest(resolved)
    after = resolved.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"input changed while hashing: {resolved}")
    return {
        "path": str(resolved),
        "sha256": digest,
        "bytes": int(after.st_size),
        "mtime_ns": int(after.st_mtime_ns),
    }


def _record_key(raw: dict[str, Any]) -> str:
    return f"{str(raw.get('episode_id') or '')}\x1f{int(raw.get('seat') or 0)}"


def scan_replay_shards(paths: Sequence[str | Path]) -> dict[str, Any]:
    """Hash and count the exact compact-shard stream in file/line order."""
    if not paths:
        raise ValueError("at least one replay shard is required")
    shards: list[dict[str, Any]] = []
    order = hashlib.sha256()
    unique_records: set[str] = set()
    unique_episodes: set[str] = set()
    duplicate_records = 0
    records = decisions = 0
    for shard_index, supplied in enumerate(paths):
        path = Path(supplied).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        before = path.stat()
        content_digest = hashlib.sha256()
        shard_records = shard_decisions = 0
        with path.open("rb") as handle:
            for line_index, raw_line in enumerate(handle):
                content_digest.update(raw_line)
                if not raw_line.strip():
                    continue
                raw = json.loads(raw_line)
                if not isinstance(raw, dict):
                    raise TypeError(
                        f"compact replay row is not an object: {path}:{line_index + 1}"
                    )
                key = _record_key(raw)
                episode = str(raw.get("episode_id") or "")
                if not episode:
                    raise ValueError(
                        f"compact replay row has no episode_id: {path}:{line_index + 1}"
                    )
                row_decisions = len(list(raw.get("decisions") or ()))
                order.update(
                    canonical_json_bytes(
                        [shard_index, line_index, key, row_decisions]
                    )
                )
                duplicate_records += int(key in unique_records)
                unique_records.add(key)
                unique_episodes.add(episode)
                records += 1
                decisions += row_decisions
                shard_records += 1
                shard_decisions += row_decisions
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RuntimeError(f"input changed while scanning: {path}")
        identity = {
            "path": str(path),
            "sha256": "sha256:" + content_digest.hexdigest(),
            "bytes": int(after.st_size),
            "mtime_ns": int(after.st_mtime_ns),
        }
        identity.update(
            {
                "records": shard_records,
                "decisions": shard_decisions,
                "input_index": shard_index,
            }
        )
        shards.append(identity)
    return {
        "shards": shards,
        "replay_set_sha256": canonical_digest(
            [row["sha256"] for row in shards]
        ),
        "source_order_sha256": "sha256:" + order.hexdigest(),
        "records": records,
        "unique_game_sequences": len(unique_records),
        "unique_episode_ids": len(unique_episodes),
        "duplicate_game_sequences": duplicate_records,
        "decisions": decisions,
    }


def _verify_completed_collection_receipt(
    shard: dict[str, Any], receipt_path: str | Path
) -> dict[str, Any]:
    identity = _file_identity(receipt_path)
    receipt = read_json(identity["path"])
    if receipt.get("schema") != "poke_bot.completed_collection/v1":
        raise ValueError(
            f"replay input lacks a completed-collection receipt: {receipt_path}"
        )
    claimed = dict(receipt.get("shard") or {})
    exact = {
        "path": str(Path(shard["path"]).resolve()),
        "sha256": shard["sha256"],
        "size": int(shard["bytes"]),
        "games": int(shard["records"]),
        "decisions": int(shard["decisions"]),
    }
    for key, expected in exact.items():
        observed = claimed.get(key)
        if key == "path" and observed is not None:
            observed = str(Path(str(observed)).expanduser().resolve())
        if observed != expected:
            raise ValueError(
                f"completed-collection receipt mismatch for {key}: "
                f"expected={expected!r} observed={observed!r}"
            )
    cache = dict(receipt.get("replay_cache") or {})
    cache_signature = dict(cache.get("signature") or {})
    cache_exact = {
        "covered_bytes": int(shard["bytes"]),
        "records": int(shard["records"]),
        "sequences": int(shard["records"]),
        "dropped": 0,
    }
    for key, expected in cache_exact.items():
        if int(cache.get(key, -1)) != expected:
            raise ValueError(
                f"completed replay cache mismatch for {key}: "
                f"expected={expected!r} observed={cache.get(key)!r}"
            )
    if int(cache_signature.get("max_context", -1)) != FIXED_MAX_CONTEXT:
        raise ValueError(
            f"completed replay cache context must be {FIXED_MAX_CONTEXT}"
        )
    if int(receipt.get("requested_games", -1)) != int(shard["records"]):
        raise ValueError("completed-collection requested/retained game count differs")
    return {
        **identity,
        "schema": receipt["schema"],
        "iteration": receipt.get("iteration"),
        "requested_games": int(receipt["requested_games"]),
    }


def bind_completed_collection_receipts(
    replay: dict[str, Any], receipt_paths: Sequence[str | Path]
) -> dict[str, Any]:
    shards = list(replay.get("shards") or ())
    if len(receipt_paths) != len(shards):
        raise ValueError("exactly one completed-collection receipt is required per shard")
    for shard, receipt_path in zip(shards, receipt_paths):
        shard["collection_receipt"] = _verify_completed_collection_receipt(
            shard, receipt_path
        )
    replay["boundary_contract"] = "completed_collection_receipts_v1"
    replay["collection_receipts_sha256"] = canonical_digest(
        [row["collection_receipt"]["sha256"] for row in shards]
    )
    return replay


def verify_frozen_inputs(manifest: dict[str, Any]) -> dict[str, Any]:
    expected_parent = dict(manifest["parent_checkpoint"])
    observed_parent = _file_identity(expected_parent["path"])
    if observed_parent["sha256"] != expected_parent["sha256"]:
        raise RuntimeError("shadow-study parent checkpoint bytes changed")
    expected_data = dict(manifest["replay"])
    try:
        observed_data = scan_replay_shards(
            [row["path"] for row in expected_data["shards"]]
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"shadow-study replay invariant changed: unreadable row: {exc}"
        ) from exc
    invariant_keys = (
        "replay_set_sha256",
        "source_order_sha256",
        "records",
        "unique_game_sequences",
        "unique_episode_ids",
        "duplicate_game_sequences",
        "decisions",
    )
    for key in invariant_keys:
        if observed_data[key] != expected_data[key]:
            raise RuntimeError(
                f"shadow-study replay invariant changed: {key} "
                f"expected={expected_data[key]!r} observed={observed_data[key]!r}"
            )
    if expected_data.get("boundary_contract") != "completed_collection_receipts_v1":
        raise RuntimeError("shadow-study replay input is not boundary committed")
    for expected_shard, observed_shard in zip(
        expected_data["shards"], observed_data["shards"]
    ):
        receipt = expected_shard.get("collection_receipt") or {}
        if not receipt:
            raise RuntimeError("shadow-study replay shard has no boundary receipt")
        if file_digest(receipt["path"]) != receipt["sha256"]:
            raise RuntimeError("shadow-study collection receipt bytes changed")
        try:
            _verify_completed_collection_receipt(observed_shard, receipt["path"])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"shadow-study collection receipt no longer reconciles: {exc}"
            ) from exc
    return {"parent_checkpoint": observed_parent, "replay": observed_data}


def assert_shadow_gpu_idle(device: str) -> None:
    """Fail closed when another process already owns the physical shadow GPU."""
    value = str(device).strip().lower()
    if not value.startswith("cuda:"):
        return
    try:
        requested_index = int(value.split(":", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"shadow device must name an explicit CUDA index: {device}") from exc
    try:
        gpu_rows = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
        app_rows = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("cannot prove the shadow CUDA device is idle") from exc
    gpu_by_index: dict[int, str] = {}
    for row in gpu_rows.splitlines():
        fields = [field.strip() for field in row.split(",")]
        if len(fields) >= 2:
            gpu_by_index[int(fields[0])] = fields[1]
    if requested_index not in gpu_by_index:
        raise RuntimeError(f"shadow CUDA device does not exist: {device}")
    target_uuid = gpu_by_index[requested_index]
    foreign: list[str] = []
    for row in app_rows.splitlines():
        fields = [field.strip() for field in row.split(",")]
        if len(fields) < 2 or fields[0] != target_uuid:
            continue
        try:
            pid = int(fields[1])
        except ValueError:
            pid = -1
        if pid != os.getpid():
            foreign.append(row.strip())
    if foreign:
        raise RuntimeError(
            f"shadow GPU {device} is not isolated; active compute apps: {foreign}"
        )


def _evaluation_schedule(seed: int) -> dict[str, Any]:
    official: list[dict[str, Any]] = []
    schedule_index = 0
    # 250 games/opponent: 125 matched requested seeds in both seats.
    for pair_index in range(125):
        for opponent_id in OFFICIAL_BASELINE_IDS:
            requested_seed = int(seed + pair_index * len(OFFICIAL_BASELINE_IDS))
            for candidate_seat in (0, 1):
                official.append(
                    {
                        "schedule_id": f"official-{schedule_index:04d}",
                        "target_type": "official_baseline",
                        "target_id": str(opponent_id),
                        "requested_seed": requested_seed,
                        "candidate_seat": candidate_seat,
                        "target_sequence_index": pair_index * 2 + candidate_seat,
                    }
                )
                schedule_index += 1
    parent: list[dict[str, Any]] = []
    for pair_index in range(200):
        requested_seed = int(seed + 10_000 + pair_index)
        for candidate_seat in (0, 1):
            parent.append(
                {
                    "schedule_id": f"parent-{pair_index * 2 + candidate_seat:04d}",
                    "target_type": "current_checkpoint",
                    "target_id": "immutable_parent",
                    "requested_seed": requested_seed,
                    "candidate_seat": candidate_seat,
                    "target_sequence_index": pair_index * 2 + candidate_seat,
                }
            )
    family_comparisons = len(BETA_VALUES) * (len(OFFICIAL_BASELINE_IDS) + 1)
    looks_per_comparison = 5
    return {
        "contract": "matched_requested_seed_opponent_seat_v1",
        "official_jobs": official,
        "parent_jobs": parent,
        "official_looks_per_opponent": [50, 100, 150, 200, 250],
        "parent_looks": [80, 160, 240, 320, 400],
        "familywise_alpha": EVALUATION_ALPHA,
        "alpha_spending": "bonferroni_over_variants_targets_and_looks",
        "family_comparisons": family_comparisons,
        "looks_per_comparison": looks_per_comparison,
        "per_interval_alpha": EVALUATION_ALPHA
        / (family_comparisons * looks_per_comparison),
        "interval": "two_sided_hoeffding_for_bounded_draw_aware_scores",
        "engine_seedable": False,
        "pairing_claimed": False,
        "seed_note": (
            "Python/Torch job seeds, opponents, and seats are matched. The "
            "official engine has no native RNG seed API, so confidence bounds "
            "do not claim paired engine randomness."
        ),
    }


def _research_proposals() -> dict[str, Any]:
    return {
        "game_balanced_loss": {
            "enabled": False,
            "status": "proposal_only",
            "opt_in_flag": "--research-game-balanced-loss",
            "auto_apply": False,
        },
        "bce_win_probability_head": {
            "enabled": False,
            "status": "proposal_only",
            "opt_in_flag": "--research-bce-win-head",
            "auto_apply": False,
        },
        "psro_mixture": {
            "enabled": False,
            "status": "proposal_only",
            "opt_in_flag": "--research-psro-mixture",
            "auto_apply": False,
        },
    }


def prepare_beta_manifest(
    *,
    output_dir: str | Path,
    parent_checkpoint: str | Path,
    replay_shards: Sequence[str | Path],
    replay_receipts: Sequence[str | Path],
    baseline_contract: str | Path,
    shadow_device: str,
    production_device: str,
    seed: int,
    epochs: int = 2,
    games_per_batch: int = 240,
    val_frac: float = 0.1,
    max_context: int = FIXED_MAX_CONTEXT,
    archetype_id: str = "alakazam",
    loss_weights: dict[str, float] | None = None,
) -> Path:
    output = Path(output_dir).expanduser().resolve()
    if str(shadow_device).strip() == str(production_device).strip():
        raise ValueError("shadow and production devices must be distinct")
    if int(epochs) <= 0 or int(games_per_batch) <= 0:
        raise ValueError("epochs and games_per_batch must be positive")
    if int(max_context) != FIXED_MAX_CONTEXT:
        raise ValueError(
            f"shadow studies fix temporal context at {FIXED_MAX_CONTEXT}; "
            f"got {max_context}"
        )
    parent = _file_identity(parent_checkpoint)
    replay = bind_completed_collection_receipts(
        scan_replay_shards(replay_shards), replay_receipts
    )
    baseline = _file_identity(baseline_contract)
    # Keep experiment outputs out of every live source run directory.  A
    # directory named shards is the durable run boundary in this pipeline.
    for row in replay["shards"]:
        shard = Path(row["path"])
        source_run = shard.parent.parent if shard.parent.name == "shards" else None
        if source_run is not None and (
            output == source_run or source_run in output.parents
        ):
            raise ValueError(
                f"shadow output must be outside source run directory: {source_run}"
            )
    weights = {
        "value": 1.0,
        "archetype_aux": 0.05,
        "opponent_hand": 0.05,
        "opponent_remainder": 0.05,
        "lethal_threat": 0.025,
        "prize_race": 0.025,
        "alakazam_guide": 0.0,
        **dict(loss_weights or {}),
    }
    controls = {
        "optimizer": "AdamW",
        "learning_rate": FIXED_LR,
        "weight_decay": FIXED_WEIGHT_DECAY,
        "gradient_clip_norm": FIXED_GRAD_CLIP,
        "max_decisions_per_batch": FIXED_DECISION_CAP,
        "max_context": FIXED_MAX_CONTEXT,
        "games_per_batch": int(games_per_batch),
        "epochs": int(epochs),
        "val_frac": float(val_frac),
        "split_by_episode": False,
        "split_seed": int(seed),
        "batch_order_seed": int(seed),
        "awr_weight_max": DEFAULT_AWR_WEIGHT_MAX,
        "awr_normalize_advantages": True,
        "awr_freeze_baseline": True,
        "entropy_bonus": 0.01,
        "early_stop_patience": 0,
        "loss_weights": weights,
    }
    variants = [
        {
            "id": f"beta_{str(beta).replace('.', '_')}",
            "stage": "beta",
            "awr_beta": beta,
            "parent_checkpoint_sha256": parent["sha256"],
            "replay_set_sha256": replay["replay_set_sha256"],
            "split_seed": int(seed),
            "batch_order_seed": int(seed),
            "controls_sha256": canonical_digest(controls),
        }
        for beta in BETA_VALUES
    ]
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": utc_timestamp(),
        "study_kind": "temporal_alakazam_awr_beta_ab",
        "state": "prepared_beta_only",
        "promotion_enabled": False,
        "promotion_disabled": True,
        "auto_apply": False,
        "production_mutation_allowed": False,
        "service_control_allowed": False,
        "parent_checkpoint": parent,
        "replay": replay,
        "official_baseline_contract": baseline,
        "resource_isolation": {
            "shadow_device": str(shadow_device),
            "production_device": str(production_device),
            "must_be_distinct": True,
            "shadow_gpu_must_be_idle_at_launch": True,
            "live_service_changes": "forbidden",
        },
        "archetype_id": str(archetype_id),
        "controls": controls,
        "beta_variants": variants,
        "evaluation": _evaluation_schedule(seed + 1_000_000),
        "next_stage": {
            "status": "blocked_until_final_beta_report",
            "requires": "immutable matched-evaluation beta_report.final.json",
            "axes": {
                "learning_rate": [FIXED_LR, 2e-4],
                "awr_weight_max": [DEFAULT_AWR_WEIGHT_MAX, 10.0],
                "replay_exposure": ["4_shards_x_1_epoch", "2_shards_x_2_epochs"],
            },
            "comparison_rule": (
                "same selected beta, immutable parent, seeds, batching controls; "
                "report unique games/decisions and total decision exposures"
            ),
            "auto_start": False,
        },
        "research_proposals": _research_proposals(),
    }
    validate_beta_manifest(manifest)
    output.mkdir(parents=True, exist_ok=True)
    target = immutable_json(output / "study_manifest.json", manifest)
    immutable_json(
        output / "study_manifest.receipt.json",
        {
            "schema": SCHEMA,
            "artifact": str(target),
            "artifact_sha256": file_digest(target),
            "parent_checkpoint_sha256": parent["sha256"],
            "replay_set_sha256": replay["replay_set_sha256"],
            "promotion_disabled": True,
            "auto_apply": False,
            "created_at": utc_timestamp(),
        },
    )
    return target


def validate_beta_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != SCHEMA:
        raise ValueError("not an AWR shadow-study manifest")
    forbidden = (
        bool(manifest.get("promotion_enabled"))
        or not bool(manifest.get("promotion_disabled"))
        or bool(manifest.get("auto_apply"))
        or bool(manifest.get("production_mutation_allowed"))
        or bool(manifest.get("service_control_allowed"))
    )
    if forbidden:
        raise ValueError("shadow manifest permits promotion or production mutation")
    resources = manifest.get("resource_isolation") or {}
    if resources.get("shadow_device") == resources.get("production_device"):
        raise ValueError("shadow resource isolation is false")
    controls = dict(manifest.get("controls") or {})
    exact = {
        "optimizer": "AdamW",
        "learning_rate": FIXED_LR,
        "weight_decay": FIXED_WEIGHT_DECAY,
        "gradient_clip_norm": FIXED_GRAD_CLIP,
        "max_decisions_per_batch": FIXED_DECISION_CAP,
        "max_context": FIXED_MAX_CONTEXT,
    }
    for key, expected in exact.items():
        if controls.get(key) != expected:
            raise ValueError(
                f"beta study changed fixed control {key}: {controls.get(key)!r}"
            )
    variants = list(manifest.get("beta_variants") or ())
    if [float(row.get("awr_beta")) for row in variants] != list(BETA_VALUES):
        raise ValueError("beta study must run exactly beta=0.5/0.75/1.0")
    invariant = {
        (
            row.get("parent_checkpoint_sha256"),
            row.get("replay_set_sha256"),
            row.get("split_seed"),
            row.get("batch_order_seed"),
            row.get("controls_sha256"),
        )
        for row in variants
    }
    if len(invariant) != 1:
        raise ValueError("beta variants do not share parent/data/split/order controls")
    proposals = manifest.get("research_proposals") or {}
    if any(bool(row.get("enabled")) for row in proposals.values()):
        raise ValueError("research proposals cannot be enabled in beta study")


def _sequence_identity(sequence: Any) -> list[Any]:
    return [
        str(getattr(sequence, "episode_id", "")),
        int(getattr(sequence, "seat", 0)),
        str(getattr(sequence, "archetype", "")),
        len(getattr(sequence, "decisions", ()) or ()),
    ]


def exact_training_order_contract(dataset: Any, controls: dict[str, Any]) -> dict[str, Any]:
    """Preview the exact split and per-epoch game order used by rl_train_step."""
    from poke_bot.train import _iter_game_batches, cap_history_sequences, split_dataset

    usable = [sequence for sequence in dataset.sequences if sequence.decisions]
    usable, truncated = cap_history_sequences(usable, int(controls["max_context"]))
    train, validation = split_dataset(
        type(dataset)(sequences=usable),
        float(controls["val_frac"]),
        int(controls["split_seed"]),
    )
    if not train:
        train, validation = list(usable), []
    split_payload = {
        "train": [_sequence_identity(row) for row in train],
        "validation": [_sequence_identity(row) for row in validation],
    }
    order_hasher = hashlib.sha256()
    batch_counts: list[int] = []
    decision_exposures = 0
    for epoch in range(int(controls["epochs"])):
        batches = _iter_game_batches(
            train,
            int(controls["games_per_batch"]),
            int(controls["max_decisions_per_batch"]),
            shuffle=True,
            seed=int(controls["batch_order_seed"]),
            epoch=epoch,
        )
        batch_counts.append(len(batches))
        for batch_index, batch in enumerate(batches):
            payload = [
                epoch,
                batch_index,
                [_sequence_identity(sequence) for sequence in batch],
            ]
            order_hasher.update(canonical_json_bytes(payload))
            decision_exposures += sum(len(sequence.decisions) for sequence in batch)
    return {
        "split_sha256": canonical_digest(split_payload),
        "batch_order_sha256": "sha256:" + order_hasher.hexdigest(),
        "train_unique_game_sequences": len(train),
        "validation_unique_game_sequences": len(validation),
        "train_unique_episode_ids": len({row.episode_id for row in train}),
        "validation_unique_episode_ids": len(
            {row.episode_id for row in validation}
        ),
        "train_decisions": sum(len(row.decisions) for row in train),
        "validation_decisions": sum(len(row.decisions) for row in validation),
        "optimizer_decision_exposures": decision_exposures,
        "batches_per_epoch": batch_counts,
        "truncated_sequences": int(truncated),
    }


_METRIC_KEYS = (
    "policy_loss",
    "value_loss",
    "aux_loss",
    "opp_hand_loss",
    "opp_remainder_loss",
    "lethal_threat_loss",
    "prize_race_loss",
    "alakazam_guide_loss",
    "total_loss",
    "policy_acc",
    "target_value_mean",
    "value_pred_mean",
    "awr_weight_mean",
    "awr_weight_p50",
    "awr_weight_p95",
    "awr_weight_max_observed",
    "awr_weight_clip_frac",
    "awr_effective_sample_size",
    "awr_effective_sample_fraction",
    "policy_selected_nll",
    "n_decisions",
    "n_games",
)


def compact_training_result(result: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(result.get("metrics") or {})
    validation = dict(result.get("validation_metrics") or {})
    return {
        "candidate": {
            "path": str(result["candidate_path"]),
            "sha256": str(result["candidate_digest"]),
        },
        "parent_sha256": str(result["parent_digest"]),
        "train_metrics": {key: metrics.get(key) for key in _METRIC_KEYS},
        "validation_metrics": {key: validation.get(key) for key in _METRIC_KEYS},
        "awr_weight_quantiles_exact": bool(
            metrics.get("awr_weight_quantiles_exact")
        ),
        "update_norm_l2": float(result["update_norm_l2"]),
        "relative_update_norm_l2": float(result["relative_update_norm_l2"]),
        "policy_prev_agreement": float(result["policy_prev_agreement"]),
        "policy_prev_agreement_rows": int(result["policy_prev_agreement_rows"]),
        "optimizer_state_restored": bool(result["optimizer_state_restored"]),
        "epochs_ran": int(result["epochs_ran"]),
        "best_metric": float(result["best_metric"]),
        "awr_baseline_mode": str(result["awr_baseline_mode"]),
    }


def run_beta_study(manifest_path: str | Path, *, device: str) -> Path:
    """Run only the prepared beta variants; never evaluate, promote, or apply."""
    import gc

    source = Path(manifest_path).expanduser().resolve()
    manifest = read_json(source)
    validate_beta_manifest(manifest)
    if str(device) != str(manifest["resource_isolation"]["shadow_device"]):
        raise ValueError("execution device does not match frozen shadow device")
    assert_shadow_gpu_idle(device)
    verify_frozen_inputs(manifest)
    output_dir = source.parent
    # These must land before importing config/dataset_bridge.  Shadow
    # featurization, cache manifests, and cache pruning therefore stay entirely
    # under the experiment directory instead of mutating a live run's cache.
    os.environ["POKEBOT_CACHE_DIR"] = str(output_dir / "cache")
    os.environ["POKEBOT_REPLAY_STATUS_DISABLED"] = "1"

    import torch

    from poke_bot import config as runtime_config
    from poke_bot.dataset import BootstrapDataset
    from poke_bot.pure_rl.dataset_bridge import dataset_from_shard
    from poke_bot.train import TrainConfig, rl_train_step

    runtime_config.HARDWARE.cache_dir = output_dir / "cache"

    sequences: list[Any] = []
    for row in manifest["replay"]["shards"]:
        shard_dataset = dataset_from_shard(
            Path(row["path"]),
            verify_info_set=False,
            max_context=int(manifest["controls"]["max_context"]),
        )
        sequences.extend(shard_dataset.sequences)
    dataset = BootstrapDataset(sequences=sequences)
    order_contract = exact_training_order_contract(dataset, manifest["controls"])
    execution_contract = {
        "schema": SCHEMA,
        "study_manifest_sha256": file_digest(source),
        "parent_checkpoint_sha256": manifest["parent_checkpoint"]["sha256"],
        "replay_set_sha256": manifest["replay"]["replay_set_sha256"],
        "source_order_sha256": manifest["replay"]["source_order_sha256"],
        "training_order": order_contract,
        "promotion_disabled": True,
        "auto_apply": False,
        "created_at": utc_timestamp(),
    }
    contract_path = output_dir / "beta_execution_contract.json"
    if contract_path.exists():
        existing = read_json(contract_path)
        comparable = dict(existing)
        comparable.pop("created_at", None)
        expected = dict(execution_contract)
        expected.pop("created_at", None)
        if comparable != expected:
            raise RuntimeError("existing beta execution contract does not match")
    else:
        immutable_json(contract_path, execution_contract)

    controls = manifest["controls"]
    weights = controls["loss_weights"]
    receipts: list[dict[str, Any]] = []
    for variant in manifest["beta_variants"]:
        variant_id = str(variant["id"])
        receipt_path = output_dir / "receipts" / f"{variant_id}.json"
        candidate_path = output_dir / "candidates" / f"{variant_id}.pt"
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if (
                receipt.get("study_manifest_sha256") != file_digest(source)
                or receipt.get("training_order") != order_contract
                or file_digest(receipt["result"]["candidate"]["path"])
                != receipt["result"]["candidate"]["sha256"]
            ):
                raise RuntimeError(f"invalid existing variant receipt: {receipt_path}")
            receipts.append(receipt)
            continue
        if candidate_path.exists():
            raise RuntimeError(
                f"orphan shadow candidate exists without receipt: {candidate_path}"
            )
        cfg = TrainConfig.pure_rl_defaults(
            lr=float(controls["learning_rate"]),
            weight_decay=float(controls["weight_decay"]),
            epochs=int(controls["epochs"]),
            games_per_batch=int(controls["games_per_batch"]),
            max_decisions_per_batch=int(controls["max_decisions_per_batch"]),
            val_frac=float(controls["val_frac"]),
            split_by_episode=bool(controls["split_by_episode"]),
            early_stop_patience=int(controls["early_stop_patience"]),
            grad_clip=float(controls["gradient_clip_norm"]),
            seed=int(controls["split_seed"]),
            amp=str(device).startswith("cuda"),
            awr_beta=float(variant["awr_beta"]),
            awr_weight_max=float(controls["awr_weight_max"]),
            awr_normalize_advantages=bool(
                controls["awr_normalize_advantages"]
            ),
            awr_freeze_baseline=bool(controls["awr_freeze_baseline"]),
            entropy_bonus=float(controls["entropy_bonus"]),
            capture_awr_weight_distribution=True,
            value_loss_weight=float(weights["value"]),
            aux_loss_weight=float(weights["archetype_aux"]),
            opp_hand_loss_weight=float(weights["opponent_hand"]),
            opp_remainder_loss_weight=float(weights["opponent_remainder"]),
            lethal_threat_loss_weight=float(weights["lethal_threat"]),
            prize_race_loss_weight=float(weights["prize_race"]),
            alakazam_guide_loss_weight=float(weights["alakazam_guide"]),
        )
        result = rl_train_step(
            dataset,
            base_ckpt=manifest["parent_checkpoint"]["path"],
            out_run_name=f"shadow.{variant_id}",
            archetype_id=str(manifest["archetype_id"]),
            epochs=int(controls["epochs"]),
            device=torch.device(device),
            cfg=cfg,
            seed=int(controls["batch_order_seed"]),
            output_path=candidate_path,
            parent_digest=manifest["parent_checkpoint"]["sha256"],
            training_provenance={
                "schema": SCHEMA,
                "shadow_only": True,
                "study_manifest_sha256": file_digest(source),
                "variant_id": variant_id,
                "training_order": order_contract,
                "promotion_disabled": True,
                "auto_apply": False,
            },
            replace_existing=False,
            device_resident=False,
        )
        verify_frozen_inputs(manifest)
        receipt = {
            "schema": SCHEMA,
            "variant_id": variant_id,
            "awr_beta": float(variant["awr_beta"]),
            "study_manifest_sha256": file_digest(source),
            "execution_contract_sha256": file_digest(contract_path),
            "parent_checkpoint_sha256": manifest["parent_checkpoint"]["sha256"],
            "replay_set_sha256": manifest["replay"]["replay_set_sha256"],
            "training_order": order_contract,
            "unique_game_sequences": manifest["replay"]["unique_game_sequences"],
            "unique_episode_ids": manifest["replay"]["unique_episode_ids"],
            "unique_decisions": manifest["replay"]["decisions"],
            "result": compact_training_result(result),
            "promotion_disabled": True,
            "auto_apply": False,
            "completed_at": utc_timestamp(),
        }
        immutable_json(receipt_path, receipt)
        receipts.append(receipt)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {
        "schema": SCHEMA,
        "report_kind": "beta_training_comparison",
        "status": "training_complete_evaluation_required",
        "study_manifest_sha256": file_digest(source),
        "execution_contract_sha256": file_digest(contract_path),
        "fixed_controls": manifest["controls"],
        "same_parent_replay_split_seed_order": True,
        "variants": receipts,
        "next_stage_unlocked": False,
        "promotion_disabled": True,
        "auto_apply": False,
        "completed_at": utc_timestamp(),
    }
    report_path = output_dir / "beta_report.training.json"
    if not report_path.exists():
        immutable_json(report_path, report)
    return report_path


def _score_from_row(row: dict[str, Any]) -> float:
    if row.get("score") is not None:
        score = float(row["score"])
    else:
        winner = int(row["winner"])
        seat = int(row.get("candidate_seat", row.get("our_seat", 0)))
        score = 0.5 if winner == 2 else float(winner == seat)
    if score not in {0.0, 0.5, 1.0}:
        raise ValueError(f"evaluation score must be 0/0.5/1, got {score}")
    return score


def _hoeffding_interval(scores: Sequence[float], alpha: float) -> tuple[float, float]:
    if not scores:
        return 0.0, 1.0
    center = sum(scores) / len(scores)
    radius = math.sqrt(math.log(2.0 / float(alpha)) / (2.0 * len(scores)))
    return max(0.0, center - radius), min(1.0, center + radius)


def matched_sequential_evaluation(
    manifest: dict[str, Any],
    variant_receipts: Sequence[dict[str, Any]],
    rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Validate a complete matched schedule and calculate alpha-spent CIs."""
    variants = {str(row["variant_id"]): row for row in variant_receipts}
    expected_variants = {str(row["id"]) for row in manifest["beta_variants"]}
    if set(variants) != expected_variants:
        raise ValueError("evaluation receipts do not cover exactly the beta variants")
    schedule_rows = [
        *manifest["evaluation"]["official_jobs"],
        *manifest["evaluation"]["parent_jobs"],
    ]
    schedule = {str(row["schedule_id"]): row for row in schedule_rows}
    observed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        variant_id = str(row.get("variant_id") or "")
        schedule_id = str(row.get("schedule_id") or "")
        key = (variant_id, schedule_id)
        if variant_id not in variants or schedule_id not in schedule:
            raise ValueError(f"unexpected evaluation row: {key}")
        if key in observed:
            raise ValueError(f"duplicate evaluation row: {key}")
        expected = schedule[schedule_id]
        actual_seat = int(row.get("candidate_seat", row.get("our_seat", -1)))
        if (
            str(row.get("target_type")) != str(expected["target_type"])
            or str(row.get("target_id")) != str(expected["target_id"])
            or int(row.get("requested_seed", -1)) != int(expected["requested_seed"])
            or actual_seat != int(expected["candidate_seat"])
        ):
            raise ValueError(f"evaluation schedule mismatch: {key}")
        candidate = variants[variant_id]["result"]["candidate"]
        if str(row.get("candidate_sha256")) != str(candidate["sha256"]):
            raise ValueError(f"candidate digest mismatch: {key}")
        if expected["target_type"] == "current_checkpoint":
            if str(row.get("target_sha256")) != str(
                manifest["parent_checkpoint"]["sha256"]
            ):
                raise ValueError(f"parent target digest mismatch: {key}")
        if bool(row.get("invalid")) or row.get("error"):
            raise ValueError(f"invalid matched evaluation row: {key}")
        observed[key] = {**row, "score": _score_from_row(row)}
    expected_keys = {
        (variant_id, schedule_id)
        for variant_id in expected_variants
        for schedule_id in schedule
    }
    missing = expected_keys - set(observed)
    if missing:
        raise ValueError(f"matched evaluation is incomplete: missing={len(missing)}")

    alpha = float(manifest["evaluation"]["per_interval_alpha"])
    comparisons: dict[str, Any] = {}
    for variant_id in sorted(expected_variants):
        per_target: dict[str, Any] = {}
        target_specs = [
            ("current_checkpoint", "immutable_parent", manifest["evaluation"]["parent_looks"]),
            *[
                (
                    "official_baseline",
                    opponent_id,
                    manifest["evaluation"]["official_looks_per_opponent"],
                )
                for opponent_id in OFFICIAL_BASELINE_IDS
            ],
        ]
        for target_type, target_id, looks in target_specs:
            selected_schedule = sorted(
                (
                    row
                    for row in schedule_rows
                    if row["target_type"] == target_type
                    and row["target_id"] == target_id
                ),
                key=lambda row: int(row["target_sequence_index"]),
            )
            scores = [
                float(observed[(variant_id, str(row["schedule_id"]))]["score"])
                for row in selected_schedule
            ]
            sequential = []
            for look in looks:
                prefix = scores[: int(look)]
                lower, upper = _hoeffding_interval(prefix, alpha)
                sequential.append(
                    {
                        "games": int(look),
                        "win_rate": sum(prefix) / len(prefix),
                        "confidence_lower": lower,
                        "confidence_upper": upper,
                        "alpha": alpha,
                    }
                )
            per_target[target_id] = {
                "target_type": target_type,
                "games": len(scores),
                "win_rate": sum(scores) / len(scores),
                "sequential": sequential,
                "final": sequential[-1],
            }
        official = [per_target[target] for target in OFFICIAL_BASELINE_IDS]
        comparisons[variant_id] = {
            "targets": per_target,
            "minimum_official_win_rate": min(row["win_rate"] for row in official),
            "minimum_official_confidence_lower": min(
                row["final"]["confidence_lower"] for row in official
            ),
            "mean_official_win_rate": sum(row["win_rate"] for row in official)
            / len(official),
            "beats_every_official_at_sequential_ci": all(
                row["final"]["confidence_lower"] > 0.5 for row in official
            ),
        }
    ranking = sorted(
        comparisons,
        key=lambda variant_id: (
            comparisons[variant_id]["minimum_official_confidence_lower"],
            comparisons[variant_id]["minimum_official_win_rate"],
            comparisons[variant_id]["mean_official_win_rate"],
        ),
        reverse=True,
    )
    return {
        "contract": manifest["evaluation"]["contract"],
        "rows": len(observed),
        "matched_complete": True,
        "alpha_spending": manifest["evaluation"]["alpha_spending"],
        "familywise_alpha": manifest["evaluation"]["familywise_alpha"],
        "per_interval_alpha": alpha,
        "engine_seedable": False,
        "pairing_claimed": False,
        "comparisons": comparisons,
        "ranking": ranking,
        "selected_variant": ranking[0],
    }


def finalize_beta_report(
    manifest_path: str | Path,
    *,
    evaluation_rows_path: str | Path,
) -> Path:
    source = Path(manifest_path).expanduser().resolve()
    manifest = read_json(source)
    validate_beta_manifest(manifest)
    training_report = read_json(source.parent / "beta_report.training.json")
    receipts = list(training_report.get("variants") or ())
    raw_rows = json.loads(Path(evaluation_rows_path).read_text())
    if not isinstance(raw_rows, list):
        raise TypeError("evaluation rows must be a JSON array")
    evaluation = matched_sequential_evaluation(manifest, receipts, raw_rows)
    report = {
        "schema": SCHEMA,
        "report_kind": "beta_final_matched_evaluation",
        "status": "complete",
        "study_manifest_sha256": file_digest(source),
        "training_report_sha256": file_digest(
            source.parent / "beta_report.training.json"
        ),
        "evaluation_rows_sha256": file_digest(evaluation_rows_path),
        "same_parent_replay_split_seed_order": True,
        "training_variants": receipts,
        "evaluation": evaluation,
        "selected_variant": evaluation["selected_variant"],
        "next_stage_unlocked": True,
        "promotion_disabled": True,
        "auto_apply": False,
        "completed_at": utc_timestamp(),
    }
    return immutable_json(source.parent / "beta_report.final.json", report)


def _subset_counts(replay: dict[str, Any], indices: Sequence[int]) -> dict[str, Any]:
    selected = [replay["shards"][index] for index in indices]
    return {
        "shard_indices": list(indices),
        "shard_sha256s": [row["sha256"] for row in selected],
        "records": sum(int(row["records"]) for row in selected),
        "decisions": sum(int(row["decisions"]) for row in selected),
        # Exact deduped counts are added by prepare_stage2_manifest, which
        # rescans each subset rather than assuming source shards are disjoint.
    }


def prepare_stage2_manifest(
    *,
    beta_manifest_path: str | Path,
    final_beta_report_path: str | Path,
    output_dir: str | Path,
    replay_shards: Sequence[str | Path],
    replay_receipts: Sequence[str | Path],
    seed: int,
) -> Path:
    beta_path = Path(beta_manifest_path).expanduser().resolve()
    beta = read_json(beta_path)
    validate_beta_manifest(beta)
    final_path = Path(final_beta_report_path).expanduser().resolve()
    final = read_json(final_path)
    if (
        final.get("status") != "complete"
        or not final.get("next_stage_unlocked")
        or final.get("study_manifest_sha256") != file_digest(beta_path)
    ):
        raise ValueError("stage 2 remains blocked until the final beta report")
    if len(replay_shards) != 4:
        raise ValueError("stage 2 requires exactly four immutable replay shards")
    replay = bind_completed_collection_receipts(
        scan_replay_shards(replay_shards), replay_receipts
    )
    all_four = scan_replay_shards(replay_shards)
    latest_two = scan_replay_shards(replay_shards[-2:])
    selected_id = str(final["selected_variant"])
    receipts = {
        str(row["variant_id"]): row for row in final["training_variants"]
    }
    if selected_id not in receipts:
        raise ValueError("selected beta variant has no immutable training receipt")
    selected = receipts[selected_id]
    parent = _file_identity(selected["result"]["candidate"]["path"])
    if parent["sha256"] != selected["result"]["candidate"]["sha256"]:
        raise ValueError("selected beta checkpoint no longer matches its receipt")
    profiles = {
        "4_shards_x_1_epoch": {
            "shards": [row["path"] for row in replay["shards"]],
            "epochs": 1,
            "unique_game_sequences": all_four["unique_game_sequences"],
            "unique_episode_ids": all_four["unique_episode_ids"],
            "unique_decisions": all_four["decisions"],
            "decision_exposures": all_four["decisions"],
            "replay_set_sha256": all_four["replay_set_sha256"],
        },
        "2_shards_x_2_epochs": {
            "shards": [row["path"] for row in replay["shards"][-2:]],
            "epochs": 2,
            "unique_game_sequences": latest_two["unique_game_sequences"],
            "unique_episode_ids": latest_two["unique_episode_ids"],
            "unique_decisions": latest_two["decisions"],
            "decision_exposures": latest_two["decisions"] * 2,
            "replay_set_sha256": latest_two["replay_set_sha256"],
        },
    }
    variants: list[dict[str, Any]] = []
    for lr in (FIXED_LR, 2e-4):
        for weight_max in (DEFAULT_AWR_WEIGHT_MAX, 10.0):
            for profile_id, profile in profiles.items():
                variants.append(
                    {
                        "id": (
                            f"lr_{lr:.0e}_wmax_{weight_max:g}_{profile_id}"
                        ).replace("+", ""),
                        "selected_awr_beta": float(selected["awr_beta"]),
                        "learning_rate": lr,
                        "awr_weight_max": weight_max,
                        "replay_profile": profile_id,
                        "replay_set_sha256": profile["replay_set_sha256"],
                        "unique_game_sequences": profile["unique_game_sequences"],
                        "unique_decisions": profile["unique_decisions"],
                        "decision_exposures": profile["decision_exposures"],
                        "split_seed": int(seed),
                        "batch_order_seed": int(seed),
                        "parent_checkpoint_sha256": parent["sha256"],
                    }
                )
    stage2 = {
        "schema": SCHEMA,
        "study_kind": "awr_stage2_guarded_matrix",
        "status": "prepared_not_started",
        "created_at": utc_timestamp(),
        "source_beta_manifest_sha256": file_digest(beta_path),
        "final_beta_report_sha256": file_digest(final_path),
        "selected_beta_variant": selected_id,
        "parent_checkpoint": parent,
        "replay": replay,
        "replay_profiles": profiles,
        "matrix": variants,
        "fixed_controls": {
            "optimizer": "AdamW",
            "weight_decay": FIXED_WEIGHT_DECAY,
            "gradient_clip_norm": FIXED_GRAD_CLIP,
            "max_decisions_per_batch": FIXED_DECISION_CAP,
            "max_context": FIXED_MAX_CONTEXT,
            "seed": int(seed),
        },
        "promotion_disabled": True,
        "auto_apply": False,
        "auto_start": False,
        "research_proposals": _research_proposals(),
    }
    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    return immutable_json(target_dir / "stage2_manifest.json", stage2)
