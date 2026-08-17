"""Isolated guide-on/guide-off training and realized-win evaluation.

This module consumes only revision-44/45 future-run review requests.  It fits
two shadow checkpoints from the same immutable parent, replay stream, split,
batch order, optimizer, and seed.  The only changed training input is the
current-deck guide loss multiplier.  Both candidates are then evaluated on
the same opponent/seat/seed schedule.  No candidate is eligible for replay,
serving, promotion, or a formal gate.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from poke_bot.pure_rl.awr_shadow_study import (
    bind_completed_collection_receipts,
    exact_training_order_contract,
    file_digest,
    immutable_json,
    scan_replay_shards,
)
from poke_bot.pure_rl.guide_weight_evidence import (
    EVIDENCE_SCHEMA,
    compile_schedule,
)
from poke_bot.pure_rl.guide_weight_review import REQUEST_SCHEMA
from poke_bot.pure_rl.eval_public import (
    OFFICIAL_BASELINE_IDS as CANONICAL_OFFICIAL_BASELINE_IDS,
)


MANIFEST_SCHEMA = "poke_bot.future_guide_weight_shadow_pair/v1"
OFFICIAL_BASELINE_IDS = tuple(CANONICAL_OFFICIAL_BASELINE_IDS)


def _read(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def _identity(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return {
        "path": str(source),
        "sha256": file_digest(source),
        "bytes": source.stat().st_size,
    }


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def prepare_manifest(
    *,
    request_path: str | Path,
    output_dir: str | Path,
    shadow_device: str,
    production_device: str,
    baseline_manifest: str | Path,
    epochs: int = 1,
    games_per_batch: int = 240,
    max_decisions_per_batch: int = 8192,
) -> Path:
    request_source = Path(request_path).expanduser().resolve()
    request = _read(request_source)
    specialist = str(request.get("specialist_id") or "")
    if (
        request.get("schema") != REQUEST_SCHEMA
        or request.get("status") != "ready_for_isolated_shadow_pair"
        or request.get("scope") != "future_specialist_training_runs_only"
        or int(request.get("prospective_policy_revision") or 0) != 44
        or int(request.get("learning_semantics_revision") or 0) != 46
        or request.get("retroactive_application_allowed") is not False
        or specialist == "teal-mask-ogerpon-ex"
        or not specialist
    ):
        raise ValueError("review request is not an authorized future-run pair")
    if shadow_device.strip() == production_device.strip():
        raise ValueError("shadow and production devices must be distinct")
    if epochs <= 0 or games_per_batch <= 0 or max_decisions_per_batch <= 0:
        raise ValueError("shadow training controls must be positive")

    window = dict(request.get("review_window") or {})
    receipt_rows = list(window.get("collection_receipts") or ())
    receipt_paths = [Path(str(row["path"])).resolve() for row in receipt_rows]
    receipts = [_read(path) for path in receipt_paths]
    replay_paths = [
        Path(str(dict(receipt.get("shard") or {}).get("path") or "")).resolve()
        for receipt in receipts
    ]
    replay = bind_completed_collection_receipts(
        scan_replay_shards(replay_paths),
        receipt_paths,
    )
    parent = dict(window.get("seed_checkpoint") or {})
    if _identity(parent.get("path", ""))["sha256"] != parent.get("sha256"):
        raise RuntimeError("shadow-pair parent checkpoint identity changed")

    seed = 45_000_000 + int(request["completed_iteration"]) * 10_000
    controls = {
        "optimizer": "AdamW",
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "epochs": int(epochs),
        "games_per_batch": int(games_per_batch),
        "max_decisions_per_batch": int(max_decisions_per_batch),
        "max_context": 320,
        "val_frac": 0.10,
        "split_by_episode": False,
        "split_seed": seed,
        "batch_order_seed": seed,
        "awr_beta": 1.0,
        "awr_weight_max": 20.0,
        "awr_normalize_advantages": True,
        "awr_freeze_baseline": True,
        "entropy_bonus": 0.01,
        "loss_weights": {
            "value": 1.0,
            "archetype_aux": 0.05,
            "opponent_hand": 0.05,
            "opponent_remainder": 0.05,
            "lethal_threat": 0.025,
            "prize_race": 0.025,
        },
    }
    variants = [
        {"id": "guide_on", "guide_loss_weight": float(request["current_weight"])},
        {"id": "guide_off", "guide_loss_weight": 0.0},
    ]
    schedule = []
    for index in range(1000):
        schedule.append(
            {
                "schedule_id": f"pair-{index:04d}",
                "opponent_id": OFFICIAL_BASELINE_IDS[(index // 2) % 4],
                "candidate_seat": index % 2,
                "requested_seed": seed + 1_000_000 + index,
            }
        )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "prepared",
        "request": _identity(request_source),
        "specialist_id": specialist,
        "completed_iteration": int(request["completed_iteration"]),
        "current_weight": float(request["current_weight"]),
        "consecutive_nonpositive_evaluations": int(
            request.get("consecutive_nonpositive_evaluations", 0)
        ),
        "parent_checkpoint": parent,
        "replay": replay,
        "guide_contract": dict(request["guide_contract"]),
        "guide_version": str(request["guide_version"]),
        "controls": controls,
        "controls_sha256": _canonical_digest(controls),
        "variants": variants,
        "evaluation_schedule": schedule,
        "baseline_manifest": _identity(baseline_manifest),
        "resource_isolation": {
            "shadow_device": shadow_device,
            "production_device": production_device,
            "must_be_distinct": True,
        },
        "training_eligible": False,
        "replay_eligible": False,
        "formal_gate": False,
        "serving_allowed": False,
        "promotion_allowed": False,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    target = output / "shadow_pair.manifest.json"
    if target.exists():
        existing = _read(target)
        if (
            existing.get("request") != manifest["request"]
            or existing.get("controls_sha256") != manifest["controls_sha256"]
            or existing.get("evaluation_schedule")
            != manifest["evaluation_schedule"]
        ):
            raise RuntimeError("existing guide shadow manifest identity changed")
        return target
    return immutable_json(target, manifest)


def run_training(manifest_path: str | Path, *, device: str) -> Path:
    source = Path(manifest_path).expanduser().resolve()
    manifest = _read(source)
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("training_eligible") is not False
        or manifest.get("serving_allowed") is not False
        or str(device)
        != str(manifest["resource_isolation"]["shadow_device"])
    ):
        raise ValueError("invalid guide shadow-pair training manifest")
    if str(device) == str(manifest["resource_isolation"]["production_device"]):
        raise ValueError("shadow training requested on production device")

    os.environ["POKEBOT_CURRENT_DECK_GUIDE"] = str(manifest["specialist_id"])
    os.environ["POKEBOT_CURRENT_DECK_GUIDE_TARGETS"] = "1"
    os.environ["POKEBOT_CURRENT_DECK_GUIDE_CONTRACT"] = str(
        manifest["guide_contract"]["path"]
    )
    os.environ["POKEBOT_REPLAY_STATUS_DISABLED"] = "1"
    os.environ["POKEBOT_CACHE_DIR"] = str(source.parent / "cache")

    import torch
    from poke_bot import config as runtime_config
    from poke_bot.dataset import BootstrapDataset
    from poke_bot.pure_rl.dataset_bridge import dataset_from_shard
    from poke_bot.train import TrainConfig, rl_train_step

    runtime_config.HARDWARE.cache_dir = source.parent / "cache"
    sequences = []
    for row in manifest["replay"]["shards"]:
        sequences.extend(
            dataset_from_shard(
                Path(row["path"]),
                verify_info_set=False,
                max_context=int(manifest["controls"]["max_context"]),
            ).sequences
        )
    dataset = BootstrapDataset(sequences=sequences)
    controls = dict(manifest["controls"])
    order = exact_training_order_contract(dataset, controls)
    receipts = []
    for variant in manifest["variants"]:
        variant_id = str(variant["id"])
        output = source.parent / "checkpoints" / f"{variant_id}.pt"
        variant_receipt_path = (
            source.parent / "training_receipts" / f"{variant_id}.json"
        )
        if variant_receipt_path.exists():
            receipt = _read(variant_receipt_path)
            checkpoint = dict(receipt.get("checkpoint") or {})
            if (
                receipt.get("manifest_sha256") != file_digest(source)
                or receipt.get("training_order") != order
                or _identity(checkpoint.get("path", ""))["sha256"]
                != checkpoint.get("sha256")
            ):
                raise RuntimeError(
                    f"existing guide shadow receipt changed: {variant_id}"
                )
            receipts.append(receipt)
            continue
        if output.exists():
            raise RuntimeError(
                f"orphan guide shadow checkpoint exists: {output}"
            )
        cfg = TrainConfig.pure_rl_defaults(
            lr=float(controls["learning_rate"]),
            weight_decay=float(controls["weight_decay"]),
            epochs=int(controls["epochs"]),
            games_per_batch=int(controls["games_per_batch"]),
            max_decisions_per_batch=int(controls["max_decisions_per_batch"]),
            val_frac=float(controls["val_frac"]),
            split_by_episode=bool(controls["split_by_episode"]),
            early_stop_patience=0,
            grad_clip=float(controls["gradient_clip_norm"]),
            seed=int(controls["split_seed"]),
            amp=str(device).startswith("cuda"),
            awr_beta=float(controls["awr_beta"]),
            awr_weight_max=float(controls["awr_weight_max"]),
            awr_normalize_advantages=True,
            awr_freeze_baseline=True,
            entropy_bonus=float(controls["entropy_bonus"]),
            value_loss_weight=1.0,
            aux_loss_weight=0.05,
            opp_hand_loss_weight=0.05,
            opp_remainder_loss_weight=0.05,
            lethal_threat_loss_weight=0.025,
            prize_race_loss_weight=0.025,
            alakazam_guide_loss_weight=float(variant["guide_loss_weight"]),
        )
        result = rl_train_step(
            dataset,
            base_ckpt=manifest["parent_checkpoint"]["path"],
            out_run_name=f"guide-shadow.{variant_id}",
            archetype_id=str(manifest["specialist_id"]),
            epochs=int(controls["epochs"]),
            device=torch.device(device),
            cfg=cfg,
            seed=int(controls["batch_order_seed"]),
            output_path=output,
            parent_digest=manifest["parent_checkpoint"]["sha256"],
            training_provenance={
                "schema": MANIFEST_SCHEMA,
                "shadow_only": True,
                "variant": variant_id,
                "training_order": order,
                "serving_allowed": False,
                "promotion_allowed": False,
            },
            replace_existing=False,
            device_resident=False,
        )
        receipt = {
                "variant": variant_id,
                "guide_loss_weight": float(variant["guide_loss_weight"]),
                "checkpoint": {
                    "path": str(Path(result["candidate_path"]).resolve()),
                    "sha256": str(result["candidate_digest"]),
                },
                "parent_sha256": str(result["parent_digest"]),
                "training_order": order,
                "serving_allowed": False,
                "promotion_allowed": False,
                "manifest_sha256": file_digest(source),
            }
        immutable_json(variant_receipt_path, receipt)
        receipts.append(receipt)
    if receipts[0]["training_order"] != receipts[1]["training_order"]:
        raise RuntimeError("guide shadow variants used different training order")
    target = source.parent / "shadow_pair.training.json"
    payload = {
            "schema": MANIFEST_SCHEMA,
            "status": "training_complete",
            "manifest_sha256": file_digest(source),
            "same_parent_replay_split_batch_order_and_optimizer": True,
            "variants": receipts,
            "serving_allowed": False,
            "promotion_allowed": False,
        }
    if target.exists():
        if _read(target) != payload:
            raise RuntimeError("existing guide shadow training receipt changed")
        return target
    return immutable_json(target, payload)


def finalize(
    manifest_path: str | Path,
    training_receipt_path: str | Path,
    evaluation_rows_path: str | Path,
) -> tuple[Path, Path]:
    source = Path(manifest_path).expanduser().resolve()
    manifest = _read(source)
    training = _read(training_receipt_path)
    rows = json.loads(Path(evaluation_rows_path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError("evaluation rows must be a JSON array")
    checkpoints = {
        row["variant"]: row["checkpoint"] for row in training["variants"]
    }
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "specialist_id": manifest["specialist_id"],
        "completed_iteration": manifest["completed_iteration"],
        "current_weight": manifest["current_weight"],
        "consecutive_nonpositive_evaluations": manifest[
            "consecutive_nonpositive_evaluations"
        ],
        "guide_on_checkpoint": checkpoints["guide_on"],
        "guide_off_checkpoint": checkpoints["guide_off"],
        "training_eligible": False,
        "replay_eligible": False,
        "formal_gate": False,
        "serving_allowed": False,
        "promotion_allowed": False,
        "rows": rows,
    }
    schedule = compile_schedule(evidence)
    evidence_path = source.parent / "paired_evidence.json"
    schedule_path = source.parent / "guide_weight_schedule.json"
    if evidence_path.exists() or schedule_path.exists():
        existing_schedule = (
            _read(schedule_path) if schedule_path.exists() else {}
        )
        expected_schedule = dict(schedule)
        existing_schedule.pop("compiled_at_utc", None)
        expected_schedule.pop("compiled_at_utc", None)
        if (
            not evidence_path.exists()
            or not schedule_path.exists()
            or _read(evidence_path) != evidence
            or existing_schedule != expected_schedule
        ):
            raise RuntimeError("existing guide shadow final artifacts changed")
        return evidence_path, schedule_path
    immutable_json(evidence_path, evidence)
    immutable_json(schedule_path, schedule)
    return evidence_path, schedule_path


def run_evaluation(
    manifest_path: str | Path,
    training_receipt_path: str | Path,
    *,
    workers: int,
    game_timeout_seconds: int = 600,
) -> Path:
    """Run the exact paired schedule locally without checkpoint publication."""

    if workers <= 0 or game_timeout_seconds <= 0:
        raise ValueError("evaluation worker controls must be positive")
    source = Path(manifest_path).expanduser().resolve()
    manifest = _read(source)
    training = _read(training_receipt_path)
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or training.get("schema") != MANIFEST_SCHEMA
        or training.get("status") != "training_complete"
        or training.get(
            "same_parent_replay_split_batch_order_and_optimizer"
        )
        is not True
    ):
        raise ValueError("guide shadow-pair training is not complete")
    if _identity(manifest["baseline_manifest"]["path"]) != dict(
        manifest["baseline_manifest"]
    ):
        raise RuntimeError("official baseline manifest identity changed")

    from poke_bot.baselines_runtime import (
        ensure_baselines_installed,
        filter_loadable_baselines,
        load_manifest,
    )
    from poke_bot.worker_pool import WorkerPool
    from scripts.train_pure_rl import _build_collect_jobs, _our_decks
    from scripts.train_round_robin import _worker_play

    ensure_baselines_installed()
    loadable, _failed = filter_loadable_baselines(load_manifest())
    by_id = {spec.id: spec for spec in loadable}
    missing = [value for value in OFFICIAL_BASELINE_IDS if value not in by_id]
    if missing:
        raise RuntimeError(f"official guide-weight opponents missing: {missing}")
    specs = [by_id[value] for value in OFFICIAL_BASELINE_IDS]
    decks = _our_decks("specialist", str(manifest["specialist_id"]))
    if [name for name, _deck in decks] != [manifest["specialist_id"]]:
        raise RuntimeError("guide-weight evaluation deck identity changed")
    schedule = list(manifest["evaluation_schedule"])
    base_seed = int(schedule[0]["requested_seed"])
    all_rows: list[dict[str, Any]] = []
    for variant in training["variants"]:
        checkpoint = dict(variant["checkpoint"])
        _self, jobs = _build_collect_jobs(
            n_games=len(schedule),
            ckpt=Path(checkpoint["path"]),
            digest=str(checkpoint["sha256"]),
            model_generation=0,
            decks=decks,
            specs=specs,
            seed=base_seed,
            game_timeout_s=game_timeout_seconds,
            mode="specialist",
            self_play_frac=0.0,
            balanced_eval=True,
        )
        for job in jobs:
            job["training_eligible"] = False
            job["sample_actions"] = False
            job["greedy"] = True
            job["action_temperature"] = 1.0
            job["collect_privileged_belief"] = False
        with WorkerPool(num_workers=min(int(workers), len(jobs))) as pool:
            results = list(pool.imap_unordered(_worker_play, jobs))
        by_index = {int(row["job_index"]): row for row in results}
        if len(by_index) != len(schedule):
            raise RuntimeError("guide-weight evaluation result count changed")
        for index, expected in enumerate(schedule):
            row = by_index[index]
            if (
                row.get("error")
                or row.get("baseline_failed")
                or row.get("our_failed")
                or row.get("resource_error")
                or int(row.get("our_seat", -1))
                != int(expected["candidate_seat"])
                or int(row.get("seed", -1))
                != int(expected["requested_seed"])
                or str(row.get("opponent_id") or "")
                != str(expected["opponent_id"])
            ):
                raise RuntimeError(
                    f"invalid guide-weight evaluation row: {index}"
                )
            winner = int(row["winner"])
            seat = int(row["our_seat"])
            score = 0.5 if winner == 2 else float(winner == seat)
            all_rows.append(
                {
                    "variant": str(variant["variant"]),
                    "schedule_id": str(expected["schedule_id"]),
                    "opponent_id": str(expected["opponent_id"]),
                    "candidate_seat": seat,
                    "requested_seed": int(expected["requested_seed"]),
                    "checkpoint_sha256": str(checkpoint["sha256"]),
                    "score": score,
                    "training_eligible": False,
                    "replay_eligible": False,
                    "formal_gate": False,
                    "invalid": False,
                    "error": None,
                }
            )
    output = source.parent / "paired_evaluation_rows.json"
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if not isinstance(existing, list) or len(existing) != 2000:
            raise RuntimeError("existing paired evaluation rows changed")
        return output
    return immutable_json(output, all_rows)  # type: ignore[arg-type]


__all__ = [
    "MANIFEST_SCHEMA",
    "OFFICIAL_BASELINE_IDS",
    "finalize",
    "prepare_manifest",
    "run_evaluation",
    "run_training",
]
