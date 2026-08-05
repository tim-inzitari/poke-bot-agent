#!/usr/bin/env python3
"""Run revision-130's real, isolated Marnie family shadow study.

The trainer must already be stopped at its immutable status-75 boundary.  This
program never starts or stops production, never publishes a checkpoint, and
never writes into a production replay directory.  A passing run materializes
only the selected loss vector, candidate registry, and activation request that
the separately managed atomic installer consumes.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.archetype_family_activation import (  # noqa: E402
    PAUSE_SCHEMA,
    REQUEST_SCHEMA,
    sha256,
    validate_activation_request,
    validate_iteration9_upload_trigger,
)
from poke_bot.archetype_family_study import (  # noqa: E402
    SCHEMA,
    activation_gate,
    antithetic_spsa,
    compile_activation_metrics,
    compile_development_round,
    family_shadow_plan,
    selected_loss_vector,
    validate_same_parent_shadow,
)
from poke_bot.archetype_loss_contract import (  # noqa: E402
    canonical_residual_weights,
    validate_loss_contract,
)
from poke_bot.specialist_archetype_family import (  # noqa: E402
    FamilyDeckMix,
    validate_manifest,
)


SPECIALIST_ID = "marnie-s-grimmsnarl-ex"
RUN_SCHEMA = "poke_bot.marnie_archetype_family_shadow_run/v1"
TRAINING_GAMES = 8192
SELF_PLAY_FRACTION = 0.125
SHADOW_GUIDE_LOSS_WEIGHT = 0.0


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _canonical_digest(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _validated_shadow_guide_loss_weight(controls: Mapping[str, Any]) -> float:
    """Mask the guide objective when sealed family rows have no guide labels.

    The family shadow study tunes only the separately typed archetype residual
    objectives.  Its observed-list rows are deliberately training-ineligible
    and do not carry current-deck guide targets.  Production keeps its existing
    0.05 directional-guide contract through the unchanged parent registry; a
    nonzero guide objective inside this isolated study would be both unusable
    and a change outside the SPSA study's authority.
    """

    weight = float(controls.get("shadow_guide_loss_weight", math.nan))
    if not math.isfinite(weight) or weight != SHADOW_GUIDE_LOSS_WEIGHT:
        raise RuntimeError(
            "family shadow guide objective must be exactly masked while "
            "production guide authority remains unchanged"
        )
    return weight


@contextmanager
def _cpu_only_panel_spawn_environment():
    """Hide CUDA before spawn imports run, then restore the parent env.

    Setting ``POKEBOT_WORKER_CPU_ONLY`` only in WorkerPool's initializer is too
    late for spawn children: they import the main module (and CUDA-capable
    dependencies) before the initializer executes.  Masking CUDA in the parent
    while the pool is created makes CPU-only evaluation an OS-process-start
    invariant rather than a best-effort post-import setting.
    """
    previous_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    previous_cpu_only = os.environ.get("POKEBOT_WORKER_CPU_ONLY")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["POKEBOT_WORKER_CPU_ONLY"] = "1"
    try:
        yield
    finally:
        if previous_cuda is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous_cuda
        if previous_cpu_only is None:
            os.environ.pop("POKEBOT_WORKER_CPU_ONLY", None)
        else:
            os.environ["POKEBOT_WORKER_CPU_ONLY"] = previous_cpu_only


def _identity(path: Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return {"path": str(source), "sha256": sha256(source), "bytes": source.stat().st_size}


def _write_immutable(path: Path, value: Any) -> Path:
    target = Path(path).expanduser().resolve()
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if target.exists():
        if target.read_bytes() != encoded:
            raise RuntimeError(f"immutable artifact identity changed: {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return target


def _hash_torch_value(value: Any) -> str:
    """Deterministically seal nested optimizer/scaler/RNG checkpoint state."""
    import torch

    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode())
            digest.update(json.dumps(list(tensor.shape)).encode())
            # NumPy cannot represent every torch dtype (notably bfloat16).
            # Hash the exact contiguous storage bytes without a dtype cast.
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=lambda raw: repr(raw)):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode() + b"\0")
            for row in item:
                visit(row)
        elif isinstance(item, bytes):
            digest.update(b"bytes\0" + item)
        else:
            digest.update(
                json.dumps(item, sort_keys=True, default=repr).encode("utf-8")
            )

    visit(value)
    return "sha256:" + digest.hexdigest()


def _load_gate_specs(active_gate_path: Path) -> tuple[list[Any], list[str]]:
    from poke_bot.baselines_runtime import (
        baseline_content_digest,
        ensure_baselines_installed,
        filter_loadable_baselines,
        load_manifest,
    )
    from poke_bot.pure_rl.strong_public_gate import load_active_gate_contract

    gate = load_active_gate_contract(active_gate_path)["next_gate"]
    opponent_ids = [str(row["opponent_id"]) for row in gate["roster"]]
    if len(opponent_ids) != 17:
        raise RuntimeError("family study active gate is not the exact 17-opponent roster")
    ensure_baselines_installed()
    loadable, _failed = filter_loadable_baselines(load_manifest())
    by_id = {str(spec.id): spec for spec in loadable}
    missing = [opponent_id for opponent_id in opponent_ids if opponent_id not in by_id]
    if missing:
        raise RuntimeError(f"family study opponents are unavailable: {missing}")
    specs = [by_id[opponent_id] for opponent_id in opponent_ids]
    expected = {
        str(row["opponent_id"]): str(row["content_digest"])
        for row in gate["roster"]
    }
    for spec in specs:
        if baseline_content_digest(spec.path) != expected[str(spec.id)]:
            raise RuntimeError(f"family study opponent digest changed: {spec.id}")
    return specs, opponent_ids


def _collect_training_shard(
    *,
    output_dir: Path,
    parent: Mapping[str, Any],
    manifest: Mapping[str, Any],
    specs: list[Any],
    workers: int,
    game_timeout_seconds: int,
    seed: int,
) -> tuple[Path, dict[str, Any]]:
    from poke_bot.pure_rl.dataset_bridge import dataset_from_shard
    from poke_bot.remote_sim_jobs import remote_self_play_job
    from scripts.train_pure_rl import _build_collect_jobs, _collect_wave
    from scripts.train_round_robin import _worker_play

    shard = output_dir / "sealed_training_rows.jsonl"
    receipt_path = output_dir / "sealed_training_rows.receipt.json"
    if shard.is_file() and receipt_path.is_file():
        receipt = _read(receipt_path)
        if receipt.get("shard_sha256") != sha256(shard):
            raise RuntimeError("sealed family shadow rows changed")
        return shard, receipt
    if shard.exists() or receipt_path.exists():
        raise RuntimeError("partial family shadow collection exists")
    mix = FamilyDeckMix.from_manifest(manifest)
    by_id = {str(row["variant_id"]): row for row in manifest["variants"]}
    decks = [
        (entry.deck_id, [int(card) for card in by_id[entry.deck_id]["card_ids"]])
        for entry in mix.decks
    ]
    self_jobs, public_jobs = _build_collect_jobs(
        n_games=TRAINING_GAMES,
        ckpt=Path(str(parent["path"])),
        digest=str(parent["sha256"]),
        model_generation=0,
        decks=decks,
        specs=specs,
        seed=int(seed),
        game_timeout_s=int(game_timeout_seconds),
        mode="specialist",
        collect_temperature=1.0,
        self_play_frac=SELF_PLAY_FRACTION,
        ladder_mix=mix,
        family_manifest=manifest,
        iteration=0,
        exact_training_seat_split=True,
    )
    os.environ["POKEBOT_WORKER_CPU_ONLY"] = "1"
    writer, runtime_rows, stats = _collect_wave(
        self_play_jobs=self_jobs,
        baseline_jobs=public_jobs,
        shard_path=shard,
        n_workers=int(workers),
        leaf_channel=None,
        remote_farm=None,
        worker_play=_worker_play,
        worker_self_play=remote_self_play_job,
        iteration=0,
        stage_label="family-shadow:sealed-training",
        replay_eligible=True,
        required_checkpoint_digest=str(parent["sha256"]),
        required_mirror_archetype=SPECIALIST_ID,
    )
    if int(stats.get("with_record", 0)) != TRAINING_GAMES:
        raise RuntimeError("family shadow collection did not retain every source game")
    dataset = dataset_from_shard(shard, verify_info_set=False, max_context=320)
    receipt = {
        "schema": "poke_bot.marnie_family_sealed_shadow_rows/v1",
        "training_eligible": False,
        "production_replay_eligible": False,
        "shadow_training_only": True,
        "parent": dict(parent),
        "manifest_sha256": str(manifest["artifact_sha256"]),
        "shard": str(shard.resolve()),
        "shard_sha256": sha256(shard),
        "source_games": int(stats["with_record"]),
        "trajectories": int(writer.n_games),
        "decisions": int(writer.n_decisions),
        "dataset_sequences": len(dataset.sequences),
        "runtime_rows": len(runtime_rows),
        "self_play_fraction": SELF_PLAY_FRACTION,
        "seed": int(seed),
    }
    _write_immutable(receipt_path, receipt)
    return shard, receipt


def _train_candidate(
    *,
    output_dir: Path,
    label: str,
    parent: Mapping[str, Any],
    dataset: Any,
    controls: Mapping[str, Any],
    residuals: Mapping[str, float],
    common_provenance: Mapping[str, Any],
    device: str,
) -> dict[str, Any]:
    import torch
    from poke_bot.train import TrainConfig, rl_train_step

    output = output_dir / "checkpoints" / f"{label}.pt"
    receipt_path = output_dir / "training_receipts" / f"{label}.json"
    if receipt_path.is_file():
        receipt = _read(receipt_path)
        checkpoint = receipt.get("checkpoint") or {}
        if sha256(Path(str(checkpoint.get("path", "")))) != checkpoint.get("sha256"):
            raise RuntimeError("existing family shadow checkpoint changed")
        return receipt
    if output.exists():
        raise RuntimeError(f"orphan family shadow checkpoint: {output}")
    shadow_guide_loss_weight = _validated_shadow_guide_loss_weight(controls)
    cfg = TrainConfig.pure_rl_defaults(
        lr=float(controls["learning_rate"]),
        weight_decay=float(controls["weight_decay"]),
        epochs=int(controls["epochs"]),
        games_per_batch=int(controls["games_per_batch"]),
        max_decisions_per_batch=int(controls["max_decisions_per_batch"]),
        val_frac=float(controls["val_frac"]),
        split_by_episode=False,
        early_stop_patience=0,
        grad_clip=float(controls["gradient_clip_norm"]),
        seed=int(controls["seed"]),
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
        alakazam_guide_loss_weight=shadow_guide_loss_weight,
        current_deck_guide_training_mode="strategic_directional_v2",
        setup_board_outcome_loss_weight=0.025,
        combo_state_loss_weight=0.025,
        current_deck_guide_curriculum_spec=str(controls["curriculum_spec"]),
        current_deck_guide_head_role_map=str(controls["head_role_map"]),
        current_deck_guide_curriculum_validation_receipt=str(
            controls["curriculum_validation"]
        ),
        archetype_residual_loss_weights=dict(residuals),
        dormant_matchup_adapter_epochs=0,
    )
    result = rl_train_step(
        dataset,
        base_ckpt=str(parent["path"]),
        out_run_name=f"marnie-family-shadow.{label}",
        archetype_id=SPECIALIST_ID,
        epochs=int(controls["epochs"]),
        device=torch.device(device),
        cfg=cfg,
        seed=int(controls["seed"]),
        output_path=output,
        parent_digest=str(parent["sha256"]),
        training_provenance={
            **dict(common_provenance),
            "shadow_only": True,
            "serving_allowed": False,
            "promotion_allowed": False,
            "production_replay_eligible": False,
            "candidate": label,
            "residual_weights": dict(residuals),
            "shadow_guide_loss_weight": shadow_guide_loss_weight,
            "production_directional_guide_contract_unchanged": True,
        },
        replace_existing=False,
        device_resident=False,
    )
    receipt = {
        "schema": RUN_SCHEMA,
        "status": "training_complete",
        "variant": label,
        "checkpoint": _identity(Path(result["candidate_path"])),
        "parent_checkpoint_sha256": str(parent["sha256"]),
        "loss_vector_sha256": _canonical_digest(dict(residuals)),
        "residual_weights": dict(residuals),
        **dict(common_provenance),
        "served": False,
        "promoted": False,
        "replay_eligible": False,
        "training_result": result,
    }
    _write_immutable(receipt_path, receipt)
    return receipt


def _panel_jobs(
    *,
    panel: Sequence[Mapping[str, Any]],
    treatment: str,
    checkpoint: Mapping[str, Any],
    manifest: Mapping[str, Any],
    specs: Sequence[Any],
    game_timeout_seconds: int,
) -> tuple[list[dict[str, Any]], dict[int, Mapping[str, Any]]]:
    from scripts.train_pure_rl import _build_collect_jobs

    variants = {str(row["variant_id"]): row for row in manifest["variants"]}
    spec_by_id = {str(spec.id): spec for spec in specs}
    templates: dict[tuple[str, str, int], dict[str, Any]] = {}
    jobs: list[dict[str, Any]] = []
    schedule_by_index: dict[int, Mapping[str, Any]] = {}
    for job_index, schedule in enumerate(panel):
        variant_id = str(schedule["variant_id"])
        opponent_id = str(schedule["opponent_id"])
        seat = int(schedule["treatment_seat"])
        key = (variant_id, opponent_id, seat)
        if key not in templates:
            _self, pair = _build_collect_jobs(
                n_games=2,
                ckpt=Path(str(checkpoint["path"])),
                digest=str(checkpoint["sha256"]),
                model_generation=0,
                decks=[
                    (
                        SPECIALIST_ID,
                        [int(card) for card in variants[variant_id]["card_ids"]],
                    )
                ],
                specs=[spec_by_id[opponent_id]],
                seed=0,
                game_timeout_s=int(game_timeout_seconds),
                mode="specialist",
                self_play_frac=0.0,
                balanced_eval=True,
            )
            by_seat = {int(row["our_seat"]): row for row in pair}
            if set(by_seat) != {0, 1}:
                raise RuntimeError("family panel template is not seat-balanced")
            templates[(variant_id, opponent_id, 0)] = by_seat[0]
            templates[(variant_id, opponent_id, 1)] = by_seat[1]
        job = copy.deepcopy(templates[key])
        job.update(
            {
                "job_index": job_index,
                "checkpoint": str(checkpoint["path"]),
                "checkpoint_digest": str(checkpoint["sha256"]),
                "seed": int(schedule["requested_seed"]),
                "pair_id": str(schedule["schedule_id"]),
                "our_seat": seat,
                "training_eligible": False,
                "sample_actions": False,
                "greedy": True,
                "action_temperature": 1.0,
                "collect_privileged_belief": False,
            }
        )
        provenance = dict(job.get("target_provenance") or {})
        provenance.update(
            {
                "family_id": SPECIALIST_ID,
                "variant_id": variant_id,
                "manifest_digest": str(manifest["artifact_sha256"]),
                "ordered_digest": str(variants[variant_id]["ordered_digest"]),
                "multiset_digest": str(variants[variant_id]["multiset_digest"]),
                "study_phase": str(schedule["phase"]),
                "schedule_id": str(schedule["schedule_id"]),
            }
        )
        job["target_provenance"] = provenance
        jobs.append(job)
        schedule_by_index[job_index] = schedule
    return jobs, schedule_by_index


def _run_panel(
    *,
    output_path: Path,
    panel: Sequence[Mapping[str, Any]],
    treatments: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
    specs: Sequence[Any],
    workers: int,
    game_timeout_seconds: int,
) -> list[dict[str, Any]]:
    from poke_bot.worker_pool import WorkerPool
    from scripts.train_round_robin import _worker_play

    if output_path.is_file():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(existing, list):
            raise RuntimeError("existing family panel rows are malformed")
        return existing
    all_rows: list[dict[str, Any]] = []
    for treatment, checkpoint in treatments.items():
        jobs, schedule_by_index = _panel_jobs(
            panel=panel,
            treatment=treatment,
            checkpoint=checkpoint,
            manifest=manifest,
            specs=specs,
            game_timeout_seconds=game_timeout_seconds,
        )
        with _cpu_only_panel_spawn_environment():
            with WorkerPool(
                num_workers=min(max(1, int(workers)), len(jobs))
            ) as pool:
                results = list(pool.imap_unordered(_worker_play, jobs))
        by_index = {int(row["job_index"]): row for row in results}
        if set(by_index) != set(schedule_by_index):
            raise RuntimeError("family panel result cardinality changed")
        for job_index in sorted(schedule_by_index):
            result = by_index[job_index]
            schedule = schedule_by_index[job_index]
            invalid = bool(
                result.get("error")
                or result.get("baseline_failed")
                or result.get("our_failed")
                or result.get("resource_error")
                or result.get("trust_failure")
                or result.get("cancelled")
            )
            winner = int(result.get("winner", 2))
            seat = int(result.get("our_seat", -1))
            score = 0.5 if winner == 2 else float(winner == seat)
            decisions = max(1, int(result.get("n_decisions") or 0))
            wall_seconds = float(result.get("wall_s") or 0.0)
            causal_integrity = bool(
                not invalid
                and result.get("training_eligible") is False
                and result.get("record_json") is None
                and result.get("action_selection") == "greedy"
            )
            all_rows.append(
                {
                    **dict(schedule),
                    "treatment": treatment,
                    "checkpoint_sha256": str(checkpoint["sha256"]),
                    "score": score,
                    "invalid": invalid,
                    "error": result.get("error"),
                    "wall_seconds": wall_seconds,
                    "decisions": decisions,
                    "decision_latency_seconds": wall_seconds / decisions,
                    "causal_integrity": causal_integrity,
                    "training_eligible": False,
                    "replay_eligible": False,
                    "formal_gate": False,
                }
            )
    _write_immutable(output_path, all_rows)
    return all_rows


def _study_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "epochs": 5,
        "games_per_batch": 240,
        "max_decisions_per_batch": 2048,
        "val_frac": 0.10,
        "max_context": 320,
        "split_seed": int(args.seed),
        "batch_order_seed": int(args.seed),
        "awr_beta": 0.5,
        "awr_weight_max": 20.0,
        "entropy_bonus": 0.01,
        "shadow_guide_loss_weight": SHADOW_GUIDE_LOSS_WEIGHT,
        "seed": int(args.seed),
        "curriculum_spec": str(args.curriculum_spec.resolve()),
        "head_role_map": str(args.head_role_map.resolve()),
        "curriculum_validation": str(args.curriculum_validation.resolve()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from poke_bot.pure_rl.awr_shadow_study import exact_training_order_contract
    from poke_bot.pure_rl.dataset_bridge import dataset_from_shard
    from poke_bot.train import (
        TrainConfig,
        archetype_residual_gradient_diagnostics,
        policy_drift_diagnostics,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trigger = _read(args.trigger)
    validate_iteration9_upload_trigger(trigger)
    pause = _read(args.pause)
    if (
        pause.get("schema") != PAUSE_SCHEMA
        or pause.get("next_collection_started") is not False
        or int(pause.get("restart_prevent_status", -1)) != 75
    ):
        raise RuntimeError("family study lacks the trainer's exact status-75 pause")
    manifest = validate_manifest(_read(args.manifest), require_activation_ready=True)
    loss_contract = validate_loss_contract(_read(args.loss_contract))
    loop = _read(args.loop_state)
    parent = dict(loop.get("learner") or {})
    if (
        parent.get("digest") != pause.get("learner_sha256")
        or sha256(Path(str(parent.get("path", "")))) != parent.get("digest")
    ):
        raise RuntimeError("paused learner and loop-state learner differ")
    parent_identity = {
        "path": str(Path(str(parent["path"])).resolve()),
        "sha256": str(parent["digest"]),
    }
    specs, opponent_ids = _load_gate_specs(args.active_gate_contract)
    plan = family_shadow_plan(manifest, opponent_ids, seed_book=str(args.seed_book))
    plan_path = _write_immutable(output_dir / "shadow_plan.json", plan)
    shard, sealed_rows = _collect_training_shard(
        output_dir=output_dir,
        parent=parent_identity,
        manifest=manifest,
        specs=specs,
        workers=int(args.workers),
        game_timeout_seconds=int(args.game_timeout_seconds),
        seed=int(args.seed),
    )
    dataset = dataset_from_shard(shard, verify_info_set=False, max_context=320)
    controls = _study_config(args)
    order = exact_training_order_contract(dataset, controls)
    common = {
        "parent_checkpoint_sha256": str(parent_identity["sha256"]),
        "sealed_rows_sha256": sha256(shard),
        "split_sha256": str(order["split_sha256"]),
        "batch_order_sha256": str(order["batch_order_sha256"]),
        "optimizer_settings_sha256": _canonical_digest(controls),
        "seed_book_sha256": _canonical_digest(
            {"seed_book": args.seed_book, "seed": int(args.seed)}
        ),
        "update_count": sum(int(value) for value in order["batches_per_epoch"]),
        "manifest_sha256": sha256(args.manifest),
        "shadow_plan_sha256": sha256(plan_path),
    }
    base_weights = canonical_residual_weights(
        {
            name: float(row["weight"])
            for name, row in loss_contract["residual_objectives"].items()
        }
    )
    rounds: list[dict[str, Any]] = []
    selected_training: dict[str, Any] | None = None
    selected_weights: dict[str, float] | None = None
    for round_index in (1, 2):
        perturbation = antithetic_spsa(
            base_weights,
            round_index=round_index,
            seed_book=f"{args.seed_book}:round-{round_index}",
        )
        training: dict[str, dict[str, Any]] = {}
        for direction in ("plus", "minus"):
            label = f"round{round_index}-{direction}"
            training[direction] = _train_candidate(
                output_dir=output_dir,
                label=label,
                parent=parent_identity,
                dataset=dataset,
                controls={**controls, "seed": int(args.seed) + round_index * 1000},
                residuals=perturbation["candidates"][direction],
                common_provenance={
                    **common,
                    "seed_book_sha256": _canonical_digest(
                        {
                            "seed_book": args.seed_book,
                            "round": round_index,
                            "seed": int(args.seed) + round_index * 1000,
                        }
                    ),
                },
                device=str(args.device),
            )
        same_parent = validate_same_parent_shadow(
            {"plus": training["plus"], "minus": training["minus"]}
        )
        development_rows = _run_panel(
            output_path=output_dir / f"round{round_index}-development.json",
            panel=plan["panels"]["development"],
            treatments={
                "plus": training["plus"]["checkpoint"],
                "minus": training["minus"]["checkpoint"],
            },
            manifest=manifest,
            specs=specs,
            workers=int(args.workers),
            game_timeout_seconds=int(args.game_timeout_seconds),
        )
        result = compile_development_round(
            development_rows, round_index=round_index
        )
        round_receipt = {
            **result,
            "perturbation": perturbation,
            "same_parent_validation": same_parent,
            "training_receipts": {
                direction: _identity(
                    output_dir
                    / "training_receipts"
                    / f"round{round_index}-{direction}.json"
                )
                for direction in ("plus", "minus")
            },
            "development_rows": _identity(
                output_dir / f"round{round_index}-development.json"
            ),
        }
        rounds.append(round_receipt)
        _write_immutable(output_dir / f"round{round_index}.json", round_receipt)
        direction = result.get("selected_direction")
        if result.get("status") == "conclusive" and direction in {"plus", "minus"}:
            selected_training = training[str(direction)]
            selected_weights = dict(perturbation["candidates"][str(direction)])
            break
    if selected_training is None or selected_weights is None:
        failed = {
            "schema": SCHEMA,
            "passed": False,
            "status": "failed_closed_inconclusive_after_two_rounds",
            "rounds": rounds,
            "same_parent_validation": rounds[-1]["same_parent_validation"],
            "training_eligible": False,
            "replay_eligible": False,
        }
        _write_immutable(output_dir / "study.json", failed)
        return {"status": "failed_closed", "study": str(output_dir / "study.json")}
    selected_checkpoint = dict(selected_training["checkpoint"])
    locked_rows = _run_panel(
        output_path=output_dir / "locked-confirmation.json",
        panel=plan["panels"]["locked"],
        treatments={"candidate": selected_checkpoint, "parent": parent_identity},
        manifest=manifest,
        specs=specs,
        workers=int(args.workers),
        game_timeout_seconds=int(args.game_timeout_seconds),
    )
    package_rows = _run_panel(
        output_path=output_dir / "package-guard.json",
        panel=plan["panels"]["package"],
        treatments={"candidate": selected_checkpoint, "parent": parent_identity},
        manifest=manifest,
        specs=specs,
        workers=int(args.workers),
        game_timeout_seconds=int(args.game_timeout_seconds),
    )
    shadow_guide_loss_weight = _validated_shadow_guide_loss_weight(controls)
    diagnostic_cfg = TrainConfig.pure_rl_defaults(
        games_per_batch=int(controls["games_per_batch"]),
        max_decisions_per_batch=int(controls["max_decisions_per_batch"]),
        seed=int(controls["seed"]),
        amp=str(args.device).startswith("cuda"),
        value_loss_weight=1.0,
        aux_loss_weight=0.05,
        opp_hand_loss_weight=0.05,
        opp_remainder_loss_weight=0.05,
        lethal_threat_loss_weight=0.025,
        prize_race_loss_weight=0.025,
        alakazam_guide_loss_weight=shadow_guide_loss_weight,
        current_deck_guide_training_mode="strategic_directional_v2",
        setup_board_outcome_loss_weight=0.025,
        combo_state_loss_weight=0.025,
        current_deck_guide_curriculum_spec=str(args.curriculum_spec.resolve()),
        current_deck_guide_head_role_map=str(args.head_role_map.resolve()),
        current_deck_guide_curriculum_validation_receipt=str(
            args.curriculum_validation.resolve()
        ),
        archetype_residual_loss_weights=selected_weights,
    )
    gradient = archetype_residual_gradient_diagnostics(
        checkpoint_path=parent_identity["path"],
        dataset=dataset,
        cfg=diagnostic_cfg,
        device=torch.device(str(args.device)),
    )
    drift = policy_drift_diagnostics(
        parent_checkpoint=parent_identity["path"],
        candidate_checkpoint=selected_checkpoint["path"],
        dataset=dataset,
        cfg=diagnostic_cfg,
        device=torch.device(str(args.device)),
    )
    metrics = compile_activation_metrics(
        locked_rows=locked_rows,
        package_rows=package_rows,
        gradient_diagnostics=gradient,
        policy_drift=drift,
    )
    gate = activation_gate(metrics)
    selected_round = int(rounds[-1]["round"])
    selected_direction = str(rounds[-1]["selected_direction"])
    study = {
        "schema": SCHEMA,
        "passed": bool(gate["passed"]),
        "status": "passed" if gate["passed"] else "failed_closed",
        "selection": {
            "activate": bool(gate["passed"]),
            "round": selected_round,
            "direction": selected_direction,
            "weights": selected_weights,
            "checkpoint": selected_checkpoint,
        },
        "rounds": rounds,
        "same_parent_validation": rounds[-1]["same_parent_validation"],
        "activation_metrics": metrics,
        "activation_gate": gate,
        "gradient_diagnostics": gradient,
        "policy_drift": drift,
        "plan": _identity(plan_path),
        "sealed_training_rows": sealed_rows,
        "locked_rows": _identity(output_dir / "locked-confirmation.json"),
        "package_rows": _identity(output_dir / "package-guard.json"),
        "training_eligible": False,
        "replay_eligible": False,
        "serving_allowed": False,
        "promotion_allowed": False,
        "kaggle_evidence_used_for_training_or_tuning": False,
    }
    study_path = _write_immutable(output_dir / "study.json", study)
    if not gate["passed"]:
        return {"status": "failed_closed", "study": str(study_path)}
    vector = selected_loss_vector(
        weights=selected_weights,
        manifest_sha256=sha256(args.manifest),
        loss_contract_sha256=sha256(args.loss_contract),
        study_sha256=sha256(study_path),
        selected_round=selected_round,
        selected_direction=selected_direction,
    )
    vector_path = _write_immutable(output_dir / "selected-loss-vector.json", vector)
    before_registry = _read(args.before_registry)
    candidate_registry = copy.deepcopy(before_registry)
    candidate_runtime_root = args.candidate_runtime_root.expanduser().resolve()
    if not candidate_runtime_root.is_dir():
        raise FileNotFoundError(candidate_runtime_root)
    for field in ("trainer", "active_gate", "research_gate", "frozen_gate"):
        relative = Path(str(candidate_registry.get(field) or ""))
        if relative.is_absolute() or not str(relative):
            raise ValueError(f"candidate registry {field} must be relative")
        if not (candidate_runtime_root / relative).is_file():
            raise FileNotFoundError(candidate_runtime_root / relative)
    candidate_registry["runtime_root"] = str(candidate_runtime_root)
    trainer_args = list(candidate_registry.get("common_trainer_args") or [])
    try:
        reason_index = trainer_args.index("--boundary-design-migration-reason") + 1
    except ValueError as exc:
        raise RuntimeError(
            "runtime registry lacks a clean-boundary migration reason"
        ) from exc
    if reason_index >= len(trainer_args):
        raise RuntimeError("runtime registry has an empty migration reason")
    trainer_args[reason_index] = "receipt_backed_marnie_family_activation_r133"
    candidate_registry.update(
        {
            "common_trainer_args": trainer_args,
            "deck_family_distribution": "package_20_equal_cluster_80_v1",
            "family_manifest": _identity(args.manifest),
            "variant_provenance": "checksum_bound_exact_observed_lists_v1",
            "replay_list_sampler": "deterministic_hamilton_family_macro_v1",
            "archetype_loss_contract": _identity(args.loss_contract),
            "selected_loss_vector": _identity(vector_path),
        }
    )
    candidate_registry_path = _write_immutable(
        output_dir / "candidate-runtime-registry.json", candidate_registry
    )
    selector_identity = _identity(args.selector)
    checkpoint_payload = torch.load(
        parent_identity["path"], map_location="cpu", weights_only=False
    )
    design_contract_path = args.run_manifest.resolve()
    absent_manifest = _canonical_digest({"status": "absent_before_activation"})
    absent_vector = _canonical_digest({"status": "absent_before_activation"})
    request = {
        "schema": REQUEST_SCHEMA,
        "status": "ready_for_atomic_activation",
        "trigger": _identity(args.trigger),
        "manifest": _identity(args.manifest),
        "loss_contract": _identity(args.loss_contract),
        "study": _identity(study_path),
        "bindings": {
            "learner_sha256": str(parent_identity["sha256"]),
            "checkpoint": _identity(Path(str(parent_identity["path"]))),
            "registry": _identity(args.before_registry),
            "selector": selector_identity,
            "selected_loss_vector": _identity(vector_path),
            "candidate_registry": _identity(candidate_registry_path),
        },
        "sealed_pre_activation": {
            "registry": sha256(args.before_registry),
            "selector": selector_identity["sha256"],
            "learner": str(parent_identity["sha256"]),
            "checkpoint": str(parent_identity["sha256"]),
            "optimizer": _hash_torch_value(
                checkpoint_payload.get("optimizer_state_dict")
            ),
            "scaler": _hash_torch_value(
                checkpoint_payload.get("scaler_state_dict")
            ),
            "rng": _hash_torch_value(
                {
                    key: checkpoint_payload.get(key)
                    for key in (
                        "rng_state", "torch_rng_state", "cuda_rng_state",
                        "python_rng_state", "numpy_rng_state",
                    )
                }
            ),
            "design_contract": sha256(design_contract_path),
            "manifest": absent_manifest,
            "loss_vector": absent_vector,
        },
        "atomic_changes_only": [
            "runtime_root", "deck_family_distribution", "family_manifest", "variant_provenance",
            "replay_list_sampler", "archetype_loss_contract", "selected_loss_vector",
        ],
        "next_collection_started": False,
    }
    request_path = _write_immutable(args.activation_request, request)
    validate_activation_request(
        _read(request_path),
        expected_learner_digest=str(parent_identity["sha256"]),
    )
    return {
        "status": "ready_for_atomic_activation",
        "study": str(study_path),
        "selected_loss_vector": str(vector_path),
        "candidate_registry": str(candidate_registry_path),
        "activation_request": str(request_path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trigger", type=Path, required=True)
    parser.add_argument("--pause", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--loss-contract", type=Path, required=True)
    parser.add_argument("--loop-state", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--active-gate-contract", type=Path, required=True)
    parser.add_argument("--before-registry", type=Path, required=True)
    parser.add_argument("--candidate-runtime-root", type=Path, default=Path.cwd())
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--curriculum-spec", type=Path, required=True)
    parser.add_argument("--head-role-map", type=Path, required=True)
    parser.add_argument("--curriculum-validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--activation-request", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--game-timeout-seconds", type=int, default=600)
    parser.add_argument("--seed", type=int, default=130009)
    parser.add_argument("--seed-book", default="marnie-family-r130-v1")
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run(args)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "ready_for_atomic_activation" else 76


if __name__ == "__main__":
    raise SystemExit(main())
