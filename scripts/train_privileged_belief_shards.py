#!/usr/bin/env python
"""Memory-bounded training over the exact hidden-state replay corpus.

The canonical corpus remains sharded on Inzi and one shard is resident in host
memory at a time.  Head-only mode updates the five belief/strategy heads.  In
``--full-model --pure-rl`` mode the entire policy/value model is continued with
outcome-weighted AWR plus the exact hidden-state auxiliary targets.  Policy
inputs are always ordinary competition-masked observations; privileged cards
appear only as loss targets.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from tqdm.auto import tqdm

from poke_bot import archetypes, checkpoint
from poke_bot.dataset import BootstrapDataset, GameSequence
from poke_bot.train import (
    batch_losses,
    expand_aux_head_to_current_registry,
    load_model_from_checkpoint,
)


HEAD_PREFIXES: tuple[str, ...] = (
    "aux_head.",
    "opp_hand_head.",
    "opp_remainder_head.",
    "lethal_threat_head.",
    "prize_race_head.",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--latest-checkpoint", type=Path, required=True)
    parser.add_argument("--best-checkpoint", type=Path, required=True)
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--full-model",
        action="store_true",
        help="Update the complete policy/value/trunk plus auxiliary heads.",
    )
    parser.add_argument(
        "--pure-rl",
        action="store_true",
        help=(
            "Use outcome-weighted AWR on selected actions. This is required "
            "for full-model continuation from trajectories collected by an "
            "older policy; it avoids behavior-cloning that older policy."
        ),
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    # One 128-game collector shard is ~20-22k decision stages.  The 3080 Ti
    # measured this all-shard batch at 4.31 GiB peak, so 256/32768 maximizes
    # useful CUDA work while retaining >7 GiB VRAM headroom.
    parser.add_argument("--games-per-batch", type=int, default=256)
    parser.add_argument("--max-decisions-per-batch", type=int, default=32768)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Persistent per-shard feature cache (default: .feature-cache next "
            "to the canonical manifest). Later epochs/restarts avoid redoing "
            "the expensive JSON-to-feature conversion."
        ),
    )
    parser.add_argument("--val-modulus", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="Minimum held-out objective decrease counted as improvement.",
    )
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--aux-weight", type=float, default=0.10)
    parser.add_argument("--hand-weight", type=float, default=0.40)
    parser.add_argument("--remainder-weight", type=float, default=0.30)
    parser.add_argument("--lethal-weight", type=float, default=0.10)
    parser.add_argument("--prize-race-weight", type=float, default=0.10)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--awr-beta", type=float, default=0.5)
    parser.add_argument("--awr-weight-max", type=float, default=20.0)
    parser.add_argument("--entropy-bonus", type=float, default=0.01)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _tensor_digest(model: nn.Module, *, heads: bool) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        is_head = name.startswith(HEAD_PREFIXES)
        if is_head != heads:
            continue
        name_b = name.encode("utf-8")
        tensor = value.detach().contiguous().cpu()
        digest.update(len(name_b).to_bytes(8, "big"))
        digest.update(name_b)
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(bytes(tensor.numpy()))
    return "sha256:" + digest.hexdigest()


def _expand_aux_head(model: nn.Module) -> bool:
    return expand_aux_head_to_current_registry(model)


def _manifest_shards(path: Path) -> tuple[dict[str, Any], list[Path]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "poke_bot.privileged_belief_corpus/v1":
        raise RuntimeError("head warm-start requires the canonical Inzi corpus")
    if payload.get("storage_authority") != "inzi":
        raise RuntimeError("privileged corpus storage authority is not Inzi")
    shards: list[Path] = []
    for row in payload.get("shards") or []:
        shard = (path.parent / str(row["path"])).resolve()
        if not shard.is_file() or _sha256(shard) != row.get("sha256"):
            raise RuntimeError(f"canonical shard failed checksum: {shard}")
        if int(row.get("decisions", -1)) != int(
            row.get("hand_labeled_decisions", -2)
        ):
            raise RuntimeError(f"canonical shard is partially labeled: {shard}")
        shards.append(shard)
    if len(shards) != int((payload.get("totals") or {}).get("shards", -1)):
        raise RuntimeError("canonical manifest shard count mismatch")
    return payload, shards


def _is_validation(path: Path, modulus: int) -> bool:
    # Include the source directory because distributed collectors reuse shard
    # names (for example, every host has shard_00000.jsonl).
    split_key = f"{path.parent.name}/{path.name}"
    value = int(hashlib.sha256(split_key.encode("utf-8")).hexdigest()[:16], 16)
    return value % modulus == 0


def _batches(
    sequences: list[GameSequence],
    *,
    games_per_batch: int,
    max_decisions: int,
    rng: random.Random | None,
) -> Iterable[list[GameSequence]]:
    order = list(range(len(sequences)))
    if rng is not None:
        rng.shuffle(order)
    current: list[GameSequence] = []
    decisions = 0
    for index in order:
        sequence = sequences[index]
        n = len(sequence)
        if current and (
            len(current) >= games_per_batch or decisions + n > max_decisions
        ):
            yield current
            current = []
            decisions = 0
        current.append(sequence)
        decisions += n
    if current:
        yield current


def _head_objective(metrics: Any, args: argparse.Namespace) -> float:
    if bool(args.full_model):
        return float(metrics.total_loss)
    return (
        args.aux_weight * float(metrics.aux_loss)
        + args.hand_weight * float(metrics.opp_hand_loss)
        + args.remainder_weight * float(metrics.opp_remainder_loss)
        + args.lethal_weight * float(metrics.lethal_threat_loss)
        + args.prize_race_weight * float(metrics.prize_race_loss)
    )


def _run_shard(
    model: nn.Module,
    shard: Path,
    args: argparse.Namespace,
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    train: bool,
    seed: int,
) -> dict[str, float]:
    dataset = BootstrapDataset.from_jsonl(
        shard,
        max_context=int(model.cfg.max_context),
        verify_info_set=True,
        use_cache=True,
        cache_dir=Path(args.cache_dir),
    )
    if not dataset.info_set_ok_all or dataset.n_decisions <= 0:
        raise RuntimeError(f"invalid/empty canonical shard: {shard}")
    if bool(args.pure_rl):
        # The collector records diagnostic soft probabilities from the older
        # epoch-4 behavior policy.  Pure RL must use only the selected action
        # plus game outcome; retaining these would either clone the older
        # policy or trigger batch_losses' deliberate fail-closed contract.
        for sequence in dataset.sequences:
            sequence.policy_targets = None
            sequence.factorized_policy_targets = None
    if train and bool(args.full_model):
        model.train()
    elif train:
        model.eval()
        for module_name in (
            "aux_head",
            "opp_hand_head",
            "opp_remainder_head",
            "lethal_threat_head",
            "prize_race_head",
        ):
            getattr(model, module_name).train()
    else:
        model.eval()
    use_amp = model.aux_head[-1].weight.device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if use_amp and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    totals = {
        "objective": 0.0,
        "total": 0.0,
        "policy": 0.0,
        "value": 0.0,
        "hand": 0.0,
        "remainder": 0.0,
        "aux": 0.0,
        "lethal": 0.0,
        "prize_race": 0.0,
        "policy_accuracy": 0.0,
        "policy_selected_nll": 0.0,
        "raw_advantage_mean_abs": 0.0,
        "awr_weight_mean": 0.0,
        "awr_weight_p95": 0.0,
        "awr_weight_clip_frac": 0.0,
        "awr_effective_sample_fraction": 0.0,
        "decisions": 0.0,
        "batches": 0.0,
    }
    batches = list(_batches(
        list(dataset.sequences),
        games_per_batch=int(args.games_per_batch),
        max_decisions=int(args.max_decisions_per_batch),
        rng=random.Random(seed) if train else None,
    ))

    # Match the production pure-RL contract: do not let V(s) drift underneath
    # the AWR weights as optimizer steps advance.  The corpus is deliberately
    # sharded for memory containment, so freeze once at each shard boundary.
    awr_baseline_cache: dict[tuple[int, int, int], float] | None = None
    if train and bool(args.full_model) and bool(args.pure_rl):
        awr_baseline_cache = {}
        model.eval()
        with torch.no_grad():
            for batch in batches:
                with torch.amp.autocast(
                    "cuda", enabled=use_amp, dtype=amp_dtype
                ):
                    batch_losses(
                        model,
                        batch,
                        value_weight=float(args.value_weight),
                        aux_weight=float(args.aux_weight),
                        opp_hand_weight=float(args.hand_weight),
                        opp_remainder_weight=float(args.remainder_weight),
                        lethal_threat_weight=float(args.lethal_weight),
                        prize_race_weight=float(args.prize_race_weight),
                        pure_rl=True,
                        awr_beta=float(args.awr_beta),
                        awr_weight_max=float(args.awr_weight_max),
                        awr_normalize_advantages=True,
                        entropy_bonus=float(args.entropy_bonus),
                        awr_capture_baseline=awr_baseline_cache,
                    )
        model.train()

    for batch in batches:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        grad_context = torch.enable_grad() if train else torch.no_grad()
        with grad_context:
            with torch.amp.autocast(
                "cuda", enabled=use_amp, dtype=amp_dtype
            ):
                loss, metrics = batch_losses(
                    model,
                    batch,
                    value_weight=float(args.value_weight),
                    aux_weight=float(args.aux_weight),
                    opp_hand_weight=float(args.hand_weight),
                    opp_remainder_weight=float(args.remainder_weight),
                    lethal_threat_weight=float(args.lethal_weight),
                    prize_race_weight=float(args.prize_race_weight),
                    pure_rl=bool(args.pure_rl),
                    awr_beta=float(args.awr_beta),
                    awr_weight_max=float(args.awr_weight_max),
                    awr_normalize_advantages=True,
                    entropy_bonus=float(args.entropy_bonus),
                    awr_baseline_cache=awr_baseline_cache,
                )
        if train:
            assert optimizer is not None and scaler is not None
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            trainable = [
                parameter for parameter in model.parameters() if parameter.requires_grad
            ]
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
        weight = max(1, int(metrics.n_decisions))
        totals["objective"] += _head_objective(metrics, args) * weight
        totals["total"] += float(metrics.total_loss) * weight
        totals["policy"] += float(metrics.policy_loss) * weight
        totals["value"] += float(metrics.value_loss) * weight
        totals["hand"] += float(metrics.opp_hand_loss) * weight
        totals["remainder"] += float(metrics.opp_remainder_loss) * weight
        totals["aux"] += float(metrics.aux_loss) * weight
        totals["lethal"] += float(metrics.lethal_threat_loss) * weight
        totals["prize_race"] += float(metrics.prize_race_loss) * weight
        totals["policy_accuracy"] += float(metrics.policy_acc) * weight
        totals["policy_selected_nll"] += float(metrics.policy_selected_nll) * weight
        totals["raw_advantage_mean_abs"] += (
            float(metrics.raw_advantage_mean_abs) * weight
        )
        totals["awr_weight_mean"] += float(metrics.awr_weight_mean) * weight
        totals["awr_weight_p95"] += float(metrics.awr_weight_p95) * weight
        totals["awr_weight_clip_frac"] += (
            float(metrics.awr_weight_clip_frac) * weight
        )
        totals["awr_effective_sample_fraction"] += (
            float(metrics.awr_effective_sample_fraction) * weight
        )
        totals["decisions"] += weight
        totals["batches"] += 1
    denom = max(1.0, totals["decisions"])
    for key in (
        "objective",
        "total",
        "policy",
        "value",
        "hand",
        "remainder",
        "aux",
        "lethal",
        "prize_race",
        "policy_accuracy",
        "policy_selected_nll",
        "raw_advantage_mean_abs",
        "awr_weight_mean",
        "awr_weight_p95",
        "awr_weight_clip_frac",
        "awr_effective_sample_fraction",
    ):
        totals[key] /= denom
    del dataset
    gc.collect()
    return totals


def _weighted_merge(rows: list[dict[str, float]]) -> dict[str, float]:
    decisions = sum(row["decisions"] for row in rows)
    merged = {
        "decisions": decisions,
        "batches": sum(row["batches"] for row in rows),
    }
    for key in (
        "objective",
        "total",
        "policy",
        "value",
        "hand",
        "remainder",
        "aux",
        "lethal",
        "prize_race",
        "policy_accuracy",
        "policy_selected_nll",
        "raw_advantage_mean_abs",
        "awr_weight_mean",
        "awr_weight_p95",
        "awr_weight_clip_frac",
        "awr_effective_sample_fraction",
    ):
        merged[key] = sum(row[key] * row["decisions"] for row in rows) / max(
            1.0, decisions
        )
    return merged


def main() -> int:
    args = _parse_args()
    if args.epochs < 1 or args.val_modulus < 2:
        raise ValueError("epochs must be positive and val-modulus >= 2")
    if args.min_delta < 0:
        raise ValueError("min-delta must be non-negative")
    if args.full_model and not args.pure_rl:
        raise ValueError(
            "--full-model requires --pure-rl so older collector actions are "
            "never treated as behavior-cloning targets"
        )
    manifest_path = args.manifest.expanduser().resolve()
    args.cache_dir = (
        args.cache_dir.expanduser().resolve()
        if args.cache_dir is not None
        else manifest_path.parent / ".feature-cache"
    )
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    init_path = args.init_checkpoint.expanduser().resolve()
    latest_path = args.latest_checkpoint.expanduser().resolve()
    best_path = args.best_checkpoint.expanduser().resolve()
    status_path = args.status_json.expanduser().resolve()
    manifest, shards = _manifest_shards(manifest_path)
    train_shards = [p for p in shards if not _is_validation(p, args.val_modulus)]
    val_shards = [p for p in shards if _is_validation(p, args.val_modulus)]
    if not train_shards or not val_shards:
        raise RuntimeError("deterministic shard split produced an empty partition")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("privileged replay training requires a CUDA GPU")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    resume = latest_path.is_file()
    source = latest_path if resume else init_path
    model = load_model_from_checkpoint(source, device=device)
    expanded = _expand_aux_head(model)
    expected_ids = list(archetypes.archetype_ids())
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            bool(args.full_model) or name.startswith(HEAD_PREFIXES)
        )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )
    use_bf16 = torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)

    epoch = 0
    shard_cursor = 0
    global_step = 0
    best_metric = float("inf")
    patience_left = int(args.patience)
    history: list[dict[str, Any]] = []
    init_digest = checkpoint.checkpoint_digest(init_path)
    init_saved = checkpoint.load_checkpoint(init_path, map_location="cpu")
    source_epoch_index = int(init_saved.get("epoch", -1))
    source_history = list((init_saved.get("extra") or {}).get("history") or [])
    source_epochs_completed = max(source_epoch_index + 1, len(source_history))
    current_non_head_digest = _tensor_digest(model, heads=False)
    source_non_head_digest = current_non_head_digest
    if resume:
        saved = checkpoint.load_checkpoint(latest_path, map_location=device)
        extra = dict(saved.get("extra") or {}).get("privileged_belief_training") or {}
        if extra.get("corpus_digest") != manifest.get("corpus_digest"):
            raise RuntimeError("resume checkpoint belongs to another corpus")
        if list(extra.get("archetype_ids") or []) != expected_ids:
            raise RuntimeError("resume checkpoint archetype-row mapping changed")
        source_non_head_digest = str(
            extra.get("source_non_head_tensor_digest")
            or extra.get("frozen_tensor_digest")
            or ""
        )
        expected_current = str(
            extra.get("current_non_head_tensor_digest")
            or extra.get("frozen_tensor_digest")
            or ""
        )
        if expected_current != current_non_head_digest:
            raise RuntimeError("resume checkpoint policy/trunk digest changed")
        if bool(extra.get("full_model", False)) != bool(args.full_model):
            raise RuntimeError("resume checkpoint full-model mode changed")
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        if saved.get("scaler_state_dict"):
            scaler.load_state_dict(saved["scaler_state_dict"])
        epoch = int(extra.get("epoch", 0))
        shard_cursor = int(extra.get("shard_cursor", 0))
        global_step = int(saved.get("step", 0))
        best_metric = float(saved.get("best_metric", float("inf")))
        patience_left = int(extra.get("patience_left", args.patience))
        history = list(extra.get("history") or [])
    elif expanded:
        print(
            f"[belief-train] expanded aux classes to {len(expected_ids) + 1} "
            f"({len(expected_ids)} named + unknown)",
            flush=True,
        )

    weights = {
        "aux": float(args.aux_weight),
        "hand": float(args.hand_weight),
        "remainder": float(args.remainder_weight),
        "lethal": float(args.lethal_weight),
        "prize_race": float(args.prize_race_weight),
    }

    def save(path: Path, *, next_epoch: int, next_cursor: int) -> None:
        current_non_head = _tensor_digest(model, heads=False)
        if not args.full_model and current_non_head != source_non_head_digest:
            raise RuntimeError("policy/trunk tensor changed during head-only training")
        extra = {
            "pure_rl": bool(args.pure_rl),
            "smoke": False,
            "privileged_belief_training": {
                "schema": "poke_bot.privileged_belief_train/v2",
                "corpus_manifest": str(manifest_path),
                "corpus_digest": manifest["corpus_digest"],
                "source_checkpoint": str(init_path),
                "source_checkpoint_digest": init_digest,
                "policy_inputs": "competition_masked_observation_only",
                "privileged_values": "loss_targets_only",
                "policy_trunk_frozen": not bool(args.full_model),
                "full_model": bool(args.full_model),
                "pure_rl": bool(args.pure_rl),
                "source_non_head_tensor_digest": source_non_head_digest,
                "current_non_head_tensor_digest": current_non_head,
                "frozen_tensor_digest": (
                    source_non_head_digest if not args.full_model else None
                ),
                "archetype_ids": expected_ids,
                "unknown_class_index": len(expected_ids),
                "weights": weights,
                "source_epoch_index": source_epoch_index,
                "source_epochs_completed": source_epochs_completed,
                "continuation_epochs_target": int(args.epochs),
                "min_delta": float(args.min_delta),
                "epoch": next_epoch,
                "shard_cursor": next_cursor,
                "patience_left": patience_left,
                "history": history,
            },
        }
        payload = checkpoint.build_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            step=global_step,
            epoch=source_epoch_index + next_epoch,
            best_metric=best_metric,
            model_config=model.cfg,
            model_id="state_core_privileged_belief_20k_v1",
            archetype_id="top-ladder-core",
            extra=extra,
        )
        checkpoint.atomic_torch_save(payload, path)

    _atomic_json(
        status_path,
        {
            "schema": "poke_bot.privileged_belief_train_status/v2",
            "status": "training",
            "mode": "full_model_pure_rl" if args.full_model else "heads_only",
            "device": str(device),
            "corpus_digest": manifest["corpus_digest"],
            "train_shards": len(train_shards),
            "val_shards": len(val_shards),
            "epoch": epoch,
            "total_epoch_index": source_epoch_index + epoch + 1,
            "source_epoch_index": source_epoch_index,
            "shard_cursor": shard_cursor,
            "updated_unix": int(time.time()),
        },
    )
    print(
        f"[belief-train] device={device} train_shards={len(train_shards)} "
        f"val_shards={len(val_shards)} full_model={bool(args.full_model)} "
        f"pure_rl={bool(args.pure_rl)} source_epoch={source_epoch_index} "
        f"target_epoch={source_epoch_index + int(args.epochs)} "
        f"source_non_head={source_non_head_digest}",
        flush=True,
    )

    while epoch < args.epochs and patience_left > 0:
        order = list(train_shards)
        random.Random(args.seed + epoch * 10007).shuffle(order)
        train_rows: list[dict[str, float]] = []
        bar = tqdm(
            range(shard_cursor, len(order)),
            desc=f"belief full ep{source_epoch_index + epoch + 1}" if args.full_model else f"belief heads ep{epoch}",
            unit="shard",
            initial=shard_cursor,
            total=len(order),
        )
        for position in bar:
            row = _run_shard(
                model,
                order[position],
                args,
                optimizer=optimizer,
                scaler=scaler,
                train=True,
                seed=args.seed + epoch * 100000 + position,
            )
            train_rows.append(row)
            global_step += int(row["batches"])
            bar.set_postfix(
                obj=f"{row['objective']:.4f}",
                hand=f"{row['hand']:.3f}",
                rem=f"{row['remainder']:.3f}",
            )
            save(latest_path, next_epoch=epoch, next_cursor=position + 1)
            _atomic_json(
                status_path,
                {
                    "schema": "poke_bot.privileged_belief_train_status/v2",
                    "status": "training",
                    "mode": "full_model_pure_rl" if args.full_model else "heads_only",
                    "device": str(device),
                    "corpus_digest": manifest["corpus_digest"],
                    "epoch": epoch,
                    "total_epoch_index": source_epoch_index + epoch + 1,
                    "source_epoch_index": source_epoch_index,
                    "shard_cursor": position + 1,
                    "train_shards": len(order),
                    "val_shards": len(val_shards),
                    "last_train": row,
                    "updated_unix": int(time.time()),
                },
            )

        val_rows = []
        for position, shard in enumerate(
            tqdm(
                val_shards,
                desc=f"belief val ep{source_epoch_index + epoch + 1}" if args.full_model else f"belief val ep{epoch}",
                unit="shard",
            )
        ):
            val_rows.append(
                _run_shard(
                    model,
                    shard,
                    args,
                    optimizer=None,
                    scaler=None,
                    train=False,
                    seed=args.seed + position,
                )
            )
        train_metrics = _weighted_merge(train_rows) if train_rows else {}
        val_metrics = _weighted_merge(val_rows)
        metric = float(val_metrics["objective"])
        improved = metric < best_metric - float(args.min_delta)
        if improved:
            best_metric = metric
            patience_left = int(args.patience)
        else:
            patience_left -= 1
        history.append(
            {
                "epoch": source_epoch_index + epoch + 1,
                "continuation_epoch": epoch,
                "step": global_step,
                "train": train_metrics,
                "val": val_metrics,
                "best_metric": best_metric,
                "improved": improved,
                "time_unix": int(time.time()),
            }
        )
        epoch += 1
        shard_cursor = 0
        save(latest_path, next_epoch=epoch, next_cursor=0)
        if improved:
            save(best_path, next_epoch=epoch, next_cursor=0)
        print(
            f"[belief-train] epoch={source_epoch_index + epoch} "
            f"continuation_epoch={epoch - 1} val_obj={metric:.5f} "
            f"hand={val_metrics['hand']:.5f} rem={val_metrics['remainder']:.5f} "
            f"best={best_metric:.5f} patience={patience_left}",
            flush=True,
        )

    final_status = {
        "schema": "poke_bot.privileged_belief_train_status/v2",
        "status": "complete",
        "mode": "full_model_pure_rl" if args.full_model else "heads_only",
        "device": str(device),
        "corpus_digest": manifest["corpus_digest"],
        "source_epoch_index": source_epoch_index,
        "source_epochs_completed": source_epochs_completed,
        "continuation_epochs_completed": epoch,
        "epochs_completed": source_epochs_completed + epoch,
        "last_epoch_index": source_epoch_index + epoch,
        "best_metric": best_metric,
        "latest_checkpoint": str(latest_path),
        "latest_digest": checkpoint.checkpoint_digest(latest_path),
        "best_checkpoint": str(best_path) if best_path.is_file() else None,
        "best_digest": (
            checkpoint.checkpoint_digest(best_path) if best_path.is_file() else None
        ),
        "source_non_head_tensor_digest": source_non_head_digest,
        "current_non_head_tensor_digest": _tensor_digest(model, heads=False),
        "head_tensor_digest": _tensor_digest(model, heads=True),
        "history": history,
        "updated_unix": int(time.time()),
    }
    _atomic_json(status_path, final_status)
    print(json.dumps(final_status, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
