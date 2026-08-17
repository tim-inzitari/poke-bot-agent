#!/usr/bin/env python3
"""Train the exact hidden-state replay corpus entirely from Blackwell VRAM.

Raw JSONL and the persistent feature cache are read exactly once while packing.
After that, the complete sparse corpus, every exact target, shuffle order, AWR
baseline, model, gradients, and optimizer state remain on the CUDA device.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import signal
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from tqdm.auto import tqdm

from poke_bot import archetypes, checkpoint, config
from poke_bot.device_corpus import DeviceResidentBootstrapCorpus
from poke_bot.train import (
    BatchMetrics,
    device_exact_batch_losses,
    device_exact_value_predictions,
    load_model_from_checkpoint,
)
from scripts.train_privileged_belief_shards import (
    HEAD_PREFIXES,
    _atomic_json,
    _expand_aux_head,
    _is_validation,
    _manifest_shards,
    _tensor_digest,
    _weighted_merge,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--latest-checkpoint", type=Path, required=True)
    parser.add_argument("--best-checkpoint", type=Path, required=True)
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=26)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--min-free-gib", type=float, default=12.0)
    parser.add_argument("--min-step-headroom-gib", type=float, default=3.0)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--val-modulus", type=int, default=10)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--aux-weight", type=float, default=0.10)
    parser.add_argument("--hand-weight", type=float, default=0.40)
    parser.add_argument("--remainder-weight", type=float, default=0.30)
    parser.add_argument("--lethal-weight", type=float, default=0.10)
    parser.add_argument("--prize-race-weight", type=float, default=0.10)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--awr-beta", type=float, default=0.5)
    parser.add_argument("--awr-weight-max", type=float, default=20.0)
    parser.add_argument("--entropy-bonus", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260719)
    return parser.parse_args()


def metric_row(metrics: BatchMetrics) -> dict[str, float]:
    return {
        "objective": float(metrics.total_loss),
        "total": float(metrics.total_loss),
        "policy": float(metrics.policy_loss),
        "value": float(metrics.value_loss),
        "hand": float(metrics.opp_hand_loss),
        "remainder": float(metrics.opp_remainder_loss),
        "aux": float(metrics.aux_loss),
        "lethal": float(metrics.lethal_threat_loss),
        "prize_race": float(metrics.prize_race_loss),
        "policy_accuracy": float(metrics.policy_acc),
        "policy_selected_nll": float(metrics.policy_selected_nll),
        "raw_advantage_mean_abs": float(metrics.raw_advantage_mean_abs),
        "awr_weight_mean": float(metrics.awr_weight_mean),
        "awr_weight_p95": float(metrics.awr_weight_p95),
        "awr_weight_clip_frac": float(metrics.awr_weight_clip_frac),
        "awr_effective_sample_fraction": float(
            metrics.awr_effective_sample_fraction
        ),
        "decisions": float(metrics.n_decisions),
        "batches": 1.0,
    }


def fit_batch_size(
    model: nn.Module,
    corpus: DeviceResidentBootstrapCorpus,
    requested: int,
    *,
    amp_dtype: torch.dtype,
    args: argparse.Namespace,
) -> int:
    """Use the largest proven full-loss batch with a real backward pass."""
    size = min(int(requested), int(corpus.train_samples))
    while size >= 256:
        ids = torch.arange(size, device=corpus.device, dtype=torch.long)
        widest = torch.argmax(corpus.n_options[: corpus.train_samples]).long()
        ids[-1] = widest
        baseline = torch.zeros(size, device=corpus.device, dtype=torch.float32)
        model.train()
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(corpus.device)
        try:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                loss, _ = device_exact_batch_losses(
                    model,
                    corpus,
                    ids,
                    baseline_pred=baseline,
                    value_weight=args.value_weight,
                    aux_weight=args.aux_weight,
                    opp_hand_weight=args.hand_weight,
                    opp_remainder_weight=args.remainder_weight,
                    lethal_threat_weight=args.lethal_weight,
                    prize_race_weight=args.prize_race_weight,
                    awr_beta=args.awr_beta,
                    awr_weight_max=args.awr_weight_max,
                    entropy_bonus=args.entropy_bonus,
                )
            loss.backward()
            torch.cuda.synchronize(corpus.device)
            peak = torch.cuda.max_memory_allocated(corpus.device)
            total = torch.cuda.get_device_properties(corpus.device).total_memory
            headroom = total - peak
            if headroom < int(args.min_step_headroom_gib * 2**30):
                raise torch.OutOfMemoryError(
                    f"fit passed but peak headroom was only {headroom / 2**30:.2f} GiB"
                )
        except (RuntimeError, torch.OutOfMemoryError) as exc:
            if not isinstance(exc, torch.OutOfMemoryError) and not config.is_cuda_oom(exc):
                raise
            model.zero_grad(set_to_none=True)
            del ids, baseline
            torch.cuda.empty_cache()
            reduced = size // 2
            print(
                f"[resident-fit] batch={size} rejected ({exc}); trying {reduced}",
                flush=True,
            )
            size = reduced
            continue
        model.zero_grad(set_to_none=True)
        del ids, baseline
        torch.cuda.empty_cache()
        print(
            f"[resident-fit] batch={size} passed peak={peak / 2**30:.2f} GiB "
            f"headroom={headroom / 2**30:.2f} GiB",
            flush=True,
        )
        return size
    raise MemoryError("no safe exact-resident training batch fits on the GPU")


@torch.no_grad()
def build_baseline(
    model: nn.Module,
    corpus: DeviceResidentBootstrapCorpus,
    *,
    batch_size: int,
    amp_dtype: torch.dtype,
    epoch_index: int,
    status_cb,
) -> torch.Tensor:
    """Freeze V(s) for AWR once per epoch, entirely in device memory."""
    model.eval()
    baseline = torch.empty(
        corpus.train_samples, device=corpus.device, dtype=torch.float32
    )
    batches = corpus.batches(
        train=True, batch_size=batch_size, shuffle=False, seed=0, epoch=0
    )
    started = time.monotonic()
    bar = tqdm(batches, desc=f"resident baseline ep{epoch_index}", unit="batch")
    for cursor, ids in enumerate(bar, 1):
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            prediction = device_exact_value_predictions(model, corpus, ids)
        baseline.index_copy_(0, ids, prediction.float())
        if cursor == 1 or cursor % 10 == 0 or cursor == len(batches):
            status_cb(
                phase="baseline",
                current=cursor,
                total=len(batches),
                elapsed=time.monotonic() - started,
            )
    return baseline


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.val_modulus < 2 or args.batch_size < 1:
        raise ValueError("invalid epoch, split, or batch configuration")
    manifest_path = args.manifest.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    init_path = args.init_checkpoint.expanduser().resolve()
    latest_path = args.latest_checkpoint.expanduser().resolve()
    best_path = args.best_checkpoint.expanduser().resolve()
    status_path = args.status_json.expanduser().resolve()
    manifest, shards = _manifest_shards(manifest_path)
    train_shards = [p for p in shards if not _is_validation(p, args.val_modulus)]
    val_shards = [p for p in shards if _is_validation(p, args.val_modulus)]
    if not train_shards or not val_shards:
        raise RuntimeError("deterministic split produced an empty partition")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("exact resident training requires CUDA")
    torch.cuda.set_device(device)
    config.apply_runtime_perf()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    resume = latest_path.is_file()
    source = latest_path if resume else init_path
    model = load_model_from_checkpoint(source, device=device)
    expanded = _expand_aux_head(model)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.train()

    init_saved = checkpoint.load_checkpoint(init_path, map_location="cpu")
    source_epoch = int(init_saved.get("epoch", -1))
    source_digest = checkpoint.checkpoint_digest(init_path)
    source_non_head = _tensor_digest(model, heads=False)
    expected_ids = list(archetypes.archetype_ids())

    epoch = 0
    batch_cursor = 0
    global_step = 0
    best_metric = math.inf
    patience_left = int(args.patience)
    history: list[dict[str, Any]] = []
    saved_baseline: torch.Tensor | None = None
    saved_rows: list[dict[str, float]] = []
    if resume:
        saved = checkpoint.load_checkpoint(latest_path, map_location=device)
        extra = dict(saved.get("extra") or {}).get("resident_exact_training") or {}
        if extra.get("corpus_digest") != manifest.get("corpus_digest"):
            raise RuntimeError("resident resume checkpoint belongs to another corpus")
        if list(extra.get("archetype_ids") or []) != expected_ids:
            raise RuntimeError("resident resume archetype mapping changed")
        epoch = int(extra.get("epoch", 0))
        batch_cursor = int(extra.get("batch_cursor", 0))
        global_step = int(saved.get("step", 0))
        best_metric = float(saved.get("best_metric", math.inf))
        patience_left = int(extra.get("patience_left", args.patience))
        history = list(extra.get("history") or [])
        saved_rows = list(extra.get("epoch_rows") or [])
        value = saved.get("resident_awr_baseline")
        if isinstance(value, torch.Tensor):
            saved_baseline = value.to(device=device, dtype=torch.float32)
    elif expanded:
        print(f"[resident] expanded aux head to {len(expected_ids) + 1} classes")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    if resume:
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        if saved.get("scaler_state_dict"):
            scaler.load_state_dict(saved["scaler_state_dict"])

    corpus: DeviceResidentBootstrapCorpus | None = None
    batch_size = 0
    stop_requested = False

    def on_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, on_stop)
    signal.signal(signal.SIGINT, on_stop)

    def write_status(
        *,
        phase: str,
        current: int = 0,
        total: int = 0,
        elapsed: float = 0.0,
        metrics: dict[str, float] | None = None,
        state: str = "training",
    ) -> None:
        resident_bytes = corpus.input_bytes if corpus is not None else 0
        rate = current / elapsed if elapsed > 0 and current > 0 else None
        _atomic_json(
            status_path,
            {
                "schema": "poke_bot.privileged_belief_resident_status/v1",
                "status": state,
                "mode": "full_model_pure_rl_exact_resident",
                "phase": phase,
                "device": str(device),
                "gpu_name": torch.cuda.get_device_name(device),
                "corpus_digest": manifest["corpus_digest"],
                "all_training_tensors_device_resident": corpus is not None,
                "resident_bytes": resident_bytes,
                "resident_gib": resident_bytes / 2**30,
                "train_samples": corpus.train_samples if corpus else None,
                "val_samples": corpus.val_samples if corpus else None,
                "decisions": corpus.decisions if corpus else None,
                "batch_size": batch_size or None,
                "epoch": epoch,
                "total_epoch_index": source_epoch + epoch + 1,
                "source_epoch_index": source_epoch,
                "epochs_target": args.epochs,
                "current": current,
                "total": total,
                "percent": 100.0 * current / total if total else None,
                "batch_per_second": rate,
                "eta_seconds": ((total - current) / rate) if rate else None,
                "global_step": global_step,
                "metrics": metrics or {},
                "updated_unix": int(time.time()),
            },
        )

    write_status(phase="packing")
    corpus = DeviceResidentBootstrapCorpus.from_exact_shards(
        train_shards,
        val_shards,
        cache_dir=cache_dir,
        max_context=int(model.cfg.max_context),
        card_vocab=int(model.belief_card_vocab),
        device=device,
        min_free_gib=float(args.min_free_gib),
    )
    if corpus.device != device or not corpus.has_exact_targets:
        raise RuntimeError("full exact corpus did not become device resident")
    batch_size = fit_batch_size(
        model, corpus, args.batch_size, amp_dtype=amp_dtype, args=args
    )
    free, total_vram = torch.cuda.mem_get_info(device)
    print(
        f"[resident] ALL training tensors on {torch.cuda.get_device_name(device)} "
        f"corpus={corpus.input_bytes / 2**30:.2f} GiB "
        f"free={free / 2**30:.2f}/{total_vram / 2**30:.2f} GiB "
        f"batch={batch_size}",
        flush=True,
    )

    current_baseline: torch.Tensor | None = saved_baseline
    current_rows = saved_rows

    def build_payload(next_epoch: int, next_cursor: int) -> dict[str, Any]:
        extra = {
            "pure_rl": True,
            "resident_exact_training": {
                "schema": "poke_bot.privileged_belief_resident_train/v1",
                "corpus_manifest": str(manifest_path),
                "corpus_digest": manifest["corpus_digest"],
                "source_checkpoint": str(init_path),
                "source_checkpoint_digest": source_digest,
                "source_epoch_index": source_epoch,
                "full_model": True,
                "pure_rl": True,
                "all_training_tensors_device_resident": True,
                "resident_bytes": corpus.input_bytes,
                "batch_size": batch_size,
                "epoch": next_epoch,
                "batch_cursor": next_cursor,
                "patience_left": patience_left,
                "archetype_ids": expected_ids,
                "history": history,
                "epoch_rows": current_rows,
            },
        }
        payload = checkpoint.build_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            step=global_step,
            epoch=source_epoch + next_epoch,
            best_metric=best_metric,
            model_config=model.cfg,
            model_id="state_core_privileged_belief_20k_resident_v1",
            archetype_id="top-ladder-core",
            extra=extra,
        )
        if next_cursor > 0 and current_baseline is not None:
            payload["resident_awr_baseline"] = current_baseline
        return payload

    def save_latest(next_epoch: int, next_cursor: int) -> None:
        checkpoint.atomic_torch_save(
            build_payload(next_epoch, next_cursor), latest_path
        )

    while epoch < args.epochs and patience_left > 0:
        if current_baseline is None:
            current_baseline = build_baseline(
                model,
                corpus,
                batch_size=batch_size,
                amp_dtype=amp_dtype,
                epoch_index=source_epoch + epoch + 1,
                status_cb=write_status,
            )
        batches = corpus.batches(
            train=True,
            batch_size=batch_size,
            shuffle=True,
            seed=args.seed,
            epoch=epoch,
        )
        started = time.monotonic()
        bar = tqdm(
            range(batch_cursor, len(batches)),
            total=len(batches),
            initial=batch_cursor,
            desc=f"resident train ep{source_epoch + epoch + 1}",
            unit="batch",
        )
        model.train()
        for position in bar:
            ids = batches[position]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                loss, metrics = device_exact_batch_losses(
                    model,
                    corpus,
                    ids,
                    baseline_pred=current_baseline.index_select(0, ids),
                    value_weight=args.value_weight,
                    aux_weight=args.aux_weight,
                    opp_hand_weight=args.hand_weight,
                    opp_remainder_weight=args.remainder_weight,
                    lethal_threat_weight=args.lethal_weight,
                    prize_race_weight=args.prize_race_weight,
                    awr_beta=args.awr_beta,
                    awr_weight_max=args.awr_weight_max,
                    entropy_bonus=args.entropy_bonus,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            row = metric_row(metrics)
            current_rows.append(row)
            batch_cursor = position + 1
            merged = _weighted_merge(current_rows)
            elapsed = time.monotonic() - started
            bar.set_postfix(
                acc=f"{100 * row['policy_accuracy']:.1f}%",
                aux=f"{row['aux']:.3f}",
                hand=f"{row['hand']:.3f}",
                loss=f"{row['total']:.3f}",
            )
            write_status(
                phase="training",
                current=batch_cursor,
                total=len(batches),
                elapsed=elapsed,
                metrics=merged,
            )
            if batch_cursor % int(args.checkpoint_every) == 0 or stop_requested:
                save_latest(epoch, batch_cursor)
            if stop_requested:
                write_status(
                    phase="stopped",
                    current=batch_cursor,
                    total=len(batches),
                    elapsed=elapsed,
                    metrics=merged,
                    state="stopped",
                )
                return 0

        train_metrics = _weighted_merge(current_rows)
        model.eval()
        val_rows: list[dict[str, float]] = []
        val_batches = corpus.batches(
            train=False,
            batch_size=batch_size,
            shuffle=False,
            seed=args.seed,
            epoch=epoch,
        )
        val_started = time.monotonic()
        with torch.no_grad():
            for position, ids in enumerate(
                tqdm(val_batches, desc=f"resident val ep{source_epoch + epoch + 1}", unit="batch"),
                1,
            ):
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    _, metrics = device_exact_batch_losses(
                        model,
                        corpus,
                        ids,
                        value_weight=args.value_weight,
                        aux_weight=args.aux_weight,
                        opp_hand_weight=args.hand_weight,
                        opp_remainder_weight=args.remainder_weight,
                        lethal_threat_weight=args.lethal_weight,
                        prize_race_weight=args.prize_race_weight,
                        awr_beta=args.awr_beta,
                        awr_weight_max=args.awr_weight_max,
                        entropy_bonus=args.entropy_bonus,
                    )
                val_rows.append(metric_row(metrics))
                write_status(
                    phase="validation",
                    current=position,
                    total=len(val_batches),
                    elapsed=time.monotonic() - val_started,
                    metrics=_weighted_merge(val_rows),
                )
        val_metrics = _weighted_merge(val_rows)
        objective = float(val_metrics["objective"])
        improved = objective < best_metric - float(args.min_delta)
        if improved:
            best_metric = objective
            patience_left = int(args.patience)
        else:
            patience_left -= 1
        history.append(
            {
                "epoch": source_epoch + epoch + 1,
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
        batch_cursor = 0
        current_rows = []
        current_baseline = None
        save_latest(epoch, 0)
        if improved:
            checkpoint.atomic_torch_save(build_payload(epoch, 0), best_path)
        print(
            f"[resident] epoch={source_epoch + epoch} val={objective:.5f} "
            f"acc={100 * val_metrics['policy_accuracy']:.2f}% "
            f"hand={val_metrics['hand']:.5f} aux={val_metrics['aux']:.5f} "
            f"best={best_metric:.5f} patience={patience_left}",
            flush=True,
        )

    write_status(phase="complete", state="complete", metrics=history[-1]["val"] if history else {})
    print(json.dumps({"status": "complete", "epochs": epoch, "best": best_metric}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
