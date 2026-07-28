#!/usr/bin/env python3
"""Profile Bert MPS host packing, dispatch, readback, sync, and cache costs."""

from __future__ import annotations

import json
import math
import time
from typing import Any, Callable

import torch

from run_apple_optimization import CHECKPOINT, STATUS, _publish
from benchmark_apple_leaf import _error, _forward, _sparse, _sync
from poke_bot import features
from poke_bot.batched_infer import _policy_probs
from poke_bot.model import pack_sparse_batch
from poke_bot.train import load_model_from_checkpoint


def _memory() -> dict[str, int | None]:
    values: dict[str, int | None] = {}
    for name in ("current_allocated_memory", "driver_allocated_memory"):
        fn = getattr(torch.mps, name, None)
        try:
            values[name] = int(fn()) if fn is not None else None
        except RuntimeError:
            values[name] = None
    return values


def _measure(
    fn: Callable[[], Any], *, iterations: int, sync: bool = True
) -> tuple[float, Any]:
    for _ in range(3):
        value = fn()
    if sync:
        torch.mps.synchronize()
    started = time.perf_counter()
    value = None
    for _ in range(iterations):
        value = fn()
    if sync:
        torch.mps.synchronize()
    return time.perf_counter() - started, value


def main() -> int:
    report = json.loads(STATUS.read_text()) if STATUS.is_file() else {}
    report.update(
        {
            "status": "running",
            "stage": "mps-profile:loading",
            "active_variant": "mps-transfer-sync-profile",
            "profile_started_at": time.time(),
        }
    )
    report["mps_pipeline_profile"] = {
        "status": "running",
        "rows": [],
        "notes": {
            "end_to_end": "sparse host inputs through per-row CPU results",
            "dispatch_compute": "sparse packing + MPS model; one sync after loop",
            "resident_compute": "prepacked MPS tensors + model; one sync after loop",
            "bulk_readback": "one logits and one value tensor copy to CPU",
            "per_row_readback": "current result path: CPU copy/item per batch row",
        },
    }
    _publish(report)

    cpu_model = load_model_from_checkpoint(CHECKPOINT, device=torch.device("cpu"))
    cpu_model.eval()
    mps_model = load_model_from_checkpoint(CHECKPOINT, device=torch.device("mps"))
    mps_model.eval()
    board = _sparse(features.NUM_BOARD_TOKENS, 6, int(cpu_model.encoder_vocab), 7)
    options_count = 8
    options = _sparse(options_count, 4, int(cpu_model.decoder_vocab), 19)
    with torch.inference_mode():
        reference = _forward(cpu_model, board, options, options_count, 32)

    for batch in (1, 4, 16, 32):
        report["stage"] = f"mps-profile:batch-{batch}"
        _publish(report)
        iterations = max(8, min(128, math.ceil(512 / batch)))
        boards = [board] * batch
        opts = [options] * batch
        counts = [options_count] * batch
        seats = [0] * batch

        with torch.inference_mode():
            cpu_started = time.perf_counter()
            cpu_result = None
            for _ in range(iterations):
                cpu_result = _forward(
                    cpu_model, board, options, options_count, batch
                )
            cpu_elapsed = time.perf_counter() - cpu_started

            e2e_elapsed, mps_result = _measure(
                lambda: _forward(
                    mps_model, board, options, options_count, batch
                ),
                iterations=iterations,
            )

            dispatch_elapsed, raw = _measure(
                lambda: mps_model.forward(
                    boards,
                    opts,
                    kv_cache=None,
                    append_cache=False,
                    n_options=counts,
                ),
                iterations=iterations,
            )

            board_packed = pack_sparse_batch(
                boards, features.NUM_BOARD_TOKENS, torch.device("mps")
            )
            option_packed = pack_sparse_batch(
                opts, options_count, torch.device("mps")
            )

            def resident_forward():
                spatial = mps_model.encode_board_packed(
                    board_packed, batch_size=batch
                )
                cls = mps_model.history_tokens(spatial, [None] * batch).unsqueeze(1)
                state, _cache = mps_model.temporal_encode(cls, append=False)
                logits = mps_model.decode_options_packed(
                    option_packed,
                    spatial,
                    state,
                    n_options=counts,
                    batch_size=batch,
                )
                value = torch.tanh(mps_model.value_head(state)).squeeze(-1)
                return logits, value

            resident_elapsed, resident = _measure(
                resident_forward, iterations=iterations
            )

            def pack_transfer():
                return (
                    pack_sparse_batch(
                        boards, features.NUM_BOARD_TOKENS, torch.device("mps")
                    ),
                    pack_sparse_batch(opts, options_count, torch.device("mps")),
                )

            pack_elapsed, _packed = _measure(
                pack_transfer, iterations=iterations
            )

            logits, values = resident

            def bulk_readback():
                return (
                    logits.detach().float().cpu(),
                    values.detach().float().cpu(),
                )

            bulk_elapsed, _bulk = _measure(
                bulk_readback, iterations=iterations, sync=False
            )

            def per_row_readback():
                return [
                    (float(values[i].item()), _policy_probs(logits[i, :options_count]))
                    for i in range(batch)
                ]

            row_elapsed, _rows = _measure(
                per_row_readback, iterations=iterations, sync=False
            )

            sync_started = time.perf_counter()
            for _ in range(iterations):
                torch.mps.synchronize()
            sync_elapsed = time.perf_counter() - sync_started

            cache_started = time.perf_counter()
            for _ in range(iterations):
                torch.mps.synchronize()
                torch.mps.empty_cache()
            cache_elapsed = time.perf_counter() - cache_started

        row = {
            "batch": batch,
            "iterations": iterations,
            "decisions": batch * iterations,
            "cpu_end_to_end_decisions_per_s": batch * iterations / cpu_elapsed,
            "mps_end_to_end_decisions_per_s": batch * iterations / e2e_elapsed,
            "mps_dispatch_compute_decisions_per_s": batch * iterations / dispatch_elapsed,
            "mps_resident_compute_decisions_per_s": batch * iterations / resident_elapsed,
            "pack_transfer_ms_per_batch": 1000.0 * pack_elapsed / iterations,
            "bulk_readback_ms_per_batch": 1000.0 * bulk_elapsed / iterations,
            "per_row_readback_ms_per_batch": 1000.0 * row_elapsed / iterations,
            "empty_sync_ms": 1000.0 * sync_elapsed / iterations,
            "sync_empty_cache_ms": 1000.0 * cache_elapsed / iterations,
            "max_abs_error_vs_cpu": _error(reference[:batch], mps_result),
            "memory": _memory(),
        }
        report["mps_pipeline_profile"]["rows"].append(row)
        report["mps_pipeline_profile"]["updated_at"] = time.time()
        _publish(report)
        print(json.dumps(row, sort_keys=True), flush=True)

    report["mps_pipeline_profile"]["status"] = "complete"
    report["mps_pipeline_profile"]["completed_at"] = time.time()
    report["status"] = "complete"
    report["stage"] = "mps-profile:complete"
    report["active_variant"] = None
    report["profile_completed_at"] = time.time()
    _publish(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
