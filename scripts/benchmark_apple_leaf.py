#!/usr/bin/env python3
"""Sweep native Apple CPU and MPS policy-leaf throughput with parity checks.

This is an isolated leaf benchmark, not a game-engine benchmark.  It measures
the exact production state evaluator's sparse packing, forward, and result-copy
path across realistic coalesced batch sizes.  A row is only eligible when its
values and policy probabilities agree with the CPU reference within tolerance.
The report is replaced after every row so dashboard/terminal observers can see
the sweep while it is still running.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from poke_bot import features
from poke_bot.batched_infer import forward_featurized
from poke_bot.train import load_model_from_checkpoint


def _sparse(words: int, nnz_per_word: int, vocab: int, offset: int) -> features.SparseVector:
    value = features.SparseVector()
    for word in range(words):
        value.word_start()
        for lane in range(nnz_per_word):
            value.add((offset + word * 17 + lane * 31) % vocab, 1.0 / (lane + 1))
    return value


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _forward(model: Any, board: Any, options: Any, n_options: int, batch: int):
    return forward_featurized(
        model,
        [board] * batch,
        [options] * batch,
        [n_options] * batch,
        [0] * batch,
        [0] * batch,
    )


def _error(reference: list[tuple[float, list[float]]], actual: list[tuple[float, list[float]]]) -> float:
    if len(reference) != len(actual):
        return float("inf")
    worst = 0.0
    for (rv, rp), (av, ap) in zip(reference, actual):
        worst = max(worst, abs(rv - av))
        if len(rp) != len(ap):
            return float("inf")
        worst = max(worst, *(abs(x - y) for x, y in zip(rp, ap)))
    return worst


def _publish(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--batches", default="1,4,16,32,64,128,256")
    parser.add_argument("--cpu-threads", default="2,4,6,8,10")
    parser.add_argument("--options", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--target-decisions", type=int, default=4096)
    parser.add_argument("--max-iters", type=int, default=300)
    parser.add_argument("--parity-atol", type=float, default=2e-3)
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    batches = [int(value) for value in args.batches.split(",") if value.strip()]
    cpu_threads = [int(value) for value in args.cpu_threads.split(",") if value.strip()]
    if not batches or min(batches) < 1 or not cpu_threads or min(cpu_threads) < 1:
        raise SystemExit("batch sizes and CPU thread counts must be positive")

    cpu_model = load_model_from_checkpoint(checkpoint, device=torch.device("cpu"))
    cpu_model.eval()
    board = _sparse(features.NUM_BOARD_TOKENS, 6, int(cpu_model.encoder_vocab), 7)
    options = _sparse(args.options, 4, int(cpu_model.decoder_vocab), 19)
    with torch.inference_mode():
        reference = _forward(cpu_model, board, options, args.options, max(batches))

    report: dict[str, Any] = {
        "schema": "poke_bot.apple_leaf_sweep/v1",
        "status": "running",
        "started_at": time.time(),
        "checkpoint": str(checkpoint),
        "torch": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "model_params": sum(parameter.numel() for parameter in cpu_model.parameters()),
        "rows": [],
        "best_eligible": None,
    }
    _publish(args.json_out, report)

    variants: list[tuple[str, torch.device, int | None]] = [
        (f"cpu-{threads}t", torch.device("cpu"), threads) for threads in cpu_threads
    ]
    if torch.backends.mps.is_available():
        variants.append(("mps", torch.device("mps"), None))

    try:
        for variant, device, threads in variants:
            if threads is not None:
                torch.set_num_threads(threads)
            model = cpu_model if device.type == "cpu" else load_model_from_checkpoint(checkpoint, device=device)
            model.eval()
            with torch.inference_mode():
                for batch in batches:
                    for _ in range(max(1, args.warmup)):
                        _forward(model, board, options, args.options, batch)
                    _sync(device)
                    iterations = max(
                        8,
                        min(args.max_iters, math.ceil(args.target_decisions / batch)),
                    )
                    started = time.perf_counter()
                    result = []
                    for _ in range(iterations):
                        result = _forward(model, board, options, args.options, batch)
                    _sync(device)
                    elapsed = time.perf_counter() - started
                    error = _error(reference[:batch], result)
                    row = {
                        "variant": variant,
                        "device": device.type,
                        "threads": threads,
                        "batch": batch,
                        "iterations": iterations,
                        "decisions": batch * iterations,
                        "elapsed_s": elapsed,
                        "decisions_per_s": batch * iterations / elapsed,
                        "batch_latency_ms": elapsed * 1000.0 / iterations,
                        "max_abs_error_vs_cpu": error,
                        "parity_passed": error <= args.parity_atol,
                    }
                    report["rows"].append(row)
                    eligible = [item for item in report["rows"] if item["parity_passed"]]
                    report["best_eligible"] = (
                        max(eligible, key=lambda item: item["decisions_per_s"])
                        if eligible
                        else None
                    )
                    report["updated_at"] = time.time()
                    _publish(args.json_out, report)
                    print(json.dumps(row, sort_keys=True), flush=True)
            if device.type == "mps":
                del model
                torch.mps.synchronize()
                torch.mps.empty_cache()
        report["status"] = "complete"
        report["completed_at"] = time.time()
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["failed_at"] = time.time()
        raise
    finally:
        _publish(args.json_out, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
