#!/usr/bin/env python3
"""Benchmark the real CABT policy leaf on one explicitly isolated CUDA GPU.

This is intentionally not described as a CUDA game engine: libcg rule
execution remains native CPU code.  The benchmark measures the production
GPU portion (sparse packing, transfer, policy/value forward, and result copy)
and can emit a PyTorch CPU/CUDA trace for deciding which kernel work is real.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from poke_bot import features
from poke_bot.batched_infer import forward_featurized
from poke_bot.train import load_model_from_checkpoint


def _sparse(words: int, nnz_per_word: int, vocab: int, offset: int) -> features.SparseVector:
    if vocab < 1:
        raise ValueError("vocabulary must be positive")
    sv = features.SparseVector()
    for word in range(words):
        sv.word_start()
        for lane in range(nnz_per_word):
            index = (offset + word * 17 + lane * 31) % vocab
            sv.add(index, 1.0 / float(lane + 1))
    return sv


def _precision(name: str) -> torch.dtype | None:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return None
    raise ValueError(name)


def _run_once(model, board, options, n_options: int, batch: int, dtype):
    return forward_featurized(
        model,
        [board] * batch,
        [options] * batch,
        [n_options] * batch,
        [0] * batch,
        [0] * batch,
        autocast_dtype=dtype,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-device", default="")
    parser.add_argument("--batches", default="1,4,16,32,64,128")
    parser.add_argument("--options", type=int, default=8)
    parser.add_argument("--board-nnz", type=int, default=6)
    parser.add_argument("--option-nnz", type=int, default=4)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--target-decisions", type=int, default=4096)
    parser.add_argument("--max-iters", type=int, default=500)
    parser.add_argument("--trace-out", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("benchmark requires an isolated CUDA device")
    gpu_name = torch.cuda.get_device_name(device)
    if args.expected_device and args.expected_device.lower() not in gpu_name.lower():
        raise SystemExit(
            f"refusing wrong GPU: expected {args.expected_device!r}, got {gpu_name!r}"
        )

    model = load_model_from_checkpoint(args.checkpoint.expanduser().resolve(), device=device)
    model.eval()
    dtype = _precision(args.precision)
    board = _sparse(
        features.NUM_BOARD_TOKENS,
        max(1, args.board_nnz),
        int(model.encoder_vocab),
        7,
    )
    options = _sparse(
        max(1, args.options),
        max(1, args.option_nnz),
        int(model.decoder_vocab),
        19,
    )
    batches = [int(value) for value in args.batches.split(",") if value.strip()]
    if not batches or any(value < 1 for value in batches):
        raise SystemExit("--batches must contain positive integers")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    rows: list[dict] = []
    with torch.inference_mode():
        for batch in batches:
            for _ in range(max(1, args.warmup)):
                _run_once(model, board, options, args.options, batch, dtype)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            iterations = max(
                10,
                min(
                    max(10, args.max_iters),
                    int(math.ceil(max(1, args.target_decisions) / batch)),
                ),
            )
            started = time.perf_counter()
            for _ in range(iterations):
                result = _run_once(model, board, options, args.options, batch, dtype)
                if len(result) != batch:
                    raise RuntimeError("leaf benchmark returned the wrong batch size")
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            rows.append(
                {
                    "batch": batch,
                    "iterations": iterations,
                    "decisions": batch * iterations,
                    "elapsed_s": elapsed,
                    "decisions_per_s": batch * iterations / elapsed,
                    "batch_latency_ms": elapsed * 1000.0 / iterations,
                    "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                }
            )

        if args.trace_out:
            args.trace_out.parent.mkdir(parents=True, exist_ok=True)
            trace_batch = max(batches)
            with torch.profiler.profile(
                activities=(
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ),
                record_shapes=True,
                profile_memory=True,
            ) as profile:
                _run_once(model, board, options, args.options, trace_batch, dtype)
                torch.cuda.synchronize(device)
            profile.export_chrome_trace(str(args.trace_out))

    report = {
        "schema": "poke_bot.cuda_leaf_benchmark/v1",
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "gpu": gpu_name,
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "precision": args.precision,
        "model_params": int(sum(parameter.numel() for parameter in model.parameters())),
        "decision_context": str(model.decision_context),
        "spatial_layers": int(model.cfg.spatial_layers),
        "temporal_layers": int(model.cfg.temporal_layers),
        "option_decoder_layers": int(model.cfg.option_decoder_layers),
        "options_per_decision": int(args.options),
        "rows": rows,
        "trace": str(args.trace_out) if args.trace_out else None,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.json_out.with_suffix(args.json_out.suffix + ".tmp")
        tmp.write_text(encoded, encoding="utf-8")
        tmp.replace(args.json_out)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
