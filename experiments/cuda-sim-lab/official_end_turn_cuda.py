#!/usr/bin/env python3
"""Device-resident CUDA slice for clean official End/draw/deckout transitions."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl


INPUT_KEYS = (
    "turn", "your_index", "first_player", "result",
    "deck_count_0", "deck_count_1", "hand_count_0", "hand_count_1",
)
OUTPUT_KEYS = (
    "turn", "your_index", "first_player", "result",
    "deck_count_0", "deck_count_1", "hand_count_0", "hand_count_1",
    "select_type", "select_context", "select_min", "select_max", "terminal",
)


@triton.jit
def clean_end_turn_kernel(
    turn,
    your_index,
    first_player,
    result,
    deck0,
    deck1,
    hand0,
    hand1,
    out_turn,
    out_your_index,
    out_first_player,
    out_result,
    out_deck0,
    out_deck1,
    out_hand0,
    out_hand1,
    out_select_type,
    out_select_context,
    out_select_min,
    out_select_max,
    out_terminal,
    n: tl.constexpr,
    block: tl.constexpr,
):
    lane = tl.program_id(0) * block + tl.arange(0, block)
    mask = lane < n
    t = tl.load(turn + lane, mask=mask, other=0).to(tl.int32)
    current = tl.load(your_index + lane, mask=mask, other=0).to(tl.int32)
    first = tl.load(first_player + lane, mask=mask, other=0).to(tl.int32)
    d0 = tl.load(deck0 + lane, mask=mask, other=0).to(tl.int32)
    d1 = tl.load(deck1 + lane, mask=mask, other=0).to(tl.int32)
    h0 = tl.load(hand0 + lane, mask=mask, other=0).to(tl.int32)
    h1 = tl.load(hand1 + lane, mask=mask, other=0).to(tl.int32)

    next_player = 1 - current
    next_deck = tl.where(next_player == 0, d0, d1)
    deckout = next_deck == 0
    draw0 = (next_player == 0) & (~deckout)
    draw1 = (next_player == 1) & (~deckout)

    tl.store(out_turn + lane, t + 1, mask=mask)
    # Official terminal observations retain the selecting player; otherwise
    # the observer/control player advances to the player who drew.
    tl.store(out_your_index + lane, tl.where(deckout, current, next_player), mask=mask)
    tl.store(out_first_player + lane, first, mask=mask)
    tl.store(out_result + lane, tl.where(deckout, current, -1), mask=mask)
    tl.store(out_deck0 + lane, d0 - draw0.to(tl.int32), mask=mask)
    tl.store(out_deck1 + lane, d1 - draw1.to(tl.int32), mask=mask)
    tl.store(out_hand0 + lane, h0 + draw0.to(tl.int32), mask=mask)
    tl.store(out_hand1 + lane, h1 + draw1.to(tl.int32), mask=mask)
    # Clean no-effect turns return to the main selection gate; terminal rows
    # retain these scalar control fields but expose zero options.
    tl.store(out_select_type + lane, 0, mask=mask)
    tl.store(out_select_context + lane, 0, mask=mask)
    tl.store(out_select_min + lane, 1, mask=mask)
    tl.store(out_select_max + lane, 1, mask=mask)
    tl.store(out_terminal + lane, deckout.to(tl.int32), mask=mask)


def publish(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def pack(
    fixtures: list[dict[str, Any]], device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    inputs = {
        key: torch.tensor(
            [int(row["before"][key]) for row in fixtures],
            dtype=torch.int32,
            device=device,
        )
        for key in INPUT_KEYS
    }
    targets: dict[str, torch.Tensor] = {}
    for key in OUTPUT_KEYS:
        if key == "terminal":
            values = [int(row["after"]["result"] != -1) for row in fixtures]
        else:
            values = [int(row["after"][key]) for row in fixtures]
        targets[key] = torch.tensor(values, dtype=torch.int32, device=device)
    return inputs, targets


def launch(
    inputs: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]
) -> None:
    n = inputs["turn"].numel()
    clean_end_turn_kernel[(triton.cdiv(n, 256),)](
        *[inputs[key] for key in INPUT_KEYS],
        *[outputs[key] for key in OUTPUT_KEYS],
        n=n, block=256, num_warps=4,
    )


def measure_transfers(
    inputs: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    cpu_inputs = {key: value.cpu() for key, value in inputs.items()}
    input_bytes = sum(value.numel() * value.element_size() for value in cpu_inputs.values())
    output_bytes = sum(value.numel() * value.element_size() for value in outputs.values())
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    pageable_inputs = {key: value.to(device) for key, value in cpu_inputs.items()}
    torch.cuda.synchronize(device)
    pageable_h2d_s = time.perf_counter() - started
    started = time.perf_counter()
    pageable_outputs = {key: value.cpu() for key, value in outputs.items()}
    torch.cuda.synchronize(device)
    pageable_d2h_s = time.perf_counter() - started
    report: dict[str, Any] = {
        "input_h2d_bytes": input_bytes,
        "output_d2h_bytes": output_bytes,
        "pageable_h2d_s": pageable_h2d_s,
        "pageable_d2h_s": pageable_d2h_s,
        "pageable_h2d_gib_per_s": input_bytes / max(pageable_h2d_s, 1e-12) / (1024**3),
        "pageable_d2h_gib_per_s": output_bytes / max(pageable_d2h_s, 1e-12) / (1024**3),
    }
    try:
        pinned_inputs = {key: value.pin_memory() for key, value in cpu_inputs.items()}
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        pinned_device = {
            key: value.to(device, non_blocking=True)
            for key, value in pinned_inputs.items()
        }
        torch.cuda.synchronize(device)
        pinned_h2d_s = time.perf_counter() - started
        pinned_outputs = {
            key: torch.empty_like(value, device="cpu", pin_memory=True)
            for key, value in outputs.items()
        }
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for key in OUTPUT_KEYS:
            pinned_outputs[key].copy_(outputs[key], non_blocking=True)
        torch.cuda.synchronize(device)
        pinned_d2h_s = time.perf_counter() - started
        report.update({
            "pinned_async_h2d_s": pinned_h2d_s,
            "pinned_async_d2h_s": pinned_d2h_s,
            "pinned_async_h2d_gib_per_s": input_bytes / max(pinned_h2d_s, 1e-12) / (1024**3),
            "pinned_async_d2h_gib_per_s": output_bytes / max(pinned_d2h_s, 1e-12) / (1024**3),
        })
        del pinned_inputs, pinned_device, pinned_outputs
    except RuntimeError as exc:
        report["pinned_async_error"] = f"{type(exc).__name__}: {exc}"
    del cpu_inputs, pageable_inputs, pageable_outputs
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--benchmark-lanes", type=int, default=262144)
    parser.add_argument("--benchmark-iters", type=int, default=100)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    device = torch.device("cuda:0")
    gpu = torch.cuda.get_device_name(device)
    if "3080 Ti" not in gpu:
        raise SystemExit(f"refusing non-3080 GPU: {gpu}")
    fixture_report = json.loads(args.fixtures.read_text())
    fixtures = fixture_report["fixtures"]
    report: dict[str, Any] = {
        "schema": "poke_bot.official_clean_end_turn_cuda/v1",
        "status": "running",
        "started_at": time.time(),
        "gpu": gpu,
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "official_lib_sha256": fixture_report["official_lib_sha256"],
        "scope": fixture_report["scope"],
        "official_fixture_count": len(fixtures),
        "official_games": fixture_report["games"],
        "official_terminal_transitions": fixture_report["terminal_transition_count"],
        "implemented_fields": list(OUTPUT_KEYS),
        "excluded_state": [
            "card identities and option construction",
            "attacks/card effects and non-clean end-of-turn status effects",
            "active/bench/prize/discard mutation",
        ],
        "full_engine_transition_coverage": False,
        "production_eligible": False,
    }
    publish(args.json_out, report)

    setup_started = time.perf_counter()
    inputs, targets = pack(fixtures, device)
    outputs = {
        key: torch.empty(len(fixtures), device=device, dtype=torch.int32)
        for key in OUTPUT_KEYS
    }
    torch.cuda.synchronize(device)
    setup_h2d_s = time.perf_counter() - setup_started
    launch(inputs, outputs)
    torch.cuda.synchronize(device)
    mismatches = {
        key: int((outputs[key] != targets[key]).sum().item())
        for key in OUTPUT_KEYS
    }
    report["parity"] = {
        "exact": all(value == 0 for value in mismatches.values()),
        "field_mismatches": mismatches,
        "fixture_setup_h2d_s": setup_h2d_s,
        "parity_result_d2h_bytes": len(OUTPUT_KEYS) * 8,
    }
    if not report["parity"]["exact"]:
        report["status"] = "failed"
        publish(args.json_out, report)
        raise RuntimeError(f"clean End slice parity failed: {mismatches}")

    lanes = max(len(fixtures), int(args.benchmark_lanes))
    repeats = (lanes + len(fixtures) - 1) // len(fixtures)
    bench_inputs = {
        key: value.repeat(repeats)[:lanes].contiguous()
        for key, value in inputs.items()
    }
    bench_outputs = {
        key: torch.empty(lanes, device=device, dtype=torch.int32)
        for key in OUTPUT_KEYS
    }
    for _ in range(10):
        launch(bench_inputs, bench_outputs)
    torch.cuda.synchronize(device)
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_started = time.perf_counter()
    begin.record()
    for _ in range(max(1, args.benchmark_iters)):
        launch(bench_inputs, bench_outputs)
    end.record()
    torch.cuda.synchronize(device)
    wall_s = time.perf_counter() - wall_started
    kernel_s = begin.elapsed_time(end) / 1000.0
    transitions = lanes * max(1, args.benchmark_iters)
    report["throughput"] = {
        "lanes": lanes,
        "iterations": args.benchmark_iters,
        "transitions": transitions,
        "kernel_s": kernel_s,
        "wall_s": wall_s,
        "kernel_transitions_per_s": transitions / max(kernel_s, 1e-12),
        "whole_loop_transitions_per_s": transitions / max(wall_s, 1e-12),
        "whole_loop_vs_kernel_efficiency": kernel_s / max(wall_s, 1e-12),
        "measured_bulk_h2d_bytes": 0,
        "measured_bulk_d2h_bytes": 0,
        "device_resident_bytes": sum(
            value.numel() * value.element_size()
            for value in (*bench_inputs.values(), *bench_outputs.values())
        ),
    }
    report["transfer_probe"] = measure_transfers(bench_inputs, bench_outputs, device)
    probe = report["transfer_probe"]
    kernel_per_iteration_s = kernel_s / max(1, args.benchmark_iters)
    pageable_roundtrip = float(probe["pageable_h2d_s"]) + float(probe["pageable_d2h_s"])
    report["transfer_probe"]["pageable_roundtrip_vs_one_kernel_step"] = pageable_roundtrip / max(kernel_per_iteration_s, 1e-12)
    if "pinned_async_h2d_s" in probe:
        pinned_roundtrip = float(probe["pinned_async_h2d_s"]) + float(probe["pinned_async_d2h_s"])
        report["transfer_probe"]["pinned_roundtrip_vs_one_kernel_step"] = pinned_roundtrip / max(kernel_per_iteration_s, 1e-12)
    report["status"] = "complete"
    report["completed_at"] = time.time()
    publish(args.json_out, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
