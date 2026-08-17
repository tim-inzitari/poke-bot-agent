#!/usr/bin/env python3
"""CUDA parity/throughput gate for an official libcg rule slice.

Implemented official slice:
* selection error precedence from State::checkPlayerSelect;
* terminal predicate (`current.result != -1`);
* rejected-step public selection/terminal state remains unchanged.

Accepted card-effect transitions are deliberately out of scope and reported as
such.  Ground truth comes from the unmodified official libcg binary fixtures.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl


MAX_ACTIONS = 64


@triton.jit
def official_select_gate_kernel(
    select_min,
    select_max,
    option_count,
    result,
    action_length,
    actions,
    out_error,
    out_terminal,
    out_rejected_unchanged,
    n: tl.constexpr,
    block_actions: tl.constexpr,
):
    lane = tl.program_id(0)
    if lane < n:
        length = tl.load(action_length + lane).to(tl.int32)
        positions = tl.arange(0, block_actions)
        action_mask = positions < length
        row = actions + lane * block_actions
        values = tl.load(row + positions, mask=action_mask, other=0).to(tl.int32)

        # Exact official precedence: duplicate (only when count <= 60), then
        # range, then min/max cardinality.  The loop is bounded by the ABI
        # slice's explicit 64-choice capacity.
        duplicate = False
        for j in tl.static_range(0, block_actions):
            selected_j = tl.load(row + j, mask=j < length, other=-2147483648).to(tl.int32)
            occurrences = tl.sum(((values == selected_j) & action_mask).to(tl.int32), axis=0)
            duplicate = duplicate | ((j < length) & (occurrences > 1))
        duplicate = duplicate & (length <= 60)

        options = tl.load(option_count + lane).to(tl.int32)
        out_of_range = tl.sum(
            (action_mask & ((values < 0) | (values >= options))).to(tl.int32), axis=0
        ) > 0
        minimum = tl.load(select_min + lane).to(tl.int32)
        maximum = tl.load(select_max + lane).to(tl.int32)
        cardinality = (length < minimum) | (length > maximum)
        error = tl.where(
            duplicate,
            6,
            tl.where(out_of_range, 5, tl.where(cardinality, 4, 0)),
        )
        terminal = (tl.load(result + lane).to(tl.int32) != -1).to(tl.int32)
        tl.store(out_error + lane, error)
        tl.store(out_terminal + lane, terminal)
        # The implemented step slice rejects invalid selections before the
        # authoritative game transition. Public select/result fields persist.
        tl.store(out_rejected_unchanged + lane, (error != 0).to(tl.int32))


@triton.jit
def official_first_player_step_kernel(
    turn,
    your_index,
    result,
    selected_option_type,
    out_turn,
    out_first_player,
    out_result,
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
    current_turn = tl.load(turn + lane, mask=mask, other=0).to(tl.int32)
    current_player = tl.load(your_index + lane, mask=mask, other=0).to(tl.int32)
    current_result = tl.load(result + lane, mask=mask, other=-1).to(tl.int32)
    option_type = tl.load(selected_option_type + lane, mask=mask, other=1).to(tl.int32)
    # Official SelectOptionType: Yes=1, No=2.  The accepted IsFirst step sets
    # first player, then enters the one-card setup selection control state.
    next_first = tl.where(option_type == 1, current_player, 1 - current_player)
    tl.store(out_turn + lane, current_turn, mask=mask)
    tl.store(out_first_player + lane, next_first, mask=mask)
    tl.store(out_result + lane, current_result, mask=mask)
    tl.store(out_select_type + lane, 1, mask=mask)
    tl.store(out_select_context + lane, 1, mask=mask)
    tl.store(out_select_min + lane, 1, mask=mask)
    tl.store(out_select_max + lane, 1, mask=mask)
    tl.store(out_terminal + lane, (current_result != -1).to(tl.int32), mask=mask)


def publish(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def pack(fixtures: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    n = len(fixtures)
    actions = torch.full((n, MAX_ACTIONS), -1, dtype=torch.int32)
    lengths = torch.empty(n, dtype=torch.int32)
    select_min = torch.empty(n, dtype=torch.int32)
    select_max = torch.empty(n, dtype=torch.int32)
    option_count = torch.empty(n, dtype=torch.int32)
    result = torch.empty(n, dtype=torch.int32)
    target_error = torch.empty(n, dtype=torch.int32)
    compare_error = torch.empty(n, dtype=torch.bool)
    target_terminal = torch.empty(n, dtype=torch.int32)
    target_rejected = torch.empty(n, dtype=torch.int32)
    compare_rejected = torch.empty(n, dtype=torch.bool)
    for i, fixture in enumerate(fixtures):
        state = fixture["state"]
        action = [int(value) for value in fixture["action"]]
        if len(action) > MAX_ACTIONS:
            raise ValueError(f"fixture {i} action exceeds {MAX_ACTIONS}")
        if action:
            actions[i, : len(action)] = torch.tensor(action, dtype=torch.int32)
        lengths[i] = len(action)
        select_min[i] = int(state["select_min"])
        select_max[i] = int(state["select_max"])
        option_count[i] = int(state["option_count"])
        result[i] = int(state["result"])
        target_error[i] = int(fixture["official_error"])
        compare_error[i] = fixture["kind"] != "terminal"
        target_terminal[i] = int(bool(fixture["official_terminal"]))
        rejected = fixture.get("rejected_public_slice_unchanged")
        target_rejected[i] = int(bool(rejected)) if rejected is not None else 0
        compare_rejected[i] = rejected is not None
    return {
        "select_min": select_min.to(device),
        "select_max": select_max.to(device),
        "option_count": option_count.to(device),
        "result": result.to(device),
        "action_length": lengths.to(device),
        "actions": actions.to(device),
        "target_error": target_error.to(device),
        "compare_error": compare_error.to(device),
        "target_terminal": target_terminal.to(device),
        "target_rejected": target_rejected.to(device),
        "compare_rejected": compare_rejected.to(device),
    }


def launch(packed: dict[str, torch.Tensor], outputs: tuple[torch.Tensor, ...]) -> None:
    n = packed["select_min"].numel()
    official_select_gate_kernel[(n,)](
        packed["select_min"], packed["select_max"], packed["option_count"],
        packed["result"], packed["action_length"], packed["actions"],
        *outputs, n=n, block_actions=MAX_ACTIONS, num_warps=1,
    )


FIRST_OUTPUT_KEYS = (
    "turn", "first_player", "result", "select_type", "select_context",
    "select_min", "select_max", "terminal",
)


def pack_first_player(
    fixtures: list[dict[str, Any]], device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    inputs = {
        "turn": torch.tensor([row["before"]["turn"] for row in fixtures], dtype=torch.int32, device=device),
        "your_index": torch.tensor([row["before"]["your_index"] for row in fixtures], dtype=torch.int32, device=device),
        "result": torch.tensor([row["before"]["result"] for row in fixtures], dtype=torch.int32, device=device),
        "selected_option_type": torch.tensor([row["selected_option_type"] for row in fixtures], dtype=torch.int32, device=device),
    }
    targets = {
        "turn": torch.tensor([row["after"]["turn"] for row in fixtures], dtype=torch.int32, device=device),
        "first_player": torch.tensor([row["after"]["first_player"] for row in fixtures], dtype=torch.int32, device=device),
        "result": torch.tensor([row["after"]["result"] for row in fixtures], dtype=torch.int32, device=device),
        "select_type": torch.tensor([row["after"]["select_type"] for row in fixtures], dtype=torch.int32, device=device),
        "select_context": torch.tensor([row["after"]["select_context"] for row in fixtures], dtype=torch.int32, device=device),
        "select_min": torch.tensor([row["after"]["select_min"] for row in fixtures], dtype=torch.int32, device=device),
        "select_max": torch.tensor([row["after"]["select_max"] for row in fixtures], dtype=torch.int32, device=device),
        "terminal": torch.tensor([int(row["after"]["result"] != -1) for row in fixtures], dtype=torch.int32, device=device),
    }
    return inputs, targets


def launch_first_player(
    inputs: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]
) -> None:
    n = inputs["turn"].numel()
    official_first_player_step_kernel[(triton.cdiv(n, 256),)](
        inputs["turn"], inputs["your_index"], inputs["result"],
        inputs["selected_option_type"],
        *[outputs[key] for key in FIRST_OUTPUT_KEYS],
        n=n, block=256, num_warps=4,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--first-player-fixtures", type=Path, required=True)
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
    first_report = json.loads(args.first_player_fixtures.read_text())
    if first_report["official_lib_sha256"] != fixture_report["official_lib_sha256"]:
        raise RuntimeError("official fixture reports use different libcg binaries")
    fixtures = fixture_report["fixtures"]
    first_fixtures = first_report["fixtures"]
    report: dict[str, Any] = {
        "schema": "poke_bot.official_rule_slice_cuda/v2",
        "status": "running",
        "started_at": time.time(),
        "gpu": gpu,
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "official_lib_sha256": fixture_report["official_lib_sha256"],
        "official_fixture_count": len(fixtures),
        "official_accepted_step_fixture_count": len(first_fixtures),
        "official_games": fixture_report["games_completed"],
        "official_select_contexts": len(fixture_report["select_context_counts"]),
        "scope": (
            fixture_report["scope"]
            + "; accepted IsFirst control-flow transition"
        ),
        "accepted_transition_slice_coverage": True,
        "accepted_transition_coverage": False,
        "full_card_effect_transition_coverage": False,
        "production_eligible": False,
    }
    publish(args.json_out, report)

    started = time.perf_counter()
    packed = pack(fixtures, device)
    outputs = tuple(
        torch.empty(len(fixtures), device=device, dtype=torch.int32)
        for _ in range(3)
    )
    torch.cuda.synchronize(device)
    fixture_h2d_and_setup_s = time.perf_counter() - started
    launch(packed, outputs)
    torch.cuda.synchronize(device)
    error, terminal, rejected = outputs
    error_mismatches = int(
        ((error != packed["target_error"]) & packed["compare_error"]).sum().item()
    )
    terminal_mismatches = int((terminal != packed["target_terminal"]).sum().item())
    rejected_mismatches = int(
        ((rejected != packed["target_rejected"]) & packed["compare_rejected"]).sum().item()
    )
    report["parity"] = {
        "selection_error_mismatches": error_mismatches,
        "terminal_mismatches": terminal_mismatches,
        "rejected_step_public_slice_mismatches": rejected_mismatches,
        "selection_error_exact": error_mismatches == 0,
        "terminal_exact": terminal_mismatches == 0,
        "rejected_step_public_slice_exact": rejected_mismatches == 0,
        "fixture_h2d_and_setup_s": fixture_h2d_and_setup_s,
        "result_d2h_bytes": 24,
    }
    if error_mismatches or terminal_mismatches or rejected_mismatches:
        report["status"] = "failed"
        publish(args.json_out, report)
        raise RuntimeError(f"official slice parity failed: {report['parity']}")

    first_inputs, first_targets = pack_first_player(first_fixtures, device)
    first_outputs = {
        key: torch.empty(len(first_fixtures), device=device, dtype=torch.int32)
        for key in FIRST_OUTPUT_KEYS
    }
    launch_first_player(first_inputs, first_outputs)
    torch.cuda.synchronize(device)
    first_mismatches = {
        key: int((first_outputs[key] != first_targets[key]).sum().item())
        for key in FIRST_OUTPUT_KEYS
    }
    report["accepted_step_parity"] = {
        "scope": first_report["scope"],
        "fixtures": len(first_fixtures),
        "yes_fixtures": first_report["selected_yes_count"],
        "no_fixtures": first_report["selected_no_count"],
        "field_mismatches": first_mismatches,
        "exact": all(value == 0 for value in first_mismatches.values()),
        "compared_fields": list(FIRST_OUTPUT_KEYS),
        "excluded_next_state_fields": [
            "your_index and option list/count depend on hidden randomized setup hands"
        ],
        "result_d2h_bytes": len(FIRST_OUTPUT_KEYS) * 8,
    }
    if not report["accepted_step_parity"]["exact"]:
        report["status"] = "failed"
        publish(args.json_out, report)
        raise RuntimeError(
            f"accepted IsFirst step parity failed: {first_mismatches}"
        )

    # Replicate the official corpus while preserving device residency. No
    # actions or state cross PCIe during the measured iterations.
    lanes = max(len(fixtures), int(args.benchmark_lanes))
    repeats = (lanes + len(fixtures) - 1) // len(fixtures)
    bench: dict[str, torch.Tensor] = {}
    for key, value in packed.items():
        if value.ndim == 2:
            bench[key] = value.repeat((repeats, 1))[:lanes].contiguous()
        else:
            bench[key] = value.repeat(repeats)[:lanes].contiguous()
    bench_outputs = tuple(
        torch.empty(lanes, device=device, dtype=torch.int32) for _ in range(3)
    )
    for _ in range(10):
        launch(bench, bench_outputs)
    torch.cuda.synchronize(device)
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_started = time.perf_counter()
    begin.record()
    for _ in range(max(1, args.benchmark_iters)):
        launch(bench, bench_outputs)
    end.record()
    torch.cuda.synchronize(device)
    wall_s = time.perf_counter() - wall_started
    kernel_s = begin.elapsed_time(end) / 1000.0
    decisions = lanes * max(1, args.benchmark_iters)
    device_bytes = sum(value.numel() * value.element_size() for value in bench.values())
    device_bytes += sum(value.numel() * value.element_size() for value in bench_outputs)
    report["throughput"] = {
        "lanes": lanes,
        "iterations": args.benchmark_iters,
        "decisions": decisions,
        "kernel_s": kernel_s,
        "wall_s": wall_s,
        "kernel_decisions_per_s": decisions / max(kernel_s, 1e-12),
        "whole_loop_decisions_per_s": decisions / max(wall_s, 1e-12),
        "whole_loop_vs_kernel_efficiency": kernel_s / max(wall_s, 1e-12),
        "measured_bulk_h2d_bytes": 0,
        "measured_bulk_d2h_bytes": 0,
        "device_resident_bytes": device_bytes,
    }

    first_repeats = (lanes + len(first_fixtures) - 1) // len(first_fixtures)
    first_bench_inputs = {
        key: value.repeat(first_repeats)[:lanes].contiguous()
        for key, value in first_inputs.items()
    }
    first_bench_outputs = {
        key: torch.empty(lanes, device=device, dtype=torch.int32)
        for key in FIRST_OUTPUT_KEYS
    }
    for _ in range(10):
        launch_first_player(first_bench_inputs, first_bench_outputs)
    torch.cuda.synchronize(device)
    first_begin = torch.cuda.Event(enable_timing=True)
    first_end = torch.cuda.Event(enable_timing=True)
    first_wall_started = time.perf_counter()
    first_begin.record()
    for _ in range(max(1, args.benchmark_iters)):
        launch_first_player(first_bench_inputs, first_bench_outputs)
    first_end.record()
    torch.cuda.synchronize(device)
    first_wall_s = time.perf_counter() - first_wall_started
    first_kernel_s = first_begin.elapsed_time(first_end) / 1000.0
    first_device_bytes = sum(
        value.numel() * value.element_size()
        for value in (*first_bench_inputs.values(), *first_bench_outputs.values())
    )
    report["accepted_step_throughput"] = {
        "lanes": lanes,
        "iterations": args.benchmark_iters,
        "accepted_steps": decisions,
        "kernel_s": first_kernel_s,
        "wall_s": first_wall_s,
        "kernel_steps_per_s": decisions / max(first_kernel_s, 1e-12),
        "whole_loop_steps_per_s": decisions / max(first_wall_s, 1e-12),
        "whole_loop_vs_kernel_efficiency": first_kernel_s / max(first_wall_s, 1e-12),
        "measured_bulk_h2d_bytes": 0,
        "measured_bulk_d2h_bytes": 0,
        "device_resident_bytes": first_device_bytes,
    }
    report["status"] = "complete"
    report["completed_at"] = time.time()
    publish(args.json_out, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
