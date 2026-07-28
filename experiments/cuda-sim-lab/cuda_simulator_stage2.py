#!/usr/bin/env python3
"""Persistent CUDA simulator R&D lane for the RTX 3080 Ti.

This is an intentionally isolated feasibility prototype, not a replacement for
the official Pokemon TCG engine.  It exercises the execution shape required by
an eventual device-resident simulator: stateful multi-step games, legality
fallbacks, prizes, knockouts, deck-out, turn limits, active-game masks, and
deterministic trace hashes.  Every published configuration must first match a
CPU reference bit-for-bit.  Production eligibility always remains false until
the complete official rules engine passes step/legal/terminal and seeded
full-game parity.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import signal
import time
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def stateful_step_kernel(
    hp0,
    hp1,
    energy0,
    energy1,
    prizes0,
    prizes1,
    deck0,
    deck1,
    active_player,
    turn,
    status0,
    status1,
    terminal,
    winner,
    seed,
    step,
    n: tl.constexpr,
    block: tl.constexpr,
):
    idx = tl.program_id(0) * block + tl.arange(0, block)
    mask = idx < n
    h0 = tl.load(hp0 + idx, mask=mask, other=0).to(tl.int32)
    h1 = tl.load(hp1 + idx, mask=mask, other=0).to(tl.int32)
    e0 = tl.load(energy0 + idx, mask=mask, other=0).to(tl.int32)
    e1 = tl.load(energy1 + idx, mask=mask, other=0).to(tl.int32)
    p0 = tl.load(prizes0 + idx, mask=mask, other=0).to(tl.int32)
    p1 = tl.load(prizes1 + idx, mask=mask, other=0).to(tl.int32)
    d0 = tl.load(deck0 + idx, mask=mask, other=0).to(tl.int32)
    d1 = tl.load(deck1 + idx, mask=mask, other=0).to(tl.int32)
    ap = tl.load(active_player + idx, mask=mask, other=0).to(tl.int32)
    t = tl.load(turn + idx, mask=mask, other=0).to(tl.int32)
    s0 = tl.load(status0 + idx, mask=mask, other=0).to(tl.int32)
    s1 = tl.load(status1 + idx, mask=mask, other=0).to(tl.int32)
    done = tl.load(terminal + idx, mask=mask, other=1).to(tl.int32)
    win = tl.load(winner + idx, mask=mask, other=-1).to(tl.int32)

    # Integer-only deterministic action stream. The selected action still
    # depends on evolving state so repeated steps cannot be optimized away.
    z = idx * 1103515245 + (step + seed) * 12345 + t * 97 + h0 * 7 + h1 * 11
    action = (z & 0x7fffffff) % 7
    amount = 10 + (((z >> 7) & 0x7fffffff) % 12) * 10
    alive = done == 0
    is_p0 = ap == 0

    actor_e = tl.where(is_p0, e0, e1)
    actor_h = tl.where(is_p0, h0, h1)
    target_h = tl.where(is_p0, h1, h0)
    actor_s = tl.where(is_p0, s0, s1)

    # Actions: attack, heal, attach, retreat, status, draw, end-turn.
    # Illegal selections deterministically fall back to end-turn.
    legal_attack = (action == 0) & (actor_e >= 2) & (actor_s != 2)
    legal_heal = (action == 1) & (actor_h < 340)
    legal_attach = (action == 2) & (actor_e < 12)
    legal_retreat = (action == 3) & (actor_e >= 1)
    legal_status = (action == 4) & (actor_e >= 1)
    legal_draw = action == 5
    chosen = legal_attack | legal_heal | legal_attach | legal_retreat | legal_status | legal_draw | (action == 6)
    end_turn = alive & (~chosen | (action == 3) | (action == 6))

    damage = tl.where(legal_attack & alive, amount + actor_e * 2, 0)
    next_target = tl.maximum(0, target_h - damage)
    next_actor = tl.where(legal_heal & alive, tl.minimum(340, actor_h + amount), actor_h)
    next_energy = tl.where(legal_attack & alive, tl.maximum(0, actor_e - 2), actor_e)
    next_energy = tl.where(legal_attach & alive, tl.minimum(12, next_energy + 1), next_energy)
    next_energy = tl.where(legal_retreat & alive, tl.maximum(0, next_energy - 1), next_energy)

    h0n = tl.where(is_p0, next_actor, next_target)
    h1n = tl.where(is_p0, next_target, next_actor)
    e0n = tl.where(is_p0, next_energy, e0)
    e1n = tl.where(is_p0, e1, next_energy)
    s0n = tl.where(legal_status & alive & is_p0, (s0 + 1) % 3, s0)
    s1n = tl.where(legal_status & alive & (~is_p0), (s1 + 1) % 3, s1)

    # Draw consumes the acting player's deck; zero-card draw loses.
    draw0 = legal_draw & alive & is_p0
    draw1 = legal_draw & alive & (~is_p0)
    d0n = tl.where(draw0, tl.maximum(0, d0 - 1), d0)
    d1n = tl.where(draw1, tl.maximum(0, d1 - 1), d1)
    deckout0 = draw0 & (d0 == 0)
    deckout1 = draw1 & (d1 == 0)

    ko0 = alive & (h0 > 0) & (h0n == 0)
    ko1 = alive & (h1 > 0) & (h1n == 0)
    p0n = tl.where(ko1, tl.maximum(0, p0 - 1), p0)
    p1n = tl.where(ko0, tl.maximum(0, p1 - 1), p1)
    # Replace a knocked-out active with a deterministic fresh bench proxy.
    h0n = tl.where(ko0 & (p1n > 0), 180 + ((idx + step) % 17) * 10, h0n)
    h1n = tl.where(ko1 & (p0n > 0), 180 + ((idx + step * 3) % 17) * 10, h1n)
    s0n = tl.where(ko0, 0, s0n)
    s1n = tl.where(ko1, 0, s1n)

    tn = t + end_turn.to(tl.int32)
    apn = tl.where(end_turn, 1 - ap, ap)
    timeout = tn >= 240
    finished = alive & ((p0n == 0) | (p1n == 0) | deckout0 | deckout1 | timeout)
    winner_now = tl.where(p0n == 0, 0, tl.where(p1n == 0, 1, tl.where(deckout0, 1, tl.where(deckout1, 0, (h0n >= h1n).to(tl.int32)))))
    donen = tl.where(finished, 1, done)
    winn = tl.where(finished, winner_now, win)

    # Terminal lanes are immutable.
    tl.store(hp0 + idx, tl.where(alive, h0n, h0), mask=mask)
    tl.store(hp1 + idx, tl.where(alive, h1n, h1), mask=mask)
    tl.store(energy0 + idx, tl.where(alive, e0n, e0), mask=mask)
    tl.store(energy1 + idx, tl.where(alive, e1n, e1), mask=mask)
    tl.store(prizes0 + idx, tl.where(alive, p0n, p0), mask=mask)
    tl.store(prizes1 + idx, tl.where(alive, p1n, p1), mask=mask)
    tl.store(deck0 + idx, tl.where(alive, d0n, d0), mask=mask)
    tl.store(deck1 + idx, tl.where(alive, d1n, d1), mask=mask)
    tl.store(active_player + idx, tl.where(alive, apn, ap), mask=mask)
    tl.store(turn + idx, tl.where(alive, tn, t), mask=mask)
    tl.store(status0 + idx, tl.where(alive, s0n, s0), mask=mask)
    tl.store(status1 + idx, tl.where(alive, s1n, s1), mask=mask)
    tl.store(terminal + idx, donen, mask=mask)
    tl.store(winner + idx, winn, mask=mask)


@triton.jit
def reset_state_kernel(
    hp0,
    hp1,
    energy0,
    energy1,
    prizes0,
    prizes1,
    deck0,
    deck1,
    active_player,
    turn,
    status0,
    status1,
    terminal,
    winner,
    n: tl.constexpr,
    block: tl.constexpr,
):
    """Reset already-allocated state buffers without a host-to-device copy."""
    idx = tl.program_id(0) * block + tl.arange(0, block)
    mask = idx < n
    tl.store(hp0 + idx, 180 + (idx % 17) * 10, mask=mask)
    tl.store(hp1 + idx, 180 + ((idx * 3) % 17) * 10, mask=mask)
    tl.store(energy0 + idx, idx % 5, mask=mask)
    tl.store(energy1 + idx, (idx * 7) % 5, mask=mask)
    tl.store(prizes0 + idx, 6, mask=mask)
    tl.store(prizes1 + idx, 6, mask=mask)
    tl.store(deck0 + idx, 40 + (idx % 13), mask=mask)
    tl.store(deck1 + idx, 40 + ((idx * 5) % 13), mask=mask)
    tl.store(active_player + idx, idx % 2, mask=mask)
    tl.store(turn + idx, 0, mask=mask)
    tl.store(status0 + idx, idx % 3, mask=mask)
    tl.store(status1 + idx, (idx * 11) % 3, mask=mask)
    tl.store(terminal + idx, 0, mask=mask)
    tl.store(winner + idx, -1, mask=mask)


@triton.jit
def compact_active_kernel(
    terminal,
    compacted_indices,
    compacted_count,
    n: tl.constexpr,
    block: tl.constexpr,
):
    """Compact live lane indices entirely on-device using one atomic counter."""
    idx = tl.program_id(0) * block + tl.arange(0, block)
    mask = idx < n
    live = mask & (tl.load(terminal + idx, mask=mask, other=1) == 0)
    # Atomic slots are intentionally unordered; the scheduler only requires a
    # dense active set, not stable lane order.
    live_i32 = live.to(tl.int32)
    prefix = tl.cumsum(live_i32, axis=0)
    block_count = tl.sum(live_i32, axis=0)
    base = tl.atomic_add(compacted_count, block_count)
    slots = base + prefix - 1
    tl.store(compacted_indices + slots, idx, mask=live)


FIELDS = (
    "hp0", "hp1", "energy0", "energy1", "prizes0", "prizes1",
    "deck0", "deck1", "active_player", "turn", "status0", "status1",
    "terminal", "winner",
)


def allocate_state(n: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: torch.empty(n, device=device, dtype=torch.int32)
        for name in FIELDS
    }


def reset_state(state: dict[str, torch.Tensor]) -> None:
    n = state["hp0"].numel()
    reset_state_kernel[(triton.cdiv(n, 256),)](
        *[state[name] for name in FIELDS], n=n, block=256, num_warps=4
    )


def initial_state(n: int, device: torch.device) -> dict[str, torch.Tensor]:
    idx = torch.arange(n, device=device, dtype=torch.int32)
    return {
        "hp0": 180 + (idx % 17) * 10,
        "hp1": 180 + ((idx * 3) % 17) * 10,
        "energy0": idx % 5,
        "energy1": (idx * 7) % 5,
        "prizes0": torch.full((n,), 6, device=device, dtype=torch.int32),
        "prizes1": torch.full((n,), 6, device=device, dtype=torch.int32),
        "deck0": 40 + (idx % 13),
        "deck1": 40 + ((idx * 5) % 13),
        "active_player": idx % 2,
        "turn": torch.zeros(n, device=device, dtype=torch.int32),
        "status0": idx % 3,
        "status1": (idx * 11) % 3,
        "terminal": torch.zeros(n, device=device, dtype=torch.int32),
        "winner": torch.full((n,), -1, device=device, dtype=torch.int32),
    }


def cpu_step(state: dict[str, torch.Tensor], seed: int, step: int) -> None:
    # The reference mirrors the kernel using int64 intermediates solely to
    # avoid Python-side overflow while preserving the kernel's positive mask.
    idx = torch.arange(state["hp0"].numel(), dtype=torch.int64)
    h0, h1 = state["hp0"], state["hp1"]
    e0, e1 = state["energy0"], state["energy1"]
    p0, p1 = state["prizes0"], state["prizes1"]
    d0, d1 = state["deck0"], state["deck1"]
    ap, t = state["active_player"], state["turn"]
    s0, s1 = state["status0"], state["status1"]
    done, win = state["terminal"], state["winner"]
    # Triton evaluates the lane arithmetic in signed int32.  Reproduce that
    # wrap exactly before extracting the positive 31-bit action stream.
    z = (
        idx * 1103515245
        + (step + seed) * 12345
        + t.to(torch.int64) * 97
        + h0.to(torch.int64) * 7
        + h1.to(torch.int64) * 11
    ).to(torch.int32).to(torch.int64)
    action = (z & 0x7fffffff) % 7
    amount = (10 + (((z >> 7) & 0x7fffffff) % 12) * 10).to(torch.int32)
    alive, is_p0 = done == 0, ap == 0
    actor_e = torch.where(is_p0, e0, e1)
    actor_h = torch.where(is_p0, h0, h1)
    target_h = torch.where(is_p0, h1, h0)
    actor_s = torch.where(is_p0, s0, s1)
    legal_attack = (action == 0) & (actor_e >= 2) & (actor_s != 2)
    legal_heal = (action == 1) & (actor_h < 340)
    legal_attach = (action == 2) & (actor_e < 12)
    legal_retreat = (action == 3) & (actor_e >= 1)
    legal_status = (action == 4) & (actor_e >= 1)
    legal_draw = action == 5
    chosen = legal_attack | legal_heal | legal_attach | legal_retreat | legal_status | legal_draw | (action == 6)
    end_turn = alive & (~chosen | (action == 3) | (action == 6))
    damage = torch.where(legal_attack & alive, amount + actor_e * 2, 0)
    next_target = torch.clamp_min(target_h - damage, 0)
    next_actor = torch.where(legal_heal & alive, torch.clamp_max(actor_h + amount, 340), actor_h)
    next_energy = torch.where(legal_attack & alive, torch.clamp_min(actor_e - 2, 0), actor_e)
    next_energy = torch.where(legal_attach & alive, torch.clamp_max(next_energy + 1, 12), next_energy)
    next_energy = torch.where(legal_retreat & alive, torch.clamp_min(next_energy - 1, 0), next_energy)
    h0n, h1n = torch.where(is_p0, next_actor, next_target), torch.where(is_p0, next_target, next_actor)
    e0n, e1n = torch.where(is_p0, next_energy, e0), torch.where(is_p0, e1, next_energy)
    s0n = torch.where(legal_status & alive & is_p0, (s0 + 1) % 3, s0)
    s1n = torch.where(legal_status & alive & ~is_p0, (s1 + 1) % 3, s1)
    draw0, draw1 = legal_draw & alive & is_p0, legal_draw & alive & ~is_p0
    d0n, d1n = torch.where(draw0, torch.clamp_min(d0 - 1, 0), d0), torch.where(draw1, torch.clamp_min(d1 - 1, 0), d1)
    deckout0, deckout1 = draw0 & (d0 == 0), draw1 & (d1 == 0)
    ko0, ko1 = alive & (h0 > 0) & (h0n == 0), alive & (h1 > 0) & (h1n == 0)
    p0n, p1n = torch.where(ko1, torch.clamp_min(p0 - 1, 0), p0), torch.where(ko0, torch.clamp_min(p1 - 1, 0), p1)
    h0n = torch.where(ko0 & (p1n > 0), 180 + ((idx + step) % 17).to(torch.int32) * 10, h0n)
    h1n = torch.where(ko1 & (p0n > 0), 180 + ((idx + step * 3) % 17).to(torch.int32) * 10, h1n)
    s0n, s1n = torch.where(ko0, 0, s0n), torch.where(ko1, 0, s1n)
    tn, apn = t + end_turn.to(torch.int32), torch.where(end_turn, 1 - ap, ap)
    timeout = tn >= 240
    finished = alive & ((p0n == 0) | (p1n == 0) | deckout0 | deckout1 | timeout)
    winner_now = torch.where(p0n == 0, 0, torch.where(p1n == 0, 1, torch.where(deckout0, 1, torch.where(deckout1, 0, (h0n >= h1n).to(torch.int32)))))
    updates = (h0n, h1n, e0n, e1n, p0n, p1n, d0n, d1n, apn, tn, s0n, s1n, torch.where(finished, 1, done), torch.where(finished, winner_now, win))
    for name, value in zip(FIELDS, updates):
        state[name] = torch.where(alive, value, state[name]).to(torch.int32)


def run_steps(state: dict[str, torch.Tensor], seed: int, steps: int) -> None:
    n = state["hp0"].numel()
    grid = (triton.cdiv(n, 256),)
    for step in range(steps):
        stateful_step_kernel[grid](*[state[name] for name in FIELDS], seed, step, n=n, block=256, num_warps=4)


def trace_hash_device(state: dict[str, torch.Tensor]) -> torch.Tensor:
    weights = torch.arange(1, len(FIELDS) + 1, device=state["hp0"].device, dtype=torch.int64)
    values = torch.stack([state[name].to(torch.int64).sum() for name in FIELDS])
    return (values * weights).sum()


def trace_hash(state: dict[str, torch.Tensor]) -> int:
    return int(trace_hash_device(state).item())


def transfer_probe(
    n: int, device: torch.device
) -> dict[str, float | int | str]:
    """Measure the avoidable full-state copies used by a naive host pipeline."""
    cpu = initial_state(n, torch.device("cpu"))
    state_bytes = sum(value.numel() * value.element_size() for value in cpu.values())
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    cuda = {name: value.to(device) for name, value in cpu.items()}
    torch.cuda.synchronize(device)
    h2d_s = time.perf_counter() - started
    started = time.perf_counter()
    copied = {name: value.cpu() for name, value in cuda.items()}
    torch.cuda.synchronize(device)
    d2h_s = time.perf_counter() - started
    # Touch a copied scalar so this cannot become a dead transfer.
    checksum = int(sum(int(value.reshape(-1)[0]) for value in copied.values()))
    result: dict[str, float | int | str] = {
        "states": n,
        "state_bytes_each_direction": state_bytes,
        "pageable_h2d_s": h2d_s,
        "pageable_d2h_s": d2h_s,
        "pageable_h2d_gib_per_s": state_bytes / max(h2d_s, 1e-12) / (1024**3),
        "pageable_d2h_gib_per_s": state_bytes / max(d2h_s, 1e-12) / (1024**3),
        # Backward-compatible aliases used by the top-level comparison.
        "h2d_s": h2d_s,
        "d2h_s": d2h_s,
        "h2d_gib_per_s": state_bytes / max(h2d_s, 1e-12) / (1024**3),
        "d2h_gib_per_s": state_bytes / max(d2h_s, 1e-12) / (1024**3),
        "checksum": checksum,
    }
    try:
        pinned = {name: value.pin_memory() for name, value in cpu.items()}
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        cuda_pinned = {
            name: value.to(device, non_blocking=True)
            for name, value in pinned.items()
        }
        torch.cuda.synchronize(device)
        pinned_h2d_s = time.perf_counter() - started
        pinned_out = {
            name: torch.empty_like(value, device="cpu", pin_memory=True)
            for name, value in pinned.items()
        }
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for name in FIELDS:
            pinned_out[name].copy_(cuda_pinned[name], non_blocking=True)
        torch.cuda.synchronize(device)
        pinned_d2h_s = time.perf_counter() - started
        result.update({
            "pinned_async_h2d_s": pinned_h2d_s,
            "pinned_async_d2h_s": pinned_d2h_s,
            "pinned_async_h2d_gib_per_s": state_bytes / max(pinned_h2d_s, 1e-12) / (1024**3),
            "pinned_async_d2h_gib_per_s": state_bytes / max(pinned_d2h_s, 1e-12) / (1024**3),
        })
        del pinned, cuda_pinned, pinned_out
    except RuntimeError as exc:
        result["pinned_async_error"] = f"{type(exc).__name__}: {exc}"
    del cpu, cuda, copied
    gc.collect()
    torch.cuda.empty_cache()
    return result


def publish(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--sizes", default="4096,16384,65536,262144,1048576")
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--parity-states", type=int, default=2048)
    parser.add_argument("--parity-steps", type=int, default=96)
    parser.add_argument("--publish-interval", type=float, default=1.0)
    parser.add_argument(
        "--official-slice-report",
        type=Path,
        default=Path("/home/inzi/cuda-sim-lab/outputs/official-rule-slice-cuda.json"),
    )
    parser.add_argument(
        "--official-fixtures-report",
        type=Path,
        default=Path("/home/inzi/cuda-sim-lab/outputs/official-rule-slice-fixtures.json"),
    )
    parser.add_argument(
        "--official-end-turn-report",
        type=Path,
        default=Path("/home/inzi/cuda-sim-lab/outputs/official-clean-end-turn-cuda.json"),
    )
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    device = torch.device("cuda:0")
    gpu = torch.cuda.get_device_name(device)
    if "3080 Ti" not in gpu:
        raise SystemExit(f"refusing non-3080 device: {gpu}")
    report: dict[str, Any] = {
        "schema": "poke_bot.cuda_simulator_lab/v3",
        "service": "pokebot-cuda-3080-lab.service",
        "role": "OUT OF FLEET - CUDA simulator testing",
        "pid": os.getpid(),
        "gpu": gpu,
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "scope": "end-to-end device-resident Pokemon-shaped CUDA simulator pipeline prototype",
        "device_residency": {
            "state": "GPU-persistent across all measured steps",
            "actions": "generated inside the CUDA kernel",
            "active_compaction": "GPU atomic compaction into a persistent fixed-capacity buffer",
            "results": "GPU-resident; only active-count and trace-hash scalars copied to host",
            "bulk_h2d_bytes_per_cycle": 0,
            "scalar_d2h_bytes_per_cycle": 12,
        },
        "profiler": {
            "method": "CUDA events plus wall-clock phase accounting",
            "ncu_available": bool(shutil.which("ncu")),
            "note": (
                "ncu is available but CUDA events remain the always-on low-overhead profiler"
                if shutil.which("ncu")
                else "ncu is not installed; no driver/toolkit mutation was made during production training"
            ),
        },
        "official_engine_parity": False,
        "production_eligible": False,
        "production_blockers": [
            "complete Pokemon rule coverage",
            "step/legal/terminal parity against official libcg",
            "seeded full-game transition-hash parity against official libcg",
            "end-to-end games/s win including policy inference",
        ],
        "status": "starting",
        "started_at": time.time(),
        "rows": [],
    }
    if args.official_slice_report.is_file():
        validated = json.loads(args.official_slice_report.read_text())
        report["official_rule_slice"] = {
            key: validated.get(key)
            for key in (
                "schema", "status", "completed_at", "scope", "gpu", "gpu_uuid",
                "official_lib_sha256", "official_fixture_count", "official_games",
                "official_select_contexts", "official_accepted_step_fixture_count",
                "accepted_transition_slice_coverage", "accepted_transition_coverage",
                "full_card_effect_transition_coverage", "production_eligible",
                "parity", "accepted_step_parity", "throughput",
                "accepted_step_throughput",
            )
        }
    if args.official_fixtures_report.is_file():
        fixtures = json.loads(args.official_fixtures_report.read_text())
        report["official_engine_port_audit"] = fixtures.get("source_audit")
        report["official_fixture_error_counts"] = fixtures.get("official_error_counts")
    if args.official_end_turn_report.is_file():
        report["official_clean_end_turn_slice"] = json.loads(
            args.official_end_turn_report.read_text()
        )
    publish(args.json_out, report)
    stop = False
    def request_stop(_sig, _frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    # Exact reference gate before any performance result is accepted.
    cpu = initial_state(args.parity_states, torch.device("cpu"))
    cuda = initial_state(args.parity_states, device)
    for step in range(args.parity_steps):
        cpu_step(cpu, args.seed, step)
    run_steps(cuda, args.seed, args.parity_steps)
    torch.cuda.synchronize()
    mismatches = {name: int((cuda[name].cpu() != cpu[name]).sum().item()) for name in FIELDS}
    parity = all(value == 0 for value in mismatches.values())
    report["prototype_cpu_parity"] = parity
    report["prototype_parity_mismatches"] = mismatches
    report["prototype_trace_hash_cpu"] = trace_hash(cpu)
    report["prototype_trace_hash_cuda"] = trace_hash(cuda)
    if not parity or report["prototype_trace_hash_cpu"] != report["prototype_trace_hash_cuda"]:
        report["status"] = "failed"
        publish(args.json_out, report)
        raise RuntimeError(f"prototype parity failed: {mismatches}")
    report["transfer_probe"] = transfer_probe(1_048_576, device)
    report["status"] = "allocating_persistent_device_state"
    report["updated_at"] = time.time()
    publish(args.json_out, report)

    cycle = 0
    last_publish = 0.0
    sizes = [int(value) for value in args.sizes.split(",") if value]
    persistent: dict[int, tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]] = {}
    for n in sizes:
        state = allocate_state(n, device)
        compacted_indices = torch.empty(n, device=device, dtype=torch.int32)
        compacted_count = torch.zeros(1, device=device, dtype=torch.int32)
        reset_state(state)
        # Compile/warm all measured kernels before timing.
        run_steps(state, args.seed, 1)
        compact_active_kernel[(triton.cdiv(n, 256),)](
            state["terminal"], compacted_indices, compacted_count,
            n=n, block=256, num_warps=4,
        )
        persistent[n] = (state, compacted_indices, compacted_count)
    torch.cuda.synchronize(device)
    report["persistent_device_state_bytes"] = sum(
        sum(value.numel() * value.element_size() for value in state.values())
        + indices.numel() * indices.element_size()
        + count.numel() * count.element_size()
        for state, indices, count in persistent.values()
    )
    report["status"] = "running"
    report["updated_at"] = time.time()
    publish(args.json_out, report)

    while not stop:
        n = sizes[cycle % len(sizes)]
        state, compacted_indices, compacted_count = persistent[n]
        reset_begin = torch.cuda.Event(enable_timing=True)
        reset_end = torch.cuda.Event(enable_timing=True)
        kernel_begin = torch.cuda.Event(enable_timing=True)
        kernel_end = torch.cuda.Event(enable_timing=True)
        compact_begin = torch.cuda.Event(enable_timing=True)
        compact_end = torch.cuda.Event(enable_timing=True)
        reduce_begin = torch.cuda.Event(enable_timing=True)
        reduce_end = torch.cuda.Event(enable_timing=True)

        pipeline_started = time.perf_counter()
        reset_begin.record()
        reset_state(state)
        reset_end.record()
        kernel_begin.record()
        run_steps(state, args.seed + cycle * 1000003, args.steps)
        kernel_end.record()
        compact_begin.record()
        compacted_count.zero_()
        compact_active_kernel[(triton.cdiv(n, 256),)](
            state["terminal"], compacted_indices, compacted_count,
            n=n, block=256, num_warps=4,
        )
        compact_end.record()
        reduce_begin.record()
        trace_scalar = trace_hash_device(state)
        reduce_end.record()
        submit_finished = time.perf_counter()
        torch.cuda.synchronize(device)
        synchronized = time.perf_counter()
        d2h_started = synchronized
        active = int(compacted_count.item())
        trace = int(trace_scalar.item())
        d2h_finished = time.perf_counter()

        reset_gpu_s = reset_begin.elapsed_time(reset_end) / 1000.0
        kernel_gpu_s = kernel_begin.elapsed_time(kernel_end) / 1000.0
        compaction_gpu_s = compact_begin.elapsed_time(compact_end) / 1000.0
        reduction_gpu_s = reduce_begin.elapsed_time(reduce_end) / 1000.0
        gpu_stages_s = reset_gpu_s + kernel_gpu_s + compaction_gpu_s + reduction_gpu_s
        host_submit_s = submit_finished - pipeline_started
        sync_wait_s = synchronized - submit_finished
        scalar_d2h_s = d2h_finished - d2h_started
        pipeline_s = d2h_finished - pipeline_started
        transitions = n * args.steps
        row = {
            "cycle": cycle,
            "states": n,
            "steps": args.steps,
            "state_bytes_gpu_resident": sum(value.numel() * value.element_size() for value in state.values()),
            "actions_gpu_generated": True,
            "bulk_h2d_bytes": 0,
            "scalar_d2h_bytes": 12,
            "reset_gpu_s": reset_gpu_s,
            "kernel_only_gpu_s": kernel_gpu_s,
            "compaction_gpu_s": compaction_gpu_s,
            "reduction_gpu_s": reduction_gpu_s,
            "gpu_stages_s": gpu_stages_s,
            "host_submit_s": host_submit_s,
            "synchronization_wait_s": sync_wait_s,
            "scalar_d2h_s": scalar_d2h_s,
            "whole_pipeline_s": pipeline_s,
            "state_transitions": transitions,
            "kernel_only_transitions_per_s": transitions / max(kernel_gpu_s, 1e-12),
            "whole_pipeline_transitions_per_s": transitions / max(pipeline_s, 1e-12),
            "whole_pipeline_vs_kernel_efficiency": kernel_gpu_s / max(pipeline_s, 1e-12),
            "wall_fraction_host_submit": host_submit_s / max(pipeline_s, 1e-12),
            "wall_fraction_synchronization_wait": sync_wait_s / max(pipeline_s, 1e-12),
            "wall_fraction_scalar_d2h": scalar_d2h_s / max(pipeline_s, 1e-12),
            "gpu_fraction_reset": reset_gpu_s / max(gpu_stages_s, 1e-12),
            "gpu_fraction_transition_kernel": kernel_gpu_s / max(gpu_stages_s, 1e-12),
            "gpu_fraction_compaction": compaction_gpu_s / max(gpu_stages_s, 1e-12),
            "gpu_fraction_reduction": reduction_gpu_s / max(gpu_stages_s, 1e-12),
            "active_after_steps": active,
            "terminal_after_steps": n - active,
            "prototype_trace_hash": trace,
            "prototype_cpu_parity_gate": parity,
            "completed_at": time.time(),
        }
        probe = report["transfer_probe"]
        if n == int(probe["states"]):
            full_copy_s = float(probe["h2d_s"]) + float(probe["d2h_s"])
            row["counterfactual_full_state_h2d_plus_d2h_s"] = full_copy_s
            row["counterfactual_full_state_copy_vs_kernel"] = (
                full_copy_s / max(kernel_gpu_s, 1e-12)
            )
            if "pinned_async_h2d_s" in probe and "pinned_async_d2h_s" in probe:
                pinned_copy_s = float(probe["pinned_async_h2d_s"]) + float(probe["pinned_async_d2h_s"])
                row["counterfactual_pinned_full_state_h2d_plus_d2h_s"] = pinned_copy_s
                row["counterfactual_pinned_full_state_copy_vs_kernel"] = (
                    pinned_copy_s / max(kernel_gpu_s, 1e-12)
                )
        report["rows"] = (report["rows"] + [row])[-30:]
        report["latest_profile"] = row
        report.setdefault("last_profile_by_size", {})[str(n)] = row
        report["cycles_completed"] = cycle + 1
        report["updated_at"] = time.time()
        # The benchmark can complete dozens of cycles per second.  Throttle
        # durable writes/log lines so a long-running lab cannot churn disk or
        # grow its log aggressively while dashboard freshness stays sub-second
        # to roughly one second.
        now = time.monotonic()
        if now - last_publish >= max(0.1, args.publish_interval):
            publish(args.json_out, report)
            print(json.dumps(row, sort_keys=True), flush=True)
            last_publish = now
        cycle += 1
    report["status"] = "stopped"
    report["stopped_at"] = time.time()
    publish(args.json_out, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
