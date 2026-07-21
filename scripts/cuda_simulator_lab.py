#!/usr/bin/env python3
"""CUDA simulator feasibility lane: exact batched transition/compaction gate.

This prototype deliberately does not claim to implement Pokémon rules.  It
proves the device-resident execution shape needed by such an implementation:
one independent game state per lane, deterministic integer transitions,
terminal flags, and compacted active-game indices.  Every measured CUDA row is
checked bit-for-bit against a CPU reference.  Production eligibility remains
false until the complete rule engine and seeded full games match official
libcg.
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


@triton.jit
def transition_kernel(
    hp,
    prizes,
    energy,
    turn,
    action,
    amount,
    out_hp,
    out_prizes,
    out_energy,
    out_turn,
    terminal,
    n: tl.constexpr,
    block: tl.constexpr,
):
    idx = tl.program_id(0) * block + tl.arange(0, block)
    mask = idx < n
    h = tl.load(hp + idx, mask=mask, other=0).to(tl.int32)
    p = tl.load(prizes + idx, mask=mask, other=0).to(tl.int32)
    e = tl.load(energy + idx, mask=mask, other=0).to(tl.int32)
    t = tl.load(turn + idx, mask=mask, other=0).to(tl.int32)
    a = tl.load(action + idx, mask=mask, other=0).to(tl.int32)
    x = tl.load(amount + idx, mask=mask, other=0).to(tl.int32)

    # Four deterministic prototype actions: damage, heal, attach, end turn.
    h2 = tl.where(a == 0, tl.maximum(0, h - x), h)
    h2 = tl.where(a == 1, tl.minimum(340, h + x), h2)
    e2 = tl.where(a == 2, tl.minimum(12, e + 1), e)
    t2 = tl.where(a == 3, t + 1, t)
    knockout = (a == 0) & (h > 0) & (h2 == 0)
    p2 = tl.where(knockout, tl.maximum(0, p - 1), p)
    done = (p2 == 0) | (t2 >= 200)

    tl.store(out_hp + idx, h2, mask=mask)
    tl.store(out_prizes + idx, p2, mask=mask)
    tl.store(out_energy + idx, e2, mask=mask)
    tl.store(out_turn + idx, t2, mask=mask)
    tl.store(terminal + idx, done.to(tl.int32), mask=mask)


def cpu_reference(hp, prizes, energy, turn, action, amount):
    h2 = torch.where(action == 0, torch.clamp_min(hp - amount, 0), hp)
    h2 = torch.where(action == 1, torch.clamp_max(hp + amount, 340), h2)
    e2 = torch.where(action == 2, torch.clamp_max(energy + 1, 12), energy)
    t2 = torch.where(action == 3, turn + 1, turn)
    knockout = (action == 0) & (hp > 0) & (h2 == 0)
    p2 = torch.where(knockout, torch.clamp_min(prizes - 1, 0), prizes)
    done = ((p2 == 0) | (t2 >= 200)).to(torch.int32)
    return h2, p2, e2, t2, done


def publish(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--sizes", default="1024,4096,16384,65536,262144")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    device = torch.device("cuda:0")
    gpu = torch.cuda.get_device_name(device)
    report: dict[str, Any] = {
        "schema": "poke_bot.cuda_simulator_lab/v1",
        "status": "running",
        "scope": "device-resident transition and active-game compaction skeleton",
        "production_eligible": False,
        "production_blockers": [
            "complete Pokemon rule coverage",
            "step/legal/terminal parity against official libcg",
            "seeded full-game transition-hash parity",
            "end-to-end games/s win including policy inference",
        ],
        "gpu": gpu,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "started_at": time.time(),
        "rows": [],
    }
    publish(args.json_out, report)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    try:
        for n in [int(value) for value in args.sizes.split(",") if value.strip()]:
            cpu = {
                "hp": torch.randint(1, 341, (n,), generator=generator, dtype=torch.int32),
                "prizes": torch.randint(1, 7, (n,), generator=generator, dtype=torch.int32),
                "energy": torch.randint(0, 13, (n,), generator=generator, dtype=torch.int32),
                "turn": torch.randint(0, 200, (n,), generator=generator, dtype=torch.int32),
                "action": torch.randint(0, 4, (n,), generator=generator, dtype=torch.int32),
                "amount": torch.randint(10, 351, (n,), generator=generator, dtype=torch.int32),
            }
            expected = cpu_reference(**cpu)
            src = {key: value.to(device) for key, value in cpu.items()}
            outputs = [torch.empty(n, device=device, dtype=torch.int32) for _ in range(5)]
            grid = (triton.cdiv(n, 256),)
            for _ in range(10):
                transition_kernel[grid](
                    *src.values(), *outputs, n=n, block=256, num_warps=4
                )
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            for _ in range(max(1, args.steps)):
                transition_kernel[grid](
                    *src.values(), *outputs, n=n, block=256, num_warps=4
                )
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            parity = all(torch.equal(actual.cpu(), want) for actual, want in zip(outputs, expected))
            active = torch.nonzero(outputs[-1] == 0, as_tuple=False).flatten()
            torch.cuda.synchronize(device)
            row = {
                "states": n,
                "steps": args.steps,
                "elapsed_s": elapsed,
                "transitions_per_s": n * args.steps / elapsed,
                "ns_per_transition": elapsed * 1e9 / (n * args.steps),
                "active_after_step": int(active.numel()),
                "bit_exact_cpu_parity": parity,
            }
            if not parity:
                raise RuntimeError(f"transition parity failed at n={n}")
            report["rows"].append(row)
            report["updated_at"] = time.time()
            publish(args.json_out, report)
            print(json.dumps(row, sort_keys=True), flush=True)
        report["status"] = "complete"
        report["completed_at"] = time.time()
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["failed_at"] = time.time()
        raise
    finally:
        publish(args.json_out, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
