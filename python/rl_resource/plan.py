"""Advisory knob ratchet: slow up, fast down, opaque env names."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from .sample import sample_cpu, sample_gpus, sample_ram


@dataclass
class Knob:
    env: str
    value: int
    step: int
    ceiling: int
    floor: int

    def bump(self) -> bool:
        if self.value + self.step <= self.ceiling:
            self.value += self.step
            return True
        return False

    def backoff(self) -> bool:
        if self.value - self.step >= self.floor:
            self.value -= self.step
            return True
        return False


@dataclass
class ResourcePlan:
    knobs: dict[str, Knob]
    vram_max_pct: float = 85.0
    ram_max_gb: float = 110.0
    ram_cushion_gb: float = 10.0
    cpu_max_pct: float = 92.0
    cpu_headroom_pct: float = 70.0
    hysteresis: int = 4
    min_bump_interval_s: float = 180.0
    headroom_streak: int = 0
    last_bump_monotonic: float = 0.0
    reason: str = "init"

    def env_map(self) -> dict[str, int]:
        return {k.env: int(k.value) for k in self.knobs.values()}

    def as_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "env": self.env_map(),
            "knobs": {name: asdict(k) for name, k in self.knobs.items()},
        }

    def write_json(self, path: str | Path) -> None:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + f".tmp.{os.getpid()}")
        tmp.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n")
        os.replace(tmp, dest)


def ratchet_step(
    plan: ResourcePlan,
    *,
    now: Optional[float] = None,
    ram: Optional[Mapping[str, float]] = None,
    cpu_pct: Optional[float] = None,
    gpus: Optional[Iterable[Mapping[str, Any]]] = None,
) -> ResourcePlan:
    """One sample: backoff immediately under pressure; bump only with hysteresis."""
    now = time.monotonic() if now is None else float(now)
    ram = dict(ram if ram is not None else sample_ram())
    cpu_pct = float(cpu_pct if cpu_pct is not None else sample_cpu())
    gpu_rows = list(gpus if gpus is not None else sample_gpus())
    max_vram = max((float(g.get("mem_pct") or 0.0) for g in gpu_rows), default=0.0)
    pressure = (
        max_vram >= plan.vram_max_pct
        or float(ram.get("used_gb") or 0.0) >= plan.ram_max_gb
        or float(ram.get("available_gb") or 0.0) < plan.ram_cushion_gb
        or cpu_pct >= plan.cpu_max_pct
    )
    if pressure:
        changed = False
        for knob in plan.knobs.values():
            changed = knob.backoff() or changed
        plan.headroom_streak = 0
        plan.reason = "pressure_backoff" if changed else "pressure_hold"
        return plan
    headroom = cpu_pct <= plan.cpu_headroom_pct and max_vram < plan.vram_max_pct * 0.9
    if not headroom:
        plan.headroom_streak = 0
        plan.reason = "neutral"
        return plan
    plan.headroom_streak += 1
    if plan.headroom_streak < plan.hysteresis:
        plan.reason = "headroom_wait"
        return plan
    if now - plan.last_bump_monotonic < plan.min_bump_interval_s:
        plan.reason = "bump_cooldown"
        return plan
    changed = False
    for knob in plan.knobs.values():
        changed = knob.bump() or changed
    if changed:
        plan.last_bump_monotonic = now
        plan.headroom_streak = 0
        plan.reason = "bump"
    else:
        plan.reason = "at_ceiling"
    return plan
