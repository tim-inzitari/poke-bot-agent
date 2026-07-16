"""Full-box hardware profile for a single pure-RL trainee."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from poke_bot import config


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(f"PURE_RL_{name}")
    if raw is None:
        raw = os.environ.get(f"POKEBOT_{name}")
    return int(raw) if raw is not None else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(f"PURE_RL_{name}")
    if raw is None:
        raw = os.environ.get(f"POKEBOT_{name}")
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class FullHardwareProfile:
    """Saturate CPU + dual GPU leaf servers for one active trainee."""

    sim_workers: int
    games_in_flight: int
    train_cuda_device: int  # PCI order: 1 = Blackwell
    leaf_gpu1_replicas: int
    leaf_gpu0_replicas: int
    torch_threads: int
    allow_single_gpu: bool = False
    per_worker_rss_gb: float = 0.8

    @property
    def leaf_replicas_total(self) -> int:
        return int(self.leaf_gpu0_replicas) + int(self.leaf_gpu1_replicas)

    def leaf_cuda_devices(self) -> list[int]:
        devices: list[int] = []
        devices.extend([1] * int(self.leaf_gpu1_replicas))
        devices.extend([0] * int(self.leaf_gpu0_replicas))
        return devices

    def validate_or_raise(self, *, visible_gpu_count: int) -> None:
        if self.sim_workers < 1:
            raise ValueError("sim_workers must be >= 1")
        if self.leaf_replicas_total < 1:
            raise ValueError("need at least one leaf replica")
        if visible_gpu_count >= 2 and not self.allow_single_gpu:
            if self.leaf_gpu0_replicas < 1 or self.leaf_gpu1_replicas < 1:
                raise ValueError(
                    "full-hardware launch requires leaf replicas on GPU0 and GPU1 "
                    "(set PURE_RL_ALLOW_SINGLE_GPU=1 to override)"
                )
            if int(self.train_cuda_device) != 1:
                raise ValueError(
                    "train device must be CUDA:1 (Blackwell) for full-hardware profile"
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "sim_workers": self.sim_workers,
            "games_in_flight": self.games_in_flight,
            "train_cuda_device": self.train_cuda_device,
            "leaf_gpu0_replicas": self.leaf_gpu0_replicas,
            "leaf_gpu1_replicas": self.leaf_gpu1_replicas,
            "torch_threads": self.torch_threads,
            "allow_single_gpu": self.allow_single_gpu,
            "per_worker_rss_gb": self.per_worker_rss_gb,
            "leaf_cuda_devices": self.leaf_cuda_devices(),
        }


def full_hardware_profile() -> FullHardwareProfile:
    """Aggressive defaults: use the whole dedicated box for one trainee."""
    cpu = int(config.CPU_THREADS)
    # Reserve a few threads for parent + torch train process.
    workers = _env_int("SIM_WORKERS", max(1, min(cpu, 40)))
    in_flight = _env_int("GAMES_IN_FLIGHT", max(workers, int(config.HARDWARE.rl_games_in_flight)))
    return FullHardwareProfile(
        sim_workers=workers,
        games_in_flight=in_flight,
        train_cuda_device=_env_int("TRAIN_CUDA_DEVICE", 1),
        leaf_gpu1_replicas=_env_int("LEAF_GPU1_REPLICAS", 6),
        leaf_gpu0_replicas=_env_int("LEAF_GPU0_REPLICAS", 3),
        torch_threads=_env_int("TORCH_THREADS", 8),
        allow_single_gpu=_env_bool("ALLOW_SINGLE_GPU", False),
        per_worker_rss_gb=float(
            os.environ.get(
                "PURE_RL_PER_WORKER_RSS_GB",
                os.environ.get(
                    "POKEBOT_PER_WORKER_RSS_GB",
                    str(config.HARDWARE.per_worker_rss_gb),
                ),
            )
        ),
    )
