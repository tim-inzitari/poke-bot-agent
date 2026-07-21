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
        """Spread GPU0 leaves evenly across the full server index map.

        A contiguous ``[1]*N + [0]*M`` map is fatal when ``num_workers <= N``:
        every self-play client lands on Blackwell and the 3080 Ti farm idles.

        Pairwise ``[1,0]*M + [1]*(N-M)`` still bunches all GPU0 indices into the
        first ``2*M`` slots — workers bound past that window never touch the
        3080, so util boom-busts when the small GPU0 client cohort is in CPU
        sim between MCTS leaf waves. Even spacing keeps sticky modulo binds
        mixed across the whole worker range.
        """
        n1 = max(0, int(self.leaf_gpu1_replicas))
        n0 = max(0, int(self.leaf_gpu0_replicas))
        total = n0 + n1
        if total <= 0:
            return []
        if n0 <= 0:
            return [1] * n1
        if n1 <= 0:
            return [0] * n0
        devices = [1] * total
        for k in range(n0):
            devices[(k * total) // n0] = 0
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


def sticky_leaf_server_index(
    slot: int,
    devices: list[int],
    *,
    gpu0_client_frac: float = 0.38,
) -> int:
    """Map a worker slot to a leaf server, oversubscribing GPU0 clients.

    Proportional ``slot % n`` only gives GPU0 ``n0/n`` of clients (~20/96 at
    10/50 leaves). That small cohort goes idle together between MCTS waves
    while Blackwell's larger client set keeps GPU1 saturated. Bias ~38% of
    slots onto the existing GPU0 leaves (no extra VRAM / leaf procs).
    """
    n = len(devices)
    if n <= 0:
        raise ValueError("devices must be non-empty")
    slot = int(slot)
    g0 = [i for i, d in enumerate(devices) if int(d) == 0]
    g1 = [i for i, d in enumerate(devices) if int(d) != 0]
    if not g0:
        return slot % n
    if not g1:
        return g0[slot % len(g0)]
    frac = min(0.70, max(0.15, float(gpu0_client_frac)))
    period = 100
    thresh = max(1, min(period - 1, int(round(period * frac))))
    within = slot % period
    block = slot // period
    if within < thresh:
        rank = block * thresh + within
        return g0[rank % len(g0)]
    rank = block * (period - thresh) + (within - thresh)
    return g1[rank % len(g1)]


def pick_leaf_server_index(
    *,
    req_qs: list,
    devices: list[int],
    alive_evts: list | None = None,
) -> int:
    """Least-queue leaf pick; empty GPU0 queues win ties (feed the 3080)."""
    n = len(req_qs)
    if n <= 0:
        raise ValueError("req_qs must be non-empty")
    best_i = 0
    best_key: tuple[int, int, int] | None = None
    for i, q in enumerate(req_qs):
        if alive_evts is not None and i < len(alive_evts):
            ev = alive_evts[i]
            if ev is not None and not ev.is_set():
                continue
        try:
            depth = max(0, int(q.qsize()))
        except (AttributeError, NotImplementedError, OSError):
            depth = 0
        dev = int(devices[i]) if i < len(devices) else 1
        # Lower depth first; prefer GPU0 on ties; stable by index.
        key = (depth, 0 if dev == 0 else 1, i)
        if best_key is None or key < best_key:
            best_key = key
            best_i = i
    return int(best_i)


def full_hardware_profile() -> FullHardwareProfile:
    """Aggressive defaults: use the whole dedicated box for one trainee."""
    # Import here to avoid import cycles at module load.
    from poke_bot.live_pool import _MAX_LEAF_GPU0, _MAX_LEAF_GPU1

    cpu = int(config.CPU_THREADS)
    # Reserve a few threads for parent + torch train process.
    workers = _env_int("SIM_WORKERS", max(1, min(cpu, 40)))
    in_flight = _env_int("GAMES_IN_FLIGHT", max(workers, int(config.HARDWARE.rl_games_in_flight)))
    # Env may request more (stale relaunch scripts); hard-cap 3080 Ti for 12GB.
    allow_single_gpu = _env_bool("ALLOW_SINGLE_GPU", False)
    leaf_gpu0 = min(_env_int("LEAF_GPU0_REPLICAS", 12), _MAX_LEAF_GPU0)
    leaf_gpu1 = min(_env_int("LEAF_GPU1_REPLICAS", 30), _MAX_LEAF_GPU1)
    return FullHardwareProfile(
        sim_workers=workers,
        games_in_flight=in_flight,
        train_cuda_device=_env_int("TRAIN_CUDA_DEVICE", 1),
        # Steady: GPU0=12 (12GB, no headroom); GPU1=30 (Blackwell growth to 48).
        leaf_gpu1_replicas=max(0 if allow_single_gpu else 1, leaf_gpu1),
        leaf_gpu0_replicas=max(0 if allow_single_gpu else 1, leaf_gpu0),
        torch_threads=_env_int("TORCH_THREADS", 8),
        allow_single_gpu=allow_single_gpu,
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
