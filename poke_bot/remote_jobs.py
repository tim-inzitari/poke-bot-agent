"""TCP job protocol for remote poke-bot sim workers (TrueNAS / LAN).

Design intent
-------------
Remote hosts run **CPU sims + co-located GPU leaf** (e.g. TrueNAS elmo 3060 LHR 12 GB).
The training box ships whole-game jobs (and receives records), not per-leaf
RPCs — that keeps LAN chatter to ~1 RTT per game instead of thousands of
leaf forwards per move.

Wire format: length-prefixed JSON frames (``!I`` big-endian uint32 + UTF-8
JSON body). Additive / optional: local trainers keep using in-process
``WorkerPool`` until ``--remote-worker-endpoints`` (or the canary client) is
explicitly set.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import queue
import shlex
import shutil
import socket
import struct
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

try:  # Optional fast, GIL-friendly codec; wire format remains ordinary JSON.
    import orjson as _orjson
except ImportError:  # pragma: no cover - exercised on minimal worker images
    _orjson = None

PROTO_VERSION = 1
DEFAULT_PORT = 8765
_HDR = struct.Struct("!I")
_MAX_FRAME = 256 * 1024 * 1024  # 256 MiB — self-play records can be large


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

# TrueNAS docker worker only mounts ``./checkpoint`` → ``/workspace/checkpoint``.
# Bert native checkout mirrors under ``/Users/tsinzitari/workspace/poke-bot-agent``.
_TRAIN_ROOT = Path("/home/inzi/poke-bot-agent")
_BERT_ROOT = Path("/Users/tsinzitari/workspace/poke-bot-agent")
_ELMO_HOSTS = frozenset({"192.168.1.143", "truenas.local", "truenas"})
_BERT_HOSTS = frozenset(
    {"192.168.1.157", "192.168.1.158", "bert.local", "bert"}
)
_BERT_SSH = os.environ.get("POKEBOT_BERT_SSH", "tsinzitari@bert.local")
_ELMO_STAGE_LOCK = threading.RLock()
_ELMO_STAGE_CACHE: dict[tuple[str, str, str], str] = {}
_BERT_STAGE_LOCK = threading.RLock()
_BERT_STAGE_CACHE: dict[tuple[str, str], str] = {}
_MATCHUP_RUNTIME_MARKER_NAME = "matchup-runtime-activation.json"
_MATCHUP_RUNTIME_MARKER_SCHEMA = "poke_bot.remote_matchup_runtime_activation/v1"

# Remote demand contract (max ≫ default; scheduler grows into max):
#   Elmo default 20 / max 40; Bert default 10 / max 20.
# Remotes should *run* a pool at max capacity and advertise default via
# hello ``workers`` so legacy trainers don't jump to the ceiling.
_ELMO_DEFAULT_WORKERS = 20
_ELMO_MAX_WORKERS = 40
_BERT_DEFAULT_WORKERS = 10
_BERT_MAX_WORKERS = 20


def result_queue_capacity(*, local_workers: int, remote_workers: int) -> int:
    """Bound completed trajectory buffering to two waves of active slots.

    Each result may carry a large game record, so an unbounded queue can turn
    a slow shard writer into a parent-process memory leak.  Producers now
    block after roughly two results per local/remote execution slot.
    """

    slots = max(1, max(0, int(local_workers)) + max(0, int(remote_workers)))
    return 2 * slots


class _SpillableResultQueue:
    """Bound heavy result payloads in RAM and spill overflow to local disk.

    Remote request threads must be able to acknowledge a completed trajectory
    and submit their next game even while the main shard consumer is briefly
    slower.  Blocking those threads on the in-memory result queue drains the
    *remote* process queues to zero.  Making that queue unbounded recreates the
    parent-process memory leak this module previously fixed.

    This handoff keeps at most ``memory_capacity`` full payloads in RAM.  Extra
    payloads are pickled into a per-dispatch directory and the metadata queue
    carries only their small paths.  Disk overflow has its own hard byte cap;
    once reached, normal producer backpressure resumes.  Thus refill can run
    independently on Elmo and Bert without an unbounded RAM or disk queue.
    """

    def __init__(
        self,
        *,
        memory_capacity: int,
        spool_root: Optional[Path] = None,
        max_spool_bytes: Optional[int] = None,
    ) -> None:
        self.memory_capacity = max(1, int(memory_capacity))
        self._memory_slots = threading.BoundedSemaphore(self.memory_capacity)
        self._items: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._state = threading.Condition()
        self._closed = False
        self._memory_items = 0
        self._spool_bytes = 0
        self._spool_files = 0
        self._spilled_total = 0
        self._spool_peak_bytes = 0
        self._sequence = 0

        if max_spool_bytes is None:
            try:
                max_gb = float(
                    os.environ.get("POKEBOT_REMOTE_RESULT_SPOOL_MAX_GB", "16")
                    or "16"
                )
            except (TypeError, ValueError):
                max_gb = 16.0
            max_spool_bytes = int(max(0.25, max_gb) * (1024**3))
        self.max_spool_bytes = max(1024 * 1024, int(max_spool_bytes))

        if spool_root is None:
            configured = os.environ.get("POKEBOT_REMOTE_RESULT_SPOOL_DIR", "").strip()
            spool_root = (
                Path(configured).expanduser()
                if configured
                else Path(tempfile.gettempdir()) / "pokebot-remote-results"
            )
        root = Path(spool_root)
        root.mkdir(parents=True, exist_ok=True)
        self.spool_dir = Path(
            tempfile.mkdtemp(prefix=f"dispatch-{os.getpid()}-", dir=str(root))
        )

    def put(self, item: tuple[str, Any], timeout: Optional[float] = None) -> None:
        tag, payload = item
        if tag != "ok":
            self._items.put(("control", item))
            return

        if self._memory_slots.acquire(blocking=False):
            with self._state:
                if self._closed:
                    self._memory_slots.release()
                    raise queue.Full
                self._memory_items += 1
            self._items.put(("memory", item))
            return

        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._state:
            while self._spool_bytes >= self.max_spool_bytes and not self._closed:
                if deadline is None:
                    self._state.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise queue.Full
                self._state.wait(timeout=remaining)
            if self._closed:
                raise queue.Full

            # Serialize under the state lock.  That admits at most one payload
            # beyond the byte cap (its size is unknown until written), rather
            # than letting hundreds of producers overshoot simultaneously.
            self._sequence += 1
            path = self.spool_dir / f"result-{self._sequence:08d}.pickle"
            try:
                with path.open("wb") as fh:
                    pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
                size = int(path.stat().st_size)
            except Exception:
                path.unlink(missing_ok=True)
                raise
            self._spool_bytes += size
            self._spool_files += 1
            self._spilled_total += 1
            self._spool_peak_bytes = max(
                self._spool_peak_bytes, self._spool_bytes
            )
        self._items.put(("spool", (tag, path, size)))

    def get(self, block: bool = True, timeout: Optional[float] = None) -> tuple[str, Any]:
        storage, item = self._items.get(block=block, timeout=timeout)
        if storage == "control":
            return item
        if storage == "memory":
            with self._state:
                self._memory_items = max(0, self._memory_items - 1)
            self._memory_slots.release()
            return item

        tag, path, size = item
        try:
            with Path(path).open("rb") as fh:
                payload = pickle.load(fh)
        finally:
            Path(path).unlink(missing_ok=True)
            with self._state:
                self._spool_bytes = max(0, self._spool_bytes - int(size))
                self._spool_files = max(0, self._spool_files - 1)
                self._state.notify_all()
        return tag, payload

    def telemetry(self) -> dict[str, int]:
        with self._state:
            return {
                "memory_capacity": self.memory_capacity,
                "memory_items": self._memory_items,
                "spool_files": self._spool_files,
                "spool_bytes": self._spool_bytes,
                "spilled_total": self._spilled_total,
                "spool_peak_bytes": self._spool_peak_bytes,
                "max_spool_bytes": self.max_spool_bytes,
            }

    def close(self) -> None:
        with self._state:
            self._closed = True
            self._state.notify_all()
        shutil.rmtree(self.spool_dir, ignore_errors=True)


def _put_thread_result(
    out_q: Any,
    stop_event: threading.Event,
    item: tuple[str, Any],
) -> bool:
    """Put one producer result without becoming immortal after cancellation."""
    while not stop_event.is_set():
        try:
            out_q.put(item, timeout=0.1)
            return True
        except queue.Full:
            continue
    return False


def _local_batch_is_tail_straggled(batch_size: int, outstanding: int) -> bool:
    """Return whether a fixed local batch is down to its final stragglers."""
    size = max(0, int(batch_size))
    left = max(0, int(outstanding))
    threshold = max(2, (size + 19) // 20)
    return bool(size > 0 and 0 < left <= threshold)


def _local_tail_override_ready(
    *,
    emitter_done: bool,
    batch_size: int,
    outstanding: int,
    last_progress_mono: float,
    now_mono: float,
    stale_s: float,
) -> bool:
    """Allow immediate done handoff, or a near-tail batch stale past grace."""
    if bool(emitter_done):
        return True
    if not _local_batch_is_tail_straggled(batch_size, outstanding):
        return False
    age_s = max(0.0, float(now_mono) - float(last_progress_mono))
    return age_s >= max(0.0, float(stale_s))


class _EndpointClaimCredits:
    """Bound shared-deque claims while preserving work for each endpoint.

    ``inflight`` counts jobs claimed by an endpoint but not yet completed.
    Healthy endpoints with live emitters receive a bounded credit target; a
    faster peer may refill its own credit but cannot consume another
    endpoint's deficit.  Reservations deliberately stop in the final 5% of a
    wave, where avoiding stranded tail work matters more than steady-state
    queue depth.

    The caller serializes this object with the dispatcher's claim lock.  It is
    intentionally bookkeeping-only: no health probe, connect, submit, or
    other network operation belongs in these methods.
    """

    def __init__(self, *, total_jobs: int) -> None:
        total = max(0, int(total_jobs))
        self.tail_jobs = max(1, (total + 19) // 20) if total > 0 else 0
        self.targets: dict[str, int] = {}
        self.inflight: dict[str, int] = {}
        self.claimed_total: dict[str, int] = {}
        self.live_claimants: dict[str, int] = {}
        self.healthy: dict[str, bool] = {}

    def register(self, endpoint: str) -> None:
        ep = str(endpoint)
        self.live_claimants[ep] = int(self.live_claimants.get(ep, 0)) + 1
        self.inflight.setdefault(ep, 0)
        self.claimed_total.setdefault(ep, 0)
        self.healthy[ep] = True

    def unregister(self, endpoint: str) -> None:
        ep = str(endpoint)
        live = max(0, int(self.live_claimants.get(ep, 0)) - 1)
        self.live_claimants[ep] = live
        if live <= 0:
            # No claimant can use this endpoint's unused allowance.  Treat it
            # as done so healthy peers may spill that credit rather than
            # strand the deque.
            self.healthy[ep] = False

    def set_healthy(self, endpoint: str, healthy: bool) -> None:
        ep = str(endpoint)
        self.healthy[ep] = bool(healthy) and int(
            self.live_claimants.get(ep, 0)
        ) > 0

    def set_target(self, endpoint: str, target: int) -> None:
        self.targets[str(endpoint)] = max(0, int(target))

    def note_claimed(self, endpoint: str, n: int) -> None:
        ep = str(endpoint)
        count = max(0, int(n))
        self.inflight[ep] = int(self.inflight.get(ep, 0)) + count
        self.claimed_total[ep] = int(self.claimed_total.get(ep, 0)) + count

    def note_finished(self, endpoint: str, n: int = 1) -> None:
        ep = str(endpoint)
        count = max(0, int(n))
        have = max(0, int(self.inflight.get(ep, 0)))
        if count > have:
            raise RuntimeError(
                f"remote endpoint credit underflow for {ep}: "
                f"finished={count} inflight={have}"
            )
        self.inflight[ep] = have - count

    def _eligible_endpoints(self) -> list[str]:
        return sorted(
            ep
            for ep, target in self.targets.items()
            if int(target) > 0
            and int(self.live_claimants.get(ep, 0)) > 0
            and bool(self.healthy.get(ep, False))
        )

    def _effective_targets(self, remaining_count: int) -> dict[str, int]:
        """Scale targets only when too little work remains to fill them all."""
        endpoints = self._eligible_endpoints()
        if not endpoints:
            return {}
        raw = {ep: max(0, int(self.targets.get(ep, 0))) for ep in endpoints}
        total_target = sum(raw.values())
        pending = max(0, int(remaining_count)) + sum(
            max(0, int(self.inflight.get(ep, 0))) for ep in endpoints
        )
        if pending >= total_target:
            return raw
        if pending <= 0 or total_target <= 0:
            return {ep: 0 for ep in endpoints}

        # When less than one full combined credit remains, use cumulative
        # weighted entitlement.  This both avoids a small-wave deadlock and
        # remembers that a fast endpoint already consumed most of a long wave:
        # the scarce final pre-tail jobs refill the under-served endpoint.
        claimed = {
            ep: max(0, int(self.claimed_total.get(ep, 0))) for ep in endpoints
        }
        future_total = sum(claimed.values()) + max(0, int(remaining_count))
        desired = {
            ep: (future_total * raw[ep]) // total_target
            for ep in endpoints
        }
        left = future_total - sum(desired.values())
        order = sorted(
            endpoints,
            key=lambda ep: (
                -((future_total * raw[ep]) % total_target),
                ep,
            ),
        )
        for ep in order[:left]:
            desired[ep] += 1
        return {
            ep: min(
                raw[ep],
                max(0, int(self.inflight.get(ep, 0)))
                + max(0, desired[ep] - claimed[ep]),
            )
            for ep in endpoints
        }

    def claim_limit(
        self,
        endpoint: str,
        *,
        want: int,
        remaining_count: int,
        wave_remaining: int,
    ) -> int:
        """Return this endpoint's safe claim without consuming peer credit."""
        requested = min(max(0, int(want)), max(0, int(remaining_count)))
        if requested <= 0:
            return 0
        # The allocation tail is an explicit correctness/throughput override.
        if max(0, int(wave_remaining)) <= self.tail_jobs:
            return requested

        ep = str(endpoint)
        effective = self._effective_targets(remaining_count)
        if ep not in effective:
            # Unhealthy/done endpoints own no reserved work.  A still-running
            # socket waits for a successful data result or health probe to
            # restore eligibility; healthy peers may use the released credit.
            return 0
        own_room = max(
            0, int(effective[ep]) - max(0, int(self.inflight.get(ep, 0)))
        )
        protected_for_peers = sum(
            max(
                0,
                int(target) - max(0, int(self.inflight.get(other, 0))),
            )
            for other, target in effective.items()
            if other != ep
        )
        shared_room = max(0, int(remaining_count) - protected_for_peers)
        return min(requested, own_room, shared_room)


def expand_endpoint_specs(specs: list[str]) -> list[str]:
    """Accept space-separated nargs or a single comma-separated token."""
    out: list[str] = []
    for spec in specs:
        for part in str(spec).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


def _parse_endpoint_float_map(env_name: str) -> dict[str, float]:
    """Parse ``host[:port]=value,...`` maps from env."""
    raw = os.environ.get(env_name, "").strip()
    out: dict[str, float] = {}
    if not raw:
        return out
    for part in raw.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, _, val = part.partition("=")
        key = key.strip().lower()
        try:
            w = float(val.strip())
        except ValueError:
            continue
        out[key] = max(0.0, w)
        # Also index bare host when key includes port.
        if ":" in key:
            host_only, _, _port = key.rpartition(":")
            out.setdefault(host_only, max(0.0, w))
    return out


def _parse_endpoint_weight_map() -> dict[str, float]:
    """Parse ``POKEBOT_REMOTE_ENDPOINT_WEIGHTS`` as ``host[:port]=w,...``."""
    return _parse_endpoint_float_map("POKEBOT_REMOTE_ENDPOINT_WEIGHTS")


def _parse_endpoint_int_map(env_name: str) -> dict[str, int]:
    raw_map = _parse_endpoint_float_map(env_name)
    return {k: max(0, int(round(v))) for k, v in raw_map.items()}


def _lookup_endpoint_int(
    mapping: dict[str, int], host: str, port: Optional[int] = None
) -> Optional[int]:
    host_l = str(host).strip().lower()
    if port is not None:
        keyed = f"{host_l}:{int(port)}"
        if keyed in mapping:
            return int(mapping[keyed])
    if host_l in mapping:
        return int(mapping[host_l])
    return None


def endpoint_role(host: str) -> str:
    """Return ``elmo`` / ``bert`` / ``other`` for known LAN remotes."""
    host_l = str(host).strip().lower()
    if host_l in _ELMO_HOSTS or host_l.startswith("192.168.1.143"):
        return "elmo"
    if host_l in _BERT_HOSTS:
        return "bert"
    return "other"


def endpoint_default_workers(host: str, port: Optional[int] = None) -> int:
    """Steady-state demand floor (not the ceiling).

    Override via ``POKEBOT_REMOTE_DEFAULT_WORKERS`` e.g.
    ``192.168.1.143:8765=20,bert.local:8766=10``.
    """
    override = _lookup_endpoint_int(
        _parse_endpoint_int_map("POKEBOT_REMOTE_DEFAULT_WORKERS"), host, port
    )
    if override is not None:
        return max(1, int(override))
    role = endpoint_role(host)
    if role == "elmo":
        return _ELMO_DEFAULT_WORKERS
    if role == "bert":
        return _BERT_DEFAULT_WORKERS
    return 8


def endpoint_max_workers(host: str, port: Optional[int] = None) -> int:
    """Hard demand ceiling (scheduler may grow up to this).

    Override via ``POKEBOT_REMOTE_MAX_WORKERS`` e.g.
    ``192.168.1.143:8765=40,bert.local:8766=20``.
    """
    override = _lookup_endpoint_int(
        _parse_endpoint_int_map("POKEBOT_REMOTE_MAX_WORKERS"), host, port
    )
    if override is not None:
        return max(1, int(override))
    role = endpoint_role(host)
    if role == "elmo":
        return _ELMO_MAX_WORKERS
    if role == "bert":
        return _BERT_MAX_WORKERS
    return 16


def endpoint_dispatch_chunk(
    host: str,
    port: Optional[int] = None,
    *,
    default: int,
) -> int:
    """Return the endpoint's total shared-queue backlog target.

    This is *not* a LAN/RPC batch: every reserved job is still submitted as a
    separate request.  The dispatcher divides this total across the endpoint's
    live sockets, preventing the first socket threads from hoarding the whole
    backlog while preserving a useful ready queue per remote host.
    """
    override = _lookup_endpoint_int(
        _parse_endpoint_int_map("POKEBOT_REMOTE_ENDPOINT_CHUNKS"), host, port
    )
    return max(1, int(default if override is None else override))


def remote_socket_prefetch_factor() -> int:
    """Return request sockets opened per remote execution worker.

    Production starts with one executing wave plus four server-queued waves.
    A separate bounded reserve is admitted only if server telemetry reports
    that the local queue fell below its low-water mark.  The additional
    sockets do not add simulator processes, policy leaves, or model replicas.
    """
    try:
        value = int(os.environ.get("POKEBOT_REMOTE_SOCKET_PREFETCH", "1") or "1")
    except (TypeError, ValueError):
        value = 1
    return max(1, min(8, value))


def remote_socket_prefetch_max_factor() -> int:
    """Return the hard per-worker request-socket ceiling for low-water refill."""
    base = remote_socket_prefetch_factor()
    try:
        value = int(
            os.environ.get("POKEBOT_REMOTE_SOCKET_PREFETCH_MAX", str(base))
            or str(base)
        )
    except (TypeError, ValueError):
        value = base
    return max(base, min(8, value))


def remote_socket_target(execution_workers: int) -> int:
    """Map scheduled execution-worker demand to queued request sockets."""
    return max(0, int(execution_workers)) * remote_socket_prefetch_factor()


def remote_socket_max_target(execution_workers: int) -> int:
    """Map execution demand to the bounded low-water reserve ceiling."""
    return max(0, int(execution_workers)) * remote_socket_prefetch_max_factor()


def remote_queue_low_water_fraction() -> float:
    """Fraction of the desired server-side waiting queue that triggers refill."""
    try:
        value = float(
            os.environ.get("POKEBOT_REMOTE_QUEUE_LOW_WATER_FRAC", "0.5") or "0.5"
        )
    except (TypeError, ValueError):
        value = 0.5
    return max(0.1, min(0.9, value))


def remote_queue_probe_interval_s() -> float:
    """Minimum interval between parallel Elmo/Bert queue-depth probes."""
    try:
        value = float(
            os.environ.get("POKEBOT_REMOTE_QUEUE_PROBE_S", "2") or "2"
        )
    except (TypeError, ValueError):
        value = 2.0
    return max(0.2, min(30.0, value))


def remote_refill_games_per_socket() -> int:
    """Return the private game reservation made by every socket refill."""
    try:
        value = int(os.environ.get("POKEBOT_REMOTE_REFILL_GAMES", "1") or "1")
    except (TypeError, ValueError):
        value = 1
    return max(1, min(64, value))


def endpoint_dispatch_weight(host: str, port: Optional[int] = None) -> float:
    """Per-endpoint dispatch weight (default 1.0).

    Override via ``POKEBOT_REMOTE_ENDPOINT_WEIGHTS`` e.g.
    ``192.168.1.143:8765=1.0,bert.local:8766=0.4``.

    When demand scheduling is active, slot counts use demand/max caps instead;
    weights remain for legacy additive dispatch / multi-trainer sharing.
    """
    weights = _parse_endpoint_weight_map()
    host_l = str(host).strip().lower()
    if port is not None:
        keyed = f"{host_l}:{int(port)}"
        if keyed in weights:
            return float(weights[keyed])
    if host_l in weights:
        return float(weights[host_l])
    return 1.0


def weighted_slot_count(advertised_workers: int, weight: float) -> int:
    """Scale advertised worker sockets by dispatch weight (0 → skip endpoint)."""
    n = max(0, int(advertised_workers))
    w = max(0.0, float(weight))
    if n <= 0 or w <= 0.0:
        return 0
    return max(1, int(round(n * w)))


def remote_capacity_workers(info: "RemoteWorkerInfo") -> int:
    """Process-pool capacity (= max) from hello; falls back to ``workers``."""
    adv = max(0, int(info.workers))
    mx = max(0, int(getattr(info, "max_workers", 0) or 0))
    return max(adv, mx) if mx > 0 else adv


def demand_slot_count(
    *,
    capacity: int,
    demand: int,
    max_cap: int,
) -> int:
    """In-flight sockets: ``min(capacity, max_cap, demand)`` (0 if any side 0)."""
    cap = max(0, int(capacity))
    dem = max(0, int(demand))
    hi = max(0, int(max_cap))
    if cap <= 0 or dem <= 0 or hi <= 0:
        return 0
    return max(1, min(cap, hi, dem))


def weighted_remote_capacity(
    clients: list["RemoteJobClient"],
    *,
    demand_by_endpoint: Optional[dict[str, int]] = None,
) -> int:
    """Sum of in-flight slots across connected farm clients.

    With ``demand_by_endpoint``, uses scheduler demand clamped by per-host max
    and hello capacity (max ≫ default). Otherwise legacy weight×advertised.
    """
    total = 0
    for client in clients:
        if client.info is None:
            continue
        if demand_by_endpoint is not None:
            capacity = remote_capacity_workers(client.info)
            max_cap = endpoint_max_workers(client.host, client.port)
            demand = int(
                demand_by_endpoint.get(
                    client.endpoint,
                    endpoint_default_workers(client.host, client.port),
                )
            )
            total += demand_slot_count(
                capacity=capacity, demand=demand, max_cap=max_cap
            )
            continue
        w = endpoint_dispatch_weight(client.host, client.port)
        total += weighted_slot_count(int(client.info.workers), w)
    return total


def describe_endpoint_weights(
    clients: list["RemoteJobClient"],
    *,
    demand_by_endpoint: Optional[dict[str, int]] = None,
) -> list[dict[str, Any]]:
    """Per-endpoint weight/slot summary for collect-start logs."""
    rows: list[dict[str, Any]] = []
    for client in clients:
        adv = int(client.info.workers) if client.info is not None else 0
        capacity = (
            remote_capacity_workers(client.info) if client.info is not None else 0
        )
        w = endpoint_dispatch_weight(client.host, client.port)
        default = endpoint_default_workers(client.host, client.port)
        max_cap = endpoint_max_workers(client.host, client.port)
        if demand_by_endpoint is not None:
            demand = int(demand_by_endpoint.get(client.endpoint, default))
            slots = demand_slot_count(
                capacity=capacity, demand=demand, max_cap=max_cap
            )
        else:
            demand = default
            slots = weighted_slot_count(adv, w)
        rows.append(
            {
                "endpoint": client.endpoint,
                "host": client.host,
                "weight": w,
                "advertised_workers": adv,
                "capacity_workers": capacity,
                "default_workers": default,
                "max_workers": max_cap,
                "demand_workers": demand,
                "dispatch_slots": slots,
            }
        )
    return rows


def _smb_checkpoint_dir() -> Path | None:
    explicit = os.environ.get("POKEBOT_TRUENAS_CHECKPOINT_SMB")
    if explicit:
        path = Path(explicit)
        return path if path.is_dir() else None
    uid = os.getuid()
    # Host Docker compose mounts containers/truenas-worker/checkpoint →
    # /workspace/checkpoint (NOT the top-level poke-bot-agent/checkpoint/).
    candidates = [
        Path(
            f"/run/user/{uid}/gvfs/smb-share:server=truenas.local,"
            "share=main/poke-bot-agent/containers/truenas-worker/checkpoint"
        ),
        Path(
            f"/run/user/{uid}/gvfs/smb-share:server=truenas.local,"
            "share=main/poke-bot-agent/checkpoint"
        ),
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def _bert_sftp_root() -> Path | None:
    """gvfs SFTP mount of bert's native checkout (no interactive SSH)."""
    explicit = os.environ.get("POKEBOT_BERT_STAGE_ROOT")
    if explicit:
        path = Path(explicit)
        return path if path.is_dir() else None
    uid = os.getuid()
    # Absolute ``/Users/...`` under the sftp gvfs root mirrors bert's FS.
    candidate = Path(
        f"/run/user/{uid}/gvfs/sftp:host=bert.local{_BERT_ROOT}"
    )
    return candidate if candidate.is_dir() else None


def _needs_stage(src: Path, dest: Path) -> bool:
    if not dest.is_file():
        return True
    try:
        return (
            dest.stat().st_size != src.stat().st_size
            or int(dest.stat().st_mtime) < int(src.stat().st_mtime)
        )
    except OSError:
        return True


def digest_addressed_basename(src: Path, digest: Optional[str] = None) -> str:
    """Stable remote filename that cannot collide across distinct checkpoint bytes.

    Elmo only bind-mounts a flat ``checkpoint/`` dir. Staging as bare
    ``iter_00001.pt`` overwrote prior digests and broke pin/reload on the
    long-lived worker (expected old sha, file had new sha).
    """
    from .checkpoint import checkpoint_digest

    resolved = Path(src).expanduser()
    dig = digest or checkpoint_digest(resolved)
    short = str(dig).split(":", 1)[-1][:16]
    if not short:
        raise RemoteJobsError(f"empty checkpoint digest for {resolved}")
    return f"{resolved.stem}.{short}{resolved.suffix}"


def _gvfs_safe_copy(src: Path, dest: Path) -> None:
    """Copy bytes onto gvfs SMB/SFTP (os.replace often fails with Errno 95)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f"pokebot_remote_stage_{os.getpid()}_",
        suffix=src.suffix,
        dir="/tmp",
    )
    os.close(descriptor)
    local_tmp = Path(temporary)
    try:
        shutil.copy2(src, local_tmp)
        with open(local_tmp, "rb") as src_f, open(dest, "wb") as dst_f:
            shutil.copyfileobj(src_f, dst_f, length=16 * 1024 * 1024)
    finally:
        try:
            local_tmp.unlink(missing_ok=True)
        except TypeError:
            if local_tmp.exists():
                local_tmp.unlink()


def _bert_remote_digest(remote_native: Path) -> Optional[str]:
    """Return Bert's SHA-256 for one staged file, or ``None`` when absent."""

    command = (
        f"test -f {shlex.quote(remote_native.as_posix())} && "
        f"shasum -a 256 -- {shlex.quote(remote_native.as_posix())}"
    )
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "StrictHostKeyChecking=accept-new",
            _BERT_SSH,
            command,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    token = (result.stdout.strip().split() or [""])[0].lower()
    if len(token) != 64 or any(ch not in "0123456789abcdef" for ch in token):
        return None
    return "sha256:" + token


def _matchup_runtime_companions() -> tuple[Path, bytes] | None:
    """Return the canonical routing tree and strict remote activation marker.

    Remote checkpoints are content-addressed and can live in a different
    directory from the worker's startup checkpoint.  The worker intentionally
    requires the marker and tree beside every reloaded checkpoint, so staging
    the checkpoint without these files makes a safe runtime reload fail.
    """

    if not _env_truthy("POKEBOT_MATCHUP_ADAPTER_RUNTIME"):
        return None
    tree = Path(
        os.environ.get("POKEBOT_PUBLIC_MATCHUP_TREE_PATH", "")
    ).expanduser()
    if not tree.is_file():
        raise RemoteJobsError(
            "matchup runtime staging requires a readable canonical tree"
        )
    raw = tree.read_bytes()
    try:
        tree_payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RemoteJobsError(
            f"matchup runtime tree is invalid JSON: {tree}"
        ) from exc
    runtime = dict(tree_payload.get("runtime_contract") or {})
    accepted = sorted(
        str(value) for value in runtime.get("accepted_archetype_ids") or ()
    )
    if not (
        accepted
        and runtime.get("one_route_per_decision") is True
        and runtime.get("unknown_route_exact_bypass") is True
        and int(runtime.get("consecutive_required") or 0) >= 1
    ):
        raise RemoteJobsError(
            "matchup runtime tree lacks its enabled routing contract"
        )
    binding_keys = (
        "adapter_format",
        "route_target_ids",
        "route_physical_slots",
        "physical_slot_capacity",
        "slot_registry_digest",
    )
    if (
        runtime.get("adapter_format") == "poke-bot-matchup-adapter-bank-v6"
        and any(key not in runtime for key in binding_keys)
    ):
        raise RemoteJobsError(
            "V6 matchup runtime tree lacks its registry/slot binding"
        )
    payload = {
        "schema": _MATCHUP_RUNTIME_MARKER_SCHEMA,
        "runtime_enabled": True,
        "tree_file": tree.name,
        "tree_digest": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "accepted_archetype_ids": accepted,
        "continuous_reevaluation": True,
        "one_route_per_decision": True,
        "zero_materialized_adapters_allowed": bool(
            runtime.get("zero_materialized_adapters_allowed")
        ),
        **{
            key: runtime[key]
            for key in binding_keys
            if key in runtime
        },
    }
    marker = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    return tree, marker


def _stage_bert_runtime_companions(remote_dir: Path) -> None:
    companions = _matchup_runtime_companions()
    if companions is None:
        return
    tree, marker = companions
    tree_digest = "sha256:" + hashlib.sha256(tree.read_bytes()).hexdigest()
    tree_remote = remote_dir / tree.name
    if _bert_remote_digest(tree_remote) != tree_digest:
        _rsync_to_bert(tree, tree_remote, digest=tree_digest)

    descriptor, temporary = tempfile.mkstemp(
        prefix="pokebot-matchup-runtime-marker-",
        suffix=".json",
        dir="/tmp",
    )
    marker_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(marker)
            stream.flush()
            os.fsync(stream.fileno())
        marker_digest = "sha256:" + hashlib.sha256(marker).hexdigest()
        marker_remote = remote_dir / _MATCHUP_RUNTIME_MARKER_NAME
        if _bert_remote_digest(marker_remote) != marker_digest:
            _rsync_to_bert(marker_path, marker_remote, digest=marker_digest)
    finally:
        marker_path.unlink(missing_ok=True)


def _stage_elmo_runtime_companions(checkpoint_dir: Path) -> None:
    companions = _matchup_runtime_companions()
    if companions is None:
        return
    tree, marker = companions
    tree_dest = checkpoint_dir / tree.name
    tree_digest = "sha256:" + hashlib.sha256(tree.read_bytes()).hexdigest()
    if (
        not tree_dest.is_file()
        or "sha256:" + hashlib.sha256(tree_dest.read_bytes()).hexdigest()
        != tree_digest
    ):
        _gvfs_safe_copy(tree, tree_dest)

    marker_dest = checkpoint_dir / _MATCHUP_RUNTIME_MARKER_NAME
    if not marker_dest.is_file() or marker_dest.read_bytes() != marker:
        descriptor, temporary = tempfile.mkstemp(
            prefix="pokebot-matchup-runtime-marker-",
            suffix=".json",
            dir="/tmp",
        )
        marker_path = Path(temporary)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(marker)
                stream.flush()
                os.fsync(stream.fileno())
            _gvfs_safe_copy(marker_path, marker_dest)
        finally:
            marker_path.unlink(missing_ok=True)


def _rsync_to_bert(src: Path, remote_native: Path, *, digest: str) -> None:
    """Atomically stage and checksum ``src`` on Bert via BatchMode rsync."""
    remote_dir = remote_native.parent.as_posix()
    remote_partial = remote_native.with_name(
        remote_native.name + f".partial.{os.getpid()}"
    )
    ssh_base = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "StrictHostKeyChecking=accept-new",
        _BERT_SSH,
    ]
    mkdir = subprocess.run(
        [*ssh_base, f"mkdir -p {shlex.quote(remote_dir)}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if mkdir.returncode != 0:
        raise RemoteJobsError(
            f"bert ssh mkdir failed ({_BERT_SSH}:{remote_dir}): "
            f"{mkdir.stderr.strip() or mkdir.stdout.strip() or mkdir.returncode}"
        )
    rsync = subprocess.run(
        [
            "rsync",
            "-a",
            "--partial",
            "-e",
            "ssh -o BatchMode=yes -o ConnectTimeout=10 "
            "-o StrictHostKeyChecking=accept-new",
            str(src),
            f"{_BERT_SSH}:{remote_partial.as_posix()}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if rsync.returncode != 0:
        raise RemoteJobsError(
            f"bert rsync failed ({src.name} -> {_BERT_SSH}:{remote_partial}): "
            f"{rsync.stderr.strip() or rsync.stdout.strip() or rsync.returncode}"
        )
    expected = str(digest).split(":", 1)[-1].lower()
    publish = subprocess.run(
        [
            *ssh_base,
            "set -e; "
            f"actual=$(shasum -a 256 -- {shlex.quote(remote_partial.as_posix())} | "
            "awk '{print $1}'); "
            f"test \"$actual\" = {shlex.quote(expected)}; "
            f"mv -f -- {shlex.quote(remote_partial.as_posix())} "
            f"{shlex.quote(remote_native.as_posix())}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if publish.returncode != 0:
        raise RemoteJobsError(
            f"bert checkpoint checksum/publish failed ({remote_native}): "
            f"{publish.stderr.strip() or publish.stdout.strip() or publish.returncode}"
        )


def _stage_bert_checkpoint(src: Path) -> str:
    """Ensure digest-named .pt bytes exist on bert; return path bert opens."""
    if not src.is_file():
        raise RemoteJobsError(f"local checkpoint missing for bert stage: {src}")
    try:
        rel = src.relative_to(_TRAIN_ROOT)
    except ValueError as exc:
        raise RemoteJobsError(
            f"bert path remap requires path under {_TRAIN_ROOT}, got {src}"
        ) from exc
    from .checkpoint import checkpoint_digest

    digest = checkpoint_digest(src)
    remote_native = (_BERT_ROOT / rel).with_name(
        digest_addressed_basename(src, digest=digest)
    )
    cache_key = (str(src), digest)
    # Hundreds of remote request threads may prepare the same heldout job at
    # once.  The old per-job mapping launched one rsync per thread, creating a
    # process/memory storm.  Serialize the rare content-addressed publish and
    # cache it for the lifetime of this trainer process.
    with _BERT_STAGE_LOCK:
        cached = _BERT_STAGE_CACHE.get(cache_key)
        if cached is not None:
            _stage_bert_runtime_companions(remote_native.parent)
            return cached
        if _bert_remote_digest(remote_native) == digest:
            _stage_bert_runtime_companions(remote_native.parent)
            result = str(remote_native)
            _BERT_STAGE_CACHE[cache_key] = result
            return result
        sftp_root = _bert_sftp_root()
        try:
            _rsync_to_bert(src, remote_native, digest=digest)
        except RemoteJobsError as rsync_exc:
            if sftp_root is None:
                raise
            dest = sftp_root / remote_native.relative_to(_BERT_ROOT)
            try:
                _gvfs_safe_copy(src, dest)
            except OSError as copy_exc:
                raise RemoteJobsError(
                    f"bert stage failed via rsync ({rsync_exc}) and gvfs SFTP "
                    f"({copy_exc})"
                ) from copy_exc
            if checkpoint_digest(dest) != digest:
                raise RemoteJobsError("bert gvfs checkpoint digest mismatch")
        if _bert_remote_digest(remote_native) != digest and sftp_root is None:
            raise RemoteJobsError("bert checkpoint publish did not verify")
        _stage_bert_runtime_companions(remote_native.parent)
        result = str(remote_native)
        _BERT_STAGE_CACHE[cache_key] = result
        return result


def resolve_remote_checkpoint_path(host: str, local_path: str) -> str:
    """Map a trainer-local .pt path to a path the remote process can open.

    Elmo (TrueNAS host Docker): stage bytes onto the SMB ``checkpoint/``
    bind-mount and reload ``/workspace/checkpoint/<basename>``.
    Bert: stage bytes onto the native checkout (gvfs SFTP or SSH rsync), then
    remap the training-box repo root onto ``/Users/tsinzitari/workspace/...``.
    """
    raw = str(local_path)
    if raw.startswith("/workspace/checkpoint/"):
        return raw
    src = Path(raw).expanduser().resolve()
    host_l = host.strip().lower()
    if host_l in _ELMO_HOSTS:
        smb = _smb_checkpoint_dir()
        if smb is None:
            raise RemoteJobsError(
                "TrueNAS SMB checkpoint dir not mounted "
                "(open smb://truenas.local/main then retry)"
            )
        if not src.is_file():
            raise RemoteJobsError(f"local checkpoint missing for stage: {src}")
        # Digest-addressed basename: pins/reloads stay valid across iters/runs.
        from .checkpoint import checkpoint_digest

        digest = checkpoint_digest(src)
        dest_name = digest_addressed_basename(src, digest=digest)
        dest = smb / dest_name
        cache_key = (str(src), digest, str(smb))
        with _ELMO_STAGE_LOCK:
            cached = _ELMO_STAGE_CACHE.get(cache_key)
            if cached is not None:
                _stage_elmo_runtime_companions(smb)
                return cached
            destination_valid = False
            if dest.is_file() and dest.stat().st_size == src.stat().st_size:
                try:
                    destination_valid = checkpoint_digest(dest) == digest
                except OSError:
                    destination_valid = False
            if not destination_valid:
                _gvfs_safe_copy(src, dest)
            if checkpoint_digest(dest) != digest:
                raise RemoteJobsError("Elmo gvfs checkpoint digest mismatch")
            _stage_elmo_runtime_companions(smb)
            result = f"/workspace/checkpoint/{dest_name}"
            _ELMO_STAGE_CACHE[cache_key] = result
            return result
    if host_l in _BERT_HOSTS:
        return _stage_bert_checkpoint(src)
    return str(src)


def resolve_remote_workdir_path(host: str, local_path: str) -> str:
    """Map a trainer-local repo path onto the remote checkout (no staging).

    Used for baseline ``spec.path`` (and similar) so bert/elmo workers open
    their native trees instead of ``/home/inzi/poke-bot-agent/...``.
    """
    raw = str(local_path)
    host_l = host.strip().lower()
    if host_l in _BERT_HOSTS and raw.startswith(str(_BERT_ROOT)):
        return raw
    if host_l in _ELMO_HOSTS and raw.startswith("/workspace/"):
        return raw
    try:
        resolved = Path(raw).expanduser().resolve()
    except OSError:
        resolved = Path(raw).expanduser()
    try:
        rel = resolved.relative_to(_TRAIN_ROOT)
    except ValueError:
        return raw
    if host_l in _BERT_HOSTS:
        return str(_BERT_ROOT / rel)
    if host_l in _ELMO_HOSTS:
        return f"/workspace/{rel.as_posix()}"
    return raw


def prepare_remote_play_job(host: str, job: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy a play/promotion job with host-native filesystem paths."""
    out = copy.deepcopy(job)
    host_l = host.strip().lower()
    mapped_checkpoints: dict[str, str] = {}
    for field_name in (
        "checkpoint",
        "opponent_checkpoint",
        "candidate_checkpoint",
        "parent_checkpoint",
    ):
        checkpoint = out.get(field_name)
        if not checkpoint:
            continue
        raw = str(checkpoint)
        mapped = mapped_checkpoints.get(raw)
        if mapped is None:
            # Elmo only bind-mounts ``/workspace/checkpoint/`` (not the full
            # repo), so every checkpoint used by either seat must be staged.
            # Bert keeps a full checkout; remapping both seats to that checkout
            # is sufficient and avoids an rsync on every submitted game.
            if host_l in _ELMO_HOSTS or host_l in _BERT_HOSTS:
                mapped = resolve_remote_checkpoint_path(host, raw)
            else:
                mapped = raw
            mapped_checkpoints[raw] = mapped
        out[field_name] = mapped
    spec = out.get("spec")
    if isinstance(spec, dict) and spec.get("path"):
        spec["path"] = resolve_remote_workdir_path(host, str(spec["path"]))
    return out


class RemoteJobsError(RuntimeError):
    """Protocol or transport failure talking to a remote worker."""


class RemoteResultError(RemoteJobsError):
    """A well-formed remote reply whose game result is unusable.

    Keeping semantic failures in the same exception family as transport
    failures lets the existing additive collectors retry locally (or fail
    closed when local fallback is disabled) before remote completion metrics
    are incremented.
    """

    def __init__(self, reason: str, result: dict[str, Any]) -> None:
        super().__init__(f"remote result failed: {reason}")
        self.reason = str(reason)
        self.result = result


_REMOTE_RESULT_FAILURE_FLAGS = (
    "our_failed",
    "resource_error",
    "cancelled",
    "trust_failure",
    "game_timeout",
    "search_budget_exhausted",
    "baseline_failed",
)


def remote_result_failure_reason(result: dict[str, Any]) -> Optional[str]:
    """Return why a remote game payload is unusable, or ``None`` if valid.

    Remote workers intentionally return game/runtime failures inside an
    otherwise successful ``{type: result, ok: true}`` protocol frame.  Those
    payloads must not look like productive remote games to demand scheduling.
    The helper is public so callers with stricter collection policy can choose
    explicit local retry or fail-closed behavior without duplicating flags.
    """
    failed_flags = [
        flag for flag in _REMOTE_RESULT_FAILURE_FLAGS if bool(result.get(flag))
    ]
    if result.get("ok") is False:
        failed_flags.append("ok=false")

    status = str(result.get("status") or "").strip().lower()
    if status in {"error", "failed", "failure"}:
        failed_flags.append(f"status={status}")

    result_type = str(result.get("type") or "").strip().lower()
    if result_type in {"error", "failed", "failure"}:
        failed_flags.append(f"type={result_type}")

    error = result.get("error")
    has_error = error not in (None, "", False, [], {})
    if not failed_flags and not has_error:
        return None

    reason = ", ".join(dict.fromkeys(failed_flags)) or "error payload"
    if has_error:
        reason = f"{reason}: {error}"
    return reason


def require_remote_result_success(result: dict[str, Any]) -> dict[str, Any]:
    """Return a usable result or raise :class:`RemoteResultError`.

    Call this before crediting a remote completion.  ``RemoteJobClient`` does
    so centrally, which preserves valid-job behavior while activating the
    collectors' existing local-fallback/fail-closed paths for semantic errors.
    """
    reason = remote_result_failure_reason(result)
    if reason is not None:
        raise RemoteResultError(reason, result)
    return result


def encode_frame(payload: dict[str, Any]) -> bytes:
    if _orjson is not None:
        try:
            # Match stdlib json's acceptance/stringification of integer keys.
            # The bytes are still ordinary UTF-8 JSON and remain cross-codec.
            body = _orjson.dumps(payload, option=_orjson.OPT_NON_STR_KEYS)
        except (TypeError, ValueError) as exc:
            raise RemoteJobsError(f"invalid JSON payload: {exc}") from exc
    else:
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    if len(body) > _MAX_FRAME:
        raise RemoteJobsError(f"frame too large: {len(body)} bytes")
    return _HDR.pack(len(body)) + body


def _recvexact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise RemoteJobsError("connection closed while reading frame")
        buf.extend(chunk)
    return bytes(buf)


def read_frame(sock: socket.socket) -> dict[str, Any]:
    header = _recvexact(sock, _HDR.size)
    (length,) = _HDR.unpack(header)
    if length > _MAX_FRAME:
        raise RemoteJobsError(f"frame length {length} exceeds max {_MAX_FRAME}")
    body = _recvexact(sock, length)
    try:
        payload = (
            _orjson.loads(body)
            if _orjson is not None
            else json.loads(body.decode("utf-8"))
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RemoteJobsError(f"invalid JSON frame: {exc}") from exc
    if not isinstance(payload, dict):
        raise RemoteJobsError(f"frame root must be object, got {type(payload)!r}")
    return payload


def send_frame(sock: socket.socket, payload: dict[str, Any]) -> None:
    sock.sendall(encode_frame(payload))


@dataclass
class RemoteWorkerInfo:
    endpoint: str
    workers: int
    leaf_servers: int
    gpu_name: str
    device: str
    checkpoint_digest: Optional[str]
    hostname: str
    free_ram_gb: Optional[float] = None
    job_kinds: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    #: Process-pool capacity (max). 0 → treat ``workers`` as capacity.
    max_workers: int = 0
    #: Steady-state advertise hint from remote (optional).
    default_workers: int = 0
    #: Digest-verified runtime-router contract advertised by the worker. ``None``
    #: means the worker started without an adjacent activation marker and must
    #: not receive games when production matchup routing is required.
    matchup_runtime: Optional[dict[str, Any]] = None


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    return float(raw)


class RemoteJobClient:
    """One TCP session to a remote worker that already has leaf+sim ready."""

    def __init__(
        self,
        host: str,
        port: int = DEFAULT_PORT,
        *,
        timeout_s: float = 30.0,
        connect_timeout_s: Optional[float] = None,
        control_timeout_s: Optional[float] = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        # Data-plane bound for in-flight game result frames (often >> control).
        self.timeout_s = float(timeout_s)
        # Connect / hello / ping / reload / pin — keep separate so a slow game
        # socket timeout does not also force multi-minute control ops, and so
        # busy-but-alive peers are not false-killed on short default reads.
        self.connect_timeout_s = float(
            connect_timeout_s
            if connect_timeout_s is not None
            else _env_float("POKEBOT_REMOTE_CONNECT_TIMEOUT_S", 60.0)
        )
        self.control_timeout_s = float(
            control_timeout_s
            if control_timeout_s is not None
            else _env_float("POKEBOT_REMOTE_CONTROL_TIMEOUT_S", 300.0)
        )
        self._sock: Optional[socket.socket] = None
        self.info: Optional[RemoteWorkerInfo] = None

    @property
    def endpoint(self) -> str:
        return f"{self.host}:{self.port}"

    def connect(self) -> RemoteWorkerInfo:
        sock = socket.create_connection(
            (self.host, self.port), timeout=self.connect_timeout_s
        )
        sock.settimeout(self.control_timeout_s)
        self._sock = sock
        send_frame(
            sock,
            {
                "type": "hello",
                "proto": PROTO_VERSION,
                "client": "poke-bot-agent",
            },
        )
        reply = read_frame(sock)
        if reply.get("type") != "hello_ok" or int(reply.get("proto", -1)) != PROTO_VERSION:
            raise RemoteJobsError(f"unexpected hello reply: {reply!r}")
        raw_kinds = reply.get("job_kinds") or []
        if isinstance(raw_kinds, str):
            kinds = tuple(k.strip() for k in raw_kinds.split(",") if k.strip())
        else:
            kinds = tuple(str(k) for k in raw_kinds)
        raw_capabilities = reply.get("capabilities") or []
        if isinstance(raw_capabilities, str):
            capabilities = tuple(
                value.strip()
                for value in raw_capabilities.split(",")
                if value.strip()
            )
        else:
            capabilities = tuple(str(value) for value in raw_capabilities)
        workers = int(reply.get("workers", 0))
        max_workers = int(reply.get("max_workers", 0) or 0)
        default_workers = int(reply.get("default_workers", 0) or 0)
        self.info = RemoteWorkerInfo(
            endpoint=self.endpoint,
            workers=workers,
            leaf_servers=int(reply.get("leaf_servers", 0)),
            gpu_name=str(reply.get("gpu_name") or ""),
            device=str(reply.get("device") or ""),
            checkpoint_digest=reply.get("checkpoint_digest"),
            hostname=str(reply.get("hostname") or self.host),
            free_ram_gb=(
                float(reply["free_ram_gb"])
                if reply.get("free_ram_gb") is not None
                else None
            ),
            job_kinds=kinds,
            capabilities=capabilities,
            max_workers=max_workers,
            default_workers=default_workers,
            matchup_runtime=(
                copy.deepcopy(reply.get("matchup_runtime"))
                if isinstance(reply.get("matchup_runtime"), dict)
                else None
            ),
        )
        return self.info

    def close(self) -> None:
        sock = self._sock
        self._sock = None
        if sock is not None:
            try:
                send_frame(sock, {"type": "bye"})
            except Exception:
                pass
            try:
                sock.close()
            except Exception:
                pass

    def reconnect(self) -> RemoteWorkerInfo:
        """Drop any half-open socket and open a fresh hello session."""
        sock = self._sock
        self._sock = None
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass
        return self.connect()

    def ensure_alive(self) -> None:
        """Ping the session; reconnect only on real hangups.

        Farm template sockets sit idle across train/promo gaps. Older remote
        workers tear idle sessions down after ``idle_timeout_s``, which surfaces
        as ``connection closed while reading frame`` on the next wave. A cheap
        ping/reconnect here heals templates before slot clones are opened.

        A bare ``TimeoutError`` means the peer is slow/backlogged, not dead —
        do **not** tear down a healthy-but-busy Elmo session.
        """
        try:
            self.ping()
        except TimeoutError:
            return
        except (OSError, RemoteJobsError) as exc:
            if _is_remote_hangup_error(exc) or isinstance(exc, RemoteJobsError):
                self.reconnect()
                return
            raise

    def __enter__(self) -> "RemoteJobClient":
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _require_sock(self) -> socket.socket:
        if self._sock is None:
            raise RemoteJobsError("not connected")
        return self._sock

    def ping(self) -> dict[str, Any]:
        sock = self._require_sock()
        prev = sock.gettimeout()
        sock.settimeout(self.control_timeout_s)
        try:
            send_frame(sock, {"type": "ping", "t0": time.time()})
            reply = read_frame(sock)
        finally:
            try:
                sock.settimeout(prev)
            except Exception:
                pass
        if reply.get("type") != "pong":
            raise RemoteJobsError(f"unexpected ping reply: {reply!r}")
        return reply

    def health(self) -> dict[str, Any]:
        sock = self._require_sock()
        prev = sock.gettimeout()
        sock.settimeout(self.control_timeout_s)
        try:
            send_frame(sock, {"type": "health"})
            reply = read_frame(sock)
        finally:
            try:
                sock.settimeout(prev)
            except Exception:
                pass
        if reply.get("type") != "health_ok":
            raise RemoteJobsError(f"unexpected health reply: {reply!r}")
        return reply

    def submit_job(self, job: dict[str, Any], *, kind: str = "play") -> dict[str, Any]:
        """Submit one game job; blocks until the remote result frame arrives.

        One reconnect+retry is allowed when the peer closed an idle farm
        session under us (common after train/promo gaps on undeployed workers).
        """
        remote_job = prepare_remote_play_job(self.host, job)
        # Belief-MCTS games on a backlogged remote routinely exceed the nominal
        # game_timeout wall before the worker can return a watchdog result.
        # Keep a large socket buffer so slow-but-alive Elmo is not false-killed.
        job_buffer_s = _env_float("POKEBOT_REMOTE_JOB_TIMEOUT_BUFFER_S", 600.0)
        last_exc: Optional[BaseException] = None
        for attempt in range(2):
            sock: Optional[socket.socket] = None
            prev: Optional[float] = None
            try:
                # Keep the socket lookup inside the retry boundary. A farm
                # template can remain present after its worker/container has
                # recycled while its socket was cleared; that is a healable
                # hangup, not a terminal "not connected" job failure.
                sock = self._require_sock()
                prev = sock.gettimeout()
                sock.settimeout(
                    max(
                        self.timeout_s,
                        float(remote_job.get("game_timeout_s") or 900)
                        + job_buffer_s,
                    )
                )
                send_frame(sock, {"type": "job", "kind": kind, "job": remote_job})
                reply = read_frame(sock)
            except (TimeoutError, OSError, RemoteJobsError) as exc:
                last_exc = exc
                if attempt == 0 and _is_remote_hangup_error(exc):
                    try:
                        self.reconnect()
                    except (TimeoutError, OSError, RemoteJobsError):
                        raise exc
                    continue
                raise
            finally:
                if sock is not None:
                    try:
                        sock.settimeout(prev)
                    except Exception:
                        pass
            if reply.get("type") != "result":
                raise RemoteJobsError(f"unexpected job reply: {reply!r}")
            if not reply.get("ok", False):
                raise RemoteJobsError(str(reply.get("error") or "remote job failed"))
            payload = reply.get("result")
            if not isinstance(payload, dict):
                raise RemoteJobsError("remote result missing body")
            return require_remote_result_success(payload)
        assert last_exc is not None
        raise last_exc

    def _control_call(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Send a control-plane frame with ``control_timeout_s``; one hangup retry."""
        last_exc: Optional[BaseException] = None
        for attempt in range(2):
            sock: Optional[socket.socket] = None
            prev: Optional[float] = None
            try:
                # ``_require_sock`` must be covered by the reconnect retry.
                # A client may still be registered in the farm after a worker
                # replacement but have ``_sock is None``; this exact state used
                # to abort the between-iteration checkpoint hard gate.
                sock = self._require_sock()
                prev = sock.gettimeout()
                sock.settimeout(self.control_timeout_s)
                send_frame(sock, msg)
                return read_frame(sock)
            except (TimeoutError, OSError, RemoteJobsError) as exc:
                last_exc = exc
                if attempt == 0 and _is_remote_hangup_error(exc):
                    try:
                        self.reconnect()
                    except (TimeoutError, OSError, RemoteJobsError):
                        raise exc
                    continue
                raise
            finally:
                if sock is not None:
                    try:
                        sock.settimeout(prev)
                    except Exception:
                        pass
        assert last_exc is not None
        raise last_exc

    def reload_checkpoint(
        self,
        path: str,
        *,
        digest: Optional[str] = None,
        version: Optional[int] = None,
    ) -> dict[str, Any]:
        remote_path = resolve_remote_checkpoint_path(self.host, path)
        msg: dict[str, Any] = {"type": "reload", "path": remote_path}
        if digest is not None:
            msg["digest"] = digest
        if version is not None:
            msg["version"] = int(version)
        reply = self._control_call(msg)
        if reply.get("type") != "reload_ok" or not reply.get("ok", False):
            raise RemoteJobsError(
                f"reload failed host={self.host} remote_path={remote_path}: {reply!r}"
            )
        return reply

    def pin_checkpoint(
        self,
        path: str,
        *,
        digest: Optional[str] = None,
    ) -> dict[str, Any]:
        """Pin a second digest on remote leaf servers (belief-MCTS promotion)."""
        remote_path = resolve_remote_checkpoint_path(self.host, path)
        msg: dict[str, Any] = {"type": "pin", "path": remote_path}
        if digest is not None:
            msg["digest"] = digest
        reply = self._control_call(msg)
        if reply.get("type") != "pin_ok" or not reply.get("ok", False):
            raise RemoteJobsError(
                f"pin failed host={self.host} remote_path={remote_path}: {reply!r}"
            )
        return reply

    def unpin_checkpoint(self, digest: str) -> dict[str, Any]:
        reply = self._control_call({"type": "unpin", "digest": digest})
        if reply.get("type") != "unpin_ok" or not reply.get("ok", False):
            raise RemoteJobsError(f"unpin failed: {reply!r}")
        return reply

    def request_rotation(self, reason: str) -> dict[str, Any]:
        """Request a clean supervised restart at a drained iteration boundary."""

        reply = self._control_call(
            {"type": "rotate", "reason": str(reason)}
        )
        if reply.get("type") != "rotate_ok" or not reply.get("ok", False):
            raise RemoteJobsError(f"controlled rotation failed: {reply!r}")
        return reply


def parse_endpoint(spec: str) -> tuple[str, int]:
    """Parse ``host``, ``host:port``, or ``tcp://host:port``."""
    text = spec.strip()
    if text.startswith("tcp://"):
        text = text[len("tcp://") :]
    if ":" in text:
        host, _, port_s = text.rpartition(":")
        return host, int(port_s)
    return text, DEFAULT_PORT


def split_jobs_additive(
    jobs: list[dict[str, Any]],
    *,
    local_workers: int,
    remote_workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Capacity-weighted split between local IPC and remote TCP workers.

    Default: local slots claim the first ``local_workers`` positions in each
    round-robin window (local-primary). Set ``POKEBOT_REMOTE_PRIMARY=1`` to
    reverse that so remotes receive the leading slots (core remote-heavy).

    ``POKEBOT_REMOTE_ONLY=1`` (or ``local_workers<=0`` with live remotes) sends
    **all** jobs to remotes so the training box does not steal CPU from a
    co-resident trainer (Blackwell).
    """
    if not jobs or int(remote_workers) <= 0:
        return list(jobs), []
    if _env_truthy("POKEBOT_REMOTE_ONLY") or int(local_workers) <= 0:
        return [], list(jobs)
    local_slots = max(1, int(local_workers))
    remote_slots = max(1, int(remote_workers))
    total = local_slots + remote_slots
    remote_primary = _env_truthy("POKEBOT_REMOTE_PRIMARY")
    local_jobs: list[dict[str, Any]] = []
    remote_jobs: list[dict[str, Any]] = []
    for index, job in enumerate(jobs):
        slot = index % total
        to_remote = (
            slot < remote_slots if remote_primary else slot >= local_slots
        )
        if to_remote:
            remote_jobs.append(job)
        else:
            local_jobs.append(job)
    return local_jobs, remote_jobs


def _is_remote_hangup_error(exc: BaseException) -> bool:
    """True for peer hangups/resets that a reconnect can heal.

    Deliberately excludes bare ``TimeoutError``: a long in-flight game timing
    out must not be auto-retried (would double-run the job on a fresh socket).
    """
    if isinstance(exc, ConnectionError):
        return True
    if isinstance(exc, OSError) and not isinstance(exc, TimeoutError):
        # errno-based hangups (EPIPE/ECONNRESET/etc.); skip socket.timeout.
        err = getattr(exc, "errno", None)
        if err in {32, 104, 54, 107}:  # EPIPE, ECONNRESET, (mac ECONNRESET), ENOTCONN
            return True
        msg = str(exc).lower()
        return any(
            token in msg
            for token in (
                "broken pipe",
                "connection reset",
                "connection aborted",
                "not connected",
            )
        )
    if isinstance(exc, RemoteJobsError):
        msg = str(exc).lower()
        return any(
            token in msg
            for token in (
                "connection closed",
                "not connected",
                "broken pipe",
                "connection reset",
            )
        )
    return False


def _clone_remote_client(template: RemoteJobClient) -> RemoteJobClient:
    """Open an extra TCP session to the same endpoint (one in-flight game each)."""
    client = RemoteJobClient(
        template.host,
        template.port,
        timeout_s=template.timeout_s,
        connect_timeout_s=template.connect_timeout_s,
        control_timeout_s=template.control_timeout_s,
    )
    client.connect()
    return client


def _parallel_remote_slots(
    remote_clients: list[RemoteJobClient],
    *,
    demand_by_endpoint: Optional[dict[str, int]] = None,
) -> tuple[list[RemoteJobClient], list[RemoteJobClient]]:
    """Expand farm endpoints into concurrent sockets (one in-flight game each).

    With ``demand_by_endpoint``, opens ``min(capacity, max_cap, demand)`` sockets
    so defaults stay below max and the scheduler can grow into headroom.
    Without demand, legacy behaviour uses weight × hello ``workers``.

    Returns ``(all_slots, owned_clones)`` where ``owned_clones`` must be closed
    by the caller (slot 0 per endpoint reuses the farm session).
    """
    slots: list[RemoteJobClient] = []
    owned: list[RemoteJobClient] = []
    # POKEBOT_REMOTE_SLOT_DIVISOR>1 shares farm capacity across concurrent trainers.
    divisor = max(1, int(os.environ.get("POKEBOT_REMOTE_SLOT_DIVISOR", "1") or "1"))
    for template in remote_clients:
        # Heal farm templates that went idle-dead across train/promo gaps before
        # we stamp out N worker clones against a half-closed socket.
        ensure_alive = getattr(template, "ensure_alive", None)
        if callable(ensure_alive):
            try:
                ensure_alive()
            except (TimeoutError, OSError, RemoteJobsError) as exc:
                print(
                    f"[remote] {getattr(template, 'endpoint', '?')} template reconnect "
                    f"failed ({type(exc).__name__}: {exc}); skipping endpoint this wave",
                    flush=True,
                )
                continue
        advertised = max(
            1, int(template.info.workers) if template.info is not None else 1
        )
        capacity = (
            remote_capacity_workers(template.info)
            if template.info is not None
            else advertised
        )
        max_cap = endpoint_max_workers(template.host, template.port)
        default = endpoint_default_workers(template.host, template.port)
        if demand_by_endpoint is not None:
            demand = int(demand_by_endpoint.get(template.endpoint, default))
            n = demand_slot_count(
                capacity=capacity, demand=demand, max_cap=max_cap
            )
            print(
                f"[remote] {getattr(template, 'endpoint', '?')} "
                f"demand={demand} slots={n}/{capacity} "
                f"(default={default} max={max_cap} hello_workers={advertised})",
                flush=True,
            )
        else:
            weight = endpoint_dispatch_weight(template.host, template.port)
            n = weighted_slot_count(advertised, weight)
            if weight < 0.999:
                print(
                    f"[remote] {getattr(template, 'endpoint', '?')} weight={weight:.2f} "
                    f"slots={n}/{advertised}",
                    flush=True,
                )
        n = max(0, n // divisor) if divisor > 1 else n
        execution_slots = n
        n = remote_socket_target(execution_slots)
        if n <= 0:
            print(
                f"[remote] {getattr(template, 'endpoint', '?')} "
                f"→ 0 dispatch slots (skipped this wave)",
                flush=True,
            )
            continue
        if n != execution_slots:
            print(
                f"[remote] {getattr(template, 'endpoint', '?')} "
                f"socket_prefetch={n} ({execution_slots} workers × "
                f"{remote_socket_prefetch_factor()})",
                flush=True,
            )
        slots.append(template)
        for _ in range(n - 1):
            clone = _clone_remote_client(template)
            slots.append(clone)
            owned.append(clone)
    return slots, owned


def iter_additive_results(
    *,
    local_pool: Any,
    local_fn: Callable[[dict[str, Any]], dict[str, Any]],
    jobs: list[dict[str, Any]],
    remote_clients: list[RemoteJobClient],
    kind: str = "play",
    local_workers: int,
    remote_workers: int = 0,
    on_execution: Optional[Callable[[dict[str, Any]], None]] = None,
) -> Iterator[dict[str, Any]]:
    """Run local WorkerPool and optional remote TCP workers in parallel.

    Yields results in completion order (unordered), matching ``imap_unordered``.
    When ``remote_clients`` is empty, delegates to the local pool only.

    Remote concurrency matches advertised worker counts: each endpoint opens
    ``info.workers`` sockets (one job in flight per socket), so additive
    capacity is real in-flight games — not one serial pipeline per host.
    """
    if not remote_clients or int(remote_workers) <= 0:
        for row in local_pool.imap_unordered(local_fn, jobs):
            if on_execution is not None:
                on_execution({"origin": "local", "kind": kind})
            yield row
        return

    local_jobs, remote_jobs = split_jobs_additive(
        jobs,
        local_workers=local_workers,
        remote_workers=remote_workers,
    )
    out_q = _SpillableResultQueue(
        memory_capacity=result_queue_capacity(
            local_workers=local_workers, remote_workers=remote_workers
        )
    )
    errors: list[BaseException] = []
    owned_clients: list[RemoteJobClient] = []
    producer_stop = threading.Event()
    threads: list[threading.Thread] = []
    remote_slots: list[RemoteJobClient] = []

    def _emit_local() -> None:
        try:
            if local_jobs:
                for row in local_pool.imap_unordered(local_fn, local_jobs):
                    if producer_stop.is_set():
                        return
                    if on_execution is not None:
                        on_execution({"origin": "local", "kind": kind})
                    if not _put_thread_result(
                        out_q, producer_stop, ("ok", row)
                    ):
                        return
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            _put_thread_result(out_q, producer_stop, ("err", exc))
        finally:
            _put_thread_result(out_q, producer_stop, ("done", None))

    no_local_fallback = _env_truthy("POKEBOT_REMOTE_NO_LOCAL_FALLBACK") or _env_truthy(
        "POKEBOT_REMOTE_ONLY"
    )

    def _submit_remote_with_retry(
        client: RemoteJobClient, job: dict[str, Any]
    ) -> dict[str, Any]:
        """Submit one job; on failure reconnect and retry on remote (never local)."""
        max_attempts = max(
            1, int(os.environ.get("POKEBOT_REMOTE_JOB_RETRIES", "8") or "8")
        )
        last_exc: Optional[BaseException] = None
        for attempt in range(max_attempts):
            try:
                return client.submit_job(job, kind=kind)
            except (TimeoutError, OSError, RemoteJobsError) as exc:
                last_exc = exc
                print(
                    f"[remote] {client.endpoint} job attempt {attempt + 1}/"
                    f"{max_attempts} failed ({type(exc).__name__}: {exc}); "
                    "reconnect+retry on remote (no local fallback)",
                    flush=True,
                )
                try:
                    client.reconnect()
                except (TimeoutError, OSError, RemoteJobsError) as recon_exc:
                    print(
                        f"[remote] {client.endpoint} reconnect failed "
                        f"({type(recon_exc).__name__}: {recon_exc}); backing off",
                        flush=True,
                    )
                    time.sleep(min(30.0, 2.0 * (attempt + 1)))
                else:
                    time.sleep(min(5.0, 0.5 * (attempt + 1)))
        assert last_exc is not None
        raise last_exc

    def _emit_remote(client: RemoteJobClient, batch: list[dict[str, Any]]) -> None:
        try:
            for index, job in enumerate(batch):
                if producer_stop.is_set():
                    return
                try:
                    if no_local_fallback:
                        row = _submit_remote_with_retry(client, job)
                    else:
                        row = client.submit_job(job, kind=kind)
                    if on_execution is not None:
                        on_execution(
                            {
                                "origin": "remote",
                                "endpoint": client.endpoint,
                                "kind": kind,
                            }
                        )
                    if not _put_thread_result(
                        out_q, producer_stop, ("ok", row)
                    ):
                        return
                except (TimeoutError, OSError, RemoteJobsError) as exc:
                    remaining = batch[index:]
                    if no_local_fallback:
                        # Exhausted remote retries — fail the wave; never dump onto
                        # the training-box CPU (steals from co-resident Blackwell).
                        print(
                            f"[remote] {client.endpoint} slot failed after remote "
                            f"retries ({type(exc).__name__}: {exc}); "
                            f"NOT falling back {len(remaining)} job(s) to local "
                            "(POKEBOT_REMOTE_NO_LOCAL_FALLBACK)",
                            flush=True,
                        )
                        raise
                    print(
                        f"[remote] {client.endpoint} slot failed ({type(exc).__name__}: {exc}); "
                        f"falling back {len(remaining)} job(s) to local pool",
                        flush=True,
                    )
                    # Must use WorkerPool.apply (child process), never call the
                    # worker fn on this additive thread — SIGALRM / thread-local
                    # assumptions break with "signal only works in main thread".
                    for fallback_job in remaining:
                        if producer_stop.is_set():
                            return
                        row = local_pool.apply(local_fn, fallback_job)
                        if on_execution is not None:
                            on_execution(
                                {
                                    "origin": "local_fallback",
                                    "endpoint": client.endpoint,
                                    "kind": kind,
                                }
                            )
                        if not _put_thread_result(
                            out_q, producer_stop, ("ok", row)
                        ):
                            return
                    return
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            _put_thread_result(out_q, producer_stop, ("err", exc))
        finally:
            _put_thread_result(out_q, producer_stop, ("done", None))

    try:
        if remote_jobs:
            remote_slots, owned_clients = _parallel_remote_slots(remote_clients)
            if not remote_slots:
                if no_local_fallback:
                    print(
                        f"[remote] no live farm slots; reconnecting templates "
                        f"(no local fallback for {len(remote_jobs)} job(s))",
                        flush=True,
                    )
                    for template in remote_clients:
                        try:
                            template.reconnect()
                        except (TimeoutError, OSError, RemoteJobsError) as exc:
                            print(
                                f"[remote] {getattr(template, 'endpoint', '?')} "
                                f"reconnect failed ({type(exc).__name__}: {exc})",
                                flush=True,
                            )
                    remote_slots, owned_clients = _parallel_remote_slots(remote_clients)
                if not remote_slots:
                    if no_local_fallback:
                        raise RemoteJobsError(
                            f"no live farm slots for {len(remote_jobs)} job(s); "
                            "local fallback disabled (POKEBOT_REMOTE_NO_LOCAL_FALLBACK)"
                        )
                    print(
                        f"[remote] no live farm slots; falling back {len(remote_jobs)} "
                        "job(s) to local pool",
                        flush=True,
                    )
                    local_jobs.extend(remote_jobs)
                    remote_jobs = []
        if local_jobs:
            threads.append(
                threading.Thread(target=_emit_local, name="additive-local", daemon=True)
            )
        if remote_jobs and remote_slots:
            n_slots = len(remote_slots)
            per_endpoint_jobs: dict[str, int] = {}
            for slot_index, client in enumerate(remote_slots):
                batch = [
                    job
                    for job_index, job in enumerate(remote_jobs)
                    if job_index % n_slots == slot_index
                ]
                if not batch:
                    continue
                per_endpoint_jobs[client.endpoint] = (
                    int(per_endpoint_jobs.get(client.endpoint, 0)) + len(batch)
                )
                threads.append(
                    threading.Thread(
                        target=_emit_remote,
                        args=(client, batch),
                        name=f"additive-remote-{client.endpoint}-{slot_index}",
                        daemon=True,
                    )
                )
            if per_endpoint_jobs:
                parts = [
                    f"{ep}={n}"
                    for ep, n in sorted(per_endpoint_jobs.items(), key=lambda x: x[0])
                ]
                print(
                    f"[remote] dispatch kind={kind} local_jobs={len(local_jobs)} "
                    f"remote_jobs={len(remote_jobs)} slots={n_slots} "
                    f"per_endpoint[{', '.join(parts)}]",
                    flush=True,
                )
        for thread in threads:
            thread.start()

        pending = len(threads)
        while pending > 0:
            tag, payload = out_q.get()
            if tag == "done":
                pending -= 1
                continue
            if tag == "err":
                producer_stop.set()
                raise payload
            yield payload

        for thread in threads:
            thread.join(timeout=1.0)
        if errors:
            raise errors[0]
    finally:
        producer_stop.set()
        for client in owned_clients:
            try:
                client.close()
            except Exception:
                pass
        for thread in threads:
            thread.join(timeout=1.0)
        out_q.close()


def _result_decision_count(result: Any) -> int:
    """Best-effort telemetry count that can never fail a completed game."""
    try:
        return max(0, int(result.get("n_decisions") or 0))
    except (AttributeError, TypeError, ValueError):
        return 0


def iter_scheduled_additive_results(
    *,
    local_pool: Any,
    local_fn: Callable[[dict[str, Any]], dict[str, Any]],
    jobs: list[dict[str, Any]],
    remote_clients: list[RemoteJobClient],
    kind: str = "play",
    scheduler: Any,
    local_workers: int,
    remote_workers: int = 0,
    on_remote_slots: Optional[Callable[[dict[str, int]], None]] = None,
    on_execution: Optional[Callable[[dict[str, Any]], None]] = None,
) -> Iterator[dict[str, Any]]:
    """Additive collect with mid-wave rebalance of *new chunk claims*.

    Remotes still receive **large job chunks** per socket refill (LAN RTT
    amortize) — never chatty per-game scheduler RPCs. Scheduler metrics use
    wave wall-clock GPS only (see ``mid_iter_scheduler.WaveGpsTracker``).

    ``on_remote_slots`` (optional) is invoked with
    ``{active, demand, claimed_remote}`` whenever open remote sockets or
    scheduler demand change — used to keep tqdm ``remotes=`` live.
    """
    if scheduler is None or not remote_clients or int(remote_workers) <= 0:
        yield from iter_additive_results(
            local_pool=local_pool,
            local_fn=local_fn,
            jobs=jobs,
            remote_clients=remote_clients,
            kind=kind,
            local_workers=local_workers,
            remote_workers=remote_workers,
            on_execution=on_execution,
        )
        return

    from collections import deque

    remaining: deque[dict[str, Any]] = deque(jobs)
    claim_lock = threading.Lock()
    grow_lock = threading.Lock()  # also guards slots_by_endpoint / retire_tokens
    claimed_local = 0
    claimed_remote = 0
    out_q = _SpillableResultQueue(
        memory_capacity=result_queue_capacity(
            local_workers=local_workers, remote_workers=remote_workers
        )
    )
    errors: list[BaseException] = []
    owned_clients: list[RemoteJobClient] = []
    producer_stop = threading.Event()
    refill_monitor_stop = threading.Event()
    refill_monitor: Optional[threading.Thread] = None
    pending_lock = threading.Lock()
    threads: list[threading.Thread] = []
    slots_by_endpoint: dict[str, int] = {}
    refill_slot_floor: dict[str, int] = {}
    low_water_recovery: set[str] = set()
    initial_remote_tokens: set[int] = set()
    initial_remote_leaders: set[int] = set()
    initial_endpoint_leaders_ready = threading.Event()
    # A local emitter may legitimately finish after remotes have claimed the
    # whole queue.  It can also be alive but unable to refill: ``imap_unordered``
    # does not return from a fixed-size claim until its final straggler finishes.
    # In either case the remaining remote emitters must be allowed to drain the
    # tail even when their historical claim fraction is above the scheduler's
    # soft ceiling.  Without this hand-off the queue can be non-empty with no
    # eligible claimant forever.
    local_emitter_done = threading.Event()
    local_batch_size = 0
    local_batch_outstanding = 0
    local_last_progress_mono = time.monotonic()
    local_straggler_stale_s = max(
        0.0, _env_float("PURE_RL_LOCAL_STRAGGLER_STALE_S", 30.0)
    )
    tail_override_logged = False
    total_jobs = len(jobs)
    endpoint_templates: dict[str, RemoteJobClient] = {}
    for template in remote_clients:
        endpoint_templates.setdefault(template.endpoint, template)
    # Jobs reserved for a remote are staged in an endpoint-owned controller
    # queue. Elmo request threads can only drain Elmo's deque and Bert request
    # threads can only drain Bert's deque; OS thread timing can no longer turn
    # a shared-deque lock race into endpoint starvation.
    endpoint_owned_queues: dict[str, deque[dict[str, Any]]] = {
        endpoint: deque() for endpoint in endpoint_templates
    }
    endpoint_queue_high_water: dict[str, int] = {
        endpoint: 0 for endpoint in endpoint_templates
    }
    endpoint_queue_safety_ceiling: dict[str, int] = {
        endpoint: 0 for endpoint in endpoint_templates
    }
    endpoint_credits = _EndpointClaimCredits(total_jobs=total_jobs)
    credit_demand_signature: Optional[tuple[tuple[str, int], ...]] = None
    log_scheduler = getattr(scheduler, "maybe_tick", None)
    no_local_fallback = _env_truthy(
        "POKEBOT_REMOTE_NO_LOCAL_FALLBACK"
    ) or _env_truthy("POKEBOT_REMOTE_ONLY")
    remote_only = _env_truthy("POKEBOT_REMOTE_ONLY")
    demand_queue_mode = _env_truthy("POKEBOT_REMOTE_DEMAND_QUEUE")
    endpoint_owned_queue_mode = bool(demand_queue_mode or remote_only)

    def _refresh_endpoint_credit_targets(dec: Any) -> None:
        """Refresh pure per-endpoint work targets under ``claim_lock``.

        Scheduler demand is the intended execution concurrency.  Each healthy
        endpoint additionally owns its configured server-queue target (four
        execution waves by default), matching the low-water refill contract.
        This function performs environment/config arithmetic only; it never
        opens a socket or probes a remote while the shared deque is locked.
        """
        nonlocal credit_demand_signature
        demand_map = dict(getattr(dec, "remote_demand", None) or {})
        signature = tuple(
            sorted(
                (
                    ep,
                    max(
                        0,
                        int(
                            demand_map.get(
                                ep,
                                endpoint_default_workers(
                                    template.host, template.port
                                ),
                            )
                        ),
                    ),
                )
                for ep, template in endpoint_templates.items()
            )
        )
        if signature == credit_demand_signature:
            return
        credit_demand_signature = signature
        for ep, execution_demand in signature:
            template = endpoint_templates[ep]
            execution = min(
                max(0, int(execution_demand)),
                endpoint_max_workers(template.host, template.port),
            )
            if execution <= 0:
                endpoint_credits.set_target(ep, 0)
                endpoint_queue_high_water[ep] = 0
                endpoint_queue_safety_ceiling[ep] = 0
                continue
            queue_target = endpoint_dispatch_chunk(
                template.host,
                template.port,
                default=4 * execution,
            )
            # Capacity includes one executing wave plus the endpoint-specific
            # waiting queue. With production settings this is Elmo 48+336 and
            # Bert 16+112. The queues are disjoint even though their source is
            # the same finite collection wave.
            high_water = execution + queue_target
            endpoint_queue_high_water[ep] = high_water
            # The scheduling target is soft: one execution wave may be in
            # transition while a low-water refill is being admitted. This
            # ceiling exists only for device/file-descriptor safety and is not
            # used as the ordinary refill target.
            endpoint_queue_safety_ceiling[ep] = high_water + execution
            endpoint_credits.set_target(
                ep, endpoint_queue_safety_ceiling[ep]
            )

    def _endpoint_remote_available_locked() -> int:
        available = len(remaining)
        if demand_queue_mode and not remote_only:
            local_reserve = (
                0
                if local_emitter_done.is_set()
                else max(int(local_workers), int(local_batch_outstanding))
            )
            available = max(0, available - local_reserve)
        return available

    def _refill_endpoint_queue_locked(endpoint: str, dec: Any) -> int:
        """Fill one endpoint-owned queue without consuming peer credit.

        Caller holds ``claim_lock``. Credit accounting includes queued plus
        executing jobs, so a completion releases exactly one refill slot for
        the same endpoint. This is local bookkeeping only; no network I/O is
        performed while the lock is held.
        """
        nonlocal claimed_remote
        ep = str(endpoint)
        _refresh_endpoint_credit_targets(dec)
        owned = endpoint_owned_queues.setdefault(ep, deque())
        target = max(0, int(endpoint_queue_high_water.get(ep, 0)))
        if target <= 0 or not remaining:
            return 0
        need = max(0, target - len(owned))
        if need <= 0:
            return 0
        available = _endpoint_remote_available_locked()
        granted = endpoint_credits.claim_limit(
            ep,
            want=need,
            remaining_count=available,
            wave_remaining=len(remaining),
        )
        if granted <= 0:
            return 0
        for _ in range(granted):
            owned.append(remaining.popleft())
        claimed_remote += granted
        endpoint_credits.note_claimed(ep, granted)
        return granted

    def _finish_remote_credit(endpoint: str, n: int = 1) -> None:
        """Release completed remote work and restore data-path health."""
        with claim_lock:
            endpoint_credits.note_finished(endpoint, n)
            # A returned game is stronger evidence than a failed control probe.
            endpoint_credits.set_healthy(endpoint, True)

    def _finish_initial_remote_claim(token: Optional[int]) -> None:
        if token is None:
            return
        initial_remote_tokens.discard(int(token))
        initial_remote_leaders.discard(int(token))
        if not initial_remote_leaders:
            initial_endpoint_leaders_ready.set()

    def _claim(
        side: str,
        want: int,
        *,
        initial_token: Optional[int] = None,
        endpoint: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        nonlocal claimed_local, claimed_remote
        nonlocal local_batch_size, local_batch_outstanding
        nonlocal local_last_progress_mono, tail_override_logged
        requested_want = max(0, int(want))
        want = requested_want
        if want <= 0:
            return []
        with claim_lock:
            initial_attempt = bool(
                side == "remote"
                and initial_token is not None
                and initial_token in initial_remote_tokens
            )
            if side == "remote" and endpoint_owned_queue_mode:
                if endpoint is None:
                    raise RuntimeError("remote claim missing endpoint")
                dec = scheduler.decision()
                _refill_endpoint_queue_locked(str(endpoint), dec)
                owned = endpoint_owned_queues.setdefault(str(endpoint), deque())
                take = min(requested_want, len(owned))
                batch = [owned.popleft() for _ in range(take)]
                if initial_attempt:
                    _finish_initial_remote_claim(initial_token)
                if callable(log_scheduler):
                    try:
                        log_scheduler(remaining=len(remaining))
                    except Exception:
                        pass
                # Never fall through to the shared deque: an empty Bert queue
                # must wait for Bert credit/refill, not race an Elmo claimant.
                return batch
            if not remaining:
                if initial_attempt:
                    _finish_initial_remote_claim(initial_token)
                return []
            dec = scheduler.decision()
            local_share = float(dec.local_share)
            min_local = float(getattr(scheduler, "min_local_frac", 0.40))
            total_claimed = claimed_local + claimed_remote
            if side == "remote":
                if remote_only:
                    want = min(want, len(remaining))
                elif demand_queue_mode:
                    # Completion speed is the demand signal. Keep every live
                    # remote socket fed, but reserve one complete local wave so
                    # fast remotes cannot empty the shared deque before the
                    # already-full local pool asks for its next batch.
                    local_reserve = (
                        0
                        if local_emitter_done.is_set()
                        else max(int(local_workers), int(local_batch_outstanding))
                    )
                    remote_room = max(0, len(remaining) - local_reserve)
                    want = min(want, remote_room)
                else:
                    # Legacy soft-share mode. Large private socket chunks used
                    # with this gate can create burst starvation; production
                    # uses the bounded demand queue above.
                    demand_map = getattr(dec, "remote_demand", None) or {}
                    try:
                        demand_sum = int(sum(int(v) for v in demand_map.values()))
                    except Exception:
                        demand_sum = max(0, int(remote_workers))
                    demand_sum = max(1, demand_sum)
                    demand_frac = float(demand_sum) / float(
                        max(1, int(local_workers)) + demand_sum
                    )
                    max_remote_frac = min(
                        float(getattr(scheduler, "max_remote_frac", 0.60)),
                        max(
                            float(dec.remote_share),
                            float(getattr(scheduler, "min_remote_frac", 0.0)),
                            demand_frac,
                        ),
                    )
                    if total_claimed > 0:
                        room = int(
                            max_remote_frac * (total_claimed + want)
                            - claimed_remote
                        )
                        want = min(want, max(0, room))
                    prefer = float(
                        getattr(scheduler, "prefer_local_frac", min_local)
                    )
                    local_floor = max(
                        min_local, min(prefer, float(local_share))
                    )
                    target_local = local_floor * max(total_claimed + want, 1)
                    if (
                        claimed_local + 1 < target_local * 0.50
                        and claimed_local < max(8, int(local_workers) // 4)
                        and total_claimed > want
                    ):
                        want = max(0, want // 2)
                # Do not treat every partially-completed local batch as a
                # straggler.  That would bypass the remote share ceiling after
                # the first ordinary local completion and could move the whole
                # tail onto slower remotes.  The production latch had two
                # outstanding results; reserve this override for the final 5%
                # of a batch (with a two-result floor).
                now_mono = time.monotonic()
                local_progress_age_s = max(
                    0.0, now_mono - float(local_last_progress_mono)
                )
                local_claimant_stale = _local_tail_override_ready(
                    emitter_done=local_emitter_done.is_set(),
                    batch_size=local_batch_size,
                    outstanding=local_batch_outstanding,
                    last_progress_mono=local_last_progress_mono,
                    now_mono=now_mono,
                    stale_s=local_straggler_stale_s,
                )
                if (
                    want <= 0
                    and remaining
                    and local_claimant_stale
                ):
                    # Tail-drain correctness override.  The share limit is a
                    # throughput preference, not permission to strand jobs
                    # after the local claimant has exited or is stuck waiting
                    # for the last result(s) of an underfilled fixed batch.
                    want = min(max(1, requested_want), len(remaining))
                    if not tail_override_logged:
                        local_state = (
                            "done"
                            if local_emitter_done.is_set()
                            else (
                                f"underfilled "
                                f"({local_batch_outstanding}/{local_batch_size} "
                                f"outstanding; stale={local_progress_age_s:.1f}s/"
                                f"{local_straggler_stale_s:.1f}s)"
                            )
                        )
                        print(
                            "[remote] scheduled tail-drain override: local "
                            f"emitter is {local_state}; ignoring remote soft share for "
                            f"{len(remaining)} remaining job(s)",
                            flush=True,
                        )
                        tail_override_logged = True
            else:
                # Local-primary / additive: always allow local claims up to
                # remaining — demand growth never soft-caps local. Endpoint
                # credit fairness is never a steal from the local claimant.
                pass
            want = min(want, len(remaining))
            if initial_attempt:
                # First remote claims are a synchronized round. Cap each slot
                # to its fair share of the still-unclaimed work, reserving room
                # for later slots (and therefore later endpoints) instead of
                # letting the first Elmo thread take a 128-job chunk before any
                # Bert thread runs.
                unclaimed = max(1, len(initial_remote_tokens))
                fair_share = max(1, (len(remaining) + unclaimed - 1) // unclaimed)
                want = min(want, fair_share)
                if (
                    want <= 0
                    and int(initial_token) in initial_remote_leaders
                    and remaining
                ):
                    # One leader per endpoint is an execution-proof reservation.
                    # It may exceed a transient soft share by one job, never the
                    # hard remote capacity or no-fallback contract.
                    want = 1
                _finish_initial_remote_claim(initial_token)
            if side == "remote":
                if endpoint is None:
                    raise RuntimeError("remote claim missing endpoint")
                _refresh_endpoint_credit_targets(dec)
                remote_available = len(remaining)
                if demand_queue_mode and not remote_only:
                    local_reserve = (
                        0
                        if local_emitter_done.is_set()
                        else max(int(local_workers), int(local_batch_outstanding))
                    )
                    remote_available = max(0, len(remaining) - local_reserve)
                want = endpoint_credits.claim_limit(
                    endpoint,
                    want=want,
                    remaining_count=remote_available,
                    wave_remaining=len(remaining),
                )
            if want <= 0:
                return []
            batch = [remaining.popleft() for _ in range(want)]
            if side == "remote":
                claimed_remote += len(batch)
                endpoint_credits.note_claimed(str(endpoint), len(batch))
            else:
                claimed_local += len(batch)
                local_batch_size = len(batch)
                local_batch_outstanding = len(batch)
                local_last_progress_mono = time.monotonic()
            if callable(log_scheduler):
                try:
                    log_scheduler(remaining=len(remaining))
                except Exception:
                    pass
            return batch

    def _emit_local() -> None:
        nonlocal local_batch_outstanding, local_last_progress_mono
        try:
            local_chunk = max(4, int(local_workers))
            while True:
                if producer_stop.is_set():
                    return
                batch = _claim("local", local_chunk)
                if not batch:
                    break
                for row in local_pool.imap_unordered(local_fn, batch):
                    if producer_stop.is_set():
                        return
                    with claim_lock:
                        local_batch_outstanding = max(
                            0, int(local_batch_outstanding) - 1
                        )
                        local_last_progress_mono = time.monotonic()
                    scheduler.note_completed(
                        side="local",
                        n=1,
                        decisions=_result_decision_count(row),
                    )
                    if on_execution is not None:
                        on_execution({"origin": "local", "kind": kind})
                    if not _put_thread_result(
                        out_q, producer_stop, ("ok", row)
                    ):
                        return
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            _put_thread_result(out_q, producer_stop, ("err", exc))
        finally:
            local_emitter_done.set()
            _put_thread_result(out_q, producer_stop, ("done", None))

    # Cooperative retire tokens: when demand shrinks, excess emit threads
    # exit between chunks so live sockets fall with demand (not grow-only).
    retire_tokens: dict[str, int] = {}

    def _account_remote_slot_exit(endpoint: str) -> tuple[int, int]:
        """Remove one live emitter from both slot and reservation accounting."""
        ep = str(endpoint)
        with grow_lock:
            before = max(0, int(slots_by_endpoint.get(ep, 0)))
            after = max(0, before - 1)
            slots_by_endpoint[ep] = after
        with claim_lock:
            endpoint_credits.unregister(ep)
        return before, after

    def _submit_remote_with_retry(
        client: RemoteJobClient, job: dict[str, Any]
    ) -> dict[str, Any]:
        max_attempts = max(
            1, int(os.environ.get("POKEBOT_REMOTE_JOB_RETRIES", "8") or "8")
        )
        last_exc: Optional[BaseException] = None
        for attempt in range(max_attempts):
            try:
                return client.submit_job(job, kind=kind)
            except (TimeoutError, OSError, RemoteJobsError) as exc:
                last_exc = exc
                print(
                    f"[remote] {client.endpoint} scheduled job attempt "
                    f"{attempt + 1}/{max_attempts} failed "
                    f"({type(exc).__name__}: {exc}); reconnect+retry remote",
                    flush=True,
                )
                try:
                    client.reconnect()
                except (TimeoutError, OSError, RemoteJobsError):
                    time.sleep(min(30.0, 2.0 * (attempt + 1)))
                else:
                    time.sleep(min(5.0, 0.5 * (attempt + 1)))
        assert last_exc is not None
        raise last_exc

    def _emit_remote(
        client: RemoteJobClient,
        initial_token: Optional[int] = None,
    ) -> None:
        # Fair, shallow reservations keep every remote socket fed.  This claim
        # is only an in-process deque operation; it does not amortize LAN RTT.
        # Stay alive on soft-capped empty claims so demand→target→live claims
        # can resume (exiting here left tqdm remotes= and claimed_remote flat
        # while demand climbed). Retire when demand_shrink hands us a token.
        nonlocal claimed_remote
        idle_spins = 0
        ep = client.endpoint
        first_claim = initial_token is not None
        retiring = False
        try:
            while True:
                if producer_stop.is_set():
                    return
                with grow_lock:
                    tok = int(retire_tokens.get(ep, 0))
                    if tok > 0:
                        retire_tokens[ep] = tok - 1
                        retiring = True
                        try:
                            client.close()
                        except Exception:
                            pass
                        return
                dec = scheduler.decision()
                endpoint_backlog = endpoint_dispatch_chunk(
                    client.host,
                    client.port,
                    default=int(getattr(dec, "remote_chunk", 128)),
                )
                with grow_lock:
                    live_endpoint_slots = max(
                        1, int(slots_by_endpoint.get(ep, 1))
                    )
                chunk = max(
                    remote_refill_games_per_socket(),
                    (int(endpoint_backlog) + live_endpoint_slots - 1)
                    // live_endpoint_slots,
                )
                batch = _claim(
                    "remote",
                    chunk,
                    initial_token=initial_token if first_claim else None,
                    endpoint=ep,
                )
                was_initial_claim = first_claim
                if first_claim:
                    first_claim = False
                if not batch:
                    with claim_lock:
                        left = len(remaining)
                    if left <= 0:
                        break
                    idle_spins += 1
                    time.sleep(min(0.25, 0.02 * float(idle_spins)))
                    continue
                idle_spins = 0
                for index, job in enumerate(batch):
                    if producer_stop.is_set():
                        return
                    # Retire only between claimed chunks.  Returning the
                    # unstarted suffix here used to create a deterministic
                    # tail deadlock: local had already exited, while the
                    # surviving remotes were above their soft share and could
                    # not reclaim the returned jobs.  A slot now completes the
                    # batch it owns, then consumes its retire token at the top
                    # of the outer loop.
                    try:
                        row = (
                            _submit_remote_with_retry(client, job)
                            if no_local_fallback
                            else client.submit_job(job, kind=kind)
                        )
                        _finish_remote_credit(ep)
                        scheduler.note_completed(
                            side="remote",
                            n=1,
                            decisions=_result_decision_count(row),
                        )
                        if on_execution is not None:
                            on_execution(
                                {
                                    "origin": "remote",
                                    "endpoint": client.endpoint,
                                    "kind": kind,
                                }
                            )
                        if not _put_thread_result(
                            out_q, producer_stop, ("ok", row)
                        ):
                            return
                    except (TimeoutError, OSError, RemoteJobsError) as exc:
                        remaining_jobs = batch[index:]
                        if no_local_fallback:
                            print(
                                f"[remote] {client.endpoint} scheduled slot failed "
                                f"after remote retries ({type(exc).__name__}: {exc}); "
                                f"NOT falling back {len(remaining_jobs)} job(s)",
                                flush=True,
                            )
                            raise
                        print(
                            f"[remote] {client.endpoint} chunk slot failed "
                            f"({type(exc).__name__}: {exc}); "
                            f"falling back {len(remaining_jobs)} job(s) to local",
                            flush=True,
                        )
                        for fallback_job in remaining_jobs:
                            if producer_stop.is_set():
                                return
                            row = local_pool.apply(local_fn, fallback_job)
                            _finish_remote_credit(ep)
                            scheduler.note_completed(
                                side="local",
                                n=1,
                                decisions=_result_decision_count(row),
                            )
                            if on_execution is not None:
                                on_execution(
                                    {
                                        "origin": "local_fallback",
                                        "endpoint": client.endpoint,
                                        "kind": kind,
                                    }
                                )
                            if not _put_thread_result(
                                out_q, producer_stop, ("ok", row)
                            ):
                                return
                        return
                # Refill independently as soon as this socket completes.  A
                # global post-first-claim rendezvous drained both remote queues
                # to zero while the final startup token lagged on a busy host.
                if was_initial_claim:
                    # Wait only for one reservation per endpoint, never for all
                    # request sockets. This preserves Bert/Elmo admission on a
                    # tiny queue and is already set before a real game returns.
                    initial_endpoint_leaders_ready.wait(timeout=5.0)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            _put_thread_result(out_q, producer_stop, ("err", exc))
        finally:
            # Each emitter exclusively owns its request socket.  Close it as
            # soon as that emitter exits, especially after a remote restart
            # or protocol failure.  Deferring all closes to the outer
            # generator cleanup leaves dead sockets in CLOSE-WAIT while other
            # emitters are still draining; a large queued wave can otherwise
            # consume the worker's entire connection limit and lock health /
            # refill probes out of the control plane.
            try:
                client.close()
            except Exception:
                pass
            before, after = _account_remote_slot_exit(ep)
            if retiring:
                print(
                    f"[remote] {ep} demand_shrink "
                    f"slots={before}->{after} "
                    f"(cap demand; cooperative retire)",
                    flush=True,
                )
            _put_thread_result(out_q, producer_stop, ("done", None))

    try:
        bind = getattr(scheduler, "bind_remote_endpoints", None)
        if callable(bind):
            try:
                bind(remote_clients)
            except Exception as exc:
                print(f"[pure_rl] scheduler bind_remotes failed: {exc!r}", flush=True)

        # Initial tick documents metrics contract in the event log.
        try:
            dec0 = scheduler.maybe_tick(remaining=total_jobs, force=True)
            if dec0 is not None:
                print(
                    f"[pure_rl] mid_iter_rebalance=start {dec0.as_log()} "
                    f"jobs={total_jobs} kind={kind}",
                    flush=True,
                )
        except Exception as exc:
            print(f"[pure_rl] scheduler start tick failed: {exc!r}", flush=True)

        demand0 = {}
        try:
            demand0 = dict(scheduler.decision().remote_demand or {})
        except Exception:
            demand0 = {}
        remote_slots, owned_clients = _parallel_remote_slots(
            remote_clients, demand_by_endpoint=demand0 or None
        )
        if no_local_fallback and not remote_slots:
            raise RemoteJobsError(
                "scheduled dispatch has no live remote slots and local fallback "
                "is disabled"
            )
        # Start endpoint sockets round-robin.  Grouped Elmo-then-Bert startup
        # made the faster endpoint wait behind hundreds of thread starts and
        # amplified synchronized waves on the return path.
        endpoint_buckets: dict[str, list[RemoteJobClient]] = {}
        for client in remote_slots:
            endpoint_buckets.setdefault(client.endpoint, []).append(client)
        if len(endpoint_buckets) > 1:
            interleaved: list[RemoteJobClient] = []
            depth = max(len(bucket) for bucket in endpoint_buckets.values())
            for index in range(depth):
                for bucket in endpoint_buckets.values():
                    if index < len(bucket):
                        interleaved.append(bucket[index])
            remote_slots = interleaved

        initial_remote_tokens.update(range(len(remote_slots)))
        first_slot_by_endpoint: dict[str, int] = {}
        for slot_index, client in enumerate(remote_slots):
            first_slot_by_endpoint.setdefault(client.endpoint, slot_index)
        initial_remote_leaders.update(first_slot_by_endpoint.values())
        if not initial_remote_leaders:
            initial_endpoint_leaders_ready.set()
        if not remote_only:
            threads.append(
                threading.Thread(target=_emit_local, name="sched-local", daemon=True)
            )
        for slot_index, client in enumerate(remote_slots):
            slots_by_endpoint[client.endpoint] = (
                slots_by_endpoint.get(client.endpoint, 0) + 1
            )
            # Register every endpoint before any thread starts.  Thus the
            # first fast Elmo claimant already protects Bert's full credit;
            # fairness does not depend on OS thread-start order.
            endpoint_credits.register(client.endpoint)
            threads.append(
                threading.Thread(
                    target=_emit_remote,
                    args=(client, slot_index),
                    name=f"sched-remote-{client.endpoint}-{slot_index}",
                    daemon=True,
                )
            )
        endpoint_queue_depths: dict[str, int] = {}
        if endpoint_owned_queue_mode:
            # Deterministic controller-side partition before any request
            # thread starts. Both endpoint credits are registered above, so
            # the first refill protects the other endpoint's full capacity.
            with claim_lock:
                dec = scheduler.decision()
                for endpoint in sorted(endpoint_templates):
                    _refill_endpoint_queue_locked(endpoint, dec)
                endpoint_queue_depths = {
                    endpoint: len(queue_)
                    for endpoint, queue_ in endpoint_owned_queues.items()
                }
            print(
                "[remote] endpoint_owned_queues "
                f"depths={endpoint_queue_depths} "
                f"high_water={dict(endpoint_queue_high_water)} "
                f"safety_ceiling={dict(endpoint_queue_safety_ceiling)} "
                "shared_endpoint_race=disabled",
                flush=True,
            )
        print(
            f"[remote] scheduled_dispatch kind={kind} jobs={total_jobs} "
            f"remote_slots={len(remote_slots)} "
            f"remote_chunk={int(scheduler.decision().remote_chunk)} "
            f"refill_games={remote_refill_games_per_socket()} "
            f"demand={demand0 or '-'} "
            f"result_buffer=ram:{out_q.memory_capacity}+"
            f"disk:{out_q.max_spool_bytes / (1024**3):.1f}GiB "
            f"(chunked RTT amortize; metrics=wave_wall_clock_gps; "
            f"remote_demand max≫default)",
            flush=True,
        )
        for thread in threads:
            thread.start()

        pending = len(threads)
        last_log = time.monotonic()

        def _publish_remote_slots() -> None:
            """Push live active-socket + demand counts to tqdm / monitors."""
            if on_remote_slots is None:
                return
            try:
                active = int(sum(slots_by_endpoint.values()))
                dem_map = dict(scheduler.remote_demand() or {})
                demand_n = int(sum(int(v) for v in dem_map.values())) if dem_map else active
                on_remote_slots(
                    {
                        "active": active,
                        "demand": demand_n,
                        "claimed_remote": int(claimed_remote),
                    }
                )
            except Exception:
                pass

        _publish_remote_slots()

        def _maybe_grow_remote_slots(dec: Any) -> None:
            """Hot-add remote sockets when scheduler demand rises mid-wave."""
            nonlocal pending
            demand = dict(getattr(dec, "remote_demand", None) or {})
            if not demand:
                return
            grew = False
            with grow_lock:
                for template in remote_clients:
                    ep = template.endpoint
                    want = remote_socket_target(int(demand.get(ep, 0)))
                    have = int(slots_by_endpoint.get(ep, 0))
                    if want <= have:
                        continue
                    extras = want - have
                    for i in range(extras):
                        try:
                            clone = _clone_remote_client(template)
                        except (TimeoutError, OSError, RemoteJobsError) as exc:
                            print(
                                f"[remote] {ep} grow slot failed "
                                f"({type(exc).__name__}: {exc})",
                                flush=True,
                            )
                            break
                        owned_clients.append(clone)
                        slots_by_endpoint[ep] = slots_by_endpoint.get(ep, 0) + 1
                        with claim_lock:
                            endpoint_credits.register(ep)
                        t = threading.Thread(
                            target=_emit_remote,
                            args=(clone,),
                            name=f"sched-remote-grow-{ep}-{have + i}",
                            daemon=True,
                        )
                        threads.append(t)
                        with pending_lock:
                            pending += 1
                        t.start()
                    if slots_by_endpoint.get(ep, 0) > have:
                        grew = True
                        print(
                            f"[remote] {ep} demand_grow "
                            f"slots={have}->{slots_by_endpoint[ep]} "
                            f"(socket target={want})",
                            flush=True,
                        )
            if grew:
                _publish_remote_slots()

        def _maybe_shrink_remote_slots(dec: Any) -> None:
            """Retire excess emit threads when demand falls (completion-time).

            Grow-only left sockets stuck at the peak (e.g. 37) after
            ``demand_hurt_wave_gps`` shrunk demand back to 20/10 — GPS stayed
            ~6.7 while the faster 20/10 operating point was ~15 g/s. Tokens make
            emitters exit between chunks so ``remotes=`` tracks demand down.
            """
            demand = dict(getattr(dec, "remote_demand", None) or {})
            if not demand:
                return
            shrunk = False
            with grow_lock:
                for template in remote_clients:
                    ep = template.endpoint
                    want = remote_socket_target(
                        max(0, int(demand.get(ep, 0)))
                    )
                    # A low-water refill reserve is held for the rest of this
                    # collection wave.  Shrinking it immediately on the next
                    # ordinary scheduler tick recreates the starvation cycle.
                    want = max(want, int(refill_slot_floor.get(ep, 0)))
                    have = int(slots_by_endpoint.get(ep, 0))
                    pending_retire = int(retire_tokens.get(ep, 0))
                    # Effective live after already-queued retires.
                    effective = have - pending_retire
                    if effective <= want:
                        continue
                    extras = effective - want
                    retire_tokens[ep] = pending_retire + extras
                    shrunk = True
                    print(
                        f"[remote] {ep} demand_shrink_queue "
                        f"slots={have} want={want} retire+={extras} "
                        f"(live sockets must fall with completion gate)",
                        flush=True,
                    )
            if shrunk:
                _publish_remote_slots()

        def _maybe_refill_low_remote_queues() -> None:
            """Fill every low remote queue to high water in one probe pass.

            ``health.active_jobs`` is the authoritative server-side count of
            executing plus locally queued games.  Refill is based on queued
            games (active minus execution workers), not TCP socket estimates.
            Both endpoint probes and all required reserve connections happen
            concurrently, so a low Bert queue cannot wait behind Elmo.  This
            function is called by an independent cadence thread; result ingest
            can never pause the 0.2-second queue controller.
            """
            nonlocal pending
            with claim_lock:
                jobs_left = len(remaining)
            # Do not overfill the final allocation tail; idle stragglers are
            # expected and explicitly outside the steady-state queue contract.
            if jobs_left <= max(32, int(total_jobs * 0.05)):
                return

            samples: dict[str, dict[str, Any]] = {}
            failed_probes: set[str] = set()
            sample_lock = threading.Lock()

            def _probe(template: RemoteJobClient) -> None:
                endpoint = template.endpoint
                control = RemoteJobClient(
                    template.host,
                    template.port,
                    timeout_s=min(3.0, template.timeout_s),
                    connect_timeout_s=min(1.0, template.connect_timeout_s),
                    control_timeout_s=min(2.0, template.control_timeout_s),
                )
                try:
                    control.connect()
                    health = control.health()
                    with sample_lock:
                        samples[endpoint] = health
                except (TimeoutError, OSError, RemoteJobsError) as exc:
                    with sample_lock:
                        failed_probes.add(endpoint)
                    print(
                        f"[remote] {endpoint} queue_probe failed "
                        f"({type(exc).__name__}: {exc}); keeping existing slots",
                        flush=True,
                    )
                finally:
                    control.close()

            probes = [
                threading.Thread(
                    target=_probe,
                    args=(template,),
                    name=f"remote-queue-probe-{template.endpoint}",
                    daemon=True,
                )
                for template in remote_clients
            ]
            for probe in probes:
                probe.start()
            for probe in probes:
                # Never start another probe on the same socket while a prior
                # read is outstanding. Overlapped health frames corrupted the
                # stream and created false queue-empty samples under load.
                probe.join()

            # Probe I/O has completed before taking the deque lock.  A
            # successful health reply (including active_jobs=0) keeps the
            # endpoint's reservation. A transient control timeout does not
            # revoke data-path credit: doing so used to starve a busy remote.
            with claim_lock:
                for endpoint in samples:
                    endpoint_credits.set_healthy(endpoint, True)

            try:
                demand = dict(scheduler.decision().remote_demand or {})
            except Exception:
                demand = {}
            fraction = remote_queue_low_water_fraction()
            plans: list[
                tuple[RemoteJobClient, int, int, int, int, int, int]
            ] = []
            for template in remote_clients:
                health = samples.get(template.endpoint)
                if not health:
                    continue
                execution = demand_slot_count(
                    capacity=remote_capacity_workers(template.info),
                    demand=int(demand.get(template.endpoint, 0)),
                    max_cap=endpoint_max_workers(template.host, template.port),
                )
                if execution <= 0:
                    continue
                controller_high_water = endpoint_dispatch_chunk(
                    template.host,
                    template.port,
                    default=4 * execution,
                )
                active = max(0, int(health.get("active_jobs") or 0))
                queued = max(0, active - execution)
                with grow_lock:
                    have = int(slots_by_endpoint.get(template.endpoint, 0))
                max_slots = remote_socket_max_target(execution)
                # A protected controller queue may exceed the number of
                # requests that the remote server can admit. Treat both values
                # as soft high-water targets and leave the extra work on the
                # trainer rather than pretending the remote can queue it.
                server_high_water = max(
                    1,
                    min(
                        controller_high_water,
                        max(0, max_slots - execution),
                    ),
                )
                low_water = max(1, int(server_high_water * fraction))
                endpoint = template.endpoint
                if queued < low_water:
                    low_water_recovery.add(endpoint)
                elif queued >= server_high_water:
                    low_water_recovery.discard(endpoint)
                if endpoint not in low_water_recovery:
                    continue
                restore_active = execution + server_high_water
                # One poll means one complete top-up.  The old ``execution``
                # term capped this at a single 48/16-worker wave, so a drained
                # queue needed several result-driven probes to recover.
                extras = min(
                    max(0, max_slots - have),
                    max(0, restore_active - active),
                    jobs_left,
                )
                if extras > 0:
                    plans.append(
                        (
                            template,
                            extras,
                            active,
                            queued,
                            low_water,
                            server_high_water,
                            max_slots,
                        )
                    )

            if not plans:
                return

            # Build the request list round-robin, then connect concurrently.
            # This starts the *entire* high-water deficit in this one pass.
            max_extras = max(plan[1] for plan in plans)
            started: dict[str, int] = {plan[0].endpoint: 0 for plan in plans}
            limits = {plan[0].endpoint: int(plan[6]) for plan in plans}
            requests: list[RemoteJobClient] = []
            for depth in range(max_extras):
                for (
                    template,
                    extras,
                    _active,
                    _queued,
                    _low_water,
                    _high_water,
                    _max_slots,
                ) in plans:
                    if depth >= extras:
                        continue
                    requests.append(template)

            def _open_refill_client(template: RemoteJobClient) -> RemoteJobClient:
                client = RemoteJobClient(
                    template.host,
                    template.port,
                    timeout_s=template.timeout_s,
                    connect_timeout_s=min(2.0, template.connect_timeout_s),
                    control_timeout_s=min(2.0, template.control_timeout_s),
                )
                client.connect()
                return client

            open_workers = min(32, max(1, len(requests)))
            with ThreadPoolExecutor(
                max_workers=open_workers,
                thread_name_prefix="remote-refill-connect",
            ) as connector_pool:
                future_templates = {
                    connector_pool.submit(_open_refill_client, template): template
                    for template in requests
                    if not producer_stop.is_set()
                    and not refill_monitor_stop.is_set()
                }
                for future in as_completed(future_templates):
                    template = future_templates[future]
                    ep = template.endpoint
                    try:
                        clone = future.result()
                    except (TimeoutError, OSError, RemoteJobsError) as exc:
                        print(
                            f"[remote] {ep} low_water grow failed "
                            f"({type(exc).__name__}: {exc})",
                            flush=True,
                        )
                        continue
                    with grow_lock:
                        have = int(slots_by_endpoint.get(ep, 0))
                        if (
                            producer_stop.is_set()
                            or refill_monitor_stop.is_set()
                            or have >= int(limits.get(ep, have))
                        ):
                            clone.close()
                            continue
                        owned_clients.append(clone)
                        slots_by_endpoint[ep] = have + 1
                        retire_tokens[ep] = 0
                        refill_slot_floor[ep] = slots_by_endpoint[ep]
                        with claim_lock:
                            endpoint_credits.register(ep)
                    thread = threading.Thread(
                        target=_emit_remote,
                        args=(clone,),
                        name=f"sched-remote-low-water-{ep}-{have}",
                        daemon=True,
                    )
                    threads.append(thread)
                    with pending_lock:
                        pending += 1
                    started[ep] += 1
                    thread.start()

            for (
                template,
                _extras,
                active,
                queued,
                low_water,
                high_water,
                _max_slots,
            ) in plans:
                ep = template.endpoint
                added = int(started.get(ep, 0))
                if added <= 0:
                    continue
                print(
                    f"[remote] {ep} LOW_WATER_REFILL active={active} "
                    f"queued={queued}<{low_water} added={added} "
                    f"fill=high_water target_active="
                    f"{int(demand.get(ep, 0)) + high_water} "
                    f"high_water={high_water} "
                    f"slots={slots_by_endpoint.get(ep, 0)}/"
                    f"{remote_socket_max_target(int(demand.get(ep, 0)))}",
                    flush=True,
                )
            _publish_remote_slots()

        def _remote_queue_refill_monitor() -> None:
            interval = remote_queue_probe_interval_s()
            print(
                "[remote] queue_refill_controller "
                f"interval={interval:.3f}s low_water="
                f"{remote_queue_low_water_fraction():.0%} "
                "action=fill_to_high_water endpoints=parallel "
                "ingest_coupled=false",
                flush=True,
            )
            # Let the initial prefetch wave reach the servers before measuring
            # it.  Subsequent deadlines use perf_counter so tests (and callers)
            # that patch scheduler monotonic time cannot stall this controller.
            next_probe = time.perf_counter() + interval
            while (
                not producer_stop.is_set()
                and not refill_monitor_stop.is_set()
            ):
                now_counter = time.perf_counter()
                if refill_monitor_stop.wait(
                    max(0.0, next_probe - now_counter)
                ):
                    return
                try:
                    _maybe_refill_low_remote_queues()
                except Exception as refill_exc:
                    print(
                        f"[remote] low_water refill check skipped: {refill_exc!r}",
                        flush=True,
                    )
                next_probe += interval
                now_counter = time.perf_counter()
                if next_probe < now_counter:
                    next_probe = now_counter

        refill_monitor = threading.Thread(
            target=_remote_queue_refill_monitor,
            name="remote-queue-refill-controller",
            daemon=True,
        )
        refill_monitor.start()

        while True:
            with pending_lock:
                if pending <= 0:
                    break
            tag, payload = out_q.get()
            if tag == "done":
                with pending_lock:
                    pending -= 1
                _publish_remote_slots()
                continue
            if tag == "err":
                producer_stop.set()
                raise payload
            yield payload
            now = time.monotonic()
            if now - last_log >= 15.0:
                last_log = now
                try:
                    with claim_lock:
                        rem = len(remaining)
                    dec = scheduler.maybe_tick(remaining=rem, force=False)
                    if dec is not None:
                        active_n = int(sum(slots_by_endpoint.values()))
                        print(
                            f"[pure_rl] mid_iter_rebalance={dec.as_log()} "
                            f"claimed_local={claimed_local} "
                            f"claimed_remote={claimed_remote} "
                            f"remote_slots_live={active_n} remaining={rem} "
                            f"result_buffer={out_q.telemetry()}",
                            flush=True,
                        )
                        try:
                            # Shrink first so a hurt-probe can cut sockets before
                            # any re-grow on the same tick.
                            _maybe_shrink_remote_slots(dec)
                            _maybe_grow_remote_slots(dec)
                        except Exception as slot_exc:
                            print(
                                f"[remote] demand slot adjust skipped: {slot_exc!r}",
                                flush=True,
                            )
                        _publish_remote_slots()
                except Exception:
                    pass

        refill_monitor_stop.set()
        refill_monitor.join(timeout=3.0)
        for thread in threads:
            thread.join(timeout=1.0)
        if errors:
            raise errors[0]
        with claim_lock:
            stranded = len(remaining)
        if stranded:
            raise RemoteJobsError(
                "scheduled dispatch exhausted all emitters with "
                f"{stranded} unclaimed job(s)"
            )
        try:
            dec_f = scheduler.maybe_tick(remaining=0, force=True)
            if dec_f is not None:
                print(
                    f"[pure_rl] mid_iter_rebalance=done {dec_f.as_log()} "
                    f"claimed_local={claimed_local} claimed_remote={claimed_remote}",
                    flush=True,
                )
        except Exception:
            pass
    finally:
        producer_stop.set()
        refill_monitor_stop.set()
        if refill_monitor is not None:
            refill_monitor.join(timeout=3.0)
        for client in owned_clients:
            try:
                client.close()
            except Exception:
                pass
        for thread in threads:
            thread.join(timeout=1.0)
        out_q.close()


@dataclass
class RemoteWorkerFarm:
    """Connected LAN workers used as additive whole-game capacity."""

    endpoints: list[str]
    timeout_s: float = 30.0
    connect_timeout_s: float = field(
        default_factory=lambda: _env_float("POKEBOT_REMOTE_CONNECT_TIMEOUT_S", 60.0)
    )
    control_timeout_s: float = field(
        default_factory=lambda: _env_float("POKEBOT_REMOTE_CONTROL_TIMEOUT_S", 300.0)
    )
    clients: list[RemoteJobClient] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.endpoints = expand_endpoint_specs(list(self.endpoints))

    @property
    def total_workers(self) -> int:
        """Advertised default slots (hello ``workers``); not the pool ceiling."""
        return sum(
            int(client.info.workers)
            for client in self.clients
            if client.info is not None
        )

    @property
    def total_capacity(self) -> int:
        """Process-pool capacity across endpoints (``max_workers`` / headroom)."""
        return sum(
            remote_capacity_workers(client.info)
            for client in self.clients
            if client.info is not None
        )

    def connect(self, *, require_all: bool = False) -> list[RemoteWorkerInfo]:
        """Connect farm endpoints.

        ``require_all=False`` (default): skip dead endpoints so a mid-redeploy
        blip on Elmo does not zero *all* remotes (bert can still take work).
        ``require_all=True``: fail-closed if any endpoint is unreachable.
        """
        self.close()
        infos: list[RemoteWorkerInfo] = []
        errors: list[str] = []
        for endpoint in self.endpoints:
            client = RemoteJobClient(
                *parse_endpoint(endpoint),
                timeout_s=self.timeout_s,
                connect_timeout_s=self.connect_timeout_s,
                control_timeout_s=self.control_timeout_s,
            )
            try:
                infos.append(client.connect())
                self.clients.append(client)
            except Exception as exc:  # noqa: BLE001 — surface per-endpoint
                errors.append(f"{endpoint}: {type(exc).__name__}: {exc}")
                try:
                    client.close()
                except Exception:
                    pass
                if require_all:
                    self.close()
                    raise
                print(
                    f"[remote] connect skip {endpoint} ({type(exc).__name__}: {exc}); "
                    "continuing with live endpoints",
                    flush=True,
                )
        if not infos and self.endpoints:
            raise RemoteJobsError(
                "no remote endpoints reachable: " + "; ".join(errors)
            )
        return infos

    def close(self) -> None:
        for client in self.clients:
            client.close()
        self.clients.clear()

    def __enter__(self) -> "RemoteWorkerFarm":
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _reconnect_missing(self) -> None:
        """Re-attach soft-dropped endpoints from ``self.endpoints`` before reload/pin."""
        alive = {(c.host, int(c.port)) for c in self.clients}
        for endpoint in self.endpoints:
            host, port = parse_endpoint(endpoint)
            key = (host, int(port))
            if key in alive:
                continue
            client = RemoteJobClient(
                host,
                port,
                timeout_s=self.timeout_s,
                connect_timeout_s=self.connect_timeout_s,
                control_timeout_s=self.control_timeout_s,
            )
            try:
                client.connect()
                self.clients.append(client)
                alive.add(key)
                print(
                    f"[remote-farm] reconnected soft-dropped endpoint {endpoint}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                try:
                    client.close()
                except Exception:
                    pass
                print(
                    f"[remote-farm] WARN reconnect failed {endpoint}: {exc}",
                    flush=True,
                )

    def reload_all(
        self,
        path: str,
        *,
        digest: Optional[str] = None,
        version: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        self._reconnect_missing()
        replies: list[dict[str, Any]] = []
        survivors: list[RemoteJobClient] = []
        errors: list[str] = []
        for client in self.clients:
            try:
                replies.append(
                    client.reload_checkpoint(path, digest=digest, version=version)
                )
                survivors.append(client)
                continue
            except TimeoutError as exc:
                # Busy-but-alive: one reconnect+retry, then keep endpoint (no soft-drop).
                print(
                    f"[remote-farm] WARN reload timed out on {client.endpoint}: {exc}; "
                    "retrying once (slow-but-alive)",
                    flush=True,
                )
                try:
                    client.reconnect()
                    replies.append(
                        client.reload_checkpoint(
                            path, digest=digest, version=version
                        )
                    )
                    survivors.append(client)
                    continue
                except TimeoutError as exc2:
                    errors.append(
                        f"{client.endpoint}: timed out (kept; slow-but-alive): {exc2}"
                    )
                    survivors.append(client)
                    print(
                        f"[remote-farm] WARN reload still slow on {client.endpoint}; "
                        "keeping endpoint (not treating as dead)",
                        flush=True,
                    )
                    continue
                except Exception as exc2:  # noqa: BLE001
                    errors.append(f"{client.endpoint}: {exc2}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{client.endpoint}: {exc}")
            try:
                client.close()
            except Exception:
                pass
        self.clients = survivors
        if not self.clients:
            raise RemoteJobsError(
                "remote worker reload failed on all endpoints: "
                + "; ".join(errors)
            )
        dropped = [e for e in errors if "kept; slow-but-alive" not in e]
        if dropped:
            print(
                "[remote-farm] WARN dropped endpoint(s) after reload failure: "
                + "; ".join(dropped),
                flush=True,
            )
        return replies

    def pin_all(
        self,
        path: str,
        *,
        digest: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        self._reconnect_missing()
        replies: list[dict[str, Any]] = []
        survivors: list[RemoteJobClient] = []
        errors: list[str] = []
        for client in self.clients:
            try:
                replies.append(client.pin_checkpoint(path, digest=digest))
                survivors.append(client)
                continue
            except TimeoutError as exc:
                print(
                    f"[remote-farm] WARN pin timed out on {client.endpoint}: {exc}; "
                    "retrying once (slow-but-alive)",
                    flush=True,
                )
                try:
                    client.reconnect()
                    replies.append(client.pin_checkpoint(path, digest=digest))
                    survivors.append(client)
                    continue
                except TimeoutError as exc2:
                    errors.append(
                        f"{client.endpoint}: timed out (kept; slow-but-alive): {exc2}"
                    )
                    survivors.append(client)
                    print(
                        f"[remote-farm] WARN pin still slow on {client.endpoint}; "
                        "keeping endpoint (not treating as dead)",
                        flush=True,
                    )
                    continue
                except Exception as exc2:  # noqa: BLE001
                    errors.append(f"{client.endpoint}: {exc2}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{client.endpoint}: {exc}")
            try:
                client.close()
            except Exception:
                pass
        self.clients = survivors
        if not self.clients:
            raise RemoteJobsError(
                "remote worker pin failed on all endpoints: " + "; ".join(errors)
            )
        dropped = [e for e in errors if "kept; slow-but-alive" not in e]
        if dropped:
            print(
                "[remote-farm] WARN dropped endpoint(s) after pin failure: "
                + "; ".join(dropped),
                flush=True,
            )
        return replies

    def unpin_all(self, digest: str) -> list[dict[str, Any]]:
        return [client.unpin_checkpoint(digest) for client in self.clients]


def iter_remote_results(
    endpoints: list[str],
    jobs: list[dict[str, Any]],
    *,
    kind: str = "play",
    timeout_s: float = 30.0,
) -> Iterator[dict[str, Any]]:
    """Round-robin submit ``jobs`` across endpoints (one connection each).

    Yields remote result dicts. Intended for optional canary / parallel
    collection; does not replace the local WorkerPool path.
    """
    if not endpoints:
        return
    clients = [RemoteJobClient(*parse_endpoint(ep), timeout_s=timeout_s) for ep in endpoints]
    try:
        for client in clients:
            client.connect()
        for i, job in enumerate(jobs):
            client = clients[i % len(clients)]
            yield client.submit_job(job, kind=kind)
    finally:
        for client in clients:
            client.close()


def serve_forever(
    handler: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    host: str = "0.0.0.0",
    port: int = DEFAULT_PORT,
    hello: Optional[Callable[[], dict[str, Any]]] = None,
    backlog: int = 64,
    stop_event: Optional[threading.Event] = None,
    idle_timeout_s: float = 60.0,
    max_connections: int = 128,
) -> None:
    """Accept connections; each connection is handled sequentially (one job at a
    time per socket). Concurrent sockets = concurrent in-flight games, bounded
    by the caller's WorkerPool behind ``handler``.

    ``idle_timeout_s`` bounds each ``recv`` so a wedged peer cannot hang the
    handler thread forever. Idle timeouts are retried (not treated as hangup)
    so long-lived farm sockets survive train/promo gaps between waves.

    ``max_connections`` is a hard thread/socket cap.  In particular, a TCP-only
    container health check must not be able to create an unbounded number of
    daemon threads while the worker is busy.  Malformed probes and clients that
    disconnect mid-frame are normal network noise and are closed quietly.
    """

    connection_limit = threading.BoundedSemaphore(max(1, int(max_connections)))
    active_lock = threading.Lock()
    active_connections: set[socket.socket] = set()
    active_threads: set[threading.Thread] = set()

    def _handle_conn(conn: socket.socket, addr: tuple[str, int]) -> None:
        # Bounded recv so a wedged peer cannot hang the handler thread forever,
        # but idle farm sessions (train/promo between collection waves) must
        # NOT be torn down — treat read timeouts as keep-waiting, not hangup.
        conn.settimeout(float(idle_timeout_s))
        try:
            try:
                first = read_frame(conn)
            except (OSError, RemoteJobsError, TimeoutError, socket.timeout):
                return
            if first.get("type") != "hello":
                try:
                    send_frame(
                        conn,
                        {
                            "type": "error",
                            "error": "expected hello",
                        },
                    )
                except OSError:
                    pass
                return
            if int(first.get("proto", -1)) != PROTO_VERSION:
                try:
                    send_frame(
                        conn,
                        {
                            "type": "error",
                            "error": f"unsupported proto {first.get('proto')}",
                        },
                    )
                except OSError:
                    pass
                return
            info = hello() if hello is not None else {}
            try:
                send_frame(
                    conn,
                    {
                        "type": "hello_ok",
                        "proto": PROTO_VERSION,
                        **info,
                    },
                )
            except OSError:
                return
            while True:
                if stop_event is not None and stop_event.is_set():
                    break
                try:
                    msg = read_frame(conn)
                except (TimeoutError, socket.timeout):
                    # Long-lived RemoteWorkerFarm sockets sit idle for minutes
                    # during local train/promo; closing them here surfaces as
                    # client "connection closed while reading frame".
                    continue
                except (OSError, RemoteJobsError):
                    break
                mtype = msg.get("type")
                if mtype == "bye":
                    break
                if mtype == "ping":
                    try:
                        send_frame(
                            conn,
                            {
                                "type": "pong",
                                "t0": msg.get("t0"),
                                "t1": time.time(),
                            },
                        )
                    except OSError:
                        break
                    continue
                try:
                    reply = handler(msg)
                except BaseException as exc:  # noqa: BLE001 - keep connection alive
                    try:
                        send_frame(
                            conn,
                            {
                                "type": "result" if mtype == "job" else "error",
                                "ok": False,
                                "error": f"{type(exc).__name__}: {exc}",
                            },
                        )
                    except OSError:
                        break
                    continue
                try:
                    send_frame(conn, reply)
                except OSError:
                    break
        finally:
            with active_lock:
                active_connections.discard(conn)
            try:
                conn.close()
            except Exception:
                pass

    def _run_conn(conn: socket.socket, addr: tuple[str, int]) -> None:
        try:
            try:
                _handle_conn(conn, addr)
            except (OSError, RemoteJobsError, TimeoutError, socket.timeout):
                # Last-resort containment for a peer disappearing between a
                # handler operation and its reply. Never emit thread tracebacks
                # for routine health probes or trainer teardown.
                pass
        finally:
            with active_lock:
                active_threads.discard(threading.current_thread())
            connection_limit.release()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(backlog)
    server.settimeout(1.0)
    try:
        while stop_event is None or not stop_event.is_set():
            try:
                conn, addr = server.accept()
            except socket.timeout:
                continue
            if not connection_limit.acquire(blocking=False):
                # Admission is deliberately silent: the peer has not completed
                # the protocol handshake, and blocking here would stall accept.
                try:
                    conn.close()
                except OSError:
                    pass
                continue
            with active_lock:
                active_connections.add(conn)
            thread = threading.Thread(
                target=_run_conn,
                args=(conn, addr),
                name=f"remote-job-{addr[0]}:{addr[1]}",
                daemon=True,
            )
            with active_lock:
                active_threads.add(thread)
            thread.start()
    finally:
        server.close()
        # Closing accepted sockets wakes idle readers immediately instead of
        # leaving one thread per client around until idle_timeout_s expires.
        with active_lock:
            connections = list(active_connections)
            threads = list(active_threads)
        for conn in connections:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                conn.close()
            except OSError:
                pass
        deadline = time.monotonic() + 5.0
        for thread in threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
