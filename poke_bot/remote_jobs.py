"""TCP job protocol for remote poke-bot sim workers (TrueNAS / LAN).

Design intent
-------------
Remote hosts run **CPU sims + co-located GPU leaf** (e.g. TrueNAS elmo 3060 LHR 12 GB).
The training box ships whole-game jobs (and receives records), not per-leaf
RPCs — that keeps LAN chatter to ~1 RTT per game instead of thousands of
leaf forwards per move.

Wire format: length-prefixed JSON frames (``!I`` big-endian uint32 + UTF-8
JSON body).  After a plain hello handshake, peers may negotiate zlib framing;
the high length bit then marks a compressed JSON body. Additive / optional:
local trainers keep using in-process
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
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from poke_bot.public_multi_env_safety import (
    public_multi_env_legacy_opponent_ids,
    public_multi_env_safe_job,
)

try:  # Optional fast, GIL-friendly codec; wire format remains ordinary JSON.
    import orjson as _orjson
except ImportError:  # pragma: no cover - exercised on minimal worker images
    _orjson = None

PROTO_VERSION = 1
DEFAULT_PORT = 8765
_HDR = struct.Struct("!I")
_MAX_FRAME = 256 * 1024 * 1024  # 256 MiB — self-play records can be large
_COMPRESSED_FRAME_FLAG = 0x80000000
_FRAME_COMPRESSION_MIN_BYTES = 64 * 1024
_FRAME_CODEC_ZLIB_V1 = "zlib_frames_v1"


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
_BERT_ROOT = Path(
    os.environ.get(
        "POKEBOT_BERT_REMOTE_ROOT",
        "/Users/tsinzitari/workspace/poke-bot-agent",
    )
).expanduser()
_ELMO_HOSTS = frozenset({"192.168.1.143", "truenas.local", "truenas"})
_BERT_HOSTS = frozenset(
    {"192.168.1.157", "192.168.1.158", "bert.local", "bert"}
)
_BERT_SSH = os.environ.get("POKEBOT_BERT_SSH", "tsinzitari@bert.local")
_ELMO_STAGE_LOCK = threading.RLock()
# A digest identifies checkpoint bytes regardless of which immutable local
# alias named them.  Keying this cache by source path made every historical
# self-play checkpoint alias re-enter the SMB verification path; the first
# thread then held this lock while the other Elmo request sockets sat idle.
_ELMO_STAGE_CACHE: dict[tuple[str, str], str] = {}
_LOCAL_CHECKPOINT_DIGEST_LOCK = threading.RLock()
_LOCAL_CHECKPOINT_DIGEST_CACHE: dict[
    str, tuple[int, int, int, str]
] = {}
_ELMO_STAGE_RECEIPT_ROOT = Path(
    os.environ.get(
        "POKEBOT_ELMO_STAGE_RECEIPT_DIR",
        str(Path.home() / ".cache" / "pokebot" / "elmo-checkpoint-stage"),
    )
).expanduser()
_BERT_STAGE_LOCK = threading.RLock()
_BERT_STAGE_CACHE: dict[tuple[str, str], str] = {}
_MATCHUP_RUNTIME_MARKER_NAME = "matchup-runtime-activation.json"
_MATCHUP_RUNTIME_MARKER_SCHEMA = "poke_bot.remote_matchup_runtime_activation/v1"
_SAFE_ELMO_SSH_STAGE_PATH_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/._-"
)
_SAFE_ELMO_SSH_TARGET_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.@:-[]"
)

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
    endpoint's deficit. Reservations deliberately taper in the final 20% of a
    wave, where keeping the shared local pool eligible matters more than
    steady-state queue depth.

    The caller serializes this object with the dispatcher's claim lock.  It is
    intentionally bookkeeping-only: no health probe, connect, submit, or
    other network operation belongs in these methods.
    """

    def __init__(
        self,
        *,
        total_jobs: int,
        tail_fraction: Optional[float] = None,
        tail_jobs: Optional[int] = None,
    ) -> None:
        total = max(0, int(total_jobs))
        if tail_jobs is not None and int(tail_jobs) > 0:
            self.tail_jobs = min(total, max(1, int(tail_jobs))) if total else 0
            self.tail_fraction = (
                float(self.tail_jobs) / float(total) if total else 0.0
            )
        else:
            configured_fraction = (
                _env_float("POKEBOT_REMOTE_TAIL_WORK_STEAL_FRACTION", 0.20)
                if tail_fraction is None
                else float(tail_fraction)
            )
            fraction = min(0.50, max(0.05, configured_fraction))
            self.tail_fraction = float(fraction)
            self.tail_jobs = (
                max(1, int(total * fraction + 0.999999999)) if total > 0 else 0
            )
        # Endpoint-credit spill keeps its established final-5% escape hatch.
        # The broader configurable window is consumed only by the mixed
        # dispatcher, which can prove that a local work-stealing pool exists.
        self.credit_spill_tail_jobs = max(1, (total + 19) // 20) if total else 0
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

    def note_released(self, endpoint: str, n: int) -> None:
        """Return controller-owned jobs to the exact shared pool."""
        ep = str(endpoint)
        count = max(0, int(n))
        inflight = max(0, int(self.inflight.get(ep, 0)))
        claimed = max(0, int(self.claimed_total.get(ep, 0)))
        if count > inflight or count > claimed:
            raise RuntimeError(
                f"remote endpoint credit release underflow for {ep}: "
                f"released={count} inflight={inflight} claimed={claimed}"
            )
        self.inflight[ep] = inflight - count
        self.claimed_total[ep] = claimed - count

    def in_tail(self, wave_remaining: int) -> bool:
        return max(0, int(wave_remaining)) <= int(self.tail_jobs)

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
        # Preserve the endpoint allocator's final-5% spill escape hatch.
        # Mixed collection applies its broader 20% one-game cap in the
        # dispatcher, where it knows a local claimant actually exists.
        if max(0, int(wave_remaining)) <= self.credit_spill_tail_jobs:
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


def remote_socket_prefetch_factor(*, kind: Optional[str] = None) -> int:
    """Return request sockets opened per remote execution worker.

    Public ``play`` has its own reserve-depth knob so a one-extra-request
    queue can keep Bert/Elmo fed without multiplying the already four-game
    ``self_play_multi`` batches.  Callers without a collection kind retain the
    legacy generic setting for compatibility.
    """
    name = (
        "POKEBOT_REMOTE_PUBLIC_SOCKET_PREFETCH"
        if str(kind).strip().lower() == "play"
        else "POKEBOT_REMOTE_SOCKET_PREFETCH"
    )
    try:
        value = int(os.environ.get(name, "1") or "1")
    except (TypeError, ValueError):
        value = 1
    return max(1, min(8, value))


def remote_socket_prefetch_max_factor(*, kind: Optional[str] = None) -> int:
    """Return the hard per-worker request-socket ceiling for low-water refill."""
    base = remote_socket_prefetch_factor(kind=kind)
    name = (
        "POKEBOT_REMOTE_PUBLIC_SOCKET_PREFETCH_MAX"
        if str(kind).strip().lower() == "play"
        else "POKEBOT_REMOTE_SOCKET_PREFETCH_MAX"
    )
    try:
        value = int(os.environ.get(name, str(base)) or str(base))
    except (TypeError, ValueError):
        value = base
    return max(base, min(8, value))


def remote_socket_target(
    execution_workers: int, *, kind: Optional[str] = None
) -> int:
    """Map scheduled execution-worker demand to queued request sockets."""
    return max(0, int(execution_workers)) * remote_socket_prefetch_factor(
        kind=kind
    )


def remote_socket_max_target(
    execution_workers: int, *, kind: Optional[str] = None
) -> int:
    """Map execution demand to the bounded low-water reserve ceiling."""
    return max(0, int(execution_workers)) * remote_socket_prefetch_max_factor(
        kind=kind
    )


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


def remote_self_play_multi_games() -> int:
    """Games packed into one ``self_play_multi`` socket job.

    Owner override of GOAL r112/r124 single-game-per-socket for LAN remotes
    that advertise ``job_kinds`` including ``self_play_multi``. Default ``4``.
    Set ``POKEBOT_REMOTE_SELF_PLAY_MULTI_GAMES=0`` (or ``1``) to keep the
    historical one-game-per-socket submit path even when workers advertise
    the multi kind.
    """
    raw = os.environ.get("POKEBOT_REMOTE_SELF_PLAY_MULTI_GAMES", "4")
    try:
        value = int(raw or "4")
    except (TypeError, ValueError):
        value = 4
    return max(0, min(64, value))


try:
    PUBLIC_MULTI_ENV_LEGACY_OPPONENT_IDS = public_multi_env_legacy_opponent_ids()
except RuntimeError:
    # Missing/corrupt deployment metadata disables public packing through the
    # lazy classifier below; ordinary singleton remote play stays available.
    PUBLIC_MULTI_ENV_LEGACY_OPPONENT_IDS = frozenset()


def _client_supports_self_play_multi(client: "RemoteJobClient") -> bool:
    """Whether a remote advertised the existing multi-env transport kind."""

    info = getattr(client, "info", None)
    return "self_play_multi" in set(getattr(info, "job_kinds", ()) or ())


def client_self_play_multi_pack(
    client: "RemoteJobClient",
    *,
    kind: str,
) -> int:
    """Return pack size when this endpoint should receive ``self_play_multi``.

    Only true self-play is safe to pack. Public baseline agents use the
    process-global ``cg.game`` singleton and must keep the ordinary one-game
    ``play`` path even when a worker advertises ``self_play_multi``. Remotes
    remain engaged; this changes only the per-socket public-play packing.
    """
    if str(kind) != "self_play":
        return 0
    pack = remote_self_play_multi_games()
    if pack <= 1:
        return 0
    if not _client_supports_self_play_multi(client):
        return 0
    return pack


def _client_public_multi_env_pack(client: "RemoteJobClient") -> int:
    """Return the public multi-env pack size after endpoint capability gating.

    The job-level r182 allowlist is deliberately checked separately, before a
    child is placed into a pack.  Both the existing transport kind and the
    new worker capability are required, so an unupdated endpoint that happens
    to advertise ``self_play_multi`` cannot receive public packs.
    """

    pack = remote_self_play_multi_games()
    info = getattr(client, "info", None)
    capabilities = set(getattr(info, "capabilities", ()) or ())
    if (
        pack <= 1
        or not _client_supports_self_play_multi(client)
        or "public_multi_env_allowlist_r182_v1" not in capabilities
    ):
        return 0
    # The public LibcgMultiEnv contract is intentionally narrower than the
    # generic self-play transport knob.  Larger generic batches are valid for
    # self-play, but public packets must remain one exact-safe pack of at most
    # four independently accounted collection children.
    return min(4, pack)


def _remote_transport_packets(
    client: "RemoteJobClient",
    *,
    kind: str,
    jobs: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Stable-partition claimed logical jobs into transport packets.

    Public play is default-singleton.  Only exact r182-safe children form
    packets of up to four; every legacy, malformed, or provenance-drifted
    child remains its own ordinary ``play`` request.  The safe and singleton
    partitions retain their original internal order, which permits client and
    server result identity checks to remain one logical result per job.
    """

    if not jobs:
        return []
    if str(kind) == "self_play":
        pack = client_self_play_multi_pack(client, kind=kind)
        if pack <= 1:
            return [[job] for job in jobs]
        return [jobs[index : index + pack] for index in range(0, len(jobs), pack)]
    if str(kind) != "play":
        return [[job] for job in jobs]
    pack = _client_public_multi_env_pack(client)
    if pack <= 1:
        return [[job] for job in jobs]
    safe = [job for job in jobs if public_multi_env_safe_job(job)]
    singleton = [job for job in jobs if not public_multi_env_safe_job(job)]
    packets = [safe[index : index + pack] for index in range(0, len(safe), pack)]
    packets.extend([[job] for job in singleton])
    return packets


def _remote_packet_uses_multi_env(
    *, kind: str, packet: list[dict[str, Any]]
) -> bool:
    """Return whether this packet must use the multi-env wire transport."""

    if len(packet) <= 1:
        return False
    if str(kind) == "self_play":
        return True
    return str(kind) == "play" and all(public_multi_env_safe_job(job) for job in packet)


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


def _path_is_dir_bounded(
    path: Path, *, timeout_s: float = 1.0
) -> tuple[bool, bool]:
    """Return ``(is_dir, timed_out)`` without hanging on stuck gvfs."""

    if timeout_s <= 0:
        try:
            return path.is_dir(), False
        except OSError:
            return False, False

    result: dict[str, bool] = {"is_dir": False}

    def _probe() -> None:
        try:
            result["is_dir"] = bool(path.is_dir())
        except OSError:
            result["is_dir"] = False

    probe = threading.Thread(target=_probe, name="smb-dir-probe", daemon=True)
    probe.start()
    probe.join(timeout=timeout_s)
    if probe.is_alive():
        return False, True
    return bool(result["is_dir"]), False


def _smb_checkpoint_probe_timeout_s() -> float:
    """Bound one trainer-side GVFS probe before choosing SSH publication."""

    try:
        timeout_s = float(
            os.environ.get("POKEBOT_ELMO_SMB_PROBE_TIMEOUT_S", "1") or "1"
        )
    except (TypeError, ValueError):
        timeout_s = 1.0
    return max(0.05, min(10.0, timeout_s))


def _smb_checkpoint_dir() -> Path | None:
    timeout_s = _smb_checkpoint_probe_timeout_s()
    explicit = os.environ.get("POKEBOT_TRUENAS_CHECKPOINT_SMB")
    if explicit:
        path = Path(explicit)
        is_dir, _timed_out = _path_is_dir_bounded(path, timeout_s=timeout_s)
        return path if is_dir else None
    uid = os.getuid()
    # Host Docker compose mounts containers/truenas-worker/checkpoint →
    # /workspace/checkpoint (NOT the top-level poke-bot-agent/checkpoint/).
    # Probe gvfs with a bound timeout: a wedged mount previously hung every
    # Elmo resolve and starved self_play_multi packing. The NFS mirror under
    # /tmp/truenas_main is read-only and is intentionally not a stage target.
    candidates = [
        Path(
            f"/run/user/{uid}/gvfs/smb-share:server=truenas.local,"
            "share=main/poke-bot-agent/containers/truenas-worker/checkpoint"
        ),
        Path(
            f"/run/user/{uid}/gvfs/smb-share:server=truenas.local,"
            "share=main/poke-bot-agent/checkpoint"
        ),
        Path(
            f"/run/user/{uid}/gvfs/smb-share:server=192.168.1.143,"
            "share=main,user=inzi/poke-bot-agent/containers/"
            "truenas-worker/checkpoint"
        ),
        Path(
            f"/run/user/{uid}/gvfs/smb-share:server=192.168.1.143,"
            "share=main,user=inzi/poke-bot-agent/checkpoint"
        ),
    ]
    for candidate in candidates:
        is_dir, timed_out = _path_is_dir_bounded(candidate, timeout_s=timeout_s)
        if is_dir:
            return candidate
        if timed_out:
            # One wedged gvfs probe is enough; further siblings share the mount.
            return None
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


def _cached_local_checkpoint_digest(src: Path) -> str:
    """Hash immutable local checkpoint bytes once per exact file identity.

    Remote jobs still fail closed when a path is replaced or modified: size,
    nanosecond mtime, and inode must all match. The lock intentionally covers
    the first hash so 36 request threads do not concurrently read the same
    128 MiB parent before discovering the shared Elmo mapping cache.
    """

    from .checkpoint import checkpoint_digest

    resolved = Path(src).expanduser().resolve()
    with _LOCAL_CHECKPOINT_DIGEST_LOCK:
        before = resolved.stat()
        identity = (
            int(before.st_size),
            int(before.st_mtime_ns),
            int(before.st_ino),
        )
        cached = _LOCAL_CHECKPOINT_DIGEST_CACHE.get(str(resolved))
        if cached is not None and cached[:3] == identity:
            return cached[3]
        digest = checkpoint_digest(resolved)
        after = resolved.stat()
        after_identity = (
            int(after.st_size),
            int(after.st_mtime_ns),
            int(after.st_ino),
        )
        if after_identity != identity:
            raise RemoteJobsError(
                f"checkpoint changed while hashing: {resolved}"
            )
        _LOCAL_CHECKPOINT_DIGEST_CACHE[str(resolved)] = (
            *identity,
            digest,
        )
        return digest


def digest_addressed_basename(src: Path, digest: Optional[str] = None) -> str:
    """Stable remote filename that cannot collide across distinct checkpoint bytes.

    Elmo only bind-mounts a flat ``checkpoint/`` dir. Staging as bare
    ``iter_00001.pt`` overwrote prior digests and broke pin/reload on the
    long-lived worker (expected old sha, file had new sha).
    """
    resolved = Path(src).expanduser()
    dig = digest or _cached_local_checkpoint_digest(resolved)
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


def _elmo_stage_receipt_path(dest: Path) -> Path:
    key = hashlib.sha256(str(dest).encode("utf-8")).hexdigest()
    return _ELMO_STAGE_RECEIPT_ROOT / f"{key}.json"


def _write_elmo_stage_receipt(src: Path, dest: Path, digest: str) -> Path:
    """Persist one exact content-addressed SMB verification."""

    source_stat = src.stat()
    destination_stat = dest.stat()
    payload = {
        "schema": "poke_bot.elmo_checkpoint_stage_receipt/v1",
        "checkpoint_digest": str(digest),
        "source_path": str(src),
        "source_size": int(source_stat.st_size),
        "destination_path": str(dest),
        "destination_name": dest.name,
        "destination_size": int(destination_stat.st_size),
        "destination_mtime_ns": int(destination_stat.st_mtime_ns),
        "verified_at_unix": time.time(),
    }
    receipt = _elmo_stage_receipt_path(dest)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt.with_name(receipt.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, receipt)
    return receipt


def _elmo_stage_receipt_is_exact(src: Path, dest: Path, digest: str) -> bool:
    """Validate a content receipt plus current SMB metadata, not payload.

    ``source_path`` remains recorded for provenance, but it is not identity.
    Two immutable local aliases with the same SHA-256 and size are the same
    checkpoint and may safely reuse one digest-addressed remote object.
    """

    try:
        payload = json.loads(
            _elmo_stage_receipt_path(dest).read_text(encoding="utf-8")
        )
        source_stat = src.stat()
        destination_stat = dest.stat()
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return bool(
        payload.get("schema") == "poke_bot.elmo_checkpoint_stage_receipt/v1"
        and payload.get("checkpoint_digest") == str(digest)
        and int(payload.get("source_size", -1)) == int(source_stat.st_size)
        and payload.get("destination_path") == str(dest)
        and payload.get("destination_name") == dest.name
        and int(payload.get("destination_size", -1))
        == int(destination_stat.st_size)
        and int(payload.get("destination_mtime_ns", -1))
        == int(destination_stat.st_mtime_ns)
        and int(destination_stat.st_size) == int(source_stat.st_size)
    )


def _elmo_receipt_destination_for_digest(
    src: Path,
    smb: Path,
    digest: str,
) -> Optional[Path]:
    """Return any still-exact content-addressed object for ``digest``.

    The source stem is deliberately irrelevant. A population snapshot may
    refer to identical bytes as ``iter_00002.pt`` and ``parent.pt``; retaining
    both remote copies would defeat content addressing and reintroduce SMB
    verification on process restart.
    """

    try:
        receipts = tuple(_ELMO_STAGE_RECEIPT_ROOT.glob("*.json"))
        source_size = int(src.stat().st_size)
        resolved_smb = smb.resolve()
    except OSError:
        return None
    for receipt in receipts:
        try:
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            if (
                payload.get("schema")
                != "poke_bot.elmo_checkpoint_stage_receipt/v1"
                or payload.get("checkpoint_digest") != str(digest)
                or int(payload.get("source_size", -1)) != source_size
            ):
                continue
            candidate = Path(str(payload.get("destination_path") or ""))
            candidate_stat = candidate.stat()
            if candidate.parent.resolve() != resolved_smb:
                continue
            if (
                not candidate.is_file()
                or candidate.name != payload.get("destination_name")
                or int(candidate_stat.st_size)
                != int(payload.get("destination_size", -1))
                or int(candidate_stat.st_mtime_ns)
                != int(payload.get("destination_mtime_ns", -1))
                or int(candidate_stat.st_size) != source_size
            ):
                continue
            return candidate
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return None


def _elmo_remote_checkpoint_digest(host: str, remote_path: str) -> Optional[str]:
    """Hash an Elmo checkpoint on TrueNAS storage, never through SMB.

    Old content-addressed objects can predate local stage receipts.  Reading a
    100+ MiB file back over gvfs merely to prove bytes already resident on
    TrueNAS took roughly 141 seconds and serialized every Elmo dispatch thread.
    A capability-gated control call performs the same SHA-256 next to storage.
    Returning ``None`` preserves the fail-safe legacy verifier for older
    workers rather than trusting a filename or size alone.
    """

    client = RemoteJobClient(
        host,
        DEFAULT_PORT,
        timeout_s=30.0,
        connect_timeout_s=min(
            5.0, _env_float("POKEBOT_REMOTE_CONNECT_TIMEOUT_S", 60.0)
        ),
        control_timeout_s=min(
            30.0, _env_float("POKEBOT_REMOTE_CONTROL_TIMEOUT_S", 300.0)
        ),
    )
    try:
        info = client.connect()
        if "checkpoint_digest_verify_v1" not in set(info.capabilities):
            return None
        reply = client.verify_checkpoint(remote_path)
        digest = str(reply.get("checkpoint_digest") or "")
        return digest if digest.startswith("sha256:") else None
    except (TimeoutError, OSError, RemoteJobsError):
        return None
    finally:
        client.close()


def _elmo_ssh_stage_config() -> Optional[tuple[str, str]]:
    """Return the explicit host-side Elmo stage target, if enabled.

    Elmo deliberately exposes ``/workspace/checkpoint`` to its worker as a
    read-only bind.  A trainer-local GVFS mount is convenient, but it is not a
    publication requirement: a new immutable checkpoint can instead be copied
    to that same bind source over the already-authenticated LAN SSH transport.

    This fallback is opt-in because it writes to a remote host.  It is used
    only after the storage-local worker verifier has proved that the requested
    digest is *not* resident, and the caller verifies it again before reload.
    """

    if not _env_truthy("POKEBOT_ELMO_SSH_STAGE"):
        return None
    target = os.environ.get("POKEBOT_ELMO_SSH", "admin@elmo").strip()
    checkpoint_dir = os.environ.get(
        "POKEBOT_ELMO_CHECKPOINT_HOST_DIR", ""
    ).strip()
    if not target:
        raise RemoteJobsError(
            "POKEBOT_ELMO_SSH_STAGE requires a nonempty POKEBOT_ELMO_SSH"
        )
    if target.startswith("-") or any(
        character not in _SAFE_ELMO_SSH_TARGET_CHARS for character in target
    ):
        raise RemoteJobsError("POKEBOT_ELMO_SSH is not a safe SSH target")
    if not checkpoint_dir:
        raise RemoteJobsError(
            "POKEBOT_ELMO_SSH_STAGE requires "
            "POKEBOT_ELMO_CHECKPOINT_HOST_DIR"
        )
    if (
        not checkpoint_dir.startswith("/")
        or any(
            character not in _SAFE_ELMO_SSH_STAGE_PATH_CHARS
            for character in checkpoint_dir
        )
    ):
        raise RemoteJobsError(
            "POKEBOT_ELMO_CHECKPOINT_HOST_DIR must be an absolute safe path"
        )
    checkpoint_dir = checkpoint_dir.rstrip("/")
    if not checkpoint_dir:
        raise RemoteJobsError(
            "POKEBOT_ELMO_CHECKPOINT_HOST_DIR may not be filesystem root"
        )
    return target, checkpoint_dir


def _elmo_ssh_transport_options() -> list[str]:
    """Return noninteractive SSH/SCP options for one bounded stage transfer."""

    options = [
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "StrictHostKeyChecking=yes",
    ]
    identity_raw = os.environ.get("POKEBOT_ELMO_SSH_IDENTITY_FILE", "").strip()
    if identity_raw:
        identity = Path(identity_raw).expanduser()
        if not identity.is_file():
            raise RemoteJobsError(
                "POKEBOT_ELMO_SSH_IDENTITY_FILE is not a readable file: "
                f"{identity}"
            )
        options.extend(["-i", str(identity), "-o", "IdentitiesOnly=yes"])
    return options


def _elmo_ssh_stage_timeout_s() -> float:
    """Bound an SSH publish so a wedged transport cannot stall a hard gate."""

    try:
        timeout_s = float(
            os.environ.get("POKEBOT_ELMO_SSH_STAGE_TIMEOUT_S", "900") or "900"
        )
    except (TypeError, ValueError):
        timeout_s = 900.0
    return max(30.0, min(3600.0, timeout_s))


def _stage_elmo_file_via_ssh(
    src: Path,
    *,
    destination_name: str,
    digest: str,
) -> bool:
    """Atomically stage one digest-addressed Elmo bind-source file over SSH.

    The transfer lands under a unique same-directory partial name.  Elmo
    hashes the partial bytes before an atomic rename and the caller then asks
    the worker's storage-local ``checkpoint_digest_verify_v1`` verifier to
    hash the published object again.  A failed transfer never exposes partial
    bytes at the path a worker can reload.
    """

    config = _elmo_ssh_stage_config()
    if config is None:
        return False
    target, checkpoint_dir = config
    source = Path(src).expanduser().resolve()
    if not source.is_file():
        raise RemoteJobsError(f"local checkpoint missing for SSH stage: {source}")
    if (
        Path(destination_name).name != destination_name
        or destination_name in {"", ".", ".."}
        or any(
            character not in _SAFE_ELMO_SSH_STAGE_PATH_CHARS - {"/"}
            for character in destination_name
        )
    ):
        raise RemoteJobsError(
            f"unsafe Elmo SSH stage destination name: {destination_name!r}"
        )
    expected_digest = str(digest)
    expected_hex = expected_digest.removeprefix("sha256:")
    if (
        not expected_digest.startswith("sha256:")
        or len(expected_hex) != 64
        or any(character not in "0123456789abcdef" for character in expected_hex)
    ):
        raise RemoteJobsError(
            f"invalid Elmo SSH stage checkpoint digest: {expected_digest!r}"
        )
    if _cached_local_checkpoint_digest(source) != expected_digest:
        raise RemoteJobsError(
            f"local checkpoint changed before Elmo SSH stage: {source}"
        )

    # ``scp`` writes only the unique partial object.  The final object remains
    # invisible to the worker until the storage-local SHA-256 has passed and
    # the same-directory rename commits.
    nonce = f"{os.getpid()}.{threading.get_ident()}.{expected_hex[:16]}"
    remote_destination = f"{checkpoint_dir}/{destination_name}"
    remote_partial = f"{checkpoint_dir}/.{destination_name}.partial.{nonce}"
    options = _elmo_ssh_transport_options()
    timeout_s = _elmo_ssh_stage_timeout_s()
    transfer = [
        "scp",
        *options,
        str(source),
        f"{target}:{remote_partial}",
    ]
    try:
        copied = subprocess.run(
            transfer,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise RemoteJobsError(
            "Elmo SSH checkpoint transfer timed out before publication "
            f"after {timeout_s:.0f}s"
        ) from exc
    if copied.returncode != 0:
        detail = (copied.stderr or copied.stdout or str(copied.returncode)).strip()
        raise RemoteJobsError(f"Elmo SSH checkpoint transfer failed: {detail}")

    # Do not rely on filename or byte size.  If another publisher raced this
    # immutable destination, replacing it is safe only after the staged source
    # itself has its requested digest; the caller's subsequent worker-local
    # verification remains the final authority before reload/pin.
    script = "\n".join(
        (
            "set -eu",
            f"partial={shlex.quote(remote_partial)}",
            f"destination={shlex.quote(remote_destination)}",
            f"expected={shlex.quote(expected_hex)}",
            "cleanup() { rm -f -- \"$partial\"; }",
            "trap cleanup EXIT HUP INT TERM",
            "test -f \"$partial\"",
            "actual=$(sha256sum -- \"$partial\" | awk '{print $1}')",
            "test \"$actual\" = \"$expected\"",
            "chmod 0444 -- \"$partial\"",
            "if test -e \"$destination\"; then",
            "  test -f \"$destination\"",
            "fi",
            "mv -f -- \"$partial\" \"$destination\"",
            "actual=$(sha256sum -- \"$destination\" | awk '{print $1}')",
            "test \"$actual\" = \"$expected\"",
            "trap - EXIT HUP INT TERM",
        )
    )
    publish = [
        "ssh",
        *options,
        target,
        "sh -ceu " + shlex.quote(script),
    ]
    try:
        published = subprocess.run(
            publish,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise RemoteJobsError(
            "Elmo SSH checkpoint checksum/publish timed out "
            f"after {timeout_s:.0f}s"
        ) from exc
    if published.returncode != 0:
        detail = (
            published.stderr or published.stdout or str(published.returncode)
        ).strip()
        raise RemoteJobsError(f"Elmo SSH checkpoint checksum/publish failed: {detail}")
    if _cached_local_checkpoint_digest(source) != expected_digest:
        raise RemoteJobsError(
            f"local checkpoint changed during Elmo SSH stage: {source}"
        )
    return True


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


def _stage_elmo_runtime_companions_via_ssh() -> bool:
    """Publish runtime tree/marker beside an SSH-staged Elmo checkpoint.

    The worker validates these companions during reload, so an SSH fallback
    must use the same byte-bound activation files as the ordinary SMB path.
    ``True`` means staging is either unnecessary or completed; ``False``
    means host-side SSH staging is not enabled.
    """

    companions = _matchup_runtime_companions()
    if companions is None:
        return True
    tree, marker = companions
    tree_digest = "sha256:" + hashlib.sha256(tree.read_bytes()).hexdigest()
    if not _stage_elmo_file_via_ssh(
        tree,
        destination_name=tree.name,
        digest=tree_digest,
    ):
        return False

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
        return _stage_elmo_file_via_ssh(
            marker_path,
            destination_name=_MATCHUP_RUNTIME_MARKER_NAME,
            digest=marker_digest,
        )
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
    digest = _cached_local_checkpoint_digest(src)
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
            # The cache key binds immutable checkpoint bytes for this trainer
            # process. Runtime companions are staged in the same critical
            # section before the cache entry is published, so repeating their
            # SSH digest checks here would serialize every game submission.
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

    Elmo (TrueNAS host Docker): stage bytes onto the ``checkpoint/`` bind
    source through an available verified transport, then reload
    ``/workspace/checkpoint/<basename>``.  GVFS SMB is preferred when mounted;
    an explicitly enabled checksum-verified SSH host stage handles new
    digests when GVFS is unavailable.
    Bert: stage bytes onto the native checkout (gvfs SFTP or SSH rsync), then
    remap the training-box repo root onto ``/Users/tsinzitari/workspace/...``.
    """
    raw = str(local_path)
    if raw.startswith("/workspace/checkpoint/"):
        return raw
    src = Path(raw).expanduser().resolve()
    host_l = host.strip().lower()
    if host_l in _ELMO_HOSTS:
        if not src.is_file():
            raise RemoteJobsError(f"local checkpoint missing for stage: {src}")
        # Digest-addressed basename: pins/reloads stay valid across iters/runs.
        from .checkpoint import checkpoint_digest

        digest = _cached_local_checkpoint_digest(src)
        dest_name = digest_addressed_basename(src, digest=digest)
        remote_path = f"/workspace/checkpoint/{dest_name}"
        resident_key = (digest, "elmo:/workspace/checkpoint")
        with _ELMO_STAGE_LOCK:
            cached = _ELMO_STAGE_CACHE.get(resident_key)
            if cached is not None:
                return cached

        smb = _smb_checkpoint_dir()
        if smb is None:
            # BETWEEN_ITER / prior stage can leave a digest-addressed object
            # resident even when GVFS is missing or wedged. For a genuinely
            # new digest, use the opt-in host-side publisher rather than an
            # interactive GUI mount. Hold the rare publish under the existing
            # digest lock: a hard-gate has many client calls, but one immutable
            # object must be copied and verified once.
            with _ELMO_STAGE_LOCK:
                cached = _ELMO_STAGE_CACHE.get(resident_key)
                if cached is not None:
                    return cached
                remote_digest = _elmo_remote_checkpoint_digest(
                    host, remote_path
                )
                if remote_digest != digest:
                    staged = _stage_elmo_file_via_ssh(
                        src,
                        destination_name=dest_name,
                        digest=digest,
                    )
                    if not staged:
                        raise RemoteJobsError(
                            "TrueNAS checkpoint is not resident on Elmo and "
                            "no writable trainer SMB mount or enabled "
                            "checksum-verified SSH stage is available; set "
                            "POKEBOT_ELMO_SSH_STAGE=1 with "
                            "POKEBOT_ELMO_CHECKPOINT_HOST_DIR"
                        )
                    if not _stage_elmo_runtime_companions_via_ssh():
                        raise RemoteJobsError(
                            "Elmo SSH checkpoint stage is enabled but runtime "
                            "companions could not be published"
                        )
                    remote_digest = _elmo_remote_checkpoint_digest(
                        host, remote_path
                    )
                if remote_digest != digest:
                    raise RemoteJobsError(
                        "Elmo SSH checkpoint stage did not pass storage-local "
                        "digest verification: "
                        f"path={remote_path} expected={digest} "
                        f"actual={remote_digest!r}"
                    )
                _ELMO_STAGE_CACHE[resident_key] = remote_path
                return remote_path
        dest = smb / dest_name
        cache_key = (digest, str(smb))
        with _ELMO_STAGE_LOCK:
            cached = _ELMO_STAGE_CACHE.get(cache_key)
            if cached is not None:
                # Companion staging completed before this immutable cache
                # entry was published. Re-hashing the SMB files on every game
                # serializes all remote request sockets and starves Elmo.
                return cached
            receipt_destination = _elmo_receipt_destination_for_digest(
                src, smb, digest
            )
            if receipt_destination is not None:
                _stage_elmo_runtime_companions(smb)
                result = (
                    f"/workspace/checkpoint/{receipt_destination.name}"
                )
                _ELMO_STAGE_CACHE[cache_key] = result
                _ELMO_STAGE_CACHE[resident_key] = result
                return result
            destination_verified = False
            if dest.is_file() and dest.stat().st_size == src.stat().st_size:
                if _elmo_stage_receipt_is_exact(src, dest, digest):
                    destination_verified = True
                else:
                    remote_digest = _elmo_remote_checkpoint_digest(
                        host, remote_path
                    )
                    if remote_digest is not None:
                        destination_verified = remote_digest == digest
                    else:
                        # Compatibility fallback for an older worker. This is
                        # intentionally safe but slow and disappears once the
                        # advertised verify capability is deployed.
                        try:
                            destination_verified = (
                                checkpoint_digest(dest) == digest
                            )
                        except OSError:
                            destination_verified = False
            if not destination_verified:
                try:
                    _gvfs_safe_copy(src, dest)
                except OSError as exc:
                    raise RemoteJobsError(
                        "TrueNAS checkpoint not writable for stage "
                        f"({smb}): {exc}"
                    ) from exc
                if checkpoint_digest(dest) != digest:
                    raise RemoteJobsError("Elmo gvfs checkpoint digest mismatch")
            _write_elmo_stage_receipt(src, dest, digest)
            _stage_elmo_runtime_companions(smb)
            result = f"/workspace/checkpoint/{dest_name}"
            _ELMO_STAGE_CACHE[cache_key] = result
            _ELMO_STAGE_CACHE[resident_key] = result
            return result
    if host_l in _BERT_HOSTS:
        return _stage_bert_checkpoint(src)
    return str(src)


def cache_exact_resident_remote_checkpoint(
    host: str,
    local_path: str,
    *,
    digest: str,
) -> Optional[str]:
    """Cache an Elmo mapping after a complete remote resident-health proof.

    The caller must have just verified that the controller and every leaf are
    healthy on ``digest`` and that the digest is pinned. That proof is stronger
    and newer than re-reading the same digest-addressed object over SMB. We
    still require the expected object and exact byte size before caching its
    deterministic path. A new digest or missing object continues through the
    ordinary copy-and-checksum staging path.
    """

    host_l = host.strip().lower()
    if host_l not in _ELMO_HOSTS:
        return None
    dig = str(digest)
    if not dig.startswith("sha256:"):
        raise RemoteJobsError(f"invalid resident checkpoint digest: {dig!r}")
    src = Path(local_path).expanduser().resolve()
    if not src.is_file():
        raise RemoteJobsError(f"resident checkpoint source is missing: {src}")
    dest_name = digest_addressed_basename(src, digest=dig)
    result = f"/workspace/checkpoint/{dest_name}"
    resident_key = (dig, "elmo:/workspace/checkpoint")
    smb = _smb_checkpoint_dir()
    if smb is not None:
        dest = smb / dest_name
        if not dest.is_file() or dest.stat().st_size != src.stat().st_size:
            raise RemoteJobsError(
                "resident checkpoint object is missing or has the wrong byte "
                f"size: {dest}"
            )
        cache_key = (dig, str(smb))
        with _ELMO_STAGE_LOCK:
            _ELMO_STAGE_CACHE[cache_key] = result
            _ELMO_STAGE_CACHE[resident_key] = result
            _write_elmo_stage_receipt(src, dest, dig)
        return result
    # Health proof already verified digest residency; cache the deterministic
    # Elmo bind-mount path even when the trainer's SMB/gvfs mount is down.
    with _ELMO_STAGE_LOCK:
        _ELMO_STAGE_CACHE[resident_key] = result
    return result


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


def encode_frame(payload: dict[str, Any], *, compress: bool = False) -> bytes:
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
    wire_body = body
    wire_length = len(body)
    if bool(compress) and len(body) >= _FRAME_COMPRESSION_MIN_BYTES:
        compressed = zlib.compress(body, level=1)
        if len(compressed) < len(body):
            wire_body = compressed
            wire_length = len(compressed) | _COMPRESSED_FRAME_FLAG
    return _HDR.pack(wire_length) + wire_body


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
    (wire_length,) = _HDR.unpack(header)
    compressed = bool(wire_length & _COMPRESSED_FRAME_FLAG)
    length = wire_length & ~_COMPRESSED_FRAME_FLAG
    if length > _MAX_FRAME:
        raise RemoteJobsError(f"frame length {length} exceeds max {_MAX_FRAME}")
    body = _recvexact(sock, length)
    if compressed:
        try:
            body = zlib.decompress(body)
        except zlib.error as exc:
            raise RemoteJobsError(f"invalid compressed frame: {exc}") from exc
        if len(body) > _MAX_FRAME:
            raise RemoteJobsError(
                f"decompressed frame length {len(body)} exceeds max {_MAX_FRAME}"
            )
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


def send_frame(
    sock: socket.socket,
    payload: dict[str, Any],
    *,
    compress: bool = False,
) -> None:
    sock.sendall(encode_frame(payload, compress=compress))


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
        self._compress_frames = False

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
                "frame_codecs": [_FRAME_CODEC_ZLIB_V1],
            },
        )
        reply = read_frame(sock)
        if reply.get("type") != "hello_ok" or int(reply.get("proto", -1)) != PROTO_VERSION:
            raise RemoteJobsError(f"unexpected hello reply: {reply!r}")
        raw_frame_codecs = reply.get("frame_codecs") or []
        if isinstance(raw_frame_codecs, str):
            frame_codecs = {
                value.strip() for value in raw_frame_codecs.split(",") if value.strip()
            }
        else:
            frame_codecs = {str(value) for value in raw_frame_codecs}
        self._compress_frames = _FRAME_CODEC_ZLIB_V1 in frame_codecs
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
                send_frame(
                    sock,
                    {"type": "bye"},
                    compress=self._compress_frames,
                )
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
        self._compress_frames = False
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

    def _send(self, sock: socket.socket, payload: dict[str, Any]) -> None:
        send_frame(sock, payload, compress=self._compress_frames)

    def ping(self) -> dict[str, Any]:
        sock = self._require_sock()
        prev = sock.gettimeout()
        sock.settimeout(self.control_timeout_s)
        try:
            self._send(sock, {"type": "ping", "t0": time.time()})
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
            self._send(sock, {"type": "health"})
            reply = read_frame(sock)
        finally:
            try:
                sock.settimeout(prev)
            except Exception:
                pass
        if reply.get("type") != "health_ok":
            raise RemoteJobsError(f"unexpected health reply: {reply!r}")
        return reply

    def verify_checkpoint(self, path: str) -> dict[str, Any]:
        """Return a storage-local SHA-256 proof for one checkpoint object."""

        reply = self._control_call(
            {"type": "verify_checkpoint", "path": str(path)}
        )
        if reply.get("type") != "verify_checkpoint_ok" or not reply.get(
            "ok", False
        ):
            raise RemoteJobsError(
                "remote checkpoint verification failed: "
                f"{reply.get('error') or reply!r}"
            )
        digest = str(reply.get("checkpoint_digest") or "")
        if not digest.startswith("sha256:"):
            raise RemoteJobsError(
                f"remote checkpoint verification returned invalid digest: {digest!r}"
            )
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
                self._send(sock, {"type": "job", "kind": kind, "job": remote_job})
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

    def submit_self_play_multi(
        self, jobs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Submit a packed Libcg self-play batch as one ``self_play_multi`` job.

        The remote worker runs :func:`remote_self_play_multi_job` and returns a
        dict envelope with ``results``. Each child result is validated with the
        ordinary remote-result contract before credit.
        """
        if not jobs:
            return []
        prepared = [prepare_remote_play_job(self.host, job) for job in jobs]
        batch_job: dict[str, Any] = {"jobs": prepared}
        first = prepared[0]
        if first.get("checkpoint"):
            batch_job["checkpoint"] = first["checkpoint"]
        if first.get("checkpoint_digest"):
            batch_job["checkpoint_digest"] = first["checkpoint_digest"]
        job_buffer_s = _env_float("POKEBOT_REMOTE_JOB_TIMEOUT_BUFFER_S", 600.0)
        # Packed batches share one socket wait; keep the buffer sized for the
        # slowest child rather than multiplying timeouts by pack size.
        timeout_games = max(
            float(job.get("game_timeout_s") or 900) for job in prepared
        )
        last_exc: Optional[BaseException] = None
        for attempt in range(2):
            sock: Optional[socket.socket] = None
            prev: Optional[float] = None
            try:
                sock = self._require_sock()
                prev = sock.gettimeout()
                sock.settimeout(max(self.timeout_s, timeout_games + job_buffer_s))
                self._send(
                    sock,
                    {
                        "type": "job",
                        "kind": "self_play_multi",
                        "job": batch_job,
                    },
                )
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
            if isinstance(payload, dict) and bool(payload.get("self_play_multi")):
                rows = list(payload.get("results") or [])
            elif isinstance(payload, list):
                rows = list(payload)
            else:
                raise RemoteJobsError(
                    "remote self_play_multi result missing results list"
                )
            if len(rows) != len(jobs):
                raise RemoteJobsError(
                    "remote self_play_multi result count mismatch: "
                    f"expected={len(jobs)} got={len(rows)}"
                )
            expected_by_index: dict[int, dict[str, Any]] = {}
            for job in jobs:
                try:
                    job_index = int(job.get("job_index", -1))
                    our_seat = int(job.get("our_seat", 0))
                except (TypeError, ValueError) as exc:
                    raise RemoteJobsError(
                        "self_play_multi request has non-integer child identity"
                    ) from exc
                if job_index < 0 or job_index in expected_by_index:
                    raise RemoteJobsError(
                        "self_play_multi request has invalid child job indexes"
                    )
                expected_by_index[job_index] = {**job, "our_seat": our_seat}
            checked_by_index: dict[int, dict[str, Any]] = {}
            for position, row in enumerate(rows):
                if not isinstance(row, dict):
                    raise RemoteJobsError(
                        "remote self_play_multi child is not an object: "
                        f"position={position}"
                    )
                try:
                    job_index = int(row.get("job_index", -1))
                    row_seat = int(row.get("our_seat", 0))
                except (TypeError, ValueError) as exc:
                    raise RemoteJobsError(
                        "remote self_play_multi child has non-integer identity: "
                        f"position={position}"
                    ) from exc
                if job_index in checked_by_index:
                    raise RemoteJobsError(
                        "remote self_play_multi returned a duplicate child: "
                        f"job_index={job_index}"
                    )
                job = expected_by_index.get(job_index)
                if job is None:
                    raise RemoteJobsError(
                        "remote self_play_multi returned an unknown child: "
                        f"job_index={job_index}"
                    )
                expected_identity = (
                    job_index,
                    str(job.get("opponent_id") or "self"),
                    int(job.get("our_seat", 0)),
                )
                actual_identity = (
                    job_index,
                    str(row.get("opponent_id") or "self"),
                    row_seat,
                )
                if actual_identity != expected_identity:
                    raise RemoteJobsError(
                        "remote self_play_multi child identity mismatch: "
                        f"position={position} expected={expected_identity!r} "
                        f"actual={actual_identity!r}"
                    )
                # Public r182 packets carry a portable baseline ``spec``;
                # unlike self-play, their per-child collection identity also
                # includes the deterministic seed and the exact full-model
                # checkpoint digest.  ``run_play_multi`` echoes both fields,
                # so accepting a row without them would allow a same-index
                # stale/foreign public result to consume an exact cell.
                if job.get("spec") is not None and (
                    "seed" not in row
                    or "checkpoint_digest" not in row
                    or row.get("seed") != job.get("seed")
                    or row.get("checkpoint_digest")
                    != job.get("checkpoint_digest")
                ):
                    raise RemoteJobsError(
                        "remote self_play_multi public child proof mismatch: "
                        f"position={position} job_index={job_index} "
                        f"expected_seed={job.get('seed')!r} "
                        f"actual_seed={row.get('seed')!r} "
                        "expected_checkpoint_digest="
                        f"{job.get('checkpoint_digest')!r} "
                        "actual_checkpoint_digest="
                        f"{row.get('checkpoint_digest')!r}"
                    )
                checked_by_index[job_index] = require_remote_result_success(row)
            if set(checked_by_index) != set(expected_by_index):
                raise RemoteJobsError(
                    "remote self_play_multi result child set mismatch"
                )
            return [checked_by_index[int(job["job_index"])] for job in jobs]
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
                self._send(sock, msg)
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
    fanout_connect_timeout_s = max(
        0.5,
        _env_float("POKEBOT_REMOTE_FANOUT_CONNECT_TIMEOUT_S", 3.0),
    )
    fanout_hello_timeout_s = max(
        fanout_connect_timeout_s,
        _env_float("POKEBOT_REMOTE_FANOUT_HELLO_TIMEOUT_S", 10.0),
    )
    client = RemoteJobClient(
        template.host,
        template.port,
        timeout_s=template.timeout_s,
        connect_timeout_s=min(
            float(template.connect_timeout_s), fanout_connect_timeout_s
        ),
        control_timeout_s=min(
            float(template.control_timeout_s), fanout_hello_timeout_s
        ),
    )
    client.connect()
    return client


def _parallel_remote_slots(
    remote_clients: list[RemoteJobClient],
    *,
    demand_by_endpoint: Optional[dict[str, int]] = None,
    kind: Optional[str] = None,
) -> tuple[list[RemoteJobClient], list[RemoteJobClient]]:
    """Expand farm endpoints into concurrent sockets (one in-flight game each).

    With ``demand_by_endpoint``, opens ``min(capacity, max_cap, demand)`` sockets
    so defaults stay below max and the scheduler can grow into headroom.
    Without demand, legacy behaviour uses weight × hello ``workers``.

    Returns ``(all_slots, owned_clones)`` where ``owned_clones`` must be closed
    by the caller (slot 0 per endpoint reuses the farm session).
    """
    ordered_slots: list[tuple[int, int, RemoteJobClient, bool]] = []
    clone_requests: list[tuple[int, int, RemoteJobClient]] = []
    # POKEBOT_REMOTE_SLOT_DIVISOR>1 shares farm capacity across concurrent trainers.
    divisor = max(1, int(os.environ.get("POKEBOT_REMOTE_SLOT_DIVISOR", "1") or "1"))
    for endpoint_index, template in enumerate(remote_clients):
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
        n = remote_socket_target(execution_slots, kind=kind)
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
                f"{remote_socket_prefetch_factor(kind=kind)})",
                flush=True,
            )
        ordered_slots.append((endpoint_index, 0, template, False))
        for clone_index in range(1, n):
            clone_requests.append((endpoint_index, clone_index, template))

    # Open every endpoint's request sockets concurrently. The old serial loop
    # opened all Elmo sockets, then waited on each Bert TCP handshake before it
    # started *any* emitter thread. One slow SYN therefore left a fully ready
    # Elmo at 0 jobs while the dashboard misleadingly showed 36 reservations.
    # A bounded LAN connect/hello deadline makes one unhealthy slot reduce only
    # that endpoint's realized capacity instead of freezing the whole wave.
    failed = 0
    fanout_started = time.monotonic()
    if clone_requests:
        configured_workers = int(
            os.environ.get("POKEBOT_REMOTE_CONNECT_FANOUT", "64") or "64"
        )
        fanout_workers = min(
            len(clone_requests), max(1, min(128, configured_workers))
        )
        with ThreadPoolExecutor(
            max_workers=fanout_workers,
            thread_name_prefix="remote-initial-connect",
        ) as connector_pool:
            futures = {
                connector_pool.submit(_clone_remote_client, template): (
                    endpoint_index,
                    clone_index,
                    template,
                )
                for endpoint_index, clone_index, template in clone_requests
            }
            for future in as_completed(futures):
                endpoint_index, clone_index, template = futures[future]
                try:
                    clone = future.result()
                except (TimeoutError, OSError, RemoteJobsError) as exc:
                    failed += 1
                    print(
                        f"[remote] {template.endpoint} initial slot "
                        f"{clone_index} failed "
                        f"({type(exc).__name__}: {exc}); continuing with "
                        "remaining live slots",
                        flush=True,
                    )
                    continue
                ordered_slots.append(
                    (endpoint_index, clone_index, clone, True)
                )
    ordered_slots.sort(key=lambda row: (row[0], row[1]))
    slots = [row[2] for row in ordered_slots]
    owned = [row[2] for row in ordered_slots if row[3]]
    print(
        "[remote] parallel_slot_fanout "
        f"requested={len(clone_requests)} connected={len(owned)} "
        f"failed={failed} wall_s={time.monotonic() - fanout_started:.3f}",
        flush=True,
    )
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
    on_producers_drained: Optional[Callable[[], None]] = None,
) -> Iterator[dict[str, Any]]:
    """Run local WorkerPool and optional remote TCP workers in parallel.

    Yields results in completion order (unordered), matching ``imap_unordered``.
    When ``remote_clients`` is empty, delegates to the local pool only.

    Remote concurrency matches advertised worker counts: each endpoint opens
    ``info.workers`` sockets (one job in flight per socket), so additive
    capacity is real in-flight games — not one serial pipeline per host.
    """
    remote_enabled = bool(remote_clients and int(remote_workers) > 0)
    if not remote_enabled and on_producers_drained is None:
        for row in local_pool.imap_unordered(local_fn, jobs):
            if on_execution is not None:
                on_execution({"origin": "local", "kind": kind})
            yield row
        return

    if remote_enabled:
        local_jobs, remote_jobs = split_jobs_additive(
            jobs,
            local_workers=local_workers,
            remote_workers=remote_workers,
        )
    else:
        # With an early-release callback, put the local iterator behind the
        # same bounded spill queue as additive dispatch. This lets the pool's
        # producer finish and release its worker processes while the serialized
        # consumer is still compacting already-durable rows.
        local_jobs, remote_jobs = list(jobs), []
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
    producer_lifecycle_lock = threading.Lock()
    producers_registered = 0
    producers_finished = 0
    producers_drained_notified = False

    def _producer_finished() -> None:
        nonlocal producers_finished, producers_drained_notified
        callback: Optional[Callable[[], None]] = None
        with producer_lifecycle_lock:
            producers_finished += 1
            if (
                not producers_drained_notified
                and producers_registered > 0
                and producers_finished == producers_registered
            ):
                producers_drained_notified = True
                callback = on_producers_drained
        if callback is not None:
            try:
                callback()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
                producer_stop.set()

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
            _producer_finished()

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

    def _submit_remote_multi_with_retry(
        client: RemoteJobClient, jobs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Submit a packed self_play_multi batch with remote-only retries."""
        max_attempts = max(
            1, int(os.environ.get("POKEBOT_REMOTE_JOB_RETRIES", "8") or "8")
        )
        last_exc: Optional[BaseException] = None
        for attempt in range(max_attempts):
            try:
                return client.submit_self_play_multi(jobs)
            except (TimeoutError, OSError, RemoteJobsError) as exc:
                last_exc = exc
                print(
                    f"[remote] {client.endpoint} self_play_multi attempt "
                    f"{attempt + 1}/{max_attempts} failed "
                    f"({type(exc).__name__}: {exc}); reconnect+retry on remote "
                    f"pack_games={len(jobs)}",
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
            packets = _remote_transport_packets(client, kind=kind, jobs=batch)
            for packet_index, packet in enumerate(packets):
                if producer_stop.is_set():
                    return
                multi_env = _remote_packet_uses_multi_env(
                    kind=kind, packet=packet
                )
                try:
                    if multi_env:
                        rows = (
                            _submit_remote_multi_with_retry(client, packet)
                            if no_local_fallback
                            else client.submit_self_play_multi(packet)
                        )
                    elif no_local_fallback:
                        rows = [_submit_remote_with_retry(client, packet[0])]
                    else:
                        rows = [client.submit_job(packet[0], kind=kind)]
                    for row in rows:
                        if on_execution is not None:
                            on_execution(
                                {
                                    "origin": "remote",
                                    "endpoint": client.endpoint,
                                    "kind": kind,
                                    "transport_kind": (
                                        "self_play_multi" if multi_env else kind
                                    ),
                                    "pack_games": 1,
                                }
                            )
                        if not _put_thread_result(
                            out_q, producer_stop, ("ok", row)
                        ):
                            return
                except (TimeoutError, OSError, RemoteJobsError) as exc:
                    remaining = [
                        job
                        for pending_packet in packets[packet_index:]
                        for job in pending_packet
                    ]
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
            _producer_finished()

    try:
        if remote_jobs:
            remote_slots, owned_clients = _parallel_remote_slots(
                remote_clients, kind=kind
            )
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
                    remote_slots, owned_clients = _parallel_remote_slots(
                        remote_clients, kind=kind
                    )
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
        with producer_lifecycle_lock:
            producers_registered = len(threads)
        if not threads and on_producers_drained is not None:
            on_producers_drained()
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
    on_producers_drained: Optional[Callable[[], None]] = None,
) -> Iterator[dict[str, Any]]:
    """Additive collect with mid-wave rebalance of *new chunk claims*.

    Remotes still receive **large job chunks** per socket refill (LAN RTT
    amortize) — never chatty per-game scheduler RPCs. Scheduler metrics use
    wave wall-clock GPS only (see ``mid_iter_scheduler.WaveGpsTracker``).

    ``on_remote_slots`` (optional) is invoked with live socket, scheduler
    demand, and exact outstanding-game counts. Socket count is capacity, not
    work ownership, and must never be displayed as remaining remote games.

    ``on_producers_drained`` runs exactly once after every local/remote
    producer has finished and all result/done records are durably queued, but
    potentially before the consumer has compacted the queue.  Production uses
    this only behind an explicit boundary-gated flag to join the exhausted
    local simulator pool and release anonymous model memory during a long
    result-spool drain.
    """
    # Self-play is a short 1,024-game wave on the current H10 runtime.  A
    # remote endpoint that accepts work but stops answering control probes can
    # strand its final executing claims after Blackwell has exhausted the
    # shared queue.  There is no exact-once-safe way to steal an already
    # executing claim, so the fail-closed production profile keeps self-play
    # local while leaving public mix and every evaluation phase additive.
    # This is deliberately checked before the scheduled path so no remote
    # producer, socket, reservation, or endpoint-owned queue is created.
    if kind == "self_play" and _env_truthy("POKEBOT_SELF_PLAY_LOCAL_ONLY"):
        print(
            "[remote] self_play_local_only=true remote_claims=0 "
            "reason=fail_closed_no_remote_return_tail",
            flush=True,
        )
        remote_clients = []
        remote_workers = 0

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
            on_producers_drained=on_producers_drained,
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
    producer_lifecycle_lock = threading.Lock()
    producers_registered = 0
    producers_finished = 0
    producers_drained_notified = False
    slots_by_endpoint: dict[str, int] = {}
    refill_slot_floor: dict[str, int] = {}
    low_water_recovery: set[str] = set()
    # A listener restart invalidates every persistent request socket at once,
    # but the corresponding emitter threads can remain alive while their
    # reconnect retries back off.  ``slots_by_endpoint`` intentionally counts
    # those threads until they exit, so an ordinary low-water refill sees its
    # socket ceiling as full even though the restarted listener has no live
    # data-plane sessions.  Keep that exceptional state at *emitter* grain:
    # a successful fresh control-plane probe may open one replacement for each
    # emitter that has actually observed a refused reconnect.  The old emitter
    # is then retired at its next between-claim boundary.  This never releases
    # or reassigns its logical credits, which can include an in-flight packed
    # r182 public request.
    next_remote_emitter_id = 0
    remote_emitter_endpoint: dict[int, str] = {}
    remote_emitter_reconnect_backoff: set[int] = set()
    remote_emitter_rejoin_retire: set[int] = set()
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
    endpoint_execution_target: dict[str, int] = {
        endpoint: 0 for endpoint in endpoint_templates
    }
    self_play_elmo_tail_only = bool(
        kind == "self_play"
        and _env_truthy("POKEBOT_SELF_PLAY_ELMO_TAIL_ONLY")
    )
    tail_fraction = (
        _env_float("POKEBOT_SELF_PLAY_TAIL_WORK_STEAL_FRACTION", 0.35)
        if self_play_elmo_tail_only
        else None
    )
    try:
        configured_tail_games = int(
            os.environ.get("POKEBOT_SELF_PLAY_TAIL_WORK_STEAL_GAMES", "0")
            or "0"
        )
    except (TypeError, ValueError):
        configured_tail_games = 0
    tail_games = (
        max(1, configured_tail_games)
        if self_play_elmo_tail_only and configured_tail_games > 0
        else None
    )
    endpoint_credits = _EndpointClaimCredits(
        total_jobs=total_jobs,
        tail_fraction=tail_fraction,
        tail_jobs=tail_games,
    )
    tail_work_steal_started = False
    tail_work_steal_released = 0
    credit_demand_signature: Optional[tuple[tuple[str, int], ...]] = None
    log_scheduler = getattr(scheduler, "maybe_tick", None)
    no_local_fallback = _env_truthy(
        "POKEBOT_REMOTE_NO_LOCAL_FALLBACK"
    ) or _env_truthy("POKEBOT_REMOTE_ONLY")
    remote_only = _env_truthy("POKEBOT_REMOTE_ONLY")
    demand_queue_mode = _env_truthy("POKEBOT_REMOTE_DEMAND_QUEUE")
    endpoint_owned_queue_mode = bool(demand_queue_mode or remote_only)

    # ``eout`` / ``bout`` are already exact *logical-game* credits.  That is
    # deliberately a different unit from both request sockets and the worker
    # server's ``active_jobs`` counter: one r182 public request can contain up
    # to four independent logical games.  Keep a compact per-wave transport
    # audit so the event log proves which public rows really used LibcgMultiEnv
    # and which stayed singleton without printing one line per game.
    public_transport_enabled = str(kind) == "play"
    public_transport_lock = threading.Lock()
    public_transport_by_endpoint: dict[str, dict[str, int]] = {}

    def _public_transport_bucket(
        packet: list[dict[str, Any]], *, multi_env: bool
    ) -> str | None:
        if not public_transport_enabled or not packet:
            return None
        if multi_env:
            # ``_remote_packet_uses_multi_env`` has already proven that every
            # child is exact-r182 safe.  Keep the assertion local to telemetry
            # as a second guard against a misleading success log.
            if not all(public_multi_env_safe_job(job) for job in packet):
                return "default_singleton"
            return "safe_multi"
        job = packet[0]
        if public_multi_env_safe_job(job):
            # A final 1--3 safe-row remainder, or an endpoint without the
            # r182 capability, correctly remains an ordinary one-game play.
            return "safe_singleton"
        opponent_id = str(job.get("opponent_id") or "")
        if opponent_id in PUBLIC_MULTI_ENV_LEGACY_OPPONENT_IDS:
            return "legacy_id_singleton"
        return "default_singleton"

    def _note_public_transport(
        endpoint: str,
        packet: list[dict[str, Any]],
        *,
        multi_env: bool,
        returned_rows: int,
    ) -> None:
        """Record completed public transport at packet and logical-game grain."""

        bucket = _public_transport_bucket(packet, multi_env=multi_env)
        if bucket is None:
            return
        logical_games = max(0, int(returned_rows))
        with public_transport_lock:
            row = public_transport_by_endpoint.setdefault(
                str(endpoint),
                {
                    "safe_multi_requests": 0,
                    "safe_multi_games": 0,
                    "safe_singleton_requests": 0,
                    "safe_singleton_games": 0,
                    "legacy_id_singleton_requests": 0,
                    "legacy_id_singleton_games": 0,
                    "default_singleton_requests": 0,
                    "default_singleton_games": 0,
                },
            )
            row[f"{bucket}_requests"] += 1
            row[f"{bucket}_games"] += logical_games

    def _public_transport_summary() -> str:
        """Format a bounded, deterministic public transport receipt line."""

        with public_transport_lock:
            rows = {
                endpoint: dict(values)
                for endpoint, values in public_transport_by_endpoint.items()
            }
        if not rows:
            return "-"
        parts: list[str] = []
        for endpoint in sorted(rows):
            row = rows[endpoint]
            parts.append(
                f"{endpoint}:"
                f"safe_multi={row['safe_multi_games']}/"
                f"{row['safe_multi_requests']}req,"
                f"safe_singleton={row['safe_singleton_games']}/"
                f"{row['safe_singleton_requests']}req,"
                f"legacy_id_singleton={row['legacy_id_singleton_games']}/"
                f"{row['legacy_id_singleton_requests']}req,"
                f"default_singleton={row['default_singleton_games']}/"
                f"{row['default_singleton_requests']}req"
            )
        return ";".join(parts)

    def _publish_remote_slots() -> None:
        """Push sockets and exact remote-owned games as separate values.

        This definition must precede the initial endpoint-queue fill: that fill
        updates ownership immediately, before producer threads are started.
        Keeping telemetry available at that first claim also prevents the
        dashboard from briefly presenting socket capacity as owned work.
        """
        if on_remote_slots is None:
            return
        try:
            active = int(sum(slots_by_endpoint.values()))
            dem_map = dict(scheduler.remote_demand() or {})
            demand_n = (
                int(sum(int(v) for v in dem_map.values()))
                if dem_map
                else active
            )
            outstanding_by_role = {"elmo": 0, "bert": 0, "other": 0}
            for endpoint, count in endpoint_credits.inflight.items():
                template = endpoint_templates.get(endpoint)
                role = (
                    endpoint_role(template.host)
                    if template is not None
                    else "other"
                )
                outstanding_by_role[role] = (
                    int(outstanding_by_role.get(role, 0))
                    + max(0, int(count))
                )
            on_remote_slots(
                {
                    "active": active,
                    "demand": demand_n,
                    "claimed_remote": int(claimed_remote),
                    "outstanding": int(sum(outstanding_by_role.values())),
                    "outstanding_elmo": int(outstanding_by_role["elmo"]),
                    "outstanding_bert": int(outstanding_by_role["bert"]),
                }
            )
        except Exception:
            # Progress telemetry is observational and may never break exact
            # job ownership or result delivery.
            pass

    def _new_remote_emitter_id_locked(endpoint: str) -> int:
        """Register one request-emitter identity while ``grow_lock`` is held."""

        nonlocal next_remote_emitter_id
        emitter_id = int(next_remote_emitter_id)
        next_remote_emitter_id += 1
        remote_emitter_endpoint[emitter_id] = str(endpoint)
        return emitter_id

    def _discard_remote_emitter_id_locked(emitter_id: int) -> None:
        """Forget an emitter that was allocated but never allowed to start."""

        token = int(emitter_id)
        remote_emitter_endpoint.pop(token, None)
        remote_emitter_reconnect_backoff.discard(token)
        remote_emitter_rejoin_retire.discard(token)

    def _mark_remote_emitter_reconnect_backoff(
        endpoint: str, emitter_id: Optional[int]
    ) -> None:
        """Record one data-path refused-reconnect without touching credits.

        The marker is deliberately per emitter rather than per error attempt:
        one dead socket may retry several times while the worker listener is
        down, but it grants at most one bounded replacement socket after the
        listener's fresh health probe succeeds.
        """

        if emitter_id is None:
            return
        token = int(emitter_id)
        ep = str(endpoint)
        with grow_lock:
            if remote_emitter_endpoint.get(token) != ep:
                return
            if token not in remote_emitter_rejoin_retire:
                remote_emitter_reconnect_backoff.add(token)

    def _mark_remote_emitter_progress(
        endpoint: str, emitter_id: Optional[int]
    ) -> None:
        """Clear a stale-reconnect marker after a real logical result."""

        if emitter_id is None:
            return
        token = int(emitter_id)
        with grow_lock:
            if remote_emitter_endpoint.get(token) == str(endpoint):
                remote_emitter_reconnect_backoff.discard(token)

    def _listener_rejoin_replacements_locked(
        endpoint: str,
        *,
        active: int,
        execution: int,
        deficit: int,
    ) -> list[int]:
        """Return bounded stale emitters that a public listener rejoin may replace.

        A low active server sample by itself is not enough: normal fast
        completions routinely report it.  The replacement path requires both
        a successful *new* health socket and a data-path reconnect refusal from
        the old emitter.  It is public-play-only because self-play's one-wave
        cap deliberately forbids low-water second-wave admission.
        """

        if str(kind) != "play" or int(active) >= max(1, int(execution)):
            return []
        cap = max(0, int(deficit))
        if cap <= 0:
            return []
        ep = str(endpoint)
        return [
            token
            for token in sorted(remote_emitter_reconnect_backoff)
            if remote_emitter_endpoint.get(token) == ep
            and token not in remote_emitter_rejoin_retire
        ][:cap]

    def _tail_endpoint_target(endpoint: str) -> int:
        """Return the controller-owned target once tail stealing starts.

        Self-play has a materially different tail profile from public play:
        Bert's CPU games are the long stragglers, while even an extra queued
        Elmo wave can outlive Blackwell's shared work.  The opt-in policy stops
        new Bert tail claims and keeps only one Elmo execution wave. Other
        collection kinds retain the existing one-wave target.
        """
        ep = str(endpoint)
        execution = max(0, int(endpoint_execution_target.get(ep, 0)))
        if not self_play_elmo_tail_only:
            return execution
        template = endpoint_templates.get(ep)
        if template is None or template.host.strip().lower() not in _ELMO_HOSTS:
            return 0
        return min(
            max(0, int(endpoint_queue_safety_ceiling.get(ep, 0))),
            execution,
        )

    def _register_producer(thread: threading.Thread) -> bool:
        """Register ``thread`` unless the producer boundary already sealed.

        Refill monitors open sockets concurrently with the final claims.  A
        monitor may therefore finish connecting after the last registered
        producer has durably queued its done record.  That late socket owns no
        work and must be declined rather than turning a sealed producer
        boundary back into a live one.
        """
        nonlocal producers_registered
        with producer_lifecycle_lock:
            if producers_drained_notified:
                return False
            producers_registered += 1
            threads.append(thread)
            return True

    def _producer_finished() -> None:
        nonlocal producers_finished, producers_drained_notified
        callback: Optional[Callable[[], None]] = None
        with producer_lifecycle_lock:
            producers_finished += 1
            with claim_lock:
                no_unclaimed_jobs = not remaining
            if (
                not producers_drained_notified
                and no_unclaimed_jobs
                and producers_registered > 0
                and producers_finished == producers_registered
            ):
                producers_drained_notified = True
                callback = on_producers_drained
        if callback is not None:
            try:
                callback()
            except BaseException as exc:  # noqa: BLE001
                # The final producer has already queued its ``done`` record.
                # Preserve that ordering, retain the failure for the consumer's
                # post-drain error check, and stop any ancillary monitor work.
                errors.append(exc)
                producer_stop.set()

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
            endpoint_execution_target[ep] = execution
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
            # Self-play jobs are long and remote execution can be far slower
            # than Blackwell.  Do not pre-reserve hundreds of them: one
            # controller-owned wave plus one executing wave preserves remote
            # throughput without creating an unreclaimable remote-only tail.
            high_water = (
                execution
                if self_play_elmo_tail_only
                else execution + queue_target
            )
            endpoint_queue_high_water[ep] = high_water
            # The scheduling target is soft: one execution wave may be in
            # transition while a low-water refill is being admitted. This
            # ceiling exists only for device/file-descriptor safety and is not
            # used as the ordinary refill target.
            endpoint_queue_safety_ceiling[ep] = (
                execution
                if self_play_elmo_tail_only
                else high_water + execution
            )
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

    def _start_tail_work_steal_locked() -> None:
        """Return excess endpoint reservations before the remote-only tail.

        Caller holds ``claim_lock``. Only controller-owned deque rows can be
        returned; jobs already executing or held by a request emitter retain
        their exact owner and finish normally.
        """
        nonlocal claimed_remote
        nonlocal tail_work_steal_started, tail_work_steal_released
        if tail_work_steal_started:
            return
        if not endpoint_owned_queue_mode or remote_only:
            return
        if not endpoint_credits.in_tail(len(remaining)):
            return
        tail_work_steal_started = True
        returned: list[dict[str, Any]] = []
        released_by_endpoint: dict[str, int] = {}
        for endpoint in sorted(endpoint_owned_queues):
            owned = endpoint_owned_queues[endpoint]
            execution_wave = _tail_endpoint_target(endpoint)
            # Credit inflight includes this controller deque plus jobs already
            # executing or held in an emitter-private request list. Only the
            # deque is safely returnable, so subtract those non-deque owners
            # before deciding how much of the deque may remain reserved.
            non_deque_inflight = max(
                0,
                int(endpoint_credits.inflight.get(endpoint, 0)) - len(owned),
            )
            keep = max(0, execution_wave - non_deque_inflight)
            count = max(0, len(owned) - keep)
            if count <= 0:
                continue
            rows = [owned.pop() for _ in range(count)]
            endpoint_credits.note_released(endpoint, count)
            claimed_remote -= count
            released_by_endpoint[endpoint] = count
            returned.extend(rows)
        returned.sort(key=lambda row: int(row.get("job_index", 0)))
        for row in reversed(returned):
            remaining.appendleft(row)
        tail_work_steal_released = len(returned)
        print(
            "[remote] tail_work_steal=start "
            f"fraction={endpoint_credits.tail_fraction:.2f} "
            f"threshold_jobs={endpoint_credits.tail_jobs} "
            f"returned_to_shared={len(returned)} "
            f"released_by_endpoint={released_by_endpoint} "
            f"shared_remaining={len(remaining)} "
            "remote_claim_games=1 local_pool_remains_eligible=true "
            f"self_play_elmo_tail_only={self_play_elmo_tail_only}",
            flush=True,
        )

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
        _start_tail_work_steal_locked()
        owned = endpoint_owned_queues.setdefault(ep, deque())
        target = max(
            0,
            int(
                _tail_endpoint_target(ep)
                if tail_work_steal_started
                else endpoint_queue_high_water.get(ep, 0)
            ),
        )
        if target <= 0 or not remaining:
            return 0
        need = max(0, target - len(owned))
        if need <= 0:
            return 0
        available = _endpoint_remote_available_locked()
        granted = endpoint_credits.claim_limit(
            ep,
            want=(min(1, need) if tail_work_steal_started else need),
            remaining_count=available,
            wave_remaining=len(remaining),
        )
        if granted <= 0:
            return 0
        for _ in range(granted):
            owned.append(remaining.popleft())
        claimed_remote += granted
        endpoint_credits.note_claimed(ep, granted)
        _publish_remote_slots()
        return granted

    def _finish_remote_credit(endpoint: str, n: int = 1) -> None:
        """Release completed remote work and restore data-path health."""
        with claim_lock:
            endpoint_credits.note_finished(endpoint, n)
            # A returned game is stronger evidence than a failed control probe.
            endpoint_credits.set_healthy(endpoint, True)
        _publish_remote_slots()

    def _finish_initial_remote_claim(token: Optional[int]) -> None:
        if token is None:
            return
        initial_remote_tokens.discard(int(token))
        initial_remote_leaders.discard(int(token))
        if not initial_remote_leaders:
            initial_endpoint_leaders_ready.set()

    def _take_public_transport_packet_locked(
        source: deque[dict[str, Any]],
        *,
        packet_size: int,
    ) -> list[dict[str, Any]]:
        """Atomically select one exact-safe public packet from ``source``.

        The caller holds ``claim_lock``.  A scheduled public emitter owns at
        most one transport request at a time: scan the controller deque for
        up to four r182-safe children, remove only those children, and leave
        every legacy/default-singleton row in its original source order.  If
        no safe child remains, release exactly one front row as the ordinary
        singleton fallback.  The source may be the shared deque or an
        endpoint-owned controller queue; in the latter case credit was already
        reserved at queue fill, so this changes packet selection only.
        """

        if not source:
            return []
        limit = max(1, min(4, int(packet_size)))
        rows = list(source)
        selected_indexes: list[int] = []
        for index, job in enumerate(rows):
            if not public_multi_env_safe_job(job):
                continue
            selected_indexes.append(index)
            if len(selected_indexes) >= limit:
                break
        if not selected_indexes:
            # No exact-safe child is available anywhere in this controller
            # queue.  Preserve legacy/default ordering and give the emitter a
            # single ordinary transport request rather than a private chunk.
            return [source.popleft()]

        selected = set(selected_indexes)
        source.clear()
        source.extend(
            job for index, job in enumerate(rows) if index not in selected
        )
        return [rows[index] for index in selected_indexes]

    def _claim(
        side: str,
        want: int,
        *,
        initial_token: Optional[int] = None,
        endpoint: Optional[str] = None,
        public_packet_size: int = 0,
    ) -> list[dict[str, Any]]:
        nonlocal claimed_local, claimed_remote
        nonlocal local_batch_size, local_batch_outstanding
        nonlocal local_last_progress_mono, tail_override_logged
        requested_want = max(0, int(want))
        want = requested_want
        if want <= 0:
            return []
        with claim_lock:
            _start_tail_work_steal_locked()
            if side == "remote" and tail_work_steal_started:
                requested_want = min(requested_want, 1)
                want = min(want, 1)
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
                if public_packet_size > 0:
                    batch = _take_public_transport_packet_locked(
                        owned,
                        packet_size=min(take, public_packet_size),
                    )
                else:
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
            if side == "remote" and public_packet_size > 0:
                batch = _take_public_transport_packet_locked(
                    remaining,
                    packet_size=min(want, public_packet_size),
                )
            else:
                batch = [remaining.popleft() for _ in range(want)]
            if side == "remote":
                claimed_remote += len(batch)
                endpoint_credits.note_claimed(str(endpoint), len(batch))
                _publish_remote_slots()
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
            _producer_finished()

    # Cooperative retire tokens: when demand shrinks, excess emit threads
    # exit between chunks so live sockets fall with demand (not grow-only).
    retire_tokens: dict[str, int] = {}

    def _account_remote_slot_exit(
        endpoint: str, emitter_id: Optional[int] = None
    ) -> tuple[int, int]:
        """Remove one live emitter from both slot and reservation accounting."""
        ep = str(endpoint)
        with grow_lock:
            before = max(0, int(slots_by_endpoint.get(ep, 0)))
            after = max(0, before - 1)
            slots_by_endpoint[ep] = after
            if emitter_id is not None:
                _discard_remote_emitter_id_locked(int(emitter_id))
        with claim_lock:
            endpoint_credits.unregister(ep)
        return before, after

    def _submit_remote_with_retry(
        client: RemoteJobClient,
        job: dict[str, Any],
        *,
        on_reconnect_backoff: Optional[Callable[[], None]] = None,
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
                    if on_reconnect_backoff is not None:
                        on_reconnect_backoff()
                    time.sleep(min(30.0, 2.0 * (attempt + 1)))
                else:
                    time.sleep(min(5.0, 0.5 * (attempt + 1)))
        assert last_exc is not None
        raise last_exc

    def _submit_remote_multi_with_retry(
        client: RemoteJobClient,
        jobs: list[dict[str, Any]],
        *,
        on_reconnect_backoff: Optional[Callable[[], None]] = None,
    ) -> list[dict[str, Any]]:
        max_attempts = max(
            1, int(os.environ.get("POKEBOT_REMOTE_JOB_RETRIES", "8") or "8")
        )
        last_exc: Optional[BaseException] = None
        for attempt in range(max_attempts):
            try:
                return client.submit_self_play_multi(jobs)
            except (TimeoutError, OSError, RemoteJobsError) as exc:
                last_exc = exc
                print(
                    f"[remote] {client.endpoint} scheduled self_play_multi "
                    f"attempt {attempt + 1}/{max_attempts} failed "
                    f"({type(exc).__name__}: {exc}); reconnect+retry remote "
                    f"pack_games={len(jobs)}",
                    flush=True,
                )
                try:
                    client.reconnect()
                except (TimeoutError, OSError, RemoteJobsError):
                    if on_reconnect_backoff is not None:
                        on_reconnect_backoff()
                    time.sleep(min(30.0, 2.0 * (attempt + 1)))
                else:
                    time.sleep(min(5.0, 0.5 * (attempt + 1)))
        assert last_exc is not None
        raise last_exc

    def _emit_remote(
        client: RemoteJobClient,
        initial_token: Optional[int] = None,
        emitter_id: Optional[int] = None,
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
        listener_rejoin_retiring = False
        try:
            while True:
                if producer_stop.is_set():
                    return
                with grow_lock:
                    if (
                        emitter_id is not None
                        and int(emitter_id) in remote_emitter_rejoin_retire
                    ):
                        listener_rejoin_retiring = True
                        retiring = True
                        try:
                            client.close()
                        except Exception:
                            pass
                        return
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
                self_play_pack = client_self_play_multi_pack(client, kind=kind)
                public_pack = (
                    _client_public_multi_env_pack(client)
                    if str(kind) == "play"
                    else 0
                )
                pack_n = max(self_play_pack, public_pack)
                public_packet_size = public_pack if public_pack > 1 else 0
                chunk = max(
                    remote_refill_games_per_socket(),
                    (int(endpoint_backlog) + live_endpoint_slots - 1)
                    // live_endpoint_slots,
                )
                if public_packet_size:
                    # Do not privately reserve a generic lookahead chunk.  The
                    # packet-aware claim below scans its controller deque under
                    # the exact ownership lock and removes only one up-to-four
                    # safe packet, leaving legacy rows available for later
                    # singleton dispatch.  This keeps every public prefetch
                    # socket productive before an earlier request returns.
                    chunk = public_packet_size
                elif pack_n > 1:
                    chunk = pack_n
                elif self_play_elmo_tail_only:
                    # One socket represents one execution worker in the
                    # revision-123 self-play profile.  Letting the first few
                    # socket threads privately claim multi-game batches leaves
                    # most of Elmo/Bert idle even though the dashboard reports
                    # every socket as live.  A one-game claim keeps all 36+16
                    # workers fed and makes ``inflight`` exactly represent
                    # remote-owned work.
                    chunk = 1
                if (
                    endpoint_owned_queue_mode
                    and not remote_only
                    and not public_packet_size
                ):
                    # A mixed collection may not hide more than half of the
                    # tail window in emitter-private lists. Production's many
                    # sockets remain at their existing small per-slot chunks,
                    # while a small fleet cannot privately span the boundary
                    # before its controller-owned reservations are released.
                    private_tail_cap = max(
                        1,
                        int(endpoint_credits.tail_jobs)
                        // max(1, 2 * live_endpoint_slots),
                    )
                    chunk = min(chunk, private_tail_cap)
                    if tail_work_steal_started and pack_n <= 1:
                        chunk = 1
                batch = _claim(
                    "remote",
                    chunk,
                    initial_token=initial_token if first_claim else None,
                    endpoint=ep,
                    public_packet_size=public_packet_size,
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
                packets = _remote_transport_packets(client, kind=kind, jobs=batch)
                for packet_index, packet in enumerate(packets):
                    if producer_stop.is_set():
                        return
                    # Retire only between claimed chunks.  Returning the
                    # unstarted suffix here used to create a deterministic
                    # tail deadlock: local had already exited, while the
                    # surviving remotes were above their soft share and could
                    # not reclaim the returned jobs.  A slot now completes the
                    # batch it owns, then consumes its retire token at the top
                    # of the outer loop.
                    multi_env = _remote_packet_uses_multi_env(
                        kind=kind, packet=packet
                    )
                    try:
                        if multi_env:
                            rows = _submit_remote_multi_with_retry(
                                client,
                                packet,
                                on_reconnect_backoff=lambda: (
                                    _mark_remote_emitter_reconnect_backoff(
                                        ep, emitter_id
                                    )
                                ),
                            )
                        else:
                            row = (
                                _submit_remote_with_retry(
                                    client,
                                    packet[0],
                                    on_reconnect_backoff=lambda: (
                                        _mark_remote_emitter_reconnect_backoff(
                                            ep, emitter_id
                                        )
                                    ),
                                )
                                if no_local_fallback
                                else client.submit_job(packet[0], kind=kind)
                            )
                            rows = [row]
                        _note_public_transport(
                            client.endpoint,
                            packet,
                            multi_env=multi_env,
                            returned_rows=len(rows),
                        )
                        for row in rows:
                            _mark_remote_emitter_progress(ep, emitter_id)
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
                                        "transport_kind": (
                                            "self_play_multi" if multi_env else kind
                                        ),
                                        "pack_games": 1,
                                    }
                                )
                            if not _put_thread_result(
                                out_q, producer_stop, ("ok", row)
                            ):
                                return
                    except (TimeoutError, OSError, RemoteJobsError) as exc:
                        remaining_jobs = [
                            job
                            for pending_packet in packets[packet_index:]
                            for job in pending_packet
                        ]
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
            before, after = _account_remote_slot_exit(ep, emitter_id)
            if listener_rejoin_retiring:
                print(
                    f"[remote] {ep} listener_rejoin_retire "
                    f"slots={before}->{after} "
                    "(replaced refused persistent socket)",
                    flush=True,
                )
            elif retiring:
                print(
                    f"[remote] {ep} demand_shrink "
                    f"slots={before}->{after} "
                    f"(cap demand; cooperative retire)",
                    flush=True,
                )
            _put_thread_result(out_q, producer_stop, ("done", None))
            _producer_finished()

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
            remote_clients,
            demand_by_endpoint=demand0 or None,
            kind=kind,
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
            _register_producer(
                threading.Thread(target=_emit_local, name="sched-local", daemon=True)
            )
        for slot_index, client in enumerate(remote_slots):
            with grow_lock:
                emitter_id = _new_remote_emitter_id_locked(client.endpoint)
            slots_by_endpoint[client.endpoint] = (
                slots_by_endpoint.get(client.endpoint, 0) + 1
            )
            # Register every endpoint before any thread starts.  Thus the
            # first fast Elmo claimant already protects Bert's full credit;
            # fairness does not depend on OS thread-start order.
            endpoint_credits.register(client.endpoint)
            _register_producer(
                threading.Thread(
                    target=_emit_remote,
                    args=(client, slot_index, emitter_id),
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
                "shared_endpoint_race=disabled units=logical_games",
                flush=True,
            )
        multi_pack_by_endpoint = {
            client.endpoint: (
                _client_public_multi_env_pack(client)
                if str(kind) == "play"
                else client_self_play_multi_pack(client, kind=kind)
            )
            for client in remote_clients
        }
        if public_transport_enabled:
            safe_allowlisted = sum(
                1 for job in jobs if public_multi_env_safe_job(job)
            )
            named_legacy_singleton = sum(
                1
                for job in jobs
                if str(job.get("opponent_id") or "")
                in PUBLIC_MULTI_ENV_LEGACY_OPPONENT_IDS
            )
            default_singleton = max(
                0,
                total_jobs - safe_allowlisted - named_legacy_singleton,
            )
            print(
                "[remote] public_multi_env_plan "
                f"logical_games={total_jobs} "
                f"safe_allowlisted={safe_allowlisted} "
                f"legacy_id_singleton={named_legacy_singleton} "
                f"default_singleton={default_singleton} "
                f"endpoint_pack_cap={multi_pack_by_endpoint} "
                "default_deny=true",
                flush=True,
            )
        print(
            f"[remote] scheduled_dispatch kind={kind} jobs={total_jobs} "
            f"remote_slots={len(remote_slots)}(request_sockets) "
            f"remote_chunk={int(scheduler.decision().remote_chunk)} "
            f"refill_games={remote_refill_games_per_socket()} "
            f"multi_env_pack={multi_pack_by_endpoint} "
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

        _publish_remote_slots()

        def _maybe_grow_remote_slots(dec: Any) -> None:
            """Hot-add remote sockets when scheduler demand rises mid-wave."""
            nonlocal pending
            # A scheduler decision can retain its last nonzero demand after
            # every job has already been claimed. Never create empty producer
            # threads during the consumer-only result-spool tail: besides
            # wasting sockets, that would violate the producer-drained
            # callback's once-only lifecycle boundary.
            with claim_lock:
                if not remaining:
                    return
            demand = dict(getattr(dec, "remote_demand", None) or {})
            if not demand:
                return
            grew = False
            with grow_lock:
                for template in remote_clients:
                    ep = template.endpoint
                    want = remote_socket_target(
                        int(demand.get(ep, 0)), kind=kind
                    )
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
                        emitter_id = _new_remote_emitter_id_locked(ep)
                        t = threading.Thread(
                            target=_emit_remote,
                            args=(clone, None, emitter_id),
                            name=f"sched-remote-grow-{ep}-{have + i}",
                            daemon=True,
                        )
                        if not _register_producer(t):
                            _discard_remote_emitter_id_locked(emitter_id)
                            clone.close()
                            break
                        owned_clients.append(clone)
                        slots_by_endpoint[ep] = slots_by_endpoint.get(ep, 0) + 1
                        with claim_lock:
                            endpoint_credits.register(ep)
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
                        max(0, int(demand.get(ep, 0))), kind=kind
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
            executing plus locally queued *request packets*. An r182 safe
            public packet can contain up to four logical games, so the exact
            logical credits remain separate in progress telemetry. Refill is
            based on queued request packets (active minus execution workers),
            not TCP socket estimates. Both endpoint probes and all required
            reserve connections happen concurrently, so a low Bert queue
            cannot wait behind Elmo. This function is called by an independent
            cadence thread; result ingest can never pause the 0.2-second queue
            controller.
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
                tuple[
                    RemoteJobClient,
                    int,
                    int,
                    int,
                    int,
                    int,
                    int,
                    int,
                    tuple[int, ...],
                ]
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
                # ``active_jobs`` / ``queued`` below are remote-server request
                # packets.  The separate controller credit is the exact
                # logical-game count shown as eout/bout in trainer progress.
                with claim_lock:
                    logical_credits = max(
                        0,
                        int(
                            endpoint_credits.inflight.get(
                                template.endpoint, 0
                            )
                        ),
                    )
                max_slots = remote_socket_max_target(execution, kind=kind)
                if self_play_elmo_tail_only:
                    # Self-play request sockets are the execution wave. They
                    # persist and claim one next game after each return; a
                    # low-water probe must never manufacture a second wave.
                    max_slots = min(
                        max_slots,
                        max(
                            0,
                            int(
                                endpoint_queue_safety_ceiling.get(
                                    template.endpoint, execution
                                )
                            ),
                        ),
                    )
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
                restore_deficit = max(0, restore_active - active)
                with grow_lock:
                    # Normal room uses the physical request-socket ceiling.
                    # A listener-rejoin replacement can exceed that count
                    # only one-for-one with a specifically refused old
                    # emitter, which is retired before it can make another
                    # claim.  Thus a stale ``have=72`` cannot suppress the
                    # fresh public prefetch wave, but repeated probes cannot
                    # grow an unbounded second fleet.
                    normal_slot_room = max(0, max_slots - have)
                    stale_replacements = _listener_rejoin_replacements_locked(
                        endpoint,
                        active=active,
                        execution=execution,
                        deficit=restore_deficit,
                    )
                # One poll means one complete top-up.  The old ``execution``
                # term capped this at a single 48/16-worker wave, so a drained
                # queue needed several result-driven probes to recover.
                extras = min(
                    normal_slot_room + len(stale_replacements),
                    restore_deficit,
                    jobs_left,
                )
                if extras > 0:
                    rejoin_count = max(0, extras - normal_slot_room)
                    plans.append(
                        (
                            template,
                            extras,
                            active,
                            queued,
                            low_water,
                            server_high_water,
                            max_slots,
                            logical_credits,
                            tuple(stale_replacements[:rejoin_count]),
                        )
                    )

            if not plans:
                return

            # Build the request list round-robin, then connect concurrently.
            # This starts the *entire* high-water deficit in this one pass.
            max_extras = max(plan[1] for plan in plans)
            started: dict[str, int] = {plan[0].endpoint: 0 for plan in plans}
            rejoin_started: dict[str, int] = {
                plan[0].endpoint: 0 for plan in plans
            }
            limits = {plan[0].endpoint: int(plan[6]) for plan in plans}
            requests: list[tuple[RemoteJobClient, Optional[int]]] = []
            for depth in range(max_extras):
                for (
                    template,
                    extras,
                    _active,
                    _queued,
                    _low_water,
                    _high_water,
                    _max_slots,
                    _logical_credits,
                    rejoin_replacements,
                ) in plans:
                    if depth >= extras:
                        continue
                    ordinary_extras = max(
                        0, int(extras) - len(rejoin_replacements)
                    )
                    replacement_id = (
                        int(rejoin_replacements[depth - ordinary_extras])
                        if depth >= ordinary_extras
                        else None
                    )
                    requests.append((template, replacement_id))

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
                    connector_pool.submit(_open_refill_client, template): (
                        template,
                        replacement_id,
                    )
                    for template, replacement_id in requests
                    if not producer_stop.is_set()
                    and not refill_monitor_stop.is_set()
                }
                for future in as_completed(future_templates):
                    template, replacement_id = future_templates[future]
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
                        replaces_stale_emitter = bool(
                            replacement_id is not None
                            and remote_emitter_endpoint.get(int(replacement_id))
                            == ep
                            and int(replacement_id)
                            in remote_emitter_reconnect_backoff
                            and int(replacement_id)
                            not in remote_emitter_rejoin_retire
                        )
                        if (
                            producer_stop.is_set()
                            or refill_monitor_stop.is_set()
                            or (
                                not replaces_stale_emitter
                                and have >= int(limits.get(ep, have))
                            )
                        ):
                            clone.close()
                            continue
                        emitter_id = _new_remote_emitter_id_locked(ep)
                        thread = threading.Thread(
                            target=_emit_remote,
                            args=(clone, None, emitter_id),
                            name=(
                                f"sched-remote-listener-rejoin-{ep}-{have}"
                                if replaces_stale_emitter
                                else f"sched-remote-low-water-{ep}-{have}"
                            ),
                            daemon=True,
                        )
                        if not _register_producer(thread):
                            _discard_remote_emitter_id_locked(emitter_id)
                            clone.close()
                            continue
                        owned_clients.append(clone)
                        slots_by_endpoint[ep] = have + 1
                        retire_tokens[ep] = 0
                        refill_slot_floor[ep] = max(
                            int(refill_slot_floor.get(ep, 0)),
                            min(
                                int(slots_by_endpoint[ep]),
                                int(limits.get(ep, slots_by_endpoint[ep])),
                            ),
                        )
                        if replaces_stale_emitter:
                            remote_emitter_rejoin_retire.add(int(replacement_id))
                        with claim_lock:
                            endpoint_credits.register(ep)
                    with pending_lock:
                        pending += 1
                    started[ep] += 1
                    if replaces_stale_emitter:
                        rejoin_started[ep] += 1
                    thread.start()

            for (
                template,
                _extras,
                active,
                queued,
                low_water,
                high_water,
                _max_slots,
                logical_credits,
                rejoin_replacements,
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
                    f"{remote_socket_max_target(int(demand.get(ep, 0)), kind=kind)} "
                    "server_request_units=packets "
                    f"logical_credits_at_probe={logical_credits}",
                    flush=True,
                )
                rejoined = int(rejoin_started.get(ep, 0))
                if rejoined:
                    print(
                        f"[remote] {ep} LISTENER_REJOIN_REFILL "
                        f"stale_emitters={len(rejoin_replacements)} "
                        f"replaced={rejoined} have_before="
                        f"{max(0, int(slots_by_endpoint.get(ep, 0)) - added)} "
                        f"socket_ceiling={int(_max_slots)} "
                        "logical_credits_preserved=true",
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
        if public_transport_enabled:
            print(
                "[remote] public_multi_env_transport complete "
                "units=logical_games/transport_requests "
                f"by_endpoint={_public_transport_summary()}",
                flush=True,
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
            raw_requested_codecs = first.get("frame_codecs") or []
            if isinstance(raw_requested_codecs, str):
                requested_codecs = {
                    value.strip()
                    for value in raw_requested_codecs.split(",")
                    if value.strip()
                }
            else:
                requested_codecs = {str(value) for value in raw_requested_codecs}
            compress_frames = _FRAME_CODEC_ZLIB_V1 in requested_codecs
            info = hello() if hello is not None else {}
            try:
                send_frame(
                    conn,
                    {
                        "type": "hello_ok",
                        "proto": PROTO_VERSION,
                        "frame_codecs": (
                            [_FRAME_CODEC_ZLIB_V1] if compress_frames else []
                        ),
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
                            compress=compress_frames,
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
                            compress=compress_frames,
                        )
                    except OSError:
                        break
                    continue
                try:
                    send_frame(conn, reply, compress=compress_frames)
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
