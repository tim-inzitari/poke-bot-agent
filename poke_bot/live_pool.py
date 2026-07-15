"""Live pool-plan control for round-robin (iteration-boundary resize).

Writes/reads ``outputs/state/live_pool_plan.json`` so a running
``train_round_robin`` can grow/shrink collection workers, leaf servers, and
promotion parallelism **between iterations** without a process restart.

Design contract
---------------
* Apply only when no collection ``WorkerPool`` is active (iteration start).
* Full leaf-topology rebuild on change (simpler than in-place chop).
* Opt-in consumption: RR polls the file; absence means leave launch flags alone.
* Never auto-applied mid-wave / mid-``imap_unordered``.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from . import paths

LIVE_POOL_PLAN_PATH: Path = paths.OUTPUTS_DIR / "state" / "live_pool_plan.json"

# Hard clamps — defensive against bogus watcher / hand edits.
_MIN_WORKERS = 1
_MAX_WORKERS = 128
_MIN_LEAF_SERVERS = 1
_MAX_LEAF_SERVERS = 16
_MIN_PROMOTION_WORKERS = 1
_MAX_PROMOTION_WORKERS = 64


@dataclass
class LivePoolPlan:
    """Parsed live pool plan (iteration-boundary knobs)."""

    seq: int = 0
    workers: Optional[int] = None
    leaf_servers: Optional[int] = None
    promotion_workers: Optional[int] = None
    apply: str = "next_iter"
    updated: str = ""
    reason: str = ""

    def clamped(
        self,
        *,
        max_workers: Optional[int] = None,
        max_leaf_servers: Optional[int] = None,
    ) -> "LivePoolPlan":
        """Return a copy with knobs clamped to safe ranges."""
        hi_w = int(max_workers if max_workers is not None else _MAX_WORKERS)
        hi_l = int(
            max_leaf_servers if max_leaf_servers is not None else _MAX_LEAF_SERVERS
        )
        hi_w = max(_MIN_WORKERS, min(hi_w, _MAX_WORKERS))
        hi_l = max(_MIN_LEAF_SERVERS, min(hi_l, _MAX_LEAF_SERVERS))

        workers = self.workers
        if workers is not None:
            workers = max(_MIN_WORKERS, min(int(workers), hi_w))

        leaf_servers = self.leaf_servers
        if leaf_servers is not None:
            leaf_servers = max(_MIN_LEAF_SERVERS, min(int(leaf_servers), hi_l))
            if workers is not None:
                leaf_servers = min(leaf_servers, workers)

        promotion_workers = self.promotion_workers
        if promotion_workers is not None:
            promotion_workers = max(
                _MIN_PROMOTION_WORKERS,
                min(int(promotion_workers), _MAX_PROMOTION_WORKERS),
            )

        return LivePoolPlan(
            seq=int(self.seq),
            workers=workers,
            leaf_servers=leaf_servers,
            promotion_workers=promotion_workers,
            apply=str(self.apply or "next_iter"),
            updated=str(self.updated or ""),
            reason=str(self.reason or ""),
        )


def live_pool_enabled() -> bool:
    """True unless ``POKEBOT_LIVE_POOL=0/false/off``."""
    raw = os.environ.get("POKEBOT_LIVE_POOL", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace ``path`` with JSON (tmp + replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_live_pool_plan(
    *,
    seq: int,
    workers: Optional[int] = None,
    leaf_servers: Optional[int] = None,
    promotion_workers: Optional[int] = None,
    reason: str = "",
    path: Optional[Path] = None,
    apply: str = "next_iter",
) -> LivePoolPlan:
    """Write a new live pool plan and return the clamped plan object."""
    plan = LivePoolPlan(
        seq=int(seq),
        workers=workers,
        leaf_servers=leaf_servers,
        promotion_workers=promotion_workers,
        apply=apply,
        updated=time.strftime("%Y-%m-%dT%H:%M:%S"),
        reason=str(reason or ""),
    ).clamped()
    target = Path(path) if path is not None else LIVE_POOL_PLAN_PATH
    atomic_write_json(
        target,
        {
            "seq": plan.seq,
            "workers": plan.workers,
            "leaf_servers": plan.leaf_servers,
            "promotion_workers": plan.promotion_workers,
            "apply": plan.apply,
            "updated": plan.updated,
            "reason": plan.reason,
        },
    )
    return plan


def read_live_pool_plan(path: Optional[Path] = None) -> Optional[LivePoolPlan]:
    """Load plan from disk, or ``None`` if missing/invalid."""
    target = Path(path) if path is not None else LIVE_POOL_PLAN_PATH
    if not target.is_file():
        return None
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    try:
        seq = int(raw.get("seq", 0))
    except (TypeError, ValueError):
        return None
    def _opt_int(key: str) -> Optional[int]:
        if key not in raw or raw[key] is None:
            return None
        try:
            return int(raw[key])
        except (TypeError, ValueError):
            return None

    return LivePoolPlan(
        seq=seq,
        workers=_opt_int("workers"),
        leaf_servers=_opt_int("leaf_servers"),
        promotion_workers=_opt_int("promotion_workers"),
        apply=str(raw.get("apply") or "next_iter"),
        updated=str(raw.get("updated") or ""),
        reason=str(raw.get("reason") or ""),
    ).clamped()


def should_apply_plan(plan: LivePoolPlan, last_seq: int) -> bool:
    """True when plan is newer than ``last_seq`` and marked for next_iter."""
    if int(plan.seq) <= int(last_seq):
        return False
    apply = str(plan.apply or "next_iter").strip().lower()
    return apply in ("next_iter", "next-iter", "iteration", "")


@dataclass
class LeafTopology:
    """Spawned leaf-server fan-out for GPU-server leaf eval."""

    servers: list = field(default_factory=list)
    req_qs: list = field(default_factory=list)
    ctrl_qs: list = field(default_factory=list)
    status_qs: list = field(default_factory=list)
    alive_evts: list = field(default_factory=list)
    resp_qs: list = field(default_factory=list)
    slot_counter: Any = None
    n_servers: int = 0
    n_workers: int = 0
    digest: str = ""
    version: int = 0
    leaf_gpu: str = ""
    queue_depth: int = 32
    max_batch: int = 64
    coalesce_ms: float = 2.0
    timeout_s: float = 30.0

    def remote_channel(self, *, generation: int = 0) -> dict[str, Any]:
        return {
            "req_qs": self.req_qs,
            "resp_qs": self.resp_qs,
            "slot_counter": self.slot_counter,
            "ctrl_qs": self.ctrl_qs,
            "generation": int(generation),
            "alive_evts": self.alive_evts,
            "expected_digest": self.digest,
            "expected_version": int(self.version),
            "timeout_s": float(self.timeout_s),
        }

    def shutdown(self, *, join_timeout_s: float = 10.0) -> None:
        """Stop all leaf servers and close queues."""
        for j, proc in enumerate(self.servers):
            try:
                if j < len(self.ctrl_qs):
                    self.ctrl_qs[j].put_nowait({"cmd": "stop"})
            except Exception:
                pass
            try:
                if j < len(self.req_qs):
                    self.req_qs[j].put_nowait(None)
            except Exception:
                pass
        for proc in self.servers:
            try:
                proc.join(timeout=join_timeout_s)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)
            except Exception:
                pass
        for queue_obj in (
            *self.req_qs,
            *self.ctrl_qs,
            *self.status_qs,
            *self.resp_qs,
        ):
            try:
                queue_obj.cancel_join_thread()
            except Exception:
                pass
            try:
                queue_obj.close()
            except Exception:
                pass
        self.servers.clear()
        self.req_qs.clear()
        self.ctrl_qs.clear()
        self.status_qs.clear()
        self.alive_evts.clear()
        self.resp_qs.clear()
        self.slot_counter = None
        self.n_servers = 0
        self.n_workers = 0


def spawn_leaf_topology(
    *,
    checkpoint_path: str,
    leaf_gpu: str,
    n_workers: int,
    n_servers: int,
    digest: str,
    version: int = 0,
    queue_depth: int = 32,
    max_batch: int = 64,
    coalesce_ms: float = 2.0,
    timeout_s: float = 30.0,
    ready_timeout_s: float = 240.0,
    bf16: bool = True,
) -> LeafTopology:
    """Spawn ``n_servers`` leaf replicas serving ``n_workers`` response slots."""
    from .batched_infer import run_leaf_server

    n_workers = max(_MIN_WORKERS, min(int(n_workers), _MAX_WORKERS))
    n_servers = max(
        _MIN_LEAF_SERVERS,
        min(int(n_servers), n_workers, _MAX_LEAF_SERVERS),
    )
    mpctx = mp.get_context("spawn")
    topo = LeafTopology(
        n_servers=n_servers,
        n_workers=n_workers,
        digest=str(digest),
        version=int(version),
        leaf_gpu=str(leaf_gpu),
        queue_depth=int(queue_depth),
        max_batch=int(max_batch),
        coalesce_ms=float(coalesce_ms),
        timeout_s=float(timeout_s),
    )
    topo.resp_qs = [mpctx.Queue(maxsize=2) for _ in range(n_workers)]
    topo.slot_counter = mpctx.Value("i", 0)
    readies = []
    for _ in range(n_servers):
        rq = mpctx.Queue(maxsize=int(queue_depth))
        cq = mpctx.Queue(maxsize=8)
        sq = mpctx.Queue(maxsize=16)
        ev = mpctx.Event()
        alive = mpctx.Event()
        proc = mpctx.Process(
            target=run_leaf_server,
            args=(str(checkpoint_path), str(leaf_gpu), rq, topo.resp_qs),
            kwargs=dict(
                ready_evt=ev,
                alive_evt=alive,
                ctrl_q=cq,
                status_q=sq,
                expected_digest=str(digest),
                initial_version=int(version),
                bf16=bool(bf16),
                max_batch=int(max_batch),
                coalesce_ms=float(coalesce_ms),
            ),
            daemon=True,
        )
        proc.start()
        topo.req_qs.append(rq)
        topo.ctrl_qs.append(cq)
        topo.status_qs.append(sq)
        topo.alive_evts.append(alive)
        topo.servers.append(proc)
        readies.append(ev)

    for j, ev in enumerate(readies):
        if not ev.wait(timeout=float(ready_timeout_s)):
            topo.shutdown()
            raise RuntimeError(
                f"leaf server {j} did not signal ready in {ready_timeout_s}s"
            )
        try:
            status = topo.status_qs[j].get(timeout=5)
        except Exception as exc:
            topo.shutdown()
            raise RuntimeError(f"missing leaf ready status: {exc}") from exc
        if (
            not status.get("ok")
            or status.get("checkpoint_digest") != digest
            or int(status.get("version", -1)) != int(version)
            or not topo.servers[j].is_alive()
            or not topo.alive_evts[j].is_set()
        ):
            topo.shutdown()
            raise RuntimeError(f"invalid leaf ready ack: {status}")
    return topo


def rebuild_leaf_topology(
    previous: Optional[LeafTopology],
    **spawn_kwargs: Any,
) -> LeafTopology:
    """Shut down ``previous`` (if any) and spawn a fresh topology."""
    if previous is not None:
        previous.shutdown()
    return spawn_leaf_topology(**spawn_kwargs)
