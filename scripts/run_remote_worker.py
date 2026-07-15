#!/usr/bin/env python3
"""Remote sim + co-located GPU leaf worker for LAN offload (TrueNAS elmo).

Starts leaf-eval server(s) on the local NVIDIA GPU (RTX 3060 LHR 12 GB on elmo)
and a CPU WorkerPool for game jobs. Training-box clients submit whole-game jobs
over TCP (:mod:`poke_bot.remote_jobs`); leaf forwards stay on this host via IPC.

Sized for Ryzen 9 5900X headroom (~20–22 SIM workers) + 12 GB 3060 LHR leaf.
Does not touch the training-box overnight loops — run this only on the remote
box / container.
"""

from __future__ import annotations

import argparse
import importlib.util
import multiprocessing as mp
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _default_leaf_gpu() -> str:
    """Prefer env, else CUDA, else native MPS (bert), else cuda:0 for NVIDIA hosts."""
    env = os.environ.get("LEAF_GPU")
    if env:
        return env
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cuda:0"


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SIM_WORKERS", "20")),
        help="CPU sim workers (5900X: 20–22 leaves headroom for leaf+OS)",
    )
    p.add_argument(
        "--leaf-servers",
        type=int,
        default=int(os.environ.get("LEAF_SERVERS", "2")),
        help="GPU leaf replicas on the local card (3060 LHR 12 GB: 2 is safe)",
    )
    p.add_argument(
        "--leaf-gpu",
        default=_default_leaf_gpu(),
        help="Leaf device: cuda:N on NVIDIA (elmo), mps on Mac bert (auto if unset)",
    )
    p.add_argument(
        "--leaf-max-batch",
        type=int,
        default=int(os.environ.get("LEAF_MAX_BATCH", "192")),
        help="Per-replica leaf batch on 12 GB 3060 LHR (keep under ~192–224)",
    )
    p.add_argument(
        "--leaf-queue-depth",
        type=int,
        default=int(os.environ.get("LEAF_QUEUE_DEPTH", "96")),
    )
    p.add_argument(
        "--leaf-coalesce-ms",
        type=float,
        default=float(os.environ.get("LEAF_COALESCE_MS", "2")),
    )
    p.add_argument(
        "--checkpoint",
        default=os.environ.get("POKEBOT_CHECKPOINT", ""),
        help="Initial champion .pt (mounted volume path inside container)",
    )
    p.add_argument(
        "--cg-lib-path",
        default=os.environ.get("CG_LIB_PATH", ""),
    )
    p.add_argument(
        "--cpu-only-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mask CUDA in sim workers (leaf servers keep the GPU)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Boot leaf+pool, ping health once, exit (no listen loop)",
    )
    return p.parse_args(argv)


def _load_rr_module():
    """Import train_round_robin worker callables without executing main."""
    path = REPO_ROOT / "scripts" / "train_round_robin.py"
    spec = importlib.util.spec_from_file_location("train_round_robin_remote", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    # Avoid running ``if __name__ == '__main__'``.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _gpu_name(device_str: str) -> str:
    try:
        import torch

        kind = str(device_str).split(":", 1)[0].lower()
        if kind == "mps":
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "Apple MPS"
            return "mps (unavailable)"
        if kind == "cpu" or not torch.cuda.is_available():
            return "cpu"
        idx = 0
        if ":" in device_str:
            idx = int(device_str.split(":")[-1])
        return torch.cuda.get_device_name(idx)
    except Exception as exc:  # noqa: BLE001
        return f"unavailable ({exc})"


def _free_ram_gb() -> Optional[float]:
    try:
        import psutil

        return float(psutil.virtual_memory().available) / (1024**3)
    except Exception:
        try:
            page = os.sysconf("SC_PAGE_SIZE")
            avail = os.sysconf("SC_AVPHYS_PAGES")
            return float(page * avail) / (1024**3)
        except Exception:
            return None


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if args.cpu_only_workers:
        os.environ["POKEBOT_WORKER_CPU_ONLY"] = "1"
    if args.cg_lib_path:
        os.environ["CG_LIB_PATH"] = args.cg_lib_path

    from poke_bot import config
    from poke_bot.batched_infer import run_leaf_server
    from poke_bot.checkpoint import checkpoint_digest
    from poke_bot.remote_jobs import serve_forever
    from poke_bot.worker_pool import WorkerPool

    config.apply_runtime_perf()

    ckpt = Path(args.checkpoint).expanduser() if args.checkpoint else None
    if ckpt is None or not ckpt.is_file():
        print(
            "[remote-worker] ERROR: --checkpoint / POKEBOT_CHECKPOINT must point "
            f"to a .pt file (got {args.checkpoint!r})",
            file=sys.stderr,
        )
        return 2

    digest = checkpoint_digest(str(ckpt))
    n_workers = max(1, int(args.workers))
    n_servers = max(1, min(int(args.leaf_servers), n_workers))
    leaf_gpu = str(args.leaf_gpu)

    print(
        f"[remote-worker] host={socket.gethostname()} "
        f"gpu={_gpu_name(leaf_gpu)!r} device={leaf_gpu} "
        f"workers={n_workers} leaf_servers={n_servers} "
        f"leaf_max_batch={args.leaf_max_batch} ckpt={ckpt.name} "
        f"digest={digest[:12]}…",
        flush=True,
    )

    mpctx = mp.get_context("spawn")
    leaf_resp_qs = [mpctx.Queue(maxsize=2) for _ in range(n_workers)]
    leaf_req_qs = []
    leaf_ctrl_qs = []
    leaf_status_qs = []
    leaf_alive_evts = []
    leaf_servers = []
    leaf_version = 0
    readies = []

    for j in range(n_servers):
        rq = mpctx.Queue(maxsize=int(args.leaf_queue_depth))
        cq = mpctx.Queue(maxsize=8)
        sq = mpctx.Queue(maxsize=16)
        ev = mpctx.Event()
        alive = mpctx.Event()
        proc = mpctx.Process(
            target=run_leaf_server,
            args=(str(ckpt), leaf_gpu, rq, leaf_resp_qs),
            kwargs=dict(
                ready_evt=ev,
                alive_evt=alive,
                ctrl_q=cq,
                status_q=sq,
                expected_digest=digest,
                initial_version=leaf_version,
                bf16=True,
                max_batch=int(args.leaf_max_batch),
                coalesce_ms=float(args.leaf_coalesce_ms),
            ),
            daemon=True,
        )
        proc.start()
        leaf_req_qs.append(rq)
        leaf_ctrl_qs.append(cq)
        leaf_status_qs.append(sq)
        leaf_alive_evts.append(alive)
        leaf_servers.append(proc)
        readies.append(ev)

    startup_ok = True
    for j, ev in enumerate(readies):
        if not ev.wait(timeout=240):
            print(f"[remote-worker] ERROR: leaf[{j}] not ready in 240s", file=sys.stderr)
            startup_ok = False
            continue
        try:
            status = leaf_status_qs[j].get(timeout=5)
        except Exception as exc:
            print(f"[remote-worker] ERROR: leaf[{j}] status: {exc}", file=sys.stderr)
            startup_ok = False
            continue
        if not status.get("ok") or status.get("checkpoint_digest") != digest:
            print(f"[remote-worker] ERROR: leaf[{j}] bad ack: {status}", file=sys.stderr)
            startup_ok = False
        else:
            print(f"[remote-worker] leaf[{j}] ready ok={status.get('ok')}", flush=True)

    if not startup_ok:
        for j, proc in enumerate(leaf_servers):
            try:
                leaf_ctrl_qs[j].put({"cmd": "stop"})
                proc.join(timeout=5)
            except Exception:
                pass
        return 3

    remote_channel = {
        "req_qs": leaf_req_qs,
        "resp_qs": leaf_resp_qs,
        "alive_evts": leaf_alive_evts,
        "expected_digest": digest,
        "expected_version": leaf_version,
        "timeout_s": float(config.SEARCH.remote_request_timeout_s),
        "generation": 0,
    }

    rr = _load_rr_module()
    pool = WorkerPool(
        num_workers=n_workers,
        cg_lib_path=args.cg_lib_path or None,
        remote_channel=remote_channel,
    )
    pool.__enter__()
    stop_event = threading.Event()
    state: dict[str, Any] = {
        "digest": digest,
        "version": leaf_version,
        "checkpoint": str(ckpt),
        "jobs_completed": 0,
        "jobs_failed": 0,
        "started_at": time.time(),
    }

    def hello() -> dict[str, Any]:
        return {
            "hostname": socket.gethostname(),
            "workers": n_workers,
            "leaf_servers": n_servers,
            "gpu_name": _gpu_name(leaf_gpu),
            "device": leaf_gpu,
            "checkpoint_digest": state["digest"],
            "checkpoint_version": state["version"],
            "free_ram_gb": _free_ram_gb(),
            "jobs_completed": state["jobs_completed"],
            "jobs_failed": state["jobs_failed"],
            "uptime_s": time.time() - state["started_at"],
        }

    def handler(msg: dict[str, Any]) -> dict[str, Any]:
        mtype = msg.get("type")
        if mtype == "health":
            alive = all(p.is_alive() and ev.is_set() for p, ev in zip(leaf_servers, leaf_alive_evts))
            return {
                "type": "health_ok",
                "ok": alive,
                **hello(),
                "leaf_alive": alive,
            }
        if mtype == "reload":
            path = str(msg["path"])
            requested_digest = msg.get("digest")
            new_version = int(msg.get("version", state["version"] + 1))
            actual = checkpoint_digest(path)
            if requested_digest is not None and actual != requested_digest:
                return {
                    "type": "reload_ok",
                    "ok": False,
                    "error": f"digest mismatch: expected {requested_digest}, got {actual}",
                }
            for cq in leaf_ctrl_qs:
                cq.put(
                    {
                        "cmd": "reload",
                        "path": path,
                        "digest": actual,
                        "version": new_version,
                    }
                )
            # Wait for each replica ack.
            for j, sq in enumerate(leaf_status_qs):
                status = sq.get(timeout=120)
                if not status.get("ok"):
                    return {
                        "type": "reload_ok",
                        "ok": False,
                        "error": f"leaf[{j}] reload failed: {status}",
                    }
            state["digest"] = actual
            state["version"] = new_version
            state["checkpoint"] = path
            remote_channel["expected_digest"] = actual
            remote_channel["expected_version"] = new_version
            return {
                "type": "reload_ok",
                "ok": True,
                "checkpoint_digest": actual,
                "version": new_version,
            }
        if mtype == "pin":
            path = str(msg["path"])
            requested_digest = msg.get("digest")
            actual = checkpoint_digest(path)
            if requested_digest is not None and actual != requested_digest:
                return {
                    "type": "pin_ok",
                    "ok": False,
                    "error": f"digest mismatch: expected {requested_digest}, got {actual}",
                }
            for cq in leaf_ctrl_qs:
                cq.put({"cmd": "pin", "path": path, "digest": actual})
            for j, sq in enumerate(leaf_status_qs):
                status = sq.get(timeout=120)
                if not status.get("ok"):
                    return {
                        "type": "pin_ok",
                        "ok": False,
                        "error": f"leaf[{j}] pin failed: {status}",
                    }
            return {
                "type": "pin_ok",
                "ok": True,
                "checkpoint_digest": actual,
                "pinned_digests": status.get("pinned_digests"),
            }
        if mtype == "unpin":
            digest = str(msg["digest"])
            for cq in leaf_ctrl_qs:
                cq.put({"cmd": "unpin", "digest": digest})
            for j, sq in enumerate(leaf_status_qs):
                status = sq.get(timeout=120)
                if not status.get("ok"):
                    return {
                        "type": "unpin_ok",
                        "ok": False,
                        "error": f"leaf[{j}] unpin failed: {status}",
                    }
            return {"type": "unpin_ok", "ok": True, "digest": digest}
        if mtype == "job":
            kind = str(msg.get("kind") or "play")
            job = msg.get("job")
            if not isinstance(job, dict):
                return {"type": "result", "ok": False, "error": "job must be object"}
            if kind == "play":
                # Collection jobs inherit the worker's pinned incumbent digest.
                job = {
                    **job,
                    "checkpoint": state["checkpoint"],
                    "checkpoint_digest": state["digest"],
                    "device": "cpu",
                }
            else:
                job = {**job, "device": "cpu"}
            worker_fn = rr._worker_play if kind == "play" else rr._worker_promotion
            # One job per request; pool may have many concurrent sockets.
            try:
                result = next(pool.imap_unordered(worker_fn, [job]))
                state["jobs_completed"] += 1
                return {"type": "result", "ok": True, "result": result}
            except BaseException as exc:  # noqa: BLE001
                state["jobs_failed"] += 1
                return {
                    "type": "result",
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        return {"type": "error", "error": f"unknown message type {mtype!r}"}

    if args.smoke:
        health = handler({"type": "health"})
        print(f"[remote-worker] smoke health={health}", flush=True)
        ok = bool(health.get("ok"))
        pool.__exit__(None, None, None)
        for j, proc in enumerate(leaf_servers):
            try:
                leaf_ctrl_qs[j].put({"cmd": "stop"})
                proc.join(timeout=5)
            except Exception:
                pass
        return 0 if ok else 4

    print(
        f"[remote-worker] listening on {args.host}:{args.port} "
        f"(5900X-sized workers={n_workers}, leaf on 3060 LHR 12 GB)",
        flush=True,
    )
    try:
        serve_forever(
            handler,
            host=args.host,
            port=int(args.port),
            hello=hello,
            stop_event=stop_event,
        )
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        pool.__exit__(None, None, None)
        for j, proc in enumerate(leaf_servers):
            try:
                leaf_ctrl_qs[j].put({"cmd": "stop"})
                proc.join(timeout=5)
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
