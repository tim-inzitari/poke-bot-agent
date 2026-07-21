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
import hashlib
import json
import multiprocessing as mp
import os
import signal
import socket
import stat
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Advertised capability set for trainer dispatch / redeploy gates.
REMOTE_JOB_KINDS = ("play", "promotion", "self_play")
REMOTE_WORKER_SAFETY_VERSION = "20260717"
REMOTE_WORKER_ARM_FILE = REPO_ROOT / "outputs" / "state" / "REMOTE_WORKER_ARMED"
# Production supervisors may opt into this reserved code to distinguish a
# completed max-service-jobs rotation from a resource or health failure. The
# default remains zero for existing canary/Bert wrappers.
REMOTE_WORKER_PLANNED_ROTATION_EXIT_CODE = 75
REMOTE_WORKER_WATCHDOG_EXIT_CODE = 70
REMOTE_ACTIVE_CHECKPOINT_FILE_ENV = "POKEBOT_REMOTE_ACTIVE_CHECKPOINT_FILE"
REMOTE_CHECKPOINT_ROOT_ENV = "POKEBOT_REMOTE_CHECKPOINT_ROOT"


def _raw_sha256_digest(path: Path) -> str:
    """Return the checkpoint identity without importing Torch/project modules."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _persist_active_checkpoint(
    path: str | Path,
    digest: str,
    *,
    source: Optional[dict[str, str]] = None,
) -> Optional[Path]:
    """Atomically publish the checkpoint a production rotation must reload.

    Canary and Bert processes do not configure ``REMOTE_ACTIVE_CHECKPOINT_FILE``
    and therefore remain unchanged.  Elmo production configures both this file
    and a checkpoint root on its durable ``runtime-logs`` / read-only
    ``checkpoint`` mounts.  Re-hashing immediately before publication prevents
    a host-side file replacement between leaf acknowledgement and persistence.
    """
    env = os.environ if source is None else source
    raw_state_file = str(env.get(REMOTE_ACTIVE_CHECKPOINT_FILE_ENV, "")).strip()
    if not raw_state_file:
        return None

    state_file = Path(raw_state_file).expanduser()
    if not state_file.is_absolute():
        raise ValueError(
            f"{REMOTE_ACTIVE_CHECKPOINT_FILE_ENV} must be absolute: {state_file}"
        )
    raw_root = str(env.get(REMOTE_CHECKPOINT_ROOT_ENV, "")).strip()
    if not raw_root:
        raise ValueError(
            f"{REMOTE_CHECKPOINT_ROOT_ENV} is required when "
            f"{REMOTE_ACTIVE_CHECKPOINT_FILE_ENV} is configured"
        )
    checkpoint_root = Path(raw_root).expanduser()
    if not checkpoint_root.is_absolute():
        raise ValueError(
            f"{REMOTE_CHECKPOINT_ROOT_ENV} must be absolute: {checkpoint_root}"
        )

    resolved_root = checkpoint_root.resolve(strict=True)
    resolved_checkpoint = Path(path).expanduser().resolve(strict=True)
    try:
        resolved_checkpoint.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"active checkpoint {resolved_checkpoint} escapes root {resolved_root}"
        ) from exc
    if not resolved_checkpoint.is_file():
        raise ValueError(f"active checkpoint is not a regular file: {resolved_checkpoint}")

    expected = str(digest)
    if (
        not expected.startswith("sha256:")
        or len(expected) != len("sha256:") + 64
        or any(ch not in "0123456789abcdef" for ch in expected[7:])
    ):
        raise ValueError(f"invalid checkpoint digest: {expected!r}")
    actual = _raw_sha256_digest(resolved_checkpoint)
    if actual != expected:
        raise ValueError(
            f"checkpoint changed before durable publication: {actual} != {expected}"
        )

    state_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_file.with_name(
        f"{state_file.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    payload = {
        "version": 1,
        "path": str(resolved_checkpoint),
        "digest": expected,
        "published_at_epoch": time.time(),
    }
    fd: Optional[int] = None
    try:
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            fd = None
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, state_file)
        try:
            directory_fd = os.open(state_file.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if fd is not None:
            os.close(fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return state_file


def _select_startup_checkpoint(
    configured_checkpoint: str | Path,
    *,
    source: Optional[dict[str, str]] = None,
) -> tuple[Path, Optional[str]]:
    """Select and verify the checkpoint that a new service lifetime must load.

    Without ``REMOTE_ACTIVE_CHECKPOINT_FILE`` this is intentionally a no-op so
    canary and Bert retain their existing startup behavior. Once production
    configures durable continuity, however, the record is authoritative:
    missing/corrupt state, a path outside the read-only checkpoint root, or a
    digest mismatch all raise. Falling back to the image's original
    ``model.pt`` would silently undo a trainer reload at the next rotation.
    """
    env = os.environ if source is None else source
    raw_state_file = str(env.get(REMOTE_ACTIVE_CHECKPOINT_FILE_ENV, "")).strip()
    if not raw_state_file:
        return Path(configured_checkpoint).expanduser(), None

    state_file = Path(raw_state_file).expanduser()
    if not state_file.is_absolute():
        raise ValueError(
            f"{REMOTE_ACTIVE_CHECKPOINT_FILE_ENV} must be absolute: {state_file}"
        )
    raw_root = str(env.get(REMOTE_CHECKPOINT_ROOT_ENV, "")).strip()
    if not raw_root:
        raise ValueError(
            f"{REMOTE_CHECKPOINT_ROOT_ENV} is required when "
            f"{REMOTE_ACTIVE_CHECKPOINT_FILE_ENV} is configured"
        )
    checkpoint_root = Path(raw_root).expanduser()
    if not checkpoint_root.is_absolute():
        raise ValueError(
            f"{REMOTE_CHECKPOINT_ROOT_ENV} must be absolute: {checkpoint_root}"
        )
    try:
        resolved_root = checkpoint_root.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"checkpoint root is unavailable: {checkpoint_root}: {exc}") from exc
    if not resolved_root.is_dir():
        raise ValueError(f"checkpoint root is not a directory: {resolved_root}")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(state_file, flags)
    except OSError as exc:
        raise ValueError(
            f"active checkpoint state is unreadable: {state_file}: {exc}"
        ) from exc
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(
                f"active checkpoint state is not a regular file: {state_file}"
            )
        if metadata.st_size > 16 * 1024:
            raise ValueError(
                f"active checkpoint state is unexpectedly large: {metadata.st_size}"
            )
        with os.fdopen(fd, "r", encoding="utf-8") as stream:
            fd = -1
            raw_payload = stream.read(16 * 1024 + 1)
    except (OSError, UnicodeError) as exc:
        raise ValueError(
            f"active checkpoint state could not be read: {state_file}: {exc}"
        ) from exc
    finally:
        if fd >= 0:
            os.close(fd)

    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"active checkpoint state is invalid JSON: {state_file}: {exc}"
        ) from exc
    required_keys = {"version", "path", "digest", "published_at_epoch"}
    if not isinstance(payload, dict) or set(payload) != required_keys:
        raise ValueError(
            "active checkpoint state must contain exactly "
            f"{sorted(required_keys)}"
        )
    if type(payload["version"]) is not int or payload["version"] != 1:
        raise ValueError(
            f"unsupported active checkpoint state version: {payload['version']!r}"
        )
    published = payload["published_at_epoch"]
    if (
        isinstance(published, bool)
        or not isinstance(published, (int, float))
        or float(published) <= 0
    ):
        raise ValueError(f"invalid active checkpoint publication time: {published!r}")

    raw_path = payload["path"]
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"invalid active checkpoint path: {raw_path!r}")
    active_path = Path(raw_path).expanduser()
    if not active_path.is_absolute():
        raise ValueError(f"active checkpoint path must be absolute: {active_path}")
    try:
        resolved_checkpoint = active_path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(
            f"active checkpoint is unavailable: {active_path}: {exc}"
        ) from exc
    try:
        resolved_checkpoint.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"active checkpoint {resolved_checkpoint} escapes root {resolved_root}"
        ) from exc
    if not resolved_checkpoint.is_file():
        raise ValueError(
            f"active checkpoint is not a regular file: {resolved_checkpoint}"
        )

    expected = payload["digest"]
    if (
        not isinstance(expected, str)
        or not expected.startswith("sha256:")
        or len(expected) != len("sha256:") + 64
        or any(ch not in "0123456789abcdef" for ch in expected[7:])
    ):
        raise ValueError(f"invalid active checkpoint digest: {expected!r}")
    actual = _raw_sha256_digest(resolved_checkpoint)
    if actual != expected:
        raise ValueError(
            f"active checkpoint digest mismatch: state={expected} actual={actual}"
        )
    return resolved_checkpoint, expected


def _service_shutdown_exit_code(
    *,
    planned_rotation: bool,
    planned_rotation_exit_code: int,
) -> int:
    """Return a supervisor-safe code for one service shutdown cause."""
    if planned_rotation:
        return int(planned_rotation_exit_code)
    return REMOTE_WORKER_WATCHDOG_EXIT_CODE


def _remote_worker_arm_error(*, smoke: bool) -> Optional[str]:
    """Return why production startup is unarmed, or ``None`` when armed."""
    if smoke:
        return None
    safety_version = os.environ.get("POKEBOT_REMOTE_WORKER_SAFETY_VERSION", "")
    if safety_version != REMOTE_WORKER_SAFETY_VERSION:
        return (
            "service is not armed with memory safety version "
            f"{REMOTE_WORKER_SAFETY_VERSION}"
        )
    arm_file = Path(
        os.environ.get("POKEBOT_REMOTE_WORKER_ARM_FILE", str(REMOTE_WORKER_ARM_FILE))
    ).expanduser()
    try:
        token = arm_file.read_text(encoding="utf-8")
    except OSError as exc:
        return f"arm token is unreadable at {arm_file}: {type(exc).__name__}: {exc}"
    if token != REMOTE_WORKER_SAFETY_VERSION:
        return (
            f"arm token at {arm_file} must contain exactly "
            f"{REMOTE_WORKER_SAFETY_VERSION!r}"
        )
    return None


def _validated_leaf_status(
    index: int,
    status: object,
    *,
    expected_type: str,
    expected_digest: str,
    expected_version: int,
) -> tuple[dict[str, Any], list[str]]:
    """Normalize one leaf acknowledgement and verify its exact identity.

    A process/event pair only proves that a process is running.  It does not
    prove which model that process has resident.  Keep the normalized frame so
    health can continue to expose the last identity acknowledged by each leaf.
    """
    problems: list[str] = []
    if not isinstance(status, dict):
        return (
            {
                "index": int(index),
                "type": "invalid",
                "ok": False,
                "checkpoint_digest": None,
                "version": None,
                "error": f"non-object status: {status!r}",
            },
            [f"leaf[{index}] returned non-object status: {status!r}"],
        )

    got_type = str(status.get("type") or "")
    got_digest = status.get("checkpoint_digest")
    raw_version = status.get("version")
    try:
        got_version: Optional[int] = int(raw_version)
    except (TypeError, ValueError):
        got_version = None
    ok = status.get("ok") is True
    view = {
        "index": int(index),
        "type": got_type,
        "ok": ok,
        "checkpoint_digest": got_digest,
        "version": got_version,
        "error": status.get("error"),
    }
    if not ok:
        problems.append(f"leaf[{index}] {expected_type} failed: {status}")
    if got_type != expected_type:
        problems.append(
            f"leaf[{index}] expected type={expected_type!r}, got {got_type!r}"
        )
    if got_digest != expected_digest:
        problems.append(
            f"leaf[{index}] digest mismatch: expected {expected_digest}, "
            f"got {got_digest}"
        )
    if got_version != int(expected_version):
        problems.append(
            f"leaf[{index}] version mismatch: expected {int(expected_version)}, "
            f"got {got_version}"
        )
    return view, problems


def _status_is_current_attempt(
    status: object,
    *,
    expected_type: str,
    expected_version: int,
) -> bool:
    """Return whether a status frame can belong to this control attempt.

    Status queues survive parent-side timeouts. A late acknowledgement from an
    older version must be consumed and ignored rather than being attributed to
    the next reload. Digest/``ok`` validation is deliberately left to
    :func:`_validated_leaf_status`: a current-attempt failure often reports the
    previous resident digest and still needs to fail the attempt immediately.
    """
    if not isinstance(status, dict):
        return False
    if str(status.get("type") or "") != expected_type:
        return False
    try:
        return int(status.get("version")) == int(expected_version)
    except (TypeError, ValueError):
        return False


def _leaf_health_report(
    leaf_statuses: list[dict[str, Any]],
    *,
    process_alive: list[bool],
    event_alive: list[bool],
    expected_digest: str,
    expected_version: int,
    controller_healthy: bool,
    controller_error: Optional[str] = None,
) -> dict[str, Any]:
    """Build fail-closed health from process liveness *and* leaf identity."""
    n = max(len(leaf_statuses), len(process_alive), len(event_alive))
    leaves: list[dict[str, Any]] = []
    for index in range(n):
        status = (
            dict(leaf_statuses[index])
            if index < len(leaf_statuses)
            else {
                "index": index,
                "type": "missing",
                "ok": False,
                "checkpoint_digest": None,
                "version": None,
                "error": "no acknowledged leaf identity",
            }
        )
        proc_ok = bool(process_alive[index]) if index < len(process_alive) else False
        event_ok = bool(event_alive[index]) if index < len(event_alive) else False
        identity_ok = bool(
            status.get("ok") is True
            and status.get("checkpoint_digest") == expected_digest
            and status.get("version") == int(expected_version)
        )
        status.update(
            {
                "process_alive": proc_ok,
                "event_alive": event_ok,
                "identity_ok": identity_ok,
                "healthy": proc_ok and event_ok and identity_ok,
            }
        )
        leaves.append(status)

    leaf_alive = n > 0 and all(
        leaf["process_alive"] and leaf["event_alive"] for leaf in leaves
    )
    leaf_identity_ok = n > 0 and all(leaf["identity_ok"] for leaf in leaves)
    ok = bool(controller_healthy and leaf_alive and leaf_identity_ok)
    return {
        "ok": ok,
        "leaf_alive": leaf_alive,
        "leaf_identity_ok": leaf_identity_ok,
        "leaves": leaves,
        "controller_error": controller_error,
    }


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
        default=int(os.environ.get("SIM_WORKERS", "4")),
        help=(
            "CPU sim worker pool size. The fail-safe default is a four-worker "
            "canary; scale only after measured RSS plateaus."
        ),
    )
    p.add_argument(
        "--default-workers",
        type=int,
        default=int(os.environ.get("SIM_DEFAULT_WORKERS", "0")),
        help=(
            "Steady-state hello advertise (default demand). 0 → auto: "
            "the full bounded canary pool."
        ),
    )
    p.add_argument(
        "--leaf-servers",
        type=int,
        default=int(os.environ.get("LEAF_SERVERS", "1")),
        help="GPU leaf replicas (fail-safe default: one)",
    )
    p.add_argument(
        "--leaf-gpu",
        default=_default_leaf_gpu(),
        help="Leaf device: cuda:N on NVIDIA (elmo), mps on Mac bert (auto if unset)",
    )
    p.add_argument(
        "--leaf-max-batch",
        type=int,
        default=int(os.environ.get("LEAF_MAX_BATCH", "96")),
        help="Per-replica leaf batch (fail-safe default: 96)",
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
    p.add_argument(
        "--max-connections",
        type=int,
        default=int(os.environ.get("POKEBOT_REMOTE_MAX_CONNECTIONS", "32")),
        help="Hard cap on simultaneous TCP connection threads",
    )
    p.add_argument(
        "--tree-rss-limit-gb",
        type=float,
        default=float(os.environ.get("POKEBOT_REMOTE_TREE_RSS_LIMIT_GB", "32")),
        help="Fail closed when parent+descendant RSS reaches this bound (0 disables)",
    )
    p.add_argument(
        "--min-free-ram-gb",
        type=float,
        default=float(os.environ.get("POKEBOT_REMOTE_MIN_FREE_RAM_GB", "8")),
        help="Fail closed below this host available-RAM floor (0 disables)",
    )
    p.add_argument(
        "--max-service-jobs",
        type=int,
        default=int(os.environ.get("POKEBOT_REMOTE_MAX_SERVICE_JOBS", "0")),
        help="Drain and exit after this many completed jobs (0 disables rotation)",
    )
    p.add_argument(
        "--planned-rotation-exit-code",
        type=int,
        default=int(
            os.environ.get("POKEBOT_REMOTE_PLANNED_ROTATION_EXIT_CODE", "0")
        ),
        help=(
            "Exit code used only after a completed max-service-jobs drain. "
            "Production supervision reserves 75; zero preserves legacy wrappers."
        ),
    )
    p.add_argument(
        "--watchdog-interval-s",
        type=float,
        default=float(os.environ.get("POKEBOT_REMOTE_WATCHDOG_INTERVAL_S", "5")),
    )
    return p.parse_args(argv)


def _load_rr_module():
    """Import train_round_robin (side-effect warm); job dispatch uses package wrappers."""
    from poke_bot.remote_sim_jobs import load_round_robin_module

    return load_round_robin_module()


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


def _process_tree_rss_gb() -> Optional[float]:
    """Best-effort RSS for this service and every current descendant."""
    try:
        import psutil

        root = psutil.Process()
        processes = [root, *root.children(recursive=True)]
        total = 0
        for proc in processes:
            try:
                total += int(proc.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return float(total) / (1024**3)
    except Exception:
        return None


def _close_mp_queue(queue_obj: object) -> None:
    """Close one multiprocessing queue without waiting on stale payloads."""
    try:
        cancel = getattr(queue_obj, "cancel_join_thread")
        cancel()
    except Exception:
        pass
    try:
        close = getattr(queue_obj, "close")
        close()
    except Exception:
        pass


def _close_mp_queues(*groups: list[object]) -> None:
    seen: set[int] = set()
    for group in groups:
        for queue_obj in group:
            ident = id(queue_obj)
            if ident in seen:
                continue
            seen.add(ident)
            _close_mp_queue(queue_obj)


def _shutdown_leaf_servers(
    leaf_servers: list[mp.Process],
    control_queues: list[object],
    *,
    graceful_timeout_s: float = 8.0,
) -> tuple[int, ...]:
    """Stop, terminate, then kill leaf children; return any survivors."""
    for control_queue in control_queues:
        try:
            control_queue.put({"cmd": "stop"}, timeout=0.2)
        except Exception:
            pass

    deadline = time.monotonic() + max(0.0, float(graceful_timeout_s))
    for proc in leaf_servers:
        try:
            proc.join(timeout=max(0.0, deadline - time.monotonic()))
        except Exception:
            pass
    for proc in leaf_servers:
        try:
            if proc.is_alive():
                proc.terminate()
        except Exception:
            pass
    for proc in leaf_servers:
        try:
            proc.join(timeout=3.0)
        except Exception:
            pass
    for proc in leaf_servers:
        try:
            if proc.is_alive():
                kill = getattr(proc, "kill", None)
                if callable(kill):
                    kill()
        except Exception:
            pass
    survivors: list[int] = []
    for proc in leaf_servers:
        try:
            proc.join(timeout=3.0)
            if proc.is_alive() and proc.pid is not None:
                survivors.append(int(proc.pid))
        except Exception:
            if proc.pid is not None:
                survivors.append(int(proc.pid))
    return tuple(survivors)


def main(argv: Optional[list[str]] = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    # Gate before _parse_args: its leaf-device default probes torch. No project
    # runtime import or child creation is allowed until both independent arming
    # signals are present. Smoke remains available for deployment preflight.
    arm_error = _remote_worker_arm_error(smoke="--smoke" in raw_argv)
    if arm_error is not None:
        print(
            f"[remote-worker] ERROR: {arm_error}; refusing to import or create a pool",
            file=sys.stderr,
        )
        return 78
    args = _parse_args(raw_argv)
    if int(args.planned_rotation_exit_code) not in (
        0,
        REMOTE_WORKER_PLANNED_ROTATION_EXIT_CODE,
    ):
        print(
            "[remote-worker] ERROR: --planned-rotation-exit-code must be 0 "
            f"or {REMOTE_WORKER_PLANNED_ROTATION_EXIT_CODE}",
            file=sys.stderr,
        )
        return 64
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if args.cpu_only_workers:
        os.environ["POKEBOT_WORKER_CPU_ONLY"] = "1"
    if args.cg_lib_path:
        os.environ["CG_LIB_PATH"] = args.cg_lib_path

    configured_checkpoint = args.checkpoint or ""
    try:
        ckpt, durable_startup_digest = _select_startup_checkpoint(
            configured_checkpoint
        )
    except (OSError, ValueError) as exc:
        print(
            "[remote-worker] ERROR: durable active-checkpoint state is invalid; "
            f"refusing configured-checkpoint fallback: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 78

    from poke_bot import config
    from poke_bot.batched_infer import run_leaf_server
    from poke_bot.checkpoint import checkpoint_digest
    from poke_bot.remote_jobs import serve_forever
    from poke_bot.remote_sim_jobs import (
        remote_play_job,
        remote_promotion_job,
        remote_self_play_job,
    )
    from poke_bot.worker_pool import WorkerPool

    config.apply_runtime_perf()

    if not ckpt.is_file():
        print(
            "[remote-worker] ERROR: --checkpoint / POKEBOT_CHECKPOINT must point "
            f"to a .pt file (got {configured_checkpoint!r})",
            file=sys.stderr,
        )
        return 2

    digest = checkpoint_digest(str(ckpt))
    if durable_startup_digest is not None and digest != durable_startup_digest:
        print(
            "[remote-worker] ERROR: active checkpoint changed after durable "
            f"state validation: {digest} != {durable_startup_digest}",
            file=sys.stderr,
        )
        return 78
    n_workers = max(1, int(args.workers))
    n_servers = max(1, min(int(args.leaf_servers), n_workers))
    leaf_gpu = str(args.leaf_gpu)
    # An explicit lower advertise is allowed, but the unconfigured path exposes
    # only the deliberately small canary pool.
    if int(args.default_workers) > 0:
        n_default = max(1, min(int(args.default_workers), n_workers))
    else:
        n_default = n_workers

    print(
        f"[remote-worker] host={socket.gethostname()} "
        f"gpu={_gpu_name(leaf_gpu)!r} device={leaf_gpu} "
        f"workers_pool={n_workers} default_advertise={n_default} "
        f"leaf_servers={n_servers} "
        f"leaf_max_batch={args.leaf_max_batch} ckpt={ckpt.name} "
        f"digest={digest[:12]}… "
        f"rss_limit={args.tree_rss_limit_gb:.1f}GiB "
        f"min_free={args.min_free_ram_gb:.1f}GiB",
        flush=True,
    )

    # Install signal capture before the first child exists. The prior late
    # install left a 240-second leaf-startup window where SIGTERM used its
    # default action and bypassed every pool/leaf cleanup path.
    stop_event = threading.Event()
    received_signal: dict[str, Optional[int]] = {"value": None}

    def _capture_shutdown(signum: int, _frame: object) -> None:
        if received_signal["value"] is None:
            received_signal["value"] = int(signum)
        stop_event.set()

    previous_handlers = {
        sig: signal.signal(sig, _capture_shutdown)
        for sig in (signal.SIGINT, signal.SIGTERM)
    }

    def _restore_signal_handlers() -> None:
        for sig, previous in previous_handlers.items():
            signal.signal(sig, previous)

    def _wait_ready_or_stop(event: object, timeout_s: float) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        wait = getattr(event, "wait")
        while not stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            if wait(timeout=min(0.25, remaining)):
                return True
        return False

    mpctx = mp.get_context("spawn")
    leaf_resp_qs = [mpctx.Queue(maxsize=2) for _ in range(n_workers)]
    leaf_req_qs = []
    leaf_ctrl_qs = []
    leaf_status_qs = []
    leaf_alive_evts = []
    leaf_servers = []
    leaf_version = 0
    readies = []
    leaf_statuses: list[dict[str, Any]] = []

    startup_ok = True
    for j in range(n_servers):
        if stop_event.is_set():
            startup_ok = False
            break
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
        try:
            proc.start()
        except BaseException as exc:
            print(
                f"[remote-worker] ERROR: leaf[{j}] spawn failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            startup_ok = False
            break
        leaf_req_qs.append(rq)
        leaf_ctrl_qs.append(cq)
        leaf_status_qs.append(sq)
        leaf_alive_evts.append(alive)
        leaf_servers.append(proc)
        readies.append(ev)

    for j, ev in enumerate(readies):
        if not _wait_ready_or_stop(ev, 240):
            detail = (
                f"interrupted by signal {received_signal['value']}"
                if stop_event.is_set()
                else "not ready in 240s"
            )
            print(f"[remote-worker] ERROR: leaf[{j}] {detail}", file=sys.stderr)
            startup_ok = False
            break
        try:
            status = leaf_status_qs[j].get(timeout=5)
        except Exception as exc:
            print(f"[remote-worker] ERROR: leaf[{j}] status: {exc}", file=sys.stderr)
            startup_ok = False
            continue
        normalized, problems = _validated_leaf_status(
            j,
            status,
            expected_type="ready",
            expected_digest=digest,
            expected_version=leaf_version,
        )
        leaf_statuses.append(normalized)
        if problems:
            print(
                f"[remote-worker] ERROR: leaf[{j}] bad ack: "
                + "; ".join(problems),
                file=sys.stderr,
            )
            startup_ok = False
        else:
            print(
                f"[remote-worker] leaf[{j}] ready "
                f"digest={digest[:12]}… version={leaf_version}",
                flush=True,
            )

    if not startup_ok:
        survivors = _shutdown_leaf_servers(leaf_servers, leaf_ctrl_qs)
        _close_mp_queues(
            leaf_resp_qs,
            leaf_req_qs,
            leaf_ctrl_qs,
            leaf_status_qs,
        )
        if survivors:
            print(
                f"[remote-worker] ERROR: leaf shutdown survivors={survivors}",
                file=sys.stderr,
            )
        _restore_signal_handlers()
        signal_number = received_signal["value"]
        return 128 + int(signal_number) if signal_number is not None else 3

    # Manager dict is a true cross-process proxy: reload publishes digest/version
    # here and long-lived WorkerPool children re-read on every leaf RPC.
    manager = None
    pool = None
    try:
        if stop_event.is_set():
            raise InterruptedError("signal received before pool startup")
        manager = mpctx.Manager()
        leaf_expect = manager.dict()
        leaf_expect["digest"] = digest
        leaf_expect["version"] = int(leaf_version)
        leaf_expect["pinned"] = []

        remote_channel = {
            "req_qs": leaf_req_qs,
            "resp_qs": leaf_resp_qs,
            "alive_evts": leaf_alive_evts,
            "ctrl_qs": leaf_ctrl_qs,
            # Shared proxy so reload updates are visible to long-lived pool workers
            # without respawning (plain ints were snapshotted at worker init).
            "expected_digest": leaf_expect,
            "expected_version": leaf_expect,
            "timeout_s": float(config.SEARCH.remote_request_timeout_s),
            "generation": 0,
        }

        _load_rr_module()
        pool = WorkerPool(
            num_workers=n_workers,
            cg_lib_path=args.cg_lib_path or None,
            remote_channel=remote_channel,
        )
        pool_start_finished = threading.Event()

        def _interrupt_pool_startup() -> None:
            stop_event.wait()
            while not pool_start_finished.wait(timeout=0.05):
                if getattr(pool, "_pool", None) is not None:
                    pool.request_stop(
                        f"received signal {received_signal['value']} during startup"
                    )
                    return

        startup_interrupt = threading.Thread(
            target=_interrupt_pool_startup,
            name="remote-startup-signal-watcher",
            daemon=True,
        )
        startup_interrupt.start()
        try:
            pool.__enter__()
        finally:
            pool_start_finished.set()
    except BaseException:
        _shutdown_leaf_servers(leaf_servers, leaf_ctrl_qs)
        _close_mp_queues(
            leaf_resp_qs,
            leaf_req_qs,
            leaf_ctrl_qs,
            leaf_status_qs,
        )
        if manager is not None:
            try:
                manager.shutdown()
            except Exception:
                pass
        _restore_signal_handlers()
        signal_number = received_signal["value"]
        if signal_number is not None:
            return 128 + int(signal_number)
        raise
    assert manager is not None
    assert pool is not None
    # Serialize checkpoint-control operations across concurrent TCP handlers.
    ctrl_lock = threading.RLock()
    ctrl_cond = threading.Condition(ctrl_lock)
    # The current leaf server routes a single resident model. Keep this map for
    # wire compatibility, but reject secondary pins immediately instead of
    # sending unsupported control messages that time out after two minutes.
    pins: dict[str, str] = {}
    state: dict[str, Any] = {
        "digest": digest,
        "version": leaf_version,
        "checkpoint": str(ckpt),
        "healthy": True,
        "accepting_jobs": True,
        "active_jobs": 0,
        "terminal_reload_failure": False,
        "controller_error": None,
        "jobs_completed": 0,
        "jobs_failed": 0,
        "started_at": time.time(),
        "tree_rss_gb": None,
        "free_ram_gb": _free_ram_gb(),
        "live_worker_pids": list(pool.ready_worker_pids),
        "shutdown_reason": None,
        "shutdown_exit_code": 0,
        "leaf_expect": leaf_expect,
        "pins": pins,
    }
    reload_drain_timeout_s = max(
        1.0,
        float(os.environ.get("POKEBOT_REMOTE_RELOAD_DRAIN_TIMEOUT_S", "240")),
    )

    def _publish_expect(*, primary: str, version: int) -> None:
        """Publish the live primary digest/version to long-lived workers."""
        leaf_expect["digest"] = primary
        leaf_expect["version"] = int(version)
        leaf_expect["pinned"] = []

    def _drain_status_queues() -> None:
        """Drop any stale leaf status frames (best-effort, non-blocking)."""
        for sq in leaf_status_qs:
            while True:
                try:
                    sq.get_nowait()
                except Exception:
                    break

    def _terminalize_reload_failure(reason: str) -> None:
        """Stop serving after an ambiguous/partial leaf control attempt.

        Without an acknowledgement correlation token, issuing another reload
        after a timeout could consume a late matching frame from the abandoned
        attempt. Require a clean worker restart instead.
        """
        state["terminal_reload_failure"] = True
        state["accepting_jobs"] = False
        state["healthy"] = False
        state["controller_error"] = reason
        for cq in leaf_ctrl_qs:
            try:
                cq.put({"cmd": "stop"}, timeout=0.1)
            except Exception:
                pass

    def _await_leaf_statuses(
        expected_type: str,
        *,
        expected_digest: str,
        expected_version: int,
        sent: Optional[list[bool]] = None,
        timeout_s: float = 120.0,
    ) -> tuple[bool, Optional[str], list[dict[str, Any]]]:
        """Collect and verify one exact identity acknowledgement per leaf."""
        statuses: list[dict[str, Any]] = []
        problems: list[str] = []
        stale_frames: list[str] = []
        deadline = time.monotonic() + float(timeout_s)
        for j, sq in enumerate(leaf_status_qs):
            if sent is not None and (j >= len(sent) or not sent[j]):
                status: object = {
                    "type": expected_type,
                    "ok": False,
                    "version": None,
                    "checkpoint_digest": None,
                    "error": "control command was not queued",
                }
            else:
                while True:
                    try:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError("shared status deadline expired")
                        candidate = sq.get(timeout=remaining)
                    except Exception as exc:
                        status = {
                            "type": expected_type,
                            "ok": False,
                            "version": None,
                            "checkpoint_digest": None,
                            "error": f"status timeout: {exc}",
                        }
                        break
                    if _status_is_current_attempt(
                        candidate,
                        expected_type=expected_type,
                        expected_version=expected_version,
                    ):
                        status = candidate
                        break
                    stale_frames.append(f"leaf[{j}] ignored stale status {candidate!r}")
            normalized, leaf_problems = _validated_leaf_status(
                j,
                status,
                expected_type=expected_type,
                expected_digest=expected_digest,
                expected_version=expected_version,
            )
            statuses.append(normalized)
            problems.extend(leaf_problems)
        if problems:
            return False, "; ".join(stale_frames + problems), statuses
        return True, None, statuses

    def _current_leaf_health() -> dict[str, Any]:
        return _leaf_health_report(
            leaf_statuses,
            process_alive=[proc.is_alive() for proc in leaf_servers],
            event_alive=[evt.is_set() for evt in leaf_alive_evts],
            expected_digest=str(state["digest"]),
            expected_version=int(state["version"]),
            controller_healthy=bool(state["healthy"]),
            controller_error=(
                str(state["controller_error"])
                if state.get("controller_error")
                else None
            ),
        )

    def _advertised_identity() -> tuple[Optional[str], Optional[int]]:
        health = _current_leaf_health()
        if not health["ok"]:
            return None, None
        return str(state["digest"]), int(state["version"])

    def _hello_unlocked() -> dict[str, Any]:
        # Legacy trainers read ``workers`` as in-flight slots — keep that at the
        # *default* so a live collect does not jump to the pool ceiling.
        # New schedulers use ``max_workers`` (pool capacity) + demand caps.
        health = _current_leaf_health()
        advertised_digest, advertised_version = _advertised_identity()
        return {
            "hostname": socket.gethostname(),
            "workers": n_default,
            "default_workers": n_default,
            "max_workers": n_workers,
            "leaf_servers": n_servers,
            "gpu_name": _gpu_name(leaf_gpu),
            "device": leaf_gpu,
            "checkpoint_digest": advertised_digest,
            "checkpoint_version": advertised_version,
            "pinned_digests": [advertised_digest] if advertised_digest else [],
            "controller_healthy": health["ok"],
            "leaf_alive": health["leaf_alive"],
            "leaf_identity_ok": health["leaf_identity_ok"],
            "leaves": health["leaves"],
            "controller_error": health["controller_error"],
            "accepting_jobs": bool(state["accepting_jobs"]),
            "active_jobs": int(state["active_jobs"]),
            "terminal_reload_failure": bool(state["terminal_reload_failure"]),
            "free_ram_gb": state.get("free_ram_gb"),
            "tree_rss_gb": state.get("tree_rss_gb"),
            "tree_rss_limit_gb": float(args.tree_rss_limit_gb),
            "live_worker_pids": list(state.get("live_worker_pids") or []),
            "worker_capacity_healthy": pool.worker_capacity_healthy,
            "shutdown_reason": state.get("shutdown_reason"),
            "jobs_completed": state["jobs_completed"],
            "jobs_failed": state["jobs_failed"],
            "uptime_s": time.time() - state["started_at"],
            "job_kinds": list(REMOTE_JOB_KINDS),
        }

    def hello() -> dict[str, Any]:
        # A handshake is also an identity advertisement. Serialize it with
        # reload so it cannot combine pre-reload health with post-reload state.
        with ctrl_lock:
            return _hello_unlocked()

    def handler(msg: dict[str, Any]) -> dict[str, Any]:
        mtype = msg.get("type")
        if mtype == "health":
            with ctrl_lock:
                health = _current_leaf_health()
                return {"type": "health_ok", **hello(), **health}
        if mtype == "reload":
            with ctrl_lock:
                if state["terminal_reload_failure"]:
                    return {
                        "type": "reload_ok",
                        "ok": False,
                        "error": (
                            "worker restart required after an ambiguous or partial "
                            f"leaf reload: {state.get('controller_error')}"
                        ),
                        "checkpoint_digest": None,
                        "version": None,
                    }
                path = str(msg["path"])
                requested_digest = msg.get("digest")
                new_version = int(msg.get("version", state["version"] + 1))
                try:
                    actual = checkpoint_digest(path)
                except BaseException as exc:  # noqa: BLE001
                    return {
                        "type": "reload_ok",
                        "ok": False,
                        "error": f"checkpoint preflight failed: {type(exc).__name__}: {exc}",
                    }
                if requested_digest is not None and actual != requested_digest:
                    return {
                        "type": "reload_ok",
                        "ok": False,
                        "error": (
                            f"digest mismatch: expected {requested_digest}, "
                            f"got {actual}"
                        ),
                    }
                # Close admission before draining. Existing games retain the
                # old resident identity until they finish; reload never changes
                # leaf weights underneath an in-flight game.
                prior_healthy = bool(state["healthy"])
                prior_accepting = bool(state["accepting_jobs"])
                prior_error = state.get("controller_error")
                state["accepting_jobs"] = False
                state["healthy"] = False
                state["controller_error"] = (
                    f"reload draining active_jobs={state['active_jobs']} "
                    f"digest={actual} version={new_version}"
                )
                drain_deadline = time.monotonic() + reload_drain_timeout_s
                while int(state["active_jobs"]) > 0:
                    remaining = drain_deadline - time.monotonic()
                    if remaining <= 0:
                        active = int(state["active_jobs"])
                        state["controller_error"] = prior_error
                        state["healthy"] = prior_healthy
                        state["accepting_jobs"] = prior_accepting
                        return {
                            "type": "reload_ok",
                            "ok": False,
                            "error": (
                                f"reload drain timed out with {active} active job(s); "
                                "no leaf control command was sent"
                            ),
                            "checkpoint_digest": state["digest"],
                            "version": state["version"],
                        }
                    ctrl_cond.wait(timeout=remaining)
                state["controller_error"] = (
                    f"reload in progress digest={actual} version={new_version}"
                )
                _drain_status_queues()
                sent: list[bool] = []
                queue_errors: list[str] = []
                for j, cq in enumerate(leaf_ctrl_qs):
                    try:
                        cq.put(
                            {
                                "cmd": "reload",
                                "path": path,
                                "digest": actual,
                                "version": new_version,
                            },
                            timeout=5,
                        )
                        sent.append(True)
                    except Exception as exc:
                        sent.append(False)
                        queue_errors.append(
                            f"leaf[{j}] reload command failed: {type(exc).__name__}: {exc}"
                        )
                ok, err, statuses = _await_leaf_statuses(
                    "reload",
                    expected_digest=actual,
                    expected_version=new_version,
                    sent=sent,
                )
                leaf_statuses[:] = statuses
                if queue_errors:
                    ok = False
                    err = "; ".join(queue_errors + ([err] if err else []))
                if not ok:
                    # A partial reload is not recoverable by pretending the old
                    # parent identity still represents every resident leaf.
                    # Stop leaves and require a clean process restart; another
                    # uncorrelated control attempt could consume a late ack.
                    _terminalize_reload_failure(
                        err or "leaf reload verification failed"
                    )
                    return {
                        "type": "reload_ok",
                        "ok": False,
                        "error": err,
                        "checkpoint_digest": None,
                        "version": None,
                        "leaf_alive": _current_leaf_health()["leaf_alive"],
                        "leaf_identity_ok": False,
                        "leaves": list(leaf_statuses),
                    }
                try:
                    _persist_active_checkpoint(path, actual)
                except (OSError, ValueError) as exc:
                    # Leaves have acknowledged the new identity, but a process
                    # rotation cannot safely reproduce it. Stop serving and
                    # force a clean restart from the last durable identity;
                    # never report reload success to the trainer.
                    persistence_error = (
                        "durable active-checkpoint publication failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    _terminalize_reload_failure(persistence_error)
                    return {
                        "type": "reload_ok",
                        "ok": False,
                        "error": persistence_error,
                        "checkpoint_digest": None,
                        "version": None,
                        "leaf_alive": _current_leaf_health()["leaf_alive"],
                        "leaf_identity_ok": False,
                        "leaves": list(leaf_statuses),
                    }
                pins.clear()
                state["digest"] = actual
                state["version"] = new_version
                state["checkpoint"] = path
                # Publish to long-lived workers BEFORE returning so the next job
                # cannot observe server version N while still expecting N-1.
                _publish_expect(primary=actual, version=new_version)
                state["controller_error"] = None
                state["accepting_jobs"] = True
                state["healthy"] = True
                return {
                    "type": "reload_ok",
                    "ok": True,
                    "checkpoint_digest": actual,
                    "version": new_version,
                    "pinned_digests": [actual],
                    "restored_pins": [],
                }
        if mtype == "pin":
            with ctrl_lock:
                health = _current_leaf_health()
                if not health["ok"]:
                    return {
                        "type": "pin_ok",
                        "ok": False,
                        "error": "worker is fail-closed: leaf identity is unhealthy",
                        "leaves": health["leaves"],
                    }
                path = str(msg["path"])
                requested_digest = msg.get("digest")
                actual = checkpoint_digest(path)
                if requested_digest is not None and actual != requested_digest:
                    return {
                        "type": "pin_ok",
                        "ok": False,
                        "error": (
                            f"digest mismatch: expected {requested_digest}, "
                            f"got {actual}"
                        ),
                    }
                if actual != state["digest"]:
                    return {
                        "type": "pin_ok",
                        "ok": False,
                        "error": (
                            "secondary checkpoint pinning is unsupported by the "
                            "single-resident leaf server; reload it as primary first"
                        ),
                    }
                return {
                    "type": "pin_ok",
                    "ok": True,
                    "checkpoint_digest": actual,
                    "pinned_digests": [state["digest"]],
                }
        if mtype == "unpin":
            with ctrl_lock:
                digest = str(msg["digest"])
                if digest == state["digest"]:
                    return {
                        "type": "unpin_ok",
                        "ok": False,
                        "error": "cannot unpin the active primary checkpoint",
                    }
                pins.pop(digest, None)
                return {"type": "unpin_ok", "ok": True, "digest": digest}
        if mtype == "job":
            kind = str(msg.get("kind") or "play")
            job = msg.get("job")
            if not isinstance(job, dict):
                return {"type": "result", "ok": False, "error": "job must be object"}
            # Must use package-level callables (not rr._worker_*), or spawn
            # Pool workers fail to unpickle ``train_round_robin_remote``.
            if kind == "play":
                worker_fn = remote_play_job
            elif kind == "self_play":
                worker_fn = remote_self_play_job
            elif kind == "promotion":
                worker_fn = remote_promotion_job
            else:
                return {
                    "type": "result",
                    "ok": False,
                    "error": f"unsupported job kind {kind!r}",
                }

            # Admission and the active-job increment are atomic with reload's
            # admission close. Once counted, this job keeps the old leaf
            # identity resident until its finally block releases the barrier.
            with ctrl_cond:
                health = _current_leaf_health()
                if not health["ok"] or not state["accepting_jobs"]:
                    state["jobs_failed"] += 1
                    return {
                        "type": "result",
                        "ok": False,
                        "error": (
                            "remote worker fail-closed: leaf checkpoint identity "
                            "is unhealthy or reload admission is closed "
                            f"({health.get('controller_error')})"
                        ),
                        "leaves": health["leaves"],
                    }
                active_digest = str(state["digest"])
                active_checkpoint = str(state["checkpoint"])
                if kind in ("play", "self_play"):
                    # A job may use the active leaf checkpoint plus a CPU-local
                    # opponent checkpoint. Different leaf primaries cannot
                    # share this worker concurrently.
                    req_digest = job.get("checkpoint_digest")
                    req_path = job.get("checkpoint")
                    if req_digest and req_path:
                        req_digest_s = str(req_digest)
                        if req_digest_s != active_digest:
                            return {
                                "type": "result",
                                "ok": False,
                                "error": (
                                    f"{kind} requested inactive checkpoint "
                                    f"{req_digest_s}; active={active_digest}; "
                                    "reload_checkpoint is required before dispatch"
                                ),
                            }
                        job = {
                            **job,
                            "checkpoint": active_checkpoint,
                            "device": "cpu",
                        }
                    else:
                        job = {
                            **job,
                            "checkpoint": active_checkpoint,
                            "checkpoint_digest": active_digest,
                            "device": "cpu",
                        }
                else:
                    job = {**job, "device": "cpu"}
                state["active_jobs"] += 1

            # One job per socket request. Many sockets call this concurrently;
            # use Pool.apply (thread-safe), never concurrent imap_unordered.
            succeeded = False
            try:
                result = pool.apply(worker_fn, job)
                succeeded = True
                response = {"type": "result", "ok": True, "result": result}
            except BaseException as exc:  # noqa: BLE001
                response = {
                    "type": "result",
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            finally:
                with ctrl_cond:
                    state["active_jobs"] = max(0, int(state["active_jobs"]) - 1)
                    if succeeded:
                        state["jobs_completed"] += 1
                    else:
                        state["jobs_failed"] += 1
                    ctrl_cond.notify_all()
            return response
        return {"type": "error", "error": f"unknown message type {mtype!r}"}

    def _mark_shutdown(reason: str, *, exit_code: int) -> None:
        """Close admission immediately; the main thread performs teardown."""
        if state.get("shutdown_reason") is None:
            state["shutdown_reason"] = str(reason)
            state["shutdown_exit_code"] = int(exit_code)
        state["accepting_jobs"] = False
        state["healthy"] = False
        state["controller_error"] = str(reason)
        stop_event.set()

    def _watch_resources() -> None:
        missing_pool_samples = 0
        interval = max(1.0, float(args.watchdog_interval_s))
        while not stop_event.wait(interval):
            # A multiprocessing child can be alive while repeatedly failing
            # its initializer or waiting for a response-slot lease. Only the
            # ready handshake proves usable capacity.
            ready_pids = pool.ready_worker_pids
            state["live_worker_pids"] = list(ready_pids)
            state["tree_rss_gb"] = _process_tree_rss_gb()
            state["free_ram_gb"] = _free_ram_gb()

            if not pool.worker_capacity_healthy:
                missing_pool_samples += 1
            else:
                missing_pool_samples = 0

            reason: Optional[str] = None
            shutdown_exit_code = _service_shutdown_exit_code(
                planned_rotation=False,
                planned_rotation_exit_code=int(args.planned_rotation_exit_code),
            )
            dead_leaves = [
                index
                for index, proc in enumerate(leaf_servers)
                if not proc.is_alive()
                or index >= len(leaf_alive_evts)
                or not leaf_alive_evts[index].is_set()
            ]
            if dead_leaves:
                reason = f"leaf process/event died indices={dead_leaves}"
            elif missing_pool_samples >= 3:
                reason = (
                    "sim worker capacity did not recover: "
                    f"ready={len(ready_pids)} expected={n_workers} "
                    f"init_attempts={pool.initializer_attempts} "
                    f"init_failures={pool.initializer_failures}"
                )

            rss_gb = state.get("tree_rss_gb")
            if (
                reason is None
                and float(args.tree_rss_limit_gb) > 0
                and rss_gb is not None
                and float(rss_gb) >= float(args.tree_rss_limit_gb)
            ):
                reason = (
                    f"process-tree RSS {float(rss_gb):.2f}GiB reached "
                    f"limit {float(args.tree_rss_limit_gb):.2f}GiB"
                )

            free_gb = state.get("free_ram_gb")
            if (
                reason is None
                and float(args.min_free_ram_gb) > 0
                and free_gb is not None
                and float(free_gb) <= float(args.min_free_ram_gb)
            ):
                reason = (
                    f"available RAM {float(free_gb):.2f}GiB reached floor "
                    f"{float(args.min_free_ram_gb):.2f}GiB"
                )

            if int(args.max_service_jobs) > 0 and int(
                state["jobs_completed"]
            ) >= int(args.max_service_jobs):
                # First stop new admissions. Let already-counted jobs drain so
                # their result frames are not cut off by routine rotation.
                state["accepting_jobs"] = False
                state["controller_error"] = (
                    f"service rotation draining after {state['jobs_completed']} jobs"
                )
                if int(state["active_jobs"]) == 0 and reason is None:
                    reason = str(state["controller_error"])
                    shutdown_exit_code = _service_shutdown_exit_code(
                        planned_rotation=True,
                        planned_rotation_exit_code=int(
                            args.planned_rotation_exit_code
                        ),
                    )

            if reason is not None:
                print(
                    f"[remote-worker] WATCHDOG fail-closed: {reason}",
                    file=sys.stderr,
                    flush=True,
                )
                _mark_shutdown(reason, exit_code=shutdown_exit_code)
                return

    watchdog = threading.Thread(
        target=_watch_resources,
        name="remote-resource-watchdog",
        daemon=True,
    )
    watchdog.start()

    def _request_shutdown(signum: int, _frame: object) -> None:
        _capture_shutdown(signum, _frame)
        _mark_shutdown(f"received signal {int(signum)}", exit_code=0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _request_shutdown)
    exit_code = 0
    survivors: tuple[int, ...] = ()
    try:
        if args.smoke:
            health = handler({"type": "health"})
            print(f"[remote-worker] smoke health={health}", flush=True)
            exit_code = 0 if bool(health.get("ok")) else 4
        else:
            print(
                f"[remote-worker] listening on {args.host}:{args.port} "
                f"(pool={n_workers} advertise_default={n_default}, "
                f"leaf on local GPU max_connections={args.max_connections})",
                flush=True,
            )
            serve_forever(
                handler,
                host=args.host,
                port=int(args.port),
                hello=hello,
                stop_event=stop_event,
                max_connections=max(1, int(args.max_connections)),
            )
    except KeyboardInterrupt:
        _request_shutdown(signal.SIGINT, None)
    finally:
        if state.get("shutdown_reason") is None:
            state["shutdown_reason"] = "remote worker main exiting"
        state["accepting_jobs"] = False
        state["healthy"] = False
        stop_event.set()
        watchdog.join(timeout=5.0)
        pool.request_stop(str(state["shutdown_reason"]))
        try:
            pool.__exit__(None, None, None)
        finally:
            survivors = _shutdown_leaf_servers(leaf_servers, leaf_ctrl_qs)
            _close_mp_queues(
                leaf_resp_qs,
                leaf_req_qs,
                leaf_ctrl_qs,
                leaf_status_qs,
            )
            try:
                manager.shutdown()
            except Exception:
                pass
            _restore_signal_handlers()
    if survivors:
        print(
            f"[remote-worker] ERROR: leaf shutdown survivors={survivors}",
            file=sys.stderr,
            flush=True,
        )
        return 5
    shutdown_exit_code = int(state.get("shutdown_exit_code", 0))
    if shutdown_exit_code != 0:
        return shutdown_exit_code
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
