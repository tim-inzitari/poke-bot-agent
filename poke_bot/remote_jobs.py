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
import json
import os
import queue
import shlex
import shutil
import socket
import struct
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

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
_BERT_HOSTS = frozenset({"192.168.1.157", "bert.local", "bert"})
_BERT_SSH = os.environ.get("POKEBOT_BERT_SSH", "tsinzitari@bert.local")


def expand_endpoint_specs(specs: list[str]) -> list[str]:
    """Accept space-separated nargs or a single comma-separated token."""
    out: list[str] = []
    for spec in specs:
        for part in str(spec).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


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
    local_tmp = Path(f"/tmp/pokebot_remote_stage_{os.getpid()}_{src.name}")
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


def _rsync_to_bert(src: Path, remote_native: Path) -> None:
    """Stage ``src`` to bert via BatchMode SSH + rsync (preferred for large .pt)."""
    remote_dir = remote_native.parent.as_posix()
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
            f"{_BERT_SSH}:{remote_native.as_posix()}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if rsync.returncode != 0:
        raise RemoteJobsError(
            f"bert rsync failed ({src.name} -> {_BERT_SSH}:{remote_native}): "
            f"{rsync.stderr.strip() or rsync.stdout.strip() or rsync.returncode}"
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
    remote_native = _BERT_ROOT / rel
    sftp_root = _bert_sftp_root()
    if sftp_root is not None:
        dest = sftp_root / rel
        if not _needs_stage(src, dest):
            return str(remote_native)
    # Prefer BatchMode SSH+rsync for large digest .pt; gvfs SFTP as fallback.
    try:
        _rsync_to_bert(src, remote_native)
        return str(remote_native)
    except RemoteJobsError as rsync_exc:
        if sftp_root is None:
            raise
        dest = sftp_root / rel
        try:
            _gvfs_safe_copy(src, dest)
        except OSError as copy_exc:
            raise RemoteJobsError(
                f"bert stage failed via rsync ({rsync_exc}) and gvfs SFTP "
                f"({copy_exc})"
            ) from copy_exc
        return str(remote_native)


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
        dest_name = digest_addressed_basename(src)
        dest = smb / dest_name
        if _needs_stage(src, dest):
            _gvfs_safe_copy(src, dest)
        return f"/workspace/checkpoint/{dest_name}"
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
    ckpt = out.get("checkpoint")
    if ckpt:
        host_l = host.strip().lower()
        # Elmo only bind-mounts ``/workspace/checkpoint/`` (not the full repo).
        # Play jobs may auto-pin a peer digest from ``job["checkpoint"]``, so the
        # path must be the staged SMB basename — not ``/workspace/outputs/...``.
        # Bert keeps a full checkout; workdir remap is enough there.
        if host_l in _ELMO_HOSTS:
            out["checkpoint"] = resolve_remote_checkpoint_path(host, str(ckpt))
        elif host_l in _BERT_HOSTS:
            out["checkpoint"] = resolve_remote_workdir_path(host, str(ckpt))
    spec = out.get("spec")
    if isinstance(spec, dict) and spec.get("path"):
        spec["path"] = resolve_remote_workdir_path(host, str(spec["path"]))
    return out


class RemoteJobsError(RuntimeError):
    """Protocol or transport failure talking to a remote worker."""


def encode_frame(payload: dict[str, Any]) -> bytes:
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
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
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
        self.info = RemoteWorkerInfo(
            endpoint=self.endpoint,
            workers=int(reply.get("workers", 0)),
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
            sock = self._require_sock()
            prev = sock.gettimeout()
            sock.settimeout(
                max(
                    self.timeout_s,
                    float(remote_job.get("game_timeout_s") or 900) + job_buffer_s,
                )
            )
            try:
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
            return payload
        assert last_exc is not None
        raise last_exc

    def _control_call(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Send a control-plane frame with ``control_timeout_s``; one hangup retry."""
        last_exc: Optional[BaseException] = None
        for attempt in range(2):
            sock = self._require_sock()
            prev = sock.gettimeout()
            sock.settimeout(self.control_timeout_s)
            try:
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
) -> tuple[list[RemoteJobClient], list[RemoteJobClient]]:
    """Expand farm endpoints into ``info.workers`` concurrent sockets each.

    ``serve_forever`` handles one job per socket; concurrent sockets are the
    unit of remote in-flight capacity. Returns ``(all_slots, owned_clones)``
    where ``owned_clones`` must be closed by the caller (slot 0 per endpoint
    reuses the farm's existing session and is not closed here).
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
        n = max(1, int(template.info.workers) if template.info is not None else 1)
        n = max(1, n // divisor)
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
) -> Iterator[dict[str, Any]]:
    """Run local WorkerPool and optional remote TCP workers in parallel.

    Yields results in completion order (unordered), matching ``imap_unordered``.
    When ``remote_clients`` is empty, delegates to the local pool only.

    Remote concurrency matches advertised worker counts: each endpoint opens
    ``info.workers`` sockets (one job in flight per socket), so additive
    capacity is real in-flight games — not one serial pipeline per host.
    """
    if not remote_clients or int(remote_workers) <= 0:
        yield from local_pool.imap_unordered(local_fn, jobs)
        return

    local_jobs, remote_jobs = split_jobs_additive(
        jobs,
        local_workers=local_workers,
        remote_workers=remote_workers,
    )
    out_q: queue.Queue[tuple[str, Any]] = queue.Queue()
    errors: list[BaseException] = []
    owned_clients: list[RemoteJobClient] = []

    def _emit_local() -> None:
        try:
            if local_jobs:
                for row in local_pool.imap_unordered(local_fn, local_jobs):
                    out_q.put(("ok", row))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            out_q.put(("err", exc))
        finally:
            out_q.put(("done", None))

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
                try:
                    if no_local_fallback:
                        out_q.put(("ok", _submit_remote_with_retry(client, job)))
                    else:
                        out_q.put(("ok", client.submit_job(job, kind=kind)))
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
                        out_q.put(("ok", local_pool.apply(local_fn, fallback_job)))
                    return
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            out_q.put(("err", exc))
        finally:
            out_q.put(("done", None))

    try:
        threads: list[threading.Thread] = []
        remote_slots: list[RemoteJobClient] = []
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
            for slot_index, client in enumerate(remote_slots):
                batch = [
                    job
                    for job_index, job in enumerate(remote_jobs)
                    if job_index % n_slots == slot_index
                ]
                if not batch:
                    continue
                threads.append(
                    threading.Thread(
                        target=_emit_remote,
                        args=(client, batch),
                        name=f"additive-remote-{client.endpoint}-{slot_index}",
                        daemon=True,
                    )
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
                raise payload
            yield payload

        for thread in threads:
            thread.join(timeout=1.0)
        if errors:
            raise errors[0]
    finally:
        for client in owned_clients:
            try:
                client.close()
            except Exception:
                pass


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
        return sum(
            int(client.info.workers)
            for client in self.clients
            if client.info is not None
        )

    def connect(self) -> list[RemoteWorkerInfo]:
        self.close()
        infos: list[RemoteWorkerInfo] = []
        for endpoint in self.endpoints:
            client = RemoteJobClient(
                *parse_endpoint(endpoint),
                timeout_s=self.timeout_s,
                connect_timeout_s=self.connect_timeout_s,
                control_timeout_s=self.control_timeout_s,
            )
            infos.append(client.connect())
            self.clients.append(client)
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
) -> None:
    """Accept connections; each connection is handled sequentially (one job at a
    time per socket). Concurrent sockets = concurrent in-flight games, bounded
    by the caller's WorkerPool behind ``handler``.

    ``idle_timeout_s`` bounds each ``recv`` so a wedged peer cannot hang the
    handler thread forever. Idle timeouts are retried (not treated as hangup)
    so long-lived farm sockets survive train/promo gaps between waves.
    """

    def _handle_conn(conn: socket.socket, addr: tuple[str, int]) -> None:
        # Bounded recv so a wedged peer cannot hang the handler thread forever,
        # but idle farm sessions (train/promo between collection waves) must
        # NOT be torn down — treat read timeouts as keep-waiting, not hangup.
        conn.settimeout(float(idle_timeout_s))
        try:
            first = read_frame(conn)
            if first.get("type") != "hello":
                send_frame(
                    conn,
                    {
                        "type": "error",
                        "error": "expected hello",
                    },
                )
                return
            if int(first.get("proto", -1)) != PROTO_VERSION:
                send_frame(
                    conn,
                    {
                        "type": "error",
                        "error": f"unsupported proto {first.get('proto')}",
                    },
                )
                return
            info = hello() if hello is not None else {}
            send_frame(
                conn,
                {
                    "type": "hello_ok",
                    "proto": PROTO_VERSION,
                    **info,
                },
            )
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
                except RemoteJobsError:
                    break
                mtype = msg.get("type")
                if mtype == "bye":
                    break
                if mtype == "ping":
                    send_frame(
                        conn,
                        {
                            "type": "pong",
                            "t0": msg.get("t0"),
                            "t1": time.time(),
                        },
                    )
                    continue
                try:
                    reply = handler(msg)
                except BaseException as exc:  # noqa: BLE001 - keep connection alive
                    send_frame(
                        conn,
                        {
                            "type": "result" if mtype == "job" else "error",
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                    continue
                send_frame(conn, reply)
        finally:
            try:
                conn.close()
            except Exception:
                pass

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
            thread = threading.Thread(
                target=_handle_conn,
                args=(conn, addr),
                name=f"remote-job-{addr[0]}:{addr[1]}",
                daemon=True,
            )
            thread.start()
    finally:
        server.close()
