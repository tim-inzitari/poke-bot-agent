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

import json
import os
import queue
import shutil
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

PROTO_VERSION = 1
DEFAULT_PORT = 8765
_HDR = struct.Struct("!I")
_MAX_FRAME = 256 * 1024 * 1024  # 256 MiB — self-play records can be large

# TrueNAS docker worker only mounts ``./checkpoint`` → ``/workspace/checkpoint``.
# Bert native checkout mirrors under ``/Users/tsinzitari/workspace/poke-bot-agent``.
_TRAIN_ROOT = Path("/home/inzi/poke-bot-agent")
_BERT_ROOT = Path("/Users/tsinzitari/workspace/poke-bot-agent")
_ELMO_HOSTS = frozenset({"192.168.1.143", "truenas.local", "truenas"})
_BERT_HOSTS = frozenset({"192.168.1.157", "bert.local", "bert"})


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


def resolve_remote_checkpoint_path(host: str, local_path: str) -> str:
    """Map a trainer-local .pt path to a path the remote process can open.

    Elmo (TrueNAS host Docker): stage bytes onto the SMB ``checkpoint/``
    bind-mount and reload ``/workspace/checkpoint/<basename>``.
    Bert: remap the training-box repo root onto bert's native checkout.
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
        dest = smb / src.name
        if (
            not dest.is_file()
            or dest.stat().st_size != src.stat().st_size
            or int(dest.stat().st_mtime) < int(src.stat().st_mtime)
        ):
            tmp = dest.with_suffix(dest.suffix + f".staging.{os.getpid()}")
            shutil.copy2(src, tmp)
            os.replace(tmp, dest)
        return f"/workspace/checkpoint/{src.name}"
    if host_l in _BERT_HOSTS:
        try:
            rel = src.relative_to(_TRAIN_ROOT)
        except ValueError as exc:
            raise RemoteJobsError(
                f"bert path remap requires path under {_TRAIN_ROOT}, got {src}"
            ) from exc
        return str(_BERT_ROOT / rel)
    return str(src)


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


class RemoteJobClient:
    """One TCP session to a remote worker that already has leaf+sim ready."""

    def __init__(
        self,
        host: str,
        port: int = DEFAULT_PORT,
        *,
        timeout_s: float = 30.0,
        connect_timeout_s: float = 10.0,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.connect_timeout_s = float(connect_timeout_s)
        self._sock: Optional[socket.socket] = None
        self.info: Optional[RemoteWorkerInfo] = None

    @property
    def endpoint(self) -> str:
        return f"{self.host}:{self.port}"

    def connect(self) -> RemoteWorkerInfo:
        sock = socket.create_connection(
            (self.host, self.port), timeout=self.connect_timeout_s
        )
        sock.settimeout(self.timeout_s)
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
        send_frame(sock, {"type": "ping", "t0": time.time()})
        reply = read_frame(sock)
        if reply.get("type") != "pong":
            raise RemoteJobsError(f"unexpected ping reply: {reply!r}")
        return reply

    def health(self) -> dict[str, Any]:
        sock = self._require_sock()
        send_frame(sock, {"type": "health"})
        reply = read_frame(sock)
        if reply.get("type") != "health_ok":
            raise RemoteJobsError(f"unexpected health reply: {reply!r}")
        return reply

    def submit_job(self, job: dict[str, Any], *, kind: str = "play") -> dict[str, Any]:
        """Submit one game job; blocks until the remote result frame arrives."""
        sock = self._require_sock()
        # Game wall times routinely exceed 12–15 minutes on belief-MCTS; bump
        # the socket timeout for the duration of the call.
        prev = sock.gettimeout()
        sock.settimeout(max(self.timeout_s, float(job.get("game_timeout_s") or 900) + 120.0))
        try:
            send_frame(sock, {"type": "job", "kind": kind, "job": job})
            reply = read_frame(sock)
        finally:
            sock.settimeout(prev)
        if reply.get("type") != "result":
            raise RemoteJobsError(f"unexpected job reply: {reply!r}")
        if not reply.get("ok", False):
            raise RemoteJobsError(str(reply.get("error") or "remote job failed"))
        payload = reply.get("result")
        if not isinstance(payload, dict):
            raise RemoteJobsError("remote result missing body")
        return payload

    def reload_checkpoint(
        self,
        path: str,
        *,
        digest: Optional[str] = None,
        version: Optional[int] = None,
    ) -> dict[str, Any]:
        sock = self._require_sock()
        remote_path = resolve_remote_checkpoint_path(self.host, path)
        msg: dict[str, Any] = {"type": "reload", "path": remote_path}
        if digest is not None:
            msg["digest"] = digest
        if version is not None:
            msg["version"] = int(version)
        send_frame(sock, msg)
        reply = read_frame(sock)
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
        sock = self._require_sock()
        remote_path = resolve_remote_checkpoint_path(self.host, path)
        msg: dict[str, Any] = {"type": "pin", "path": remote_path}
        if digest is not None:
            msg["digest"] = digest
        send_frame(sock, msg)
        reply = read_frame(sock)
        if reply.get("type") != "pin_ok" or not reply.get("ok", False):
            raise RemoteJobsError(
                f"pin failed host={self.host} remote_path={remote_path}: {reply!r}"
            )
        return reply

    def unpin_checkpoint(self, digest: str) -> dict[str, Any]:
        sock = self._require_sock()
        send_frame(sock, {"type": "unpin", "digest": digest})
        reply = read_frame(sock)
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
    """Capacity-weighted split: local IPC stays primary, remote is extra."""
    if not jobs or int(remote_workers) <= 0:
        return list(jobs), []
    local_slots = max(1, int(local_workers))
    remote_slots = max(1, int(remote_workers))
    total = local_slots + remote_slots
    local_jobs: list[dict[str, Any]] = []
    remote_jobs: list[dict[str, Any]] = []
    for index, job in enumerate(jobs):
        if (index % total) < local_slots:
            local_jobs.append(job)
        else:
            remote_jobs.append(job)
    return local_jobs, remote_jobs


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

    def _emit_remote(client: RemoteJobClient, batch: list[dict[str, Any]]) -> None:
        try:
            for job in batch:
                out_q.put(("ok", client.submit_job(job, kind=kind)))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            out_q.put(("err", exc))
        finally:
            out_q.put(("done", None))

    threads: list[threading.Thread] = []
    if local_jobs:
        threads.append(
            threading.Thread(target=_emit_local, name="additive-local", daemon=True)
        )
    if remote_jobs:
        n_clients = len(remote_clients)
        for client_index, client in enumerate(remote_clients):
            batch = [
                job
                for job_index, job in enumerate(remote_jobs)
                if job_index % n_clients == client_index
            ]
            if not batch:
                continue
            threads.append(
                threading.Thread(
                    target=_emit_remote,
                    args=(client, batch),
                    name=f"additive-remote-{client.endpoint}",
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


@dataclass
class RemoteWorkerFarm:
    """Connected LAN workers used as additive whole-game capacity."""

    endpoints: list[str]
    timeout_s: float = 30.0
    connect_timeout_s: float = 10.0
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

    def reload_all(
        self,
        path: str,
        *,
        digest: Optional[str] = None,
        version: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        replies: list[dict[str, Any]] = []
        survivors: list[RemoteJobClient] = []
        errors: list[str] = []
        for client in self.clients:
            try:
                replies.append(
                    client.reload_checkpoint(path, digest=digest, version=version)
                )
                survivors.append(client)
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
        if errors:
            print(
                "[remote-farm] WARN dropped endpoint(s) after reload failure: "
                + "; ".join(errors),
                flush=True,
            )
        return replies

    def pin_all(
        self,
        path: str,
        *,
        digest: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        replies: list[dict[str, Any]] = []
        survivors: list[RemoteJobClient] = []
        errors: list[str] = []
        for client in self.clients:
            try:
                replies.append(client.pin_checkpoint(path, digest=digest))
                survivors.append(client)
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
        if errors:
            print(
                "[remote-farm] WARN dropped endpoint(s) after pin failure: "
                + "; ".join(errors),
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
) -> None:
    """Accept connections; each connection is handled sequentially (one job at a
    time per socket). Concurrent sockets = concurrent in-flight games, bounded
    by the caller's WorkerPool behind ``handler``.
    """

    def _handle_conn(conn: socket.socket, addr: tuple[str, int]) -> None:
        conn.settimeout(60.0)
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
