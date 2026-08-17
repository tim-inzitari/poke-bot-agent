"""Manifest-driven stable endpoint for an elastic remote worker fleet.

The trainer speaks the existing :mod:`poke_bot.remote_jobs` protocol to this
gateway.  The gateway keeps the trainer-facing endpoint stable while worker
endpoints are added or removed through an atomically replaced JSON manifest.

This module deliberately does not provision hosts, open firewalls, or alter a
trainer service.  A backend is admitted only after its hello identity and
storage-local checkpoint digest pass the manifest gates.  An optional,
    explicit local SSM-backed SSH staging descriptor can copy an immutable checkpoint to a
backend before the gateway performs its normal verify-before-reload fanout.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from poke_bot.remote_jobs import (
    RemoteJobClient,
    RemoteJobsError,
    parse_endpoint,
    read_frame,
    serve_forever,
)


MANIFEST_SCHEMA = "poke_bot.remote_fleet_gateway/v1"
_TOP_LEVEL_FIELDS = {
    "schema",
    "gateway_id",
    "activation_allowed",
    "bind_host",
    "bind_port",
    "default_workers",
    "fleet_worker_ceiling",
    "registry_poll_seconds",
    "backends",
}
_BACKEND_FIELDS = {
    "id",
    "endpoint",
    "enabled",
    "capacity",
    "expected_hostname",
    "required_checkpoint_digest",
    "checkpoint_paths",
    "checkpoint_path_template",
    "required_job_kinds",
    "required_capabilities",
    "path_rewrites",
    "checkpoint_stage",
}
_CHECKPOINT_STAGE_FIELDS = {
    "mode",
    "ssh_host",
    "ssh_port",
    "ssh_user",
    "identity_file",
    "known_hosts_file",
}
_CHECKPOINT_FIELDS = {
    "checkpoint": "checkpoint_digest",
    "opponent_checkpoint": "opponent_checkpoint_digest",
    "candidate_checkpoint": "candidate_checkpoint_digest",
    "parent_checkpoint": "parent_checkpoint_digest",
}


class FleetManifestError(ValueError):
    """The registry is malformed or cannot be admitted safely."""


@dataclass(frozen=True)
class CheckpointStageConfig:
    """Create-only checkpoint transport through a local SSM SSH forward."""

    mode: str
    ssh_host: str
    ssh_port: int
    ssh_user: str
    identity_file: str
    known_hosts_file: str

    @classmethod
    def from_dict(cls, value: Any, where: str) -> Optional["CheckpointStageConfig"]:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise FleetManifestError(f"{where} must be an object")
        _strict_fields(value, _CHECKPOINT_STAGE_FIELDS, where)
        mode = str(value.get("mode") or "").strip()
        if mode != "ssm_ssh_v1":
            raise FleetManifestError(f"{where}.mode must be 'ssm_ssh_v1'")
        ssh_host = str(value.get("ssh_host") or "127.0.0.1").strip()
        if ssh_host not in {"127.0.0.1", "::1"}:
            raise FleetManifestError(f"{where}.ssh_host must be loopback")
        ssh_port = int(value.get("ssh_port", 0))
        if not 1024 <= ssh_port <= 65535:
            raise FleetManifestError(f"{where}.ssh_port must be 1024..65535")
        ssh_user = str(value.get("ssh_user") or "ec2-user").strip()
        if not ssh_user or not all(ch.isalnum() or ch in "-_" for ch in ssh_user):
            raise FleetManifestError(f"{where}.ssh_user is invalid")
        identity_file = str(value.get("identity_file") or "").strip()
        if not identity_file.startswith("/"):
            raise FleetManifestError(f"{where}.identity_file must be absolute")
        known_hosts_file = str(value.get("known_hosts_file") or "").strip()
        if not known_hosts_file.startswith("/"):
            raise FleetManifestError(f"{where}.known_hosts_file must be absolute")
        return cls(
            mode=mode,
            ssh_host=ssh_host,
            ssh_port=ssh_port,
            ssh_user=ssh_user,
            identity_file=identity_file,
            known_hosts_file=known_hosts_file,
        )


def _strict_fields(value: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise FleetManifestError(f"{where} has unknown fields: {', '.join(unknown)}")


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _required_digest(value: Any, where: str) -> str:
    digest = str(value or "")
    if not digest.startswith("sha256:") or len(digest) != 71:
        raise FleetManifestError(f"{where} must be a full sha256 digest")
    try:
        int(digest[7:], 16)
    except ValueError as exc:
        raise FleetManifestError(f"{where} must be a full sha256 digest") from exc
    return digest


@dataclass(frozen=True)
class BackendConfig:
    id: str
    endpoint: str
    enabled: bool
    capacity: int
    expected_hostname: Optional[str]
    required_checkpoint_digest: str
    checkpoint_paths: dict[str, str]
    checkpoint_path_template: Optional[str]
    required_job_kinds: tuple[str, ...]
    required_capabilities: tuple[str, ...]
    path_rewrites: tuple[tuple[str, str], ...]
    checkpoint_stage: Optional[CheckpointStageConfig]

    @classmethod
    def from_dict(cls, value: Any, index: int) -> "BackendConfig":
        where = f"backends[{index}]"
        if not isinstance(value, dict):
            raise FleetManifestError(f"{where} must be an object")
        _strict_fields(value, _BACKEND_FIELDS, where)
        backend_id = str(value.get("id") or "").strip()
        if not backend_id or not all(ch.isalnum() or ch in "-_" for ch in backend_id):
            raise FleetManifestError(f"{where}.id must use letters, digits, '-' or '_'")
        endpoint = str(value.get("endpoint") or "").strip()
        try:
            endpoint_host, endpoint_port = parse_endpoint(endpoint)
            if not endpoint_host or not 1 <= endpoint_port <= 65535:
                raise ValueError("host or port outside allowed range")
        except Exception as exc:
            raise FleetManifestError(f"{where}.endpoint is invalid: {endpoint!r}") from exc
        if not isinstance(value.get("enabled", False), bool):
            raise FleetManifestError(f"{where}.enabled must be a boolean")
        enabled = value.get("enabled", False)
        capacity = int(value.get("capacity", 0))
        if capacity < 1 or capacity > 4096:
            raise FleetManifestError(f"{where}.capacity must be between 1 and 4096")
        digest = _required_digest(
            value.get("required_checkpoint_digest"),
            f"{where}.required_checkpoint_digest",
        )
        raw_paths = value.get("checkpoint_paths") or {}
        if not isinstance(raw_paths, dict):
            raise FleetManifestError(f"{where}.checkpoint_paths must be an object")
        paths: dict[str, str] = {}
        for raw_digest, raw_path in raw_paths.items():
            key = _required_digest(raw_digest, f"{where}.checkpoint_paths key")
            path = str(raw_path or "").strip()
            if not path.startswith("/"):
                raise FleetManifestError(
                    f"{where}.checkpoint_paths[{key!r}] must be absolute"
                )
            paths[key] = path
        template = str(value.get("checkpoint_path_template") or "").strip() or None
        if template is not None and (
            not template.startswith("/") or "{basename}" not in template
        ):
            raise FleetManifestError(
                f"{where}.checkpoint_path_template must be absolute and contain "
                "{basename}"
            )
        if template is not None:
            try:
                template.format(basename="checkpoint.pt", digest="0" * 64)
            except (KeyError, ValueError) as exc:
                raise FleetManifestError(
                    f"{where}.checkpoint_path_template may contain only "
                    "{{basename}} and {{digest}} placeholders"
                ) from exc
        if digest not in paths and template is None:
            raise FleetManifestError(
                f"{where} needs an exact checkpoint_paths entry for its required "
                "digest or a checkpoint_path_template"
            )
        rewrites: list[tuple[str, str]] = []
        raw_rewrites = value.get("path_rewrites") or []
        if not isinstance(raw_rewrites, list):
            raise FleetManifestError(f"{where}.path_rewrites must be a list")
        for rewrite_index, rewrite in enumerate(raw_rewrites):
            if not isinstance(rewrite, dict) or set(rewrite) != {"from", "to"}:
                raise FleetManifestError(
                    f"{where}.path_rewrites[{rewrite_index}] must contain only from/to"
                )
            source = str(rewrite["from"] or "")
            destination = str(rewrite["to"] or "")
            if not source.startswith("/") or not destination.startswith("/"):
                raise FleetManifestError(
                    f"{where}.path_rewrites[{rewrite_index}] paths must be absolute"
                )
            rewrites.append((source.rstrip("/") + "/", destination.rstrip("/") + "/"))
        raw_kinds = value.get("required_job_kinds") or []
        raw_capabilities = value.get("required_capabilities") or []
        if not isinstance(raw_kinds, list):
            raise FleetManifestError(f"{where}.required_job_kinds must be a list")
        if not isinstance(raw_capabilities, list):
            raise FleetManifestError(f"{where}.required_capabilities must be a list")
        expected_hostname = str(value.get("expected_hostname") or "").strip() or None
        required_job_kinds = tuple(
            sorted({str(item) for item in raw_kinds})
        )
        required_capabilities = tuple(
            sorted({str(item) for item in raw_capabilities})
        )
        if enabled and expected_hostname is None:
            raise FleetManifestError(
                f"{where}.expected_hostname is required when enabled"
            )
        if enabled and not required_job_kinds:
            raise FleetManifestError(
                f"{where}.required_job_kinds is required when enabled"
            )
        if enabled and "checkpoint_digest_verify_v1" not in required_capabilities:
            raise FleetManifestError(
                f"{where}.required_capabilities must include "
                "checkpoint_digest_verify_v1 when enabled"
            )
        return cls(
            id=backend_id,
            endpoint=endpoint,
            enabled=enabled,
            capacity=capacity,
            expected_hostname=expected_hostname,
            required_checkpoint_digest=digest,
            checkpoint_paths=paths,
            checkpoint_path_template=template,
            required_job_kinds=required_job_kinds,
            required_capabilities=required_capabilities,
            path_rewrites=tuple(rewrites),
            checkpoint_stage=CheckpointStageConfig.from_dict(
                value.get("checkpoint_stage"), f"{where}.checkpoint_stage"
            ),
        )

    def checkpoint_path(self, source_path: str, digest: Optional[str]) -> str:
        selected_digest = str(digest or self.required_checkpoint_digest)
        exact = self.checkpoint_paths.get(selected_digest)
        if exact:
            return exact
        if self.checkpoint_path_template:
            return self.checkpoint_path_template.format(
                basename=Path(source_path).name,
                digest=selected_digest.removeprefix("sha256:"),
            )
        raise FleetManifestError(
            f"backend {self.id} has no path for checkpoint {selected_digest}"
        )

    def rewrite_path(self, path: str) -> str:
        for source, destination in self.path_rewrites:
            if path.startswith(source):
                return destination + path[len(source) :]
        return path


@dataclass(frozen=True)
class FleetManifest:
    path: Path
    identity: str
    gateway_id: str
    activation_allowed: bool
    bind_host: str
    bind_port: int
    default_workers: int
    fleet_worker_ceiling: int
    registry_poll_seconds: float
    backends: tuple[BackendConfig, ...]

    @classmethod
    def load(cls, path: Path) -> "FleetManifest":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FleetManifestError(f"cannot read fleet manifest {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise FleetManifestError("fleet manifest must be an object")
        _strict_fields(value, _TOP_LEVEL_FIELDS, "manifest")
        if value.get("schema") != MANIFEST_SCHEMA:
            raise FleetManifestError(f"manifest.schema must be {MANIFEST_SCHEMA!r}")
        if not isinstance(value.get("activation_allowed", False), bool):
            raise FleetManifestError("manifest.activation_allowed must be a boolean")
        gateway_id = str(value.get("gateway_id") or "").strip()
        if not gateway_id:
            raise FleetManifestError("manifest.gateway_id is required")
        bind_host = str(value.get("bind_host") or "127.0.0.1").strip()
        if bind_host not in {"127.0.0.1", "::1"}:
            raise FleetManifestError(
                "manifest.bind_host must be loopback; expose it only through an "
                "authenticated tunnel or private load balancer"
            )
        bind_port = int(value.get("bind_port", 8770))
        if not 1 <= bind_port <= 65535:
            raise FleetManifestError("manifest.bind_port must be 1..65535")
        ceiling = int(value.get("fleet_worker_ceiling", 0))
        default_workers = int(value.get("default_workers", 0))
        if not 1 <= default_workers <= ceiling <= 4096:
            raise FleetManifestError(
                "require 1 <= default_workers <= fleet_worker_ceiling <= 4096"
            )
        poll = float(value.get("registry_poll_seconds", 2.0))
        if not 0.25 <= poll <= 60.0:
            raise FleetManifestError("registry_poll_seconds must be 0.25..60")
        raw_backends = value.get("backends")
        if not isinstance(raw_backends, list):
            raise FleetManifestError("manifest.backends must be a list")
        backends = tuple(
            BackendConfig.from_dict(item, index)
            for index, item in enumerate(raw_backends)
        )
        ids = [backend.id for backend in backends]
        endpoints = [backend.endpoint for backend in backends]
        if len(ids) != len(set(ids)):
            raise FleetManifestError("backend ids must be unique")
        if len(endpoints) != len(set(endpoints)):
            raise FleetManifestError("backend endpoints must be unique")
        enabled_capacity = sum(b.capacity for b in backends if b.enabled)
        if enabled_capacity > ceiling:
            raise FleetManifestError(
                f"enabled capacity {enabled_capacity} exceeds fleet ceiling {ceiling}"
            )
        enabled_digests = {
            backend.required_checkpoint_digest for backend in backends if backend.enabled
        }
        if len(enabled_digests) > 1:
            raise FleetManifestError(
                "all enabled backends must declare the same required checkpoint digest"
            )
        return cls(
            path=path.resolve(),
            identity=_sha256_json(value),
            gateway_id=gateway_id,
            activation_allowed=bool(value.get("activation_allowed", False)),
            bind_host=bind_host,
            bind_port=bind_port,
            default_workers=default_workers,
            fleet_worker_ceiling=ceiling,
            registry_poll_seconds=poll,
            backends=backends,
        )


class _BackendRuntime:
    def __init__(
        self,
        config: BackendConfig,
        client_factory: Callable[..., RemoteJobClient],
    ) -> None:
        self.config = config
        self._client_factory = client_factory
        self._slots = threading.BoundedSemaphore(config.capacity)
        self._idle: queue.LifoQueue[RemoteJobClient] = queue.LifoQueue()
        self._lock = threading.Lock()
        self.active = 0
        self.admission_info: Optional[Any] = None

    def _new_client(self) -> RemoteJobClient:
        host, port = parse_endpoint(self.config.endpoint)
        client = self._client_factory(host, port)
        client.connect()
        return client

    def preflight(self) -> None:
        client = self._new_client()
        try:
            info = client.info
            if info is None:
                raise FleetManifestError(f"backend {self.config.id} returned no hello")
            if (
                self.config.expected_hostname
                and info.hostname != self.config.expected_hostname
            ):
                raise FleetManifestError(
                    f"backend {self.config.id} hostname mismatch: "
                    f"{info.hostname!r} != {self.config.expected_hostname!r}"
                )
            if info.checkpoint_digest != self.config.required_checkpoint_digest:
                raise FleetManifestError(
                    f"backend {self.config.id} hello checkpoint mismatch: "
                    f"{info.checkpoint_digest!r} != "
                    f"{self.config.required_checkpoint_digest!r}"
                )
            missing_kinds = sorted(
                set(self.config.required_job_kinds) - set(info.job_kinds)
            )
            missing_caps = sorted(
                set(self.config.required_capabilities) - set(info.capabilities)
            )
            if missing_kinds or missing_caps:
                raise FleetManifestError(
                    f"backend {self.config.id} missing job_kinds={missing_kinds} "
                    f"capabilities={missing_caps}"
                )
            checkpoint_path = self.config.checkpoint_path(
                "checkpoint.pt", self.config.required_checkpoint_digest
            )
            proof = self._raw_verify(client, checkpoint_path)
            if proof.get("checkpoint_digest") != self.config.required_checkpoint_digest:
                raise FleetManifestError(
                    f"backend {self.config.id} storage checkpoint mismatch"
                )
            self.admission_info = copy.deepcopy(info)
            self._idle.put(client)
            client = None  # type: ignore[assignment]
        finally:
            if client is not None:
                client.close()

    @staticmethod
    def _raw_verify(client: RemoteJobClient, path: str) -> dict[str, Any]:
        sock = client._require_sock()
        client._send(sock, {"type": "verify_checkpoint", "path": path})
        reply = read_frame(sock)
        if reply.get("type") != "verify_checkpoint_ok" or reply.get("ok") is not True:
            raise FleetManifestError(
                f"checkpoint verification failed: {reply.get('error') or reply!r}"
            )
        return reply

    def try_acquire(self) -> Optional[RemoteJobClient]:
        if not self._slots.acquire(blocking=False):
            return None
        with self._lock:
            self.active += 1
        try:
            try:
                return self._idle.get_nowait()
            except queue.Empty:
                return self._new_client()
        except Exception:
            with self._lock:
                self.active -= 1
            self._slots.release()
            raise

    def release(self, client: RemoteJobClient, *, reusable: bool) -> None:
        if reusable:
            self._idle.put(client)
        else:
            client.close()
        with self._lock:
            self.active -= 1
        self._slots.release()

    def close_idle(self) -> None:
        while True:
            try:
                self._idle.get_nowait().close()
            except queue.Empty:
                return


class RemoteFleetGateway:
    """Route existing protocol requests across a checksum-gated fleet."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        client_factory: Callable[..., RemoteJobClient] = RemoteJobClient,
        checkpoint_stager: Optional[
            Callable[[Path, str, BackendConfig, str], None]
        ] = None,
    ) -> None:
        self.manifest_path = manifest_path.resolve()
        self._client_factory = client_factory
        self._checkpoint_stager = checkpoint_stager
        self._lock = threading.RLock()
        self._available = threading.Condition(self._lock)
        self._manifest: Optional[FleetManifest] = None
        self._runtimes: tuple[_BackendRuntime, ...] = ()
        self._next_backend = 0
        self._last_poll = 0.0
        self._pending_error: Optional[str] = None
        self._waiting_jobs = 0
        self._jobs_completed = 0
        self._jobs_failed = 0

    @property
    def manifest(self) -> FleetManifest:
        with self._lock:
            if self._manifest is None:
                raise FleetManifestError("no admitted fleet manifest")
            return self._manifest

    def refresh(self, *, force: bool = False) -> bool:
        now = time.monotonic()
        created: list[_BackendRuntime] = []
        with self._lock:
            if (
                not force
                and self._manifest is not None
                and now - self._last_poll < self._manifest.registry_poll_seconds
            ):
                return False
            self._last_poll = now
        try:
            candidate = FleetManifest.load(self.manifest_path)
            if not candidate.activation_allowed:
                raise FleetManifestError(
                    "manifest is staged only (activation_allowed=false)"
                )
            with self._lock:
                if self._manifest and candidate.identity == self._manifest.identity:
                    self._pending_error = None
                    return False
                current = {runtime.config.id: runtime for runtime in self._runtimes}
                if any(runtime.active for runtime in self._runtimes):
                    current_ids = {runtime.config.id for runtime in self._runtimes}
                    candidate_ids = {b.id for b in candidate.backends if b.enabled}
                    changed_ids = {
                        backend.id
                        for backend in candidate.backends
                        if backend.enabled
                        and backend.id in current
                        and current[backend.id].config != backend
                    }
                    if not current_ids.issubset(candidate_ids) or changed_ids:
                        raise FleetManifestError(
                            "backend removal/change waits until all active jobs drain"
                        )
            runtimes: list[_BackendRuntime] = []
            for backend in candidate.backends:
                if not backend.enabled:
                    continue
                runtime = current.get(backend.id)
                if runtime is not None and runtime.config != backend:
                    runtime = None
                if runtime is None:
                    runtime = _BackendRuntime(backend, self._client_factory)
                    runtime.preflight()
                    created.append(runtime)
                runtimes.append(runtime)
            if not runtimes:
                raise FleetManifestError("at least one enabled backend is required")
            old: tuple[_BackendRuntime, ...]
            with self._lock:
                old = self._runtimes
                self._manifest = candidate
                self._runtimes = tuple(runtimes)
                self._pending_error = None
                self._available.notify_all()
            retained = set(runtimes)
            for runtime in old:
                if runtime not in retained and runtime.active == 0:
                    runtime.close_idle()
            return True
        except Exception as exc:
            for runtime in created:
                runtime.close_idle()
            with self._lock:
                self._pending_error = f"{type(exc).__name__}: {exc}"
            if self._manifest is None:
                raise
            return False

    def _acquire(self) -> tuple[_BackendRuntime, RemoteJobClient]:
        self.refresh()
        while True:
            with self._available:
                runtimes = self._runtimes
                for offset in range(len(runtimes)):
                    index = (self._next_backend + offset) % len(runtimes)
                    runtime = runtimes[index]
                    client = runtime.try_acquire()
                    if client is not None:
                        self._next_backend = (index + 1) % len(runtimes)
                        return runtime, client
                self._available.wait(timeout=0.25)
            self.refresh()

    def _release(
        self, runtime: _BackendRuntime, client: RemoteJobClient, *, reusable: bool
    ) -> None:
        runtime.release(client, reusable=reusable)
        with self._available:
            self._available.notify()

    @staticmethod
    def _rewrite_job(config: BackendConfig, job: dict[str, Any]) -> dict[str, Any]:
        rewritten = copy.deepcopy(job)
        for path_field, digest_field in _CHECKPOINT_FIELDS.items():
            raw_path = rewritten.get(path_field)
            if raw_path:
                rewritten[path_field] = config.checkpoint_path(
                    str(raw_path), rewritten.get(digest_field)
                )
        spec = rewritten.get("spec")
        if isinstance(spec, dict) and spec.get("path"):
            spec["path"] = config.rewrite_path(str(spec["path"]))
        children = rewritten.get("jobs")
        if isinstance(children, list):
            rewritten["jobs"] = [
                RemoteFleetGateway._rewrite_job(config, child)
                for child in children
            ]
        return rewritten

    @staticmethod
    def _forward(client: RemoteJobClient, msg: dict[str, Any]) -> dict[str, Any]:
        sock = client._require_sock()  # one exclusively leased connection
        previous = sock.gettimeout()
        job = msg.get("job") if isinstance(msg.get("job"), dict) else {}
        timeout_s = max(
            client.control_timeout_s,
            float(job.get("game_timeout_s") or 900) + 600.0,
        )
        sock.settimeout(timeout_s if msg.get("type") == "job" else client.control_timeout_s)
        try:
            client._send(sock, msg)
            reply = read_frame(sock)
        finally:
            try:
                sock.settimeout(previous)
            except OSError:
                pass
        if not isinstance(reply, dict):
            raise RemoteJobsError("backend returned a non-object frame")
        return reply

    def _job(self, msg: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._waiting_jobs += 1
        try:
            runtime, client = self._acquire()
        except Exception:
            with self._lock:
                self._waiting_jobs -= 1
                self._jobs_failed += 1
            raise
        with self._lock:
            self._waiting_jobs -= 1
        reusable = True
        try:
            forwarded = copy.deepcopy(msg)
            forwarded["job"] = self._rewrite_job(
                runtime.config, dict(msg.get("job") or {})
            )
            reply = self._forward(client, forwarded)
            if reply.get("type") != "result":
                raise RemoteJobsError(f"unexpected backend job reply: {reply!r}")
            with self._lock:
                if reply.get("ok") is True:
                    self._jobs_completed += 1
                else:
                    self._jobs_failed += 1
            return reply
        except (OSError, TimeoutError, RemoteJobsError):
            reusable = False
            with self._lock:
                self._jobs_failed += 1
            raise
        finally:
            self._release(runtime, client, reusable=reusable)

    def _fanout_control(self, msg: dict[str, Any]) -> dict[str, Any]:
        self.refresh()
        runtimes = self._runtimes
        replies: list[tuple[str, dict[str, Any]]] = []
        errors: list[str] = []
        prepared: list[tuple[_BackendRuntime, RemoteJobClient, dict[str, Any]]] = []
        request_type = str(msg.get("type"))
        requested_digest = str(msg.get("digest") or "")
        if request_type in {"reload", "pin"} and not requested_digest.startswith(
            "sha256:"
        ):
            return {
                "type": f"{request_type}_ok",
                "ok": False,
                "error": "fleet reload/pin requires an explicit sha256 digest",
            }
        if request_type in {"reload", "pin"}:
            source_path = Path(str(msg.get("path") or ""))
            staged = [r for r in runtimes if r.config.checkpoint_stage is not None]
            if staged:
                try:
                    if self._checkpoint_stager is None:
                        from poke_bot.aws_remote_fleet import stage_checkpoint_for_backend

                        stager = stage_checkpoint_for_backend
                    else:
                        stager = self._checkpoint_stager
                    for runtime in staged:
                        target_path = runtime.config.checkpoint_path(
                            str(source_path), requested_digest
                        )
                        stager(
                            source_path,
                            requested_digest,
                            runtime.config,
                            target_path,
                        )
                except Exception as exc:
                    return {
                        "type": f"{request_type}_ok",
                        "ok": False,
                        "error": (
                            "checkpoint staging failed before any backend reload/pin: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                    }
        # Prepare every connection and verify every target object before any
        # mutating reload/pin is sent. A failed preflight leaves all leaves on
        # their prior identity.
        for runtime in runtimes:
            client: Optional[RemoteJobClient] = None
            try:
                client = runtime._new_client()
                forwarded = copy.deepcopy(msg)
                if request_type in {"reload", "pin", "verify_checkpoint"}:
                    digest = msg.get("digest")
                    source_path = str(msg.get("path") or "checkpoint.pt")
                    if (
                        request_type == "verify_checkpoint"
                        and digest is None
                        and runtime.config.checkpoint_path_template
                    ):
                        forwarded["path"] = (
                            runtime.config.checkpoint_path_template.format(
                                basename=Path(source_path).name,
                                digest="",
                            )
                        )
                    else:
                        forwarded["path"] = runtime.config.checkpoint_path(
                            source_path, digest
                        )
                if request_type in {"reload", "pin"}:
                    proof = runtime._raw_verify(client, forwarded["path"])
                    if proof.get("checkpoint_digest") != requested_digest:
                        raise FleetManifestError(
                            f"storage preflight returned "
                            f"{proof.get('checkpoint_digest')!r}, expected "
                            f"{requested_digest!r}"
                        )
                prepared.append((runtime, client, forwarded))
            except Exception as exc:
                errors.append(f"{runtime.config.id}: {type(exc).__name__}: {exc}")
                if client is not None:
                    client.close()
        if errors:
            for _, client, _ in prepared:
                client.close()
            return {
                "type": f"{request_type}_ok",
                "ok": False,
                "error": "; ".join(errors),
            }
        for runtime, client, forwarded in prepared:
            try:
                reply = self._forward(client, forwarded)
                replies.append((runtime.config.id, reply))
                if reply.get("ok") is not True:
                    errors.append(f"{runtime.config.id}: {reply.get('error') or reply}")
            except Exception as exc:
                errors.append(f"{runtime.config.id}: {type(exc).__name__}: {exc}")
            finally:
                client.close()
        reply_type = f"{request_type}_ok"
        if errors:
            return {"type": reply_type, "ok": False, "error": "; ".join(errors)}
        digests = {
            str(reply.get("checkpoint_digest"))
            for _, reply in replies
            if reply.get("checkpoint_digest")
        }
        if len(digests) > 1:
            return {
                "type": reply_type,
                "ok": False,
                "error": f"backend checkpoint identities diverged: {sorted(digests)}",
            }
        result: dict[str, Any] = {
            "type": reply_type,
            "ok": True,
            "backends": [backend for backend, _ in replies],
        }
        if digests:
            result["checkpoint_digest"] = next(iter(digests))
        if request_type == "reload":
            if msg.get("version") is not None:
                result["version"] = int(msg["version"])
            for runtime in runtimes:
                runtime.admission_info.checkpoint_digest = requested_digest
        if request_type == "rotate":
            for runtime in runtimes:
                runtime.close_idle()
        return result

    def hello(self) -> dict[str, Any]:
        self.refresh()
        with self._lock:
            manifest = self.manifest
            infos = [runtime.admission_info for runtime in self._runtimes]
            job_kinds = set(infos[0].job_kinds)
            capabilities = set(infos[0].capabilities)
            for info in infos[1:]:
                job_kinds.intersection_update(info.job_kinds)
                capabilities.intersection_update(info.capabilities)
            digests = {info.checkpoint_digest for info in infos}
            runtimes = list(self._runtimes)
            return {
                "workers": min(
                    manifest.default_workers, sum(r.config.capacity for r in runtimes)
                ),
                "default_workers": manifest.default_workers,
                "max_workers": manifest.fleet_worker_ceiling,
                "leaf_servers": sum(int(info.leaf_servers) for info in infos),
                "gpu_name": "fleet:" + ",".join(sorted({info.gpu_name for info in infos})),
                "device": "fleet-gateway",
                "hostname": manifest.gateway_id,
                "checkpoint_digest": next(iter(digests)) if len(digests) == 1 else None,
                "job_kinds": sorted(job_kinds),
                "capabilities": sorted(capabilities | {"elastic_fleet_gateway_v1"}),
                "matchup_runtime": copy.deepcopy(infos[0].matchup_runtime)
                if all(info.matchup_runtime == infos[0].matchup_runtime for info in infos)
                else None,
                "fleet_manifest_sha256": manifest.identity,
                "fleet_backend_ids": [r.config.id for r in runtimes],
            }

    def health(self) -> dict[str, Any]:
        hello = self.hello()
        with self._lock:
            executing = sum(runtime.active for runtime in self._runtimes)
            queued = self._waiting_jobs
            return {
                "type": "health_ok",
                "ok": True,
                **hello,
                "controller_healthy": True,
                "leaf_alive": True,
                "leaf_identity_ok": hello.get("checkpoint_digest") is not None,
                "accepting_jobs": True,
                "active_jobs": executing + queued,
                "queued_jobs": queued,
                "jobs_completed": self._jobs_completed,
                "jobs_failed": self._jobs_failed,
                "pending_registry_error": self._pending_error,
                "backends": [
                    {
                        "id": runtime.config.id,
                        "endpoint": runtime.config.endpoint,
                        "capacity": runtime.config.capacity,
                        "active_jobs": runtime.active,
                        "checkpoint_digest": runtime.admission_info.checkpoint_digest,
                    }
                    for runtime in self._runtimes
                ],
            }

    def handle(self, msg: dict[str, Any]) -> dict[str, Any]:
        request_type = msg.get("type")
        if request_type == "health":
            return self.health()
        if request_type == "job":
            return self._job(msg)
        if request_type in {
            "verify_checkpoint",
            "reload",
            "pin",
            "unpin",
            "rotate",
        }:
            return self._fanout_control(msg)
        return {
            "type": "error",
            "ok": False,
            "error": f"fleet gateway does not permit request type {request_type!r}",
        }

    def close(self) -> None:
        with self._lock:
            runtimes = self._runtimes
        for runtime in runtimes:
            runtime.close_idle()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the staged manifest without connecting or serving",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _parser().parse_args(argv)
    manifest = FleetManifest.load(args.manifest)
    if args.check:
        print(
            json.dumps(
                {
                    "ok": True,
                    "schema": MANIFEST_SCHEMA,
                    "manifest_sha256": manifest.identity,
                    "activation_allowed": manifest.activation_allowed,
                    "enabled_backends": [b.id for b in manifest.backends if b.enabled],
                    "enabled_capacity": sum(
                        b.capacity for b in manifest.backends if b.enabled
                    ),
                },
                sort_keys=True,
            )
        )
        return 0
    if not manifest.activation_allowed:
        raise SystemExit(
            "refusing to serve: manifest is staged with activation_allowed=false"
        )
    gateway = RemoteFleetGateway(args.manifest)
    try:
        gateway.refresh(force=True)
        serve_forever(
            gateway.handle,
            host=manifest.bind_host,
            port=manifest.bind_port,
            hello=gateway.hello,
            max_connections=max(128, manifest.fleet_worker_ceiling * 2),
        )
    finally:
        gateway.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
