"""Safe atomic registry edits for :mod:`poke_bot.remote_fleet_gateway`."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import re
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

from poke_bot.remote_fleet_gateway import FleetManifest, FleetManifestError
from poke_bot.remote_jobs import RemoteJobClient, RemoteJobsError, parse_endpoint


DEFAULT_MANIFEST = Path("/etc/pokebot/remote-fleet-gateway.active.json")
DEFAULT_TRAINER_ROOT = os.environ.get(
    "POKEBOT_TRAINER_ROOT", str(Path.cwd()) + "/"
)
DEFAULT_WORKER_ROOT = os.environ.get(
    "POKEBOT_WORKER_ROOT", "/opt/pokebot/app/"
)


@dataclass(frozen=True)
class DiscoveredBackend:
    entry: dict[str, Any]
    health: dict[str, Any]


def _manifest_digest(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _safe_id(hostname: str, host: str) -> str:
    value = re.sub(r"[^a-z0-9_-]+", "-", hostname.lower()).strip("-")
    if value:
        return value
    fallback = re.sub(r"[^a-z0-9_-]+", "-", host.lower()).strip("-")
    if not fallback:
        raise FleetManifestError("remote did not provide a usable hostname or host")
    return fallback


def normalize_endpoint(value: str, *, default_port: int = 8765) -> str:
    text = str(value).strip()
    if not text:
        raise FleetManifestError("remote endpoint cannot be empty")
    if text.startswith("tcp://"):
        text = text[6:]
    if ":" not in text:
        text = f"{text}:{int(default_port)}"
    host, port = parse_endpoint(text)
    if not host or not 1 <= int(port) <= 65535:
        raise FleetManifestError(f"invalid remote endpoint: {value!r}")
    return f"{host}:{int(port)}"


def discover_backend(
    endpoint: str,
    *,
    checkpoint_path: Optional[str] = None,
    capacity: Optional[int] = None,
    backend_id: Optional[str] = None,
    timeout_s: float = 10.0,
    trainer_root: Optional[str] = DEFAULT_TRAINER_ROOT,
    worker_root: Optional[str] = DEFAULT_WORKER_ROOT,
) -> DiscoveredBackend:
    """Probe one worker and build a checksum-bound registry entry."""

    normalized = normalize_endpoint(endpoint)
    host, port = parse_endpoint(normalized)
    client = RemoteJobClient(
        host,
        port,
        timeout_s=timeout_s,
        connect_timeout_s=timeout_s,
        control_timeout_s=timeout_s,
    )
    try:
        info = client.connect()
        health = client.health()
        if health.get("ok") is not True:
            raise FleetManifestError(
                f"{normalized} health failed: {health.get('error') or health!r}"
            )
        if health.get("controller_healthy") is not True:
            raise FleetManifestError(f"{normalized} controller is not healthy")
        if health.get("leaf_alive") is not True or health.get("leaf_identity_ok") is not True:
            raise FleetManifestError(f"{normalized} leaf identity is not healthy")
        digest = str(info.checkpoint_digest or "")
        if not digest.startswith("sha256:") or len(digest) != 71:
            raise FleetManifestError(
                f"{normalized} did not advertise a full checkpoint digest"
            )
        capabilities = sorted(set(info.capabilities))
        if "checkpoint_digest_verify_v1" not in capabilities:
            raise FleetManifestError(
                f"{normalized} lacks checkpoint_digest_verify_v1"
            )
        remote_path = str(
            checkpoint_path
            or info.checkpoint_path
            or health.get("checkpoint_path")
            or ""
        ).strip()
        if not remote_path:
            raise FleetManifestError(
                f"{normalized} is an older worker that does not advertise its "
                "checkpoint path; pass --checkpoint-path"
            )
        if not remote_path.startswith("/"):
            raise FleetManifestError("remote checkpoint path must be absolute")
        proof = client.verify_checkpoint(remote_path)
        if proof.get("checkpoint_digest") != digest:
            raise FleetManifestError(
                f"{normalized} checkpoint proof mismatch: "
                f"{proof.get('checkpoint_digest')!r} != {digest!r}"
            )
        discovered_capacity = max(
            1,
            int(capacity or info.max_workers or info.workers or 1),
        )
        if discovered_capacity > max(1, int(info.max_workers or info.workers or 1)):
            raise FleetManifestError(
                f"requested capacity {discovered_capacity} exceeds worker-advertised "
                f"capacity {int(info.max_workers or info.workers or 1)}"
            )
        path_rewrites: list[dict[str, str]] = []
        if trainer_root and worker_root:
            path_rewrites.append({"from": trainer_root, "to": worker_root})
        parent = str(Path(remote_path).parent)
        entry = {
            "id": backend_id or _safe_id(info.hostname, host),
            "endpoint": normalized,
            "enabled": True,
            "capacity": discovered_capacity,
            "expected_hostname": info.hostname,
            "required_checkpoint_digest": digest,
            "checkpoint_paths": {digest: remote_path},
            "checkpoint_path_template": f"{parent}/{{basename}}",
            "required_job_kinds": sorted(set(info.job_kinds)),
            "required_capabilities": capabilities,
            "path_rewrites": path_rewrites,
        }
        return DiscoveredBackend(entry=entry, health=copy.deepcopy(health))
    except (OSError, TimeoutError, RemoteJobsError) as exc:
        raise FleetManifestError(
            f"cannot preflight {normalized}: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        client.close()


@contextmanager
def registry_lock(manifest_path: Path) -> Iterator[None]:
    lock_path = manifest_path.with_name(manifest_path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def read_registry(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FleetManifestError(f"cannot read registry {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FleetManifestError("registry must be a JSON object")
    return value


def atomic_write_registry(
    path: Path,
    value: dict[str, Any],
    *,
    dry_run: bool = False,
) -> str:
    """Validate, fsync, and atomically replace one registry snapshot."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o640
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        FleetManifest.load(temporary)
        identity = _manifest_digest(value)
        if dry_run:
            return identity
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return identity
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def add_backends(
    manifest_path: Path,
    discovered: list[DiscoveredBackend],
    *,
    replace: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Add a fully preflighted batch with one all-or-nothing registry write."""

    if not discovered:
        raise FleetManifestError("at least one discovered backend is required")
    with registry_lock(manifest_path):
        registry = read_registry(manifest_path)
        backends = list(registry.get("backends") or [])
        by_id = {str(item.get("id")): index for index, item in enumerate(backends)}
        by_endpoint = {
            str(item.get("endpoint")): index for index, item in enumerate(backends)
        }
        added: list[str] = []
        updated: list[str] = []
        for candidate in discovered:
            entry = copy.deepcopy(candidate.entry)
            backend_id = str(entry["id"])
            endpoint = str(entry["endpoint"])
            id_index = by_id.get(backend_id)
            endpoint_index = by_endpoint.get(endpoint)
            if id_index is not None or endpoint_index is not None:
                indexes = {value for value in (id_index, endpoint_index) if value is not None}
                if len(indexes) != 1:
                    raise FleetManifestError(
                        f"backend id/endpoint collide with different entries: "
                        f"id={backend_id} endpoint={endpoint}"
                    )
                index = next(iter(indexes))
                existing = backends[index]
                if existing == entry:
                    updated.append(backend_id)
                    continue
                if not replace:
                    raise FleetManifestError(
                        f"backend {backend_id}/{endpoint} already exists; pass --replace"
                    )
                backends[index] = entry
                updated.append(backend_id)
            else:
                by_id[backend_id] = len(backends)
                by_endpoint[endpoint] = len(backends)
                backends.append(entry)
                added.append(backend_id)
        registry["backends"] = backends
        identity = atomic_write_registry(manifest_path, registry, dry_run=dry_run)
    return {
        "ok": True,
        "dry_run": dry_run,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": identity,
        "added": added,
        "updated": updated,
        "enabled_capacity": sum(
            int(item.get("capacity", 0))
            for item in backends
            if item.get("enabled") is True
        ),
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def remove_backends(
    manifest_path: Path,
    selectors: list[str],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Remove IDs/endpoints atomically; a live gateway drains before adoption."""

    wanted = set(selectors)
    with registry_lock(manifest_path):
        registry = read_registry(manifest_path)
        backends = list(registry.get("backends") or [])
        removed = [
            str(item.get("id"))
            for item in backends
            if str(item.get("id")) in wanted
            or str(item.get("endpoint")) in wanted
        ]
        missing = sorted(wanted - {
            value
            for item in backends
            for value in (str(item.get("id")), str(item.get("endpoint")))
        })
        if missing:
            raise FleetManifestError(f"unknown backend selectors: {', '.join(missing)}")
        registry["backends"] = [
            item
            for item in backends
            if str(item.get("id")) not in wanted
            and str(item.get("endpoint")) not in wanted
        ]
        identity = atomic_write_registry(manifest_path, registry, dry_run=dry_run)
    return {
        "ok": True,
        "dry_run": dry_run,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": identity,
        "removed": removed,
        "note": "a running gateway adopts removal only after active jobs drain",
    }
