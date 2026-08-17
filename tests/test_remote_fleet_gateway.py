from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from poke_bot.remote_fleet_gateway import (
    FleetManifest,
    FleetManifestError,
    RemoteFleetGateway,
    _BackendRuntime,
)
from poke_bot.remote_jobs import RemoteJobClient, read_frame, serve_forever


DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64


def _backend(
    backend_id: str,
    port: int,
    *,
    enabled: bool = True,
    digest: str = DIGEST_A,
) -> dict:
    return {
        "id": backend_id,
        "endpoint": f"127.0.0.1:{port}",
        "enabled": enabled,
        "capacity": 2,
        "expected_hostname": backend_id,
        "required_checkpoint_digest": digest,
        "checkpoint_paths": {digest: f"/checkpoints/{backend_id}/active.pt"},
        "checkpoint_path_template": f"/checkpoints/{backend_id}/{{basename}}",
        "required_job_kinds": ["play", "self_play"],
        "required_capabilities": ["checkpoint_digest_verify_v1"],
        "path_rewrites": [{"from": "/trainer/", "to": "/worker/"}],
    }


def _stage(port: int = 19022) -> dict:
    return {
        "mode": "ssm_ssh_v1",
        "ssh_host": "127.0.0.1",
        "ssh_port": port,
        "ssh_user": "ec2-user",
        "identity_file": "/run/pokebot/aws/id_ed25519",
        "known_hosts_file": "/run/pokebot/aws/known_hosts",
    }


def _manifest(backends: list[dict], *, activation_allowed: bool = True) -> dict:
    return {
        "schema": "poke_bot.remote_fleet_gateway/v1",
        "gateway_id": "test-fleet",
        "activation_allowed": activation_allowed,
        "bind_host": "127.0.0.1",
        "bind_port": 18770,
        "default_workers": 2,
        "fleet_worker_ceiling": 16,
        "registry_poll_seconds": 0.25,
        "backends": backends,
    }


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _unused_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


class _FakeClient:
    fail_ports: set[int] = set()

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.control_timeout_s = 3.0
        self.info = None
        self.closed = False

    def connect(self):
        if self.port in self.fail_ports:
            raise OSError("preflight refused")
        backend_id = {18001: "one", 18002: "two"}.get(self.port, "unknown")
        self.info = SimpleNamespace(
            hostname=backend_id,
            checkpoint_digest=DIGEST_A,
            job_kinds=("play", "self_play", "self_play_multi"),
            capabilities=("checkpoint_digest_verify_v1",),
            workers=2,
            max_workers=2,
            default_workers=2,
            leaf_servers=1,
            gpu_name="fake-gpu",
            device="cuda:0",
            matchup_runtime=None,
        )
        return self.info

    def close(self) -> None:
        self.closed = True


def test_staged_manifest_validates_without_backends(tmp_path: Path) -> None:
    path = tmp_path / "fleet.json"
    _write(path, _manifest([], activation_allowed=False))
    manifest = FleetManifest.load(path)
    assert manifest.activation_allowed is False
    assert manifest.backends == ()
    assert manifest.identity.startswith("sha256:")


def test_manifest_rejects_public_bind_unknown_fields_and_mixed_digests(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fleet.json"
    value = _manifest([_backend("one", 18001)])
    value["bind_host"] = "0.0.0.0"
    _write(path, value)
    with pytest.raises(FleetManifestError, match="loopback"):
        FleetManifest.load(path)

    value["bind_host"] = "127.0.0.1"
    value["surprise"] = True
    _write(path, value)
    with pytest.raises(FleetManifestError, match="unknown fields"):
        FleetManifest.load(path)

    value.pop("surprise")
    value["backends"].append(_backend("two", 18002, digest=DIGEST_B))
    _write(path, value)
    with pytest.raises(FleetManifestError, match="same required checkpoint"):
        FleetManifest.load(path)


def test_checkpoint_stage_is_strict_and_loopback_only(tmp_path: Path) -> None:
    path = tmp_path / "fleet.json"
    backend = _backend("one", 18001)
    backend["checkpoint_stage"] = _stage()
    _write(path, _manifest([backend]))
    parsed = FleetManifest.load(path).backends[0]
    assert parsed.checkpoint_stage is not None
    assert parsed.checkpoint_stage.ssh_port == 19022

    backend["checkpoint_stage"]["ssh_host"] = "203.0.113.9"
    _write(path, _manifest([backend]))
    with pytest.raises(FleetManifestError, match="loopback"):
        FleetManifest.load(path)


def test_job_rewrite_is_backend_local() -> None:
    from poke_bot.remote_fleet_gateway import BackendConfig

    backend = BackendConfig.from_dict(_backend("one", 18001), 0)
    rewritten = RemoteFleetGateway._rewrite_job(
        backend,
        {
            "checkpoint": "/trainer/checkpoints/candidate.pt",
            "checkpoint_digest": DIGEST_A,
            "spec": {"path": "/trainer/config/spec.json"},
            "jobs": [
                {
                    "checkpoint": "/trainer/checkpoints/child.pt",
                    "checkpoint_digest": DIGEST_B,
                }
            ],
        },
    )
    assert rewritten["checkpoint"] == "/checkpoints/one/active.pt"
    assert rewritten["spec"]["path"] == "/worker/config/spec.json"
    assert rewritten["jobs"][0]["checkpoint"] == "/checkpoints/one/child.pt"


def test_failed_candidate_registry_keeps_last_good_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "fleet.json"
    _write(path, _manifest([_backend("one", 18001)]))
    monkeypatch.setattr(
        _BackendRuntime,
        "_raw_verify",
        staticmethod(
            lambda _client, _path: {
                "type": "verify_checkpoint_ok",
                "ok": True,
                "checkpoint_digest": DIGEST_A,
            }
        ),
    )
    _FakeClient.fail_ports = set()
    gateway = RemoteFleetGateway(path, client_factory=_FakeClient)
    assert gateway.refresh(force=True) is True
    first_identity = gateway.manifest.identity
    assert gateway.hello()["fleet_backend_ids"] == ["one"]

    _FakeClient.fail_ports = {18002}
    _write(path, _manifest([_backend("one", 18001), _backend("two", 18002)]))
    assert gateway.refresh(force=True) is False
    assert gateway.manifest.identity == first_identity
    health = gateway.health()
    assert health["fleet_backend_ids"] == ["one"]
    assert "preflight refused" in health["pending_registry_error"]
    gateway.close()


def test_new_backend_can_be_hot_added_while_existing_backend_has_active_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "fleet.json"
    _write(path, _manifest([_backend("one", 18001)]))
    monkeypatch.setattr(
        _BackendRuntime,
        "_raw_verify",
        staticmethod(
            lambda _client, _path: {
                "type": "verify_checkpoint_ok",
                "ok": True,
                "checkpoint_digest": DIGEST_A,
            }
        ),
    )
    _FakeClient.fail_ports = set()
    gateway = RemoteFleetGateway(path, client_factory=_FakeClient)
    gateway.refresh(force=True)
    gateway._runtimes[0].active = 1
    _write(path, _manifest([_backend("one", 18001), _backend("two", 18002)]))
    assert gateway.refresh(force=True) is True
    assert gateway.hello()["fleet_backend_ids"] == ["one", "two"]
    gateway._runtimes[0].active = 0
    gateway.close()


def test_control_preflights_every_backend_before_first_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "fleet.json"
    _write(path, _manifest([_backend("one", 18001), _backend("two", 18002)]))
    events: list[str] = []

    def verify(client: _FakeClient, _path: str) -> dict:
        events.append(f"verify:{client.port}")
        return {
            "type": "verify_checkpoint_ok",
            "ok": True,
            "checkpoint_digest": DIGEST_A,
        }

    def forward(client: _FakeClient, msg: dict) -> dict:
        events.append(f"{msg['type']}:{client.port}")
        return {
            "type": f"{msg['type']}_ok",
            "ok": True,
            "checkpoint_digest": DIGEST_A,
            "version": msg.get("version"),
        }

    monkeypatch.setattr(_BackendRuntime, "_raw_verify", staticmethod(verify))
    monkeypatch.setattr(RemoteFleetGateway, "_forward", staticmethod(forward))
    _FakeClient.fail_ports = set()
    gateway = RemoteFleetGateway(path, client_factory=_FakeClient)
    gateway.refresh(force=True)
    events.clear()
    reply = gateway.handle(
        {
            "type": "reload",
            "path": "/trainer/candidate.pt",
            "digest": DIGEST_A,
            "version": 7,
        }
    )
    assert reply["ok"] is True
    assert reply["checkpoint_digest"] == DIGEST_A
    assert events[:2] == ["verify:18001", "verify:18002"]
    assert events[2:] == ["reload:18001", "reload:18002"]
    gateway.close()


def test_checkpoint_staging_precedes_verify_and_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "fleet.json"
    first = _backend("one", 18001)
    second = _backend("two", 18002)
    first["checkpoint_stage"] = _stage(19022)
    second["checkpoint_stage"] = _stage(19023)
    _write(path, _manifest([first, second]))
    events: list[str] = []

    def stage(_source: Path, _digest: str, backend, target: str) -> None:
        events.append(f"stage:{backend.id}:{target}")

    def verify(client: _FakeClient, _path: str) -> dict:
        events.append(f"verify:{client.port}")
        return {
            "type": "verify_checkpoint_ok",
            "ok": True,
            "checkpoint_digest": DIGEST_A,
        }

    def forward(client: _FakeClient, msg: dict) -> dict:
        events.append(f"{msg['type']}:{client.port}")
        return {
            "type": f"{msg['type']}_ok",
            "ok": True,
            "checkpoint_digest": DIGEST_A,
            "version": msg.get("version"),
        }

    monkeypatch.setattr(_BackendRuntime, "_raw_verify", staticmethod(verify))
    monkeypatch.setattr(RemoteFleetGateway, "_forward", staticmethod(forward))
    gateway = RemoteFleetGateway(
        path, client_factory=_FakeClient, checkpoint_stager=stage
    )
    gateway.refresh(force=True)
    events.clear()
    reply = gateway.handle(
        {
            "type": "reload",
            "path": "/trainer/candidate.pt",
            "digest": DIGEST_A,
            "version": 8,
        }
    )
    assert reply["ok"] is True
    assert [event.split(":", 1)[0] for event in events] == [
        "stage",
        "stage",
        "verify",
        "verify",
        "reload",
        "reload",
    ]
    gateway.close()


def test_checkpoint_staging_failure_sends_no_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "fleet.json"
    backend = _backend("one", 18001)
    backend["checkpoint_stage"] = _stage()
    _write(path, _manifest([backend]))
    forwarded: list[str] = []

    monkeypatch.setattr(
        _BackendRuntime,
        "_raw_verify",
        staticmethod(
            lambda _client, _path: {
                "type": "verify_checkpoint_ok",
                "ok": True,
                "checkpoint_digest": DIGEST_A,
            }
        ),
    )
    monkeypatch.setattr(
        RemoteFleetGateway,
        "_forward",
        staticmethod(lambda _client, msg: forwarded.append(msg["type"]) or {}),
    )

    def fail(*_args) -> None:
        raise RuntimeError("copy refused")

    gateway = RemoteFleetGateway(
        path, client_factory=_FakeClient, checkpoint_stager=fail
    )
    gateway.refresh(force=True)
    reply = gateway.handle(
        {"type": "reload", "path": "/trainer/candidate.pt", "digest": DIGEST_A}
    )
    assert reply["ok"] is False
    assert "before any backend reload/pin" in reply["error"]
    assert forwarded == []
    gateway.close()


def test_existing_wire_protocol_routes_through_stable_gateway(
    tmp_path: Path,
) -> None:
    backend_port = _unused_port()
    gateway_port = _unused_port()
    observed_jobs: list[dict] = []
    backend_stop = threading.Event()
    gateway_stop = threading.Event()

    def backend_handler(msg: dict) -> dict:
        if msg.get("type") == "verify_checkpoint":
            return {
                "type": "verify_checkpoint_ok",
                "ok": True,
                "checkpoint_digest": DIGEST_A,
            }
        if msg.get("type") == "job":
            observed_jobs.append(msg["job"])
            return {"type": "result", "ok": True, "result": {"ok": True}}
        raise AssertionError(msg)

    backend_thread = threading.Thread(
        target=serve_forever,
        kwargs={
            "handler": backend_handler,
            "host": "127.0.0.1",
            "port": backend_port,
            "hello": lambda: {
                "workers": 2,
                "max_workers": 2,
                "default_workers": 2,
                "leaf_servers": 1,
                "gpu_name": "fake-gpu",
                "device": "cuda:0",
                "hostname": "one",
                "checkpoint_digest": DIGEST_A,
                "job_kinds": ["play", "self_play"],
                "capabilities": ["checkpoint_digest_verify_v1"],
            },
            "stop_event": backend_stop,
        },
        daemon=True,
    )
    backend_thread.start()

    path = tmp_path / "fleet.json"
    value = _manifest([_backend("one", backend_port)])
    value["bind_port"] = gateway_port
    _write(path, value)
    gateway = RemoteFleetGateway(path)
    deadline = time.monotonic() + 2.0
    while True:
        try:
            gateway.refresh(force=True)
            break
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.01)
    gateway_thread = threading.Thread(
        target=serve_forever,
        kwargs={
            "handler": gateway.handle,
            "host": "127.0.0.1",
            "port": gateway_port,
            "hello": gateway.hello,
            "stop_event": gateway_stop,
        },
        daemon=True,
    )
    gateway_thread.start()
    client = RemoteJobClient(
        "127.0.0.1", gateway_port, connect_timeout_s=1.0, control_timeout_s=1.0
    )
    try:
        deadline = time.monotonic() + 2.0
        while True:
            try:
                info = client.connect()
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)
        assert info.hostname == "test-fleet"
        assert "elastic_fleet_gateway_v1" in info.capabilities
        client._send(
            client._require_sock(),
            {
                "type": "job",
                "kind": "play",
                "job": {
                    "checkpoint": "/trainer/candidate.pt",
                    "checkpoint_digest": DIGEST_A,
                    "spec": {"path": "/trainer/spec.json"},
                },
            },
        )
        assert read_frame(client._require_sock()) == {
            "type": "result",
            "ok": True,
            "result": {"ok": True},
        }
        assert observed_jobs == [
            {
                "checkpoint": "/checkpoints/one/active.pt",
                "checkpoint_digest": DIGEST_A,
                "spec": {"path": "/worker/spec.json"},
            }
        ]
    finally:
        client.close()
        gateway_stop.set()
        backend_stop.set()
        gateway_thread.join(timeout=3.0)
        backend_thread.join(timeout=3.0)
        gateway.close()
    assert not gateway_thread.is_alive()
    assert not backend_thread.is_alive()
