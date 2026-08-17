from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from poke_bot.remote_fleet_gateway import FleetManifestError
from poke_bot.remote_fleet_registry import (
    DiscoveredBackend,
    add_backends,
    discover_backend,
    remove_backends,
)


DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64


def _manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "poke_bot.remote_fleet_gateway/v1",
                "gateway_id": "test-fleet",
                "activation_allowed": False,
                "bind_host": "127.0.0.1",
                "bind_port": 18770,
                "default_workers": 2,
                "fleet_worker_ceiling": 16,
                "registry_poll_seconds": 2.0,
                "backends": [],
            }
        ),
        encoding="utf-8",
    )


def _entry(backend_id: str, endpoint: str, digest: str = DIGEST_A) -> dict:
    return {
        "id": backend_id,
        "endpoint": endpoint,
        "enabled": True,
        "capacity": 4,
        "expected_hostname": backend_id,
        "required_checkpoint_digest": digest,
        "checkpoint_paths": {digest: "/opt/pokebot/checkpoint/active.pt"},
        "checkpoint_path_template": "/opt/pokebot/checkpoint/{basename}",
        "required_job_kinds": ["play", "self_play"],
        "required_capabilities": ["checkpoint_digest_verify_v1"],
        "path_rewrites": [
            {
                "from": "/home/inzi/poke-bot-agent/",
                "to": "/opt/pokebot/app/",
            }
        ],
    }


class _FakeClient:
    def __init__(self, host: str, port: int, **_kwargs) -> None:
        self.host = host
        self.port = port
        self.closed = False

    def connect(self):
        return SimpleNamespace(
            hostname="aws-node-1",
            checkpoint_digest=DIGEST_A,
            checkpoint_path="/opt/pokebot/checkpoint/active.pt",
            job_kinds=("play", "self_play"),
            capabilities=("checkpoint_digest_verify_v1",),
            max_workers=28,
            workers=16,
        )

    def health(self):
        return {
            "ok": True,
            "controller_healthy": True,
            "leaf_alive": True,
            "leaf_identity_ok": True,
        }

    def verify_checkpoint(self, path: str):
        assert path == "/opt/pokebot/checkpoint/active.pt"
        return {"ok": True, "checkpoint_digest": DIGEST_A}

    def close(self) -> None:
        self.closed = True


def test_discover_backend_needs_only_ip_for_updated_worker(monkeypatch) -> None:
    monkeypatch.setattr(
        "poke_bot.remote_fleet_registry.RemoteJobClient", _FakeClient
    )
    discovered = discover_backend("10.0.0.42")
    assert discovered.entry["id"] == "aws-node-1"
    assert discovered.entry["endpoint"] == "10.0.0.42:8765"
    assert discovered.entry["capacity"] == 28
    assert discovered.entry["required_checkpoint_digest"] == DIGEST_A
    assert discovered.entry["checkpoint_paths"] == {
        DIGEST_A: "/opt/pokebot/checkpoint/active.pt"
    }


def test_add_batch_is_atomic_and_remove_is_selector_based(tmp_path: Path) -> None:
    path = tmp_path / "fleet.json"
    _manifest(path)
    result = add_backends(
        path,
        [
            DiscoveredBackend(
                entry=_entry("one", "10.0.0.1:8765"), health={"ok": True}
            ),
            DiscoveredBackend(
                entry=_entry("two", "10.0.0.2:8765"), health={"ok": True}
            ),
        ],
    )
    assert result["added"] == ["one", "two"]
    assert result["enabled_capacity"] == 8
    registry = json.loads(path.read_text(encoding="utf-8"))
    assert [row["id"] for row in registry["backends"]] == ["one", "two"]

    remove = remove_backends(path, ["10.0.0.2:8765"])
    assert remove["removed"] == ["two"]
    registry = json.loads(path.read_text(encoding="utf-8"))
    assert [row["id"] for row in registry["backends"]] == ["one"]


def test_invalid_batch_preserves_original_registry(tmp_path: Path) -> None:
    path = tmp_path / "fleet.json"
    _manifest(path)
    before = path.read_bytes()
    with pytest.raises(FleetManifestError, match="same required checkpoint"):
        add_backends(
            path,
            [
                DiscoveredBackend(
                    entry=_entry("one", "10.0.0.1:8765", DIGEST_A),
                    health={"ok": True},
                ),
                DiscoveredBackend(
                    entry=_entry("two", "10.0.0.2:8765", DIGEST_B),
                    health={"ok": True},
                ),
            ],
        )
    assert path.read_bytes() == before


def test_dry_run_does_not_replace_registry(tmp_path: Path) -> None:
    path = tmp_path / "fleet.json"
    _manifest(path)
    before = path.read_bytes()
    result = add_backends(
        path,
        [
            DiscoveredBackend(
                entry=_entry("one", "10.0.0.1:8765"), health={"ok": True}
            )
        ],
        dry_run=True,
    )
    assert result["dry_run"] is True
    assert path.read_bytes() == before
