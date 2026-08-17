"""Static safety checks for the authenticated dashboard inspector gateway."""

from __future__ import annotations

import plistlib
from pathlib import Path
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parents[1]


def test_inspector_assets_and_api_are_prefix_relative() -> None:
    index = (ROOT / "replay_inspector/web/index.html").read_text(encoding="utf-8")
    app = (ROOT / "replay_inspector/web/app.js").read_text(encoding="utf-8")

    assert 'href="./styles.css"' in index
    assert 'src="./app.js"' in index
    assert 'const API = "./api";' in app
    assert urljoin("http://127.0.0.1:8791/", "./api/health") == (
        "http://127.0.0.1:8791/api/health"
    )
    assert (
        urljoin("https://mc.tsinzitari.com/replay-inspector/", "./api/health")
        == "https://mc.tsinzitari.com/replay-inspector/api/health"
    )


def test_caddy_gateway_is_fixed_read_only_plus_bounded_manual_sync() -> None:
    config = (ROOT / "deploy/caddy/Caddyfile").read_text(encoding="utf-8")

    assert "path /replay-inspector/*" in config
    assert "not method GET" in config
    assert "handle_path /replay-inspector/*" in config
    assert "reverse_proxy 127.0.0.1:8792" in config
    assert "header_up Host 127.0.0.1:8791" in config
    for header in ("Authorization", "Cookie", "Origin", "Referer", "Forwarded"):
        assert f"header_up -{header}" in config
    assert "reverse_proxy 127.0.0.1:8780" in config
    assert "path /replay-inspector/api/sync" in config
    assert "method POST" in config
    assert "header X-Replay-Sync-Intent manual" in config
    assert "path /replay-inspector/api/sync-status" in config
    assert "0.0.0.0:879" not in config


def test_launchd_tunnel_binds_only_bert_loopback_to_elmo_loopback() -> None:
    path = ROOT / "deploy/launchd/com.pokebot.replay-model-inspector-tunnel.plist"
    with path.open("rb") as handle:
        payload = plistlib.load(handle)
    arguments = payload["ProgramArguments"]

    assert payload["Label"] == "com.pokebot.replay-model-inspector-tunnel"
    assert payload["RunAtLoad"] is True
    assert payload["KeepAlive"] is True
    assert "BatchMode=yes" in arguments
    assert "StrictHostKeyChecking=yes" in arguments
    assert "ExitOnForwardFailure=yes" in arguments
    assert "127.0.0.1:8792:127.0.0.1:8791" in arguments
    assert arguments[-1] == "admin@elmo"
    assert all("0.0.0.0" not in item for item in arguments)
