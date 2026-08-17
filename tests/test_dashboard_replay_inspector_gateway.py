"""Focused integration coverage for the LAN replay-inspector bridge."""

from __future__ import annotations

import http.client
import json
import subprocess
import threading
import time
from collections.abc import Iterator
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from dashboard.lan import server as dashboard_server


@pytest.fixture
def gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[int, dict[str, list[dict[str, Any]]]]]:
    observed: dict[str, list[dict[str, Any]]] = {"requests": []}

    class Upstream(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            observed["requests"].append(
                {
                    "path": self.path,
                    "headers": {
                        key.lower(): value for key, value in self.headers.items()
                    },
                }
            )
            if self.path == "/redirect":
                self.send_response(HTTPStatus.FOUND)
                self.send_header("Location", "https://example.invalid/")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if self.path == "/oversize":
                self.send_response(HTTPStatus.OK)
                self.send_header(
                    "Content-Length",
                    str(dashboard_server.INSPECTOR_PROXY_MAX_RESPONSE_BYTES + 1),
                )
                self.end_headers()
                return
            if self.path in {
                "/delayed-non-trace",
                "/api/submissions/987/games/654/steps/3?stage=0",
            }:
                # The trace target deliberately waits past the test's legacy
                # generic upstream deadline. Non-trace resources must not.
                time.sleep(0.075)
            body = json.dumps({"upstream_path": self.path}).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Set-Cookie", "upstream=must-not-pass")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                # The bounded non-trace gateway intentionally closed first.
                return

        def log_message(self, _fmt: str, *_args: object) -> None:
            return

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    monkeypatch.setattr(
        dashboard_server,
        "INSPECTOR_UPSTREAM_ADDRESS",
        ("127.0.0.1", upstream.server_address[1]),
    )

    bridge = dashboard_server.DashboardHTTPServer(
        ("127.0.0.1", 0), dashboard_server.Handler
    )
    bridge_thread = threading.Thread(target=bridge.serve_forever, daemon=True)
    bridge_thread.start()
    try:
        yield bridge.server_address[1], observed
    finally:
        bridge.shutdown()
        bridge.server_close()
        bridge_thread.join(timeout=2)
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=2)


def _request(
    port: int,
    method: str,
    target: str,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request(method, target, headers=headers or {})
        response = connection.getresponse()
        return (
            response.status,
            {key.lower(): value for key, value in response.getheaders()},
            response.read(),
        )
    finally:
        connection.close()


def test_gateway_strips_prefix_and_browser_headers(
    gateway: tuple[int, dict[str, list[dict[str, Any]]]],
) -> None:
    port, observed = gateway
    status, headers, body = _request(
        port,
        "GET",
        "/replay-inspector/api/health?stage=0",
        {
            "Authorization": "Basic opaque-browser-credential",
            "Cookie": "dashboard_session=opaque",
            "Origin": f"http://127.0.0.1:{port}",
            "Referer": f"http://127.0.0.1:{port}/",
            "Forwarded": "for=192.0.2.1",
            "X-Forwarded-For": "192.0.2.1",
            "Sec-Fetch-Site": "same-origin",
        },
    )

    assert status == HTTPStatus.OK
    assert json.loads(body) == {"upstream_path": "/api/health?stage=0"}
    assert headers["content-type"] == "application/json; charset=utf-8"
    assert "access-control-allow-origin" not in headers
    assert "set-cookie" not in headers
    assert "location" not in headers

    forwarded = observed["requests"]
    assert len(forwarded) == 1
    assert forwarded[0]["path"] == "/api/health?stage=0"
    assert forwarded[0]["headers"]["host"] == "127.0.0.1:8791"
    for header in (
        "authorization",
        "cookie",
        "origin",
        "referer",
        "forwarded",
        "x-forwarded-for",
        "sec-fetch-site",
    ):
        assert header not in forwarded[0]["headers"]

    dashboard_status, _headers, _body = _request(port, "GET", "/api/status")
    assert dashboard_status == HTTPStatus.OK
    assert len(observed["requests"]) == 1

    status, headers, _body = _request(port, "GET", "/replay-inspector")
    assert status == HTTPStatus.PERMANENT_REDIRECT
    assert headers["location"] == "/replay-inspector/"


def test_gateway_rejects_cross_site_and_non_get_before_upstream(
    gateway: tuple[int, dict[str, list[dict[str, Any]]]],
) -> None:
    port, observed = gateway

    status, _headers, _body = _request(
        port,
        "GET",
        "/replay-inspector/api/health",
        {"Sec-Fetch-Site": "cross-site"},
    )
    assert status == HTTPStatus.FORBIDDEN

    status, _headers, _body = _request(port, "POST", "/replay-inspector/api/health")
    assert status == HTTPStatus.METHOD_NOT_ALLOWED
    assert _headers["allow"] == "GET"
    assert observed["requests"] == []


def test_gateway_allows_only_bounded_manual_replay_sync(
    gateway: tuple[int, dict[str, list[dict[str, Any]]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    port, observed = gateway
    commands: list[list[str]] = []

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        stdout = "active\n" if "is-active" in command else ""
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(dashboard_server.subprocess, "run", fake_run)

    status, _headers, body = _request(
        port,
        "POST",
        dashboard_server.INSPECTOR_MANUAL_SYNC_PATH,
        {dashboard_server.INSPECTOR_MANUAL_SYNC_HEADER: "manual"},
    )
    assert status == HTTPStatus.ACCEPTED
    assert json.loads(body)["accepted"] is True
    assert commands[-1][-3:] == [
        "start",
        "--no-block",
        dashboard_server.ELMO_REPLAY_SYNC_SERVICE,
    ]

    status, _headers, body = _request(
        port, "GET", dashboard_server.INSPECTOR_MANUAL_SYNC_STATUS_PATH
    )
    assert status == HTTPStatus.OK
    assert json.loads(body)["running"] is True
    assert "is-active" in commands[-1]

    status, _headers, _body = _request(
        port, "POST", dashboard_server.INSPECTOR_MANUAL_SYNC_PATH
    )
    assert status == HTTPStatus.FORBIDDEN
    assert len(commands) == 2
    assert observed["requests"] == []


def test_gateway_rejects_unsafe_targets_and_get_bodies(
    gateway: tuple[int, dict[str, list[dict[str, Any]]]],
) -> None:
    port, observed = gateway
    cases = (
        ("/replay-inspector/%2fapi/health", HTTPStatus.BAD_REQUEST),
        ("/replay-inspector/%2e%2e/api/health", HTTPStatus.BAD_REQUEST),
        ("/replay-inspector/api/%00health", HTTPStatus.BAD_REQUEST),
        ("/replay-inspector/api/health?query=%00", HTTPStatus.BAD_REQUEST),
        (
            "/replay-inspector/"
            + "a" * dashboard_server.INSPECTOR_PROXY_MAX_TARGET_BYTES,
            HTTPStatus.REQUEST_URI_TOO_LONG,
        ),
        ("http://example.invalid/replay-inspector/api/health", HTTPStatus.BAD_REQUEST),
    )
    for target, expected in cases:
        status, _headers, _body = _request(port, "GET", target)
        assert status == expected

    status, _headers, _body = _request(
        port,
        "GET",
        "/replay-inspector/api/health",
        {"Content-Length": "1"},
    )
    assert status == HTTPStatus.BAD_REQUEST
    assert observed["requests"] == []

    with pytest.raises(dashboard_server.InspectorProxyTargetError) as exc_info:
        dashboard_server._validate_request_target("/replay-inspector/api/\x7fhealth")
    assert exc_info.value.status == HTTPStatus.BAD_REQUEST


def test_gateway_rejects_upstream_redirects_and_oversized_responses(
    gateway: tuple[int, dict[str, list[dict[str, Any]]]],
) -> None:
    port, observed = gateway

    status, headers, _body = _request(port, "GET", "/replay-inspector/redirect")
    assert status == HTTPStatus.BAD_GATEWAY
    assert "location" not in headers

    status, _headers, _body = _request(port, "GET", "/replay-inspector/oversize")
    assert status == HTTPStatus.BAD_GATEWAY
    assert [request["path"] for request in observed["requests"]] == [
        "/redirect",
        "/oversize",
    ]


def test_trace_wait_is_unbounded_but_non_trace_wait_remains_bounded(
    gateway: tuple[int, dict[str, list[dict[str, Any]]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only exact trace headers bypass the legacy generic upstream deadline."""

    port, observed = gateway
    monkeypatch.setattr(dashboard_server, "INSPECTOR_PROXY_TIMEOUT_SECONDS", 0.02)
    monkeypatch.setattr(
        dashboard_server, "INSPECTOR_PROXY_CONNECT_TIMEOUT_SECONDS", 0.02
    )

    started = time.monotonic()
    status, _headers, body = _request(
        port,
        "GET",
        "/replay-inspector/api/submissions/987/games/654/steps/3?stage=0",
    )
    elapsed = time.monotonic() - started
    assert status == HTTPStatus.OK
    assert json.loads(body)["upstream_path"].endswith("steps/3?stage=0")
    assert elapsed >= 0.05

    status, _headers, _body = _request(
        port, "GET", "/replay-inspector/delayed-non-trace"
    )
    assert status == HTTPStatus.GATEWAY_TIMEOUT
    assert [request["path"] for request in observed["requests"]] == [
        "/api/submissions/987/games/654/steps/3?stage=0",
        "/delayed-non-trace",
    ]


def test_trace_waiter_capacity_rejects_before_upstream(
    gateway: tuple[int, dict[str, list[dict[str, Any]]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    port, observed = gateway
    monkeypatch.setattr(
        dashboard_server,
        "_INSPECTOR_TRACE_WAITERS",
        threading.BoundedSemaphore(0),
    )

    status, _headers, _body = _request(
        port,
        "GET",
        "/replay-inspector/api/submissions/987/games/654/steps/3?stage=0",
    )
    assert status == HTTPStatus.SERVICE_UNAVAILABLE
    assert observed["requests"] == []


def test_trace_wait_policy_matches_only_exact_trace_targets() -> None:
    assert dashboard_server._is_inspector_trace_target(
        "/api/submissions/987/games/654/steps/3?stage=0"
    )
    assert dashboard_server._is_inspector_trace_target(
        "/api/submissions/987/games/654/steps/3?stage=0&scales=value:0.5"
    )
    assert not dashboard_server._is_inspector_trace_target(
        "/api/submissions/987/games/654/steps"
    )
    assert not dashboard_server._is_inspector_trace_target(
        "/api/submissions/987/games/654/parameters"
    )
    assert dashboard_server.INSPECTOR_PROXY_CONNECT_TIMEOUT_SECONDS > 0
    assert dashboard_server.INSPECTOR_PROXY_TRACE_RESPONSE_READ_TIMEOUT_SECONDS > 0
    assert dashboard_server.INSPECTOR_PROXY_MAX_PENDING_TRACE_REQUESTS > 0


def test_gateway_rejects_dns_rebinding_and_conflicting_origins(
    gateway: tuple[int, dict[str, list[dict[str, Any]]]],
) -> None:
    port, observed = gateway
    status, _headers, _body = _request(
        port,
        "GET",
        "/replay-inspector/api/health",
        {"Host": "public-name.example"},
    )
    assert status == HTTPStatus.FORBIDDEN

    status, _headers, _body = _request(
        port,
        "GET",
        "/replay-inspector/api/health",
        {"Origin": f"http://localhost:{port}"},
    )
    assert status == HTTPStatus.FORBIDDEN
    assert observed["requests"] == []


def test_gateway_only_trusts_local_or_tailscale_peers() -> None:
    assert dashboard_server.INSPECTOR_UPSTREAM_ADDRESS == ("127.0.0.1", 8792)
    assert dashboard_server._private_or_loopback_address("127.0.0.1")
    assert dashboard_server._private_or_loopback_address("192.168.1.40")
    assert dashboard_server._private_or_loopback_address("192.168.1.160")
    assert dashboard_server._private_or_loopback_address("fd00::5")
    assert dashboard_server._private_or_loopback_address(
        "fe80::4fd:7b09:c7d6:916d%en0"
    )
    assert dashboard_server._private_or_loopback_address("100.64.0.0")
    assert dashboard_server._private_or_loopback_address("100.106.229.91")
    assert dashboard_server._private_or_loopback_address("100.127.255.255")
    assert not dashboard_server._private_or_loopback_address("100.63.255.255")
    assert not dashboard_server._private_or_loopback_address("100.128.0.0")
    assert not dashboard_server._private_or_loopback_address("8.8.8.8")
    assert not dashboard_server._private_or_loopback_address("dashboard.example")
    assert dashboard_server._local_dashboard_authority("bert.local:8780") == (
        "bert.local",
        8780,
    )
    assert dashboard_server._local_dashboard_authority("100.106.229.91:8780") == (
        "100.106.229.91",
        8780,
    )
    assert dashboard_server._local_dashboard_authority("[::1]:8780") == (
        "::1",
        8780,
    )
    assert dashboard_server._local_dashboard_authority("public-name.example") is None
