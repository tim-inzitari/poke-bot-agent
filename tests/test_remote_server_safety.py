import socket
import threading
import time

import pytest

import poke_bot.remote_jobs as remote_jobs
from poke_bot.remote_jobs import (
    PROTO_VERSION,
    RemoteJobsError,
    read_frame,
    send_frame,
    serve_forever,
)


def test_fast_and_stdlib_json_frames_are_wire_compatible(monkeypatch) -> None:
    """Changing codecs must not change any training payload semantics."""
    fast_codec = remote_jobs._orjson
    if fast_codec is None:
        pytest.skip("orjson is optional")
    payload = {
        "job_index": 7,
        "record_jsons": ['{"steps":[{"action":3,"value":-1.0}]}'],
        "unicode": "Alakazam Ψ",
        "integer_key_map": {1: "one"},
    }

    monkeypatch.setattr(remote_jobs, "_orjson", None)
    std_frame = remote_jobs.encode_frame(payload)
    monkeypatch.setattr(remote_jobs, "_orjson", fast_codec)
    left, right = socket.socketpair()
    try:
        left.sendall(std_frame)
        assert remote_jobs.read_frame(right) == {
            **payload,
            "integer_key_map": {"1": "one"},
        }
    finally:
        left.close()
        right.close()

    fast_frame = remote_jobs.encode_frame(payload)
    monkeypatch.setattr(remote_jobs, "_orjson", None)
    left, right = socket.socketpair()
    try:
        left.sendall(fast_frame)
        assert remote_jobs.read_frame(right) == {
            **payload,
            "integer_key_map": {"1": "one"},
        }
    finally:
        left.close()
        right.close()


def _unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _connect(port: int, *, timeout: float = 2.0) -> socket.socket:
    deadline = time.monotonic() + timeout
    while True:
        try:
            sock = socket.create_connection(("127.0.0.1", port), timeout=0.25)
            sock.settimeout(1.0)
            return sock
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


def _hello(sock: socket.socket) -> dict:
    send_frame(sock, {"type": "hello", "proto": PROTO_VERSION})
    return read_frame(sock)


def _server(port: int, stop: threading.Event, *, max_connections: int) -> None:
    serve_forever(
        lambda msg: {"type": "ok", "echo": msg},
        host="127.0.0.1",
        port=port,
        hello=lambda: {"workers": 1},
        stop_event=stop,
        idle_timeout_s=0.05,
        max_connections=max_connections,
    )


def test_tcp_only_probe_does_not_poison_single_connection_slot() -> None:
    port = _unused_port()
    stop = threading.Event()
    thread = threading.Thread(target=_server, args=(port, stop), kwargs={"max_connections": 1})
    thread.start()
    try:
        probe = _connect(port)
        probe.close()

        deadline = time.monotonic() + 2.0
        while True:
            client = _connect(port)
            try:
                reply = _hello(client)
                break
            except (OSError, RemoteJobsError):
                client.close()
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)
        assert reply["type"] == "hello_ok"
        client.close()
    finally:
        stop.set()
        thread.join(timeout=3.0)
    assert not thread.is_alive()


def test_connection_thread_count_is_hard_bounded() -> None:
    port = _unused_port()
    stop = threading.Event()
    thread = threading.Thread(target=_server, args=(port, stop), kwargs={"max_connections": 1})
    thread.start()
    first = _connect(port)
    assert _hello(first)["type"] == "hello_ok"
    try:
        rejected = _connect(port)
        with pytest.raises((OSError, RemoteJobsError)):
            _hello(rejected)
        rejected.close()
    finally:
        stop.set()
        first.close()
        thread.join(timeout=3.0)
    assert not thread.is_alive()
