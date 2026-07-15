"""Unit tests for length-prefixed remote job framing (no network)."""

from __future__ import annotations

import socket
import threading

import pytest

from poke_bot.remote_jobs import (
    PROTO_VERSION,
    RemoteJobClient,
    RemoteWorkerFarm,
    encode_frame,
    iter_additive_results,
    parse_endpoint,
    read_frame,
    send_frame,
    serve_forever,
    split_jobs_additive,
)


def test_parse_endpoint():
    assert parse_endpoint("truenas.local") == ("truenas.local", 8765)
    assert parse_endpoint("truenas.local:9000") == ("truenas.local", 9000)
    assert parse_endpoint("tcp://192.168.1.143:8765") == ("192.168.1.143", 8765)


def test_split_jobs_additive_keeps_local_primary():
    jobs = [{"job_index": i} for i in range(10)]
    local, remote = split_jobs_additive(
        jobs, local_workers=2, remote_workers=8
    )
    assert len(local) + len(remote) == 10
    assert len(local) == 2
    assert len(remote) == 8
    assert local[0]["job_index"] == 0
    assert remote[0]["job_index"] == 2


def test_split_jobs_additive_no_remote_returns_all_local():
    jobs = [{"job_index": 0}, {"job_index": 1}]
    local, remote = split_jobs_additive(jobs, local_workers=4, remote_workers=0)
    assert local == jobs
    assert remote == []


def test_iter_additive_results_local_only_matches_pool():
    class FakePool:
        def imap_unordered(self, fn, batch):
            return (fn(job) for job in batch)

    jobs = [{"n": 1}, {"n": 2}]
    rows = list(
        iter_additive_results(
            local_pool=FakePool(),
            local_fn=lambda job: {"ok": job["n"]},
            jobs=jobs,
            remote_clients=[],
            local_workers=2,
            remote_workers=0,
        )
    )
    assert sorted(row["ok"] for row in rows) == [1, 2]


def test_remote_worker_farm_connect_and_pin_protocol():
    stop = threading.Event()

    def hello():
        return {
            "hostname": "test-worker",
            "workers": 4,
            "leaf_servers": 1,
            "gpu_name": "NVIDIA GeForce RTX 3060 Ti",
            "device": "cuda:0",
            "checkpoint_digest": "abc",
        }

    def handler(msg):
        if msg.get("type") == "health":
            return {"type": "health_ok", "ok": True, **hello()}
        if msg.get("type") == "pin":
            return {
                "type": "pin_ok",
                "ok": True,
                "checkpoint_digest": msg.get("digest") or "abc",
                "pinned_digests": ["abc", "def"],
            }
        if msg.get("type") == "job":
            return {"type": "result", "ok": True, "result": {"value": 2.0}}
        return {"type": "error", "error": "bad"}

    srv = threading.Thread(
        target=serve_forever,
        kwargs=dict(
            handler=handler,
            host="127.0.0.1",
            port=18766,
            hello=hello,
            stop_event=stop,
        ),
        daemon=True,
    )
    srv.start()
    try:
        with RemoteWorkerFarm(["127.0.0.1:18766"], timeout_s=5.0) as farm:
            assert farm.total_workers == 4
            pin = farm.pin_all("/tmp/candidate.pt", digest="def")
            assert pin[0]["ok"] is True
            result = farm.clients[0].submit_job({"seed": 2}, kind="play")
            assert result["value"] == 2.0
    finally:
        stop.set()
        srv.join(timeout=3)


def test_encode_roundtrip_local_socketpair():
    a, b = socket.socketpair()
    try:
        send_frame(a, {"type": "ping", "n": 1})
        msg = read_frame(b)
        assert msg == {"type": "ping", "n": 1}
        raw = encode_frame({"ok": True})
        assert len(raw) == 4 + len(raw[4:])
    finally:
        a.close()
        b.close()


def test_serve_hello_ping_health():
    stop = threading.Event()

    def hello():
        return {
            "hostname": "test-worker",
            "workers": 20,
            "leaf_servers": 2,
            "gpu_name": "NVIDIA GeForce RTX 3060 Lite Hash Rate",
            "device": "cuda:0",
            "checkpoint_digest": "abc",
            "free_ram_gb": 64.0,
        }

    def handler(msg):
        if msg.get("type") == "health":
            return {"type": "health_ok", "ok": True, "leaf_alive": True, **hello()}
        if msg.get("type") == "job":
            return {"type": "result", "ok": True, "result": {"value": 1.0}}
        return {"type": "error", "error": "bad"}

    srv = threading.Thread(
        target=serve_forever,
        kwargs=dict(
            handler=handler,
            host="127.0.0.1",
            port=18765,
            hello=hello,
            stop_event=stop,
        ),
        daemon=True,
    )
    srv.start()
    try:
        with RemoteJobClient("127.0.0.1", 18765, timeout_s=5.0) as client:
            assert client.info is not None
            assert client.info.workers == 20
            assert "3060" in client.info.gpu_name
            assert client.ping()["type"] == "pong"
            health = client.health()
            assert health["ok"] is True
            result = client.submit_job({"seed": 1}, kind="play")
            assert result["value"] == 1.0
    finally:
        stop.set()
        srv.join(timeout=3)
