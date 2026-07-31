"""Python binding smoke tests — no poke_bot imports."""

from __future__ import annotations

import threading
import time

import pytest

import wave_dispatch as wd


def test_version_and_constants():
    assert wd.__version__
    assert wd.PROTO_VERSION == 1
    assert wd.DEFAULT_PORT == 8765


def test_encode_decode_frame():
    payload = {"type": "hello", "proto": 1}
    raw = wd.encode_frame(payload)
    assert isinstance(raw, (bytes, bytearray))
    back = wd.decode_frame(raw)
    assert back["type"] == "hello"
    assert back["proto"] == 1


def test_binary_message_roundtrip():
    raw = wd.encode_message({"type": "job", "job": {"id": 3}}, b"\x01\x02\x03\x04")
    msg = wd.decode_message(raw)
    assert msg["meta"]["job"]["id"] == 3
    assert msg["blob"] == b"\x01\x02\x03\x04"


def test_scheduler_basic():
    cfg = wd.SchedulerConfig()
    cfg.tick_s = 0.0
    cfg.remote_defaults = {"a:1": 2}
    cfg.remote_maxima = {"a:1": 8}
    sched = wd.MidWaveScheduler(cfg)
    d = sched.decision()
    assert d.remote_demand["a:1"] == 2
    sched.note_completed("local", 5, 5)
    tick = sched.maybe_tick(10, force=True)
    assert tick is not None
    assert "wave_gps" in tick.metrics


def test_tcp_roundtrip():
    stop = threading.Event()
    port = 19765
    cfg = wd.ServerConfig()
    cfg.host = "127.0.0.1"
    cfg.port = port
    cfg.idle_timeout_s = 2.0

    def hello():
        return {
            "workers": 2,
            "max_workers": 4,
            "default_workers": 2,
            "hostname": "py-test",
        }

    def handler(msg):
        if msg.get("type") == "job":
            return {
                "type": "result",
                "ok": True,
                "result": {"ok": True, "id": msg["job"]["id"]},
            }
        return {"type": "error", "error": "bad"}

    t = threading.Thread(
        target=wd.serve_forever,
        kwargs={"handler": handler, "config": cfg, "hello": hello, "stop_event": stop},
        daemon=True,
    )
    t.start()
    time.sleep(0.2)
    try:
        client = wd.JobClient("127.0.0.1", port, timeout_s=5.0, connect_timeout_s=5.0)
        info = client.connect()
        assert info.workers == 2
        assert client.ping()["type"] == "pong"
        result = client.submit_job({"id": 7}, kind="echo")
        assert result["id"] == 7
        client.close()
    finally:
        stop.set()
        t.join(timeout=3.0)


def test_run_scheduled_wave_local_only():
    cfg = wd.SchedulerConfig()
    cfg.tick_s = 0.0
    sched = wd.MidWaveScheduler(cfg)
    jobs = [{"id": i} for i in range(8)]
    results = []

    def local(job):
        return {"ok": True, "id": job["id"], "src": "local"}

    n = wd.run_scheduled_wave(
        jobs,
        local,
        [],
        sched,
        wd.CollectConfig(),
        on_result=results.append,
    )
    assert n == 8
    assert len(results) == 8
