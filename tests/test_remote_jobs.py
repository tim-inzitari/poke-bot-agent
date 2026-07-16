"""Unit tests for length-prefixed remote job framing (no network)."""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import pytest

import poke_bot.remote_jobs as remote_jobs
from poke_bot.remote_jobs import (
    PROTO_VERSION,
    RemoteJobClient,
    RemoteWorkerFarm,
    encode_frame,
    iter_additive_results,
    parse_endpoint,
    prepare_remote_play_job,
    read_frame,
    resolve_remote_checkpoint_path,
    resolve_remote_workdir_path,
    send_frame,
    serve_forever,
    split_jobs_additive,
)


def test_parse_endpoint():
    assert parse_endpoint("truenas.local") == ("truenas.local", 8765)
    assert parse_endpoint("truenas.local:9000") == ("truenas.local", 9000)
    assert parse_endpoint("tcp://192.168.1.143:8765") == ("192.168.1.143", 8765)


def test_resolve_remote_workdir_path_bert_and_elmo():
    local = "/home/inzi/poke-bot-agent/baselines/roster/iono/deck.csv"
    assert resolve_remote_workdir_path("bert.local", local) == (
        "/Users/tsinzitari/workspace/poke-bot-agent/baselines/roster/iono/deck.csv"
    )
    assert resolve_remote_workdir_path("192.168.1.143", local) == (
        "/workspace/baselines/roster/iono/deck.csv"
    )
    # Already-native paths are left alone.
    bert_native = (
        "/Users/tsinzitari/workspace/poke-bot-agent/baselines/official/iono"
    )
    assert resolve_remote_workdir_path("bert", bert_native) == bert_native


def test_prepare_remote_play_job_rewrites_spec_and_checkpoint():
    job = {
        "checkpoint": "/home/inzi/poke-bot-agent/outputs/checkpoints/x.pt",
        "spec": {
            "id": "iono",
            "path": "/home/inzi/poke-bot-agent/baselines/official/iono",
        },
        "seed": 1,
    }
    out = prepare_remote_play_job("bert.local", job)
    assert out["spec"]["path"] == (
        "/Users/tsinzitari/workspace/poke-bot-agent/baselines/official/iono"
    )
    assert out["checkpoint"] == (
        "/Users/tsinzitari/workspace/poke-bot-agent/outputs/checkpoints/x.pt"
    )
    # Original job must stay trainer-local for the local WorkerPool path.
    assert job["spec"]["path"].startswith("/home/inzi/")


def test_prepare_remote_play_job_elmo_stages_checkpoint(tmp_path, monkeypatch):
    """Elmo auto-pin needs /workspace/checkpoint/<basename>, not /workspace/outputs/..."""
    ckpt_dir = tmp_path / "outputs" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    src = ckpt_dir / "core.e32ce.pt"
    src.write_bytes(b"elmo-stage-bytes")
    smb = tmp_path / "smb_checkpoint"
    smb.mkdir()

    monkeypatch.setattr(remote_jobs, "_TRAIN_ROOT", tmp_path)
    monkeypatch.setattr(remote_jobs, "_smb_checkpoint_dir", lambda: smb)

    job = {
        "checkpoint": str(src),
        "spec": {
            "id": "iono",
            "path": str(tmp_path / "baselines" / "official" / "iono"),
        },
    }
    (tmp_path / "baselines" / "official" / "iono").mkdir(parents=True)
    out = prepare_remote_play_job("192.168.1.143", job)
    assert out["checkpoint"] == "/workspace/checkpoint/core.e32ce.pt"
    assert (smb / "core.e32ce.pt").read_bytes() == b"elmo-stage-bytes"
    assert out["spec"]["path"] == "/workspace/baselines/official/iono"


def test_resolve_remote_checkpoint_path_bert_stages_via_rsync(
    tmp_path, monkeypatch
) -> None:
    """Bert remap must stage digest .pt bytes (rsync), not only rewrite paths."""
    ckpt_dir = tmp_path / "outputs" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    src = ckpt_dir / "canary.digest.pt"
    src.write_bytes(b"fake-weights-bytes")

    staged: dict[str, Path] = {}

    def fake_stage(path: Path) -> str:
        staged["src"] = path
        rel = path.relative_to(remote_jobs._TRAIN_ROOT)
        return str(remote_jobs._BERT_ROOT / rel)

    monkeypatch.setattr(remote_jobs, "_TRAIN_ROOT", tmp_path)
    monkeypatch.setattr(remote_jobs, "_stage_bert_checkpoint", fake_stage)

    remote = resolve_remote_checkpoint_path("bert.local", str(src))
    assert staged["src"] == src.resolve()
    assert remote == str(
        remote_jobs._BERT_ROOT / "outputs" / "checkpoints" / "canary.digest.pt"
    )


def test_resolve_remote_checkpoint_path_bert_rsync_helper(monkeypatch, tmp_path) -> None:
    src = tmp_path / "a.pt"
    src.write_bytes(b"abc")
    remote_native = Path("/Users/tsinzitari/workspace/poke-bot-agent/outputs/a.pt")
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))

        class R:
            returncode = 0
            stdout = ""
            stderr = ""

        return R()

    monkeypatch.setattr(remote_jobs.subprocess, "run", fake_run)
    remote_jobs._rsync_to_bert(src, remote_native)
    assert any(cmd[0] == "ssh" for cmd in calls)
    rsync_cmds = [cmd for cmd in calls if cmd[0] == "rsync"]
    assert len(rsync_cmds) == 1
    assert str(src) in rsync_cmds[0]
    assert rsync_cmds[0][-1].endswith("/outputs/a.pt")


def test_split_jobs_additive_keeps_local_primary(monkeypatch):
    monkeypatch.delenv("POKEBOT_REMOTE_PRIMARY", raising=False)
    jobs = [{"job_index": i} for i in range(10)]
    local, remote = split_jobs_additive(
        jobs, local_workers=2, remote_workers=8
    )
    assert len(local) + len(remote) == 10
    assert len(local) == 2
    assert len(remote) == 8
    assert local[0]["job_index"] == 0
    assert remote[0]["job_index"] == 2


def test_split_jobs_additive_remote_primary(monkeypatch):
    monkeypatch.setenv("POKEBOT_REMOTE_PRIMARY", "1")
    jobs = [{"job_index": i} for i in range(36)]
    local, remote = split_jobs_additive(
        jobs, local_workers=6, remote_workers=30
    )
    assert len(local) + len(remote) == 36
    assert len(remote) == 30
    assert len(local) == 6
    # Remotes claim the leading slots so farms fill first.
    assert remote[0]["job_index"] == 0
    assert local[0]["job_index"] == 30


def test_split_jobs_additive_no_remote_returns_all_local():
    jobs = [{"job_index": 0}, {"job_index": 1}]
    local, remote = split_jobs_additive(jobs, local_workers=4, remote_workers=0)
    assert local == jobs
    assert remote == []


def test_iter_additive_fallback_uses_pool_apply(monkeypatch):
    """Remote slot failure must fall back via WorkerPool.apply, not the thread."""
    monkeypatch.delenv("POKEBOT_REMOTE_PRIMARY", raising=False)
    monkeypatch.delenv("POKEBOT_REMOTE_ONLY", raising=False)
    monkeypatch.delenv("POKEBOT_REMOTE_NO_LOCAL_FALLBACK", raising=False)
    applied = []

    class FakePool:
        def imap_unordered(self, fn, batch):
            return (fn(job) for job in batch)

        def apply(self, fn, job):
            applied.append(job["n"])
            return {"value": job["n"], "via": "pool"}

    class BoomClient:
        endpoint = "boom:1"
        info = None

        def submit_job(self, job, kind="play"):
            raise TimeoutError("simulated remote hang")

        def close(self):
            return None

    # Window size 1+2 → jobs [0]=local imap, [1],[2]=remote then fallback apply.
    jobs = [{"n": i} for i in range(3)]
    rows = list(
        iter_additive_results(
            local_pool=FakePool(),
            local_fn=lambda job: {"value": job["n"], "via": "imap"},
            jobs=jobs,
            remote_clients=[BoomClient()],
            local_workers=1,
            remote_workers=2,
        )
    )
    assert sorted(r["value"] for r in rows) == [0, 1, 2]
    assert applied == [1, 2]
    assert {r["via"] for r in rows if r["value"] in applied} == {"pool"}


def test_iter_additive_no_local_fallback_retries_remote(monkeypatch):
    """With NO_LOCAL_FALLBACK, failed remotes reconnect+retry — never pool.apply."""
    monkeypatch.setenv("POKEBOT_REMOTE_NO_LOCAL_FALLBACK", "1")
    monkeypatch.setenv("POKEBOT_REMOTE_JOB_RETRIES", "3")
    monkeypatch.delenv("POKEBOT_REMOTE_ONLY", raising=False)
    monkeypatch.delenv("POKEBOT_REMOTE_PRIMARY", raising=False)
    applied: list[int] = []
    submits = {"n": 0}

    class FakePool:
        def imap_unordered(self, fn, batch):
            return (fn(job) for job in batch)

        def apply(self, fn, job):
            applied.append(job["n"])
            return {"value": job["n"], "via": "pool"}

    class FlakyClient:
        endpoint = "flaky:1"
        info = None

        def submit_job(self, job, kind="play"):
            submits["n"] += 1
            if submits["n"] < 3:
                raise TimeoutError("slow peer")
            return {"value": job["n"], "via": "remote"}

        def reconnect(self):
            return None

        def close(self):
            return None

    rows = list(
        iter_additive_results(
            local_pool=FakePool(),
            local_fn=lambda job: {"value": job["n"], "via": "imap"},
            jobs=[{"n": 1}],
            remote_clients=[FlakyClient()],
            local_workers=0,
            remote_workers=2,
        )
    )
    assert rows == [{"value": 1, "via": "remote"}]
    assert applied == []
    assert submits["n"] == 3


def test_split_jobs_remote_only(monkeypatch):
    monkeypatch.setenv("POKEBOT_REMOTE_ONLY", "1")
    jobs = [{"n": i} for i in range(5)]
    local, remote = split_jobs_additive(jobs, local_workers=4, remote_workers=20)
    assert local == []
    assert remote == jobs


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


def test_iter_additive_results_opens_worker_many_sockets():
    """Advertised workers=N must become N concurrent sockets, not 1 serial."""
    stop = threading.Event()
    peak_inflight = {"n": 0}
    inflight = {"n": 0}
    lock = threading.Lock()
    saw_n = threading.Barrier(4, timeout=8.0)

    def hello():
        return {
            "hostname": "test-worker",
            "workers": 4,
            "leaf_servers": 1,
            "gpu_name": "fake",
            "device": "cpu",
            "checkpoint_digest": "abc",
        }

    def handler(msg):
        if msg.get("type") == "health":
            return {"type": "health_ok", "ok": True, **hello()}
        if msg.get("type") == "job":
            with lock:
                inflight["n"] += 1
                peak_inflight["n"] = max(peak_inflight["n"], inflight["n"])
            try:
                # Block until all 4 worker sockets are mid-job.
                saw_n.wait()
            except threading.BrokenBarrierError:
                pass
            time.sleep(0.02)
            with lock:
                inflight["n"] -= 1
            return {
                "type": "result",
                "ok": True,
                "result": {"value": msg["job"]["n"]},
            }
        return {"type": "error", "error": "bad"}

    srv = threading.Thread(
        target=serve_forever,
        kwargs=dict(
            handler=handler,
            host="127.0.0.1",
            port=18767,
            hello=hello,
            stop_event=stop,
        ),
        daemon=True,
    )
    srv.start()
    try:
        time.sleep(0.05)
        with RemoteJobClient("127.0.0.1", 18767, timeout_s=5.0) as client:
            assert client.info is not None
            assert client.info.workers == 4

            class FakePool:
                def imap_unordered(self, fn, batch):
                    return (fn(job) for job in batch)

            # local_workers=1 + remote_workers=4 → most jobs remote; 12 remote
            # slots fill all 4 sockets concurrently.
            jobs = [{"n": i} for i in range(20)]
            rows = list(
                iter_additive_results(
                    local_pool=FakePool(),
                    local_fn=lambda job: {"value": job["n"]},
                    jobs=jobs,
                    remote_clients=[client],
                    local_workers=1,
                    remote_workers=4,
                )
            )
            assert peak_inflight["n"] >= 4
            assert len(rows) == 20
            assert sorted(row["value"] for row in rows) == list(range(20))
    finally:
        stop.set()
        srv.join(timeout=3)


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


def test_serve_forever_idle_timeout_keeps_session():
    """Idle farm sockets must survive recv timeouts between waves."""
    stop = threading.Event()

    def hello():
        return {
            "hostname": "test-worker",
            "workers": 1,
            "leaf_servers": 1,
            "gpu_name": "test",
            "device": "cpu",
            "checkpoint_digest": "abc",
        }

    def handler(msg):
        if msg.get("type") == "health":
            return {"type": "health_ok", "ok": True, **hello()}
        if msg.get("type") == "job":
            return {"type": "result", "ok": True, "result": {"value": msg["job"]["n"]}}
        return {"type": "error", "error": "bad"}

    srv = threading.Thread(
        target=serve_forever,
        kwargs=dict(
            handler=handler,
            host="127.0.0.1",
            port=18768,
            hello=hello,
            stop_event=stop,
            idle_timeout_s=0.15,
        ),
        daemon=True,
    )
    srv.start()
    try:
        time.sleep(0.05)
        with RemoteJobClient("127.0.0.1", 18768, timeout_s=5.0) as client:
            assert client.submit_job({"n": 1}, kind="play")["value"] == 1
            # Exceed server idle recv timeout several times; session must live.
            time.sleep(0.55)
            assert client.health()["ok"] is True
            assert client.submit_job({"n": 2}, kind="play")["value"] == 2
    finally:
        stop.set()
        srv.join(timeout=3)


def test_submit_job_reconnects_once_on_connection_closed():
    """Idle-dead farm sockets must reconnect once instead of failing the wave."""
    stop = threading.Event()
    hellos = {"n": 0}

    def hello():
        hellos["n"] += 1
        return {
            "hostname": "test-worker",
            "workers": 1,
            "leaf_servers": 1,
            "gpu_name": "fake",
            "device": "cpu",
            "checkpoint_digest": "sha256:x",
        }

    def handler(msg):
        assert msg.get("type") == "job"
        return {"type": "result", "ok": True, "result": {"value": msg["job"]["n"]}}

    srv = threading.Thread(
        target=serve_forever,
        kwargs=dict(
            handler=handler,
            host="127.0.0.1",
            port=18769,
            hello=hello,
            stop_event=stop,
            idle_timeout_s=30.0,
        ),
        daemon=True,
    )
    srv.start()
    try:
        time.sleep(0.05)
        client = RemoteJobClient("127.0.0.1", 18769, timeout_s=5.0)
        client.connect()
        assert hellos["n"] == 1
        # Simulate an idle-dead peer: replace the live socket with a half-closed
        # socketpair so the next send/recv raises connection-closed.
        a, b = socket.socketpair()
        b.close()
        old = client._sock
        client._sock = a
        if old is not None:
            try:
                old.close()
            except Exception:
                pass
        assert client.submit_job({"n": 7}, kind="play")["value"] == 7
        assert hellos["n"] >= 2
        client.close()
    finally:
        stop.set()
        srv.join(timeout=3)


def test_ensure_alive_rehydrates_template():
    stop = threading.Event()

    def hello():
        return {
            "hostname": "test-worker",
            "workers": 2,
            "leaf_servers": 1,
            "gpu_name": "fake",
            "device": "cpu",
            "checkpoint_digest": "sha256:x",
        }

    def handler(msg):
        return {"type": "result", "ok": True, "result": {"ok": True}}

    srv = threading.Thread(
        target=serve_forever,
        kwargs=dict(
            handler=handler,
            host="127.0.0.1",
            port=18770,
            hello=hello,
            stop_event=stop,
        ),
        daemon=True,
    )
    srv.start()
    try:
        time.sleep(0.05)
        client = RemoteJobClient("127.0.0.1", 18770, timeout_s=5.0)
        client.connect()
        a, b = socket.socketpair()
        b.close()
        old = client._sock
        client._sock = a
        if old is not None:
            try:
                old.close()
            except Exception:
                pass
        client.ensure_alive()
        client.ping()
        client.close()
    finally:
        stop.set()
        srv.join(timeout=3)


def test_ensure_alive_ignores_timeout_as_slow_alive(monkeypatch):
    """TimeoutError must not tear down a slow-but-alive farm template."""
    client = RemoteJobClient("127.0.0.1", 9, timeout_s=1.0, control_timeout_s=0.2)
    client._sock = object()  # type: ignore[assignment]
    client.info = None
    calls = {"reconnect": 0}

    def boom_ping():
        raise TimeoutError("slow peer")

    def boom_reconnect():
        calls["reconnect"] += 1
        raise AssertionError("reconnect must not run on TimeoutError")

    monkeypatch.setattr(client, "ping", boom_ping)
    monkeypatch.setattr(client, "reconnect", boom_reconnect)
    client.ensure_alive()
    assert calls["reconnect"] == 0
