from __future__ import annotations

import os
from pathlib import Path

import pytest

rl_io = pytest.importorskip("rl_io")
rl_runtime = pytest.importorskip("rl_runtime")


def test_ordered_writer_native(tmp_path: Path):
    partial = tmp_path / "replay.jsonl"
    w = rl_io.OrderedWriter(str(partial), expected_jobs=2, fsync_batch=1)
    assert w.submit(1, '{"episode_id":"1"}', {"i": 1})
    assert w.submit(0, '{"episode_id":"0"}', {"i": 0})
    tel = w.close()
    assert tel["next_index"] == 2
    w.finalize(str(tmp_path / "final.jsonl"))
    assert (tmp_path / "final.jsonl").is_file()


def test_blob_pack_native(tmp_path: Path):
    path = tmp_path / "t.rlpk"
    w = rl_io.BlobPackWriter()
    w.set_manifest({"schema": "t"})
    w.add("x", b"abc")
    w.commit(str(path))
    r = rl_io.BlobPackReader(str(path))
    assert r.get("x") == b"abc"


def test_shm_ring_native():
    name = f"/rl_py_shm_{os.getpid()}"
    cfg = rl_runtime.RingConfig()
    cfg.name = name
    cfg.slot_count = 1
    cfg.request_slots = 8
    cfg.max_payload = 256
    server = rl_runtime.ShmRing.create(cfg)
    client = rl_runtime.ShmRing.open(name)
    rid = client.submit(0, b"ping")
    req = server.pop(1.0)
    assert req is not None
    server.respond(req.slot, req.rid, b"pong")
    assert client.wait(0, rid) == b"pong"
    server.unlink()
