from __future__ import annotations

import json
import time
from pathlib import Path

from rl_resource import Knob, OomGuard, ResourcePlan, is_cuda_oom, ratchet_step
from schedule_engine import epoch_plan, schedule_digest, validate_schedule
from submit_guard import AUTH_SCHEMA, validate_authorization
from test_profiles import ProfileRunner, load_manifest
from log_trim import trim_directory


def test_oom_and_ratchet(tmp_path: Path):
    assert is_cuda_oom(RuntimeError("CUDA out of memory"))
    assert not is_cuda_oom(RuntimeError("other"))
    g = OomGuard(min_scale=0.25)
    assert g.scaled(8) == 8
    assert g.handle_oom(RuntimeError("cuda out of memory"))
    assert g.scale == 0.5
    plan = ResourcePlan(
        knobs={"w": Knob("W", 10, 2, 20, 4)},
        hysteresis=2,
        min_bump_interval_s=0.0,
    )
    plan = ratchet_step(
        plan,
        now=1.0,
        ram={"used_gb": 1.0, "available_gb": 50.0},
        cpu_pct=10.0,
        gpus=[{"mem_pct": 10.0}],
    )
    assert plan.reason == "headroom_wait"
    plan = ratchet_step(
        plan,
        now=2.0,
        ram={"used_gb": 1.0, "available_gb": 50.0},
        cpu_pct=10.0,
        gpus=[{"mem_pct": 10.0}],
    )
    assert plan.reason == "bump"
    assert plan.knobs["w"].value == 12
    plan.write_json(tmp_path / "plan.json")
    assert (tmp_path / "plan.json").is_file()


def test_schedule_engine():
    raw = {
        "schema": "demo.schedule/v1",
        "ids": ["a", "b"],
        "weights": {"a": 1.0, "b": 0.5},
        "total_epochs": 4,
        "stages": [
            {"epochs": [1, 2], "enable": ["a"]},
            {"epochs": [3, 4], "add": ["b"]},
        ],
    }
    c = validate_schedule(raw)
    assert c["stages"][-1]["enabled"] == ["a", "b"]
    d = schedule_digest(raw)
    assert d.startswith("sha256:")
    p = epoch_plan(raw, 1)
    assert p.enabled == ("a",)
    assert p.weights["b"] == 0.0


def test_test_profiles(tmp_path: Path):
    manifest = {
        "schema": 1,
        "profiles": {
            "quick": {
                "budget_seconds": 30,
                "commands": [["{python}", "-c", "print('ok')"]],
            }
        },
    }
    path = tmp_path / "m.json"
    path.write_text(json.dumps(manifest))
    m = load_manifest(path)
    r = ProfileRunner(m, history_path=tmp_path / "h.jsonl")
    assert r.commands("quick")[0][-1] == "print('ok')"
    from test_profiles import run_profile

    assert run_profile(path, "quick", history_path=tmp_path / "h.jsonl") == 0


def test_submit_guard_auth(tmp_path: Path):
    blob = tmp_path / "sub.zip"
    blob.write_bytes(b"abc")
    from submit_guard.guard import sha256_file

    digest = sha256_file(blob)
    auth = {
        "schema": AUTH_SCHEMA,
        "explicit_user_approval": True,
        "remaining_uses": 1,
        "nonce": "n1",
        "expires_at_epoch": time.time() + 60,
        "competition": "demo",
        "file_sha256": digest,
        "message": "hi",
    }
    ok, reason, identity = validate_authorization(
        auth,
        ["competitions", "submit", "-c", "demo", "-f", str(blob), "-m", "hi"],
    )
    assert ok and identity is not None
    assert identity.competition == "demo"
    bad_ok, _, _ = validate_authorization(
        auth,
        ["competitions", "submit", "-c", "other", "-f", str(blob), "-m", "hi"],
    )
    assert bad_ok is False


def test_log_trim(tmp_path: Path):
    d = tmp_path / "logs"
    d.mkdir()
    a = d / "a.log"
    b = d / "b.log"
    a.write_text("x" * 100)
    b.write_text("y" * 100)
    old = time.time() - 10_000
    import os

    os.utime(a, (old, old))
    r = trim_directory(d, max_age_s=1000, dry_run=False)
    assert r.deleted == 1
    assert not a.exists()
    assert b.exists()
