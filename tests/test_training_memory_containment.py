"""Regression tests for long-lived Inzi trainer memory containment."""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

from poke_bot import process_memory
from poke_bot.remote_jobs import (
    _SpillableResultQueue,
    _put_thread_result,
    result_queue_capacity,
)


ROOT = Path(__file__).resolve().parents[1]


class _FakeQueue:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def close(self) -> None:
        self.calls.append("close")

    def cancel_join_thread(self) -> None:
        self.calls.append("cancel_join_thread")

    def join_thread(self) -> None:
        self.calls.append("join_thread")


def test_close_mp_queue_releases_feeder_and_handles() -> None:
    q = _FakeQueue()
    process_memory.close_mp_queue(q)
    assert q.calls == ["close", "cancel_join_thread", "join_thread"]


def test_release_process_heap_collects_and_trims_linux(monkeypatch) -> None:
    class _MallocTrim:
        argtypes = None
        restype = None

        def __init__(self) -> None:
            self.calls: list[int] = []

        def __call__(self, pad: int) -> int:
            self.calls.append(pad)
            return 1

    trim = _MallocTrim()
    libc = type("_LibC", (), {"malloc_trim": trim})()
    monkeypatch.setattr(process_memory.gc, "collect", lambda: 17)
    monkeypatch.setattr(process_memory.sys, "platform", "linux")
    monkeypatch.setattr(process_memory.ctypes, "CDLL", lambda _name: libc)

    assert process_memory.release_process_heap() == (17, True)
    assert trim.calls == [0]


def test_release_process_heap_skips_malloc_trim_off_linux(monkeypatch) -> None:
    monkeypatch.setattr(process_memory.gc, "collect", lambda: 3)
    monkeypatch.setattr(process_memory.sys, "platform", "darwin")
    monkeypatch.setattr(
        process_memory.ctypes,
        "CDLL",
        lambda _name: (_ for _ in ()).throw(AssertionError("must not load libc")),
    )
    assert process_memory.release_process_heap() == (3, False)


def test_remote_result_queue_is_two_ram_waves_plus_bounded_disk() -> None:
    assert result_queue_capacity(local_workers=24, remote_workers=4) == 56
    assert result_queue_capacity(local_workers=0, remote_workers=0) == 2

    src = (ROOT / "poke_bot" / "remote_jobs.py").read_text(encoding="utf-8")
    assert src.count("out_q = _SpillableResultQueue(") == 2
    assert src.count("memory_capacity=result_queue_capacity(") == 2
    assert "out_q: queue.Queue[tuple[str, Any]] = queue.Queue()" not in src
    assert src.count("producer_stop = threading.Event()") == 2
    assert "POKEBOT_REMOTE_RESULT_SPOOL_MAX_GB" in src


def test_remote_result_overflow_spills_and_is_removed(tmp_path: Path) -> None:
    out_q = _SpillableResultQueue(
        memory_capacity=1,
        spool_root=tmp_path,
        max_spool_bytes=1024 * 1024,
    )
    first = {"job_index": 1, "record_jsons": ["a" * 4096]}
    second = {"job_index": 2, "record_jsons": ["b" * 4096]}
    out_q.put(("ok", first), timeout=0.1)
    out_q.put(("ok", second), timeout=0.1)
    stats = out_q.telemetry()
    assert stats["memory_items"] == 1
    assert stats["spool_files"] == 1
    assert stats["spool_bytes"] > 0

    assert out_q.get() == ("ok", first)
    assert out_q.get() == ("ok", second)
    assert out_q.telemetry()["spool_files"] == 0
    spool_dir = out_q.spool_dir
    out_q.close()
    assert not spool_dir.exists()


def test_cancelled_result_producer_cannot_remain_blocked_on_full_queue() -> None:
    out_q: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)
    out_q.put(("ok", object()))
    stop = threading.Event()
    outcome: list[bool] = []
    producer = threading.Thread(
        target=lambda: outcome.append(
            _put_thread_result(out_q, stop, ("ok", object()))
        )
    )
    producer.start()
    time.sleep(0.15)
    assert producer.is_alive()
    stop.set()
    producer.join(timeout=1.0)
    assert not producer.is_alive()
    assert outcome == [False]


def test_replay_window_is_released_before_any_evaluation_or_next_collect() -> None:
    src = (ROOT / "scripts" / "train_pure_rl.py").read_text(encoding="utf-8")
    full_loop = src[src.index("def run_full_loop(") :]
    train_at = full_loop.index(
        "result = recovered_candidate_result or rl_train_step("
    )
    finally_at = full_loop.index("finally:", train_at)
    delete_at = full_loop.index("del dataset", finally_at)
    trim_at = full_loop.index("release_process_heap()", delete_at)
    promotion_at = full_loop.index("promotion begin", trim_at)
    heldout_at = full_loop.index(
        "heldout_rows, heldout_audit = _heldout_eval(", promotion_at
    )
    next_collect_at = full_loop.index(
        "pending_collect = _kick_collect(\n                    next_it", heldout_at
    )
    assert train_at < finally_at < delete_at < trim_at
    assert trim_at < promotion_at < heldout_at < next_collect_at


def test_leaf_farm_does_not_overlap_replay_expansion_or_training() -> None:
    src = (ROOT / "scripts" / "train_pure_rl.py").read_text(encoding="utf-8")
    full_loop = src[src.index("def run_full_loop(") :]
    suspend_marker = full_loop.index("suspend leaf farm before rehearsal/replay")
    stop_at = full_loop.index("leaf.stop()", suspend_marker)
    dataset_at = full_loop.index("dataset = _dataset_from_replay_window(", stop_at)
    train_at = full_loop.index(
        "result = recovered_candidate_result or rl_train_step(",
        dataset_at,
    )
    release_at = full_loop.index("replay memory released iter=", train_at)
    restore_at = full_loop.index("_rebuild_leaves_if_needed(", release_at)
    promotion_at = full_loop.index("promotion begin", restore_at)
    assert stop_at < dataset_at < train_at < release_at < restore_at < promotion_at


def test_leaf_farm_reaps_terminated_children_and_closes_every_queue() -> None:
    src = (ROOT / "scripts" / "train_pure_rl.py").read_text(encoding="utf-8")
    leaf = src[src.index("class _LeafFarm:") : src.index("def run_smoke_loop(")]
    terminate_at = leaf.index("proc.terminate()")
    assert leaf.index("proc.join(timeout=5)", terminate_at) > terminate_at
    assert "proc.kill()" in leaf
    assert "close_mp_queue(q)" in leaf
    for queue_group in ("self.req_qs", "self.ctrl_qs", "self.status_qs", "self.resp_qs"):
        assert queue_group in leaf


def test_systemd_unit_disables_dynamic_growth_and_enforces_cgroup_limits() -> None:
    unit = (
        ROOT / "deploy" / "systemd" / "pokebot-pure-rl-core.service"
    ).read_text(encoding="utf-8")
    for directive in (
        "StartLimitIntervalSec=900",
        "StartLimitBurst=3",
        "Environment=POKEBOT_LIVE_POOL=0",
        "Environment=POKEBOT_WORKER_RECYCLE_GAMES=256",
        "Environment=WORKER_RECYCLE_GAMES=256",
        "Environment=PURE_RL_LOCAL_STRAGGLER_STALE_S=86400",
        "Environment=POKEBOT_WORKER_CAPACITY_RECOVERY_GRACE_S=60",
        "Environment=PURE_RL_SIM_WORKERS=40",
        "Environment=PURE_RL_LEAF_GPU0_REPLICAS=4",
        "Environment=PURE_RL_LEAF_GPU1_REPLICAS=48",
        "MemoryHigh=60G",
        "MemoryMax=68G",
        "MemorySwapMax=0",
        "OOMPolicy=stop",
    ):
        assert directive in unit
    assert "StartLimitIntervalSec=0" not in unit
    # A promoted run resumes from its append-only loop ledger.  Pinning the
    # original bootstrap checkpoint makes every post-promotion restart fail
    # the resume identity check and eventually hit systemd's start limit.
    assert "--resume auto" in unit
    assert "--base-checkpoint" not in unit


def test_continuous_unit_resumes_promotions_and_pins_measurement_decks() -> None:
    unit = (
        ROOT
        / "deploy"
        / "systemd"
        / "pokebot-pure-rl-continuous-rehearsal.service"
    ).read_text(encoding="utf-8")
    assert "--resume auto" in unit
    # v7 is a deliberate immutable new-lineage handoff. These identities seed
    # creation once; on every later restart the ledger verifies them against
    # the manifest instead of rewinding the learner.
    handoff_checkpoint = (
        "/home/inzi/poke-bot-agent/outputs/pure_rl/"
        "pure_rl_core_continuous_rehearsal_v6_20260719/"
        "checkpoints/iter_00026.pt"
    )
    assert f"--base-checkpoint {handoff_checkpoint}" in unit
    assert f"--initial-learner-checkpoint {handoff_checkpoint}" in unit
    assert "--run-name pure_rl_core_baseline50_v7_20260720" in unit
    # Persistent sockets claim one game at a time. Larger private reservations
    # synchronize hundreds of emitters into refill waves that drain remotes.
    assert "Environment=POKEBOT_REMOTE_REFILL_GAMES=1" in unit
    assert (
        "Environment=PURE_RL_MEASUREMENT_DECKS="
        "lucario,alakazam,starmie,crustle"
    ) in unit


def test_quick_preflight_budget_covers_the_production_unit_suite() -> None:
    manifest = json.loads(
        (ROOT / "tests" / "profile_manifest.json").read_text(encoding="utf-8")
    )
    # The production unit suite is currently about 30 seconds on Inzi.  Keep
    # enough margin for scheduler noise so a passed suite cannot be reported as
    # a timeout a few milliseconds before pytest exits.
    assert int(manifest["profiles"]["quick"]["budget_seconds"]) >= 60
