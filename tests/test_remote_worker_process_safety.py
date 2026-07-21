from pathlib import Path

from scripts.run_remote_worker import (
    _close_mp_queue,
    _parse_args,
    _remote_worker_arm_error,
    _shutdown_leaf_servers,
    _worker_capacity_watchdog_update,
)


ROOT = Path(__file__).resolve().parents[1]


class _Queue:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def put(self, payload: object, timeout: float) -> None:
        self.calls.append((payload, timeout))

    def cancel_join_thread(self) -> None:
        self.calls.append("cancel")

    def close(self) -> None:
        self.calls.append("close")


class _StubbornProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.alive = True
        self.calls: list[object] = []

    def join(self, timeout: float) -> None:
        self.calls.append(("join", timeout))

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.calls.append("terminate")

    def kill(self) -> None:
        self.calls.append("kill")
        self.alive = False


def test_leaf_shutdown_escalates_and_reaps_stubborn_child() -> None:
    proc = _StubbornProcess(123)
    control = _Queue()

    survivors = _shutdown_leaf_servers(
        [proc],
        [control],
        graceful_timeout_s=0,
    )

    assert survivors == ()
    assert control.calls[0] == ({"cmd": "stop"}, 0.2)
    assert "terminate" in proc.calls
    assert "kill" in proc.calls
    assert proc.calls[-1][0] == "join"


def test_queue_cleanup_cancels_feeder_before_close() -> None:
    queue = _Queue()
    _close_mp_queue(queue)
    assert queue.calls == ["cancel", "close"]


def test_remote_worker_has_signal_watchdog_and_full_cleanup() -> None:
    source = (ROOT / "scripts/run_remote_worker.py").read_text(encoding="utf-8")
    assert 'signal.signal(sig, _request_shutdown)' in source
    assert "pool.request_stop(str(state[\"shutdown_reason\"]))" in source
    assert "_shutdown_leaf_servers(leaf_servers, leaf_ctrl_qs)" in source
    assert "manager = mpctx.Manager()" in source
    assert "manager.shutdown()" in source
    assert "ready_pids = pool.ready_worker_pids" in source
    assert "_worker_capacity_watchdog_update(" in source
    assert "missing_pool_samples" not in source
    assert "POKEBOT_REMOTE_WORKER_CAPACITY_RECOVERY_GRACE_S" in source
    assert 'default=int(os.environ.get("SIM_WORKERS", "4"))' in source
    assert 'default=int(os.environ.get("LEAF_SERVERS", "1"))' in source
    assert 'os.environ.get("POKEBOT_REMOTE_TREE_RSS_LIMIT_GB", "32")' in source


def test_capacity_watchdog_allows_full_monotonic_recycle_grace() -> None:
    unhealthy_since, reason = _worker_capacity_watchdog_update(
        now_s=100.0,
        unhealthy_since_s=None,
        capacity_healthy=False,
        pool_stopped=False,
        ready_workers=11,
        expected_workers=12,
        initializer_attempts=13,
        initializer_failures=0,
        recovery_grace_s=60.0,
    )
    assert unhealthy_since == 100.0
    assert reason is None

    unhealthy_since, reason = _worker_capacity_watchdog_update(
        now_s=159.999,
        unhealthy_since_s=unhealthy_since,
        capacity_healthy=False,
        pool_stopped=False,
        ready_workers=11,
        expected_workers=12,
        initializer_attempts=13,
        initializer_failures=0,
        recovery_grace_s=60.0,
    )
    assert unhealthy_since == 100.0
    assert reason is None

    unhealthy_since, reason = _worker_capacity_watchdog_update(
        now_s=160.0,
        unhealthy_since_s=unhealthy_since,
        capacity_healthy=False,
        pool_stopped=False,
        ready_workers=11,
        expected_workers=12,
        initializer_attempts=13,
        initializer_failures=0,
        recovery_grace_s=60.0,
    )
    assert unhealthy_since == 100.0
    assert reason is not None
    assert "unhealthy_for=60.0s grace=60.0s" in reason


def test_capacity_watchdog_recovery_resets_the_grace_window() -> None:
    unhealthy_since, reason = _worker_capacity_watchdog_update(
        now_s=10.0,
        unhealthy_since_s=5.0,
        capacity_healthy=True,
        pool_stopped=False,
        ready_workers=12,
        expected_workers=12,
        initializer_attempts=13,
        initializer_failures=0,
        recovery_grace_s=60.0,
    )
    assert unhealthy_since is None
    assert reason is None

    unhealthy_since, reason = _worker_capacity_watchdog_update(
        now_s=100.0,
        unhealthy_since_s=unhealthy_since,
        capacity_healthy=False,
        pool_stopped=False,
        ready_workers=11,
        expected_workers=12,
        initializer_attempts=14,
        initializer_failures=0,
        recovery_grace_s=60.0,
    )
    assert unhealthy_since == 100.0
    assert reason is None


def test_capacity_watchdog_treats_rotating_minority_as_healthy() -> None:
    unhealthy_since, reason = _worker_capacity_watchdog_update(
        now_s=200.0,
        unhealthy_since_s=100.0,
        capacity_healthy=False,
        pool_stopped=False,
        ready_workers=30,
        expected_workers=36,
        initializer_attempts=500,
        initializer_failures=0,
        recovery_grace_s=60.0,
        minimum_ready_workers=29,
    )
    assert unhealthy_since is None
    assert reason is None

    unhealthy_since, reason = _worker_capacity_watchdog_update(
        now_s=200.0,
        unhealthy_since_s=100.0,
        capacity_healthy=False,
        pool_stopped=False,
        ready_workers=28,
        expected_workers=36,
        initializer_attempts=500,
        initializer_failures=0,
        recovery_grace_s=60.0,
        minimum_ready_workers=29,
    )
    assert unhealthy_since == 100.0
    assert reason is not None
    assert "minimum_ready=29" in reason

def test_capacity_watchdog_fails_stopped_or_initializer_failed_pool_immediately() -> None:
    _, stopped_reason = _worker_capacity_watchdog_update(
        now_s=1.0,
        unhealthy_since_s=None,
        capacity_healthy=False,
        pool_stopped=True,
        ready_workers=11,
        expected_workers=12,
        initializer_attempts=13,
        initializer_failures=0,
        recovery_grace_s=60.0,
    )
    assert stopped_reason is not None
    assert "pool stopped" in stopped_reason

    _, initializer_reason = _worker_capacity_watchdog_update(
        now_s=1.0,
        unhealthy_since_s=None,
        capacity_healthy=False,
        pool_stopped=False,
        ready_workers=11,
        expected_workers=12,
        initializer_attempts=13,
        initializer_failures=1,
        recovery_grace_s=60.0,
    )
    assert initializer_reason is not None
    assert "initializer failed" in initializer_reason


def test_capacity_recovery_grace_is_configurable_by_env_and_cli(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "POKEBOT_REMOTE_WORKER_CAPACITY_RECOVERY_GRACE_S", "60"
    )
    assert _parse_args([]).worker_capacity_recovery_grace_s == 60.0
    assert (
        _parse_args(["--worker-capacity-recovery-grace-s", "30"])
        .worker_capacity_recovery_grace_s
        == 30.0
    )
    monkeypatch.setenv("POKEBOT_REMOTE_WORKER_MIN_READY_FRAC", "0.80")
    assert _parse_args([]).worker_capacity_min_ready_frac == 0.80
    assert (
        _parse_args(["--worker-capacity-min-ready-frac", "0.90"])
        .worker_capacity_min_ready_frac
        == 0.90
    )


def test_non_smoke_arm_requires_env_and_exact_token_file(
    tmp_path, monkeypatch
) -> None:
    arm_file = tmp_path / "REMOTE_WORKER_ARMED"
    monkeypatch.setenv("POKEBOT_REMOTE_WORKER_ARM_FILE", str(arm_file))
    monkeypatch.delenv("POKEBOT_REMOTE_WORKER_SAFETY_VERSION", raising=False)
    assert "memory safety version" in str(_remote_worker_arm_error(smoke=False))

    monkeypatch.setenv("POKEBOT_REMOTE_WORKER_SAFETY_VERSION", "20260717")
    assert "unreadable" in str(_remote_worker_arm_error(smoke=False))
    arm_file.write_text("20260717\n", encoding="utf-8")
    assert "must contain exactly" in str(_remote_worker_arm_error(smoke=False))
    arm_file.write_text("20260717", encoding="utf-8")
    assert _remote_worker_arm_error(smoke=False) is None

    # Smoke must remain usable for a deployment preflight without arming a
    # production listener.
    monkeypatch.delenv("POKEBOT_REMOTE_WORKER_SAFETY_VERSION", raising=False)
    arm_file.unlink()
    assert _remote_worker_arm_error(smoke=True) is None


def test_production_arm_gate_precedes_runtime_imports() -> None:
    source = (ROOT / "scripts/run_remote_worker.py").read_text(encoding="utf-8")
    gate = source.index("arm_error = _remote_worker_arm_error")
    parse = source.index("args = _parse_args(raw_argv)")
    runtime_import = source.index("from poke_bot import config", gate)
    remote_job_import = source.index("from poke_bot.remote_sim_jobs import", gate)
    assert gate < parse < runtime_import
    assert gate < remote_job_import
    assert 'REMOTE_WORKER_SAFETY_VERSION = "20260717"' in source
    assert "return 78" in source
