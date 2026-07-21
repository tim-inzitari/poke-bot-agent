from pathlib import Path

from scripts.run_remote_worker import (
    _close_mp_queue,
    _remote_worker_arm_error,
    _shutdown_leaf_servers,
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
    assert "if not pool.worker_capacity_healthy" in source
    assert 'default=int(os.environ.get("SIM_WORKERS", "4"))' in source
    assert 'default=int(os.environ.get("LEAF_SERVERS", "1"))' in source
    assert 'os.environ.get("POKEBOT_REMOTE_TREE_RSS_LIMIT_GB", "32")' in source


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
