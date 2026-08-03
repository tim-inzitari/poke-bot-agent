from pathlib import Path
import json

from scripts import launch_pure_rl, unattended_monitor


ROOT = Path(__file__).resolve().parents[1]


def test_explicit_attempt_offset_skips_old_fatal_tail(tmp_path: Path) -> None:
    log = tmp_path / "stable.log"
    old = b"FAIL-CLOSED from an earlier attempt\n"
    new = b"new attempt started\n"
    log.write_bytes(old + new)

    offset = unattended_monitor._initial_log_offset(
        log,
        start_at_end=False,
        start_offset=len(old),
    )

    assert log.read_bytes()[offset:] == new


def test_attempt_offset_is_clamped_after_inode_preserving_trim(tmp_path: Path) -> None:
    log = tmp_path / "stable.log"
    log.write_bytes(b"short tail")

    assert unattended_monitor._initial_log_offset(
        log,
        start_at_end=False,
        start_offset=10_000,
    ) == log.stat().st_size


def test_monitor_reads_only_lines_appended_after_attempt_boundary(
    tmp_path: Path,
) -> None:
    log = tmp_path / "stable.log"
    old = "FAIL-CLOSED from an earlier attempt\n"
    new = "heartbeat from this attempt\n"
    log.write_text(old + new, encoding="utf-8")

    chunk, offset, resynced = unattended_monitor._read_new_log_chunk(
        log,
        offset=len(old.encode()),
        current_size=log.stat().st_size,
    )

    assert chunk == new
    assert offset == log.stat().st_size
    assert not resynced
    assert not unattended_monitor.FATAL_PATTERNS["fail_closed"].search(chunk)


def test_remote_transport_soft_drop_never_stops_the_healthy_trainer() -> None:
    pipe_failure = unattended_monitor.FATAL_PATTERNS["broken_pipe"]
    assert not pipe_failure.search(
        "[remote] bert.local:8766 grow slot failed "
        "(ConnectionResetError: [Errno 104] Connection reset by peer)"
    )
    assert not pipe_failure.search(
        "[remote] elmo grow slot failed "
        "(RemoteJobsError: connection closed while reading frame)"
    )
    assert pipe_failure.search(
        "FAIL-CLOSED local writer broken pipe; committed output is unsafe"
    )


def test_retriable_remote_fail_closed_does_not_stop_trainer() -> None:
    remote = (
        "[remote] 192.168.1.143:8765 scheduled job attempt 1/8 failed "
        "(RemoteJobsError: remote worker fail-closed: sim worker capacity "
        "did not recover within grace); reconnect+retry remote"
    )
    assert not unattended_monitor.FATAL_PATTERNS["fail_closed"].search(remote)
    assert not unattended_monitor.FATAL_PATTERNS["digest_mismatch"].search(
        "[remote] worker fail-closed: leaf response checkpoint digest mismatch"
    )
    assert unattended_monitor.FATAL_PATTERNS["fail_closed"].search(
        "FAIL-CLOSED local replay writer integrity failure"
    )


def test_retriable_remote_game_timeout_does_not_stop_trainer() -> None:
    retriable = (
        "[remote] 192.168.1.158:8766 scheduled job attempt 1/8 failed "
        "(RemoteResultError: remote result failed: resource_error, "
        "game_timeout: [timeout] our-agent: RuntimeError: policy runtime "
        "failed closed: TimeoutError: game exceeded 600s); reconnect+retry remote"
    )
    timeout = unattended_monitor.FATAL_PATTERNS["game_timeout"]
    assert not timeout.search(retriable)
    assert timeout.search("local worker game_timeouts=1; collection unsafe")
    assert timeout.search("remote scheduler exhausted; game exceeded 600s")



def test_out_of_band_log_shrink_does_not_replay_retained_fatal_tail(
    tmp_path: Path,
) -> None:
    log = tmp_path / "stable.log"
    retained = "FAIL-CLOSED retained by an external trim\n"
    log.write_text(retained, encoding="utf-8")

    chunk, offset, resynced = unattended_monitor._read_new_log_chunk(
        log,
        offset=10_000,
        current_size=log.stat().st_size,
    )

    assert chunk == ""
    assert offset == log.stat().st_size
    assert resynced

    appended = "heartbeat after resync\n"
    with log.open("a", encoding="utf-8") as fh:
        fh.write(appended)
    chunk, offset, resynced = unattended_monitor._read_new_log_chunk(
        log,
        offset=offset,
        current_size=log.stat().st_size,
    )
    assert chunk == appended
    assert offset == log.stat().st_size
    assert not resynced


def test_launcher_captures_log_boundary_before_spawn() -> None:
    source = (ROOT / "scripts/launch_pure_rl.py").read_text(encoding="utf-8")
    capture = source.index("monitor_log_start_offset = log_path.stat().st_size")
    spawn = source.index("training = subprocess.Popen(")
    monitor = source.index('"--start-offset"')

    assert capture < spawn < monitor


def test_optimizer_progress_log_activity_resets_stall_evidence(
    tmp_path: Path,
) -> None:
    progress = tmp_path / "run.progress.log"
    progress.write_text(
        "rl-agreement parent:  94%|████| 363/387 [09:05<00:37]\r",
        encoding="utf-8",
    )
    first_key, first_line, first_age = unattended_monitor._progress_file_sample(
        progress,
        now=progress.stat().st_mtime + 2,
    )

    with progress.open("a", encoding="utf-8") as stream:
        stream.write("rl-train ep0:   1%| | 4/387 [00:03<05:00]\r")
    second_key, second_line, second_age = unattended_monitor._progress_file_sample(
        progress,
        now=progress.stat().st_mtime + 1,
    )

    assert first_key is not None
    assert second_key is not None
    assert second_key != first_key
    assert "rl-agreement parent" in first_line
    assert "rl-train ep0" in second_line
    assert first_age == 2
    assert second_age == 1


def test_launcher_wires_optimizer_progress_log_to_watchdog() -> None:
    source = (ROOT / "scripts/launch_pure_rl.py").read_text(encoding="utf-8")
    assert '"--progress-log",\n            str(prog_path)' in source


def test_launcher_reports_signal_exit_without_wrapping_to_241() -> None:
    assert launch_pure_rl._normalized_child_returncode(-15) == 143
    assert launch_pure_rl._normalized_child_returncode(-9) == 137
    assert launch_pure_rl._normalized_child_returncode(3) == 3


def test_monitor_stop_receipt_must_match_exact_attempt(tmp_path: Path) -> None:
    alert = tmp_path / "MONITOR_STOP_REQUESTED.json"
    alert.write_text(
        json.dumps({"timestamp": 101.0, "pid": 1234, "reason": "no_progress"}),
        encoding="utf-8",
    )

    assert launch_pure_rl._monitor_requested_this_attempt_stop(
        alert,
        training_pid=1234,
        attempt_started_at=100.0,
    )
    assert not launch_pure_rl._monitor_requested_this_attempt_stop(
        alert,
        training_pid=9999,
        attempt_started_at=100.0,
    )
    assert not launch_pure_rl._monitor_requested_this_attempt_stop(
        alert,
        training_pid=1234,
        attempt_started_at=102.0,
    )


def test_monitor_stop_receipt_rejects_missing_or_invalid_evidence(
    tmp_path: Path,
) -> None:
    alert = tmp_path / "MONITOR_STOP_REQUESTED.json"
    assert not launch_pure_rl._monitor_requested_this_attempt_stop(
        alert,
        training_pid=1234,
        attempt_started_at=100.0,
    )
    alert.write_text("not-json", encoding="utf-8")
    assert not launch_pure_rl._monitor_requested_this_attempt_stop(
        alert,
        training_pid=1234,
        attempt_started_at=100.0,
    )


def test_launcher_lock_rejects_overlapping_full_hardware_run(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "pure_rl_launcher.lock"
    first, owner = launch_pure_rl._acquire_launch_lock(
        lock_path,
        run_name="first",
    )
    assert first is not None
    assert owner == ""
    try:
        second, owner = launch_pure_rl._acquire_launch_lock(
            lock_path,
            run_name="second",
        )
        assert second is None
        assert "run=first" in owner
    finally:
        first.close()

    replacement, owner = launch_pure_rl._acquire_launch_lock(
        lock_path,
        run_name="replacement",
    )
    assert replacement is not None
    assert owner == ""
    replacement.close()


def test_launcher_requires_explicit_production_memory_safety_arm() -> None:
    source = (ROOT / "scripts/launch_pure_rl.py").read_text(encoding="utf-8")
    unit = (
        ROOT / "deploy/systemd/pokebot-pure-rl-core.service"
    ).read_text(encoding="utf-8")
    assert 'TRAINING_SAFETY_VERSION = "20260717"' in source
    assert "return 78" in source
    assert "Environment=POKEBOT_TRAINING_SAFETY_VERSION=20260717" in unit
    assert (
        "Environment=POKEBOT_TRAINING_ARM_FILE="
        "/home/inzi/poke-bot-agent/outputs/state/TRAINING_ARMED"
    ) in unit
    assert "RestartPreventExitStatus=75 78" in unit


def test_production_arm_requires_env_and_exact_token_file(tmp_path: Path) -> None:
    token = tmp_path / "TRAINING_ARMED"
    env = {
        "POKEBOT_TRAINING_SAFETY_VERSION": "20260717",
        "POKEBOT_TRAINING_ARM_FILE": str(token),
    }

    armed, resolved = launch_pure_rl._production_training_arm(env)
    assert not armed
    assert resolved == token

    token.write_bytes(b"20260717\n")
    assert not launch_pure_rl._production_training_arm(env)[0]

    token.write_bytes(b"20260717")
    assert launch_pure_rl._production_training_arm(env)[0]

    env["POKEBOT_TRAINING_SAFETY_VERSION"] = "stale"
    assert not launch_pure_rl._production_training_arm(env)[0]
