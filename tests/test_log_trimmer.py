import fcntl
import os
from pathlib import Path

from scripts import launch_blackwell, log_trimmer


def _line_payload(count: int = 40) -> bytes:
    return b"".join(f"line-{index:03d}\n".encode() for index in range(count))


def test_trim_preserves_inode_and_open_append_writer(tmp_path: Path) -> None:
    log = tmp_path / "blackwell.log"
    log.write_bytes(_line_payload())
    writer = os.open(log, os.O_WRONLY | os.O_APPEND)
    inode = log.stat().st_ino
    try:
        assert log_trimmer.trim_file_if_needed(
            log,
            threshold_bytes=100,
            keep_bytes=64,
        )
        assert log.stat().st_ino == inode
        content = log.read_bytes()
        assert content.startswith(b"[log trimmed:")
        retained = content.split(b"\n", 1)[1]
        assert retained.startswith(b"line-")
        assert len(content) <= 64 + 128

        os.write(writer, b"writer-continued\n")
    finally:
        os.close(writer)
    assert log.read_bytes().endswith(b"writer-continued\n")


def test_threshold_and_retention_behavior(tmp_path: Path) -> None:
    log = tmp_path / "blackwell.log"
    log.write_bytes(b"x" * 100)
    assert not log_trimmer.trim_file_if_needed(
        log,
        threshold_bytes=100,
        keep_bytes=20,
    )
    assert log.stat().st_size == 100

    log.write_bytes(_line_payload())
    assert log_trimmer.trim_file_if_needed(
        log,
        threshold_bytes=100,
        keep_bytes=45,
    )
    retained = log.read_bytes().split(b"\n", 1)[1]
    assert retained
    assert retained.endswith(b"\n")
    assert all(line.startswith(b"line-") for line in retained.splitlines())
    assert len(retained) <= 45


def test_symlink_trims_current_run_inode(tmp_path: Path) -> None:
    current = tmp_path / "trusted_factorized_legacy.log"
    stable = tmp_path / "blackwell.log"
    current.write_bytes(_line_payload())
    stable.symlink_to(current.name)
    writer = os.open(current, os.O_WRONLY | os.O_APPEND)
    inode = current.stat().st_ino
    try:
        assert log_trimmer.trim_file_if_needed(
            stable,
            threshold_bytes=100,
            keep_bytes=50,
        )
        assert stable.is_symlink()
        assert stable.stat().st_ino == inode == current.stat().st_ino
        os.write(writer, b"new-current-output\n")
    finally:
        os.close(writer)
    assert stable.read_bytes().endswith(b"new-current-output\n")


def test_only_one_trimmer_can_own_active_inode(tmp_path: Path) -> None:
    log = tmp_path / "blackwell.log"
    first = log_trimmer.acquire_trim_lock(log)
    assert first is not None
    try:
        assert log_trimmer.acquire_trim_lock(log) is None
        assert log_trimmer.trim_lock_matches(first, log)
    finally:
        os.close(first)


def test_future_launch_replaces_symlink_without_touching_history(
    tmp_path: Path,
) -> None:
    historical = tmp_path / "trusted_factorized_legacy.log"
    stable = tmp_path / "blackwell.log"
    historical.write_bytes(b"historical output\n")
    stable.symlink_to(historical.name)

    stream = launch_blackwell.open_stable_log(stable)
    try:
        assert not stable.is_symlink()
        assert stable.stat().st_ino != historical.stat().st_ino
        assert stable.read_bytes() == b""
        flags = fcntl.fcntl(stream.fileno(), fcntl.F_GETFL)
        assert flags & os.O_APPEND
        stream.write(b"future output\n")
    finally:
        stream.close()

    assert historical.read_bytes() == b"historical output\n"
    assert stable.read_bytes() == b"future output\n"


def test_launcher_uses_stable_log_and_passes_trim_configuration(
    tmp_path: Path,
) -> None:
    stable = tmp_path / "blackwell.log"
    args = launch_blackwell._parse_args(
        [
            "--run-name",
            "trusted_metadata_name",
            "--log",
            str(stable),
            "--log-threshold-mb",
            "12",
            "--log-keep-mb",
            "3",
            "--",
            "--archetype",
            "hammer-pult",
        ]
    )
    train, monitor = launch_blackwell.build_commands(args)

    assert train[-2:] == ["--archetype", "hammer-pult"]
    assert train[train.index("--run-name") + 1] == "trusted_metadata_name"
    assert monitor[monitor.index("--log") + 1] == str(stable)
    assert monitor[monitor.index("--log-threshold-mb") + 1] == "12.0"
    assert monitor[monitor.index("--log-keep-mb") + 1] == "3.0"
    assert not any("trusted_metadata_name.log" in value for value in train + monitor)


def test_trimmer_defaults_can_be_set_by_environment(monkeypatch) -> None:
    monkeypatch.setenv("POKEBOT_LOG_THRESHOLD_MB", "64")
    monkeypatch.setenv("POKEBOT_LOG_KEEP_MB", "4")
    monkeypatch.setenv("POKEBOT_LOG_TRIM_INTERVAL", "7")
    args = log_trimmer._parse_args(["--once"])
    assert args.threshold_mb == 64
    assert args.keep_mb == 4
    assert args.interval == 7
