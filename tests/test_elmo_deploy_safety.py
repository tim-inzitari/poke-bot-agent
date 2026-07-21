from __future__ import annotations

import os
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts.run_remote_worker import (
    REMOTE_WORKER_PLANNED_ROTATION_EXIT_CODE,
    REMOTE_WORKER_WATCHDOG_EXIT_CODE,
    _persist_active_checkpoint,
    _raw_sha256_digest,
    _select_startup_checkpoint,
    _service_shutdown_exit_code,
    main as remote_worker_main,
)
from scripts.seed_remote_active_checkpoint import main as seed_active_checkpoint_main


ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "containers" / "truenas-worker"
SUPERVISOR = DEPLOY / "supervise-production.sh"


def _text(name: str) -> str:
    return (DEPLOY / name).read_text(encoding="utf-8")


def _assert_compose_safety(text: str) -> None:
    assert 'restart: "no"' in text
    assert "init: true" in text
    assert "image: poke-bot-truenas-worker:safety-20260717.3" in text
    assert 'entrypoint: ["/entrypoint.safe.sh"]' not in text
    assert "./entrypoint.sh:/entrypoint.safe.sh:ro" not in text
    assert (
        "POKEBOT_REMOTE_WORKER_ARM_FILE: "
        "/workspace/runtime-logs/REMOTE_WORKER_ARMED" in text
    )
    assert re.search(r"^\s*SIM_WORKERS: [\"']?\$\{ELMO_SIM_WORKERS:-4\}", text, re.M)
    assert re.search(
        r"^\s*SIM_DEFAULT_WORKERS: [\"']?\$\{ELMO_SIM_DEFAULT_WORKERS:-4\}",
        text,
        re.M,
    )
    assert re.search(r'^\s*ELMO_SIM_WORKER_CEILING: ["\']20["\']', text, re.M)
    assert not re.search(r'^\s*SIM_WORKERS: ["\']?40["\']?\s*$', text, re.M)
    assert re.search(r"^\s*mem_limit:\s*40g\s*$", text, re.M)
    assert re.search(r"^\s*memswap_limit:\s*40g\s*$", text, re.M)
    assert re.search(r"^\s*pids_limit:\s*256\s*$", text, re.M)
    assert re.search(
        r'^\s*POKEBOT_WORKER_RECYCLE_GAMES: ["\']16["\']', text, re.M
    )
    assert re.search(r'^\s*WORKER_RECYCLE_GAMES: ["\']16["\']', text, re.M)
    assert re.search(
        r'^\s*POKEBOT_REMOTE_MAX_SERVICE_JOBS: ["\']100["\']', text, re.M
    )
    assert re.search(
        r'^\s*POKEBOT_REMOTE_TREE_RSS_LIMIT_GB: ["\']30["\']', text, re.M
    )
    assert re.search(
        r'^\s*POKEBOT_REMOTE_MAX_CONNECTIONS: ["\']16["\']', text, re.M
    )
    assert re.search(
        r'^\s*POKEBOT_REMOTE_WORKER_CAPACITY_RECOVERY_GRACE_S: ["\']60["\']',
        text,
        re.M,
    )
    assert re.search(r'^\s*max-size:\s*["\']10m["\']\s*$', text, re.M)
    assert re.search(r'^\s*max-file:\s*["\']3["\']\s*$', text, re.M)
    assert "RemoteJobClient" in text
    assert "h.get('controller_healthy') is True" in text
    assert "h.get('leaf_alive') is True" in text
    assert "h.get('leaf_identity_ok') is True" in text
    assert "socket.create_connection" not in text


def test_nested_docker_compose_is_fail_closed_canary() -> None:
    _assert_compose_safety(_text("docker-compose.yml"))


def test_host_docker_compose_has_the_same_safety_limits() -> None:
    _assert_compose_safety(_text("docker-compose.host.yml"))


def test_rollback_override_preserves_the_prior_immutable_image_tag() -> None:
    text = _text("docker-compose.rollback.yml")
    assert "image: poke-bot-truenas-worker:safety-20260717" in text
    assert "safety-20260717.3" not in text
    ops = _text("OPS.md")
    assert "never rebuild or retag it" in ops
    assert "-f docker-compose.rollback.yml" in ops


def test_production_override_is_bounded_and_preserves_hard_limits() -> None:
    text = _text("docker-compose.production.yml")
    assert 'command: ["production"]' in text
    assert 'restart: "on-failure:3"' in text
    assert re.search(r"^\s*mem_limit:\s*40g\s*$", text, re.M)
    assert re.search(r"^\s*memswap_limit:\s*40g\s*$", text, re.M)
    assert re.search(r"^\s*pids_limit:\s*320\s*$", text, re.M)
    assert re.search(r'^\s*SIM_WORKERS: ["\']20["\']', text, re.M)
    assert re.search(r'^\s*SIM_DEFAULT_WORKERS: ["\']20["\']', text, re.M)
    assert re.search(
        r'^\s*POKEBOT_REMOTE_MAX_CONNECTIONS: ["\']24["\']', text, re.M
    )
    assert re.search(
        r'^\s*POKEBOT_WORKER_RECYCLE_GAMES: ["\']16["\']', text, re.M
    )
    assert re.search(r'^\s*WORKER_RECYCLE_GAMES: ["\']16["\']', text, re.M)
    assert re.search(
        r'^\s*POKEBOT_REMOTE_MAX_SERVICE_JOBS: ["\']0["\']', text, re.M
    )
    assert re.search(
        r'^\s*POKEBOT_REMOTE_PLANNED_ROTATION_EXIT_CODE: ["\']75["\']',
        text,
        re.M,
    )
    assert re.search(
        r'^\s*POKEBOT_REMOTE_TREE_RSS_LIMIT_GB: ["\']30["\']', text, re.M
    )
    assert re.search(
        r'^\s*POKEBOT_REMOTE_MIN_FREE_RAM_GB: ["\']24["\']', text, re.M
    )
    assert re.search(
        r'^\s*POKEBOT_REMOTE_WORKER_CAPACITY_RECOVERY_GRACE_S: ["\']60["\']',
        text,
        re.M,
    )
    assert re.search(
        r'^\s*POKEBOT_ELMO_RESTART_LIMIT: ["\']3["\']', text, re.M
    )
    assert re.search(
        r'^\s*POKEBOT_ELMO_RESTART_WINDOW_S: ["\']3600["\']', text, re.M
    )
    assert "/workspace/runtime-logs/elmo-supervisor" in text
    assert "REMOTE_WORKER_ARMED" not in text
    assert (
        "POKEBOT_REMOTE_ACTIVE_CHECKPOINT_FILE: "
        "/workspace/runtime-logs/elmo-supervisor/active-checkpoint.json" in text
    )
    assert "POKEBOT_REMOTE_CHECKPOINT_ROOT: /workspace/checkpoint" in text
    assert re.search(
        r'^\s*POKEBOT_ELMO_CHILD_STOP_GRACE_S: ["\']75["\']', text, re.M
    )


def test_entrypoint_enforces_worker_ceiling_and_exact_advertise() -> None:
    text = _text("entrypoint.sh")
    assert 'REQUIRED_SAFETY_VERSION="20260717"' in text
    assert 'POKEBOT_REMOTE_WORKER_SAFETY_VERSION:-' in text
    assert "rebuild it before starting Elmo" in text
    assert 'SIM_WORKERS="${SIM_WORKERS:-4}"' in text
    assert 'SIM_DEFAULT_WORKERS="${SIM_DEFAULT_WORKERS:-4}"' in text
    assert 'ELMO_SIM_WORKER_CEILING="${ELMO_SIM_WORKER_CEILING:-20}"' in text
    assert "SIM_WORKERS > ELMO_SIM_WORKER_CEILING" in text
    assert "SIM_DEFAULT_WORKERS > SIM_WORKERS" in text
    assert '--workers "$SIM_WORKERS"' in text
    # New workers read SIM_DEFAULT_WORKERS from the environment. Avoid passing
    # the newer flag so the safety entrypoint can also guard the deployed image
    # while that image is being rebuilt.
    assert "export SIM_WORKERS SIM_DEFAULT_WORKERS" in text
    assert '--default-workers "$SIM_DEFAULT_WORKERS"' not in text


def test_entrypoint_enforces_production_rotation_contract() -> None:
    text = _text("entrypoint.sh")
    assert '[[ "${1:-}" == "production" ]]' in text
    assert "production mode does not accept command-line overrides" in text
    assert 'production_jobs" == "0"' in text
    assert "production_jobs >= 512 && production_jobs <= 1024" in text
    assert 'production_rotation_code" != "75"' in text
    assert '"POKEBOT_WORKER_RECYCLE_GAMES:16"' in text
    assert '"WORKER_RECYCLE_GAMES:16"' in text
    assert '"POKEBOT_REMOTE_TREE_RSS_LIMIT_GB:30"' in text
    assert '"POKEBOT_REMOTE_MIN_FREE_RAM_GB:24"' in text
    assert (
        '"POKEBOT_REMOTE_WORKER_CAPACITY_RECOVERY_GRACE_S:60"' in text
    )
    assert "exec /supervise-production.sh" in text
    assert '--planned-rotation-exit-code "$production_rotation_code"' in text
    assert '[[ "${1:-}" == "seed-active-checkpoint" ]]' in text
    assert "scripts/seed_remote_active_checkpoint.py" in text


def test_image_defaults_do_not_hide_a_larger_pool() -> None:
    text = _text("Dockerfile")
    assert "POKEBOT_REMOTE_WORKER_SAFETY_VERSION=20260717" in text
    assert "org.opencontainers.image.version=\"safety-20260717.3\"" in text
    assert (
        "POKEBOT_REMOTE_WORKER_ARM_FILE="
        "/workspace/runtime-logs/REMOTE_WORKER_ARMED" in text
    )
    assert "POKEBOT_WORKER_RECYCLE_GAMES=16" in text
    assert "WORKER_RECYCLE_GAMES=16" in text
    assert "POKEBOT_REMOTE_MAX_CONNECTIONS=16" in text
    assert "POKEBOT_REMOTE_WORKER_CAPACITY_RECOVERY_GRACE_S=60" in text
    assert re.search(r"^\s*SIM_WORKERS=4 \\\s*$", text, re.M)
    assert re.search(r"^\s*SIM_DEFAULT_WORKERS=4 \\\s*$", text, re.M)
    assert re.search(r"^\s*ELMO_SIM_WORKER_CEILING=20 \\\s*$", text, re.M)
    assert not re.search(r"^\s*SIM_WORKERS=40(?:\s|\\|$)", text, re.M)
    assert (
        "COPY containers/truenas-worker/supervise-production.sh "
        "/supervise-production.sh" in text
    )
    assert "/supervise-production.sh" in text


def test_worker_exit_codes_distinguish_planned_config_and_watchdog(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert REMOTE_WORKER_PLANNED_ROTATION_EXIT_CODE == 75
    assert REMOTE_WORKER_WATCHDOG_EXIT_CODE == 70
    assert REMOTE_WORKER_PLANNED_ROTATION_EXIT_CODE != REMOTE_WORKER_WATCHDOG_EXIT_CODE
    assert (
        _service_shutdown_exit_code(
            planned_rotation=True,
            planned_rotation_exit_code=REMOTE_WORKER_PLANNED_ROTATION_EXIT_CODE,
        )
        == 75
    )
    assert (
        _service_shutdown_exit_code(
            planned_rotation=False,
            planned_rotation_exit_code=REMOTE_WORKER_PLANNED_ROTATION_EXIT_CODE,
        )
        == 70
    )

    # Configuration/arming fails before imports or child creation and cannot be
    # mistaken for either supervisor lifecycle code.
    monkeypatch.delenv("POKEBOT_REMOTE_WORKER_SAFETY_VERSION", raising=False)
    monkeypatch.setenv(
        "POKEBOT_REMOTE_WORKER_ARM_FILE", str(tmp_path / "missing-arm-token")
    )
    assert remote_worker_main([]) == 78


def _active_checkpoint_env(state_file: Path, root: Path) -> dict[str, str]:
    return {
        "POKEBOT_REMOTE_ACTIVE_CHECKPOINT_FILE": str(state_file),
        "POKEBOT_REMOTE_CHECKPOINT_ROOT": str(root),
    }


def _write_active_checkpoint_state(
    state_file: Path,
    checkpoint: Path,
    digest: str,
) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(
        json.dumps(
            {
                "version": 1,
                "path": str(checkpoint),
                "digest": digest,
                "published_at_epoch": time.time(),
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_valid_durable_checkpoint_state_resumes_exact_reload(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    root.mkdir()
    initial = root / "model.pt"
    initial.write_bytes(b"initial")
    reloaded = root / "iter_00007.digest.pt"
    reloaded.write_bytes(b"new-iteration-weights")
    state_file = tmp_path / "runtime-logs" / "active-checkpoint.json"
    env = _active_checkpoint_env(state_file, root)
    digest = _raw_sha256_digest(reloaded)

    published = _persist_active_checkpoint(reloaded, digest, source=env)
    selected, selected_digest = _select_startup_checkpoint(initial, source=env)

    assert published == state_file
    assert selected == reloaded.resolve()
    assert selected_digest == digest
    assert selected != initial


def test_unconfigured_canary_keeps_configured_checkpoint(tmp_path: Path) -> None:
    configured = tmp_path / "model.pt"
    configured.write_bytes(b"canary")
    selected, digest = _select_startup_checkpoint(configured, source={})
    assert selected == configured
    assert digest is None


def test_missing_durable_state_fails_instead_of_falling_back(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    root.mkdir()
    configured = root / "model.pt"
    configured.write_bytes(b"fallback-must-not-load")
    env = _active_checkpoint_env(tmp_path / "missing-state.json", root)

    with pytest.raises(ValueError, match="state is unreadable"):
        _select_startup_checkpoint(configured, source=env)


def test_corrupt_durable_state_fails_instead_of_falling_back(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    root.mkdir()
    configured = root / "model.pt"
    configured.write_bytes(b"fallback-must-not-load")
    state_file = tmp_path / "active-checkpoint.json"
    state_file.write_text("{not-json\n", encoding="utf-8")
    env = _active_checkpoint_env(state_file, root)

    with pytest.raises(ValueError, match="invalid JSON"):
        _select_startup_checkpoint(configured, source=env)


def test_escaped_durable_checkpoint_path_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    root.mkdir()
    configured = root / "model.pt"
    configured.write_bytes(b"fallback-must-not-load")
    escaped = tmp_path / "outside.pt"
    escaped.write_bytes(b"outside")
    state_file = tmp_path / "active-checkpoint.json"
    _write_active_checkpoint_state(
        state_file, escaped.resolve(), _raw_sha256_digest(escaped)
    )
    env = _active_checkpoint_env(state_file, root)

    with pytest.raises(ValueError, match="escapes root"):
        _select_startup_checkpoint(configured, source=env)


def test_durable_checkpoint_digest_mismatch_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    root.mkdir()
    configured = root / "model.pt"
    configured.write_bytes(b"fallback-must-not-load")
    reloaded = root / "iter.pt"
    reloaded.write_bytes(b"actual-new-weights")
    state_file = tmp_path / "active-checkpoint.json"
    _write_active_checkpoint_state(state_file, reloaded.resolve(), "sha256:" + "0" * 64)
    env = _active_checkpoint_env(state_file, root)

    with pytest.raises(ValueError, match="digest mismatch"):
        _select_startup_checkpoint(configured, source=env)


def test_reload_success_persists_before_worker_identity_publish() -> None:
    source = (ROOT / "scripts" / "run_remote_worker.py").read_text(encoding="utf-8")
    persist = source.index("_persist_active_checkpoint(path, actual)")
    state_publish = source.index('state["digest"] = actual', persist)
    healthy = source.index('state["healthy"] = True', state_publish)
    assert persist < state_publish < healthy
    assert "durable active-checkpoint publication failed" in source


def test_explicit_seed_preserves_an_existing_reloaded_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "checkpoint"
    root.mkdir()
    initial = root / "model.pt"
    initial.write_bytes(b"initial")
    reloaded = root / "iter.pt"
    reloaded.write_bytes(b"reloaded")
    state_file = tmp_path / "runtime-logs" / "active-checkpoint.json"
    env = _active_checkpoint_env(state_file, root)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    assert seed_active_checkpoint_main(["--checkpoint", str(initial)]) == 0
    _persist_active_checkpoint(
        reloaded, _raw_sha256_digest(reloaded), source=env
    )
    assert seed_active_checkpoint_main(["--checkpoint", str(initial)]) == 0

    selected, digest = _select_startup_checkpoint(initial, source=env)
    assert selected == reloaded.resolve()
    assert digest == _raw_sha256_digest(reloaded)


def _supervisor_env(state_dir: Path, *, restart_limit: int) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "POKEBOT_ELMO_SUPERVISOR_STATE_DIR": str(state_dir),
            "POKEBOT_REMOTE_PLANNED_ROTATION_EXIT_CODE": "75",
            "POKEBOT_ELMO_RESTART_LIMIT": str(restart_limit),
            "POKEBOT_ELMO_RESTART_WINDOW_S": "60",
            "POKEBOT_ELMO_FAILURE_BACKOFF_S": "0",
            "POKEBOT_ELMO_ROTATION_DELAY_S": "0",
            "POKEBOT_ELMO_MIN_ROTATION_RUNTIME_S": "0",
            "POKEBOT_ELMO_CHILD_STOP_GRACE_S": "2",
            "POKEBOT_ELMO_SESSION_LAUNCHER_PYTHON": sys.executable,
        }
    )
    return env


def _run_supervisor(
    state_dir: Path,
    command: list[str],
    *,
    restart_limit: int,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SUPERVISOR), *command],
        cwd=ROOT,
        env=_supervisor_env(state_dir, restart_limit=restart_limit),
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )


def test_supervisor_resumes_planned_rotation_without_spending_failure(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    counter = tmp_path / "counter"
    fake = tmp_path / "planned-then-config.sh"
    fake.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
counter="$1"
n=0
if [[ -f "$counter" ]]; then n="$(cat "$counter")"; fi
n=$((n + 1))
printf '%s\n' "$n" >"$counter"
if [[ "$n" -eq 1 ]]; then exit 75; fi
exit 64
""",
        encoding="utf-8",
    )

    result = _run_supervisor(
        state_dir,
        ["bash", str(fake), str(counter)],
        restart_limit=1,
    )
    assert result.returncode == 0, result.stderr
    assert counter.read_text(encoding="utf-8").strip() == "2"
    assert len((state_dir / "failures.epoch").read_text().splitlines()) == 1
    assert (state_dir / "last_rotation.epoch").is_file()
    assert (state_dir / "circuit.state").read_text().startswith("open ")

    # Once open, another outer Docker invocation exits cleanly before starting
    # a child, so on-failure cannot defeat the durable circuit.
    second = _run_supervisor(
        state_dir,
        ["bash", str(fake), str(counter)],
        restart_limit=1,
    )
    assert second.returncode == 0, second.stderr
    assert counter.read_text(encoding="utf-8").strip() == "2"


@pytest.mark.parametrize("failure_rc", [64, 70])
def test_supervisor_circuits_config_and_watchdog_exits(
    tmp_path: Path, failure_rc: int
) -> None:
    state_dir = tmp_path / f"state-{failure_rc}"
    counter = tmp_path / f"counter-{failure_rc}"
    fake = tmp_path / f"failure-{failure_rc}.sh"
    fake.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
counter="$1"
n=0
if [[ -f "$counter" ]]; then n="$(cat "$counter")"; fi
printf '%s\n' "$((n + 1))" >"$counter"
exit {failure_rc}
""",
        encoding="utf-8",
    )

    result = _run_supervisor(
        state_dir,
        ["bash", str(fake), str(counter)],
        restart_limit=2,
    )
    assert result.returncode == 0, result.stderr
    assert counter.read_text(encoding="utf-8").strip() == "2"
    assert len((state_dir / "failures.epoch").read_text().splitlines()) == 2
    assert (state_dir / "circuit.state").read_text().startswith("open ")


def test_stale_active_attempt_consumes_outer_restart_failure(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "failures.epoch").write_text(
        f"{int(time.time())}\n", encoding="utf-8"
    )
    (state_dir / "active.attempt").write_text(
        "active epoch=1 supervisor_pid=1 command=python\n", encoding="utf-8"
    )
    ran = tmp_path / "ran"
    fake = tmp_path / "must-not-run.sh"
    fake.write_text(
        "#!/usr/bin/env bash\nprintf ran >\"$1\"\n",
        encoding="utf-8",
    )

    result = _run_supervisor(
        state_dir,
        ["bash", str(fake), str(ran)],
        restart_limit=2,
    )
    assert result.returncode == 0, result.stderr
    assert not ran.exists()
    assert len((state_dir / "failures.epoch").read_text().splitlines()) == 2
    assert (state_dir / "circuit.state").read_text().startswith("open ")


def test_supervisor_forwards_sigterm_and_waits_for_child_cleanup(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    ready = tmp_path / "ready"
    cleaned = tmp_path / "cleaned"
    fake = tmp_path / "wait-for-term.sh"
    fake.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
ready="$1"
cleaned="$2"
trap 'printf cleaned >"$cleaned"; exit 0' TERM INT HUP
printf ready >"$ready"
while :; do sleep 1; done
""",
        encoding="utf-8",
    )

    process = subprocess.Popen(
        ["bash", str(SUPERVISOR), "bash", str(fake), str(ready), str(cleaned)],
        cwd=ROOT,
        env=_supervisor_env(state_dir, restart_limit=3),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ready.exists()
        process.terminate()
        _, stderr = process.communicate(timeout=8)
        assert process.returncode == 0, stderr
        assert cleaned.read_text(encoding="utf-8") == "cleaned"
        assert (state_dir / "active.attempt").read_text().startswith("inactive ")
        assert not (state_dir / "circuit.state").read_text().startswith("open ")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_supervisor_isolates_and_bounds_complete_child_group_cleanup() -> None:
    text = SUPERVISOR.read_text(encoding="utf-8")
    assert "os.setsid()" in text
    assert 'kill -TERM -- "-$child_pgid"' in text
    assert 'kill -KILL -- "-$child_pgid"' in text
    assert "child_stop_grace_s" in text
    assert '"$child_stop_grace_s" -gt 85' in text
    assert "worker-left-process-group" in text
