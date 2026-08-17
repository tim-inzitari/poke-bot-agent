import os
import plistlib
import subprocess
import time
from pathlib import Path

from scripts.run_remote_worker import (
    _leaf_health_report,
    _status_is_current_attempt,
    _validated_leaf_status,
)


ROOT = Path(__file__).resolve().parents[1]


def _status(index: int, digest: str, version: int) -> dict:
    return {
        "index": index,
        "type": "reload",
        "ok": True,
        "checkpoint_digest": digest,
        "version": version,
        "error": None,
    }


def test_reload_ack_requires_exact_type_digest_and_version() -> None:
    expected = "sha256:new"
    valid, problems = _validated_leaf_status(
        0,
        {
            "type": "reload",
            "ok": True,
            "checkpoint_digest": expected,
            "version": 9,
        },
        expected_type="reload",
        expected_digest=expected,
        expected_version=9,
    )
    assert problems == []
    assert valid["checkpoint_digest"] == expected
    assert valid["version"] == 9

    _, problems = _validated_leaf_status(
        1,
        {
            "type": "reload",
            "ok": True,
            "checkpoint_digest": "sha256:old",
            "version": 8,
        },
        expected_type="reload",
        expected_digest=expected,
        expected_version=9,
    )
    assert any("digest mismatch" in problem for problem in problems)
    assert any("version mismatch" in problem for problem in problems)


def test_health_fails_closed_for_alive_but_mixed_leaf_identities() -> None:
    report = _leaf_health_report(
        [_status(0, "sha256:new", 9), _status(1, "sha256:old", 8)],
        process_alive=[True, True],
        event_alive=[True, True],
        expected_digest="sha256:new",
        expected_version=9,
        controller_healthy=True,
    )
    assert report["leaf_alive"] is True
    assert report["leaf_identity_ok"] is False
    assert report["ok"] is False
    assert report["leaves"][0]["healthy"] is True
    assert report["leaves"][1]["healthy"] is False


def test_stale_leaf_status_is_not_attributed_to_new_reload() -> None:
    assert not _status_is_current_attempt(
        {
            "type": "reload",
            "ok": True,
            "checkpoint_digest": "sha256:old",
            "version": 8,
        },
        expected_type="reload",
        expected_version=9,
    )
    # A failure for the current version is current even though it reports the
    # previously resident digest; exact digest/ok validation then fails closed.
    assert _status_is_current_attempt(
        {
            "type": "reload",
            "ok": False,
            "checkpoint_digest": "sha256:old",
            "version": 9,
        },
        expected_type="reload",
        expected_version=9,
    )


def test_reload_transition_cannot_report_healthy_identity() -> None:
    report = _leaf_health_report(
        [_status(0, "sha256:new", 9), _status(1, "sha256:new", 9)],
        process_alive=[True, True],
        event_alive=[True, True],
        expected_digest="sha256:new",
        expected_version=9,
        controller_healthy=False,
        controller_error="reload in progress",
    )
    assert report["leaf_alive"] is True
    assert report["leaf_identity_ok"] is True
    assert report["ok"] is False
    assert report["controller_error"] == "reload in progress"


def test_redeploy_script_is_fail_closed_and_updates_elmo_bind_source() -> None:
    text = (ROOT / "scripts/redeploy_remote_self_play.sh").read_text(
        encoding="utf-8"
    )

    # Local checkpoint/model/schema/tests run before any remote mutation.
    preflight = text.index("preflight checkpoint load + trusted schema")
    first_scp = text.index("scp -o BatchMode=yes")
    assert preflight < first_scp
    assert "checkpoint.assert_trusted_policy_checkpoint" in text
    assert "feature_schema != features.FEATURE_SCHEMA_VERSION" in text
    assert "actual_profile != expected_profile" in text
    assert text.count('os.environ["PREFLIGHT_PROFILE"] != "none"') == 3
    assert '-e PREFLIGHT_PROFILE="$preflight_profile"' in text
    assert 'PREFLIGHT_PROFILE="$preflight_profile"' in text
    assert "scripts/run_test_profile.py" in text
    assert "scripts/seed_remote_active_checkpoint.py" in text
    assert (
        'CG_LIB_PATH="$repo/kaggle/input/pokemon-tcg-ai-battle/'
        'sample_submission/sample_submission/cg"'
    ) in text
    assert text.index("Bert launchd GUI domain preflight ok") < first_scp

    # Elmo's container target is read-only: replace the host bind source via a
    # same-directory temporary file and retain the prior source.
    assert (
        "/mnt/Main/Elmo/poke-bot-agent/containers/truenas-worker/"
        "checkpoint/model.pt"
    ) in text
    assert 'backup="${host_checkpoint}.before_${deploy_id}"' in text
    assert 'tmp="${host_checkpoint}.tmp.${deploy_id}"' in text
    assert 'mv -f "$tmp" "$host_checkpoint"' in text
    assert 'docker cp "$bootstrap" "$container:/workspace/checkpoint/model.pt"' not in text

    # Elmo's durable active-checkpoint pointer is authoritative at process
    # startup, so it must participate in both activation and rollback.
    assert "POKEBOT_REMOTE_ACTIVE_CHECKPOINT_FILE" in text
    assert 'active_pointer_backup="/tmp/pokebot_active_checkpoint_before_${deploy_id}.json"' in text
    assert 'cp -p "$active_checkpoint_host" "$active_pointer_backup"' in text
    assert "_persist_active_checkpoint" in text
    assert 'python - /workspace/checkpoint/model.pt' in text
    assert 'cp -p "$active_pointer_backup" "$active_restore_tmp"' in text
    elmo_activation = text.split("<<'ELMO'\n", 1)[1].split("\nELMO\n", 1)[0]
    pointer_publish = elmo_activation.index("_persist_active_checkpoint")
    elmo_restart = elmo_activation.index('docker restart "$container"')
    assert pointer_publish < elmo_restart
    elmo_rollback = text.split("<<'ELMOROLLBACK'\n", 1)[1].split(
        "\nELMOROLLBACK\n", 1
    )[0]
    pointer_restore = elmo_rollback.index(
        'cp -p "$active_pointer_backup" "$active_restore_tmp"'
    )
    rollback_restart = elmo_rollback.index('docker restart "$container"')
    assert pointer_restore < rollback_restart

    # Every Bert process group is captured before bootout and terminated as a
    # unit; parent-only cleanup cannot leave reparented pool/MPS children.
    assert "pgrep -f '[r]un_remote_worker.py'" in text
    assert text.count("capture_worker_groups") >= 4
    assert text.count("terminate_worker_groups") >= 4
    assert text.count('kill -TERM -- "-$pgid"') >= 2
    assert text.count('kill -KILL -- "-$pgid"') >= 2
    assert text.count("active.pgid") >= 2
    assert text.count("worker_group_is_exclusive") >= 4
    assert "head -1" not in text
    assert "farm.connect(require_all=True)" in text
    assert "len(infos) == 2" in text
    assert 'health.get("ok") is True' in text
    assert 'health.get("leaf_alive") is True' in text
    assert 'health.get("leaf_identity_ok") is True' in text
    assert text.index("raise SystemExit(0 if ok else 5)") < text.index(
        'log "DONE both remotes run the coherent code/checkpoint deployment"'
    )

    # A failure after either host activates restores its source/checkpoint,
    # prior service mode, launchd assets, and worker command, then verifies the
    # old digest before returning the original non-zero deployment status.
    assert "rollback_elmo()" in text
    assert "rollback_bert()" in text
    assert "pokebot_workspace_before_${deploy_id}.tar" in text
    assert "pokebot_worker_cmds_before_${deploy_id}.json" in text
    assert "pokebot_bert_service_before_${deploy_id}" in text
    assert 'active_checkpoint_file="$repo/outputs/state/bert_worker_supervisor/active-checkpoint.json"' in text
    assert 'cp -Pp "$active_checkpoint_file" "$service_snapshot/active_checkpoint"' in text
    assert 'cp -p "$service_snapshot/active_checkpoint" "$active_checkpoint_restore"' in text
    assert "bert_active_checkpoint_published=" in text
    assert 'PYTHONPATH="$repo"' in text
    assert 'failure_file="$repo/outputs/state/bert_worker_supervisor/failures.epoch"' in text
    assert 'cp -Pp "$failure_file" "$service_snapshot/failure_file"' in text
    assert 'cp -p "$service_snapshot/failure_file" "$failure_file"' in text
    assert 'arm_file="$repo/outputs/state/REMOTE_WORKER_ARMED"' in text
    assert 'cp -Pp "$arm_file" "$service_snapshot/arm_file"' in text
    assert 'cp -p "$service_snapshot/arm_file" "$arm_file"' in text
    assert "printf %s '20260717'" in text
    assert 'sys.path.insert(0, os.environ["POKEBOT_REMOTE_CHECKPOINT_ROOT"])' in text
    assert 'service_snapshot/enable_state' in text
    assert 'launchctl print-disabled "$service_domain"' in text
    assert 'previous_mode="$(cat "$service_snapshot/mode")"' in text
    assert 'if [[ "$previous_mode" == "launchd" ]]' in text
    assert 'elif [[ "$previous_mode" == "detached" ]]' in text
    assert 'expected exactly one Bert :8766 listener' in text
    assert "refusing to restore Bert launchd service without memory guard" in text
    assert '("--workers", "4")' in text
    assert '("--leaf-servers", "1")' in text
    assert '"PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.25"' in text
    assert '"REMOTE_REQUEST_TIMEOUT_S": "120"' in text
    assert '"POKEBOT_REMOTE_WORKER_SAFETY_VERSION": "20260717"' in text
    assert "wait_for_digest" in text
    assert "ROLLBACK_RESULT" in text
    # The transaction can snapshot/restore the currently deployed pre-hardening
    # worker, while all post-deploy readiness checks still require strict `ok`.
    assert text.count('health.get("ok") is None') >= 2
    assert text.index("DEPLOY_COMMITTED=1") < text.index(
        'log "DONE both remotes run the coherent code/checkpoint deployment"'
    )

    # launchd is a participant in the Bert transaction.  KeepAlive is booted
    # out before source mutation and bootstrapped only after an atomic stable
    # checkpoint pointer and rendered plist are installed.  Detached nohup is
    # retained only as the rollback path for the one-time migration.
    activation = text.split("<<'BERT'\n", 1)[1].split("\nBERT\n", 1)[0]
    group_capture = activation.index(
        'capture_worker_groups "$activation_worker_groups"'
    )
    disable = activation.index('launchctl disable "$service_target"')
    bootout = activation.index('launchctl bootout "$service_target"')
    bootout_wait = activation.index("for _ in $(seq 1 240)")
    group_cleanup = activation.index(
        'terminate_worker_groups "$activation_worker_groups"'
    )
    source_publish = activation.index('rm -rf "$repo/poke_bot"')
    checkpoint_publish = activation.index(
        'mv -f "$checkpoint_link_tmp" "$checkpoint_current"'
    )
    durable_checkpoint_publish = activation.index(
        "bert_active_checkpoint_published="
    )
    bootstrap = activation.index(
        'launchctl bootstrap "$service_domain" "$agent_plist"'
    )
    assert (
        group_capture
        < disable
        < bootout
        < group_cleanup
        < bootout_wait
        < source_publish
        < checkpoint_publish
        < durable_checkpoint_publish
        < bootstrap
    )
    assert 'nohup "$repo/.venv/bin/python"' not in activation
    assert text.count('launchctl disable "$service_target"') >= 3
    assert 'plutil -lint "$agent_plist_tmp"' in activation
    assert '"REMOTE_REQUEST_TIMEOUT_S": "120"' in text
    assert 'POKEBOT_REMOTE_REQUEST_TIMEOUT_S": "120"' in text


def test_bert_launchagent_bounds_restarts_and_memory() -> None:
    plist_path = (
        ROOT / "deploy/launchd/com.pokebot.remote-worker-8766.plist"
    )
    with plist_path.open("rb") as handle:
        plist = plistlib.load(handle)

    assert plist["Label"] == "com.pokebot.remote-worker-8766"
    assert plist["RunAtLoad"] is True
    assert plist["KeepAlive"] == {"SuccessfulExit": False}
    assert plist["ThrottleInterval"] >= 120
    assert plist["ProcessType"] == "Standard"
    assert plist["SoftResourceLimits"]["NumberOfFiles"] >= 1024
    assert (
        plist["HardResourceLimits"]["NumberOfFiles"]
        >= plist["SoftResourceLimits"]["NumberOfFiles"]
    )
    assert plist["AbandonProcessGroup"] is False
    env = plist["EnvironmentVariables"]
    assert env["REMOTE_REQUEST_TIMEOUT_S"] == "120"
    assert env["POKEBOT_REMOTE_REQUEST_TIMEOUT_S"] == "120"
    assert env["POKEBOT_REMOTE_WORKER_SAFETY_VERSION"] == "20260717"
    assert env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.25"
    assert env["PYTORCH_MPS_LOW_WATERMARK_RATIO"] == "0.20"
    assert env["POKEBOT_MPS_EMPTY_CACHE_EVERY_BATCHES"] == "1"
    assert env["POKEBOT_BERT_MAX_GROUP_RSS_MIB"] == "18432"
    assert env["POKEBOT_BERT_MIN_FREE_PERCENT"] == "30"
    assert env["POKEBOT_BERT_RESTART_LIMIT"] == "3"
    assert env["POKEBOT_BERT_MAX_GROUP_PROCESSES"] == "32"
    assert env["POKEBOT_WORKER_RECYCLE_GAMES"] == "32"
    assert env["WORKER_RECYCLE_GAMES"] == "32"
    assert env["POKEBOT_REMOTE_TREE_RSS_LIMIT_GB"] == "18"
    assert env["POKEBOT_REMOTE_MIN_FREE_RAM_GB"] == "20"
    assert env["POKEBOT_REMOTE_MAX_SERVICE_JOBS"] == "0"
    # Runtime companions are staged beside each content-addressed checkpoint
    # by the trainer. Pinning startup sources in launchd would overwrite a
    # newer specialist's tree after the planned worker rotation.
    assert "POKEBOT_MATCHUP_RUNTIME_MARKER_SOURCE" not in env
    assert "POKEBOT_PUBLIC_MATCHUP_TREE_SOURCE" not in env
    assert "__POKEBOT_BERT_REPO__" in " ".join(plist["ProgramArguments"])

    wrapper = (
        ROOT / "scripts/run_bert_remote_worker_supervised.sh"
    ).read_text(encoding="utf-8")
    listener_probe = wrapper.index('lsof -nP -iTCP:"$port"')
    safe_wait = wrapper.index("waiting for existing worker")
    worker_spawn = wrapper.index("os.setsid()")
    assert listener_probe < safe_wait < worker_spawn
    assert "refusing takeover from unexpected" in wrapper
    assert "takeover timed out" in wrapper
    assert "pure_rl_bootstrap_current.pt" in wrapper
    assert 'export REMOTE_REQUEST_TIMEOUT_S="120"' in wrapper
    assert 'export POKEBOT_REMOTE_REQUEST_TIMEOUT_S="120"' in wrapper
    assert 'export POKEBOT_REMOTE_WORKER_SAFETY_VERSION="20260717"' in wrapper
    assert 'export PYTORCH_MPS_HIGH_WATERMARK_RATIO="0.25"' in wrapper
    assert 'export PYTORCH_MPS_LOW_WATERMARK_RATIO="0.20"' in wrapper
    assert 'export POKEBOT_MPS_EMPTY_CACHE_EVERY_BATCHES="1"' in wrapper
    assert "sim_workers=4" in wrapper
    assert "default_workers=4" in wrapper
    assert "leaf_servers=1" in wrapper
    assert "leaf_max_batch=32" in wrapper
    assert "leaf_queue_depth=8" in wrapper
    assert 'export POKEBOT_WORKER_RECYCLE_GAMES="32"' in wrapper
    assert 'export WORKER_RECYCLE_GAMES="32"' in wrapper
    assert "max_service_jobs=0" in wrapper
    assert 'worker_state="$(ps -o state=' in wrapper
    assert '--tree-rss-limit-gb "18"' in wrapper
    assert '--min-free-ram-gb "20"' in wrapper
    assert '--max-service-jobs "$max_service_jobs"' in wrapper
    assert "memory guard tripped" in wrapper
    assert "restart circuit open" in wrapper
    assert "active.pgid" in wrapper
    assert "active-checkpoint.json" in wrapper
    assert 'export POKEBOT_REMOTE_ACTIVE_CHECKPOINT_FILE="$active_checkpoint_file"' in wrapper
    assert 'export POKEBOT_REMOTE_CHECKPOINT_ROOT="$checkpoint_root"' in wrapper
    assert '"$python" "$seed_checkpoint_script" --checkpoint "$checkpoint"' in wrapper
    assert 'kill -TERM -- "-$worker_pgid"' in wrapper
    assert 'kill -KILL -- "-$worker_pgid"' in wrapper
    assert "--workers 20" not in wrapper
    assert "--leaf-servers 2" not in wrapper


def test_bert_supervisor_restart_circuit_exits_without_starting(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "outputs/state/bert_worker_supervisor"
    state_dir.mkdir(parents=True)
    now = int(time.time())
    (state_dir / "failures.epoch").write_text(
        f"{now - 3}\n{now - 2}\n{now - 1}\n",
        encoding="ascii",
    )
    wrapper = ROOT / "scripts/run_bert_remote_worker_supervised.sh"
    completed = subprocess.run(
        ["/bin/bash", str(wrapper), str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "POKEBOT_BERT_RESTART_LIMIT": "3",
            "POKEBOT_BERT_RESTART_WINDOW_S": "3600",
        },
    )
    assert completed.returncode == 0
    assert "restart circuit open" in completed.stderr
    assert "starting memory-guarded canary" not in completed.stderr


def test_worker_source_has_reload_drain_barrier() -> None:
    text = (ROOT / "scripts/run_remote_worker.py").read_text(encoding="utf-8")
    close_admission = text.index('state["accepting_jobs"] = False')
    wait_for_jobs = text.index('while int(state["active_jobs"]) > 0')
    send_reload = text.index('"cmd": "reload"')
    assert close_admission < wait_for_jobs < send_reload
    assert 'state["active_jobs"] += 1' in text
    assert "ctrl_cond.notify_all()" in text
    assert "ignored stale status" in text
    assert 'state["terminal_reload_failure"] = True' in text
    assert "worker restart required after an ambiguous or partial" in text
