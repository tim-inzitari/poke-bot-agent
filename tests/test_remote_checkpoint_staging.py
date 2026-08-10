"""Remote checkpoint staging and result validation regressions."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import threading
import time
from types import SimpleNamespace

import pytest

from poke_bot.checkpoint import checkpoint_digest
from poke_bot import remote_jobs
from poke_bot.remote_jobs import (
    RemoteJobClient,
    RemoteJobsError,
    RemoteResultError,
    RemoteWorkerInfo,
    cache_exact_resident_remote_checkpoint,
    digest_addressed_basename,
    iter_additive_results,
    iter_scheduled_additive_results,
    prepare_remote_play_job,
    remote_result_failure_reason,
    require_remote_result_success,
)
from poke_bot.worker_pool import WorkerPool


def test_digest_addressed_basename_embeds_content_digest(tmp_path: Path) -> None:
    path = tmp_path / "iter_00000.pt"
    path.write_bytes(b"weights-a")
    dig_a = checkpoint_digest(path)
    name_a = digest_addressed_basename(path, digest=dig_a)
    assert name_a.startswith("iter_00000.")
    assert name_a.endswith(".pt")
    assert dig_a.split(":", 1)[-1][:16] in name_a

    path.write_bytes(b"weights-b-different")
    dig_b = checkpoint_digest(path)
    name_b = digest_addressed_basename(path, digest=dig_b)
    assert name_a != name_b
    assert dig_b.split(":", 1)[-1][:16] in name_b
    # Same logical trainer filename, distinct remote objects.
    assert name_a.split(".", 1)[0] == name_b.split(".", 1)[0]


def test_local_checkpoint_digest_cache_rehashes_only_after_identity_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from poke_bot import checkpoint as checkpoint_module

    path = tmp_path / "parent.pt"
    path.write_bytes(b"first immutable weights")
    remote_jobs._LOCAL_CHECKPOINT_DIGEST_CACHE.clear()
    original = checkpoint_module.checkpoint_digest
    calls = 0

    def counted(candidate, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(candidate, *args, **kwargs)

    monkeypatch.setattr(checkpoint_module, "checkpoint_digest", counted)
    first = remote_jobs._cached_local_checkpoint_digest(path)
    assert remote_jobs._cached_local_checkpoint_digest(path) == first
    assert calls == 1

    path.write_bytes(b"second and different immutable weights")
    second = remote_jobs._cached_local_checkpoint_digest(path)
    assert second != first
    assert calls == 2


def test_initial_remote_slot_connections_fan_out_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    monkeypatch.setenv("POKEBOT_REMOTE_CONNECT_FANOUT", "64")
    monkeypatch.setenv("POKEBOT_REMOTE_SOCKET_PREFETCH", "1")
    active = 0
    peak = 0
    counter_lock = threading.Lock()

    class Template:
        def __init__(self, host: str, port: int) -> None:
            self.host = host
            self.port = port
            self.endpoint = f"{host}:{port}"
            self.timeout_s = 30.0
            self.connect_timeout_s = 60.0
            self.control_timeout_s = 300.0
            self.info = RemoteWorkerInfo(
                endpoint=self.endpoint,
                workers=4,
                leaf_servers=1,
                gpu_name="test",
                device="cpu",
                checkpoint_digest="sha256:test",
                hostname=host,
                max_workers=4,
                default_workers=4,
            )

        def ensure_alive(self) -> None:
            return None

    def clone(template):
        nonlocal active, peak
        with counter_lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.03)
        with counter_lock:
            active -= 1
        return SimpleNamespace(
            host=template.host,
            port=template.port,
            endpoint=template.endpoint,
        )

    monkeypatch.setattr(remote_jobs, "_clone_remote_client", clone)
    elmo = Template("192.168.1.143", 8765)
    bert = Template("192.168.1.158", 8766)
    slots, owned = remote_jobs._parallel_remote_slots(
        [elmo, bert],
        demand_by_endpoint={elmo.endpoint: 4, bert.endpoint: 4},
    )

    assert len(slots) == 8
    assert len(owned) == 6
    assert peak >= 4


def test_exact_resident_health_seeds_elmo_mapping_without_payload_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "trainer" / "model.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"verified resident weights")
    digest = checkpoint_digest(source)
    smb = tmp_path / "smb"
    smb.mkdir()
    destination = smb / digest_addressed_basename(source, digest=digest)
    # Same byte size is intentional: only the caller's strict live health proof
    # authorizes caching, so this test proves no redundant payload rehash occurs.
    destination.write_bytes(b"x" * source.stat().st_size)

    monkeypatch.setattr(remote_jobs, "_smb_checkpoint_dir", lambda: smb)
    monkeypatch.setattr(
        remote_jobs, "_ELMO_STAGE_RECEIPT_ROOT", tmp_path / "receipts"
    )
    remote_jobs._ELMO_STAGE_CACHE.clear()

    mapped = cache_exact_resident_remote_checkpoint(
        "192.168.1.143",
        str(source),
        digest=digest,
    )
    assert mapped == f"/workspace/checkpoint/{destination.name}"
    assert (
        remote_jobs.resolve_remote_checkpoint_path(
            "192.168.1.143", str(source)
        )
        == mapped
    )


def test_persistent_elmo_stage_receipt_skips_payload_rehash_after_cache_reset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from poke_bot import checkpoint as checkpoint_module

    source = tmp_path / "trainer" / "parent.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"immutable parent weights")
    digest = checkpoint_digest(source)
    smb = tmp_path / "smb"
    smb.mkdir()
    destination = smb / digest_addressed_basename(source, digest=digest)
    destination.write_bytes(source.read_bytes())

    monkeypatch.setattr(remote_jobs, "_smb_checkpoint_dir", lambda: smb)
    monkeypatch.setattr(
        remote_jobs, "_ELMO_STAGE_RECEIPT_ROOT", tmp_path / "receipts"
    )
    remote_jobs._ELMO_STAGE_CACHE.clear()
    cache_exact_resident_remote_checkpoint(
        "192.168.1.143", str(source), digest=digest
    )
    remote_jobs._ELMO_STAGE_CACHE.clear()

    original_digest = checkpoint_module.checkpoint_digest

    def _digest_without_remote_payload(path, *args, **kwargs):
        if Path(path).resolve() == destination.resolve():
            raise AssertionError("valid receipt must bypass SMB payload rehash")
        return original_digest(path, *args, **kwargs)

    monkeypatch.setattr(
        checkpoint_module, "checkpoint_digest", _digest_without_remote_payload
    )
    assert remote_jobs.resolve_remote_checkpoint_path(
        "192.168.1.143", str(source)
    ).endswith(destination.name)


def test_elmo_stage_receipt_is_content_addressed_across_source_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from poke_bot import checkpoint as checkpoint_module

    source_a = tmp_path / "trainer-a" / "iter_00002.pt"
    source_b = tmp_path / "trainer-b" / "population_parent.pt"
    source_a.parent.mkdir()
    source_b.parent.mkdir()
    payload = b"same immutable historical checkpoint"
    source_a.write_bytes(payload)
    source_b.write_bytes(payload)
    digest = checkpoint_digest(source_a)
    smb = tmp_path / "smb"
    smb.mkdir()
    destination = smb / digest_addressed_basename(source_a, digest=digest)
    destination.write_bytes(payload)

    monkeypatch.setattr(remote_jobs, "_smb_checkpoint_dir", lambda: smb)
    monkeypatch.setattr(
        remote_jobs, "_ELMO_STAGE_RECEIPT_ROOT", tmp_path / "receipts"
    )
    remote_jobs._ELMO_STAGE_CACHE.clear()
    remote_jobs._write_elmo_stage_receipt(source_a, destination, digest)

    original_digest = checkpoint_module.checkpoint_digest

    def _digest_without_remote_payload(path, *args, **kwargs):
        if Path(path).resolve() == destination.resolve():
            raise AssertionError("source alias must reuse content receipt")
        return original_digest(path, *args, **kwargs)

    monkeypatch.setattr(
        checkpoint_module, "checkpoint_digest", _digest_without_remote_payload
    )
    mapped = remote_jobs.resolve_remote_checkpoint_path(
        "192.168.1.143", str(source_b)
    )
    assert mapped.endswith(destination.name)


def test_elmo_resolve_uses_control_plane_when_smb_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "trainer" / "seed.pt"
    source.parent.mkdir()
    source.write_bytes(b"resident without trainer smb mount")
    digest = checkpoint_digest(source)
    dest_name = digest_addressed_basename(source, digest=digest)
    remote_path = f"/workspace/checkpoint/{dest_name}"

    monkeypatch.setattr(remote_jobs, "_smb_checkpoint_dir", lambda: None)
    monkeypatch.setattr(
        remote_jobs,
        "_elmo_remote_checkpoint_digest",
        lambda _host, path: digest if path == remote_path else None,
    )
    remote_jobs._ELMO_STAGE_CACHE.clear()

    assert (
        remote_jobs.resolve_remote_checkpoint_path(
            "192.168.1.143", str(source)
        )
        == remote_path
    )
    # Second call must hit the resident cache and not re-probe.
    monkeypatch.setattr(
        remote_jobs,
        "_elmo_remote_checkpoint_digest",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("resident cache must skip re-verify")
        ),
    )
    assert (
        remote_jobs.resolve_remote_checkpoint_path(
            "192.168.1.143", str(source)
        )
        == remote_path
    )


def test_elmo_missing_smb_stages_new_digest_over_verified_ssh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new digest uses atomically published SSH staging, never GUI GVFS."""

    source = tmp_path / "trainer" / "candidate.pt"
    source.parent.mkdir()
    source.write_bytes(b"new immutable candidate weights")
    digest = checkpoint_digest(source)
    destination_name = digest_addressed_basename(source, digest=digest)
    remote_path = f"/workspace/checkpoint/{destination_name}"
    checks: list[str] = []
    commands: list[list[str]] = []

    def remote_digest(_host: str, path: str) -> str | None:
        checks.append(path)
        # The first storage-local check proves this fresh candidate is absent;
        # the second proves the SSH-published object before it may be reloaded.
        return None if len(checks) == 1 else digest

    def run(command, **kwargs):
        commands.append(list(command))
        assert kwargs["check"] is False
        assert kwargs["timeout"] >= 30.0
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(remote_jobs, "_smb_checkpoint_dir", lambda: None)
    monkeypatch.setattr(
        remote_jobs, "_elmo_remote_checkpoint_digest", remote_digest
    )
    monkeypatch.setattr(remote_jobs.subprocess, "run", run)
    monkeypatch.setenv("POKEBOT_ELMO_SSH_STAGE", "1")
    monkeypatch.setenv("POKEBOT_ELMO_SSH", "admin@elmo")
    monkeypatch.setenv("POKEBOT_ELMO_CHECKPOINT_HOST_DIR", "/srv/checkpoint")
    monkeypatch.delenv("POKEBOT_ELMO_SSH_IDENTITY_FILE", raising=False)
    remote_jobs._ELMO_STAGE_CACHE.clear()

    assert (
        remote_jobs.resolve_remote_checkpoint_path("192.168.1.143", str(source))
        == remote_path
    )
    assert checks == [remote_path, remote_path]
    assert [command[0] for command in commands] == ["scp", "ssh"]
    assert commands[0][-1].startswith(
        "admin@elmo:/srv/checkpoint/.candidate."
    )
    assert ".partial." in commands[0][-1]
    assert "sha256sum" in commands[1][-1]
    assert "mv -f" in commands[1][-1]
    assert "smb://" not in commands[1][-1]

    # The resident mapping is cached only after the second storage-local
    # digest proof, so future jobs do not duplicate a large SSH transfer.
    assert (
        remote_jobs.resolve_remote_checkpoint_path("192.168.1.143", str(source))
        == remote_path
    )
    assert len(commands) == 2


def test_elmo_missing_smb_refuses_new_digest_without_explicit_ssh_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "trainer" / "candidate.pt"
    source.parent.mkdir()
    source.write_bytes(b"new immutable candidate weights")

    monkeypatch.setattr(remote_jobs, "_smb_checkpoint_dir", lambda: None)
    monkeypatch.setattr(
        remote_jobs, "_elmo_remote_checkpoint_digest", lambda *_args: None
    )
    monkeypatch.delenv("POKEBOT_ELMO_SSH_STAGE", raising=False)
    remote_jobs._ELMO_STAGE_CACHE.clear()

    with pytest.raises(RemoteJobsError, match="SSH stage"):
        remote_jobs.resolve_remote_checkpoint_path("192.168.1.143", str(source))


@pytest.mark.parametrize(
    "env_name,value",
    (
        ("POKEBOT_ELMO_CHECKPOINT_HOST_DIR", "/srv/checkpoint;not-a-path"),
        ("POKEBOT_ELMO_CHECKPOINT_HOST_DIR", "/srv/check point"),
        ("POKEBOT_ELMO_SSH", "admin@elmo;not-a-target"),
    ),
)
def test_elmo_ssh_stage_rejects_remote_spec_metacharacters(
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
    value: str,
) -> None:
    monkeypatch.setenv("POKEBOT_ELMO_SSH_STAGE", "1")
    monkeypatch.setenv("POKEBOT_ELMO_SSH", "admin@elmo")
    monkeypatch.setenv("POKEBOT_ELMO_CHECKPOINT_HOST_DIR", "/srv/checkpoint")
    monkeypatch.setenv(env_name, value)

    with pytest.raises(RemoteJobsError, match="safe SSH target|absolute safe path"):
        remote_jobs._elmo_ssh_stage_config()


def test_elmo_ssh_stage_publishes_runtime_companions_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = tmp_path / "runtime-tree.json"
    tree.write_bytes(b'{"runtime_contract":"exact"}\n')
    marker = b'{"runtime_enabled":true}\n'
    staged: list[tuple[str, str, str]] = []

    def stage(source: Path, *, destination_name: str, digest: str) -> bool:
        staged.append((source.name, destination_name, digest))
        return True

    monkeypatch.setattr(
        remote_jobs, "_matchup_runtime_companions", lambda: (tree, marker)
    )
    monkeypatch.setattr(remote_jobs, "_stage_elmo_file_via_ssh", stage)

    assert remote_jobs._stage_elmo_runtime_companions_via_ssh() is True
    assert staged[0] == (
        tree.name,
        tree.name,
        "sha256:" + hashlib.sha256(tree.read_bytes()).hexdigest(),
    )
    assert staged[1][1] == "matchup-runtime-activation.json"
    assert staged[1][2] == "sha256:" + hashlib.sha256(marker).hexdigest()


def test_elmo_missing_receipt_uses_storage_local_digest_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from poke_bot import checkpoint as checkpoint_module

    source = tmp_path / "trainer" / "iter_00003.pt"
    source.parent.mkdir()
    source.write_bytes(b"historical checkpoint without an old receipt")
    digest = checkpoint_digest(source)
    smb = tmp_path / "smb"
    smb.mkdir()
    destination = smb / digest_addressed_basename(source, digest=digest)
    destination.write_bytes(source.read_bytes())

    monkeypatch.setattr(remote_jobs, "_smb_checkpoint_dir", lambda: smb)
    monkeypatch.setattr(
        remote_jobs, "_ELMO_STAGE_RECEIPT_ROOT", tmp_path / "receipts"
    )
    monkeypatch.setattr(
        remote_jobs,
        "_elmo_remote_checkpoint_digest",
        lambda _host, _path: digest,
    )
    remote_jobs._ELMO_STAGE_CACHE.clear()
    original_digest = checkpoint_module.checkpoint_digest

    def _digest_without_remote_payload(path, *args, **kwargs):
        if Path(path).resolve() == destination.resolve():
            raise AssertionError("SMB payload must not be hashed")
        return original_digest(path, *args, **kwargs)

    monkeypatch.setattr(
        checkpoint_module, "checkpoint_digest", _digest_without_remote_payload
    )
    mapped = remote_jobs.resolve_remote_checkpoint_path(
        "192.168.1.143", str(source)
    )
    assert mapped.endswith(destination.name)
    assert remote_jobs._elmo_stage_receipt_is_exact(
        source, destination, digest
    )


def test_runtime_checkpoint_staging_builds_all_route_companions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = tmp_path / "runtime-tree.json"
    tree.write_text(
        json.dumps(
            {
                "runtime_contract": {
                    "accepted_archetype_ids": [
                        f"matchup-{index}" for index in range(22)
                    ],
                    "one_route_per_decision": True,
                    "unknown_route_exact_bypass": True,
                    "consecutive_required": 2,
                    "zero_materialized_adapters_allowed": True,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("POKEBOT_MATCHUP_ADAPTER_RUNTIME", "1")
    monkeypatch.setenv("POKEBOT_PUBLIC_MATCHUP_TREE_PATH", str(tree))

    source, marker_raw = remote_jobs._matchup_runtime_companions()
    marker = json.loads(marker_raw)

    assert source == tree
    assert marker["schema"] == "poke_bot.remote_matchup_runtime_activation/v1"
    assert marker["tree_file"] == tree.name
    assert len(marker["accepted_archetype_ids"]) == 22
    assert marker["continuous_reevaluation"] is True
    assert marker["one_route_per_decision"] is True


def test_runtime_checkpoint_staging_preserves_v6_registry_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = tmp_path / "runtime-tree-v6.json"
    route_targets = ["crustle", "teal-mask-ogerpon-ex"]
    tree.write_text(
        json.dumps(
            {
                "runtime_contract": {
                    "accepted_archetype_ids": ["teal-mask-ogerpon-ex"],
                    "one_route_per_decision": True,
                    "unknown_route_exact_bypass": True,
                    "consecutive_required": 2,
                    "zero_materialized_adapters_allowed": True,
                    "adapter_format": "poke-bot-matchup-adapter-bank-v6",
                    "route_target_ids": route_targets,
                    "route_physical_slots": [0, 18],
                    "physical_slot_capacity": 64,
                    "slot_registry_digest": "sha256:" + "a" * 64,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("POKEBOT_MATCHUP_ADAPTER_RUNTIME", "1")
    monkeypatch.setenv("POKEBOT_PUBLIC_MATCHUP_TREE_PATH", str(tree))

    source, marker_raw = remote_jobs._matchup_runtime_companions()
    marker = json.loads(marker_raw)

    assert source == tree
    assert marker["adapter_format"] == "poke-bot-matchup-adapter-bank-v6"
    assert marker["route_target_ids"] == route_targets
    assert marker["route_physical_slots"] == [0, 18]
    assert marker["physical_slot_capacity"] == 64
    assert marker["slot_registry_digest"] == "sha256:" + "a" * 64


def test_prepare_remote_play_job_stages_both_elmo_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged: list[str] = []

    def _stage(_host: str, path: str) -> str:
        staged.append(path)
        return f"/workspace/checkpoint/{Path(path).name}"

    monkeypatch.setattr(remote_jobs, "resolve_remote_checkpoint_path", _stage)
    original = {
        "checkpoint": "/home/inzi/poke-bot-agent/current.pt",
        "opponent_checkpoint": "/home/inzi/poke-bot-agent/previous.pt",
    }

    prepared = prepare_remote_play_job("192.168.1.143", original)

    assert staged == [original["checkpoint"], original["opponent_checkpoint"]]
    assert prepared["checkpoint"] == "/workspace/checkpoint/current.pt"
    assert prepared["opponent_checkpoint"] == "/workspace/checkpoint/previous.pt"
    assert original["checkpoint"].startswith("/home/inzi/")
    assert original["opponent_checkpoint"].startswith("/home/inzi/")


def test_prepare_remote_play_job_remaps_both_bert_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged: list[str] = []
    remapped: list[str] = []

    def _native(path: str) -> str:
        return path.replace(
            "/home/inzi/poke-bot-agent",
            "/Users/tsinzitari/workspace/poke-bot-agent",
        )

    def _stage(_host: str, path: str) -> str:
        staged.append(path)
        return _native(path)

    def _remap(_host: str, path: str) -> str:
        remapped.append(path)
        return _native(path)

    monkeypatch.setattr(remote_jobs, "resolve_remote_checkpoint_path", _stage)
    monkeypatch.setattr(remote_jobs, "resolve_remote_workdir_path", _remap)
    original = {
        "checkpoint": "/home/inzi/poke-bot-agent/current.pt",
        "opponent_checkpoint": "/home/inzi/poke-bot-agent/previous.pt",
        "spec": {"path": "/home/inzi/poke-bot-agent/baselines/agent.py"},
    }

    prepared = prepare_remote_play_job("bert.local", original)

    assert staged == [
        original["checkpoint"],
        original["opponent_checkpoint"],
    ]
    assert remapped == [original["spec"]["path"]]
    assert prepared["checkpoint"].startswith("/Users/tsinzitari/workspace/")
    assert prepared["opponent_checkpoint"].startswith(
        "/Users/tsinzitari/workspace/"
    )
    assert original["spec"]["path"].startswith("/home/inzi/")


def test_concurrent_bert_checkpoint_stage_publishes_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_root = tmp_path / "trainer"
    bert_root = Path("/Users/test/workspace/poke-bot-agent")
    checkpoint = train_root / "outputs" / "checkpoints" / "iter_00025.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"one immutable checkpoint")
    digest = checkpoint_digest(checkpoint)
    publishes = 0
    published = False
    companion_stages = 0

    def remote_digest(_path: Path) -> str | None:
        return digest if published else None

    def publish(_src: Path, _dest: Path, *, digest: str) -> None:
        nonlocal publishes, published
        publishes += 1
        time.sleep(0.02)
        published = True

    def stage_companions(_remote_dir: Path) -> None:
        nonlocal companion_stages
        companion_stages += 1

    monkeypatch.setattr(remote_jobs, "_TRAIN_ROOT", train_root)
    monkeypatch.setattr(remote_jobs, "_BERT_ROOT", bert_root)
    monkeypatch.setattr(remote_jobs, "_bert_sftp_root", lambda: None)
    monkeypatch.setattr(remote_jobs, "_bert_remote_digest", remote_digest)
    monkeypatch.setattr(remote_jobs, "_rsync_to_bert", publish)
    monkeypatch.setattr(remote_jobs, "_stage_bert_runtime_companions", stage_companions)
    remote_jobs._BERT_STAGE_CACHE.clear()
    barrier = threading.Barrier(24)
    results: list[str] = []

    def stage() -> None:
        barrier.wait()
        results.append(remote_jobs._stage_bert_checkpoint(checkpoint))

    threads = [threading.Thread(target=stage) for _ in range(24)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert publishes == 1
    assert companion_stages == 1
    assert len(set(results)) == 1
    assert digest.split(":", 1)[-1][:16] in results[0]


def test_concurrent_elmo_checkpoint_stage_publishes_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "trainer" / "seed.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"one immutable Elmo checkpoint")
    checkpoint_dir = tmp_path / "elmo-checkpoint"
    checkpoint_dir.mkdir()
    publishes = 0
    companion_stages = 0

    def publish(src: Path, dest: Path) -> None:
        nonlocal publishes
        publishes += 1
        time.sleep(0.02)
        dest.write_bytes(src.read_bytes())

    def stage_companions(_checkpoint_dir: Path) -> None:
        nonlocal companion_stages
        companion_stages += 1

    monkeypatch.setattr(remote_jobs, "_smb_checkpoint_dir", lambda: checkpoint_dir)
    monkeypatch.setattr(remote_jobs, "_gvfs_safe_copy", publish)
    monkeypatch.setattr(remote_jobs, "_stage_elmo_runtime_companions", stage_companions)
    remote_jobs._ELMO_STAGE_CACHE.clear()
    barrier = threading.Barrier(24)
    results: list[str] = []

    def stage() -> None:
        barrier.wait()
        results.append(
            remote_jobs.resolve_remote_checkpoint_path(
                "192.168.1.143",
                str(checkpoint),
            )
        )

    threads = [threading.Thread(target=stage) for _ in range(24)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert publishes == 1
    assert companion_stages == 1
    assert len(set(results)) == 1
    assert checkpoint_digest(checkpoint).split(":", 1)[-1][:16] in results[0]


@pytest.mark.parametrize(
    ("result", "reason_part"),
    [
        ({"our_failed": True, "error": "missing opponent checkpoint"}, "our_failed"),
        ({"resource_error": True, "error": "out of memory"}, "resource_error"),
        ({"type": "error", "error": "worker rejected job"}, "type=error"),
        ({"winner": 0, "error": "unexpected payload error"}, "error payload"),
    ],
)
def test_semantic_remote_failures_raise(
    result: dict[str, object], reason_part: str
) -> None:
    reason = remote_result_failure_reason(result)
    assert reason is not None
    assert reason_part in reason
    with pytest.raises(RemoteResultError) as exc_info:
        require_remote_result_success(result)
    assert exc_info.value.result is result


def test_valid_remote_result_is_unchanged() -> None:
    result = {
        "winner": 0,
        "our_failed": False,
        "resource_error": False,
        "cancelled": False,
        "error": None,
        "record_json": "{}",
    }

    assert remote_result_failure_reason(result) is None
    assert require_remote_result_success(result) is result


def test_public_play_stays_single_game_while_self_play_packs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POKEBOT_REMOTE_SELF_PLAY_MULTI_GAMES", "4")
    client = type(
        "PackedClient",
        (),
        {
            "info": RemoteWorkerInfo(
                endpoint="worker.example:8765",
                workers=4,
                leaf_servers=0,
                gpu_name="",
                device="cpu",
                checkpoint_digest=None,
                hostname="worker.example",
                job_kinds=("play", "self_play", "self_play_multi"),
            )
        },
    )()

    assert remote_jobs.client_self_play_multi_pack(client, kind="self_play") == 4
    assert remote_jobs.client_self_play_multi_pack(client, kind="play") == 0


def test_remote_multi_rejects_mixed_play_and_self_play_children() -> None:
    from poke_bot.remote_sim_jobs import remote_self_play_multi_job

    with pytest.raises(ValueError, match="mixes play and self-play"):
        remote_self_play_multi_job(
            {"jobs": [{"job_index": 1, "spec": {}}, {"job_index": 2}]}
        )


@pytest.mark.parametrize("count", [1, 5])
def test_remote_multi_rejects_public_packet_outside_exact_pack_size(
    count: int,
) -> None:
    """A forged public envelope may never reach LibcgMultiEnv outside 2..4."""

    from poke_bot.remote_sim_jobs import remote_self_play_multi_job

    with pytest.raises(
        ValueError,
        match="public self_play_multi packet must contain 2..4",
    ):
        remote_self_play_multi_job(
            {
                "jobs": [
                    {"job_index": index, "spec": {"id": f"public-{index}"}}
                    for index in range(count)
                ]
            }
        )


def test_client_rejects_same_count_multi_results_with_wrong_child_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Socket:
        timeout = 30.0

        def gettimeout(self) -> float:
            return self.timeout

        def settimeout(self, timeout: float) -> None:
            self.timeout = timeout

    jobs = [
        {
            "job_index": index,
            "opponent_id": f"self:opponent-{index}",
            "our_seat": index % 2,
            "game_timeout_s": 1,
        }
        for index in range(4)
    ]

    def _row(job: dict[str, object]) -> dict[str, object]:
        return {
            "job_index": job["job_index"],
            "opponent_id": job["opponent_id"],
            "our_seat": job["our_seat"],
            "winner": 0,
            "our_failed": False,
            "resource_error": False,
            "cancelled": False,
            "error": None,
            "record_json": "{}",
        }

    rows = [_row(job) for job in jobs]
    reply = {
        "type": "result",
        "ok": True,
        "result": {"self_play_multi": True, "results": rows},
    }
    monkeypatch.setattr(remote_jobs, "send_frame", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(remote_jobs, "read_frame", lambda _sock: reply)
    client = RemoteJobClient("unmapped.example")
    client._sock = _Socket()  # type: ignore[assignment]

    reply["result"] = {
        "self_play_multi": True,
        "results": [rows[2], rows[0], rows[3], rows[1]],
    }
    assert client.submit_self_play_multi(jobs) == rows

    reply["result"] = {
        "self_play_multi": True,
        "results": [rows[0], rows[0], rows[2], rows[3]],
    }
    with pytest.raises(remote_jobs.RemoteJobsError, match="duplicate child"):
        client.submit_self_play_multi(jobs)

    malformed = dict(rows[0])
    malformed["job_index"] = "not-an-index"
    reply["result"] = {
        "self_play_multi": True,
        "results": [malformed, rows[1], rows[2], rows[3]],
    }
    with pytest.raises(remote_jobs.RemoteJobsError, match="non-integer identity"):
        client.submit_self_play_multi(jobs)


def test_client_requires_public_multi_child_seed_and_checkpoint_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public Libcg packets may not credit a same-index stale child result."""

    class _Socket:
        timeout = 30.0

        def gettimeout(self) -> float:
            return self.timeout

        def settimeout(self, timeout: float) -> None:
            self.timeout = timeout

    jobs = [
        {
            "job_index": index,
            "opponent_id": f"public:opponent-{index}",
            "our_seat": index % 2,
            "seed": 70_000 + index,
            "checkpoint_digest": "sha256:" + str(index) * 64,
            "spec": {"id": f"public:opponent-{index}"},
            "game_timeout_s": 1,
        }
        for index in range(2)
    ]

    def _row(job: dict[str, object]) -> dict[str, object]:
        return {
            "job_index": job["job_index"],
            "opponent_id": job["opponent_id"],
            "our_seat": job["our_seat"],
            "seed": job["seed"],
            "checkpoint_digest": job["checkpoint_digest"],
            "winner": 0,
            "our_failed": False,
            "resource_error": False,
            "cancelled": False,
            "error": None,
            "record_json": "{}",
        }

    rows = [_row(job) for job in jobs]
    reply: dict[str, object] = {
        "type": "result",
        "ok": True,
        "result": {"self_play_multi": True, "results": rows},
    }
    monkeypatch.setattr(remote_jobs, "send_frame", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(remote_jobs, "read_frame", lambda _sock: reply)
    client = RemoteJobClient("unmapped.example")
    client._sock = _Socket()  # type: ignore[assignment]

    assert client.submit_self_play_multi(jobs) == rows

    for field, wrong_value in (
        ("seed", 9_999_999),
        ("checkpoint_digest", "sha256:" + "f" * 64),
    ):
        bad_rows = [dict(row) for row in rows]
        bad_rows[0][field] = wrong_value
        reply["result"] = {"self_play_multi": True, "results": bad_rows}
        with pytest.raises(
            remote_jobs.RemoteJobsError,
            match="public child proof mismatch",
        ):
            client.submit_self_play_multi(jobs)


def test_client_rejects_semantic_failure_before_returning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Socket:
        timeout = 30.0

        def gettimeout(self) -> float:
            return self.timeout

        def settimeout(self, timeout: float) -> None:
            self.timeout = timeout

    monkeypatch.setattr(remote_jobs, "send_frame", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        remote_jobs,
        "read_frame",
        lambda _sock: {
            "type": "result",
            "ok": True,
            "result": {
                "our_failed": True,
                "error": "opponent checkpoint does not exist",
            },
        },
    )
    client = RemoteJobClient("unmapped.example")
    client._sock = _Socket()  # type: ignore[assignment]

    with pytest.raises(RemoteResultError):
        client.submit_job({"game_timeout_s": 1})


@pytest.mark.parametrize("operation", ["control", "job"])
def test_disconnected_registered_client_reconnects_before_retry(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    """A present client with no socket must heal instead of failing the gate."""

    class _Socket:
        timeout: float | None = 30.0

        def gettimeout(self) -> float | None:
            return self.timeout

        def settimeout(self, timeout: float | None) -> None:
            self.timeout = timeout

    sent: list[dict[str, object]] = []
    reconnects: list[bool] = []
    client = RemoteJobClient("recycled-worker.example")
    assert client._sock is None

    def _reconnect() -> RemoteWorkerInfo:
        reconnects.append(True)
        client._sock = _Socket()  # type: ignore[assignment]
        return RemoteWorkerInfo(
            endpoint=client.endpoint,
            workers=1,
            leaf_servers=0,
            gpu_name="",
            device="cpu",
            checkpoint_digest=None,
            hostname=client.host,
        )

    monkeypatch.setattr(client, "reconnect", _reconnect)
    monkeypatch.setattr(
        remote_jobs,
        "send_frame",
        lambda _sock, message, **_kwargs: sent.append(message),
    )
    if operation == "control":
        monkeypatch.setattr(
            remote_jobs,
            "read_frame",
            lambda _sock: {"type": "reload_ok", "ok": True},
        )
        reply = client._control_call({"type": "reload", "path": "/tmp/a.pt"})
        assert reply["type"] == "reload_ok"
    else:
        monkeypatch.setattr(
            remote_jobs,
            "resolve_remote_checkpoint_path",
            lambda _h, path: path,
        )
        monkeypatch.setattr(
            remote_jobs,
            "read_frame",
            lambda _sock: {
                "type": "result",
                "ok": True,
                "result": {
                    "winner": 0,
                    "our_failed": False,
                    "resource_error": False,
                    "cancelled": False,
                    "error": None,
                    "record_json": "{}",
                },
            },
        )
        reply = client.submit_job({"game_timeout_s": 1})
        assert reply["winner"] == 0

    assert reconnects == [True]
    assert len(sent) == 1


def test_semantic_remote_failure_uses_existing_local_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("POKEBOT_REMOTE_NO_LOCAL_FALLBACK", raising=False)
    monkeypatch.delenv("POKEBOT_REMOTE_ONLY", raising=False)

    class _Pool:
        def imap_unordered(self, fn, jobs):
            yield from (fn(job) for job in jobs)

        def apply(self, fn, job):
            return fn(job)

    class _Remote:
        host = "unmapped.example"
        port = 8765
        endpoint = "unmapped.example:8765"
        info = RemoteWorkerInfo(
            endpoint=endpoint,
            workers=1,
            leaf_servers=0,
            gpu_name="",
            device="cpu",
            checkpoint_digest=None,
            hostname=host,
        )

        def ensure_alive(self) -> None:
            return None

        def submit_job(self, _job, *, kind="play"):
            raise RemoteResultError("our_failed: remote load error", {})

    def _local(job):
        return {"job_index": job["job_index"], "source": "local"}

    rows = list(
        iter_additive_results(
            local_pool=_Pool(),
            local_fn=_local,
            jobs=[{"job_index": 7}],
            remote_clients=[_Remote()],  # type: ignore[list-item]
            local_workers=0,
            remote_workers=1,
        )
    )

    assert rows == [{"job_index": 7, "source": "local"}]


class _ScheduledPool:
    def imap_unordered(self, fn, jobs):
        yield from (fn(job) for job in jobs)

    def apply(self, fn, job):
        return fn(job)


def _identity_job(job):
    return job


class _ScheduledRemote:
    host = "unmapped.example"
    port = 8765
    endpoint = "unmapped.example:8765"
    info = RemoteWorkerInfo(
        endpoint=endpoint,
        workers=1,
        leaf_servers=0,
        gpu_name="",
        device="cpu",
        checkpoint_digest=None,
        hostname=host,
        max_workers=1,
        default_workers=1,
    )

    def __init__(self, *, fail: bool) -> None:
        self.fail = fail

    def submit_job(self, job, *, kind="play"):
        if self.fail:
            raise RemoteResultError("our_failed: scheduled remote load error", {})
        return {"job_index": job["job_index"], "source": "remote"}

    def reconnect(self):
        return self.info

    def close(self) -> None:
        return None


class _ScheduledDecision:
    local_share = 0.0
    remote_share = 1.0
    remote_chunk = 8
    remote_demand = {_ScheduledRemote.endpoint: 1}


class _ScheduledScheduler:
    min_local_frac = 0.0
    prefer_local_frac = 0.0
    min_remote_frac = 1.0
    max_remote_frac = 1.0

    def decision(self):
        return _ScheduledDecision()

    def bind_remote_endpoints(self, _clients) -> None:
        return None

    def maybe_tick(self, **_kwargs):
        return None

    def note_completed(self, **_kwargs) -> None:
        return None

    def remote_demand(self):
        return dict(_ScheduledDecision.remote_demand)


def test_scheduled_remote_only_failure_never_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POKEBOT_REMOTE_ONLY", "1")
    monkeypatch.setenv("POKEBOT_REMOTE_JOB_RETRIES", "1")

    with pytest.raises(RemoteResultError, match="scheduled remote load error"):
        list(
            iter_scheduled_additive_results(
                local_pool=_ScheduledPool(),
                local_fn=lambda job: {
                    "job_index": job["job_index"],
                    "source": "local",
                },
                jobs=[{"job_index": 11}],
                remote_clients=[_ScheduledRemote(fail=True)],  # type: ignore[list-item]
                kind="self_play",
                scheduler=_ScheduledScheduler(),
                local_workers=1,
                remote_workers=1,
            )
        )


def test_scheduled_dispatch_reports_remote_execution_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POKEBOT_REMOTE_ONLY", "1")
    executions: list[dict[str, object]] = []

    rows = list(
        iter_scheduled_additive_results(
            local_pool=_ScheduledPool(),
            local_fn=lambda job: {
                "job_index": job["job_index"],
                "source": "local",
            },
            jobs=[{"job_index": 12}],
            remote_clients=[_ScheduledRemote(fail=False)],  # type: ignore[list-item]
            kind="self_play",
            scheduler=_ScheduledScheduler(),
            local_workers=1,
            remote_workers=1,
            on_execution=executions.append,
        )
    )

    assert rows == [{"job_index": 12, "source": "remote"}]
    assert executions == [
        {
            "origin": "remote",
            "endpoint": _ScheduledRemote.endpoint,
            "kind": "self_play",
            "transport_kind": "self_play",
            "pack_games": 1,
        }
    ]


def test_scheduled_packed_self_play_reports_four_logical_executions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A four-game transport pack remains four logical self-play results."""

    monkeypatch.setenv("POKEBOT_REMOTE_ONLY", "1")
    monkeypatch.setenv("POKEBOT_REMOTE_SELF_PLAY_MULTI_GAMES", "4")

    class PackedRemote(_ScheduledRemote):
        def __init__(self) -> None:
            super().__init__(fail=False)
            self.info = RemoteWorkerInfo(
                endpoint=self.endpoint,
                workers=1,
                leaf_servers=0,
                gpu_name="",
                device="cpu",
                checkpoint_digest=None,
                hostname=self.host,
                max_workers=1,
                default_workers=1,
                job_kinds=("play", "self_play", "self_play_multi"),
            )

        def submit_self_play_multi(self, jobs):
            return [
                {"job_index": job["job_index"], "source": "remote"}
                for job in jobs
            ]

    executions: list[dict[str, object]] = []
    rows = list(
        iter_scheduled_additive_results(
            local_pool=_ScheduledPool(),
            local_fn=lambda job: {
                "job_index": job["job_index"],
                "source": "local",
            },
            jobs=[{"job_index": index} for index in range(4)],
            remote_clients=[PackedRemote()],  # type: ignore[list-item]
            kind="self_play",
            scheduler=_ScheduledScheduler(),
            local_workers=1,
            remote_workers=1,
            on_execution=executions.append,
        )
    )

    assert [row["job_index"] for row in rows] == [0, 1, 2, 3]
    assert len(executions) == 4
    assert {
        (event["kind"], event["transport_kind"], event["pack_games"])
        for event in executions
    } == {("self_play", "self_play_multi", 1)}
    # The strict endpoint proof receives one logical completion per game,
    # while consumers that sum pack_games still see exactly four games.
    assert sum(int(event["pack_games"]) for event in executions) == 4


def test_scheduled_dispatch_notifies_after_producers_before_result_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POKEBOT_REMOTE_ONLY", "1")
    drained = threading.Event()
    callback_calls: list[str] = []

    def _on_drained() -> None:
        callback_calls.append("drained")
        drained.set()

    stream = iter_scheduled_additive_results(
        local_pool=_ScheduledPool(),
        local_fn=lambda job: job,
        jobs=[{"job_index": index} for index in range(32)],
        remote_clients=[_ScheduledRemote(fail=False)],  # type: ignore[list-item]
        kind="self_play",
        scheduler=_ScheduledScheduler(),
        local_workers=1,
        remote_workers=1,
        on_producers_drained=_on_drained,
    )

    first = next(stream)
    assert drained.wait(timeout=2.0)
    # The callback can fire while most durably queued rows are still waiting
    # for the serialized consumer, which is the memory-release opportunity.
    rows = [first, *stream]
    assert len(rows) == 32
    assert callback_calls == ["drained"]


def test_local_only_fallback_notifies_before_result_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("POKEBOT_REMOTE_ONLY", raising=False)
    drained = threading.Event()
    callback_calls: list[str] = []

    def _on_drained() -> None:
        callback_calls.append("drained")
        drained.set()

    stream = iter_scheduled_additive_results(
        local_pool=_ScheduledPool(),
        local_fn=lambda job: job,
        jobs=[{"job_index": index} for index in range(32)],
        remote_clients=[],
        kind="self_play",
        scheduler=_ScheduledScheduler(),
        local_workers=1,
        remote_workers=0,
        on_producers_drained=_on_drained,
    )

    first = next(stream)
    assert drained.wait(timeout=2.0)
    rows = [first, *stream]
    assert len(rows) == 32
    assert callback_calls == ["drained"]


def test_local_only_fallback_can_release_real_pool_without_losing_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("POKEBOT_REMOTE_ONLY", raising=False)
    with WorkerPool(num_workers=1) as pool:
        stream = iter_scheduled_additive_results(
            local_pool=pool,
            local_fn=_identity_job,
            jobs=[{"job_index": index} for index in range(24)],
            remote_clients=[],
            kind="self_play",
            scheduler=_ScheduledScheduler(),
            local_workers=1,
            remote_workers=0,
            on_producers_drained=pool.release,
        )
        rows = list(stream)
        assert pool._pool is None
    assert [row["job_index"] for row in rows] == list(range(24))


def test_producer_drain_callback_can_release_real_pool_without_losing_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POKEBOT_REMOTE_ONLY", "1")
    with WorkerPool(num_workers=1) as pool:
        stream = iter_scheduled_additive_results(
            local_pool=pool,
            local_fn=lambda job: job,
            jobs=[{"job_index": index} for index in range(24)],
            remote_clients=[_ScheduledRemote(fail=False)],  # type: ignore[list-item]
            kind="self_play",
            scheduler=_ScheduledScheduler(),
            local_workers=1,
            remote_workers=1,
            on_producers_drained=pool.release,
        )
        rows = list(stream)
        assert pool._pool is None
    assert [row["job_index"] for row in rows] == list(range(24))


def test_scheduled_dispatch_surfaces_producer_drain_callback_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POKEBOT_REMOTE_ONLY", "1")

    def _fail_release() -> None:
        raise RuntimeError("release failed")

    stream = iter_scheduled_additive_results(
        local_pool=_ScheduledPool(),
        local_fn=lambda job: job,
        jobs=[{"job_index": index} for index in range(8)],
        remote_clients=[_ScheduledRemote(fail=False)],  # type: ignore[list-item]
        kind="self_play",
        scheduler=_ScheduledScheduler(),
        local_workers=1,
        remote_workers=1,
        on_producers_drained=_fail_release,
    )
    with pytest.raises(RuntimeError, match="release failed"):
        list(stream)


def test_scheduled_emitter_closes_its_socket_before_outer_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed emitter must not retain a dead socket until wave teardown."""

    monkeypatch.setenv("POKEBOT_REMOTE_ONLY", "1")

    class ClosingRemote(_ScheduledRemote):
        def __init__(self) -> None:
            super().__init__(fail=False)
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    remote = ClosingRemote()
    # Exclude this client from the outer owned-client list so the assertion
    # specifically exercises the emitter's finally block.
    monkeypatch.setattr(
        remote_jobs,
        "_parallel_remote_slots",
        lambda *_args, **_kwargs: ([remote], []),
    )

    rows = list(
        iter_scheduled_additive_results(
            local_pool=_ScheduledPool(),
            local_fn=lambda job: job,
            jobs=[{"job_index": 13}],
            remote_clients=[remote],  # type: ignore[list-item]
            kind="self_play",
            scheduler=_ScheduledScheduler(),
            local_workers=1,
            remote_workers=1,
        )
    )

    assert len(rows) == 1
    assert remote.close_calls == 1


def test_scheduled_demand_shrink_finishes_claimed_chunks(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A shrink after ``remaining`` first hits zero must not strand a tail.

    The July 16 production run grew to two synthetic slots, then retired one
    inside its claimed chunk.  Returning that suffix after the local emitter
    had exited left no eligible claimant.  This test forces the same ordering
    and requires every job to be yielded exactly once.
    """

    monkeypatch.delenv("POKEBOT_REMOTE_ONLY", raising=False)
    monkeypatch.setenv("POKEBOT_REMOTE_NO_LOCAL_FALLBACK", "1")

    local_batch_started = threading.Event()
    local_batch_finished = threading.Event()
    both_remotes_started = threading.Event()
    release_remote_tail = threading.Event()
    first_remote_lock = threading.Lock()
    first_remote_count = 0
    consumer_thread = threading.current_thread()

    class _Pool(_ScheduledPool):
        def imap_unordered(self, fn, jobs):
            batch = list(jobs)
            assert len(batch) == 4
            local_batch_started.set()
            # Keep the local claim open until both remote slots own the entire
            # remainder; its next _claim then observes an empty deque and exits.
            assert both_remotes_started.wait(timeout=2.0)
            for job in batch:
                yield fn(job)
            local_batch_finished.set()

    class _Remote(_ScheduledRemote):
        def __init__(self) -> None:
            super().__init__(fail=False)
            self.calls = 0

        @property
        def endpoint(self) -> str:
            if threading.current_thread().name.startswith("sched-remote-"):
                assert local_batch_started.wait(timeout=2.0)
            return _ScheduledRemote.endpoint

        def submit_job(self, job, *, kind="play"):
            nonlocal first_remote_count
            self.calls += 1
            if self.calls == 1:
                with first_remote_lock:
                    first_remote_count += 1
                    if first_remote_count == 2:
                        both_remotes_started.set()
            if self.calls > 1:
                assert release_remote_tail.wait(timeout=2.0)
            return {"job_index": job["job_index"], "source": "remote"}

    class _Decision:
        local_share = 0.80
        remote_chunk = 8

        def __init__(self, demand: int) -> None:
            # Before shrink, let the two initial remote slots claim all eight
            # non-local jobs.  After shrink, a returned suffix would be above
            # this soft share and require the tail override -- proof that a
            # slot retired incorrectly inside an owned chunk.
            self.remote_share = 1.0 if demand == 2 else 0.20
            self.remote_demand = {_ScheduledRemote.endpoint: demand}

        def as_log(self) -> str:
            return f"demand={self.remote_demand}"

    class _Scheduler:
        min_local_frac = 0.40
        prefer_local_frac = 0.80
        min_remote_frac = 0.0
        max_remote_frac = 1.0

        def __init__(self) -> None:
            self.demand = 2
            self.shrunk = False

        def decision(self):
            return _Decision(self.demand)

        def bind_remote_endpoints(self, _clients) -> None:
            return None

        def maybe_tick(self, *, force=False, **_kwargs):
            if force:
                return self.decision()
            # _claim invokes maybe_tick for logging from emitter threads.  The
            # production shrink happens in the generator's consumer loop,
            # where _maybe_shrink_remote_slots can actually queue a token.
            if threading.current_thread() is not consumer_thread:
                return None
            if self.shrunk:
                return None
            self.shrunk = True
            self.demand = 1

            def _release_after_local_exit() -> None:
                assert local_batch_finished.wait(timeout=1.0)
                # Give _emit_local time to observe remaining==0 and publish its
                # done marker before either remote reaches a chunk boundary.
                threading.Event().wait(0.02)
                release_remote_tail.set()

            # Let the consumer queue the retire token before the slots move on
            # to their second already-claimed games.
            threading.Thread(
                target=_release_after_local_exit,
                name="release-remote-tail",
                daemon=True,
            ).start()
            return self.decision()

        def note_completed(self, **_kwargs) -> None:
            return None

        def remote_demand(self):
            return dict(self.decision().remote_demand)

    slots = [_Remote(), _Remote()]
    monkeypatch.setattr(
        remote_jobs,
        "_parallel_remote_slots",
        lambda *_args, **_kwargs: (slots, []),
    )
    clock = iter(range(0, 10000, 20))
    monkeypatch.setattr(remote_jobs.time, "monotonic", lambda: float(next(clock)))

    def _fail_on_idle_spin(_seconds: float) -> None:
        raise AssertionError(
            "remote slot entered the soft-capped idle spin after demand shrink"
        )

    # On the regressed mid-chunk-retire path the survivor spins forever over
    # the returned suffix.  Fail promptly instead of letting this regression
    # wedge the test process.
    monkeypatch.setattr(remote_jobs.time, "sleep", _fail_on_idle_spin)

    jobs = [{"job_index": i} for i in range(12)]
    rows = list(
        iter_scheduled_additive_results(
            local_pool=_Pool(),
            local_fn=lambda job: {
                "job_index": job["job_index"],
                "source": "local",
            },
            jobs=jobs,
            remote_clients=[slots[0]],  # type: ignore[list-item]
            kind="self_play",
            scheduler=_Scheduler(),
            local_workers=4,
            remote_workers=2,
        )
    )

    output = capsys.readouterr().out
    assert "demand_shrink_queue" in output
    assert "demand_shrink slots=2->1" in output
    assert "scheduled tail-drain override" not in output
    assert [slot.calls for slot in slots] == [4, 4]
    assert sorted(int(row["job_index"]) for row in rows) == list(range(12))
    assert len(rows) == len(jobs)


def test_scheduled_remote_drains_tail_while_local_batch_waits_for_straggler(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A live local emitter waiting on one batch straggler must not deadlock.

    The scheduler claims a fixed local-worker-sized batch and cannot refill it
    until ``imap_unordered`` yields the final result.  Once the remote side is
    above its cumulative soft share, both claimants otherwise wait forever:
    local waits for its straggler while remotes spin on a zero-sized claim and
    leave the unclaimed tail untouched.
    """

    monkeypatch.delenv("POKEBOT_REMOTE_ONLY", raising=False)
    monkeypatch.setenv("POKEBOT_REMOTE_NO_LOCAL_FALLBACK", "1")
    monkeypatch.setenv("PURE_RL_LOCAL_STRAGGLER_STALE_S", "30")

    local_batch_started = threading.Event()
    remote_soft_cap_fresh = threading.Event()
    advance_to_stale = threading.Event()
    remote_tail_drained = threading.Event()
    synthetic_now = [100.0]

    monkeypatch.setattr(
        remote_jobs.time, "monotonic", lambda: float(synthetic_now[0])
    )

    def _wait_for_stale(_seconds: float) -> None:
        # Reaching the idle path proves exact 2/24 is still soft-capped while
        # the most recent local completion is fresh.
        remote_soft_cap_fresh.set()
        assert advance_to_stale.wait(timeout=2.0)

    monkeypatch.setattr(remote_jobs.time, "sleep", _wait_for_stale)

    class _Pool(_ScheduledPool):
        def imap_unordered(self, fn, jobs):
            batch = list(jobs)
            assert len(batch) == 24
            local_batch_started.set()
            for job in batch[:-2]:
                yield fn(job)
            # The final two local results are stragglers.  They complete only
            # after the otherwise-idle remote slot is allowed to claim the
            # ten-job tail above its historical soft share.  First prove the
            # same 2/24 state is *not* enough while local progress is fresh.
            assert remote_soft_cap_fresh.wait(timeout=2.0)
            assert remote.calls == 26
            synthetic_now[0] = 131.0
            advance_to_stale.set()
            assert remote_tail_drained.wait(timeout=2.0)
            yield from (fn(job) for job in batch[-2:])

    class _Remote(_ScheduledRemote):
        def __init__(self) -> None:
            super().__init__(fail=False)
            self.calls = 0

        @property
        def endpoint(self) -> str:
            # ``threads`` starts the local emitter first.  Make that ordering
            # deterministic before this remote emitter takes its first claim.
            if threading.current_thread().name.startswith("sched-remote-"):
                assert local_batch_started.wait(timeout=2.0)
            return _ScheduledRemote.endpoint

        def submit_job(self, job, *, kind="play"):
            self.calls += 1
            if self.calls == 36:
                remote_tail_drained.set()
            return {"job_index": job["job_index"], "source": "remote"}

    class _Decision:
        local_share = 0.60
        remote_share = 0.40
        remote_chunk = 16
        remote_demand = {_ScheduledRemote.endpoint: 1}

        def as_log(self) -> str:
            return "local_share=.60 remote_share=.40"

    class _Scheduler:
        min_local_frac = 0.40
        prefer_local_frac = 0.60
        min_remote_frac = 0.0
        max_remote_frac = 0.40

        def decision(self):
            return _Decision()

        def bind_remote_endpoints(self, _clients) -> None:
            return None

        def maybe_tick(self, **_kwargs):
            return self.decision()

        def note_completed(self, **_kwargs) -> None:
            return None

        def remote_demand(self):
            return dict(_Decision.remote_demand)

    remote = _Remote()
    monkeypatch.setattr(
        remote_jobs,
        "_parallel_remote_slots",
        lambda *_args, **_kwargs: ([remote], []),
    )
    jobs = [{"job_index": i} for i in range(60)]

    rows = list(
        iter_scheduled_additive_results(
            local_pool=_Pool(),
            local_fn=lambda job: {
                "job_index": job["job_index"],
                "source": "local",
            },
            jobs=jobs,
            remote_clients=[remote],  # type: ignore[list-item]
            kind="self_play",
            scheduler=_Scheduler(),
            local_workers=24,
            remote_workers=1,
        )
    )

    output = capsys.readouterr().out
    assert "underfilled (2/24 outstanding; stale=31.0s/30.0s)" in output
    assert remote_tail_drained.is_set()
    assert sorted(int(row["job_index"]) for row in rows) == list(range(60))
    assert len(rows) == len(jobs)


def test_scheduled_tail_override_requires_stale_progress_or_done_emitter() -> None:
    """Fresh tail progress stays capped; stale tail/done handoffs are allowed."""

    assert not remote_jobs._local_batch_is_tail_straggled(24, 23)
    assert not remote_jobs._local_batch_is_tail_straggled(24, 3)
    assert remote_jobs._local_batch_is_tail_straggled(24, 2)
    assert not remote_jobs._local_tail_override_ready(
        emitter_done=False,
        batch_size=24,
        outstanding=2,
        last_progress_mono=100.0,
        now_mono=100.0,
        stale_s=30.0,
    )
    assert remote_jobs._local_tail_override_ready(
        emitter_done=False,
        batch_size=24,
        outstanding=2,
        last_progress_mono=100.0,
        now_mono=130.0,
        stale_s=30.0,
    )
    # A completed local emitter remains an immediate correctness handoff;
    # neither the near-tail count nor staleness grace applies.
    assert remote_jobs._local_tail_override_ready(
        emitter_done=True,
        batch_size=24,
        outstanding=24,
        last_progress_mono=100.0,
        now_mono=100.0,
        stale_s=30.0,
    )


def test_scheduled_initial_claim_round_reaches_each_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POKEBOT_REMOTE_ONLY", "1")
    monkeypatch.setattr(
        remote_jobs.threading,
        "Barrier",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("global remote start barrier reintroduces refill waves")
        ),
    )
    source = Path(remote_jobs.__file__).read_text(encoding="utf-8")
    assert "initial_remote_claims_ready.wait" not in source

    class EndpointRemote(_ScheduledRemote):
        def __init__(self, endpoint: str) -> None:
            super().__init__(fail=False)
            self.endpoint = endpoint
            self.host, port = endpoint.split(":")
            self.port = int(port)
            self.info = RemoteWorkerInfo(
                endpoint=endpoint,
                workers=1,
                leaf_servers=0,
                gpu_name="",
                device="cpu",
                checkpoint_digest=None,
                hostname=self.host,
                max_workers=1,
                default_workers=1,
            )

    endpoints = ["elmo.test:8765", "bert.test:8766"]

    class Decision(_ScheduledDecision):
        remote_demand = {endpoint: 1 for endpoint in endpoints}

    class Scheduler(_ScheduledScheduler):
        def decision(self):
            return Decision()

        def remote_demand(self):
            return dict(Decision.remote_demand)

    executions: list[dict[str, object]] = []
    rows = list(
        iter_scheduled_additive_results(
            local_pool=_ScheduledPool(),
            local_fn=lambda job: {"job_index": job["job_index"]},
            jobs=[{"job_index": index} for index in range(8)],
            remote_clients=[EndpointRemote(endpoint) for endpoint in endpoints],  # type: ignore[list-item]
            kind="self_play",
            scheduler=Scheduler(),
            local_workers=1,
            remote_workers=2,
            on_execution=executions.append,
        )
    )

    assert len(rows) == 8
    assert {str(row["endpoint"]) for row in executions} == set(endpoints)


def test_low_water_refills_elmo_and_bert_in_the_same_probe_round(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("POKEBOT_REMOTE_ONLY", "1")
    monkeypatch.setenv("POKEBOT_REMOTE_SOCKET_PREFETCH", "1")
    monkeypatch.setenv("POKEBOT_REMOTE_SOCKET_PREFETCH_MAX", "4")
    monkeypatch.setenv("POKEBOT_REMOTE_QUEUE_LOW_WATER_FRAC", "0.5")
    monkeypatch.setenv("POKEBOT_REMOTE_QUEUE_PROBE_S", "0.2")
    monkeypatch.setenv(
        "POKEBOT_REMOTE_ENDPOINT_CHUNKS",
        "elmo.test:8765=4,bert.test:8766=4",
    )

    endpoints = ["elmo.test:8765", "bert.test:8766"]
    reserve_started: list[str] = []

    class EndpointRemote(_ScheduledRemote):
        timeout_s = 5.0
        connect_timeout_s = 5.0
        control_timeout_s = 5.0

        def __init__(self, endpoint: str) -> None:
            super().__init__(fail=False)
            self.endpoint = endpoint
            self.host, port = endpoint.split(":")
            self.port = int(port)
            self.info = RemoteWorkerInfo(
                endpoint=endpoint,
                workers=1,
                leaf_servers=0,
                gpu_name="",
                device="cpu",
                checkpoint_digest=None,
                hostname=self.host,
                max_workers=1,
                default_workers=1,
            )

        def submit_job(self, job, *, kind="play"):
            # Keep the wave alive long enough to observe the independent probe.
            threading.Event().wait(0.03)
            return {"job_index": job["job_index"], "source": self.endpoint}

    class ProbeAndReserveClient(EndpointRemote):
        def __init__(self, host, port, **_kwargs) -> None:
            super().__init__(f"{host}:{port}")

        def connect(self):
            return self.info

        def health(self):
            return {"active_jobs": 0}

        def submit_job(self, job, *, kind="play"):
            reserve_started.append(self.endpoint)
            return {"job_index": job["job_index"], "source": self.endpoint}

    class Decision(_ScheduledDecision):
        remote_demand = {endpoint: 1 for endpoint in endpoints}

    class Scheduler(_ScheduledScheduler):
        def decision(self):
            return Decision()

        def remote_demand(self):
            return dict(Decision.remote_demand)

    monkeypatch.setattr(remote_jobs, "RemoteJobClient", ProbeAndReserveClient)
    stream = iter_scheduled_additive_results(
            local_pool=_ScheduledPool(),
            local_fn=lambda job: {"job_index": job["job_index"]},
            jobs=[{"job_index": index} for index in range(160)],
            remote_clients=[EndpointRemote(endpoint) for endpoint in endpoints],  # type: ignore[list-item]
            kind="self_play",
            scheduler=Scheduler(),
            local_workers=1,
            remote_workers=2,
        )
    # Pause result ingestion after the first row. The refill controller must
    # still run on its own 0.2 s cadence and top up both endpoints completely.
    first = next(stream)
    threading.Event().wait(0.6)
    rows = [first, *stream]

    assert len(rows) == 160
    assert set(reserve_started) == set(endpoints)
    output = capsys.readouterr().out
    assert "queue_refill_controller interval=0.200s" in output
    assert "elmo.test:8765 LOW_WATER_REFILL" in output
    assert "bert.test:8766 LOW_WATER_REFILL" in output
    assert output.count("fill=high_water") >= 2
    assert output.count("added=3") >= 2
    # The server sample is request-packet grain while controller credits are
    # exact logical games.  Keep both units visible so a packed public wave
    # cannot look starved merely because four child games share one request.
    assert "server_request_units=packets" in output
    assert "logical_credits_at_probe=" in output


def test_self_play_execution_wave_never_low_water_refills_second_socket_wave(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("POKEBOT_REMOTE_ONLY", "1")
    monkeypatch.setenv("POKEBOT_SELF_PLAY_ELMO_TAIL_ONLY", "1")
    monkeypatch.setenv("POKEBOT_SELF_PLAY_TAIL_WORK_STEAL_GAMES", "20")
    monkeypatch.setenv("POKEBOT_REMOTE_SOCKET_PREFETCH", "1")
    # Even a stale higher generic ceiling may not override the self-play hard cap.
    monkeypatch.setenv("POKEBOT_REMOTE_SOCKET_PREFETCH_MAX", "4")
    monkeypatch.setenv("POKEBOT_REMOTE_QUEUE_PROBE_S", "0.05")

    endpoints = ["elmo.test:8765", "bert.test:8766"]
    clone_submissions: list[str] = []
    slot_updates: list[dict[str, int]] = []

    class EndpointRemote(_ScheduledRemote):
        timeout_s = 5.0
        connect_timeout_s = 5.0
        control_timeout_s = 5.0

        def __init__(self, endpoint: str) -> None:
            super().__init__(fail=False)
            self.endpoint = endpoint
            self.host, port = endpoint.split(":")
            self.port = int(port)
            self.info = RemoteWorkerInfo(
                endpoint=endpoint,
                workers=1,
                leaf_servers=0,
                gpu_name="",
                device="cpu",
                checkpoint_digest=None,
                hostname=self.host,
                max_workers=1,
                default_workers=1,
            )

        def submit_job(self, job, *, kind="play"):
            threading.Event().wait(0.01)
            return {"job_index": job["job_index"], "source": self.endpoint}

    class ProbeOrCloneClient(EndpointRemote):
        def __init__(self, host, port, **_kwargs) -> None:
            super().__init__(f"{host}:{port}")

        def connect(self):
            return self.info

        def health(self):
            return {"active_jobs": 0}

        def submit_job(self, job, *, kind="play"):
            clone_submissions.append(self.endpoint)
            return {"job_index": job["job_index"], "source": self.endpoint}

    class Decision(_ScheduledDecision):
        remote_demand = {endpoint: 1 for endpoint in endpoints}

    class Scheduler(_ScheduledScheduler):
        def decision(self):
            return Decision()

        def remote_demand(self):
            return dict(Decision.remote_demand)

    monkeypatch.setattr(remote_jobs, "RemoteJobClient", ProbeOrCloneClient)
    rows = list(
        iter_scheduled_additive_results(
            local_pool=_ScheduledPool(),
            local_fn=lambda job: {"job_index": job["job_index"]},
            jobs=[{"job_index": index} for index in range(80)],
            remote_clients=[EndpointRemote(endpoint) for endpoint in endpoints],  # type: ignore[list-item]
            kind="self_play",
            scheduler=Scheduler(),
            local_workers=1,
            remote_workers=2,
            on_remote_slots=slot_updates.append,
        )
    )

    assert len(rows) == 80
    assert clone_submissions == []
    assert max(int(row.get("active", 0)) for row in slot_updates) <= 2
    assert "LOW_WATER_REFILL" not in capsys.readouterr().out


def test_public_listener_rejoin_replaces_refused_stale_prefetch_slots(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A restarted public listener cannot let stale ``have`` block r182 refill.

    The two original persistent sockets own their exact initial packets, then
    observe a refused reconnect while the listener is down.  Once a *fresh*
    health connection succeeds, the refill controller must open two bounded
    replacement sockets even though ``slots_by_endpoint`` still reports the
    prefetch-2 ceiling.  The old sockets retire only between packets, so no
    logical child/credit is released, duplicated, or repacked.
    """

    endpoint = "rejoin-public.test:8765"
    monkeypatch.setenv("POKEBOT_REMOTE_ONLY", "1")
    monkeypatch.setenv("POKEBOT_REMOTE_PUBLIC_SOCKET_PREFETCH", "2")
    monkeypatch.setenv("POKEBOT_REMOTE_PUBLIC_SOCKET_PREFETCH_MAX", "2")
    monkeypatch.setenv("POKEBOT_REMOTE_QUEUE_PROBE_S", "0.02")
    monkeypatch.setenv("POKEBOT_REMOTE_QUEUE_LOW_WATER_FRAC", "0.5")
    monkeypatch.setenv("POKEBOT_REMOTE_JOB_RETRIES", "16")
    monkeypatch.setenv(
        "POKEBOT_REMOTE_ENDPOINT_CHUNKS", f"{endpoint}=16"
    )

    listener_rejoined = threading.Event()
    stale_can_resume = threading.Event()
    both_refused = threading.Event()
    refused_lock = threading.Lock()
    refused_count = 0
    data_client_ids: set[int] = set()
    data_client_lock = threading.Lock()
    rejoin_packets: list[list[int]] = []
    rejoin_packets_lock = threading.Lock()

    def _info() -> RemoteWorkerInfo:
        return RemoteWorkerInfo(
            endpoint=endpoint,
            workers=1,
            leaf_servers=0,
            gpu_name="",
            device="cpu",
            checkpoint_digest=None,
            hostname="rejoin-public.test",
            max_workers=1,
            default_workers=1,
            job_kinds=("play", "self_play", "self_play_multi"),
            capabilities=("public_multi_env_allowlist_r182_v1",),
        )

    class StaleSlot:
        host = "rejoin-public.test"
        port = 8765
        endpoint = "rejoin-public.test:8765"
        timeout_s = 5.0
        connect_timeout_s = 5.0
        control_timeout_s = 5.0

        def __init__(self) -> None:
            self.info = _info()

        def reconnect(self):
            nonlocal refused_count
            if listener_rejoined.is_set():
                return self.info
            with refused_lock:
                refused_count += 1
                if refused_count >= 2:
                    both_refused.set()
            raise ConnectionRefusedError("listener restart in progress")

        def submit_self_play_multi(self, jobs):
            if not listener_rejoined.is_set():
                raise ConnectionResetError("listener restarted")
            # The replacement must be admitted before an old client may use
            # its existing packet again.  This simulates a dead socket still
            # sleeping in retry backoff while the restarted listener is live.
            assert stale_can_resume.wait(timeout=2.0)
            return [
                {"job_index": job["job_index"], "source": "stale-resumed"}
                for job in jobs
            ]

        def submit_job(self, job, *, kind="play"):
            if not listener_rejoined.is_set():
                raise ConnectionResetError("listener restarted")
            assert stale_can_resume.wait(timeout=2.0)
            return {"job_index": job["job_index"], "source": "stale-resumed"}

        def close(self) -> None:
            return None

    class RejoinedClient:
        timeout_s = 5.0
        connect_timeout_s = 5.0
        control_timeout_s = 5.0

        def __init__(self, host, port, **_kwargs) -> None:
            self.host = str(host)
            self.port = int(port)
            self.endpoint = f"{self.host}:{self.port}"
            self.info = _info()
            self._control = False

        def connect(self):
            if not listener_rejoined.is_set():
                raise ConnectionRefusedError("listener restart in progress")
            return self.info

        def health(self):
            self._control = True
            return {"active_jobs": 0}

        def submit_self_play_multi(self, jobs):
            assert not self._control
            with data_client_lock:
                data_client_ids.add(id(self))
            packet = [int(job["job_index"]) for job in jobs]
            with rejoin_packets_lock:
                rejoin_packets.append(packet)
            stale_can_resume.set()
            return [
                {"job_index": job_index, "source": "rejoined"}
                for job_index in packet
            ]

        def submit_job(self, job, *, kind="play"):
            assert not self._control
            with data_client_lock:
                data_client_ids.add(id(self))
            stale_can_resume.set()
            return {"job_index": job["job_index"], "source": "rejoined"}

        def close(self) -> None:
            return None

    class Decision(_ScheduledDecision):
        remote_chunk = 16
        remote_demand = {endpoint: 1}

    class Scheduler(_ScheduledScheduler):
        def decision(self):
            return Decision()

        def remote_demand(self):
            return dict(Decision.remote_demand)

    def _fast_backoff(_seconds: float) -> None:
        threading.Event().wait(0.003)

    def _rejoin_listener() -> None:
        assert both_refused.wait(timeout=2.0)
        listener_rejoined.set()

    opponent_id = "generic-heuristic"
    digest = (
        "sha256:6f89fa7b6c9734731809fc9782374e94e1d0eb75cde989b0c37ac5b4b4f15e65"
    )

    def _safe_job(index: int) -> dict[str, object]:
        return {
            "job_index": index,
            "opponent_id": opponent_id,
            "require_portable_baseline_contract": True,
            "spec": {
                "id": opponent_id,
                "content_digest": digest,
                "contract_schema": "poke_bot.portable_baseline_spec/v1",
            },
            "target_provenance": {
                "opponent_id": opponent_id,
                "opponent_content_digest": digest,
                "opponent_training_group": "diverse_public",
            },
        }

    template = StaleSlot()
    stale_slots = [StaleSlot(), StaleSlot()]
    monkeypatch.setattr(remote_jobs, "RemoteJobClient", RejoinedClient)
    monkeypatch.setattr(remote_jobs.time, "sleep", _fast_backoff)
    monkeypatch.setattr(
        remote_jobs,
        "_parallel_remote_slots",
        lambda *_args, **_kwargs: (stale_slots, []),
    )
    listener_thread = threading.Thread(target=_rejoin_listener, daemon=True)
    listener_thread.start()

    rows = list(
        iter_scheduled_additive_results(
            local_pool=_ScheduledPool(),
            local_fn=lambda job: {
                "job_index": job["job_index"],
                "source": "local",
            },
            jobs=[_safe_job(index) for index in range(64)],
            remote_clients=[template],  # type: ignore[list-item]
            kind="play",
            scheduler=Scheduler(),
            local_workers=1,
            remote_workers=2,
        )
    )

    listener_thread.join(timeout=1.0)
    assert listener_rejoined.is_set()
    assert stale_can_resume.is_set()
    # Exactly two stale prefetch slots were replaced, not an unbounded second
    # wave. Both fresh clones carried true four-child r182 transport packets.
    assert len(data_client_ids) == 2
    assert any(len(packet) == 4 for packet in rejoin_packets)
    assert all(1 <= len(packet) <= 4 for packet in rejoin_packets)
    assert sorted(int(row["job_index"]) for row in rows) == list(range(64))
    assert len({int(row["job_index"]) for row in rows}) == len(rows)
    output = capsys.readouterr().out
    assert "LISTENER_REJOIN_REFILL" in output
    assert "replaced=2" in output
    assert "logical_credits_preserved=true" in output


def test_endpoint_credit_keeps_slow_bert_fed_across_a_long_wave() -> None:
    """Repeated fast Elmo claims cannot consume Bert's refill allowance."""

    elmo = "192.168.1.143:8765"
    bert = "bert.local:8766"
    credits = remote_jobs._EndpointClaimCredits(total_jobs=6963)
    for _ in range(384):
        credits.register(elmo)
    for _ in range(128):
        credits.register(bert)
    credits.set_target(elmo, 48 + 192)
    credits.set_target(bert, 16 + 64)

    remaining = 6963

    def _claim(endpoint: str, want: int) -> int:
        nonlocal remaining
        count = credits.claim_limit(
            endpoint,
            want=want,
            remaining_count=remaining,
            wave_remaining=remaining,
        )
        credits.note_claimed(endpoint, count)
        remaining -= count
        return count

    # Even when Elmo reaches the lock first, both endpoints fill only their
    # bounded execution+queue allowance.
    assert _claim(elmo, 10_000) == 240
    assert _claim(bert, 10_000) == 80
    bert_claimed = 80

    # Reproduce a sustained speed disparity: Elmo completes/refills 100 jobs
    # before the slower Bert claimant next gets CPU time.  Bert's one-job
    # deficit remains protected through every intervening Elmo lock acquire.
    for _ in range(40):
        credits.note_finished(bert)
        for _ in range(100):
            credits.note_finished(elmo)
            assert _claim(elmo, 1) == 1
        assert _claim(elmo, 1) == 0
        assert _claim(bert, 1) == 1
        bert_claimed += 1

    assert bert_claimed == 120
    assert credits.inflight[bert] == 80
    assert remaining > credits.tail_jobs

    # Unused allowance spills only after an endpoint is unhealthy/done; the
    # explicit final-5% override is the other intentional escape hatch.
    spill = remote_jobs._EndpointClaimCredits(total_jobs=1000)
    spill.register(elmo)
    spill.register(bert)
    spill.set_target(elmo, 240)
    spill.set_target(bert, 80)
    assert spill.claim_limit(
        elmo, want=100, remaining_count=100, wave_remaining=201
    ) == 75
    spill.set_healthy(bert, False)
    assert spill.claim_limit(
        elmo, want=100, remaining_count=100, wave_remaining=201
    ) == 100
    spill.set_healthy(bert, True)
    assert spill.claim_limit(
        elmo, want=100, remaining_count=50, wave_remaining=50
    ) == 50


def test_tail_work_steal_returns_remote_reservations_to_local_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The final fifth remains shared instead of becoming a remote-only tail."""

    monkeypatch.setenv("POKEBOT_REMOTE_DEMAND_QUEUE", "1")
    monkeypatch.setenv(
        "POKEBOT_REMOTE_ENDPOINT_CHUNKS",
        "unmapped.example:8765=80",
    )
    monkeypatch.setenv("POKEBOT_REMOTE_TAIL_WORK_STEAL_FRACTION", "0.20")
    remote = _ScheduledRemote(fail=False)
    monkeypatch.setattr(
        remote_jobs,
        "_parallel_remote_slots",
        lambda *_args, **_kwargs: ([remote], []),
    )
    execution_origins: list[str] = []

    def _slow_remote(job, *, kind="play"):
        time.sleep(0.002)
        return {**job, "source": "remote"}

    remote.submit_job = _slow_remote  # type: ignore[method-assign]
    jobs = [
        {"job_index": index, "seat": index % 2, "seed": 9000 + index}
        for index in range(200)
    ]
    rows = list(
        iter_scheduled_additive_results(
            local_pool=_ScheduledPool(),
            local_fn=lambda job: {**job, "source": "local"},
            jobs=jobs,
            remote_clients=[remote],  # type: ignore[list-item]
            kind="play",
            scheduler=_ScheduledScheduler(),
            local_workers=8,
            remote_workers=1,
            on_execution=lambda row: execution_origins.append(
                str(row.get("origin"))
            ),
        )
    )

    assert len(rows) == len(jobs)
    assert sorted(int(row["job_index"]) for row in rows) == list(range(200))
    assert len({int(row["job_index"]) for row in rows}) == len(jobs)
    expected = {int(row["job_index"]): row for row in jobs}
    for row in rows:
        source = expected[int(row["job_index"])]
        assert row["seat"] == source["seat"]
        assert row["seed"] == source["seed"]
    assert "local" in execution_origins
    assert "remote" in execution_origins
    output = capsys.readouterr().out
    assert "tail_work_steal=start" in output
    assert "returned_to_shared=" in output
    assert "returned_to_shared=0" not in output
    assert "remote_claim_games=1 local_pool_remains_eligible=true" in output


def test_fixed_self_play_tail_count_overrides_percentage() -> None:
    credits = remote_jobs._EndpointClaimCredits(
        total_jobs=1024,
        tail_fraction=0.75,
        tail_jobs=20,
    )

    assert credits.tail_jobs == 20
    assert credits.tail_fraction == pytest.approx(20 / 1024)
    assert not credits.in_tail(21)
    assert credits.in_tail(20)


def test_scheduled_fast_elmo_cannot_starve_slow_bert_refills(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deterministic threaded regression for the iter26 Bert starvation.

    Eighty Bert request threads finish one game, release their in-flight
    credit, then pause before asking for another.  A single fast Elmo emitter
    repeatedly reacquires the shared claim lock.  It must eventually block on
    Bert's 16+64 reservation while more than the final 5% remains, allowing
    Bert to refill.  Every job must still be returned exactly once.
    """

    monkeypatch.setenv("POKEBOT_REMOTE_ONLY", "1")
    monkeypatch.setenv("POKEBOT_REMOTE_SOCKET_PREFETCH", "1")
    monkeypatch.setenv("POKEBOT_REMOTE_QUEUE_PROBE_S", "30")
    monkeypatch.setenv(
        "POKEBOT_REMOTE_MAX_WORKERS",
        "192.168.1.143:8765=48,bert.local:8766=16",
    )
    monkeypatch.setenv(
        "POKEBOT_REMOTE_ENDPOINT_CHUNKS",
        "192.168.1.143:8765=192,bert.local:8766=64",
    )

    elmo = "192.168.1.143:8765"
    bert = "bert.local:8766"
    all_bert_submitted = threading.Event()
    release_bert_callbacks = threading.Event()
    reservation_blocked_elmo = threading.Event()
    counts_lock = threading.Lock()
    calls = {elmo: 0, bert: 0}
    bert_calls_when_released = [-1]
    slot_updates: list[dict[str, int]] = []

    class EndpointRemote(_ScheduledRemote):
        timeout_s = 5.0
        connect_timeout_s = 5.0
        control_timeout_s = 5.0

        def __init__(self, endpoint: str, *, workers: int) -> None:
            super().__init__(fail=False)
            self.endpoint = endpoint
            self.host, port = endpoint.split(":")
            self.port = int(port)
            self.info = RemoteWorkerInfo(
                endpoint=endpoint,
                workers=workers,
                leaf_servers=0,
                gpu_name="",
                device="cpu",
                checkpoint_digest=None,
                hostname=self.host,
                max_workers=workers,
                default_workers=workers,
            )

        def submit_job(self, job, *, kind="play"):
            if self.endpoint == elmo:
                assert all_bert_submitted.wait(timeout=2.0)
            with counts_lock:
                calls[self.endpoint] += 1
                if self.endpoint == bert and calls[bert] == 80:
                    all_bert_submitted.set()
            return {"job_index": job["job_index"], "source": self.endpoint}

    class Decision(_ScheduledDecision):
        remote_chunk = 192
        remote_demand = {elmo: 48, bert: 16}

    class Scheduler(_ScheduledScheduler):
        def decision(self):
            return Decision()

        def remote_demand(self):
            return dict(Decision.remote_demand)

    def _on_execution(event: dict[str, object]) -> None:
        if event.get("endpoint") != bert:
            return
        if not release_bert_callbacks.wait(timeout=2.0):
            # Prevent a broken implementation from wedging pytest forever.
            release_bert_callbacks.set()
            raise AssertionError(
                "Elmo never yielded to Bert's reserved credit "
                f"(calls={dict(calls)})"
            )

    real_wait = threading.Event().wait

    def _idle_sleep(_seconds: float) -> None:
        # The only expected Elmo idle point is its credit boundary (>5% tail).
        if elmo in threading.current_thread().name:
            with counts_lock:
                if bert_calls_when_released[0] < 0:
                    bert_calls_when_released[0] = calls[bert]
            reservation_blocked_elmo.set()
            release_bert_callbacks.set()
        real_wait(0)

    monkeypatch.setattr(remote_jobs.time, "sleep", _idle_sleep)
    slots = [EndpointRemote(elmo, workers=48)] + [
        EndpointRemote(bert, workers=16) for _ in range(80)
    ]
    monkeypatch.setattr(
        remote_jobs,
        "_parallel_remote_slots",
        lambda *_args, **_kwargs: (slots, []),
    )

    jobs = [{"job_index": index} for index in range(1500)]
    rows = list(
        iter_scheduled_additive_results(
            local_pool=_ScheduledPool(),
            local_fn=lambda job: {"job_index": job["job_index"]},
            jobs=jobs,
            remote_clients=[slots[0], slots[1]],  # type: ignore[list-item]
            kind="self_play",
            scheduler=Scheduler(),
            local_workers=1,
            remote_workers=len(slots),
            on_remote_slots=slot_updates.append,
            on_execution=_on_execution,
        )
    )

    assert reservation_blocked_elmo.is_set()
    assert bert_calls_when_released[0] == 80
    assert calls[bert] > bert_calls_when_released[0]
    assert slot_updates[0]["active"] == len(slots)
    assert slot_updates[-1]["active"] == 0
    assert sorted(int(row["job_index"]) for row in rows) == list(range(1500))
    assert len(rows) == len(jobs)


def test_self_play_one_wave_gives_every_remote_socket_one_game_before_refill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A few emitter threads must not privately hoard the execution wave."""

    endpoint = "192.168.1.143:8765"
    monkeypatch.setenv("POKEBOT_REMOTE_DEMAND_QUEUE", "1")
    monkeypatch.setenv("POKEBOT_SELF_PLAY_ELMO_TAIL_ONLY", "1")
    monkeypatch.setenv("POKEBOT_SELF_PLAY_TAIL_WORK_STEAL_FRACTION", "0.05")
    monkeypatch.setenv("POKEBOT_REMOTE_SOCKET_PREFETCH", "1")
    monkeypatch.setenv("POKEBOT_REMOTE_SOCKET_PREFETCH_MAX", "1")
    first_wave = threading.Barrier(4)
    first_call_ids: set[int] = set()
    first_call_lock = threading.Lock()

    class SlotRemote(_ScheduledRemote):
        timeout_s = 5.0
        connect_timeout_s = 5.0
        control_timeout_s = 5.0

        def __init__(self, slot_id: int) -> None:
            super().__init__(fail=False)
            self.slot_id = slot_id
            self.endpoint = endpoint
            self.host = "192.168.1.143"
            self.port = 8765
            self.info = RemoteWorkerInfo(
                endpoint=endpoint,
                workers=4,
                leaf_servers=0,
                gpu_name="",
                device="cpu",
                checkpoint_digest=None,
                hostname=self.host,
                max_workers=4,
                default_workers=4,
            )

        def submit_job(self, job, *, kind="play"):
            with first_call_lock:
                first = self.slot_id not in first_call_ids
                first_call_ids.add(self.slot_id)
            if first:
                first_wave.wait(timeout=2.0)
            return {"job_index": job["job_index"], "source": self.slot_id}

    class Decision(_ScheduledDecision):
        remote_chunk = 128
        remote_demand = {endpoint: 4}

    class Scheduler(_ScheduledScheduler):
        def decision(self):
            return Decision()

        def remote_demand(self):
            return dict(Decision.remote_demand)

    slots = [SlotRemote(index) for index in range(4)]
    monkeypatch.setattr(
        remote_jobs,
        "_parallel_remote_slots",
        lambda *_args, **_kwargs: (slots, []),
    )
    slot_updates: list[dict[str, int]] = []
    jobs = [{"job_index": index} for index in range(64)]
    rows = list(
        iter_scheduled_additive_results(
            local_pool=_ScheduledPool(),
            local_fn=lambda job: {**job, "source": "local"},
            jobs=jobs,
            remote_clients=[slots[0]],  # type: ignore[list-item]
            kind="self_play",
            scheduler=Scheduler(),
            local_workers=4,
            remote_workers=4,
            on_remote_slots=slot_updates.append,
        )
    )

    assert first_call_ids == {0, 1, 2, 3}
    assert slot_updates[0]["active"] == 4
    assert slot_updates[0]["outstanding_elmo"] == 4
    assert slot_updates[0]["outstanding_bert"] == 0
    assert slot_updates[-1]["active"] == 0
    assert slot_updates[-1]["outstanding"] == 0
    assert sorted(int(row["job_index"]) for row in rows) == list(range(64))
