"""Remote checkpoint staging and result validation regressions."""

from __future__ import annotations

from pathlib import Path
import json
import threading
import time

import pytest

from poke_bot.checkpoint import checkpoint_digest
from poke_bot import remote_jobs
from poke_bot.remote_jobs import (
    RemoteJobClient,
    RemoteResultError,
    RemoteWorkerInfo,
    digest_addressed_basename,
    iter_additive_results,
    iter_scheduled_additive_results,
    prepare_remote_play_job,
    remote_result_failure_reason,
    require_remote_result_success,
)


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
        lambda _sock, message: sent.append(message),
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
        }
    ]


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
        elmo, want=100, remaining_count=100, wave_remaining=100
    ) == 75
    spill.set_healthy(bert, False)
    assert spill.claim_limit(
        elmo, want=100, remaining_count=100, wave_remaining=100
    ) == 100
    spill.set_healthy(bert, True)
    assert spill.claim_limit(
        elmo, want=100, remaining_count=50, wave_remaining=50
    ) == 50


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
