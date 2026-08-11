from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import stat
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from itertools import count
from pathlib import Path

import pytest

from replay_inspector.game_trace_cache import (
    GameTraceCache,
    GameTraceIdentity,
    TraceAddress,
    UnsafeCacheRootError,
)


def _digest(letter: str) -> str:
    return "sha256:" + letter * 64


def _identity(
    *, replay: str = "a", checkpoint: str = "b", episode_id: int = 88001
) -> GameTraceIdentity:
    return GameTraceIdentity(
        submission_id=77001,
        episode_id=episode_id,
        replay_sha256=_digest(replay),
        provenance={
            "checkpoint_sha256": _digest(checkpoint),
            "runtime_package_sha256": _digest("c"),
            "runtime_source_tree_sha256": _digest("d"),
            "runtime_parity_receipt_sha256": _digest("e"),
        },
    )


def _cache(tmp_path: Path, **kwargs: object) -> GameTraceCache:
    return GameTraceCache(
        tmp_path.resolve() / "trace-cache",
        unsafe_test_root=True,
        min_free_bytes=0,
        **kwargs,
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value)).hexdigest()


def _game_tree_size(root: Path) -> int:
    """Mirror the cache's logical accounting of regular files and directories."""

    total = 0
    for path in (root, *root.rglob("*")):
        file_stat = path.lstat()
        if stat.S_ISDIR(file_stat.st_mode) or stat.S_ISREG(file_stat.st_mode):
            total += file_stat.st_size
    return total


def test_put_read_restart_and_private_gzip_files(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    identity = _identity()
    address = TraceAddress(step_index=17, factorized_stage=2)
    payload = {"model": {"score": 0.25}, "heads": ["a", "b"]}

    assert cache.put(identity, address, payload)
    assert cache.read(identity, address).hit
    assert cache.read(identity, address).value == payload

    entry = cache.entry_path(identity, address)
    manifest = cache.manifest_path(identity)
    assert stat.S_IMODE(cache.root.stat().st_mode) == 0o700
    assert stat.S_IMODE(entry.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(entry.stat().st_mode) == 0o600
    assert stat.S_IMODE(manifest.stat().st_mode) == 0o600
    assert not list(cache.root.rglob("*.tmp"))

    with gzip.open(entry, "rt", encoding="utf-8") as source:
        record = json.load(source)
    assert record["identity_fingerprint"] == identity.fingerprint
    assert record["address"] == address.as_dict()
    assert record["payload_sha256"] == _sha(payload)

    # A new cache instance models a worker/server restart.  The completed
    # manifest and the independently checksummed trace must both survive.
    restarted = _cache(tmp_path)
    assert restarted.read(identity, address).value == payload
    completed = restarted.read_manifest(identity)
    assert completed is not None
    assert completed.addresses == (address,)
    assert completed.trace_sha256[address] == _sha(payload)


def test_raw_entry_without_completed_manifest_is_a_restart_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _cache(tmp_path)
    identity = _identity()
    addresses = (TraceAddress(1, 0), TraceAddress(2, 0))

    monkeypatch.setattr(cache, "_write_manifest_locked", lambda *_args: False)
    result = cache.get_or_materialize(
        identity,
        addresses,
        addresses[1],
        lambda address: {"step": address.step_index},
    )
    assert result == {"step": 2}
    assert cache.entry_path(identity, addresses[0]).is_file()
    assert not cache.manifest_path(identity).exists()

    # A process cannot promote partially written raw entries after a crash.
    restarted = _cache(tmp_path)
    assert not restarted.read(identity, addresses[0]).hit
    assert not restarted.read(identity, addresses[1]).hit


def test_manifest_binds_identity_and_per_trace_payload_checksum(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    identity = _identity()
    changed_replay = _identity(replay="f")
    changed_runtime = _identity(checkpoint="f")
    address = TraceAddress(4, 1)
    assert cache.put(identity, address, {"score": 1})

    assert not cache.read(changed_replay, address).hit
    assert not cache.read(changed_runtime, address).hit
    assert identity.cache_key != changed_replay.cache_key
    assert identity.cache_key != changed_runtime.cache_key

    entry = cache.entry_path(identity, address)
    with gzip.open(entry, "rt", encoding="utf-8") as source:
        record = json.load(source)
    # Rebuild the entry's own checksums around a different payload.  Its
    # manifest still carries the original per-trace SHA, so it must fail closed.
    record["payload"] = {"score": 999}
    record["payload_sha256"] = _sha(record["payload"])
    unsigned = {key: value for key, value in record.items() if key != "entry_sha256"}
    record["entry_sha256"] = _sha(unsigned)
    entry.write_bytes(gzip.compress(_canonical_json(record), mtime=0))
    os.chmod(entry, 0o600)

    assert not cache.read(identity, address).hit


def test_corrupt_or_insecure_entry_is_ignored(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    identity = _identity()
    address = TraceAddress(7, 3)
    assert cache.put(identity, address, {"trace": True})
    entry = cache.entry_path(identity, address)

    entry.write_bytes(b"not a gzip file")
    os.chmod(entry, 0o600)
    assert not cache.read(identity, address).hit

    assert cache.put(identity, address, {"trace": True})
    os.chmod(entry, 0o644)
    assert not cache.read(identity, address).hit


def test_production_root_policy_and_explicit_test_escape(tmp_path: Path) -> None:
    non_temporary_root = tmp_path.resolve() / "not-under-tmp"
    with pytest.raises(UnsafeCacheRootError):
        GameTraceCache(non_temporary_root, min_free_bytes=0)
    escaped = GameTraceCache(
        non_temporary_root, unsafe_test_root=True, min_free_bytes=0
    )
    assert escaped.root == non_temporary_root

    target = tmp_path.resolve() / "target"
    target.mkdir()
    root_symlink = tmp_path.resolve() / "cache-link"
    root_symlink.symlink_to(target, target_is_directory=True)
    with pytest.raises(UnsafeCacheRootError):
        GameTraceCache(root_symlink, unsafe_test_root=True, min_free_bytes=0)

    # macOS commonly exposes /tmp as a sanctioned /private/tmp symlink.  A
    # production child root must remain accepted, while /tmp itself is never a
    # valid destructive/eviction target.
    with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
        production = GameTraceCache(Path(temporary) / "cache", min_free_bytes=0)
        assert stat.S_IMODE(production.root.stat().st_mode) == 0o700
    with pytest.raises(UnsafeCacheRootError):
        GameTraceCache(Path("/tmp"), min_free_bytes=0)


def test_limits_and_low_free_space_refuse_cache_without_losing_result(
    tmp_path: Path,
) -> None:
    identity = _identity()
    address = TraceAddress(0, 0)

    no_retention = _cache(tmp_path / "bounded", max_entries=0)
    assert no_retention.put(identity, address, {"small": True}) is False
    assert not no_retention.read(identity, address).hit

    available = shutil.disk_usage(tmp_path).free
    low_space = GameTraceCache(
        tmp_path.resolve() / "low-space",
        unsafe_test_root=True,
        min_free_bytes=available + 1,
    )
    assert low_space.put(identity, address, {"small": True}) is False
    assert low_space.metrics.low_space_rejections >= 1

    calls = 0

    def compute() -> dict[str, bool]:
        nonlocal calls
        calls += 1
        return {"fresh": True}

    assert low_space.get_or_compute(identity, address, compute) == {"fresh": True}
    assert calls == 1
    assert not low_space.read(identity, address).hit


def test_game_materialization_singleflights_all_addresses_and_commits_once(
    tmp_path: Path,
) -> None:
    cache = _cache(tmp_path)
    identity = _identity()
    addresses = (TraceAddress(10, 0), TraceAddress(11, 1))
    started = threading.Event()
    release = threading.Event()
    calls: list[TraceAddress] = []
    calls_lock = threading.Lock()

    def build(address: TraceAddress) -> dict[str, int]:
        with calls_lock:
            calls.append(address)
            if len(calls) == 1:
                started.set()
        release.wait(timeout=5)
        return {"step": address.step_index, "stage": address.stage}

    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(
            cache.get_or_materialize,
            identity,
            addresses,
            addresses[0],
            build,
        )
        assert started.wait(timeout=5)
        second = workers.submit(
            cache.get_or_materialize,
            identity,
            addresses,
            addresses[1],
            build,
        )
        release.set()
        assert first.result(timeout=5) == {"step": 10, "stage": 0}
        assert second.result(timeout=5) == {"step": 11, "stage": 1}

    assert calls == list(addresses)
    assert cache.read_manifest(identity) is not None
    assert cache.read(identity, addresses[0]).hit
    assert cache.read(identity, addresses[1]).hit

    def must_not_run(_address: TraceAddress) -> dict[str, int]:
        raise AssertionError("completed physical game should be a disk-cache hit")

    assert (
        cache.get_or_materialize(
            identity, addresses, addresses[1], must_not_run
        )
        == {"step": 11, "stage": 1}
    )


def test_materialization_has_no_hidden_512_address_cap(tmp_path: Path) -> None:
    # Defaults are intentionally sufficient for more than 512 selectable
    # addresses; the only default retention limits are explicit byte/global
    # limits, not a hidden per-game completeness cap.
    cache = _cache(tmp_path)
    identity = _identity()
    addresses = tuple(TraceAddress(index, 0) for index in range(513))
    calls = 0

    def build(address: TraceAddress) -> dict[str, int]:
        nonlocal calls
        calls += 1
        return {"step": address.step_index}

    assert (
        cache.get_or_materialize(identity, addresses, addresses[-1], build)
        == {"step": 512}
    )
    assert calls == 513
    manifest = cache.read_manifest(identity)
    assert manifest is not None
    assert len(manifest.addresses) == 513

    restarted = _cache(tmp_path)
    assert restarted.read(identity, addresses[0]).value == {"step": 0}
    assert restarted.read(identity, addresses[-1]).value == {"step": 512}


def test_prune_evicts_a_whole_lru_game_not_just_a_trace_row(tmp_path: Path) -> None:
    ticks = count(1_786_408_000_000_000_000)
    cache = _cache(
        tmp_path,
        max_entries=2,
        max_entries_per_game=None,
        max_total_bytes=None,
        max_bytes_per_game=None,
        clock_ns=lambda: next(ticks),
    )
    first_game = _identity(episode_id=1)
    second_game = _identity(episode_id=2)
    first_addresses = (TraceAddress(0, 0), TraceAddress(1, 0))
    second_address = TraceAddress(0, 0)

    cache.get_or_materialize(
        first_game,
        first_addresses,
        first_addresses[0],
        lambda address: {"game": 1, "step": address.step_index},
    )
    first_root = cache.manifest_path(first_game).parent
    assert cache.manifest_path(first_game).is_file()

    cache.get_or_materialize(
        second_game,
        (second_address,),
        second_address,
        lambda _address: {"game": 2},
    )

    assert not first_root.exists()
    assert not cache.manifest_path(first_game).exists()
    assert not cache.entry_path(first_game, first_addresses[0]).exists()
    assert cache.read(second_game, second_address).value == {"game": 2}
    assert cache.metrics.evictions >= len(first_addresses)


def test_prune_accounts_for_manifest_and_directory_metadata(tmp_path: Path) -> None:
    cache = _cache(
        tmp_path,
        max_entries=None,
        max_entries_per_game=None,
        max_total_bytes=None,
        max_bytes_per_game=None,
    )
    identity = _identity()
    addresses = (TraceAddress(3, 0), TraceAddress(4, 1))
    cache.get_or_materialize(
        identity,
        addresses,
        addresses[0],
        lambda address: {"step": address.step_index},
    )
    game_root = cache.manifest_path(identity).parent
    measured_bytes = _game_tree_size(game_root)
    assert measured_bytes > 0
    assert cache.manifest_path(identity).is_file()

    # Reopening under a budget one byte below the full logical game footprint
    # proves the manifest and private directories participate in eviction.
    constrained = GameTraceCache(
        cache.root,
        unsafe_test_root=True,
        max_entries=None,
        max_entries_per_game=None,
        max_total_bytes=measured_bytes - 1,
        max_bytes_per_game=measured_bytes - 1,
        min_free_bytes=0,
    )
    assert constrained.prune() >= len(addresses)
    assert not game_root.exists()
    assert not constrained.manifest_path(identity).exists()


def test_identity_provenance_is_immutable_and_trace_address_aliases_work() -> None:
    identity = _identity()
    with pytest.raises(TypeError):
        identity.provenance["checkpoint_sha256"] = _digest("f")  # type: ignore[index]
    assert TraceAddress(5, factorized_stage=3) == TraceAddress(5, stage=3)
    with pytest.raises(ValueError):
        TraceAddress(5, stage=1, factorized_stage=2)
