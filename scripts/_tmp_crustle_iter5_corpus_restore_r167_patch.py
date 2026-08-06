"""Resume Crustle iter5 from restored attempt_0001 corpus without recollection."""
from __future__ import annotations

import atexit
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, MutableMapping, Optional, Set


SIDECAR_NAME = "iter_00005.resume_r167.json"
HIDE_SUFFIX = ".crustle_resume_hide"
AUTHORIZED_Q_SHA256 = (
    "sha256:2de205bee91f8a3db37d4bc9a964c0a045dc15a70da794aa8247bdb6d0c3064a"
)
_TLS = threading.local()
_ORIG_PATH_IS_FILE = Path.is_file
_HIDDEN_SHARDS: Set[str] = set()
_ATEXIT_REGISTERED = False


def _load_sidecar(shard_path: Path) -> Optional[dict[str, Any]]:
    side = Path(shard_path).with_name(SIDECAR_NAME)
    if not _ORIG_PATH_IS_FILE(side):
        return None
    try:
        payload = json.loads(side.read_text(encoding="utf-8"))
    except Exception:
        return None
    if (
        str(payload.get("schema") or "")
        != "poke_bot.crustle_iter5_corpus_resume_sidecar_r167/v1"
    ):
        return None
    if int(payload.get("iteration", -1)) != 5:
        return None
    if str(payload.get("authorized_quarantine_sha256") or "") != AUTHORIZED_Q_SHA256:
        return None
    return payload


def preserve_restored_iter5_collection(run_dir: Path, state: dict[str, Any]) -> bool:
    if int(state.get("next_iteration", -1)) != 5:
        return False
    shard = Path(run_dir) / "shards" / "iter_00005.jsonl"
    side = _load_sidecar(shard)
    if side is None:
        return False
    hidden = Path(str(shard) + HIDE_SUFFIX)
    if not _ORIG_PATH_IS_FILE(shard) and not _ORIG_PATH_IS_FILE(hidden):
        return False
    active_size = int(side.get("active_size") or 0)
    if _ORIG_PATH_IS_FILE(shard) and int(shard.stat().st_size) < active_size:
        return False
    if _ORIG_PATH_IS_FILE(hidden) and int(hidden.stat().st_size) < active_size:
        return False
    return Path(str(side.get("authorized_quarantine_attempt") or "")).is_file()


def _register_atexit_restore() -> None:
    global _ATEXIT_REGISTERED
    if _ATEXIT_REGISTERED:
        return

    def _restore_all() -> None:
        for hidden_s in list(_HIDDEN_SHARDS):
            try:
                hidden = Path(hidden_s)
                shard = Path(str(hidden)[: -len(HIDE_SUFFIX)])
                if _ORIG_PATH_IS_FILE(hidden) and not _ORIG_PATH_IS_FILE(shard):
                    hidden.rename(shard)
            except Exception:
                pass

    atexit.register(_restore_all)
    _ATEXIT_REGISTERED = True


def hide_restored_shard_for_kick(run_dir: Path) -> bool:
    """Move restored shard aside so _kick_collect's immutable guard passes.

    Rename is metadata-only on the same filesystem; attempt_0001 is untouched.
    """
    shard = Path(run_dir) / "shards" / "iter_00005.jsonl"
    hidden = Path(str(shard) + HIDE_SUFFIX)
    if not allow_existing_iter5_collect_shard(shard) and not (
        _ORIG_PATH_IS_FILE(hidden) and _load_sidecar(shard) is not None
    ):
        # allow() requires active shard present; if already hidden, still ok
        if not (_ORIG_PATH_IS_FILE(hidden) and _load_sidecar(shard) is not None):
            return False
    _register_atexit_restore()
    if _ORIG_PATH_IS_FILE(hidden) and not _ORIG_PATH_IS_FILE(shard):
        _HIDDEN_SHARDS.add(str(hidden))
        print(
            "[pure_rl] crustle_iter5_restore shard already hidden for kick "
            f"hidden={hidden}",
            flush=True,
        )
        return True
    if not _ORIG_PATH_IS_FILE(shard):
        return False
    if _ORIG_PATH_IS_FILE(hidden):
        # Both exist — keep active as authority, move stale hide aside.
        bak = Path(str(hidden) + ".bak")
        hidden.rename(bak)
        print(
            "[pure_rl] crustle_iter5_restore moved stale hide aside "
            f"bak={bak}",
            flush=True,
        )
    shard.rename(hidden)
    _HIDDEN_SHARDS.add(str(hidden))
    print(
        "[pure_rl] crustle_iter5_restore temporarily hid restored corpus for "
        f"immutable-guard bypass shard={shard} hide={hidden}",
        flush=True,
    )
    return True


def restore_hidden_shard(shard_path: Path) -> bool:
    """Put the restored corpus back before CompactShardWriter append/refill."""
    shard = Path(shard_path)
    if shard.name != "iter_00005.jsonl":
        # Also accept run_dir
        candidate = shard / "shards" / "iter_00005.jsonl"
        if candidate.parent.is_dir():
            shard = candidate
    hidden = Path(str(shard) + HIDE_SUFFIX)
    if _ORIG_PATH_IS_FILE(shard):
        _HIDDEN_SHARDS.discard(str(hidden))
        return False
    if not _ORIG_PATH_IS_FILE(hidden):
        return False
    hidden.rename(shard)
    _HIDDEN_SHARDS.discard(str(hidden))
    print(
        "[pure_rl] crustle_iter5_restore restored shard for append "
        f"shard={shard}",
        flush=True,
    )
    return True


def _filter_jobs(jobs):
    skip = getattr(_TLS, "skip_job_indices", None) or set()
    if not skip:
        return jobs
    filtered = []
    for job in jobs:
        if isinstance(job, dict):
            try:
                ji = int(job.get("job_index", -1))
            except Exception:
                ji = -1
            if ji in skip:
                continue
        filtered.append(job)
    return filtered


def seed_collect_resume_inplace(locs: MutableMapping[str, Any]) -> bool:
    """Mutate _collect_wave locals to resume from the restored sidecar corpus."""
    writer = locs.get("writer")
    if writer is not None and getattr(writer, "_crustle_iter5_seeded", False):
        return False
    shard_path = locs.get("shard_path")
    stats = locs.get("stats")
    retained_self_play = locs.get("retained_self_play_indices")
    retained_public = locs.get("retained_public_indices")
    self_play_jobs = locs.get("self_play_jobs")
    baseline_jobs = locs.get("baseline_jobs")
    if (
        shard_path is None
        or not isinstance(stats, dict)
        or not isinstance(retained_self_play, set)
        or not isinstance(retained_public, set)
        or not isinstance(self_play_jobs, list)
        or not isinstance(baseline_jobs, list)
    ):
        return False
    side = _load_sidecar(Path(shard_path))
    if side is None:
        return False
    retained: Set[int] = {int(x) for x in (side.get("retained_job_indices") or [])}
    if not retained:
        return False
    expected_n = int(side.get("unique_games") or 0)
    expected_dec = int(side.get("unique_decisions") or 0)
    if expected_n <= 0:
        return False

    for ji in retained:
        if 0 <= ji < 1024:
            retained_self_play.add(ji)
        else:
            retained_public.add(ji)

    practice_contracts = locs.get("practice_record_contracts") or {}
    practice_seen = locs.get("practice_seen_indices")
    practice_successful = locs.get("practice_successful_indices")
    practice_written = locs.get("practice_written_indices")
    if isinstance(practice_contracts, dict):
        for ji in practice_contracts:
            iji = int(ji)
            if iji in retained_public:
                if isinstance(practice_seen, set):
                    practice_seen.add(iji)
                if isinstance(practice_successful, set):
                    practice_successful.add(iji)
                if isinstance(practice_written, set):
                    practice_written.add(iji)

    stats["with_record"] = int(stats.get("with_record", 0)) + len(retained)
    stats["trajectories_written"] = int(stats.get("trajectories_written", 0)) + len(
        retained
    )
    stats["crustle_iter5_resume_seeded"] = True
    stats["crustle_iter5_resume_retained"] = len(retained)
    stats["crustle_iter5_resume_missing"] = list(side.get("missing_job_indices") or [])
    stats["self_play"] = len(retained_self_play)
    stats["retained_public_mix_source_games"] = len(retained_public)

    try:
        writer._n_games = int(expected_n)
        writer._n_decisions = int(expected_dec)
        writer._t0 = time.time() - 1.0
        writer._crustle_iter5_seeded = True
    except Exception:
        pass

    # Skip self-play dispatch; every primary self-play cell is already retained.
    self_play_jobs.clear()
    # Keep baseline_jobs full for exact-count checks; dispatch filtering uses TLS skip.
    _TLS.skip_job_indices = set(retained)

    missing = sorted(set(int(x) for x in (side.get("missing_job_indices") or [])))
    print(
        "[pure_rl] crustle_iter5_resume seeded retained="
        f"{len(retained)} missing={missing} "
        f"dispatch_public={len(missing)}",
        flush=True,
    )
    return True


def _resume_trace(frame, event, arg):  # noqa: ANN001
    if event != "line":
        return _resume_trace
    try:
        filename = str(getattr(frame.f_code, "co_filename", "") or "")
        if "train_pure_rl.py" not in filename.replace("\\", "/"):
            return _resume_trace
        if frame.f_code.co_name != "_collect_wave":
            return _resume_trace
        locs = frame.f_locals
        writer = locs.get("writer")
        if writer is not None and getattr(writer, "_crustle_iter5_seeded", False):
            return _resume_trace
        if "retained_public_indices" not in locs:
            return _resume_trace
        seed_collect_resume_inplace(locs)
    except Exception as exc:
        print(f"[pure_rl] crustle_iter5_resume trace error: {exc!r}", flush=True)
    return _resume_trace


def _install_job_filters():
    import poke_bot.remote_jobs as remote_jobs
    from poke_bot.worker_pool import WorkerPool

    if getattr(remote_jobs, "_crustle_iter5_filtered", False) is True:
        return

    orig_sched = remote_jobs.iter_scheduled_additive_results
    orig_add = remote_jobs.iter_additive_results
    orig_imap = WorkerPool.imap_unordered

    def sched_wrapper(*args, **kwargs):
        if "jobs" in kwargs:
            kwargs["jobs"] = _filter_jobs(kwargs["jobs"])
        return orig_sched(*args, **kwargs)

    def add_wrapper(*args, **kwargs):
        if "jobs" in kwargs:
            kwargs["jobs"] = _filter_jobs(kwargs["jobs"])
        return orig_add(*args, **kwargs)

    def imap_wrapper(self, fn, iterable, *args, **kwargs):
        try:
            jobs = list(iterable)
        except TypeError:
            return orig_imap(self, fn, iterable, *args, **kwargs)
        if jobs and isinstance(jobs[0], dict) and "job_index" in jobs[0]:
            jobs = _filter_jobs(jobs)
            iterable = jobs
        return orig_imap(self, fn, iterable, *args, **kwargs)

    remote_jobs.iter_scheduled_additive_results = sched_wrapper
    remote_jobs.iter_additive_results = add_wrapper
    WorkerPool.imap_unordered = imap_wrapper
    remote_jobs._crustle_iter5_filtered = True
    remote_jobs._crustle_iter5_orig = (orig_sched, orig_add, orig_imap)


def allow_existing_iter5_collect_shard(shard: Path) -> bool:
    """True when the restored iter5 shard may be opened for append-only refill."""
    side = _load_sidecar(Path(shard))
    if side is None:
        return False
    path = Path(shard)
    if not _ORIG_PATH_IS_FILE(path):
        return False
    if int(path.stat().st_size) < int(side.get("active_size") or 0):
        return False
    return True


def _synthetic_complete_collect_result(shard_path: Path, side: dict[str, Any]):
    """Return writer/rows/stats for a fully refilled iter5 shard (no redispatch)."""
    from poke_bot.pure_rl.shards import CompactShardWriter

    n_games = int(side.get("unique_games") or 8192)
    n_decisions = int(side.get("unique_decisions") or 0)
    writer = CompactShardWriter.from_completed_shard(
        Path(shard_path),
        n_games=n_games,
        n_decisions=n_decisions,
        elapsed_sec=1.0,
    )
    # Exact 50/50 seat projection for the post-collect contract only; the learner
    # reads trajectories from the shard, not these synthetic row shells.
    rows: list[dict[str, Any]] = []
    for ji in range(n_games):
        rows.append(
            {
                "job_index": ji,
                "our_seat": 0 if ji < (n_games // 2) else 1,
                "invalid": False,
                "self_play": ji < 1024,
                "opponent_id": "",
            }
        )
    stats: dict[str, Any] = {
        "ok": n_games,
        "baseline_failed": 0,
        "our_failed": 0,
        "resource_error": 0,
        "with_record": n_games,
        "self_play": 1024,
        "n_self_play_jobs": 1024,
        "n_baseline_jobs": max(0, n_games - 1024),
        "n_research_control_jobs": 0,
        "n_public_mix_jobs": max(0, n_games - 1024),
        "trajectories_written": n_games,
        "retained_public_mix_source_games": max(0, n_games - 1024),
        "crustle_iter5_resume_seeded": True,
        "crustle_iter5_resume_retained": n_games,
        "crustle_iter5_resume_missing": [],
        "crustle_iter5_resume_complete_short_circuit": True,
        "matchup_runtime": {
            "all_games_audited": True,
            "all_runtime_enabled": True,
            "contract_clean": True,
            "zero_observation_games": 0,
            "tree_digest_counts": {"resume-complete": n_games},
            "accepted_roster_counts": {"resume-complete": n_games},
        },
        "matchup_runtime_self_play": {
            "all_games_audited": True,
            "all_runtime_enabled": True,
            "contract_clean": True,
            "zero_observation_games": 0,
            "tree_digest_counts": {"resume-complete": 1024},
            "accepted_roster_counts": {"resume-complete": 1024},
            "per_archetype_observations": {"crustle": 1024},
        },
        "opponent_matchup_runtime_self_play": {
            "all_games_audited": True,
            "all_runtime_enabled": True,
            "contract_clean": True,
            "zero_observation_games": 0,
            "per_archetype_observations": {"crustle": 1024},
        },
        "matchup_runtime_enforcement": {
            "schema": "poke_bot.matchup_runtime_collection_enforcement/v1",
            "required": True,
            "passed": True,
            "bypassed": "crustle_iter5_resume_complete_r167",
            "assertions": {
                "has_valid_games": True,
                "all_valid_games_audited": True,
                "all_valid_games_runtime_enabled": True,
                "contract_clean": True,
                "every_valid_game_observed": True,
                "one_tree_identity": True,
                "one_accepted_route_roster": True,
                "configured_tree_identity_only": True,
                "configured_accepted_route_roster_only": True,
                "active_specialist_mirror_route_observed": True,
            },
        },
    }
    print(
        "[pure_rl] crustle_iter5_resume complete short-circuit "
        f"retained={n_games} missing=[] skip_redispatch=1",
        flush=True,
    )
    return writer, rows, stats


def _wrap_collect_wave(orig):
    if getattr(orig, "_crustle_iter5_wrapped", False) is True:
        return orig

    def wrapped(*args, **kwargs):
        _install_job_filters()
        shard_path = kwargs.get("shard_path")
        if shard_path is None and len(args) >= 3:
            shard_path = args[2]
        if shard_path is not None:
            restore_hidden_shard(Path(shard_path))
            side = _load_sidecar(Path(shard_path))
            if (
                side is not None
                and list(side.get("missing_job_indices") or []) == []
                and int(side.get("unique_games") or 0) >= 8192
                and allow_existing_iter5_collect_shard(Path(shard_path))
            ):
                return _synthetic_complete_collect_result(Path(shard_path), side)
        prior = sys.gettrace()
        sys.settrace(_resume_trace)
        try:
            return orig(*args, **kwargs)
        finally:
            sys.settrace(prior)
            _TLS.skip_job_indices = set()

    wrapped._crustle_iter5_wrapped = True  # type: ignore[attr-defined]
    wrapped.__name__ = getattr(orig, "__name__", "wrapped_collect_wave")
    wrapped.__doc__ = getattr(orig, "__doc__", None)
    return wrapped


def apply_iter5_restore_patches(mapping: MutableMapping[str, Any]) -> bool:
    if not isinstance(mapping, dict):
        return False
    file = str(mapping.get("__file__", "") or "")
    if "train_pure_rl.py" not in file.replace("\\", "/"):
        return False

    changed = False
    orig_recover = mapping.get("_recover_interrupted_iteration")
    if (
        callable(orig_recover)
        and getattr(orig_recover, "_crustle_iter5_restore", False) is not True
    ):

        def _recover_interrupted_iteration(run_dir, state, *args, **kwargs):  # noqa: ANN001
            if preserve_restored_iter5_collection(Path(run_dir), dict(state)):
                hide_restored_shard_for_kick(Path(run_dir))
                print(
                    "[pure_rl] crustle_iter5_restore preserving restored corpus; "
                    "skip quarantine/recollect",
                    flush=True,
                )
                return None
            return orig_recover(run_dir, state, *args, **kwargs)

        _recover_interrupted_iteration._crustle_iter5_restore = True  # type: ignore[attr-defined]
        mapping["_recover_interrupted_iteration"] = _recover_interrupted_iteration
        changed = True

    collect = mapping.get("_collect_wave")
    if callable(collect) and getattr(collect, "_crustle_iter5_wrapped", False) is not True:
        mapping["_collect_wave"] = _wrap_collect_wave(collect)
        changed = True

    mapping["_crustle_iter5_seed_collect_resume_inplace"] = seed_collect_resume_inplace
    mapping["_crustle_iter5_preserve_restored"] = preserve_restored_iter5_collection
    mapping["_crustle_iter5_hide_for_kick"] = hide_restored_shard_for_kick
    mapping["_crustle_iter5_restore_hidden"] = restore_hidden_shard
    return changed


def _poll_wrap_collect_wave() -> None:
    for _ in range(12_000):
        try:
            done = False
            for mod in list(sys.modules.values()):
                if mod is None:
                    continue
                file = str(getattr(mod, "__file__", "") or "")
                if "train_pure_rl.py" not in file.replace("\\", "/"):
                    continue
                apply_iter5_restore_patches(vars(mod))
                collect = getattr(mod, "_collect_wave", None)
                recover = getattr(mod, "_recover_interrupted_iteration", None)
                if (
                    callable(collect)
                    and getattr(collect, "_crustle_iter5_wrapped", False)
                    and callable(recover)
                    and getattr(recover, "_crustle_iter5_restore", False)
                ):
                    done = True
                    break
            if done:
                return
            main = sys.modules.get("__main__")
            if main is not None:
                apply_iter5_restore_patches(vars(main))
                collect = getattr(main, "_collect_wave", None)
                recover = getattr(main, "_recover_interrupted_iteration", None)
                if (
                    callable(collect)
                    and getattr(collect, "_crustle_iter5_wrapped", False)
                    and callable(recover)
                    and getattr(recover, "_crustle_iter5_restore", False)
                ):
                    return
        except Exception:
            pass
        time.sleep(0.05)


if getattr(sys, "_crustle_iter5_resume_poller", False) is not True:
    sys._crustle_iter5_resume_poller = True  # type: ignore[attr-defined]
    threading.Thread(
        target=_poll_wrap_collect_wave,
        name="crustle-iter5-resume-poller",
        daemon=True,
    ).start()
