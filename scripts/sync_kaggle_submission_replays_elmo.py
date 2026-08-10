#!/usr/bin/env python3
"""Incrementally archive Kaggle submission replays on Elmo by submission id."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import queue
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi
from requests import HTTPError

SCHEMA = "poke_bot.kaggle_submission_replay_archive/v1"
DEFAULT_COMPETITION = "pokemon-tcg-ai-battle"
DEFAULT_MIN_SUBMISSION_ID = 55315274
DEFAULT_API_TIMEOUT_SECONDS = 120.0
# These are deliberately additive only for ordinary owner-submission discovery.
# In particular, they do not lower ``DEFAULT_MIN_SUBMISSION_ID`` and they do
# not leak into the explicit ``--submission`` override path below.
DEFAULT_DISCOVERY_SPECIAL_SUBMISSION_IDS = (55217604,)


class KaggleCallTimeout(TimeoutError):
    """A single unresponsive Kaggle API request exceeded its hard deadline."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _value(row: Any, *names: str) -> Any:
    if isinstance(row, dict):
        for name in names:
            if name in row and row[name] is not None:
                return row[name]
        return None
    for name in names:
        value = getattr(row, name, None)
        if value is not None:
            return value
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _positive_timeout_seconds(value: Any) -> float:
    """Parse one finite, positive per-request deadline for argparse and callers."""

    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "API timeout must be a positive number"
        ) from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("API timeout must be a positive finite number")
    return seconds


def _call_with_timeout(
    call: Any,
    *args: Any,
    timeout_s: float,
    **kwargs: Any,
) -> Any:
    """Run one blocking API call behind a bounded daemon-thread wait.

    The Kaggle client methods used here expose no timeout parameter.  A hung
    request therefore runs in a daemon thread so the archive worker can record
    a partial result and finish.  On timeout we intentionally do not join the
    still-blocked inner call or retry it; the outer sync can continue with the
    remaining episodes.
    """

    deadline = _positive_timeout_seconds(timeout_s)
    result: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def invoke() -> None:
        try:
            result.put((True, call(*args, **kwargs)))
        except BaseException as exc:  # noqa: BLE001 - relay client exceptions
            result.put((False, exc))

    worker = threading.Thread(target=invoke, name="kaggle-api-call", daemon=True)
    worker.start()
    try:
        succeeded, value = result.get(timeout=deadline)
    except queue.Empty as exc:
        raise KaggleCallTimeout(
            f"Kaggle API call timed out after {deadline:g} seconds"
        ) from exc
    # ``invoke`` has enqueued its only result, so this can only wait for the
    # thread's immediate return and never waits for an unresponsive request.
    worker.join()
    if succeeded:
        return value
    if isinstance(value, subprocess.TimeoutExpired):
        raise KaggleCallTimeout(
            f"Kaggle API call timed out after {deadline:g} seconds"
        ) from value
    raise value


def _retry_http(
    call: Any,
    *args: Any,
    attempts: int = 8,
    timeout_s: float = DEFAULT_API_TIMEOUT_SECONDS,
) -> Any:
    for attempt in range(attempts):
        try:
            return _call_with_timeout(call, *args, timeout_s=timeout_s)
        except HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status != 429 or attempt + 1 == attempts:
                raise
            retry_after = getattr(exc.response, "headers", {}).get("Retry-After")
            delay = float(retry_after) if retry_after else min(64.0, 2.0**attempt)
            time.sleep(max(1.0, delay))


def _free_kib(path: Path) -> int:
    usage = os.statvfs(path if path.exists() else path.parent)
    return int(usage.f_bavail * usage.f_frsize / 1024)


def _ensure_free_space(archive: Path, minimum_free_gib: int) -> None:
    archive.mkdir(parents=True, exist_ok=True)
    free_kib = _free_kib(archive)
    if free_kib < minimum_free_gib * 1024 * 1024:
        raise SystemExit(
            f"free-space guard failed: {free_kib} KiB available "
            f"(need {minimum_free_gib} GiB)"
        )


def _episode_payload(row: Any, submission_id: int) -> dict[str, Any]:
    agents = []
    for agent in list(_value(row, "agents") or []):
        payload = {
            "index": int(_value(agent, "index") or 0),
            "reward": _value(agent, "reward"),
            "submission_id": int(_value(agent, "submission_id", "submissionId") or 0),
            "team_id": int(_value(agent, "team_id", "teamId") or 0),
            "team_name": str(_value(agent, "team_name", "teamName") or ""),
        }
        # Preserve only an explicit rank from Kaggle's episode response.  A
        # missing rank must remain absent so the read-only inspector can say
        # "rank unavailable" rather than manufacture one from reward/score.
        rank = _value(
            agent,
            "rank",
            "Rank",
            "ranking",
            "Ranking",
            "leaderboard_rank",
            "leaderboardRank",
        )
        if rank is not None:
            payload["rank"] = rank
        agents.append(payload)
    own = [agent for agent in agents if agent["submission_id"] == submission_id]
    if len(own) != 1:
        raise RuntimeError(
            f"episode {_value(row, 'id')} does not identify submission "
            f"{submission_id} once"
        )
    return {
        "episode_id": int(_value(row, "id") or 0),
        "submission_id": submission_id,
        "state": str(_value(row, "state") or ""),
        "type": str(_value(row, "type") or ""),
        "create_time": str(_value(row, "create_time", "createTime") or ""),
        "end_time": str(_value(row, "end_time", "endTime") or ""),
        "agents": agents,
        "own_agent": own[0],
    }


def _list_team_submission_ids(
    competition: str,
    minimum_id: int,
    *,
    timeout_s: float = DEFAULT_API_TIMEOUT_SECONDS,
) -> list[int]:
    """List normal owner submissions plus explicit recurring special cases.

    The minimum remains the floor for ordinary Kaggle discovery.  A special
    case is intentionally unioned afterwards so an owner-approved historical
    submission can be rechecked for newly published games without widening the
    recurring discovery window.
    """
    completed = _call_with_timeout(
        subprocess.run,
        [
            "kaggle",
            "competitions",
            "submissions",
            competition,
            "-v",
            "--page-size",
            "200",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        timeout_s=timeout_s,
    )
    if completed.returncode:
        raise RuntimeError(
            "cannot query Kaggle submissions: "
            + (completed.stderr or completed.stdout).strip()
        )
    found: set[int] = set()
    for row in csv.DictReader(io.StringIO(completed.stdout)):
        raw = str(row.get("ref") or row.get("submissionId") or "").strip()
        if not raw.isdigit():
            continue
        submission_id = int(raw)
        if submission_id >= minimum_id:
            found.add(submission_id)
    # Always include the floor id so bootstrap can proceed even if listing lags.
    found.add(minimum_id)
    found.update(DEFAULT_DISCOVERY_SPECIAL_SUBMISSION_IDS)
    return sorted(found)


def _select_submission_ids(
    competition: str,
    minimum_id: int,
    explicit_submission_ids: list[int] | None,
    *,
    timeout_s: float = DEFAULT_API_TIMEOUT_SECONDS,
) -> list[int]:
    """Select sync targets without changing explicit CLI override semantics."""

    if explicit_submission_ids:
        # ``--submission`` has always overridden discovery.  Keep that path
        # exact: it neither discovers current submissions nor auto-adds the
        # recurring historical special case.
        return sorted(
            {
                submission_id
                for submission_id in explicit_submission_ids
                if submission_id >= minimum_id
            }
        )
    return _list_team_submission_ids(competition, minimum_id, timeout_s=timeout_s)


def _download_one(
    api: KaggleApi,
    destination: Path,
    row: dict[str, Any],
    *,
    loss_logs: bool,
    api_timeout_s: float = DEFAULT_API_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    episode_id = int(row["episode_id"])
    replay = destination / f"episode-{episode_id}-replay.json"
    downloaded: list[str] = []
    reused: list[str] = []

    if replay.is_file() and replay.stat().st_size > 0:
        reused.append(replay.name)
    else:
        try:
            _retry_http(
                api.competition_episode_replay,
                episode_id,
                str(destination),
                timeout_s=api_timeout_s,
            )
        except KaggleCallTimeout as exc:
            raise KaggleCallTimeout(
                f"episode {episode_id} replay download: {exc}"
            ) from exc
        if not replay.is_file():
            raise RuntimeError(f"replay missing after download: {replay}")
        downloaded.append(replay.name)

    reward = float(row["own_agent"].get("reward") or 0.0)
    if loss_logs and reward <= 0.0:
        index = int(row["own_agent"]["index"])
        log = destination / f"episode-{episode_id}-agent-{index}-logs.json"
        if log.is_file() and log.stat().st_size > 0:
            reused.append(log.name)
        else:
            try:
                _retry_http(
                    api.competition_episode_agent_logs,
                    episode_id,
                    index,
                    str(destination),
                    timeout_s=api_timeout_s,
                )
            except KaggleCallTimeout as exc:
                # A timeout must reach the executor so the submission receipt
                # is partial rather than silently claiming a complete sync.
                raise KaggleCallTimeout(
                    f"episode {episode_id} agent-log download: {exc}"
                ) from exc
            except Exception as exc:  # noqa: BLE001 - keep replay durable
                return {
                    "episode_id": episode_id,
                    "downloaded": downloaded,
                    "reused": reused,
                    "log_error": str(exc),
                }
            if log.is_file() and log.stat().st_size > 0:
                downloaded.append(log.name)
    return {
        "episode_id": episode_id,
        "downloaded": downloaded,
        "reused": reused,
    }


def _sync_submission(
    api: KaggleApi,
    archive: Path,
    submission_id: int,
    *,
    workers: int,
    loss_logs: bool,
    api_timeout_s: float = DEFAULT_API_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    destination = archive / str(submission_id)
    destination.mkdir(parents=True, exist_ok=True)
    rows = _retry_http(
        api.competition_list_episodes,
        submission_id,
        timeout_s=api_timeout_s,
    )
    metadata = [
        _episode_payload(row, submission_id)
        for row in rows
        if "COMPLETED" in str(_value(row, "state") or "").upper()
        and "PUBLIC" in str(_value(row, "type") or "").upper()
    ]
    metadata.sort(key=lambda row: row["episode_id"])
    _atomic_json(
        destination / "episodes.json",
        {
            "submission_id": submission_id,
            "episode_count": len(metadata),
            "updated_at_utc": _utc_now(),
            "episodes": metadata,
        },
    )

    results: list[dict[str, Any]] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [
            executor.submit(
                _download_one,
                api,
                destination,
                row,
                loss_logs=loss_logs,
                api_timeout_s=api_timeout_s,
            )
            for row in metadata
        ]
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001 - continue other episodes
                errors.append(str(exc))

    downloaded_files: list[dict[str, Any]] = []
    new_replay_episode_ids: list[int] = []
    for result in sorted(results, key=lambda item: item["episode_id"]):
        for name in result.get("downloaded") or []:
            path = destination / name
            downloaded_files.append(
                {
                    "path": name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
            if name.endswith("-replay.json") and name.startswith("episode-"):
                new_replay_episode_ids.append(int(result["episode_id"]))

    present_replays = sorted(destination.glob("episode-*-replay.json"))
    present_logs = sorted(destination.glob("episode-*-agent-*-logs.json"))
    loss_count = sum(
        float(row["own_agent"].get("reward") or 0.0) <= 0.0 for row in metadata
    )
    status = "complete" if not errors else "partial"
    receipt = {
        "schema": SCHEMA,
        "status": status,
        "created_at_utc": _utc_now(),
        "submission_id": submission_id,
        "episode_count": len(metadata),
        "replay_count_on_disk": len(present_replays),
        "loss_count": loss_count,
        "agent_log_count_on_disk": len(present_logs),
        "downloaded_this_run": len(downloaded_files),
        "new_replay_episode_ids": new_replay_episode_ids,
        "files_downloaded_this_run": downloaded_files,
        "errors": errors,
    }
    _atomic_json(destination / "SYNC_RECEIPT.json", receipt)
    return {
        "submission_id": submission_id,
        "status": status,
        "episode_count": len(metadata),
        "replay_count_on_disk": len(present_replays),
        "loss_count": loss_count,
        "agent_log_count_on_disk": len(present_logs),
        "downloaded_this_run": len(downloaded_files),
        "new_replay_episode_ids": new_replay_episode_ids,
        "error_count": len(errors),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path(
            os.environ.get(
                "POKEBOT_SUBMISSION_REPLAY_ARCHIVE",
                "/mnt/Main/main/poke-bot-agent/archive/submission-replays",
            )
        ),
    )
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument(
        "--min-submission-id",
        type=int,
        default=int(
            os.environ.get(
                "POKEBOT_SUBMISSION_REPLAY_MIN_ID", str(DEFAULT_MIN_SUBMISSION_ID)
            )
        ),
    )
    parser.add_argument(
        "--submission",
        action="append",
        type=int,
        default=None,
        help="Optional explicit submission id(s); overrides discovery when set",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--api-timeout-s",
        type=_positive_timeout_seconds,
        default=os.environ.get(
            "POKEBOT_SUBMISSION_REPLAY_API_TIMEOUT_S",
            str(DEFAULT_API_TIMEOUT_SECONDS),
        ),
        help="per-Kaggle-call timeout in seconds; timed-out calls are not retried",
    )
    parser.add_argument(
        "--loss-logs", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--min-free-gib", type=int, default=200)
    args = parser.parse_args()

    archive: Path = args.archive
    _ensure_free_space(archive, args.min_free_gib)

    submission_ids = _select_submission_ids(
        args.competition,
        args.min_submission_id,
        args.submission,
        timeout_s=args.api_timeout_s,
    )
    if args.submission and not submission_ids:
        raise SystemExit("no submission ids remain after min-submission-id filter")

    api = KaggleApi()
    api.authenticate()

    summaries: list[dict[str, Any]] = []
    hard_failure: str | None = None
    for submission_id in submission_ids:
        _ensure_free_space(archive, args.min_free_gib)
        print(
            f"[sync] begin submission_id={submission_id}",
            flush=True,
        )
        try:
            summary = _sync_submission(
                api,
                archive,
                submission_id,
                workers=args.workers,
                loss_logs=bool(args.loss_logs),
                api_timeout_s=args.api_timeout_s,
            )
        except Exception as exc:  # noqa: BLE001 - keep prior submissions durable
            hard_failure = f"submission {submission_id}: {exc}"
            summaries.append(
                {
                    "submission_id": submission_id,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            print(
                f"[sync] failed submission_id={submission_id} error={exc}", flush=True
            )
            continue
        summaries.append(summary)
        print(
            "[sync] done " + json.dumps(summary, sort_keys=True),
            flush=True,
        )

    new_downloads = {
        "schema": "poke_bot.kaggle_submission_replay_new_downloads/v1",
        "created_at_utc": _utc_now(),
        "competition": args.competition,
        "min_submission_id": args.min_submission_id,
        "submissions": [
            {
                "submission_id": int(row["submission_id"]),
                "episode_ids": sorted(
                    {int(value) for value in row.get("new_replay_episode_ids") or []}
                ),
            }
            for row in summaries
            if row.get("status") in {"complete", "partial"}
            and row.get("new_replay_episode_ids")
        ],
    }
    new_downloads["episode_count"] = sum(
        len(row["episode_ids"]) for row in new_downloads["submissions"]
    )
    _atomic_json(archive / "NEW_DOWNLOADS.json", new_downloads)

    registry = {
        "schema": SCHEMA,
        "status": (
            "failed"
            if hard_failure
            and not any(row.get("status") == "complete" for row in summaries)
            else (
                "partial"
                if hard_failure
                or any(row.get("status") == "partial" for row in summaries)
                else "complete"
            )
        ),
        "updated_at_utc": _utc_now(),
        "competition": args.competition,
        "min_submission_id": args.min_submission_id,
        "submission_ids": submission_ids,
        "new_download_episode_count": new_downloads["episode_count"],
        "submissions": summaries,
    }
    _atomic_json(archive / "registry.json", registry)
    print(
        json.dumps(
            {
                "status": registry["status"],
                "submission_ids": submission_ids,
                "new_download_episode_count": new_downloads["episode_count"],
                "submissions": summaries,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 1 if hard_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
