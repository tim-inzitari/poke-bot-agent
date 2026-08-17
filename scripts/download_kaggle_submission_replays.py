#!/usr/bin/env python3
"""Download every currently listed replay for selected Kaggle submissions."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi
from requests import HTTPError


def _value(row: Any, *names: str) -> Any:
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


def _retry_http(call: Any, *args: Any, attempts: int = 8) -> Any:
    """Retry Kaggle's transient throttling without discarding completed files."""
    for attempt in range(attempts):
        try:
            return call(*args)
        except HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status != 429 or attempt + 1 == attempts:
                raise
            retry_after = getattr(exc.response, "headers", {}).get("Retry-After")
            delay = float(retry_after) if retry_after else min(64.0, 2.0 ** attempt)
            time.sleep(max(1.0, delay))


def _episode_payload(row: Any, submission_id: int, iteration: int) -> dict[str, Any]:
    agents = []
    for agent in list(_value(row, "agents") or []):
        agents.append(
            {
                "index": int(_value(agent, "index") or 0),
                "reward": _value(agent, "reward"),
                "submission_id": int(_value(agent, "submission_id", "submissionId") or 0),
                "team_id": int(_value(agent, "team_id", "teamId") or 0),
                "team_name": str(_value(agent, "team_name", "teamName") or ""),
            }
        )
    own = [agent for agent in agents if agent["submission_id"] == submission_id]
    if len(own) != 1:
        raise RuntimeError(
            f"episode {_value(row, 'id')} does not identify submission {submission_id} once"
        )
    return {
        "episode_id": int(_value(row, "id") or 0),
        "submission_id": submission_id,
        "iteration": iteration,
        "state": str(_value(row, "state") or ""),
        "type": str(_value(row, "type") or ""),
        "create_time": str(_value(row, "create_time", "createTime") or ""),
        "end_time": str(_value(row, "end_time", "endTime") or ""),
        "agents": agents,
        "own_agent": own[0],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", action="append", required=True,
                        help="SUBMISSION_ID:ITERATION")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--loss-logs", action="store_true")
    args = parser.parse_args()
    mapping = {}
    for value in args.submission:
        submission, iteration = value.split(":", 1)
        mapping[int(submission)] = int(iteration)

    api = KaggleApi()
    api.authenticate()
    metadata: list[dict[str, Any]] = []
    for submission_id, iteration in mapping.items():
        rows = _retry_http(api.competition_list_episodes, submission_id)
        metadata.extend(
            _episode_payload(row, submission_id, iteration)
            for row in rows
            if "COMPLETED" in str(_value(row, "state") or "").upper()
            and "PUBLIC" in str(_value(row, "type") or "").upper()
        )
    metadata.sort(key=lambda row: (row["iteration"], row["episode_id"]))
    args.output.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        args.output / "episodes.json",
        {"submissions": mapping, "episodes": metadata},
    )

    def download(row: dict[str, Any]) -> list[Path]:
        destination = args.output / f"iter_{row['iteration']:05d}"
        destination.mkdir(parents=True, exist_ok=True)
        replay = destination / f"episode-{row['episode_id']}-replay.json"
        if not replay.is_file():
            _retry_http(
                api.competition_episode_replay,
                row["episode_id"],
                str(destination),
            )
        paths = [replay]
        if args.loss_logs and float(row["own_agent"].get("reward") or 0.0) <= 0.0:
            index = int(row["own_agent"]["index"])
            log = destination / f"episode-{row['episode_id']}-agent-{index}-logs.json"
            if not log.is_file():
                try:
                    _retry_http(
                        api.competition_episode_agent_logs,
                        row["episode_id"],
                        index,
                        str(destination),
                    )
                except Exception:
                    return paths
            paths.append(log)
        return paths

    files: list[Path] = [args.output / "episodes.json"]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(download, row) for row in metadata]
        for future in as_completed(futures):
            files.extend(future.result())
    files = sorted(set(files))
    receipt = {
        "schema": "poke_bot.kaggle_submission_replay_download/v1",
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "submission_ids": sorted(mapping),
        "episode_count": len(metadata),
        "loss_count": sum(
            float(row["own_agent"].get("reward") or 0.0) <= 0.0 for row in metadata
        ),
        "files": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in files
        ],
    }
    _atomic_json(args.output / "DOWNLOAD_RECEIPT.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("status", "submission_ids", "episode_count", "loss_count")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
