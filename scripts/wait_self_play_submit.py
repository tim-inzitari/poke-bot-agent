#!/usr/bin/env python3
"""Wait for notebook self-play to finish, then confirm Kaggle submission."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_agent.config import build_config
from poke_agent.paths import resolve_root


def load_manifest(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.is_file() else 0.0


def latest_iteration(manifest: dict) -> dict | None:
    iterations = manifest.get("iterations") or []
    return iterations[-1] if iterations else None


def format_status(manifest: dict) -> str:
    latest = latest_iteration(manifest)
    champion = manifest.get("champion") or {}
    parts = [
        f"iterations={len(manifest.get('iterations') or [])}",
        f"next_episode={manifest.get('next_episode', '?')}",
    ]
    if latest:
        win_rate = latest.get("eval_vs_random", {}).get("win_rate")
        if win_rate is not None:
            parts.append(f"latest_win_rate={float(win_rate):.1%}")
    champ_rate = champion.get("eval_vs_random", {}).get("win_rate")
    if champ_rate is not None:
        parts.append(f"champion_win_rate={float(champ_rate):.1%}")
    if manifest.get("stop_reason"):
        parts.append(f"stop={manifest['stop_reason']}")
    return ", ".join(parts)


def self_play_started(manifest: dict, *, baseline_iterations: int, baseline_mtime: float, path: Path) -> bool:
    if len(manifest.get("iterations") or []) > baseline_iterations:
        return True
    if manifest_mtime(path) > baseline_mtime + 1e-6:
        latest = latest_iteration(manifest) or {}
        if latest.get("training_report") is not None:
            return True
        if int(latest.get("rows_collected") or 0) > 50:
            return True
    return False


def print_submission_confirmation(manifest: dict) -> int:
    submission = manifest.get("kaggle_submission") or {}
    print("\n=== Kaggle submission confirmed ===")
    print(f"checkpoint: {submission.get('checkpoint', '?')}")
    print(f"tarball:    {submission.get('tarball', '?')}")
    print(f"message:    {submission.get('message', '?')}")
    output = (submission.get("kaggle_output") or "").strip()
    if output:
        print("kaggle:")
        print(output)
    return 0


def print_submission_error(manifest: dict) -> int:
    error = manifest.get("kaggle_submission_error") or {}
    print("\n=== Kaggle submission failed ===", file=sys.stderr)
    print(f"checkpoint: {error.get('checkpoint', '?')}", file=sys.stderr)
    print(f"returncode: {error.get('returncode', '?')}", file=sys.stderr)
    stderr = (error.get("stderr") or "").strip()
    stdout = (error.get("stdout") or "").strip()
    if stdout:
        print("stdout:", stdout, file=sys.stderr)
    if stderr:
        print("stderr:", stderr, file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Wait for self-play completion and confirm Kaggle submission.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="self-play manifest path (default: outputs/checkpoints/self_play/manifest.json)",
    )
    parser.add_argument("--poll-sec", type=int, default=30, help="poll interval in seconds")
    parser.add_argument(
        "--post-stop-timeout-sec",
        type=int,
        default=600,
        help="max seconds to wait for submission after stop_reason appears",
    )
    parser.add_argument(
        "--overall-timeout-sec",
        type=int,
        default=0,
        help="overall timeout (0 = no limit)",
    )
    args = parser.parse_args()

    root = resolve_root()
    config = build_config(root)
    manifest_path = args.manifest or Path(config["self_play"]["checkpoint_dir"])
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest_path = manifest_path / "manifest.json" if manifest_path.is_dir() else manifest_path

    baseline = load_manifest(manifest_path)
    baseline_iterations = len(baseline.get("iterations") or [])
    baseline_mtime = manifest_mtime(manifest_path)
    started_at = time.time()
    self_play_seen = False
    stop_seen_at: float | None = None

    print(f"watching {manifest_path}")
    print(f"baseline iterations={baseline_iterations}")
    print("waiting for self-play to start...")

    while True:
        now = time.time()
        if args.overall_timeout_sec > 0 and now - started_at > args.overall_timeout_sec:
            print("timed out waiting for self-play/submission", file=sys.stderr)
            return 2

        manifest = load_manifest(manifest_path)
        if manifest.get("kaggle_submission"):
            print(format_status(manifest))
            return print_submission_confirmation(manifest)
        if manifest.get("kaggle_submission_error"):
            print(format_status(manifest))
            return print_submission_error(manifest)

        if not self_play_seen:
            if self_play_started(
                manifest,
                baseline_iterations=baseline_iterations,
                baseline_mtime=baseline_mtime,
                path=manifest_path,
            ):
                self_play_seen = True
                print("self-play started")
            elif manifest.get("stop_reason"):
                self_play_seen = True
                print("self-play stop marker detected")

        if self_play_seen:
            if manifest.get("stop_reason"):
                if stop_seen_at is None:
                    stop_seen_at = now
                    print(f"self-play stopped: {manifest['stop_reason']}")
                    print("waiting for Kaggle submission record...")
                elif now - stop_seen_at > args.post_stop_timeout_sec:
                    print(
                        "self-play stopped but no kaggle_submission entry appeared in manifest",
                        file=sys.stderr,
                    )
                    return 3
            else:
                stamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{stamp}] {format_status(manifest)}")

        time.sleep(args.poll_sec)


if __name__ == "__main__":
    raise SystemExit(main())
