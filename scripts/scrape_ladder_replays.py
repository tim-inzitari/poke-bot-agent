#!/usr/bin/env python3
"""Scrape replays of the top agents on the Kaggle competition ladder.

One-shot, run-on-command tool. A single run snapshots the current top of the
leaderboard for a Kaggle Simulations competition and downloads the raw replay
JSON for the best agents, skipping any episode already pulled on a previous run.

What it does, per run:
  1. Read the leaderboard, take the top N teams.
  2. For each team, pick its highest-scoring submission.
  3. For each submission, take the most recent K episodes it played in.
  4. Download each new episode's replay JSON (dedup against a manifest).
  5. Record provenance in index.csv and update the manifest.

It is a thin, resumable wrapper around the official `kaggle` CLI:
    kaggle competitions leaderboard <comp> -s -v
    kaggle competitions team-submissions <teamId> -v
    kaggle competitions episodes <submissionId> -v
    kaggle competitions replay <episodeId> -p <dir>
    kaggle competitions logs <episodeId> <agentIndex> -p <dir>   (optional)

------------------------------------------------------------------------------
SETUP (one time)
------------------------------------------------------------------------------
1. Install the Kaggle CLI:
       pip install kaggle
2. Get an API token: kaggle.com -> Settings -> "Create New Token".
   This downloads kaggle.json. Put it at:
       Linux/Mac : ~/.kaggle/kaggle.json   (chmod 600)
       Windows   : %USERPROFILE%\\.kaggle\\kaggle.json
   (Or set KAGGLE_USERNAME / KAGGLE_KEY environment variables.)
3. Make sure you have joined / accepted the rules of the competition on the
   Kaggle website, otherwise the API returns 403.

Check your setup without downloading anything:
       python scripts/scrape_ladder_replays.py --self-check

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------
    python scripts/scrape_ladder_replays.py                  # defaults
    python scripts/scrape_ladder_replays.py --teams 5 --episodes-per-sub 10
    python scripts/scrape_ladder_replays.py --dry-run        # plan only, no downloads
    python scripts/scrape_ladder_replays.py --out data/ladder-replays --include-logs

Output (under --out, default data/ladder-replays/):
    replays/        raw replay JSON, as named by the Kaggle CLI
    logs/           per-agent logs (only with --include-logs)
    index.csv       one row per downloaded replay, with provenance
    manifest.json   set of episode ids already downloaded (dedup)
    scrape.log      every API call and outcome
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Optional

DEFAULT_COMPETITION = "pokemon-tcg-ai-battle"


# --------------------------------------------------------------------------- #
# Kaggle CLI plumbing
# --------------------------------------------------------------------------- #
def resolve_kaggle() -> list[str]:
    """Return the command prefix used to invoke the Kaggle CLI.

    Prefers a `kaggle` executable on PATH; falls back to the per-interpreter
    Scripts/bin directory. Raises with install guidance if not found.
    """
    found = shutil.which("kaggle")
    if found:
        return [found]

    import sysconfig

    scripts_dir = Path(sysconfig.get_path("scripts"))
    for name in ("kaggle.exe", "kaggle"):
        candidate = scripts_dir / name
        if candidate.exists():
            return [str(candidate)]

    raise SystemExit(
        "Could not find the `kaggle` CLI.\n"
        "Install it with:  pip install kaggle\n"
        "Then ensure your API token is at ~/.kaggle/kaggle.json "
        "(see this script's header for setup)."
    )


def run_kaggle(
    kaggle: list[str],
    args: list[str],
    *,
    log: "Logger",
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a kaggle subcommand, logging the call and surfacing auth errors."""
    cmd = kaggle + args
    log.write(f"$ kaggle {' '.join(args)}")
    # Force UTF-8 in the child: the kaggle CLI prints team names with emoji/unicode
    # and crashes on Windows' default cp1252 console encoding otherwise.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    proc = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env
    )
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        log.write(f"  ! exit {proc.returncode}: {msg[:500]}")
        lowered = msg.lower()
        if "401" in msg or "403" in msg or "credential" in lowered or "forbidden" in lowered:
            raise SystemExit(
                "Kaggle API rejected the request (auth/permission).\n"
                "  - Confirm ~/.kaggle/kaggle.json exists and is valid.\n"
                "  - Confirm you've joined the competition and accepted its rules "
                "on the Kaggle website.\n"
                f"Raw error: {msg[:500]}"
            )
        if check:
            raise SystemExit(f"kaggle {' '.join(args)} failed: {msg[:500]}")
    return proc


def parse_rows(stdout: str) -> list[dict[str, str]]:
    """Parse kaggle CLI output (CSV via -v) into a list of dict rows.

    Falls back to whitespace-delimited table parsing if it is not valid CSV.
    """
    # The kaggle CLI 2.2.2 prepends pagination lines like
    # "Next Page Token = ..." before the actual CSV; drop them.
    lines = [
        ln
        for ln in stdout.splitlines()
        if ln.strip() and not ln.lstrip().lower().startswith("next page token")
    ]
    text = "\n".join(lines).strip()
    if not text:
        return []

    # Try CSV first (what `-v` produces).
    try:
        reader = csv.DictReader(StringIO(text))
        rows = [dict(r) for r in reader]
        if rows and reader.fieldnames and len(reader.fieldnames) > 1:
            return rows
    except csv.Error:
        pass

    # Fallback: whitespace-aligned table -> split on runs of 2+ spaces.
    import re

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return []
    header = re.split(r"\s{2,}", lines[0].strip())
    out: list[dict[str, str]] = []
    for line in lines[1:]:
        cells = re.split(r"\s{2,}", line.strip())
        if len(cells) == len(header):
            out.append(dict(zip(header, cells)))
    return out


def pick(row: dict[str, str], *names: str) -> Optional[str]:
    """Return the first present, case-insensitive matching column value."""
    lowered = {k.lower(): v for k, v in row.items()}
    for n in names:
        if n.lower() in lowered and lowered[n.lower()] != "":
            return lowered[n.lower()]
    return None


def to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Small logger
# --------------------------------------------------------------------------- #
class Logger:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")

    def write(self, msg: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = f"[{stamp}] {msg}"
        print(line)
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


# --------------------------------------------------------------------------- #
# Ladder traversal
# --------------------------------------------------------------------------- #
def top_teams(kaggle: list[str], competition: str, n: int, log: Logger) -> list[dict[str, str]]:
    proc = run_kaggle(kaggle, ["competitions", "leaderboard", competition, "-s", "-v"], log=log)
    rows = parse_rows(proc.stdout)
    if not rows:
        log.write("  no leaderboard rows parsed")
    return rows[:n]


def best_submission(kaggle: list[str], team_id: str, log: Logger) -> Optional[dict[str, str]]:
    proc = run_kaggle(
        kaggle, ["competitions", "team-submissions", team_id, "-v"], log=log, check=False
    )
    rows = parse_rows(proc.stdout)
    if not rows:
        return None
    rows.sort(key=lambda r: (to_float(pick(r, "publicScore", "score")) or float("-inf")), reverse=True)
    return rows[0]


def recent_episodes(
    kaggle: list[str], submission_id: str, k: int, log: Logger
) -> list[dict[str, str]]:
    proc = run_kaggle(
        kaggle, ["competitions", "episodes", submission_id, "-v"], log=log, check=False
    )
    rows = parse_rows(proc.stdout)
    rows.sort(key=lambda r: (pick(r, "createTime", "createdTime", "create_time") or ""), reverse=True)
    return rows[:k]


def download_replay(
    kaggle: list[str], episode_id: str, dest: Path, log: Logger
) -> Optional[Path]:
    """Download one replay; return the path to the new file (best effort)."""
    dest.mkdir(parents=True, exist_ok=True)
    before = set(dest.iterdir())
    run_kaggle(kaggle, ["competitions", "replay", episode_id, "-p", str(dest)], log=log, check=False)
    new_files = sorted(set(dest.iterdir()) - before, key=lambda p: p.stat().st_mtime)
    return new_files[-1] if new_files else None


# --------------------------------------------------------------------------- #
# Manifest + index
# --------------------------------------------------------------------------- #
def load_manifest(path: Path) -> set[str]:
    if path.exists():
        try:
            return set(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            return set()
    return set()


def save_manifest(path: Path, ids: set[str]) -> None:
    path.write_text(json.dumps(sorted(ids), indent=2), encoding="utf-8")


INDEX_FIELDS = [
    "episode_id",
    "rank",
    "team_id",
    "team_name",
    "submission_id",
    "submission_score",
    "created_time",
    "replay_file",
    "downloaded_at_utc",
]


def append_index(path: Path, row: dict[str, Any]) -> None:
    is_new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in INDEX_FIELDS})


def summarize_replay(path: Path, log: Logger) -> None:
    """Print a one-time structural peek so we learn what a replay contains."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.write(f"  (could not parse replay for summary: {exc})")
        return
    log.write("  --- replay structure (first download) ---")
    if isinstance(data, dict):
        log.write(f"  top-level keys: {list(data.keys())}")
        steps = data.get("steps")
        if isinstance(steps, list):
            log.write(f"  steps: {len(steps)}")
            if steps and isinstance(steps[0], list) and steps[0]:
                log.write(f"  step[0][0] keys: {list(steps[0][0].keys())[:20]}")
    elif isinstance(data, list):
        log.write(f"  top-level: list of {len(data)}")
    log.write("  -----------------------------------------")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument("--teams", type=int, default=10, help="top N leaderboard teams")
    parser.add_argument("--episodes-per-sub", type=int, default=20, help="most recent K episodes per submission")
    parser.add_argument("--out", default="data/ladder-replays", help="output directory")
    parser.add_argument("--sleep", type=float, default=1.0, help="seconds between API calls (be polite)")
    parser.add_argument("--max-replays", type=int, default=0, help="cap total downloads this run (0 = no cap)")
    parser.add_argument("--include-logs", action="store_true", help="also download per-agent logs")
    parser.add_argument("--dry-run", action="store_true", help="plan only; download nothing")
    parser.add_argument("--self-check", action="store_true", help="verify kaggle CLI + auth, then exit")
    args = parser.parse_args()

    kaggle = resolve_kaggle()  # before any output dir is created, so a bad setup leaves nothing behind
    out_dir = Path(args.out)
    log = Logger(out_dir / "scrape.log")
    log.write(f"using kaggle CLI: {kaggle[0]}")

    if args.self_check:
        proc = run_kaggle(kaggle, ["competitions", "leaderboard", args.competition, "-s", "-v"], log=log, check=False)
        rows = parse_rows(proc.stdout)
        if rows:
            log.write(f"OK: auth works, leaderboard returned {len(rows)} rows.")
            log.close()
            return 0
        log.write("Reached the CLI but got no leaderboard rows (check auth / competition slug).")
        log.close()
        return 1

    manifest = load_manifest(out_dir / "manifest.json")
    log.write(f"manifest: {len(manifest)} episodes already downloaded")

    replays_dir = out_dir / "replays"
    logs_dir = out_dir / "logs"
    index_path = out_dir / "index.csv"

    teams = top_teams(kaggle, args.competition, args.teams, log)
    log.write(f"top teams: {len(teams)}")

    downloaded = 0
    summarized = False
    for rank, team in enumerate(teams, start=1):
        team_id = pick(team, "teamId", "team_id", "id")
        team_name = pick(team, "teamName", "team_name", "name") or ""
        if not team_id:
            log.write(f"  rank {rank}: no team id, skipping ({team})")
            continue
        time.sleep(args.sleep)
        sub = best_submission(kaggle, team_id, log)
        if not sub:
            log.write(f"  rank {rank} team {team_id}: no submissions visible, skipping")
            continue
        sub_id = pick(sub, "id", "submissionId")
        sub_score = pick(sub, "publicScore", "score") or ""
        if not sub_id:
            log.write(f"  rank {rank} team {team_id}: submission has no id, skipping")
            continue

        time.sleep(args.sleep)
        episodes = recent_episodes(kaggle, sub_id, args.episodes_per_sub, log)
        log.write(f"  rank {rank} team {team_id} sub {sub_id}: {len(episodes)} recent episodes")

        for ep in episodes:
            ep_id = pick(ep, "id", "episodeId")
            if not ep_id:
                continue
            if ep_id in manifest:
                continue
            if args.max_replays and downloaded >= args.max_replays:
                log.write(f"reached --max-replays={args.max_replays}, stopping")
                _finish(manifest, out_dir, log)
                return 0

            created = pick(ep, "createTime", "createdTime", "create_time") or ""
            if args.dry_run:
                log.write(f"    [dry-run] would download episode {ep_id} (created {created})")
                continue

            time.sleep(args.sleep)
            replay_path = download_replay(kaggle, ep_id, replays_dir, log)
            if replay_path is None:
                log.write(f"    episode {ep_id}: replay download produced no file, skipping")
                continue

            if args.include_logs:
                for agent_idx in (0, 1):
                    time.sleep(args.sleep)
                    run_kaggle(
                        kaggle,
                        ["competitions", "logs", ep_id, str(agent_idx), "-p", str(logs_dir)],
                        log=log,
                        check=False,
                    )

            append_index(
                index_path,
                {
                    "episode_id": ep_id,
                    "rank": rank,
                    "team_id": team_id,
                    "team_name": team_name,
                    "submission_id": sub_id,
                    "submission_score": sub_score,
                    "created_time": created,
                    "replay_file": replay_path.name,
                    "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            manifest.add(ep_id)
            downloaded += 1
            log.write(f"    episode {ep_id}: saved {replay_path.name} ({downloaded} this run)")

            if not summarized:
                summarize_replay(replay_path, log)
                summarized = True

    _finish(manifest, out_dir, log, downloaded)
    return 0


def _finish(manifest: set[str], out_dir: Path, log: Logger, downloaded: int = 0) -> None:
    save_manifest(out_dir / "manifest.json", manifest)
    log.write(f"done: {downloaded} new replays this run; manifest now {len(manifest)} episodes")
    log.close()


if __name__ == "__main__":
    sys.exit(main())
