#!/usr/bin/env python
"""Phase 2B smoke: filter → JSONL → dataset load + info-set assert.

Uses a small local episode sample (``data/episodes/smoke``) by default so the
smoke stays fast. Pass ``--fetch-sample`` to pull a handful of episode files
from the latest daily Kaggle bundle first.

Exits non-zero on failure.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import paths
from poke_bot.dataset import BootstrapDataset
from poke_bot.episodes_index import ensure_episodes_index, latest_n_days, load_daily_manifest
from poke_bot.replay_import import assert_info_set, load_episode_payload

PYTHON = sys.executable
KAGGLE = ROOT / ".venv" / "bin" / "kaggle"
SMOKE_DIR = paths.DATA_DIR / "episodes" / "smoke"
OUT_JSONL = paths.DATA_DIR / "bootstrap" / "dragapult.smoke.jsonl"


def _fetch_sample(n: int = 30) -> None:
    """Download ``n`` episode JSON files from the latest daily dataset."""
    ensure_episodes_index()
    manifest = load_daily_manifest()
    days = latest_n_days(manifest, 1)
    if not days:
        raise RuntimeError("no days in episodes index")
    entry = days[-1]
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(SMOKE_DIR.glob("*.json"))
    if len(existing) >= min(n, 5):
        print(f"smoke dir already has {len(existing)} json files; skipping fetch")
        return

    # List files via kaggle CLI, take first n names.
    ref = f"kaggle/{entry.slug}" if not entry.slug.startswith("kaggle/") else entry.slug
    list_cmd = [str(KAGGLE), "datasets", "files", ref]
    proc = subprocess.run(list_cmd, check=True, capture_output=True, text=True)
    names: list[str] = []
    for line in proc.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        name = parts[0]
        if name.endswith(".json") and name[0].isdigit():
            names.append(name)
        if len(names) >= n:
            break
    if not names:
        raise RuntimeError(f"could not list episode files for {ref}")

    print(f"fetching {len(names)} episodes from {ref} → {SMOKE_DIR}")
    for name in names:
        cmd = [
            str(KAGGLE),
            "datasets",
            "download",
            "-d",
            ref,
            "-f",
            name,
            "-p",
            str(SMOKE_DIR),
            "-o",
        ]
        subprocess.run(cmd, check=False, capture_output=True)
        zipped = SMOKE_DIR / f"{name}.zip"
        if zipped.is_file():
            subprocess.run(["unzip", "-o", "-q", str(zipped), "-d", str(SMOKE_DIR)], check=False)
            zipped.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch-sample", action="store_true", help="Download a small episode sample first.")
    ap.add_argument("--max-games", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    paths.ensure_runtime_dirs()
    print("== smoke_bootstrap", flush=True)

    # 1) Index present.
    manifest_path = ensure_episodes_index()
    manifest = load_daily_manifest(manifest_path)
    print(f"OK episodes index: {len(manifest)} days (latest={manifest[-1].date})")

    if args.fetch_sample:
        _fetch_sample(30)

    if not SMOKE_DIR.is_dir() or not any(SMOKE_DIR.glob("*.json")):
        print(f"ERROR: no smoke episodes in {SMOKE_DIR}; re-run with --fetch-sample", file=sys.stderr)
        return 2

    # 2) Spot-check info-set on raw episodes.
    checked = 0
    for path in sorted(SMOKE_DIR.glob("*.json"))[:5]:
        payload = load_episode_payload(path)
        for step in (payload.get("steps") or [])[:50]:
            if not isinstance(step, list):
                continue
            for entry in step[:2]:
                obs = (entry or {}).get("observation") or {}
                if not obs.get("current") or not obs.get("select"):
                    continue
                report = assert_info_set(obs, strict=True)
                assert report.ok, report.violations
                checked += 1
    print(f"OK info-set assert on {checked} raw decision observations")

    # 3) Filter + convert via bootstrap_replays.
    if OUT_JSONL.exists():
        OUT_JSONL.unlink()
    cmd = [
        PYTHON,
        str(ROOT / "scripts" / "bootstrap_replays.py"),
        "--archetype",
        "dragapult",
        "--local-dir",
        str(SMOKE_DIR),
        "--max-games",
        str(args.max_games),
        "--min-games",
        "0",
        "--min-opp-archetypes",
        "0",
        "--workers",
        str(args.workers),
        "--out",
        str(OUT_JSONL),
    ]
    print(">>", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print("ERROR: bootstrap_replays failed", file=sys.stderr)
        return proc.returncode

    if not OUT_JSONL.is_file():
        print("ERROR: JSONL not written", file=sys.stderr)
        return 1

    n_lines = sum(1 for _ in OUT_JSONL.open())
    print(f"OK JSONL lines={n_lines} path={OUT_JSONL}")
    if n_lines < 1:
        print(
            "ERROR: no dragapult sequences in smoke sample "
            "(try --fetch-sample with a larger N)",
            file=sys.stderr,
        )
        return 1

    # Schema spot-check.
    first = json.loads(OUT_JSONL.open().readline())
    required = {"episode_id", "seat", "archetype", "deck", "value", "steps", "info_set_ok", "aux_labels"}
    # aux_labels live per-step
    step0 = (first.get("steps") or [{}])[0]
    missing = required - set(first) - {"aux_labels"}
    if missing:
        print(f"ERROR: JSONL missing keys {missing}", file=sys.stderr)
        return 1
    if "aux_labels" not in step0:
        print("ERROR: step missing aux_labels", file=sys.stderr)
        return 1
    if first.get("archetype") != "dragapult":
        print(f"ERROR: expected dragapult, got {first.get('archetype')}", file=sys.stderr)
        return 1
    print(
        f"OK schema episode={first['episode_id']} seat={first['seat']} "
        f"decisions={first.get('n_decisions')} value={first.get('value')} "
        f"opp={first.get('opp_archetype')}"
    )

    # 4) Load via dataset (featurize + re-assert info-set).
    ds = BootstrapDataset.from_jsonl(
        OUT_JSONL,
        verify_info_set=True,
        use_cache=True,
    )
    summary = ds.summary()
    print(f"OK dataset {summary}")
    if not ds.info_set_ok_all:
        print("ERROR: dataset info-set integrity failed", file=sys.stderr)
        return 1
    if len(ds) < 1:
        print("ERROR: dataset empty after featurize", file=sys.stderr)
        return 1

    # Spot-check first sequence board tokens.
    seq = ds[0]
    d0 = seq.decisions[0]
    print(
        f"OK featurized board_words={d0.board.num_words} "
        f"option_words={d0.options.num_words} action={d0.action}"
    )
    print("SMOKE PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
