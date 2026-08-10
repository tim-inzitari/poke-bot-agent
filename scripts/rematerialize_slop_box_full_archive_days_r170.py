#!/usr/bin/env python3
"""Rematerialize teal/Slop Box expert features for ALL archived episode days.

Owner r170: every valid zip under archive/episode-days (currently
2026-06-16..2026-08-05), not Jul29-Aug5-only and not latest20-only.

Leaves live deep CE alone. Writes to the Elmo v5-r170 out-dir; completed
daily receipts are resumed. Does not promote into the live PROTECTED pointer.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.expert_pilot_importance import (  # noqa: E402
    canonical_digest,
    file_digest,
)

TEAL_DECK_HASH = "bcfc7b778a72d895"
TEAL_FINGERPRINT = (
    "sha256:10857495cddc95416052ff3b60e8686d2291da31de23ef064128534eb87cff28"
)
DEFAULT_ARCHIVE = Path("/mnt/Main/main/poke-bot-agent/archive/episode-days")
DEFAULT_OUT = Path(
    "/mnt/Main/main/poke-bot-agent/archive/"
    "teal-mask-ogerpon-ex-guide-corpus-full-v5-r170"
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _dates(start: str, end: str) -> list[str]:
    first = date.fromisoformat(start)
    last = date.fromisoformat(end)
    return [
        (first + timedelta(days=offset)).isoformat()
        for offset in range((last - first).days + 1)
    ]


def inventory_archive_days(archive_dir: Path) -> list[str]:
    days: list[str] = []
    for path in sorted(archive_dir.glob("pokemon-tcg-ai-battle-episodes-*.zip")):
        match = re.search(r"(20\d{2}-\d{2}-\d{2})", path.name)
        if not match:
            continue
        if path.stat().st_size <= 0:
            continue
        days.append(match.group(1))
    if not days:
        raise RuntimeError(f"no valid episode-day zips under {archive_dir}")
    return days


def supabase() -> tuple[str, dict[str, str]]:
    config_text = (
        urllib.request.urlopen("https://ptcgreplay.netlify.app/config.js", timeout=30)
        .read()
        .decode()
    )
    config = json.loads(config_text[config_text.find("{") : config_text.rfind("}") + 1])
    base = str(config["supabaseUrl"]).rstrip("/")
    anon = str(config["anonKey"])
    body = json.dumps(
        {"email": config["teamEmail"], "password": config["teamPassword"]}
    ).encode()
    req = urllib.request.Request(
        base + "/auth/v1/token?grant_type=password",
        data=body,
        method="POST",
        headers={"apikey": anon, "Content-Type": "application/json"},
    )
    token = str(json.load(urllib.request.urlopen(req, timeout=30))["access_token"])
    return base, {"apikey": anon, "Authorization": "Bearer " + token}


def fetch_teal_matches(
    base: str, headers: dict[str, str], *, start: str, end: str
) -> list[dict[str, Any]]:
    deck_q = urllib.parse.quote(TEAL_DECK_HASH, safe="")
    or_clause = f"or=(deck0_hash.eq.{deck_q},deck1_hash.eq.{deck_q})"
    query = (
        "select=episode_id,team0,team1,played_on,deck0_hash,deck1_hash&"
        + or_clause
        + f"&played_on=gte.{start}&played_on=lte.{end}&order=played_on.asc"
    )
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        url = base + "/rest/v1/matches?" + query + f"&limit=1000&offset={offset}"
        req = urllib.request.Request(
            url,
            headers={
                **headers,
                "Range": f"{offset}-{offset + 999}",
                "Prefer": "count=exact",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            chunk = json.load(response)
        if not chunk:
            break
        rows.extend(chunk)
        if len(chunk) < 1000:
            break
        offset += 1000
    return rows


def build_catalog(
    rows: list[dict[str, Any]], out: Path, *, start: str, end: str
) -> dict[str, Any]:
    by_day: Counter[str] = Counter()
    for row in rows:
        day = str(row.get("played_on") or "")[:10]
        if not day:
            continue
        if str(row.get("deck0_hash") or "") == TEAL_DECK_HASH:
            by_day[day] += 1
        if str(row.get("deck1_hash") or "") == TEAL_DECK_HASH:
            by_day[day] += 1
    days = _dates(start, end)
    observed_by_day = {day: int(by_day.get(day, 0)) for day in days}
    source_deck_rows = [
        {
            "archetype_id": 151,
            "card_ids": [
                1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 4, 4, 5, 6, 6, 63, 63, 96, 96, 96,
                108, 140, 184, 184, 272, 756, 756, 756, 978, 1071, 1071, 1071, 1088,
                1097, 1097, 1098, 1098, 1116, 1116, 1116, 1116, 1121, 1121, 1121,
                1121, 1182, 1182, 1197, 1197, 1198, 1198, 1198, 1198, 1205, 1205,
                1250, 1250, 1250, 1250,
            ],
            "deck_hash": TEAL_DECK_HASH,
        }
    ]
    catalog = {
        "schema": "poke_bot.public_deck_archetype_catalog/v1",
        "specialist_id": "teal-mask-ogerpon-ex",
        "source": "https://ptcgreplay.netlify.app/",
        "source_access": "authenticated_public_site_with_supabase_rls",
        "source_archetype": {"id": 151, "name": "Teal Mask Ogerpon ex"},
        "source_window": {"start": start, "end": end, "days": len(days)},
        "minimum_acting_seat_games": int(sum(observed_by_day.values())),
        "observed_acting_seat_games": int(sum(observed_by_day.values())),
        "observed_by_day": observed_by_day,
        "deck_fingerprints": [TEAL_FINGERPRINT],
        "source_deck_rows": source_deck_rows,
        "source_deck_rows_sha256": canonical_digest(source_deck_rows),
        "source_match_rows": len(rows),
        "source_match_facts_sha256": canonical_digest(
            [
                {
                    "episode_id": row.get("episode_id"),
                    "team0": row.get("team0"),
                    "team1": row.get("team1"),
                    "played_on": row.get("played_on"),
                    "deck0_hash": row.get("deck0_hash"),
                    "deck1_hash": row.get("deck1_hash"),
                }
                for row in rows
            ]
        ),
        "source_ingest_run": {
            "id": "r170-full-archive-rematerialize",
            "window_start": start,
            "window_end": end,
            "days_ingested": len(days),
            "note": (
                "Full archived episode-day range for Slop Box / teal-mask; "
                "not Jul29-Aug5-only"
            ),
        },
    }
    _atomic_json(out, catalog)
    return catalog


def recount_out(out_dir: Path, days: list[str]) -> dict[str, Any]:
    done: list[str] = []
    partial: list[str] = []
    missing: list[str] = []
    games = 0
    decisions = 0
    for day in days:
        feat = out_dir / f"teal-mask-ogerpon-ex-{day}.features"
        receipt = out_dir / f"teal-mask-ogerpon-ex-{day}.features.receipt.json"
        parts = list(out_dir.glob(f".teal-mask-ogerpon-ex-{day}.features.partial.*"))
        if feat.is_file() and receipt.is_file():
            stats = json.loads(receipt.read_text(encoding="utf-8")).get("stats") or {}
            games += int(stats.get("records_kept", 0))
            decisions += int(stats.get("decisions_kept", 0))
            done.append(day)
        elif parts or feat.exists():
            partial.append(day)
        else:
            missing.append(day)
    return {
        "done_days": len(done),
        "partial_days": partial,
        "missing_days": missing,
        "games_present": games,
        "decisions_present": decisions,
        "done_start": done[0] if done else None,
        "done_end": done[-1] if done else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--catalog-only", action="store_true")
    parser.add_argument("--launch-materialize", action="store_true")
    parser.add_argument("--day-parallelism", type=int, default=2)
    parser.add_argument("--workers-per-day", type=int, default=2)
    parser.add_argument("--max-in-flight-per-day", type=int, default=2)
    parser.add_argument("--memory-floor-gib", type=float, default=4.0)
    parser.add_argument("--clear-stale-partials", action="store_true")
    args = parser.parse_args()

    archive_dir = args.archive_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "status").mkdir(exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)

    days = inventory_archive_days(archive_dir)
    start, end = days[0], days[-1]
    before = recount_out(out_dir, days)

    if args.clear_stale_partials:
        removed = []
        for path in out_dir.glob(".teal-mask-ogerpon-ex-*.features.partial.*"):
            path.unlink(missing_ok=True)
            removed.append(path.name)
        print(json.dumps({"cleared_partials": removed}, indent=2), flush=True)

    base, headers = supabase()
    rows = fetch_teal_matches(base, headers, start=start, end=end)
    catalog_path = out_dir / "PUBLIC_DECK_ARCHETYPE_CATALOG.json"
    catalog = build_catalog(rows, catalog_path, start=start, end=end)

    coordination = {
        "schema": "poke_bot.slop_box_full_archive_expand_coordination_r170/v1",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "full_archive_rematerialize_active",
        "owner_clarification": (
            "do ALL archived episode days for teal/Slop Box; "
            "full-corpus train; Chao upweight only; not Jul29-Aug5-only"
        ),
        "supersedes": {
            "narrow_window": "2026-07-29..2026-08-05",
            "prior_owner": "7cbdccac",
            "reason": "owner ordered kill/supersede narrow 8-day expand",
        },
        "archive_inventory": {
            "days": len(days),
            "start": start,
            "end": end,
            "valid_zip_days": days,
        },
        "v5_r170": {
            "out_dir": str(out_dir),
            "catalog_sha256": file_digest(catalog_path),
            "observed_acting_seat_games": catalog["observed_acting_seat_games"],
            "before": before,
        },
        "train_contract": {
            "chao_only_filter": False,
            "cox_chao_upweight_only": True,
            "full_archetype_all_games": True,
            "bind_live_deep_ce": "next_clean_boundary_only",
        },
        "no_ready": True,
        "no_rl": True,
        "why_they_thought_we_had_them": {
            "archive_episode_zips_present": True,
            "archive_range": {"start": start, "end": end, "days": len(days)},
            "featurized_protected_expert_was": {
                "start": "2026-06-26",
                "end": "2026-07-28",
                "games": 1442,
            },
            "gap": (
                "archive zips on Elmo != featurized CURRENT_DECK_GUIDE / "
                "PROTECTED_EXPERT_CORPUS used by training"
            ),
        },
    }
    _atomic_json(out_dir / "FULL_ARCHIVE_EXPAND_COORDINATION_r170.json", coordination)
    print(json.dumps(coordination, indent=2, sort_keys=True), flush=True)

    if args.catalog_only or not args.launch_materialize:
        return 0

    status = out_dir / "status" / "window.json"
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "materialize_authoritative_guide_window_parallel.py"),
        "--start",
        start,
        "--end",
        end,
        "--archive-dir",
        str(archive_dir),
        "--out-dir",
        str(out_dir),
        "--status",
        str(status),
        "--required-archetype",
        "teal-mask-ogerpon-ex",
        "--current-deck-guide",
        "teal-mask-ogerpon-ex",
        "--authoritative-deck-catalog",
        str(catalog_path),
        "--authoritative-only-archetype",
        "teal-mask-ogerpon-ex",
        "--day-parallelism",
        str(args.day_parallelism),
        "--workers-per-day",
        str(args.workers_per_day),
        "--max-in-flight-per-day",
        str(args.max_in_flight_per_day),
        "--max-context",
        "320",
        "--memory-floor-gib",
        str(args.memory_floor_gib),
        "--min-records",
        "0",
    ]
    print("[full-archive] materialize", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    after = recount_out(out_dir, days)
    coordination["status"] = "full_archive_materialize_complete"
    coordination["v5_r170"]["after"] = after
    coordination["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(out_dir / "FULL_ARCHIVE_EXPAND_COORDINATION_r170.json", coordination)
    print(json.dumps({"after": after}, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
