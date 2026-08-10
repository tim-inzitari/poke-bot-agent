#!/usr/bin/env python3
"""Expand Slop Box teal-mask expert corpus through 2026-08-05 for Cox/Chao gate.

1) Rebuild public exact-deck catalog from PTCGReplay matches for deck_hash
   bcfc7b778a72d895 over 2026-06-26..2026-08-05.
2) Materialize only missing daily feature shards for 2026-07-29..2026-08-05.
3) Assemble a new protected corpus directory that reuses existing Jul28-and-
   earlier shards and adds the new days (no rewrite of old shard bytes).
4) Emit targets / pilot / held-split / Cox/Chao upweight receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
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
    TARGET_SCHEMA,
    canonical_digest,
    file_digest,
)

COX = "James Cox & Henry Chao"
TEAL_DECK_HASH = "bcfc7b778a72d895"
TEAL_FINGERPRINT = (
    "sha256:10857495cddc95416052ff3b60e8686d2291da31de23ef064128534eb87cff28"
)
OLD_CORPUS = Path(
    "/home/inzi/poke-bot-agent/data/bootstrap/"
    "expert-latest20-2026-07-04-2026-07-23-roster18-v6-strategic/"
    "teal-mask-ogerpon-ex"
)
ARCHIVE_DIR = Path("/tmp/truenas_main/poke-bot-agent/archive/episode-days")
DEFAULT_OUT = Path(
    "/home/inzi/poke-bot-agent/data/bootstrap/"
    "expert-slop-box-teal-mask-full41-r170/teal-mask-ogerpon-ex"
)
DEFAULT_STATE = Path("/home/inzi/poke-bot-agent/outputs/state")


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


def fetch_teal_matches(base: str, headers: dict[str, str]) -> list[dict[str, Any]]:
    deck_q = urllib.parse.quote(TEAL_DECK_HASH, safe="")
    or_clause = f"or=(deck0_hash.eq.{deck_q},deck1_hash.eq.{deck_q})"
    query = (
        "select=episode_id,team0,team1,played_on,deck0_hash,deck1_hash&"
        + or_clause
        + "&played_on=gte.2026-06-26&played_on=lte.2026-08-05&order=played_on.asc"
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


def build_catalog(rows: list[dict[str, Any]], out: Path) -> dict[str, Any]:
    by_day: Counter[str] = Counter()
    for row in rows:
        day = str(row.get("played_on") or "")[:10]
        if not day:
            continue
        if str(row.get("deck0_hash") or "") == TEAL_DECK_HASH:
            by_day[day] += 1
        if str(row.get("deck1_hash") or "") == TEAL_DECK_HASH:
            by_day[day] += 1
    days = _dates("2026-06-26", "2026-08-05")
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
        "source_window": {
            "start": "2026-06-26",
            "end": "2026-08-05",
            "days": len(days),
        },
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
            "id": "r170-cox-chao-expand",
            "window_start": "2026-06-26",
            "window_end": "2026-08-05",
            "days_ingested": len(days),
            "note": "Supabase matches join on exact deck_hash for Slop Box Cox/Chao expand",
        },
    }
    _atomic_json(out, catalog)
    return catalog


def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def stage_old_days(out_dir: Path) -> list[str]:
    linked = []
    for day in _dates("2026-06-26", "2026-07-28"):
        for suffix in (".features", ".features.json", ".features.receipt.json"):
            src = OLD_CORPUS / f"teal-mask-ogerpon-ex-{day}{suffix}"
            dst = out_dir / f"teal-mask-ogerpon-ex-{day}{suffix}"
            if not src.is_file():
                raise FileNotFoundError(src)
            hardlink_or_copy(src, dst)
        linked.append(day)
    return linked


def materialize_new_days(
    *,
    out_dir: Path,
    catalog: Path,
    start: str,
    end: str,
) -> None:
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
        str(ARCHIVE_DIR),
        "--out-dir",
        str(out_dir),
        "--status",
        str(status),
        "--required-archetype",
        "teal-mask-ogerpon-ex",
        "--current-deck-guide",
        "teal-mask-ogerpon-ex",
        "--authoritative-deck-catalog",
        str(catalog),
        "--authoritative-only-archetype",
        "teal-mask-ogerpon-ex",
        "--day-parallelism",
        "4",
        "--workers-per-day",
        "2",
        "--max-in-flight-per-day",
        "4",
        "--max-context",
        "320",
        "--memory-floor-gib",
        "16",
        "--min-records",
        "0",
    ]
    print("[expand] materialize", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def finalize_window(
    *,
    out_dir: Path,
    catalog: Path,
    start: str,
    end: str,
    minimum_records: int,
) -> None:
    status = out_dir / "status" / "window.json"
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "finalize_current_deck_guide_window.py"),
        "--status",
        str(status),
        "--out-dir",
        str(out_dir),
        "--start",
        start,
        "--end",
        end,
        "--specialist-id",
        "teal-mask-ogerpon-ex",
        "--guide-version",
        "teal-mask-ogerpon-ex-slop-box-north-star-v3",
        "--minimum-records",
        str(minimum_records),
        "--public-deck-catalog",
        str(catalog),
    ]
    print("[expand] finalize", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--start-new", default="2026-07-29")
    parser.add_argument("--end", default="2026-08-05")
    parser.add_argument("--skip-materialize", action="store_true")
    parser.add_argument("--catalog-only", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    state_dir = args.state_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "status").mkdir(exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)

    base, headers = supabase()
    rows = fetch_teal_matches(base, headers)
    catalog_path = out_dir / "PUBLIC_DECK_ARCHETYPE_CATALOG.json"
    catalog = build_catalog(rows, catalog_path)
    print(
        json.dumps(
            {
                "catalog": str(catalog_path),
                "catalog_sha256": file_digest(catalog_path),
                "observed_acting_seat_games": catalog["observed_acting_seat_games"],
                "jul29_aug5": sum(
                    v
                    for k, v in catalog["observed_by_day"].items()
                    if "2026-07-29" <= k <= "2026-08-05"
                ),
                "aug3_5": {
                    k: catalog["observed_by_day"][k]
                    for k in ("2026-08-03", "2026-08-04", "2026-08-05")
                },
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if args.catalog_only:
        return 0

    linked = stage_old_days(out_dir)
    print(f"[expand] linked_old_days={len(linked)}", flush=True)
    if not args.skip_materialize:
        # Materialize the full window so resume validates old days and builds new.
        materialize_new_days(
            out_dir=out_dir,
            catalog=catalog_path,
            start="2026-06-26",
            end=str(args.end),
        )
    finalize_window(
        out_dir=out_dir,
        catalog=catalog_path,
        start="2026-06-26",
        end=str(args.end),
        minimum_records=int(catalog["observed_acting_seat_games"]),
    )
    receipt = {
        "schema": "poke_bot.slop_box_cox_chao_corpus_expand_r170/v1",
        "goal_revision": 170,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "out_dir": str(out_dir),
        "catalog": str(catalog_path),
        "catalog_sha256": file_digest(catalog_path),
        "observed_acting_seat_games": catalog["observed_acting_seat_games"],
        "old_corpus": str(OLD_CORPUS),
        "window": {"start": "2026-06-26", "end": str(args.end)},
        "status": "materialized",
    }
    _atomic_json(state_dir / "slop-box-cox-chao-corpus-expand-r170.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
