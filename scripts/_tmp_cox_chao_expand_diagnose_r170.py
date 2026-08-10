#!/usr/bin/env python3
"""Diagnose Cox/Chao coverage expansion inputs (r170)."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path


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


def fetch_matches(base: str, headers: dict[str, str], query: str) -> list[dict]:
    rows: list[dict] = []
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
            if offset == 0:
                print("content-range", response.headers.get("content-range"), "query", query[:120])
            chunk = json.load(response)
        if not chunk:
            break
        rows.extend(chunk)
        if len(chunk) < 1000:
            break
        offset += 1000
    return rows


def main() -> int:
    base, headers = supabase()
    cox = "James Cox & Henry Chao"
    teal_hash = "bcfc7b778a72d895"  # from public catalog deck_hash
    # URL-encode values; PostgREST accepts *eq.value* with percent-encoding.
    cox_q = urllib.parse.quote(cox, safe="")
    # team0 or team1 equals Cox/Chao
    or_clause = f"or=(team0.eq.{cox_q},team1.eq.{cox_q})"
    query = (
        "select=episode_id,team0,team1,played_on,deck0_hash,deck1_hash&"
        + or_clause
        + "&played_on=gte.2026-06-26&order=played_on.asc"
    )
    rows = fetch_matches(base, headers, query)
    print("cox_chao_matches", len(rows))
    days = Counter(str(row.get("played_on") or "")[:10] for row in rows)
    print("day_span", min(days) if days else None, max(days) if days else None)
    for day in sorted(d for d in days if d >= "2026-07-29"):
        print("day", day, days[day])
    print(
        "aug3_5_cox",
        {day: days.get(day, 0) for day in ("2026-08-03", "2026-08-04", "2026-08-05")},
    )
    # Cox/Chao seats that also use the teal deck hash on that seat.
    teal_cox = 0
    teal_cox_by_day: Counter[str] = Counter()
    for row in rows:
        day = str(row.get("played_on") or "")[:10]
        if row.get("team0") == cox and str(row.get("deck0_hash") or "") == teal_hash:
            teal_cox += 1
            teal_cox_by_day[day] += 1
        if row.get("team1") == cox and str(row.get("deck1_hash") or "") == teal_hash:
            teal_cox += 1
            teal_cox_by_day[day] += 1
    print("teal_deck_hash_cox_seats", teal_cox)
    print(
        "aug3_5_teal_cox",
        {
            day: teal_cox_by_day.get(day, 0)
            for day in ("2026-08-03", "2026-08-04", "2026-08-05")
        },
    )
    print(
        "jul29_aug5_teal_cox",
        sum(v for k, v in teal_cox_by_day.items() if "2026-07-29" <= k <= "2026-08-05"),
    )
    print(
        "through_jul28_teal_cox",
        sum(v for k, v in teal_cox_by_day.items() if k <= "2026-07-28"),
    )

    # Existing pilot coverage for comparison.
    pilot = json.loads(
        Path(
            "/home/inzi/poke-bot-agent/outputs/state/"
            "slop-box-cox-chao-held-pilot-map-r170.json"
        ).read_text(encoding="utf-8")
    )
    held = json.loads(
        Path(
            "/home/inzi/poke-bot-agent/outputs/state/"
            "slop-box-cox-chao-held-split-pilot-map-r170.json"
        ).read_text(encoding="utf-8")
    )
    print(
        "current_pilot_cox",
        sum(1 for row in pilot["rows"] if row.get("team_name") == cox),
        "held",
        held.get("cox_chao_held_validation_games"),
        "train",
        held.get("cox_chao_train_games"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
