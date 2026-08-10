#!/usr/bin/env python3
"""Build Cox/Chao held-split pilot map for Slop Box H10 RTP (r170).

Uses PTCGReplay public matches (team0/team1) joined to expert-corpus
episode_id+seat targets. Archive NFS extract remains optional corroboration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COX = "James Cox & Henry Chao"
DEFAULT_TARGETS = Path(
    "/home/inzi/poke-bot-agent/outputs/state/slop-box-cox-chao-held-targets-r170.json"
)
DEFAULT_PILOT_OUT = Path(
    "/home/inzi/poke-bot-agent/outputs/state/slop-box-cox-chao-held-pilot-map-r170.json"
)
DEFAULT_HELD_OUT = Path(
    "/home/inzi/poke-bot-agent/outputs/state/"
    "slop-box-cox-chao-held-split-pilot-map-r170.json"
)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def supabase_token() -> tuple[str, str, dict[str, str]]:
    config_text = urllib.request.urlopen(
        "https://ptcgreplay.netlify.app/config.js", timeout=30
    ).read().decode("utf-8")
    start = config_text.find("{")
    end = config_text.rfind("}")
    config = json.loads(config_text[start : end + 1])
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
    with urllib.request.urlopen(req, timeout=30) as response:
        token = str(json.load(response)["access_token"])
    headers = {"apikey": anon, "Authorization": "Bearer " + token}
    return base, anon, headers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--pilot-out", type=Path, default=DEFAULT_PILOT_OUT)
    parser.add_argument("--held-out", type=Path, default=DEFAULT_HELD_OUT)
    parser.add_argument("--query-chunk-size", type=int, default=40)
    parser.add_argument("--query-retries", type=int, default=4)
    parser.add_argument(
        "--reuse-pilot-map",
        action="store_true",
        help=(
            "Require pilot-out to be an existing checksum-bound raw-archive "
            "map and build only the held split from it."
        ),
    )
    args = parser.parse_args()
    targets_path = args.targets.expanduser().resolve()
    pilot_out = args.pilot_out.expanduser().resolve()
    held_out = args.held_out.expanduser().resolve()

    targets = json.loads(targets_path.read_text(encoding="utf-8"))
    all_rows = list(targets["train_rows"]) + list(targets["validation_rows"])
    requested = {(str(row["episode_id"]), int(row["seat"])) for row in all_rows}
    if args.reuse_pilot_map:
        if not pilot_out.is_file():
            raise RuntimeError(f"pilot map reuse requested but missing: {pilot_out}")
        pilot = json.loads(pilot_out.read_text(encoding="utf-8"))
        if pilot.get("schema") != "poke_bot.expert_pilot_map/v1":
            raise RuntimeError("reused pilot map has the wrong schema")
        if pilot.get("targets_sha256") != file_digest(targets_path):
            raise RuntimeError("reused pilot map target digest mismatch")
        if int(pilot.get("unverifiable_rows") or 0) != 0:
            raise RuntimeError("reused pilot map contains unverifiable rows")
        rows = list(pilot.get("rows") or ())
        found = {
            (str(row["episode_id"]), int(row["seat"])): str(row["team_name"])
            for row in rows
        }
        missing = sorted(requested - set(found))
        if missing or len(rows) != len(requested):
            raise RuntimeError("reused pilot map does not exactly cover targets")
    else:
        episode_ids = sorted({episode_id for episode_id, _ in requested})
        base, _anon, headers = supabase_token()

        found: dict[tuple[str, int], str] = {}
        chunk_size = max(1, min(80, int(args.query_chunk_size)))
        query_retries = max(1, min(8, int(args.query_retries)))
        for index in range(0, len(episode_ids), chunk_size):
            chunk = episode_ids[index : index + chunk_size]
            filt = ",".join(chunk)
            url = (
                base
                + "/rest/v1/matches?select=episode_id,team0,team1&episode_id=in.("
                + filt
                + ")"
            )
            query_rows = None
            for attempt in range(query_retries):
                req = urllib.request.Request(url, headers=headers)
                try:
                    with urllib.request.urlopen(req, timeout=60) as response:
                        query_rows = json.load(response)
                    break
                except (TimeoutError, urllib.error.URLError):
                    if attempt + 1 >= query_retries:
                        raise
                    time.sleep(min(30, 3 * (2**attempt)))
            if not isinstance(query_rows, list):
                raise RuntimeError("PTCGReplay matches query did not return a list")
            for row in query_rows:
                episode_id = str(row["episode_id"])
                for seat, team in ((0, row.get("team0")), (1, row.get("team1"))):
                    key = (episode_id, seat)
                    if key in requested and isinstance(team, str) and team:
                        found[key] = team

        missing = sorted(requested - set(found))
        rows = [
            {
                "episode_id": episode_id,
                "seat": seat,
                "team_name": found[(episode_id, seat)],
            }
            for episode_id, seat in sorted(found)
        ]
        pilot = {
            "schema": "poke_bot.expert_pilot_map/v1",
            "owner_decision_revision": 170,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "targets_sha256": file_digest(targets_path),
            "rows": rows,
            "requested_rows": len(requested),
            "resolved_rows": len(rows),
            "unverifiable_rows": len(missing),
            "unverifiable": [
                {"episode_id": episode_id, "seat": seat}
                for episode_id, seat in missing
            ],
            "source_archives": [
                {
                    "method": "ptcgreplay_matches_table_team0_team1_join",
                    "source": "https://ptcgreplay.netlify.app/",
                    "table": "matches",
                    "note": (
                        "TeamNames-equivalent acting-seat join for expert corpus "
                        "episode_ids; public meta preferred over slow NFS archive scan"
                    ),
                }
            ],
            "extraction_source": "ptcgreplay_supabase_matches",
        }
        atomic_json(pilot_out, pilot)

    val_keys = {
        (str(row["episode_id"]), int(row["seat"]))
        for row in targets["validation_rows"]
    }
    train_keys = {
        (str(row["episode_id"]), int(row["seat"])) for row in targets["train_rows"]
    }
    cox_rows = [row for row in rows if row["team_name"] == COX]
    cox_val = [
        row for row in cox_rows if (row["episode_id"], row["seat"]) in val_keys
    ]
    cox_train = [
        row for row in cox_rows if (row["episode_id"], row["seat"]) in train_keys
    ]
    held = {
        "schema": "poke_bot.slop_box_cox_chao_held_split_pilot_map_r170/v1",
        "goal_revision": 170,
        "status": "ready" if not missing else "partial_unverifiable",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "team_name_exact": COX,
        "targets": str(targets_path),
        "targets_sha256": pilot["targets_sha256"],
        "pilot_map": str(pilot_out),
        "pilot_map_sha256": file_digest(pilot_out),
        "pilot_map_schema": pilot["schema"],
        "requested_rows": pilot["requested_rows"],
        "resolved_rows": pilot["resolved_rows"],
        "unverifiable_rows": pilot["unverifiable_rows"],
        "cox_chao_acting_seat_games_total": len(cox_rows),
        "cox_chao_held_validation_games": len(cox_val),
        "cox_chao_train_games": len(cox_train),
        "held_validation_rows": cox_val,
        "join_key": ["episode_id", "seat", "exact_team_name"],
        "gate_metric_id": "cox_chao_held_policy_acc",
        "gate_threshold": 0.9,
        "team_counts_top": Counter(row["team_name"] for row in rows).most_common(20),
    }
    atomic_json(held_out, held)
    for mirror in (
        ROOT / "state" / held_out.name,
        ROOT / "state" / pilot_out.name,
        Path("/home/inzi/poke-bot-agent/state") / held_out.name,
        Path("/home/inzi/poke-bot-agent/state") / pilot_out.name,
    ):
        try:
            atomic_json(mirror, held if "held-split" in mirror.name else pilot)
        except OSError:
            pass

    print(
        json.dumps(
            {
                "pilot_map": str(pilot_out),
                "pilot_map_sha256": file_digest(pilot_out),
                "held_split": str(held_out),
                "held_split_sha256": file_digest(held_out),
                "resolved_rows": len(rows),
                "unverifiable_rows": len(missing),
                "cox_chao_total": len(cox_rows),
                "cox_chao_held": len(cox_val),
                "cox_chao_train": len(cox_train),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
