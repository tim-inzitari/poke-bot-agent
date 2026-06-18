#!/usr/bin/env python3
"""Fetch official Kaggle submission results for the competition."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any


def maybe_float(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    return float(value)


def fetch_submissions(competition: str) -> list[dict[str, Any]]:
    proc = subprocess.run(
        ["kaggle", "competitions", "submissions", "-c", competition, "-v"],
        check=True,
        capture_output=True,
        text=True,
    )
    fetched_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    for raw in csv.DictReader(StringIO(proc.stdout)):
        rows.append({
            "file_name": raw.get("fileName", ""),
            "date": raw.get("date", ""),
            "description": raw.get("description", ""),
            "status": raw.get("status", ""),
            "public_score": maybe_float(raw.get("publicScore", "")),
            "private_score": maybe_float(raw.get("privateScore", "")),
            "competition": competition,
            "fetched_at_utc": fetched_at,
        })
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", default="pokemon-tcg-ai-battle")
    parser.add_argument("--out", default="data/competition-results.jsonl")
    parser.add_argument("--csv-out", default="data/competition-results.csv")
    args = parser.parse_args()

    rows = fetch_submissions(args.competition)
    write_jsonl(Path(args.out), rows)
    if args.csv_out:
        write_csv(Path(args.csv_out), rows)

    print(f"competition={args.competition}")
    print(f"wrote {len(rows)} results to {args.out}")
    if rows:
        latest = rows[0]
        print(
            "latest "
            f"status={latest['status']} "
            f"public_score={latest['public_score']} "
            f"file={latest['file_name']} "
            f"date={latest['date']}"
        )


if __name__ == "__main__":
    main()
