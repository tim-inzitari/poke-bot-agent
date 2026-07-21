#!/usr/bin/env python3
"""Stage ten feature shards through Bert and start Blackwell bootstrap."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


ROOT = Path("/home/inzi/poke-bot-agent")
FINAL = ROOT / "data/bootstrap/latest10-20260709-20260718"
BERT_STAGE = Path("/Users/tsinzitari/pokebot-staging/latest10-20260709-20260718")
BERT_VERIFIED = BERT_STAGE / ".verified"
BERT_PARTIAL = BERT_STAGE / "partial-manifests"
BERT_STATUS = FINAL / "bert-staging-status.json"
BERT_ASSEMBLER = Path(
    "/Users/tsinzitari/Documents/Codex/2026-07-16/"
    "im-doing-my-pokemon-rl-you/scripts/assemble_feature_manifest.py"
)
DAYS = [f"2026-07-{value:02d}" for value in range(9, 19)]


def run(argv: list[str], *, timeout: int = 900) -> str:
    print("[run] " + " ".join(argv), flush=True)
    result = subprocess.run(
        argv,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    if result.stdout:
        print(result.stdout.rstrip(), flush=True)
    if result.returncode:
        raise RuntimeError(f"command exited {result.returncode}: {argv[0]}")
    return result.stdout


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def day_paths(day: str) -> tuple[Path, Path]:
    feature = FINAL / f"top_ladder_all_{day}.features"
    return feature, feature.with_suffix(feature.suffix + ".json")


def update_bert_status(
    day: str,
    stage: str,
    *,
    metadata: dict[str, Any],
    message: str,
) -> None:
    try:
        payload = json.loads(BERT_STATUS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    days = dict(payload.get("days") or {})
    days[day] = {
        "day": day,
        "stage": stage,
        "host": "Inzi → Bert" if stage == "transferring" else "Bert",
        "message": message,
        "sha256": metadata.get("sha256"),
        "bytes": metadata.get("bytes"),
        "updated_at": time.time(),
    }
    payload = {
        "format": "pokebot-bert-shard-staging-status",
        "observed_at": time.time(),
        "days": days,
    }
    BERT_STATUS.parent.mkdir(parents=True, exist_ok=True)
    temp = BERT_STATUS.with_name(f".{BERT_STATUS.name}.partial.{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temp.replace(BERT_STATUS)


def link_day12() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    for suffix in (".features", ".features.json"):
        source = ROOT / f"data/bootstrap/top_ladder_all_2026-07-12{suffix}"
        target = FINAL / source.name
        if target.exists() or not source.is_file():
            continue
        os.link(source, target)
        print(f"[link] {target} -> {source}", flush=True)


def validate_metadata(day: str) -> dict[str, Any] | None:
    feature, sidecar = day_paths(day)
    if not feature.is_file() or not sidecar.is_file():
        return None
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if metadata.get("source_dates") != [day]:
        raise ValueError(f"source date mismatch for {day}")
    if int(metadata.get("bytes", -1)) != feature.stat().st_size:
        raise ValueError(f"feature byte count mismatch for {day}")
    stats = dict(metadata.get("stats") or {})
    total = int(stats.get("records_total", 0))
    kept = int(stats.get("records_kept", 0))
    if total <= 0 or kept / total < 0.98:
        raise ValueError(f"usable-record gate failed for {day}: {kept}/{total}")
    if int(stats.get("decisions_kept", 0)) <= 0:
        raise ValueError(f"empty decision shard for {day}")
    return metadata


def stage_day(day: str, metadata: dict[str, Any]) -> None:
    feature, sidecar = day_paths(day)
    update_bert_status(
        day,
        "transferring",
        metadata=metadata,
        message="Validated feature shard is transferring to Bert.",
    )
    print(f"[stage] {day} transferring to Bert", flush=True)
    try:
        run(
            [
                "rsync",
                "-a",
                "--whole-file",
                "--partial",
                str(feature),
                str(sidecar),
                f"bert:{BERT_STAGE}/",
            ]
        )
        update_bert_status(
            day,
            "concat",
            metadata=metadata,
            message="Bert is hashing the shard and assembling its manifest fragment.",
        )
        print(f"[stage] {day} concat/verify on Bert", flush=True)
        run(
            [
                "ssh",
                "bert",
                "/usr/bin/python3",
                str(BERT_ASSEMBLER),
                "--staging-dir",
                str(BERT_STAGE),
                "--out",
                str(BERT_PARTIAL / f"{day}.manifest.json"),
                "--expected-date",
                day,
                "--only-date",
                day,
                "--verified-dir",
                str(BERT_VERIFIED),
                "--min-free-gib",
                "25",
            ]
        )
    except BaseException:
        update_bert_status(
            day,
            "failed",
            metadata=metadata,
            message="Bert staging failed; the supervised finalizer will retry.",
        )
        raise
    update_bert_status(
        day,
        "ready",
        metadata=metadata,
        message="Shard digest and per-day manifest fragment are ready on Bert.",
    )
    print(f"[staged] {day} verified on Bert", flush=True)


def wait_for_shards() -> dict[str, dict[str, Any]]:
    last_report = 0.0
    staged: set[str] = set()
    while True:
        link_day12()
        ready: dict[str, dict[str, Any]] = {}
        for day in DAYS:
            metadata = validate_metadata(day)
            if metadata is not None:
                ready[day] = metadata
        for day in DAYS:
            if day in ready and day not in staged:
                stage_day(day, ready[day])
                staged.add(day)
        now = time.time()
        if now - last_report >= 15:
            print(f"[wait] validated={len(ready)}/10 days={sorted(ready)}", flush=True)
            last_report = now
        if len(ready) == len(DAYS):
            return ready
        time.sleep(5)


def main() -> int:
    run(
        [
            "ssh",
            "bert",
            "mkdir",
            "-p",
            str(BERT_STAGE),
            str(BERT_VERIFIED),
            str(BERT_PARTIAL),
        ],
        timeout=30,
    )
    metadata = wait_for_shards()
    total_decisions = sum(
        int((row.get("stats") or {}).get("decisions_kept", 0))
        for row in metadata.values()
    )
    if total_decisions < 5_500_000:
        raise ValueError(f"latest-ten decision gate failed: {total_decisions} < 5500000")
    print("[stage] all ten shards are already verified on Bert", flush=True)
    assembler = [
        "ssh",
        "bert",
        "/usr/bin/python3",
        str(BERT_ASSEMBLER),
        "--staging-dir",
        str(BERT_STAGE),
        "--out",
        str(BERT_STAGE / "manifest.json"),
        "--verified-dir",
        str(BERT_VERIFIED),
        "--min-free-gib",
        "25",
    ]
    for day in DAYS:
        assembler.extend(["--expected-date", day])
    run(assembler)
    print("[stage] returning assembled manifest and verified shards to Inzi", flush=True)
    run(
        [
            "rsync",
            "-a",
            "--whole-file",
            f"bert:{BERT_STAGE}/",
            str(FINAL) + "/",
        ]
    )
    manifest_path = FINAL / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dates") != DAYS:
        raise ValueError("assembled manifest date coverage mismatch")
    shards = list(manifest.get("shards") or [])
    if len(shards) != 10:
        raise ValueError("assembled manifest must contain ten shards")

    def verify(row: dict[str, Any]) -> None:
        path = FINAL / str(row["path"])
        actual = sha256(path)
        if actual != row.get("sha256"):
            raise ValueError(f"Inzi post-transfer digest mismatch: {path}")

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(verify, shards))
    print("[stage] Inzi post-transfer digests verified", flush=True)
    ready = {
        "ready_at": time.time(),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "total_decisions": total_decisions,
        "total_records": sum(
            int((row.get("stats") or {}).get("records_kept", 0))
            for row in metadata.values()
        ),
        "dates": DAYS,
        "bert_stage": str(BERT_STAGE),
    }
    ready_path = FINAL / "READY.json"
    temp = ready_path.with_name(f".{ready_path.name}.partial.{os.getpid()}")
    temp.write_text(json.dumps(ready, indent=2, sort_keys=True) + "\n")
    temp.replace(ready_path)
    print(f"[ready] {json.dumps(ready, sort_keys=True)}", flush=True)
    print("[stage] launching Blackwell bootstrap", flush=True)
    run(["sudo", "-n", "systemctl", "start", "pokemon-latest10-bootstrap.service"], timeout=30)
    time.sleep(3)
    state = run(
        [
            "systemctl",
            "show",
            "pokemon-latest10-bootstrap.service",
            "--property=ActiveState,SubState,MainPID",
        ],
        timeout=10,
    )
    if "ActiveState=active" not in state:
        raise RuntimeError("Blackwell bootstrap did not enter active state")
    print("[complete] Blackwell latest-ten bootstrap is active", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
