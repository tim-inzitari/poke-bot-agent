#!/usr/bin/env python3
"""Assemble Jul24-Aug5 schema-8+combo protected corpus and CPU pack (require-combo-state)."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/inzi/poke-bot-agent")
EXPERT = ROOT / "data/bootstrap/expert-slop-box-schema8-jul24-aug5-r175/teal-mask-ogerpon-ex"
SEAL = ROOT / "outputs/state/slop-box-schema8-jul24-aug5-sealed-r175.json"
CPU_READY = ROOT / "outputs/state/slop-box-schema8-jul24-aug5-cpu-pack-ready-r175.json"
CPU_PRE = ROOT / "outputs/state/slop-box-schema8-jul24-aug5-cpu-pack-prebuild-r175.json"
CPU_PACK = ROOT / "outputs/bootstrap/cpu-packs/final_format_slop_box_h10_rtp_schema8_jul24_aug5_r175"
SPECIALIST = "teal-mask-ogerpon-ex"
DAYS = [f"2026-07-{d:02d}" for d in range(24, 32)] + [
    f"2026-08-{d:02d}" for d in range(1, 6)
]
PYBIN = "/home/inzi/miniconda3/envs/poke-bot-agent/bin/python"


def _atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def assemble_manifest(seal: dict) -> Path:
    shards = []
    totals = {"records": 0, "decisions": 0, "combo_state_rows": 0}
    for day in DAYS:
        feat = EXPERT / f"{SPECIALIST}-{day}.features"
        receipt = EXPERT / f"{SPECIALIST}-{day}.features.receipt.json"
        if not feat.is_file() or not receipt.is_file():
            raise FileNotFoundError(feat)
        rec = json.loads(receipt.read_text())
        stats = rec.get("stats") or {}
        records = int(stats.get("records_kept") or 0)
        decisions = int(stats.get("decisions_kept") or 0)
        combo_rows = int(
            ((stats.get("target_coverage") or {}).get("combo_state_rows")) or 0
        )
        schema = int((rec.get("schemas") or {}).get("dataset") or 0)
        if records <= 0 or schema < 8 or combo_rows <= 0:
            raise RuntimeError(
                f"day not schema8+combo ready: {day} records={records} "
                f"schema={schema} combo={combo_rows}"
            )
        totals["records"] += records
        totals["decisions"] += decisions
        totals["combo_state_rows"] += combo_rows
        digest = "sha256:" + hashlib.sha256(feat.read_bytes()).hexdigest()
        shards.append(
            {
                "path": feat.name,
                "sha256": digest,
                "bytes": feat.stat().st_size,
                "source_dates": [day],
                "stats": {
                    "records_kept": records,
                    "decisions_kept": decisions,
                    "target_coverage": (stats.get("target_coverage") or {}),
                },
            }
        )
    manifest = {
        "format": "pokebot-bootstrap-feature-manifest",
        "format_version": 1,
        "compact_mode": "temporal-expert-v1",
        "date_start": DAYS[0],
        "date_end": DAYS[-1],
        "dates": DAYS,
        "empty_dates": [],
        "max_context": 320,
        "selection": {
            "value": SPECIALIST,
            "acting_seat_archetype": SPECIALIST,
            "seat_semantics": "acting_seat_only",
        },
        "shards": shards,
        "totals": {
            "records_kept": totals["records"],
            "decisions_kept": totals["decisions"],
            "combo_state_rows": totals["combo_state_rows"],
            "days": len(DAYS),
        },
    }
    path = EXPERT / "manifest.json"
    _atomic(path, manifest)
    ready = {
        "schema": "poke_bot.current_deck_guide_corpus_ready/v1",
        "status": "ready",
        "specialist_id": SPECIALIST,
        "guide_version": "teal-mask-ogerpon-ex-slop-box-north-star-v3",
        "manifest": str(path),
        "dataset_schema": 8,
        "packing_schema": 8,
        "records_kept": totals["records"],
        "decisions_kept": totals["decisions"],
        "combo_state_rows": totals["combo_state_rows"],
        "setup_board_labels_present": True,
        "combo_state_labels_present": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seal_receipt": str(SEAL),
        "window": {"start": DAYS[0], "end": DAYS[-1]},
    }
    _atomic(EXPERT / "CURRENT_DECK_GUIDE_CORPUS_READY.json", ready)
    man_digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    protected = {
        "schema": "poke_bot.pinned_expert_corpus/v1",
        "protected": True,
        "status": "protected",
        "specialist_id": SPECIALIST,
        "manifest": str(path),
        "manifest_sha256": man_digest,
        "guide_ready": str(EXPERT / "CURRENT_DECK_GUIDE_CORPUS_READY.json"),
        "dataset_schema": 8,
        "packing_schema": 8,
        "records_kept": totals["records"],
        "decisions_kept": totals["decisions"],
        "combo_state_rows": totals["combo_state_rows"],
        "setup_board_labels_present": True,
        "combo_state_labels_present": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "window": {"start": DAYS[0], "end": DAYS[-1]},
    }
    _atomic(EXPERT / "PROTECTED_EXPERT_CORPUS.json", protected)
    return EXPERT / "PROTECTED_EXPERT_CORPUS.json"


def main() -> int:
    print("[cpu-pack-r175] waiting for Jul24-Aug5 seal + synced shards", flush=True)
    while True:
        shard_n = sum(
            1 for day in DAYS if (EXPERT / f"{SPECIALIST}-{day}.features").is_file()
        )
        if SEAL.is_file() and shard_n == len(DAYS):
            break
        print(
            f"[cpu-pack-r175] seal={SEAL.is_file()} shards={shard_n}/{len(DAYS)}",
            flush=True,
        )
        time.sleep(30)
    seal = json.loads(SEAL.read_text())
    if int(seal.get("records") or 0) <= 0:
        raise SystemExit("seal has zero records")
    protected = assemble_manifest(seal)
    print("[cpu-pack-r175] building with --require-combo-state", protected, flush=True)
    CPU_PACK.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYBIN,
        "-u",
        str(ROOT / "scripts/prebuild_expert_cpu_pack.py"),
        "--corpus",
        str(protected),
        "--cache-root",
        str(CPU_PACK),
        "--receipt",
        str(CPU_PRE),
        "--archetype",
        SPECIALIST,
        "--belief-card-vocab",
        "2048",
        "--min-decisions",
        "1000",
        "--pack-workers",
        "4",
        "--require-combo-state",
    ]
    print("RUN", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(rc)
    pre = json.loads(CPU_PRE.read_text())
    ready = {
        "schema": "poke_bot.slop_box_schema8_jul24_aug5_cpu_pack_ready_r175/v1",
        "status": "ready_bound",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "packing_schema": 8,
        "dataset_schema": 8,
        "setup_board_labels_present": True,
        "any_setup_labels": True,
        "combo_state_labels_present": True,
        "combo_state_targets": True,
        "pack_includes_combo_state": True,
        "require_combo_state": True,
        "records": int(seal.get("records") or 0),
        "records_kept": int(seal.get("records") or 0),
        "decisions": int((seal.get("totals") or {}).get("decisions") or 0),
        "cpu_pack": {
            "root": str(CPU_PACK),
            "packing_schema": 8,
            "prebuild_receipt": str(CPU_PRE),
            "decisions": pre.get("decisions"),
            "samples": pre.get("samples"),
            "require_combo_state": True,
            "combo_state_targets": True,
        },
        "protected_expert_corpus": str(protected),
        "expert": str(protected),
        "guide_ready": str(EXPERT / "CURRENT_DECK_GUIDE_CORPUS_READY.json"),
        "seal_receipt": str(SEAL),
        "window": {"start": DAYS[0], "end": DAYS[-1]},
        "bootstrap_held": True,
        "do_not_start_bootstrap": True,
    }
    _atomic(CPU_READY, ready)
    print("CPU_PACK_READY", CPU_READY, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
