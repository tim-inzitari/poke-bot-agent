#!/usr/bin/env python3
"""Repair the sealed Slop Box full-window guide-ready projection, held/off."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/pokebot/poke-bot-agent")
CORPUS = (
    ROOT
    / "data/bootstrap/expert-slop-box-schema8-jul24-aug5-r175/"
    "teal-mask-ogerpon-ex"
)
READY = CORPUS / "CURRENT_DECK_GUIDE_CORPUS_READY.json"
MANIFEST = CORPUS / "manifest.json"
POINTER = CORPUS / "PROTECTED_EXPERT_CORPUS.json"
OUTPUT = ROOT / "outputs/state/slop-box-full-window-guide-ready-repair-r194.json"
EXPECTED_READY = "sha256:3862bb80b39e4fd7c3ae5e83ba10b33f4633f6df847cc06afdc9601c866b8b4e"
EXPECTED_MANIFEST = "sha256:a7a71d4af9de38f454412e00a9a198ac85a070a183e1ae6555230c7b3d801371"
EXPECTED_POINTER = "sha256:75284fd0065873dab2d4577c060b7c2732637ed422d999769e26aab53f39ae2e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def held(unit: str) -> bool:
    active = subprocess.check_output(
        ["systemctl", "--user", "show", unit, "-pActiveState", "--value"],
        text=True,
    ).strip()
    definition = subprocess.check_output(
        ["systemctl", "--user", "cat", unit, "--no-pager"], text=True
    )
    return active == "inactive" and "ExecStart=/bin/false" in definition


def main() -> int:
    bootstrap = "pokebot-final-format-slop-box-h10-rtp-bootstrap.service"
    rl = "pokebot-final-format-slop-box-h10-rtp-rl.service"
    if not held(bootstrap) or not held(rl):
        raise RuntimeError("refusing guide receipt repair unless CE and RL are held")
    actual = {"ready": sha256(READY), "manifest": sha256(MANIFEST), "pointer": sha256(POINTER)}
    expected = {
        "ready": EXPECTED_READY,
        "manifest": EXPECTED_MANIFEST,
        "pointer": EXPECTED_POINTER,
    }
    if actual != expected:
        raise RuntimeError(f"guide-ready repair preimage changed: {actual}")

    ready = read_json(READY)
    manifest = read_json(MANIFEST)
    pointer = read_json(POINTER)
    totals = dict(manifest.get("totals") or {})
    coverage = dict(totals.get("target_coverage") or {})
    if (
        ready.get("schema") != "poke_bot.current_deck_guide_corpus_ready/v1"
        or ready.get("status") != "ready"
        or ready.get("specialist_id") != "teal-mask-ogerpon-ex"
        or ready.get("guide_version") != "teal-mask-ogerpon-ex-slop-box-north-star-v3"
        or pointer.get("schema") != "poke_bot.pinned_expert_corpus/v1"
        or pointer.get("protected") is not True
        or pointer.get("manifest_sha256") != EXPECTED_MANIFEST
        or int(totals.get("records_kept") or 0) != 2699
        or int(totals.get("decisions_kept") or 0) != 182951
        or int(coverage.get("guide_rows") or 0) != 16580
    ):
        raise RuntimeError("sealed guide source contract changed")

    backup = READY.with_name(READY.name + ".pre-r194-3862bb80")
    if backup.exists():
        if sha256(backup) != EXPECTED_READY:
            raise RuntimeError(f"guide-ready backup identity changed: {backup}")
    else:
        shutil.copy2(READY, backup)
    repaired = {
        **ready,
        "active_training_modified": False,
        "records": 2699,
        "decisions": 182951,
        "guide_rows": 16580,
        "manifest_sha256": EXPECTED_MANIFEST,
        "pointer_manifest_sha256": EXPECTED_MANIFEST,
        "protected_pointer": str(POINTER),
        "protected_pointer_sha256": EXPECTED_POINTER,
        "repair": {
            "schema": "poke_bot.slop_box_full_window_guide_ready_repair/v1",
            "goal_revision": 194,
            "reason": "materialize_exact_trainer_required_projection_fields",
            "source_preimage_sha256": EXPECTED_READY,
            "training_started": False,
            "ce_and_rl_held": True,
        },
    }
    atomic_json(READY, repaired)
    postimage = sha256(READY)
    receipt = {
        "schema": "poke_bot.slop_box_full_window_guide_ready_repair/v1",
        "status": "repaired_while_activation_held",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "guide_ready": str(READY),
        "preimage_sha256": EXPECTED_READY,
        "postimage_sha256": postimage,
        "backup": str(backup),
        "backup_sha256": sha256(backup),
        "manifest": str(MANIFEST),
        "manifest_sha256": EXPECTED_MANIFEST,
        "protected_pointer": str(POINTER),
        "protected_pointer_sha256": EXPECTED_POINTER,
        "records": 2699,
        "decisions": 182951,
        "guide_rows": 16580,
        "bootstrap_held": True,
        "rl_held": True,
        "activation_allowed": False,
    }
    atomic_json(OUTPUT, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
