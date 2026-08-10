#!/usr/bin/env python3
"""Validate and atomically promote the r175 Slop Box matchup-markup result.

This is an operational promotion only.  It preserves the offline-oracle receipt
byte-for-byte and leaves both Slop Box bootstrap and RL held.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/inzi/poke-bot-agent")
INCOMING = (
    ROOT
    / "outputs/bootstrap/slop-box-h10-rtp/"
    ".incoming-promotable-expert-matchup-jul24-aug5-elmo-r194-f4340588"
)
CANONICAL = (
    ROOT
    / "outputs/bootstrap/slop-box-h10-rtp/"
    "expert-matchup-marked-selfplay-equiv-jul24-aug5-r175"
)
STATE = ROOT / "outputs/state"
SOURCE_RECEIPT = INCOMING / "receipt.json"
CANONICAL_RECEIPT = (
    STATE / "slop-box-h10-rtp-expert-matchup-markup-jul24-aug5-r175.json"
)
SUMMARY_RECEIPT = (
    STATE
    / "slop-box-h10-rtp-expert-matchup-markup-jul24-aug5-summary-r175.json"
)
PROMOTION_RECEIPT = (
    STATE / "slop-box-h10-rtp-expert-matchup-markup-jul24-aug5-promotion-r194.json"
)
SEAL = STATE / "slop-box-schema8-jul24-aug5-sealed-r175.json"
CPU_READY = STATE / "slop-box-schema8-jul24-aug5-cpu-pack-ready-r175.json"

EXPECTED_SOURCE_MANIFEST = (
    "sha256:a7a71d4af9de38f454412e00a9a198ac85a070a183e1ae6555230c7b3d801371"
)
EXPECTED_SOURCE_RECEIPT_FILE = (
    "sha256:f43405883d71df7596cfaeb1b27c1216af319ab1061bb2552b5c191ee9d44b8c"
)
EXPECTED_MANIFEST_FILE = (
    "sha256:eecd5f566266cafbd0c625043b6ac5b83a0207f3102f3856e3ae91b4884512e3"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_copy(source: Path, target: Path) -> None:
    temporary = target.with_name(target.name + f".tmp.{os.getpid()}")
    shutil.copyfile(source, temporary)
    temporary.replace(target)


def require_regular_directory(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"expected regular directory: {path}")


def main() -> int:
    require_regular_directory(INCOMING)
    final = INCOMING / "final"
    require_regular_directory(final)
    if CANONICAL.exists() or CANONICAL.is_symlink():
        raise RuntimeError(f"canonical destination already exists: {CANONICAL}")
    for path in (CANONICAL_RECEIPT, SUMMARY_RECEIPT, PROMOTION_RECEIPT):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"refusing to overwrite receipt: {path}")

    if sha256(SOURCE_RECEIPT) != EXPECTED_SOURCE_RECEIPT_FILE:
        raise RuntimeError("Elmo source receipt file digest mismatch")
    manifest_path = final / "manifest.json"
    if sha256(manifest_path) != EXPECTED_MANIFEST_FILE:
        raise RuntimeError("markup manifest file digest mismatch")

    receipt = json.loads(SOURCE_RECEIPT.read_text())
    manifest = json.loads(manifest_path.read_text())
    ready = json.loads((final / "MARKUP_READY.json").read_text())
    seal = json.loads(SEAL.read_text())
    cpu_ready = json.loads(CPU_READY.read_text())

    if (
        receipt.get("source_games"),
        receipt.get("source_decisions"),
        receipt.get("marked_games"),
        receipt.get("marked_decisions"),
    ) != (2699, 182951, 2515, 171243):
        raise RuntimeError("unexpected source/marked totals")
    if receipt.get("excluded") != {"unsupported_or_unroutable": 184}:
        raise RuntimeError("unexpected markup exclusion totals")
    if sum(receipt.get("route_sequences", {}).values()) != 2515:
        raise RuntimeError("route sequence total mismatch")
    if sum(receipt.get("route_decisions", {}).values()) != 171243:
        raise RuntimeError("route decision total mismatch")
    if receipt.get("source_feature_manifest_digest") != EXPECTED_SOURCE_MANIFEST:
        raise RuntimeError("source feature manifest identity mismatch")
    if not receipt.get("do_not_fit_on_pre_wipe_weights"):
        raise RuntimeError("pre-wipe training prohibition missing")
    if receipt.get("status") != "ready_for_wipe_and_fresh_adapter_bootstrap":
        raise RuntimeError("unexpected markup status")

    if not manifest.get("offline_oracle_only"):
        raise RuntimeError("offline-oracle-only flag missing")
    if manifest.get("runtime_routes_enabled") is not False:
        raise RuntimeError("runtime routes must remain disabled in markup")
    if not manifest.get("do_not_fit_on_pre_wipe_weights"):
        raise RuntimeError("manifest pre-wipe training prohibition missing")
    if manifest.get("source_feature_manifest_digest") != EXPECTED_SOURCE_MANIFEST:
        raise RuntimeError("manifest source identity mismatch")
    if manifest.get("membership_digest") != receipt.get("membership_digest"):
        raise RuntimeError("membership digest mismatch")
    if ready.get("membership_digest") != receipt.get("membership_digest"):
        raise RuntimeError("ready-marker membership digest mismatch")
    if sha256(final / "MARKUP_READY.json") != receipt.get("receipt_digest"):
        raise RuntimeError("ready-marker digest mismatch")
    if sha256(final / "ticket_index.jsonl") != receipt.get("ticket_index_digest"):
        raise RuntimeError("ticket-index digest mismatch")
    if sha256(final / "opponent-registry.json") != receipt.get(
        "package_registry_digest"
    ):
        raise RuntimeError("opponent-registry digest mismatch")

    marked_games = 0
    marked_decisions = 0
    for shard in manifest.get("shards", []):
        path = final / shard["path"]
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing regular markup shard: {path}")
        if path.stat().st_size != shard["bytes"] or sha256(path) != shard["sha256"]:
            raise RuntimeError(f"markup shard identity mismatch: {path}")
        marked_games += int(shard["stats"]["records_kept"])
        marked_decisions += int(shard["stats"]["decisions_kept"])
    if len(manifest.get("shards", [])) != 13:
        raise RuntimeError("expected exactly 13 Jul24-Aug5 markup shards")
    if (marked_games, marked_decisions) != (2515, 171243):
        raise RuntimeError("markup shard totals mismatch")

    if not seal.get("bootstrap_held") or not seal.get("do_not_start_bootstrap"):
        raise RuntimeError("source seal no longer holds bootstrap")
    if not cpu_ready.get("bootstrap_held") or not cpu_ready.get(
        "do_not_start_bootstrap"
    ):
        raise RuntimeError("CPU-pack receipt no longer holds bootstrap")
    if (
        seal.get("totals", {}).get("records"),
        seal.get("totals", {}).get("decisions"),
        cpu_ready.get("records"),
        cpu_ready.get("decisions"),
    ) != (2699, 182951, 2699, 182951):
        raise RuntimeError("sealed source/CPU-pack totals mismatch")

    # The output tree is immutable after this same-filesystem atomic rename.
    final.replace(CANONICAL)
    atomic_copy(SOURCE_RECEIPT, CANONICAL_RECEIPT)
    now = datetime.now(timezone.utc).isoformat()
    summary = {
        "schema": "poke_bot.slop_box_markup_jul24_aug5_summary_r175/v1",
        "status": "ready",
        "updated_at_utc": now,
        "receipt": str(CANONICAL_RECEIPT),
        "output_dir": str(CANONICAL),
        "marked_games": 2515,
        "marked_decisions": 171243,
        "source_games": 2699,
        "source_decisions": 182951,
        "window": {"start": "2026-07-24", "end": "2026-08-05"},
        "offline_oracle_only": True,
        "bootstrap_held": True,
        "do_not_start_bootstrap": True,
    }
    atomic_json(SUMMARY_RECEIPT, summary)
    promotion = {
        "schema": "poke_bot.slop_box_markup_elmo_promotion_r194/v1",
        "status": "promoted_verified_offline_markup",
        "promoted_at_utc": now,
        "canonical_output": str(CANONICAL),
        "canonical_receipt": str(CANONICAL_RECEIPT),
        "source": {
            "host": "192.168.1.143",
            "container_name": "pokebot-slop-box-markup-jul24-aug5-r194",
            "container_id": (
                "837c16200cb6f51c177e56a99ea87533ca2f2bcbbeae684bad1715b87e918bed"
            ),
            "image": (
                "sha256:0bcf2305438f8feecd9420cc37af8da4e3a2d81986e112597ad38fbe1e3f1aa3"
            ),
            "exit_code": 0,
            "oom_killed": False,
            "network_mode": "none",
            "read_only_root": True,
            "cpus": 16,
            "memory_bytes": 25769803776,
            "source_receipt_file_sha256": EXPECTED_SOURCE_RECEIPT_FILE,
            "manifest_file_sha256": EXPECTED_MANIFEST_FILE,
        },
        "contract": {
            "window": {"start": "2026-07-24", "end": "2026-08-05"},
            "source_games": 2699,
            "source_decisions": 182951,
            "marked_games": 2515,
            "marked_decisions": 171243,
            "excluded": {"unsupported_or_unroutable": 184},
            "offline_oracle_only": True,
            "runtime_routes_enabled": False,
            "do_not_fit_on_pre_wipe_weights": True,
            "bootstrap_held": True,
            "rl_held": True,
        },
        "inputs": {
            "sealed_source_receipt": str(SEAL),
            "sealed_source_receipt_sha256": sha256(SEAL),
            "cpu_pack_ready_receipt": str(CPU_READY),
            "cpu_pack_ready_receipt_sha256": sha256(CPU_READY),
            "source_feature_manifest_digest": EXPECTED_SOURCE_MANIFEST,
        },
    }
    atomic_json(PROMOTION_RECEIPT, promotion)
    print(json.dumps(promotion, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
