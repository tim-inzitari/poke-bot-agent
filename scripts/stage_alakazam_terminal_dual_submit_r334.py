#!/usr/bin/env python3
"""Build and queue the two owner-authorized terminal Alakazam packages."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.handle_passed_gate import (  # noqa: E402
    _copy_submission_slot,
    build_submission_bundle,
    queue_submission_copies,
)


EXPECTED_PARENT = (
    "sha256:60b3b4b697f203698de5a580fe65da376f1be4b97fc47c473d214cb7ec25331d"
)
DECKS = (
    (
        "four-enhanced-hammer",
        Path(
            "/home/inzi/poke-bot-agent-deployments/"
            "alakazam-rule-derivative-g5-r10-runtime-e241f15408c7/"
            "decks/archetype-samples/alakazam-new-list-direct-r241.csv"
        ),
        "sha256:d834c66c5a3629dd79c8533a04fde770a22ca8590ac55c9868440121b6df5fba",
    ),
    (
        "original-r195-55378392",
        Path(
            "/home/inzi/poke-bot-agent/baselines/specialists/"
            "alakazam-r195-no-rtp-submission-55378392/deck.csv"
        ),
        "sha256:1705f0f4db0c54b32f297fc9292a417b0c3abc9fdb6edf6a5370af6a635efe65",
    ),
)


def sha256(path: Path) -> str:
    block = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            block.update(chunk)
    return "sha256:" + block.hexdigest()


def canonical_digest(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(body).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--matchup-tree", type=Path, required=True)
    parser.add_argument("--cg-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--refresh-unit")
    parser.add_argument("--wait-seconds", type=int, default=0)
    args = parser.parse_args()

    terminal_path = args.run_dir / "terminal_expert_refresh.json"
    deadline = time.monotonic() + max(0, int(args.wait_seconds))
    while not terminal_path.is_file() and time.monotonic() < deadline:
        if args.refresh_unit:
            state = subprocess.check_output(
                [
                    "systemctl",
                    "--user",
                    "show",
                    args.refresh_unit,
                    "--property=ActiveState",
                    "--value",
                ],
                text=True,
            ).strip()
            if state not in {"active", "activating"}:
                raise RuntimeError(
                    "terminal refresh stopped before publishing its receipt"
                )
        time.sleep(30)
    if not terminal_path.is_file():
        raise RuntimeError("timed out waiting for terminal refresh receipt")
    terminal = read_json(terminal_path)
    if (
        terminal.get("schema") != "poke_bot.terminal_expert_soft_refresh/v1"
        or int(terminal.get("before_iteration", -1)) != 14
        or int(terminal.get("epochs_completed", -1)) != 1
        or terminal.get("next_collection_started") is not False
        or str((terminal.get("parent") or {}).get("digest") or "")
        != EXPECTED_PARENT
    ):
        raise RuntimeError("terminal refresh receipt does not match revision 334")
    refreshed = dict(terminal.get("refreshed") or {})
    checkpoint = Path(str(refreshed.get("path") or "")).resolve()
    checkpoint_digest = str(refreshed.get("digest") or "")
    if not checkpoint.is_file() or sha256(checkpoint) != checkpoint_digest:
        raise RuntimeError("terminal refreshed checkpoint is missing or drifted")

    if args.output_root.exists():
        raise RuntimeError("terminal dual-submission root already exists")
    args.output_root.mkdir(parents=True)
    copies: list[dict[str, Any]] = []
    bundles: list[dict[str, Any]] = []
    for slot, (role, source_deck, expected_deck_sha) in enumerate(DECKS, 1):
        if sha256(source_deck) != expected_deck_sha:
            raise RuntimeError(f"{role} deck digest drifted")
        cards = [int(row.strip()) for row in source_deck.read_text().splitlines() if row.strip()]
        if len(cards) != 60:
            raise RuntimeError(f"{role} deck is not 60 cards")
        role_root = args.output_root / role
        role_root.mkdir()
        pinned_deck = role_root / "deck.csv"
        shutil.copy2(source_deck, pinned_deck)
        representatives = role_root / "representatives.json"
        write_json_exclusive(
            representatives,
            {
                "schema": "poke_bot.specialist_deck_representatives/v1",
                "decks": {
                    "alakazam": {
                        "card_ids": cards,
                        "cards_sha256": canonical_digest(cards),
                        "source_deck_id": role,
                    }
                },
            },
        )
        deck_receipt = {
            "path": str(pinned_deck.resolve()),
            "cards": 60,
            "cards_sha256": canonical_digest(cards),
            "file_sha256": sha256(pinned_deck),
            "representatives": str(representatives.resolve()),
            "representatives_sha256": sha256(representatives),
        }
        bundle = build_submission_bundle(
            repo_root=args.runtime_root,
            frozen_manifest={
                "model_path": str(checkpoint),
                "checkpoint_digest": checkpoint_digest,
            },
            deck_receipt=deck_receipt,
            output_dir=role_root / "build",
            python=args.python,
            archetype="alakazam",
            matchup_tree=args.matchup_tree,
            cg_root=args.cg_root,
            turn_order_preference="first_if_allowed",
            rtp_mode="off",
            direct_no_search_assets=True,
        )
        copy = _copy_submission_slot(bundle, args.output_root, slot)
        copy["label_suffix"] = role
        copies.append(copy)
        bundles.append(bundle)

    queued = queue_submission_copies(
        queue_path=args.queue,
        copies=copies,
        gate_plan={
            "checkpoint_digest": checkpoint_digest,
            "gate_id": "alakazam-terminal-refresh-dual-submit-r334",
            "iteration": 14,
            "completion_authority": "owner_terminal_refresh",
            "owner_decision_source": "GOAL.md#revision-334",
        },
        specialist_id="alakazam",
        competition="pokemon-tcg-ai-battle",
    )
    receipt = {
        "schema": "poke_bot.alakazam_terminal_dual_submission_r334/v1",
        "status": "queued",
        "terminal_refresh": {
            "path": str(terminal_path.resolve()),
            "sha256": sha256(terminal_path),
        },
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_digest},
        "packages": bundles,
        "queue_entries": queued,
        "network_uploads_authorized": 2,
        "rtp_enabled": False,
        "search_or_mcts_packaged": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt_path = args.output_root / "terminal-dual-submission-receipt.json"
    write_json_exclusive(receipt_path, receipt)
    print(json.dumps({"receipt": str(receipt_path), **receipt}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
