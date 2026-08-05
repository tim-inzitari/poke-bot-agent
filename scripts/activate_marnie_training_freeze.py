#!/usr/bin/env python3
"""Select an existing Marnie milestone checkpoint as the training freeze.

This does not upload anything.  It verifies the exact milestone bundle and
records an owner-authorized, checksum-bound training-use freeze for Crustle.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--milestone", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    source_path = args.milestone.resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    bundle = dict(source.get("bundle") or {})
    queue = dict(source.get("queue_entry") or {})
    digest = str(source.get("checkpoint_sha256") or "")
    if (
        source.get("schema") != "poke_bot.final_format_milestone_submission/v1"
        or source.get("status") not in {"queued", "submitted", "complete"}
        or bundle.get("specialist_id") != "marnie-s-grimmsnarl-ex"
        or int(source.get("iteration", -1)) != 9
        or digest != "sha256:f20efb20f5c30820c7e23004e529d326ec87f91b026c1fe3bbb431f9c8b44381"
        or int(queue.get("copy_number", -1)) != 1
        or queue.get("turn_order_preference") != "first_if_allowed"
        or dict(bundle.get("contents") or {}).get("model_sha256") != digest
        or bundle.get("turn_order_preference") != "first_if_allowed"
    ):
        raise RuntimeError("Marnie iter-9 milestone is not the requested exact freeze")
    checkpoint = Path(str(source.get("checkpoint") or "")).expanduser().resolve()
    bundle_path = Path(str(bundle.get("path") or "")).expanduser().resolve()
    if not checkpoint.is_file() or sha(checkpoint) != digest:
        raise RuntimeError("Marnie freeze checkpoint is missing or changed")
    if not bundle_path.is_file() or sha(bundle_path) != str(bundle.get("sha256") or ""):
        raise RuntimeError("Marnie freeze bundle is missing or changed")
    receipt = {
        "schema": "poke_bot.marnie_training_freeze/v1",
        "status": "owner_selected_training_freeze",
        "owner_decision_revision": 163,
        "specialist_id": "marnie-s-grimmsnarl-ex",
        "iteration": 9,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "submission_bundle": str(bundle_path),
        "submission_bundle_sha256": str(bundle["sha256"]),
        "milestone_receipt": str(source_path),
        "milestone_receipt_sha256": sha(source_path),
        "copy_number": 1,
        "copy_count": 1,
        "turn_order_preference": "first_if_allowed",
        "kaggle_upload_requested": False,
        "training_use_only": True,
        "historical_iter20_freeze_preserved": True,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        old = json.loads(out.read_text(encoding="utf-8"))
        if old != receipt:
            raise RuntimeError("canonical Marnie training freeze already differs")
    else:
        tmp = out.with_name(f".{out.name}.{__import__('os').getpid()}.tmp")
        tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(out)
    print(json.dumps({"status": receipt["status"], "checkpoint_sha256": digest, "receipt": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
