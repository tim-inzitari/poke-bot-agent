#!/usr/bin/env python3
"""Seal Crustle iter5 resume sidecar after the 4-cell refill completed."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

RUN = Path(
    "/home/inzi/poke-bot-agent/outputs/pure_rl/final_format_crustle_r113_h10_i_v6_8k"
)
SHARD = RUN / "shards" / "iter_00005.jsonl"
SIDE = RUN / "shards" / "iter_00005.resume_r167.json"
ARMED = Path(
    "/home/inzi/poke-bot-agent/outputs/state/crustle-iter5-corpus-restore-armed-r167.json"
)
ORIG_SIZE = 2320791140
AUTHORIZED_Q = (
    "sha256:2de205bee91f8a3db37d4bc9a964c0a045dc15a70da794aa8247bdb6d0c3064a"
)
ORIG_MISSING = {8017, 8072, 8128, 8183}


def main() -> int:
    side = json.loads(SIDE.read_text(encoding="utf-8"))
    assert str(side.get("authorized_quarantine_sha256")) == AUTHORIZED_Q

    with SHARD.open("rb") as fh:
        fh.seek(ORIG_SIZE)
        tail = fh.read().decode("utf-8")
    appended = []
    appended_decisions = 0
    for line in tail.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ji = int((row.get("target_provenance") or {})["collection_job_index"])
        appended.append(ji)
        appended_decisions += len(row.get("decisions") or [])
    assert sorted(set(appended)) == sorted(ORIG_MISSING), appended
    assert len(appended) == 4, appended

    retained = sorted(set(range(8192)))
    digest_h = hashlib.sha256()
    with SHARD.open("rb") as fh:
        while True:
            chunk = fh.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest_h.update(chunk)
    digest = "sha256:" + digest_h.hexdigest()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    bak = SIDE.with_name(
        f"iter_00005.resume_r167.json.pre_seal_{now.replace(':', '')}.bak"
    )
    bak.write_bytes(SIDE.read_bytes())

    side.update(
        {
            "unique_games": 8192,
            "unique_decisions": int(370469 + appended_decisions),
            "active_size": int(SHARD.stat().st_size),
            "active_sha256": digest,
            "missing_job_indices": [],
            "retained_job_indices": retained,
            "mode": "resume_public_mix_refill_only",
            "refill_complete": {
                "completed_job_indices": sorted(ORIG_MISSING),
                "appended_decisions": appended_decisions,
                "sealed_at_utc": now,
                "notes": "all 4 missing public_mix cells appended; ready for learner",
            },
        }
    )
    SIDE.write_text(json.dumps(side) + "\n", encoding="utf-8")

    ARMED.write_text(
        json.dumps(
            {
                "schema": "poke_bot.crustle_iter5_corpus_restore_armed_r167/v1",
                "created_at_utc": now,
                "sidecar": str(SIDE),
                "active_sha256": digest,
                "active_size": int(SHARD.stat().st_size),
                "unique_games": 8192,
                "missing_job_indices": [],
                "authorized_quarantine_sha256": AUTHORIZED_Q,
                "bootstrap_hooks": [
                    "promote_poller",
                    "iter5_corpus_restore_hide_for_kick",
                ],
                "mode": "resume_complete_awaiting_learner",
                "refill_complete": side["refill_complete"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "unique_games": 8192,
                "unique_decisions": side["unique_decisions"],
                "active_size": side["active_size"],
                "active_sha256": digest,
                "missing": [],
                "bak": str(bak),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
