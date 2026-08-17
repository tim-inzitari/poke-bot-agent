#!/usr/bin/env python3
"""Bind raw-record tactical roots to the exact Inzi compact-sidecar ABI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def commit_text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--index-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source = args.overlay.resolve(strict=True)
    index = args.index.resolve(strict=True)
    output = args.output.resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("mode") != "shadow_only" or payload.get("planner_dispatch_authority") is not False:
        raise ValueError("source tactical overlay authority changed")
    rows = list(payload.get("rows") or ())
    if len(rows) != int(payload.get("roots", -1)) or len(rows) < 1200:
        raise ValueError("source tactical overlay coverage changed")

    connection = sqlite3.connect(f"file:{index}?mode=ro", uri=True)
    projected: list[dict] = []
    changed = 0
    try:
        for source_row in rows:
            row = dict(source_row)
            matches = connection.execute(
                "SELECT observation_fingerprint,payload FROM rows "
                "WHERE episode_id=? AND seat=? AND env_step=?",
                (str(row["episode_id"]), int(row["seat"]), int(row["env_step"])),
            ).fetchall()
            if len(matches) != 1:
                raise ValueError("tactical root lacks one exact sidecar three-key match")
            canonical_fingerprint, raw_sidecar = matches[0]
            sidecar = json.loads(str(raw_sidecar))
            stages = list(sidecar.get("policy_stage_option_features") or ())
            target = dict(row.get("target") or {})
            target_rows = list(target.get("rows") or ())
            if len(stages) != 1:
                raise ValueError("tactical root is not a complete single-stage action")
            stage = dict(stages[0])
            action_digest = str(stage.get("action_combos_fingerprint") or "")
            if (
                not action_digest.startswith("sha256:")
                or target.get("root_legal_order_fingerprint") != action_digest[7:]
                or int(stage.get("candidate_count", -1)) != len(target_rows)
            ):
                raise ValueError("tactical root action-menu identity changed")
            original = str(row.get("observation_fingerprint") or "")
            canonical = str(canonical_fingerprint)
            if original != canonical:
                changed += 1
            row["source_observation_fingerprint"] = original
            row["observation_fingerprint"] = canonical
            projected.append(row)
    finally:
        connection.close()

    keys = {
        (str(row["episode_id"]), int(row["seat"]), int(row["env_step"]), str(row["observation_fingerprint"]))
        for row in projected
    }
    if len(keys) != len(projected):
        raise ValueError("re-keyed tactical roots are not unique")
    payload["rows"] = projected
    payload["compact_rekey_projection"] = {
        "schema": "poke_bot.r274_tactical_compact_rekey/v1",
        "host": "inzi",
        "source_overlay": {"path": str(source), "sha256": sha256_file(source)},
        "sidecar_index": {"path": str(index), "sha256": args.index_sha256},
        "roots": len(projected),
        "changed_observation_fingerprints": changed,
        "action_menu_digest_matches": len(projected),
        "planner_dispatch_authority": False,
    }
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    commit_text(output, body)
    receipt = {
        "schema": "poke_bot.r274_tactical_compact_rekey_receipt/v1",
        "output": {
            "path": str(output),
            "sha256": sha256_file(output),
            "size_bytes": output.stat().st_size,
        },
        **payload["compact_rekey_projection"],
    }
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    commit_text(
        receipt_path,
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
    )
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
