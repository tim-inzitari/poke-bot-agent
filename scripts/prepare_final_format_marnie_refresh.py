#!/usr/bin/env python3
"""Open the Marnie refresh pre-stage only after Alakazam is registered."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    body = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != body:
            raise RuntimeError(f"Marnie preparation receipt changed: {path}")
        return
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink(missing_ok=True)


def _static(args: argparse.Namespace) -> dict[str, str]:
    paths = {
        "state": args.specialist_state.expanduser().resolve(),
        "guide": args.guide.expanduser().resolve(),
        "expert": args.expert_manifest.expanduser().resolve(),
        "original": args.original_checkpoint.expanduser().resolve(),
        "refresh_registry": args.refresh_registry.expanduser().resolve(),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if args.check and missing == ["refresh_registry"]:
        missing = []
    if missing:
        raise RuntimeError("Marnie refresh input is missing: " + ",".join(missing))
    return {name: _sha256(path) for name, path in paths.items() if path.is_file()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--specialist-state", type=Path, required=True)
    parser.add_argument("--guide", type=Path, required=True)
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--refresh-registry", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    static = _static(args)
    if args.check:
        print("FINAL_FORMAT_MARNIE_PRESTAGE_OK " + json.dumps(static, sort_keys=True))
        return 0
    state = yaml.safe_load(args.specialist_state.read_text(encoding="utf-8"))
    phase = dict(state.get("post_fleet_refresh") or {})
    if (
        phase.get("completed_refresh_specialist_ids") != ["alakazam"]
        or phase.get("pending_specialist_ids") != ["marnie-s-grimmsnarl-ex"]
        or phase.get("active_refresh_specialist_id") != "marnie-s-grimmsnarl-ex"
        or phase.get("next_refresh_specialist_id") != "marnie-s-grimmsnarl-ex"
    ):
        raise RuntimeError("Marnie refresh boundary is not authorized")
    registry = json.loads(args.refresh_registry.read_text(encoding="utf-8"))
    refreshes = list(registry.get("refreshes") or [])
    if (
        registry.get("schema") != "poke_bot.post_fleet_refresh_registry/v1"
        or len(refreshes) != 1
        or dict(refreshes[0]).get("specialist_id") != "alakazam"
    ):
        raise RuntimeError("Alakazam refresh registration is not authoritative")
    receipt = {
        "schema": "poke_bot.final_format_marnie_refresh_prestage/v1",
        "status": "authorized_preparation_started",
        "specialist_id": "marnie-s-grimmsnarl-ex",
        "predecessor_refresh": "alakazam",
        "predecessor_registry_sha256": static["refresh_registry"],
        "guide_sha256": static["guide"],
        "expert_manifest_sha256": static["expert"],
        "original_checkpoint_sha256": static["original"],
        "original_checkpoint_immutable": True,
        "training_authority": False,
        "selector_authority": False,
        "next_boundary": "resolve_current_core_and_final_format_marnie_migration",
    }
    _write_once(args.receipt.expanduser().resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
