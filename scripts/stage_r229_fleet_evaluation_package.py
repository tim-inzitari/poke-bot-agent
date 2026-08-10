#!/usr/bin/env python3
"""Safely overlay the r230/r229 evaluator onto the exact r228 archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Sequence

R228_ARCHIVE = "sha256:59531249f106d55d6606b186aee3d3a3e5ec8a3f0e9760c963e08cfd8b9d67d4"
MODEL = "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
TREE = "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049"
OVERLAYS = {
    "run_r229_process_watchdog.py": "scripts/run_r229_process_watchdog.py",
    "main.py": "submission/r228_async_eight_worker_main.py",
    "poke_bot/r228_async_shared_tree_queue.py": "poke_bot/r228_async_shared_tree_queue.py",
    "poke_bot/r228_kaggle_async_runtime.py": "poke_bot/r228_kaggle_async_runtime.py",
}


class StageError(RuntimeError):
    pass


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def tree_sha(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(row for row in root.rglob("*") if row.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        size = path.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return "sha256:" + digest.hexdigest()


def raise_packaged_action_cap(path: Path) -> None:
    old = b"MAX_ACTION_COMBOS: int = 4096"
    new = b"MAX_ACTION_COMBOS: int = 65536"
    payload = path.read_bytes()
    if payload.count(old) != 1 or new in payload:
        raise StageError("base r228 feature cap is not the exact 4,096 contract")
    path.write_bytes(payload.replace(old, new, 1))


def safe_extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        seen: set[str] = set()
        for member in tar.getmembers():
            clean = member.name.removeprefix("./")
            path = Path(clean)
            if not clean or path.is_absolute() or ".." in path.parts or clean in seen:
                raise StageError("r228 archive contains an unsafe or duplicate path")
            if not (member.isfile() or member.isdir()):
                raise StageError("r228 archive contains a link or special member")
            seen.add(clean)
            target = destination / path
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    raise StageError("r228 archive member is unreadable")
                with target.open("xb") as sink:
                    shutil.copyfileobj(source, sink)


def stage(*, source_root: Path, archive: Path, output: Path) -> dict:
    if sha(archive) != R228_ARCHIVE:
        raise StageError("input is not the exact confirmed-usable r228 archive")
    if output.exists():
        raise StageError("output already exists; refusing overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="r229-stage-", dir=output.parent) as raw:
        temporary = Path(raw) / "package"
        temporary.mkdir()
        safe_extract(archive, temporary)
        if sha(temporary / "model.pt") != MODEL or sha(temporary / "matchup_tree.json") != TREE:
            raise StageError("r228 frozen r195 identity drifted")
        overlay_hashes = {}
        feature_path = temporary / "poke_bot/features.py"
        raise_packaged_action_cap(feature_path)
        overlay_hashes["poke_bot/features.py"] = sha(feature_path)
        for destination, source in OVERLAYS.items():
            source_path = source_root / source
            if not source_path.is_file():
                raise StageError(f"missing overlay source: {source}")
            shutil.copy2(source_path, temporary / destination)
            overlay_hashes[destination] = sha(temporary / destination)
        manifest = {
            "schema": "poke_bot.alakazam_r228_vs_r195_no_mcts_fleet_bo1000_r233_package/v1",
            "status": "sealed_evaluation_only",
            "owner_goal_revision": 233,
            "base_r228_archive_sha256": R228_ARCHIVE,
            "checkpoint_sha256": MODEL,
            "matchup_tree_sha256": TREE,
            "complete_ordered_action_ceiling": 65536,
            "overlays": overlay_hashes,
            "package_payload_tree_sha256": tree_sha(temporary),
            "training_eligible": False,
        }
        (temporary / "r233_fleet_evaluation_manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n"
        )
        os.replace(temporary, output)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--r228-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = stage(source_root=args.source_root.resolve(), archive=args.r228_archive.resolve(), output=args.output.resolve())
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
