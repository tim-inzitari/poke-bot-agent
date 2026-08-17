#!/usr/bin/env python3
"""Install the exact submission-55378392 bundle as a local research baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from typing import Any


SCHEMA = "poke_bot.r274_r195_submission_55378392_research_baseline/v1"
BASELINE_ID = "alakazam-r195-no-rtp-submission-55378392"
SOURCE_BUNDLE_SHA256 = (
    "sha256:dfa8bfccf9ee41d2205c7e30d817489391bb6295fa1ed1eff78c36fd8a8b7145"
)
MODEL_SHA256 = (
    "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _write_create_only(path: Path, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_text(encoding="utf-8") != body:
            raise RuntimeError("existing r274 research-baseline receipt changed")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())


def _safe_extract(bundle: Path, target: Path) -> None:
    with tarfile.open(bundle, "r:gz") as archive:
        members = archive.getmembers()
        normalized: set[str] = set()
        for member in members:
            name = member.name.removeprefix("./")
            path = PurePosixPath(name)
            if not name or path.is_absolute() or ".." in path.parts:
                raise RuntimeError(f"unsafe r195 archive member: {member.name!r}")
            if member.issym() or member.islnk() or member.isdev():
                raise RuntimeError(f"unsupported r195 archive member: {member.name!r}")
            normalized.add(name)
        required = {"main.py", "deck.csv", "model.pt", "runtime_profile.json"}
        if not required <= normalized:
            raise RuntimeError("r195 archive lacks required runtime members")
        archive.extractall(target, filter="data")


def install(args: argparse.Namespace) -> dict[str, Any]:
    from poke_bot.baselines_runtime import baseline_content_digest

    bundle = args.bundle.expanduser().resolve()
    baseline_root = args.baseline_root.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if _sha256(bundle) != SOURCE_BUNDLE_SHA256:
        raise RuntimeError("source is not exact submission 55378392 NO-RTP bundle")
    target = baseline_root / "specialists" / BASELINE_ID
    if target.exists() or target.is_symlink():
        if not target.is_dir():
            raise RuntimeError("r274 research baseline target is not a directory")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{BASELINE_ID}.", dir=str(target.parent))
        )
        try:
            _safe_extract(bundle, temporary)
            if _sha256(temporary / "model.pt") != MODEL_SHA256:
                raise RuntimeError("installed research baseline model is not exact r195")
            os.rename(temporary, target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    profile = json.loads((target / "runtime_profile.json").read_text(encoding="utf-8"))
    if (
        profile.get("schema") != "poke_bot.submission_runtime_profile/v1"
        or profile.get("recursive_turn_planner") != "disabled"
        or profile.get("rtp_sidecar_packaged") is not False
        or _sha256(target / "model.pt") != MODEL_SHA256
    ):
        raise RuntimeError("installed research baseline is not exact direct NO-RTP r195")
    content_digest = baseline_content_digest(target)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    agents = list(manifest.get("agents") or [])
    row = {
        "id": BASELINE_ID,
        "name": "Alakazam r195 NO-RTP submission 55378392",
        "dir": BASELINE_ID,
        "group": "specialists",
        "source": "kaggle-submission:55378392",
    }
    existing = [item for item in agents if item.get("id") == BASELINE_ID]
    if existing and existing != [row]:
        raise RuntimeError("baseline manifest has a conflicting submission-55378392 row")
    if not existing:
        manifest["agents"] = [*agents, row]
        body = json.dumps(manifest, indent=2) + "\n"
        temporary = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
        temporary.write_text(body, encoding="utf-8")
        os.replace(temporary, manifest_path)
    receipt = {
        "schema": SCHEMA,
        "status": "installed_exact_local_research_baseline",
        "baseline_id": BASELINE_ID,
        "submission_id": 55378392,
        "source_bundle": str(bundle),
        "source_bundle_sha256": SOURCE_BUNDLE_SHA256,
        "installed_path": str(target.resolve()),
        "installed_content_sha256": content_digest,
        "model_sha256": MODEL_SHA256,
        "direct_policy_only": True,
        "rtp_enabled": False,
        "training_games_minimum_each_update": 128,
        "learner_first_minimum": 64,
        "learner_second_minimum": 64,
        "transport": "singleton_play",
    }
    _write_create_only(output, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = install(parser.parse_args())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
