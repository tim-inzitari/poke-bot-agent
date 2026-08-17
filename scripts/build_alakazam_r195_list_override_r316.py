#!/usr/bin/env python3
"""Clone the validated derivative package and replace only its submitted deck."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import io
import json
import os
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL_PACKAGE_SHA = "sha256:e241f15408c725ab8b6a01e27c63f4b9e8a604c5f08a525f42acf89fe4e32efe"
DECK_PACKAGE_SHA = "sha256:dfa8bfccf9ee41d2205c7e30d817489391bb6295fa1ed1eff78c36fd8a8b7145"
MODEL_SHA = "sha256:8b59af9af1d715639bd3d63a84df7d608cee686c27aadf1c6dac3c971631a248"
DECK_FILE_SHA = "sha256:1705f0f4db0c54b32f297fc9292a417b0c3abc9fdb6edf6a5370af6a635efe65"
LABEL = "8b59af9af1d7 new policy on 55468965 r195 deck list NO RTP"


def sha_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def canonical_sha(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return sha_bytes(body)


def member_key(name: str) -> str:
    return name.removeprefix("./")


def regular_members(archive: tarfile.TarFile) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for member in archive.getmembers():
        if member.isfile():
            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError(f"cannot read package member {member.name}")
            key = member_key(member.name)
            if key in result:
                raise RuntimeError(f"duplicate package member {key}")
            result[key] = handle.read()
    return result


def cards_from_deck(data: bytes) -> list[int]:
    cards: list[int] = []
    for raw in data.decode("utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            cards.append(int(line.split(",", 1)[0]))
    if len(cards) != 60:
        raise RuntimeError(f"deck must contain exactly 60 ordered cards, got {len(cards)}")
    return cards


def write_json_create_only(path: Path, payload: Any) -> str:
    body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(fd, body)
        os.fsync(fd)
    finally:
        os.close(fd)
    return sha_bytes(body)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-package", type=Path, required=True)
    parser.add_argument("--deck-package", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    args = parser.parse_args()

    if args.output_root.exists():
        raise RuntimeError(f"create-only output already exists: {args.output_root}")
    if sha_file(args.model_package) != MODEL_PACKAGE_SHA:
        raise RuntimeError("validated derivative source package identity mismatch")
    if sha_file(args.deck_package) != DECK_PACKAGE_SHA:
        raise RuntimeError("submission 55468965 deck source package identity mismatch")

    with tarfile.open(args.model_package, "r:gz") as source, tarfile.open(args.deck_package, "r:gz") as deck_source:
        source_files = regular_members(source)
        deck_files = regular_members(deck_source)
        deck_data = deck_files["deck.csv"]
        if sha_bytes(deck_data) != DECK_FILE_SHA:
            raise RuntimeError("submission 55468965 deck.csv identity mismatch")
        if sha_bytes(source_files["model.pt"]) != MODEL_SHA:
            raise RuntimeError("validated derivative model identity mismatch")
        if source_files["deck.csv"] == deck_data:
            raise RuntimeError("source and override decks unexpectedly match")
        profile = json.loads(source_files["runtime_profile.json"])
        if not (
            profile.get("display") == "NO RTP"
            and profile.get("rtp_mode") == "off"
            and profile.get("recursive_turn_planner") == "disabled"
            and profile.get("rtp_sidecar_packaged") is False
            and profile.get("model_checkpoint_sha256") == MODEL_SHA
        ):
            raise RuntimeError("source runtime is not the validated NO RTP model profile")
        turn_profile = json.loads(source_files["turn_order_profile.json"])
        if turn_profile.get("turn_order_preference") != "first_if_allowed":
            raise RuntimeError("source turn-order profile changed")

        args.output_root.mkdir(parents=True, exist_ok=False)
        package = args.output_root / "submission.tar.gz"
        with package.open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
                with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as target:
                    for original in source.getmembers():
                        member = copy.copy(original)
                        if member.isfile():
                            data = deck_data if member_key(member.name) == "deck.csv" else source_files[member_key(member.name)]
                            member.size = len(data)
                            target.addfile(member, io.BytesIO(data))
                        else:
                            target.addfile(member)

    with tarfile.open(package, "r:gz") as built:
        built_files = regular_members(built)
    if set(built_files) != set(source_files):
        raise RuntimeError("derived package member set differs from validated source")
    changed = sorted(name for name in source_files if source_files[name] != built_files[name])
    if changed != ["deck.csv"]:
        raise RuntimeError(f"derived package changed forbidden members: {changed}")
    if sha_bytes(built_files["model.pt"]) != MODEL_SHA or sha_bytes(built_files["deck.csv"]) != DECK_FILE_SHA:
        raise RuntimeError("derived model/deck identity mismatch")

    cards = cards_from_deck(deck_data)
    with tempfile.TemporaryDirectory(prefix="alakazam-r195-list-smoke-") as temporary:
        work = Path(temporary)
        with tarfile.open(package, "r:gz") as built:
            built.extractall(work, filter="data")
        smoke_code = r'''
import importlib.util, json, os, sys
from pathlib import Path
root=Path.cwd(); sys.path.insert(0,str(root))
os.environ["POKEBOT_SUBMISSION_SEARCH_DISABLE"]="1"
spec=importlib.util.spec_from_file_location("r316_package_main",root/"main.py")
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
deck=module.agent({"logs":[],"current":None,"select":None})
assert len(deck)==60
assert os.environ["POKEBOT_USE_RECURSIVE_TURN_PLANNER"]=="0"
assert module._POLICY.use_recursive_turn_planner is False
print(json.dumps({"deck":deck,"rtp_mode":"off","search":False}))
'''
        smoke = subprocess.run(
            [str(args.python), "-B", "-c", smoke_code], cwd=work,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(work)},
            text=True, capture_output=True,
        )
        if smoke.returncode:
            raise RuntimeError(f"isolated package smoke failed:\n{smoke.stdout}\n{smoke.stderr}")
        smoke_result = json.loads(smoke.stdout.strip().splitlines()[-1])
        if smoke_result["deck"] != cards:
            raise RuntimeError("isolated package did not serve the submission 55468965 deck")

    manifest = [
        {"path": name, "sha256": sha_bytes(data), "size_bytes": len(data)}
        for name, data in sorted(built_files.items())
    ]
    receipt = {
        "schema": "poke_bot.alakazam_derivative_r195_list_override_package/v1",
        "goal_revision": 316,
        "dedicated_goal_revision": 14,
        "canonical_label": LABEL,
        "source_model_submission_id": 55486518,
        "source_model_package_path": str(args.model_package.resolve()),
        "source_model_package_sha256": MODEL_PACKAGE_SHA,
        "model_checkpoint_sha256": MODEL_SHA,
        "source_deck_submission_id": 55468965,
        "source_deck_package_path": str(args.deck_package.resolve()),
        "source_deck_package_sha256": DECK_PACKAGE_SHA,
        "deck_file_sha256": DECK_FILE_SHA,
        "deck_ordered_cards_sha256": canonical_sha(cards),
        "deck_canonical_multiset_sha256": canonical_sha(sorted(cards)),
        "deck_card_count": len(cards),
        "derived_package_path": str(package.resolve()),
        "derived_package_sha256": sha_file(package),
        "derived_package_size_bytes": package.stat().st_size,
        "member_manifest_sha256": canonical_sha(manifest),
        "changed_regular_members_from_model_source": changed,
        "all_other_regular_members_byte_identical": True,
        "rtp_mode": "disabled",
        "search_or_mcts_enabled": False,
        "turn_order_preference": "first_if_allowed",
        "isolated_smoke_passed": True,
        "authorized_submission_count": 1,
        "retry_authorized": False,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt_sha = write_json_create_only(args.output_root / "package-receipt.json", receipt)
    complete_sha = write_json_create_only(args.output_root / "COMPLETE.json", {
        "schema": "poke_bot.alakazam_derivative_r195_list_override_completion/v1",
        "status": "passed",
        "package_sha256": receipt["derived_package_sha256"],
        "package_receipt_sha256": receipt_sha,
        "changed_regular_members": changed,
    })
    print(json.dumps({**receipt, "package_receipt_sha256": receipt_sha, "completion_sha256": complete_sha}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
