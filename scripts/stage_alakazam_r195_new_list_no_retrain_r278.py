#!/usr/bin/env python3
"""Build a deck-only derivative of the immutable r195 NO-RTP Kaggle bundle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import time
from typing import Any, BinaryIO


SCHEMA = "poke_bot.alakazam_r195_new_list_no_retrain_r278_build/v1"
AUTH_SCHEMA = "poke_bot.kaggle_submission_authorization/v1"
SOURCE_BUNDLE_SHA256 = "sha256:dfa8bfccf9ee41d2205c7e30d817489391bb6295fa1ed1eff78c36fd8a8b7145"
MODEL_SHA256 = "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
DECK_SHA256 = "sha256:d834c66c5a3629dd79c8533a04fde770a22ca8590ac55c9868440121b6df5fba"
DECK_CARDS_SHA256 = "sha256:e61c0a4ffcfeb730808ac561f39c1efa9de5f80aec577d3be82f3fc790b7dab2"
LABEL = "r195 NO RTP raw resubmit, new Alakazam list, not retrained otherwise"
COMPETITION = "pokemon-tcg-ai-battle"
TURN_ORDER = "first_if_allowed"
VERIFIED_CASES = {"integer_enum", "string_enum_reversed_options", "live_engine_prompt"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def sha256_stream(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    while chunk := handle.read(4 * 1024 * 1024):
        digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def canonical_cards_digest(raw: bytes) -> tuple[int, str]:
    cards = [
        int(line.split(",", 1)[0])
        for value in raw.decode("utf-8").splitlines()
        if (line := value.strip()) and not line.startswith("#")
    ]
    packed = json.dumps(
        cards, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return len(cards), "sha256:" + hashlib.sha256(packed).hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    body = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(body, encoding="utf-8")
    os.replace(temporary, path)


def safe_member_name(name: str) -> str:
    normalized = name.removeprefix("./")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or not normalized:
        raise RuntimeError(f"unsafe archive member: {name!r}")
    return normalized


def build_derivative(source: Path, deck: Path, output: Path) -> dict[str, Any]:
    if sha256_file(source) != SOURCE_BUNDLE_SHA256:
        raise RuntimeError("source is not the immutable r195 submission 55378392 bundle")
    if sha256_file(deck) != DECK_SHA256:
        raise RuntimeError("new Alakazam deck file changed")
    deck_bytes = deck.read_bytes()
    card_count, cards_digest = canonical_cards_digest(deck_bytes)
    if card_count != 60 or cards_digest != DECK_CARDS_SHA256:
        raise RuntimeError("new Alakazam deck is not the checksum-pinned 60-card list")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.partial.{os.getpid()}")
    deck_members = 0
    source_hashes: dict[str, str] = {}
    with tarfile.open(source, "r:gz") as old, tarfile.open(temporary, "w:gz") as new:
        for member in old.getmembers():
            normalized = safe_member_name(member.name)
            if not member.isfile():
                new.addfile(member)
                continue
            stream = old.extractfile(member)
            if stream is None:
                raise RuntimeError(f"unreadable source member: {member.name}")
            if normalized == "deck.csv":
                deck_members += 1
                source_hashes[normalized] = sha256_stream(stream)
                member.size = len(deck_bytes)
                new.addfile(member, io.BytesIO(deck_bytes))
            else:
                raw = stream.read()
                source_hashes[normalized] = "sha256:" + hashlib.sha256(raw).hexdigest()
                new.addfile(member, io.BytesIO(raw))
    if deck_members != 1:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"expected one deck.csv member, found {deck_members}")
    os.replace(temporary, output)

    derived_hashes: dict[str, str] = {}
    profile: dict[str, Any] = {}
    with tarfile.open(output, "r:gz") as archive:
        for member in archive.getmembers():
            normalized = safe_member_name(member.name)
            if not member.isfile():
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError(f"unreadable derived member: {member.name}")
            raw = stream.read()
            derived_hashes[normalized] = "sha256:" + hashlib.sha256(raw).hexdigest()
            if normalized == "runtime_profile.json":
                profile = json.loads(raw)
    changed = sorted(
        name for name in source_hashes if source_hashes[name] != derived_hashes.get(name)
    )
    if set(source_hashes) != set(derived_hashes) or changed != ["deck.csv"]:
        raise RuntimeError(f"raw derivative changed unexpected archive members: {changed}")
    if derived_hashes.get("model.pt") != MODEL_SHA256:
        raise RuntimeError("derived archive model is not exact r195")
    if (
        profile.get("schema") != "poke_bot.submission_runtime_profile/v1"
        or profile.get("recursive_turn_planner") != "disabled"
        or profile.get("rtp_sidecar_packaged") is not False
    ):
        raise RuntimeError("derived archive is not the exact r195 NO-RTP runtime")
    return {
        "path": str(output),
        "sha256": sha256_file(output),
        "bytes": output.stat().st_size,
        "member_count": len(derived_hashes),
        "changed_member_contents": changed,
        "model_sha256": derived_hashes["model.pt"],
        "deck_sha256": derived_hashes["deck.csv"],
        "deck_cards_sha256": cards_digest,
        "runtime_profile": profile,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--source-attestation", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_bundle.expanduser().resolve()
    source_attestation = args.source_attestation.expanduser().resolve()
    deck = args.deck.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    authorization_path = args.authorization.expanduser().resolve()
    output = output_dir / "submission.tar.gz"

    attestation = read_object(source_attestation)
    if (
        str(attestation.get("file_sha256") or "") != SOURCE_BUNDLE_SHA256
        or str(attestation.get("turn_order_preference") or TURN_ORDER) != TURN_ORDER
        or not VERIFIED_CASES <= {str(value) for value in attestation.get("verified_cases") or []}
    ):
        raise RuntimeError("source r195 turn-order attestation is incomplete")
    bundle = build_derivative(source, deck, output)
    derived_attestation = {
        **attestation,
        "schema": "poke_bot.submission_turn_order_attestation/v1",
        "file": str(output),
        "file_sha256": bundle["sha256"],
        "turn_order_preference": TURN_ORDER,
        "go_first_if_offered": True,
        "go_second_if_offered": False,
        "derived_from_unchanged_runtime_bundle_sha256": SOURCE_BUNDLE_SHA256,
        "only_changed_member_content": "deck.csv",
    }
    attestation_path = Path(str(output) + ".go-first-verified.json")
    atomic_json(attestation_path, derived_attestation)
    if authorization_path.exists():
        raise RuntimeError("another Kaggle one-shot authorization already exists")
    nonce = "alakazam-r195-new-list-no-retrain-r278-" + datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    authorization = {
        "schema": AUTH_SCHEMA,
        "explicit_user_approval": True,
        "approval_text": "Raw resubmit r195 with the new Alakazam list; label it not retrained otherwise.",
        "owner_decision_revision": 278,
        "remaining_uses": 1,
        "nonce": nonce,
        "expires_at_epoch": time.time() + 3600.0,
        "competition": COMPETITION,
        "file_sha256": bundle["sha256"],
        "message": LABEL,
        "specialist_id": "alakazam",
        "frozen_checkpoint_checksum": MODEL_SHA256,
        "submission_file_checksum": bundle["sha256"],
        "turn_order_preference": TURN_ORDER,
    }
    atomic_json(authorization_path, authorization)
    receipt = {
        "schema": SCHEMA,
        "status": "built_and_one_shot_authorized",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "owner_decision_revision": 278,
        "source_submission_id": 55378392,
        "source_bundle": str(source),
        "source_bundle_sha256": SOURCE_BUNDLE_SHA256,
        "source_turn_order_attestation": str(source_attestation),
        "new_deck": str(deck),
        "label": LABEL,
        "competition": COMPETITION,
        "training_performed": False,
        "bundle": bundle,
        "attestation": str(attestation_path),
        "authorization": str(authorization_path),
        "authorization_nonce": nonce,
    }
    atomic_json(output_dir / "build-receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
