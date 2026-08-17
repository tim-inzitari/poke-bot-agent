#!/usr/bin/env python3
"""Build the revision-33 one-off Hops second-preferring Kaggle bundle.

This intentionally derives from the already validated historical Hops bundle
instead of repackaging its V4 checkpoint with the current V5/V6 loader. Only
the isolated turn-order resolver and its immutable profile are changed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
from typing import Any


SOURCE_BUNDLE_SHA256 = (
    "sha256:6229e3dd9840268e5cd35b18516a56dc97440cd36d3372f8bc108a0536fb9231"
)
SOURCE_MAIN_SHA256 = (
    "sha256:bc28b8315d5bf08a13b9e72175168f2f3541c6b111c73664ee4b1702ae2227c3"
)
CHECKPOINT_SHA256 = (
    "sha256:462f201f8de6c07eef07b3e8f58229360972d1d64308db9c155f211d2ce3faf1"
)
DECK_SHA256 = (
    "sha256:c582b9067dd3b70a3f1e50efc874662c90d5257af04f3660d971427a9f1263b3"
)
MATCHUP_TREE_SHA256 = (
    "sha256:0bbbd1075c0c2058e07be6723c2f2bb7902193ce3132613e70d354c132f75c3d"
)
TURN_ORDER_PREFERENCE = "second_if_allowed"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _safe_extract(source: Path, destination: Path) -> None:
    with tarfile.open(source, "r:gz") as archive:
        for member in archive.getmembers():
            relative = Path(member.name.removeprefix("./"))
            if (
                member.issym()
                or member.islnk()
                or relative.is_absolute()
                or ".." in relative.parts
            ):
                raise RuntimeError(f"unsafe source-bundle member: {member.name}")
        archive.extractall(destination)


def _patch_main(source: str) -> str:
    old_doc = (
        "  - Deterministically choose first before importing cg or loading "
        "the model."
    )
    new_doc = (
        "  - Deterministically honor the packaged turn-order profile before "
        "importing cg or loading the model."
    )
    old_import = "import os\nimport random"
    new_import = "import json\nimport os\nimport random"
    old_function = '''def _go_first_choice(obs_dict: dict) -> list[int] | None:
    """Resolve IsFirst directly from the wire enum without runtime imports."""

    selection = obs_dict.get("select") if isinstance(obs_dict, dict) else None
    if not isinstance(selection, dict):
        return None
    context = selection.get("context")
    normalized_context = "".join(
        character for character in str(context).lower() if character.isalnum()
    )
    if context != 41 and normalized_context != "isfirst":
        return None
    options = list(selection.get("option") or [])
    yes = [
        index
        for index, option in enumerate(options)
        if isinstance(option, dict)
        and (
            option.get("type") == 1
            or str(option.get("type") or "").strip().lower() == "yes"
        )
    ]
    return yes if len(yes) == 1 else []
'''
    new_function = '''def _turn_order_preference() -> str:
    """Read the immutable one-off profile without importing the runtime."""

    profile = json.loads(
        (_agent_dir() / "turn_order_profile.json").read_text()
    )
    preference = str(profile.get("turn_order_preference") or "")
    if preference != "second_if_allowed":
        raise RuntimeError("invalid one-off Hops turn-order preference")
    return preference


def _go_first_choice(obs_dict: dict) -> list[int] | None:
    """Resolve IsFirst directly from the wire enum without runtime imports."""

    selection = obs_dict.get("select") if isinstance(obs_dict, dict) else None
    if not isinstance(selection, dict):
        return None
    context = selection.get("context")
    normalized_context = "".join(
        character for character in str(context).lower() if character.isalnum()
    )
    if context != 41 and normalized_context != "isfirst":
        return None
    options = list(selection.get("option") or [])
    desired = "no" if _turn_order_preference() == "second_if_allowed" else "yes"
    desired_integer = 2 if desired == "no" else 1
    matches = [
        index
        for index, option in enumerate(options)
        if isinstance(option, dict)
        and (
            option.get("type") == desired_integer
            or str(option.get("type") or "").strip().lower() == desired
        )
    ]
    return matches if len(matches) == 1 else []
'''
    replacements = (
        (old_doc, new_doc),
        (old_import, new_import),
        (old_function, new_function),
    )
    patched = source
    for old, new in replacements:
        if patched.count(old) != 1:
            raise RuntimeError("historical Hops entry point changed unexpectedly")
        patched = patched.replace(old, new, 1)
    return patched


def build(source_bundle: Path, output_dir: Path) -> dict[str, Any]:
    source_bundle = source_bundle.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if _sha256(source_bundle) != SOURCE_BUNDLE_SHA256:
        raise RuntimeError("historical Hops source bundle checksum changed")
    if output_dir.exists():
        raise FileExistsError(f"one-off output already exists: {output_dir}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        stage = temporary / "stage"
        stage.mkdir()
        _safe_extract(source_bundle, stage)
        checks = {
            "main": _sha256(stage / "main.py") == SOURCE_MAIN_SHA256,
            "checkpoint": _sha256(stage / "model.pt") == CHECKPOINT_SHA256,
            "deck": _sha256(stage / "deck.csv") == DECK_SHA256,
            "matchup_tree": (
                _sha256(stage / "matchup_tree.json") == MATCHUP_TREE_SHA256
            ),
        }
        if not all(checks.values()):
            failed = sorted(key for key, value in checks.items() if not value)
            raise RuntimeError(
                "historical Hops source identity failed: " + ",".join(failed)
            )

        main_path = stage / "main.py"
        main_path.write_text(
            _patch_main(main_path.read_text(encoding="utf-8")),
            encoding="utf-8",
        )
        _atomic_json(
            stage / "turn_order_profile.json",
            {
                "schema": "poke_bot.submission_turn_order_profile/v1",
                "specialist_id": "hops-trevenant",
                "owner_decision_revision": 33,
                "one_off": True,
                "turn_order_preference": TURN_ORDER_PREFERENCE,
            },
        )

        bundle = temporary / "submission.tar.gz"
        with tarfile.open(bundle, "w:gz") as archive:
            for path in sorted(stage.rglob("*")):
                archive.add(path, arcname=str(path.relative_to(stage)))
        result = {
            "schema": "poke_bot.hops_second_one_off_build/v1",
            "owner_decision_revision": 33,
            "one_off": True,
            "specialist_id": "hops-trevenant",
            "turn_order_preference": TURN_ORDER_PREFERENCE,
            "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source_bundle": str(source_bundle),
            "source_bundle_sha256": SOURCE_BUNDLE_SHA256,
            "bundle": str((output_dir / bundle.name)),
            "bundle_sha256": _sha256(bundle),
            "checkpoint_sha256": _sha256(stage / "model.pt"),
            "deck_sha256": _sha256(stage / "deck.csv"),
            "matchup_tree_sha256": _sha256(stage / "matchup_tree.json"),
            "source_main_sha256": SOURCE_MAIN_SHA256,
            "derived_main_sha256": _sha256(stage / "main.py"),
            "source_identity_checks": checks,
            "attestation_pending": True,
        }
        _atomic_json(temporary / "build-receipt.json", result)
        os.replace(temporary, output_dir)
        return result
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source_bundle, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
