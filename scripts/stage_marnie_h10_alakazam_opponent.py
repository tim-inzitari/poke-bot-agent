#!/usr/bin/env python3
"""Stage the frozen H10 Alakazam refresh for Marnie practice and holdout."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from typing import Any

from poke_bot.baselines_runtime import baseline_content_digest
from poke_bot.pure_rl.model_registry import sha256


CHECKPOINT_DIGEST = (
    "sha256:02c014ad7c3318d9871a2b16b57b25adb721d5c88cacb2a3d23db3c2f3ca0d92"
)
BUNDLE_DIGEST = (
    "sha256:e596630536d5052ae172ba2a42d72023709eba8b98c17a47243d1275b33a5b75"
)
OPPONENT_ID = "specialist-alakazam-final-format-h10-02c014ad7c33"
BASELINE_DIR = "alakazam-final-format-h10-02c014ad7c33"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _canonical_digest(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _extract_bundle(bundle: Path, destination: Path) -> str:
    if not bundle.is_file() or sha256(bundle) != BUNDLE_DIGEST:
        raise RuntimeError("H10 Alakazam bundle is missing or changed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{BASELINE_DIR}.", dir=str(destination.parent))
    )
    try:
        with tarfile.open(bundle, "r:gz") as archive:
            members = archive.getmembers()
            for member in members:
                member_path = PurePosixPath(member.name)
                if (
                    member_path.is_absolute()
                    or ".." in member_path.parts
                    or member.issym()
                    or member.islnk()
                    or member.isdev()
                ):
                    raise RuntimeError("unsafe member in H10 Alakazam bundle")
            archive.extractall(temporary)
        for name in ("main.py", "model.pt", "deck.csv", "matchup_tree.json"):
            if not (temporary / name).is_file():
                raise RuntimeError(f"H10 Alakazam package missing {name}")
        if sha256(temporary / "model.pt") != CHECKPOINT_DIGEST:
            raise RuntimeError("H10 Alakazam package has the wrong checkpoint")
        content_digest = baseline_content_digest(temporary)
        if destination.exists():
            if baseline_content_digest(destination) != content_digest:
                raise RuntimeError("existing H10 Alakazam package has different bytes")
            shutil.rmtree(temporary)
        else:
            os.replace(temporary, destination)
        return content_digest
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _update_manifest(path: Path) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    agents = [
        copy.deepcopy(row)
        for row in (manifest.get("agents") or [])
        if str(row.get("id") or "") != OPPONENT_ID
    ]
    agents.append(
        {
            "id": OPPONENT_ID,
            "name": "Frozen final-format H10 Alakazam refresh",
            "group": "specialists",
            "dir": BASELINE_DIR,
            "source": "checksum-bound final-format Alakazam H10 refresh "
            + BUNDLE_DIGEST,
        }
    )
    ids = [str(row.get("id") or "") for row in agents]
    if len(ids) != len(set(ids)):
        raise RuntimeError("baseline manifest contains duplicate opponent ids")
    manifest["agents"] = agents
    notes = dict(manifest.get("field_notes") or {})
    notes["total"] = len(agents)
    manifest["field_notes"] = notes
    _atomic_json(path, manifest)


def _build_registry(
    *, base_path: Path, output_path: Path, package: Path, content_digest: str,
    completion: Path,
) -> dict[str, Any]:
    base = json.loads(base_path.read_text(encoding="utf-8"))
    completion_payload = json.loads(completion.read_text(encoding="utf-8"))
    if (
        base.get("schema") != "poke_bot.frozen_specialist_registry/v1"
        or completion_payload.get("specialist_id") != "alakazam"
        or completion_payload.get("frozen") is not True
        or completion_payload.get("registered") is not True
        or completion_payload.get("refresh_checkpoint_checksum")
        != CHECKPOINT_DIGEST
    ):
        raise RuntimeError("H10 Alakazam completion authority changed")
    previous = [
        copy.deepcopy(row)
        for row in (base.get("specialists") or [])
        if str(row.get("specialist_id") or "") == "alakazam"
    ]
    if len(previous) != 1:
        raise RuntimeError("expected exactly one historical Alakazam row")
    row = {
        "adapter_route_count": int(previous[0].get("adapter_route_count") or 18),
        "archetype_id": "alakazam",
        "archetype_label": "Frozen final-format H10 Alakazam refresh",
        "baseline_dir": BASELINE_DIR,
        "baseline_group": "specialists",
        "checkpoint_digest": CHECKPOINT_DIGEST,
        "content_digest": content_digest,
        "deck_file_checksum": sha256(package / "deck.csv"),
        "frozen": True,
        "kaggle_submission_eligible": False,
        "matchup_runtime_enabled": True,
        "matchup_runtime_inference_only": True,
        "matchup_tree_checksum": sha256(package / "matchup_tree.json"),
        "opponent_id": OPPONENT_ID,
        "public_mix_eligible": True,
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_eligible": False,
        "roster_version": 6,
        "source": (
            "frozen final-format Alakazam H10 refresh under explicit owner "
            "ceiling acceptance"
        ),
        "source_passing_checkpoint_digest": CHECKPOINT_DIGEST,
        "specialist_id": "alakazam",
        "refresh_completion_receipt": str(completion),
        "refresh_completion_receipt_sha256": sha256(completion),
        "supersedes_for_marnie_opponent_id": previous[0]["opponent_id"],
    }
    rows = [
        copy.deepcopy(item)
        for item in (base.get("specialists") or [])
        if str(item.get("specialist_id") or "") != "alakazam"
    ]
    rows.append(row)
    if len(rows) != 14 or len({item["specialist_id"] for item in rows}) != 14:
        raise RuntimeError("Marnie scoped frozen roster must contain 14 specialists")
    result = copy.deepcopy(base)
    result["version"] = 14
    result["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["owner_decision_revision"] = 108
    result["scope"] = "final-format-marnie-r104-h10"
    result["specialists"] = rows
    _atomic_json(output_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--base-registry", type=Path, required=True)
    parser.add_argument("--output-registry", type=Path, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    package = args.baseline_root / "specialists" / BASELINE_DIR
    content_digest = _extract_bundle(args.bundle, package)
    _update_manifest(args.baseline_manifest)
    registry = _build_registry(
        base_path=args.base_registry,
        output_path=args.output_registry,
        package=package,
        content_digest=content_digest,
        completion=args.completion,
    )
    receipt = {
        "schema": "poke_bot.marnie_h10_alakazam_opponent_stage/v1",
        "status": "staged_for_next_committed_iteration_boundary",
        "owner_decision_revision": 108,
        "checkpoint_digest": CHECKPOINT_DIGEST,
        "bundle_digest": BUNDLE_DIGEST,
        "opponent_id": OPPONENT_ID,
        "baseline_group": "specialists",
        "baseline_dir": BASELINE_DIR,
        "baseline_content_digest": content_digest,
        "baseline_manifest": str(args.baseline_manifest),
        "baseline_manifest_sha256": sha256(args.baseline_manifest),
        "scoped_frozen_registry": str(args.output_registry),
        "scoped_frozen_registry_sha256": sha256(args.output_registry),
        "frozen_count": len(registry["specialists"]),
        "practice_public_mix_eligible": True,
        "formal_holdout_eligible": True,
        "research_control_changed": False,
        "historical_v5_rewritten": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt["receipt_payload_digest"] = _canonical_digest(receipt)
    _atomic_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
