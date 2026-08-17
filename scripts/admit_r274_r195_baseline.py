#!/usr/bin/env python3
"""Create a derivative baseline root admitting the exact frozen r195 package."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


BASELINE_ID = "alakazam-r195-no-rtp-submission-55378392"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--derivative-root", type=Path, required=True)
    parser.add_argument("--r195-package", type=Path, required=True)
    parser.add_argument("--expected-content-sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    parent = args.parent_root.resolve()
    derivative = args.derivative_root.resolve()
    package = args.r195_package.resolve()
    receipt = args.receipt.resolve()
    if derivative.exists() or receipt.exists():
        raise FileExistsError("derivative baseline root or receipt already exists")
    shutil.copytree(parent, derivative, copy_function=os.link, symlinks=True)
    derivative.chmod(derivative.stat().st_mode | 0o200)
    (derivative / "specialists").chmod(
        (derivative / "specialists").stat().st_mode | 0o200
    )
    manifest_path = derivative / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    agents = list(manifest.get("agents") or ())
    if any(str(row.get("id") or "") == BASELINE_ID for row in agents):
        raise RuntimeError("parent manifest unexpectedly already contains r195")
    agents.append(
        {
            "dir": BASELINE_ID,
            "group": "specialists",
            "id": BASELINE_ID,
            "external_path": str(package),
            "external_content_sha256": args.expected_content_sha256,
        }
    )
    manifest["agents"] = agents
    manifest_path.unlink()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    leaf = derivative / "specialists" / BASELINE_ID
    if leaf.exists() or leaf.is_symlink():
        raise RuntimeError("derivative r195 leaf already exists")
    leaf.symlink_to(package, target_is_directory=True)
    install_receipt = package / "INSTALL_RECEIPT.json"
    payload = {
        "schema": "poke_bot.r274_r195_baseline_manifest_admission/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_id": BASELINE_ID,
        "parent_root": str(parent),
        "parent_manifest_sha256": sha256(parent / "manifest.json"),
        "derivative_root": str(derivative),
        "derivative_manifest_sha256": sha256(manifest_path),
        "package_path": str(package),
        "package_leaf": str(leaf),
        "package_leaf_is_symlink": leaf.is_symlink(),
        "expected_content_sha256": args.expected_content_sha256,
        "install_receipt_path": str(install_receipt),
        "install_receipt_sha256": sha256(install_receipt) if install_receipt.is_file() else None,
        "agent_count_before": len(agents) - 1,
        "agent_count_after": len(agents),
        "passed": True,
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
