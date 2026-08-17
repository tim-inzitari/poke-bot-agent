#!/usr/bin/env python3
"""Create r274 owner-bound aggregates for the immutable imported r260 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


LEGACY_OWNER = "sha256:803c39f40d5442cdcd7e36596047271659534d5a21f0aef2667d3f714ffd2150"
LEGACY_SIDECAR = "sha256:93392171eefad6c5f56a16c1701d7374b252c464952af318910d6fe36fef10b2"
LEGACY_DATASET = "sha256:683ec6bac83051c110d7e1090259a85e68616f402a0247408bc9b2324fc78c7f"


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def load(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"binding must be a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"binding is not an object: {path}")
    return value


def semantic(value: dict[str, object]) -> str:
    unsigned = dict(value)
    unsigned.pop("binding_sha256", None)
    return sha_bytes(canonical(unsigned))


def create_only(path: Path, payload: dict[str, object]) -> None:
    body = canonical(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != body:
            raise RuntimeError(f"create-only binding changed: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    path.chmod(0o444)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-sidecar", required=True, type=Path)
    parser.add_argument("--legacy-dataset", required=True, type=Path)
    parser.add_argument("--owner-contract", required=True, type=Path)
    parser.add_argument("--output-sidecar", required=True, type=Path)
    parser.add_argument("--output-dataset", required=True, type=Path)
    parser.add_argument("--adoption-receipt", required=True, type=Path)
    args = parser.parse_args()

    owner_path = args.owner_contract.expanduser().resolve()
    owner_sha = sha_file(owner_path)
    owner = load(owner_path)
    adoption = dict(
        dict(owner.get("own_deck_head_structure_import") or {}).get(
            "local_post_transfer_completion"
        )
        or {}
    )
    if (
        adoption.get("imported_sidecar_binding_semantic_sha256") != LEGACY_SIDECAR
        or adoption.get("imported_inzi_dataset_binding_semantic_sha256") != LEGACY_DATASET
        or adoption.get("imported_binding_owner_contract_sha256") != LEGACY_OWNER
        or adoption.get("r274_create_only_rebinding_to_current_owner_contract_required") is not True
    ):
        raise RuntimeError("current owner contract does not authorize exact imported binding adoption")

    legacy_sidecar_path = args.legacy_sidecar.expanduser().resolve()
    legacy_dataset_path = args.legacy_dataset.expanduser().resolve()
    legacy_sidecar = load(legacy_sidecar_path)
    legacy_dataset = load(legacy_dataset_path)
    if (
        semantic(legacy_sidecar) != LEGACY_SIDECAR
        or semantic(legacy_dataset) != LEGACY_DATASET
        or legacy_sidecar.get("owner_contract_sha256") != LEGACY_OWNER
        or legacy_dataset.get("owner_contract_sha256") != LEGACY_OWNER
    ):
        raise RuntimeError("legacy imported bindings do not match the exact authorized identities")

    sidecar = dict(legacy_sidecar)
    sidecar["owner_contract_sha256"] = owner_sha
    sidecar["binding_sha256"] = semantic(sidecar)
    output_sidecar = args.output_sidecar.expanduser().absolute()
    create_only(output_sidecar, sidecar)
    sidecar_file = {
        "path": str(output_sidecar.resolve()),
        "sha256": sha_file(output_sidecar),
        "size_bytes": output_sidecar.stat().st_size,
    }

    dataset = dict(legacy_dataset)
    dataset["owner_contract_sha256"] = owner_sha
    dataset["sidecar_binding_sha256"] = sidecar["binding_sha256"]
    dataset["sidecar_binding_file_sha256"] = sidecar_file["sha256"]
    dataset["sidecar_binding"] = sidecar_file
    dataset["binding_sha256"] = semantic(dataset)
    output_dataset = args.output_dataset.expanduser().absolute()
    create_only(output_dataset, dataset)

    receipt = {
        "schema": "poke_bot.r274_imported_inzi_dataset_adoption/v1",
        "status": "passed",
        "owner_contract": {"path": str(owner_path), "sha256": owner_sha},
        "legacy": {
            "owner_contract_sha256": LEGACY_OWNER,
            "sidecar_binding_semantic_sha256": LEGACY_SIDECAR,
            "dataset_binding_semantic_sha256": LEGACY_DATASET,
            "sidecar_file_sha256": sha_file(legacy_sidecar_path),
            "dataset_file_sha256": sha_file(legacy_dataset_path),
        },
        "r274": {
            "sidecar_binding": {"path": str(output_sidecar.resolve()), "sha256": sha_file(output_sidecar), "size_bytes": output_sidecar.stat().st_size, "binding_sha256": sidecar["binding_sha256"]},
            "dataset_binding": {"path": str(output_dataset.resolve()), "sha256": sha_file(output_dataset), "size_bytes": output_dataset.stat().st_size, "binding_sha256": dataset["binding_sha256"]},
        },
        "large_payload_retransferred": False,
        "underlying_daily_and_joined_dataset_bytes_changed": False,
    }
    receipt["receipt_sha256"] = sha_bytes(canonical(receipt))
    create_only(args.adoption_receipt.expanduser().absolute(), receipt)
    print(json.dumps(receipt["r274"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
