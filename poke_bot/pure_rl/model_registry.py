"""Immutable model artifacts that rolling RL retention must never prune."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REGISTRY_MARKER = "PROTECTED_MODEL_REGISTRY.json"
FAMILY_MARKER = "PROTECTED_DO_NOT_PRUNE.json"
_SAFE_FAMILY = re.compile(r"^[a-z0-9][a-z0-9_-]{1,79}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_dir(path.parent)


def _atomic_copy(source: Path, target: Path, expected_digest: str) -> None:
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with source.open("rb") as src, temporary.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    actual = sha256(temporary)
    if actual != expected_digest:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"frozen model copy digest mismatch: expected={expected_digest} actual={actual}"
        )
    os.replace(temporary, target)
    _fsync_dir(target.parent)


def _exact_heldout_evidence(
    evidence: dict[str, Any], expected_digest: str
) -> dict[str, Any]:
    audit = evidence.get("audit")
    if not isinstance(audit, dict):
        raise ValueError("Deck Agnostic Core requires an exact heldout audit")
    games = int(evidence.get("games", 0))
    required = (
        audit.get("passed") is True,
        audit.get("exact_distribution") is True,
        audit.get("exact_weights") is True,
        audit.get("greedy_required") is True,
        int(audit.get("valid_games", 0)) == games == 1000,
        str(audit.get("checkpoint_digest") or "") == expected_digest,
        str(evidence.get("checkpoint_digest") or "") == expected_digest,
    )
    if not all(required):
        raise ValueError("Deck Agnostic Core heldout evidence is not exact/complete")
    return json.loads(json.dumps(evidence))


def freeze_model(
    *,
    registry_root: Path,
    family: str,
    display_name: str,
    checkpoint: Path,
    expected_digest: str,
    provenance: dict[str, Any],
    evidence: dict[str, Any] | None = None,
    require_exact_heldout: bool = False,
    harden_permissions: bool = True,
) -> dict[str, Any]:
    """Copy one model into a fail-closed, digest-addressed protected family.

    A family is write-once. Repeating the same operation is idempotent; trying
    to replace it with different bytes raises instead of silently changing the
    semantic identity future archetypes depend on.
    """
    slug = str(family).strip().lower()
    if not _SAFE_FAMILY.fullmatch(slug):
        raise ValueError(f"unsafe model family name: {family!r}")
    source = Path(checkpoint).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = str(expected_digest)
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError(f"invalid expected checkpoint digest: {digest!r}")
    actual = sha256(source)
    if actual != digest:
        raise ValueError(
            f"source checkpoint identity mismatch: expected={digest} actual={actual}"
        )
    heldout = dict(evidence or {})
    if require_exact_heldout:
        heldout = _exact_heldout_evidence(heldout, digest)

    root = Path(registry_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    registry_marker = root / REGISTRY_MARKER
    marker_payload = {
        "schema": "poke_bot.protected_model_registry/v1",
        "automatic_pruning_allowed": False,
        "purpose": "write-once reusable model artifacts",
    }
    if registry_marker.exists():
        if json.loads(registry_marker.read_text()) != marker_payload:
            raise RuntimeError("protected model registry marker changed")
    else:
        _atomic_json(registry_marker, marker_payload)

    family_dir = root / slug
    stable_manifest = family_dir / "manifest.json"
    stable_model = family_dir / "model.pt"
    if stable_manifest.is_file() or stable_model.is_file():
        manifest = verify_frozen_model(family_dir)
        if manifest.get("checkpoint_digest") != digest:
            raise RuntimeError(
                f"protected family {slug} is already frozen to another digest"
            )
        return manifest

    family_dir.mkdir(parents=True, exist_ok=False)
    versions = family_dir / "versions"
    versions.mkdir()
    version_dir = versions / digest.removeprefix("sha256:")
    version_dir.mkdir()
    version_model = version_dir / "model.pt"
    created = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema": "poke_bot.frozen_model/v1",
        "display_name": str(display_name),
        "family": slug,
        "immutable": True,
        "automatic_pruning_allowed": False,
        "checkpoint_digest": digest,
        "checkpoint_bytes": int(source.stat().st_size),
        "model_path": str(stable_model),
        "version_path": str(version_model),
        "source_checkpoint": str(source),
        "created_at_utc": created,
        "provenance": json.loads(json.dumps(provenance)),
        "evidence": heldout,
    }
    protection = {
        "schema": "poke_bot.protected_model_family/v1",
        "family": slug,
        "checkpoint_digest": digest,
        "automatic_pruning_allowed": False,
        "manual_removal_requires_explicit_model_registry_override": True,
    }
    try:
        _atomic_copy(source, version_model, digest)
        _atomic_json(version_dir / "manifest.json", manifest)
        _atomic_copy(version_model, stable_model, digest)
        _atomic_json(stable_manifest, manifest)
        _atomic_json(family_dir / FAMILY_MARKER, protection)
        _fsync_dir(version_dir)
        _fsync_dir(family_dir)
        verified = verify_frozen_model(family_dir)
        if harden_permissions:
            for path in (
                version_model,
                version_dir / "manifest.json",
                stable_model,
                stable_manifest,
                family_dir / FAMILY_MARKER,
            ):
                path.chmod(0o444)
            version_dir.chmod(0o555)
            versions.chmod(0o555)
            family_dir.chmod(0o555)
        return verified
    except BaseException:
        # Never remove a published family. Cleanup is allowed only before the
        # stable manifest exists, where no consumer can mistake it for ready.
        if not stable_manifest.exists():
            shutil.rmtree(family_dir, ignore_errors=True)
        raise


def verify_frozen_model(family_dir: Path) -> dict[str, Any]:
    family_dir = Path(family_dir).expanduser().resolve()
    marker_path = family_dir / FAMILY_MARKER
    manifest_path = family_dir / "manifest.json"
    model_path = family_dir / "model.pt"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"frozen model metadata is missing/corrupt: {family_dir}") from exc
    expected = str(manifest.get("checkpoint_digest") or "")
    if (
        manifest.get("immutable") is not True
        or manifest.get("automatic_pruning_allowed") is not False
        or marker.get("automatic_pruning_allowed") is not False
        or marker.get("checkpoint_digest") != expected
        or not model_path.is_file()
        or sha256(model_path) != expected
    ):
        raise RuntimeError(f"frozen model verification failed: {family_dir}")
    version_path = Path(str(manifest.get("version_path") or ""))
    if not version_path.is_file() or sha256(version_path) != expected:
        raise RuntimeError(f"frozen version verification failed: {version_path}")
    return manifest


def is_protected_model_path(path: Path) -> bool:
    current = Path(path).expanduser().resolve()
    for parent in (current, *current.parents):
        if (parent / REGISTRY_MARKER).is_file() or (parent / FAMILY_MARKER).is_file():
            return True
    return False
