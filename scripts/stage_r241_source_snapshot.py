#!/usr/bin/env python3
"""Materialize one immutable, receipt-bound r241 source snapshot.

The r241 Alakazam launcher must execute from a content-addressed code root,
while all run artifacts remain under the separately managed external outputs
root.  This local staging tool is the only supported way to create that code
root: it derives the launcher's execution closure, recursively includes local
Python imports, records a relative-path-only checksum inventory, seals the
tree read-only, and writes a create-only host staging receipt.

It deliberately does *not* edit the runtime registry, transfer bytes to a
remote host, start a service, load a checkpoint, collect a game, or submit to
Kaggle.  Run it independently on each host from checksum-identical source,
compare the emitted manifest identities, then bind the resulting host paths
through the normal receipt-backed activation transaction.  Its
``--published-root`` mode is intentionally narrower: it can only execute from
the already sealed snapshot itself, verifies that root without copying or
changing it, and emits the host-local create-only staging receipt.  That lets a
remote host attest a transferred snapshot without executing a mutable checkout.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import sys
import tempfile
from collections import deque
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence


# A published snapshot must never acquire an unbound ``__pycache__`` merely
# because this verifier imports its launcher.  Keep this process bytecode-free
# even if a host omits the normal ``PYTHONDONTWRITEBYTECODE=1`` environment.
sys.dont_write_bytecode = True


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE_SNAPSHOT_SCHEMA = "poke_bot.alakazam_new_list_direct_r241_source_snapshot/v1"
STAGING_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_source_snapshot_staging/v1"
)
MANIFEST_FILENAME = "r241-source-snapshot-manifest.json"
CANDIDATE_ID = "alakazam-new-list-direct-policy-r241"
REVISION = 241
_SHA256_PREFIX = "sha256:"


class R241SourceSnapshotError(RuntimeError):
    """The proposed source snapshot cannot safely be published."""


def _sha256_bytes(payload: bytes) -> str:
    return _SHA256_PREFIX + hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return _SHA256_PREFIX + digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise R241SourceSnapshotError("source-snapshot payload is not canonical JSON") from exc


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise R241SourceSnapshotError(f"{label} must be a regular non-symlink file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R241SourceSnapshotError(f"{label} is not readable JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise R241SourceSnapshotError(f"{label} must contain a JSON object: {path}")
    return payload


def _regular_directory(path: Path | str, *, label: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_dir():
        raise R241SourceSnapshotError(f"{label} must be a real non-symlink directory: {raw}")
    return raw.resolve()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _safe_relative(path: str, *, forbidden_components: frozenset[str]) -> Path:
    rendered = str(path or "").strip()
    candidate = Path(rendered)
    if (
        not rendered
        or candidate.is_absolute()
        or "." in candidate.parts
        or ".." in candidate.parts
    ):
        raise R241SourceSnapshotError(f"unsafe source-snapshot relative path: {rendered!r}")
    for component in candidate.parts:
        if component in forbidden_components:
            raise R241SourceSnapshotError(
                f"source-snapshot path uses forbidden component: {rendered}"
            )
        if component.startswith(".env") or component.endswith(".env"):
            raise R241SourceSnapshotError(
                f"source-snapshot path uses an unbound environment component: {rendered}"
            )
    return candidate


def _source_member(
    source_root: Path,
    relative: str,
    *,
    forbidden_components: frozenset[str],
) -> Path:
    parts = _safe_relative(relative, forbidden_components=forbidden_components).parts
    current = source_root
    for index, part in enumerate(parts):
        current = current / part
        if current.is_symlink():
            raise R241SourceSnapshotError(
                f"source-snapshot member is symlinked: {relative}"
            )
        if index < len(parts) - 1 and not current.is_dir():
            raise R241SourceSnapshotError(
                f"source-snapshot member parent is not a directory: {relative}"
            )
    if not current.is_file() or current.is_symlink():
        raise R241SourceSnapshotError(
            f"source-snapshot member is not a regular file: {relative}"
        )
    resolved = current.resolve()
    if source_root not in resolved.parents:
        raise R241SourceSnapshotError(
            f"source-snapshot member escapes source root: {relative}"
        )
    return resolved


def _load_launcher() -> ModuleType:
    """Load the local r241 launcher so staging shares its closure contract."""

    launcher_path = ROOT / "scripts/launch_alakazam_new_list_direct_r241.py"
    if launcher_path.is_symlink() or not launcher_path.is_file():
        raise R241SourceSnapshotError("r241 launcher is unavailable for snapshot staging")
    spec = importlib.util.spec_from_file_location(
        "_r241_source_snapshot_launcher", launcher_path
    )
    if spec is None or spec.loader is None:
        raise R241SourceSnapshotError("cannot load r241 launcher closure contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _module_name(relative: str) -> tuple[str, bool] | None:
    path = Path(relative)
    if path.suffix != ".py":
        return None
    parts = list(path.with_suffix("").parts)
    if not parts:
        return None
    if parts[-1] == "__init__":
        if len(parts) == 1:
            return None
        return ".".join(parts[:-1]), True
    return ".".join(parts), False


def _module_relative_path(source_root: Path, name: str) -> str | None:
    if not name:
        return None
    leaf = source_root.joinpath(*name.split("."))
    module_file = leaf.with_suffix(".py")
    package_init = leaf / "__init__.py"
    if module_file.is_file() and not module_file.is_symlink():
        return module_file.relative_to(source_root).as_posix()
    if package_init.is_file() and not package_init.is_symlink():
        return package_init.relative_to(source_root).as_posix()
    return None


def _package_initializers(source_root: Path, name: str) -> Iterable[str]:
    parts = name.split(".")
    for index in range(1, len(parts)):
        candidate = source_root.joinpath(*parts[:index], "__init__.py")
        if candidate.is_file() and not candidate.is_symlink():
            yield candidate.relative_to(source_root).as_posix()


def _from_target(
    *, current_name: str, current_is_package: bool, level: int, module: str | None
) -> str | None:
    if level == 0:
        return module
    package_name = current_name if current_is_package else current_name.rpartition(".")[0]
    package_parts = [part for part in package_name.split(".") if part]
    climb = level - 1
    if climb > len(package_parts):
        return None
    anchor = package_parts[: len(package_parts) - climb]
    if module:
        anchor.extend(module.split("."))
    return ".".join(anchor) or None


def _static_python_import_closure(
    source_root: Path,
    *,
    initial: set[str],
    forbidden_components: frozenset[str],
) -> set[str]:
    """Add local AST-resolved Python imports without copying a checkout wholesale.

    The execution closure intentionally remains source-only: baseline packages,
    native runtimes, output trees, caches, and environment overlays are all
    receipt-bound data mounts, not hidden code fallbacks.
    """

    resolved: set[str] = set(initial)
    queued: deque[str] = deque(sorted(relative for relative in initial if relative.endswith(".py")))
    parsed: set[str] = set()

    def include_module(name: str | None) -> None:
        if not name:
            return
        relative = _module_relative_path(source_root, name)
        if relative is None:
            return
        _source_member(
            source_root,
            relative,
            forbidden_components=forbidden_components,
        )
        if relative not in resolved:
            resolved.add(relative)
            queued.append(relative)
        for initializer in _package_initializers(source_root, name):
            _source_member(
                source_root,
                initializer,
                forbidden_components=forbidden_components,
            )
            if initializer not in resolved:
                resolved.add(initializer)
                queued.append(initializer)

    while queued:
        relative = queued.popleft()
        if relative in parsed or not relative.endswith(".py"):
            continue
        parsed.add(relative)
        source = _source_member(
            source_root,
            relative,
            forbidden_components=forbidden_components,
        )
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise R241SourceSnapshotError(
                f"cannot statically inspect source-snapshot member: {relative}"
            ) from exc
        current = _module_name(relative)
        current_name = current[0] if current else ""
        current_is_package = current[1] if current else False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    include_module(alias.name)
            elif isinstance(node, ast.ImportFrom):
                target = _from_target(
                    current_name=current_name,
                    current_is_package=current_is_package,
                    level=int(node.level or 0),
                    module=node.module,
                )
                include_module(target)
                if target:
                    for alias in node.names:
                        if alias.name != "*":
                            include_module(f"{target}.{alias.name}")
    return resolved


def _inventory(
    source_root: Path,
    *,
    closure: Iterable[str],
    forbidden_components: frozenset[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for relative in sorted(set(closure)):
        source = _source_member(
            source_root,
            relative,
            forbidden_components=forbidden_components,
        )
        rows.append(
            {
                "path": relative,
                "sha256": _sha256_file(source),
                "size_bytes": source.stat().st_size,
            }
        )
    if not rows:
        raise R241SourceSnapshotError("r241 source snapshot closure is empty")
    return rows


def _manifest_payload(
    *, launcher: ModuleType, owner_contract_sha256: str, rows: list[dict[str, object]]
) -> dict[str, object]:
    return {
        "schema": SOURCE_SNAPSHOT_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "owner_contract_sha256": owner_contract_sha256,
        "source_tree_sha256": launcher._source_tree_digest(rows),
        "external_outputs_required": True,
        "baseline_payloads_separate_and_receipted": True,
        "authenticated": True,
        "status": "authenticated_immutable_source_snapshot",
        "files": rows,
    }


def _seal_tree(
    root: Path, *, executable_paths: set[str], seal_root: bool = True
) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink():
            raise R241SourceSnapshotError(
                f"publisher refused to seal a symlinked snapshot member: {path}"
            )
        if path.is_dir():
            path.chmod(0o555)
        elif path.is_file():
            relative = path.relative_to(root).as_posix()
            path.chmod(0o555 if relative in executable_paths else 0o444)
        else:
            raise R241SourceSnapshotError(
                f"publisher found a non-regular snapshot member: {path}"
            )
    if seal_root:
        root.chmod(0o555)


def _validate_published_snapshot(
    *,
    root: Path,
    outputs_root: Path,
    manifest_bytes: bytes,
    manifest_payload: Mapping[str, object],
    required_closure: frozenset[str],
    forbidden_components: frozenset[str],
) -> dict[str, object]:
    root = _regular_directory(root, label="published r241 source snapshot root")
    outputs = _regular_directory(outputs_root, label="external r241 outputs root")
    if root.stat().st_mode & 0o222:
        raise R241SourceSnapshotError("published r241 source snapshot root is writable")
    if root == outputs or _is_within(root, outputs) or _is_within(outputs, root):
        raise R241SourceSnapshotError(
            "published r241 source snapshot and external outputs roots are not disjoint"
        )
    manifest = root / MANIFEST_FILENAME
    if manifest.is_symlink() or not manifest.is_file() or manifest.read_bytes() != manifest_bytes:
        raise R241SourceSnapshotError("published r241 source snapshot manifest drifted")
    if manifest.stat().st_mode & 0o222:
        raise R241SourceSnapshotError("published r241 source snapshot manifest is writable")
    expected_suffix = _sha256_bytes(manifest_bytes).removeprefix(_SHA256_PREFIX)[:16]
    if root.name != f"alakazam-new-list-direct-r241-src-{expected_suffix}":
        raise R241SourceSnapshotError("published r241 source root is not manifest-content-addressed")
    observed_manifest = _load_json(manifest, label="published r241 source snapshot manifest")
    if observed_manifest != dict(manifest_payload):
        raise R241SourceSnapshotError("published r241 source snapshot manifest payload drifted")
    rows = list(observed_manifest.get("files") or [])
    expected_paths = {str(row["path"]) for row in rows if isinstance(row, Mapping)}
    if len(expected_paths) != len(rows) or not required_closure.issubset(expected_paths):
        raise R241SourceSnapshotError("published r241 snapshot omits the required execution closure")
    observed_paths: set[str] = set()
    for directory_text, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        directory = Path(directory_text)
        for name in list(directory_names):
            child = directory / name
            relative = child.relative_to(root).as_posix()
            if child.is_symlink() or child.stat().st_mode & 0o222:
                raise R241SourceSnapshotError(
                    f"published r241 source snapshot has unsafe directory: {relative}"
                )
            _safe_relative(relative, forbidden_components=forbidden_components)
        for name in file_names:
            child = directory / name
            relative = child.relative_to(root).as_posix()
            if relative == MANIFEST_FILENAME:
                continue
            if child.is_symlink() or not child.is_file() or child.stat().st_mode & 0o222:
                raise R241SourceSnapshotError(
                    f"published r241 source snapshot has unsafe file: {relative}"
                )
            _safe_relative(relative, forbidden_components=forbidden_components)
            observed_paths.add(relative)
    if observed_paths != expected_paths:
        raise R241SourceSnapshotError(
            "published r241 source snapshot has unbound or missing source files"
        )
    canonical_rows: list[dict[str, object]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise R241SourceSnapshotError("published r241 source inventory row is malformed")
        relative = str(raw.get("path") or "")
        member = root / _safe_relative(relative, forbidden_components=forbidden_components)
        if member.is_symlink() or not member.is_file():
            raise R241SourceSnapshotError(
                f"published r241 source inventory member is missing: {relative}"
            )
        size = raw.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise R241SourceSnapshotError(
                f"published r241 source inventory size is invalid: {relative}"
            )
        expected_sha = str(raw.get("sha256") or "")
        if not expected_sha.startswith(_SHA256_PREFIX) or len(expected_sha) != 71:
            raise R241SourceSnapshotError(
                f"published r241 source inventory digest is invalid: {relative}"
            )
        if member.stat().st_size != size or _sha256_file(member) != expected_sha:
            raise R241SourceSnapshotError(
                f"published r241 source inventory identity drifted: {relative}"
            )
        canonical_rows.append(
            {"path": relative, "sha256": expected_sha, "size_bytes": size}
        )
    if _canonical_source_tree_digest(canonical_rows) != observed_manifest.get(
        "source_tree_sha256"
    ):
        raise R241SourceSnapshotError("published r241 source inventory tree digest drifted")
    return {
        "root": str(root),
        "manifest": str(manifest),
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "source_tree_sha256": str(observed_manifest["source_tree_sha256"]),
        "file_inventory_sha256": _sha256_bytes(_canonical_json(canonical_rows)),
        "outputs_root": str(outputs),
    }


def _canonical_source_tree_digest(rows: Sequence[Mapping[str, object]]) -> str:
    canonical = [
        {
            "path": str(row["path"]),
            "sha256": str(row["sha256"]),
            "size_bytes": int(row["size_bytes"]),
        }
        for row in sorted(rows, key=lambda item: str(item["path"]))
    ]
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _create_only_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.is_symlink():
        raise R241SourceSnapshotError(f"staging receipt path is a symlink: {path}")
    parent = _regular_directory(path.parent, label="staging receipt parent")
    target = parent / path.name
    encoded = _canonical_json(payload)
    if target.exists():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != encoded:
            raise R241SourceSnapshotError(
                f"staging receipt already exists with different bytes: {target}"
            )
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.is_symlink() or not target.is_file() or target.read_bytes() != encoded:
                raise R241SourceSnapshotError(
                    f"staging receipt already exists with different bytes: {target}"
                )
        finally:
            temporary.unlink(missing_ok=True)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _stage(
    *,
    launcher: ModuleType,
    output_base: Path,
    outputs_root: Path,
    receipt_output: Path,
    host: str,
) -> dict[str, object]:
    registry_path, registry = launcher.load_registry()
    launcher.validate_static_registry(registry)
    # The snapshot must carry only immutable intent.  Readiness and service
    # authority are supplied later through the separate create-only overlay;
    # embedding a ready registry here would form a source/receipt cycle.
    launcher._activation_overlay_base_projection_is_pending(registry)
    if registry_path.resolve().parents[1] != ROOT:
        raise R241SourceSnapshotError("r241 snapshot staging must use the local canonical registry")
    owner_contract_sha256 = str(registry["owner_contract"]["sha256"])
    required = frozenset(launcher._REQUIRED_SOURCE_SNAPSHOT_FILES)
    forbidden = frozenset(launcher._FORBIDDEN_SNAPSHOT_COMPONENTS)
    for relative in required:
        _source_member(ROOT, relative, forbidden_components=forbidden)
    closure = _static_python_import_closure(
        ROOT,
        initial=set(required),
        forbidden_components=forbidden,
    )
    rows = _inventory(ROOT, closure=closure, forbidden_components=forbidden)
    # Re-evaluate the import graph after inventory.  An edit that adds a local
    # import between discovery and publication must not leave a seemingly
    # sealed tree with an omitted executable dependency.
    final_closure = _static_python_import_closure(
        ROOT,
        initial=set(required),
        forbidden_components=forbidden,
    )
    if final_closure != closure:
        raise R241SourceSnapshotError(
            "r241 source import closure changed during inventory; retry from a stable source tree"
        )
    manifest_payload = _manifest_payload(
        launcher=launcher,
        owner_contract_sha256=owner_contract_sha256,
        rows=rows,
    )
    manifest_bytes = _canonical_json(manifest_payload)
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    output_base = _regular_directory(output_base, label="r241 source snapshot output base")
    outputs_root = _regular_directory(outputs_root, label="r241 external outputs root")
    receipt_parent = _regular_directory(
        receipt_output.expanduser().parent, label="r241 source snapshot staging receipt parent"
    )
    receipt_output = receipt_parent / receipt_output.name
    if not _is_within(receipt_output, outputs_root):
        raise R241SourceSnapshotError(
            "r241 source snapshot staging receipt must remain under external outputs"
        )
    snapshot_root = output_base / (
        "alakazam-new-list-direct-r241-src-"
        + manifest_sha256.removeprefix(_SHA256_PREFIX)[:16]
    )
    if (
        _is_within(snapshot_root, ROOT)
        or _is_within(snapshot_root, outputs_root)
        or _is_within(outputs_root, snapshot_root)
    ):
        raise R241SourceSnapshotError(
            "r241 snapshot output base must be disjoint from mutable source and outputs"
        )
    executable_paths = {
        relative
        for relative in closure
        if bool(_source_member(ROOT, relative, forbidden_components=forbidden).stat().st_mode & 0o111)
    }
    if snapshot_root.exists() or snapshot_root.is_symlink():
        identity = _validate_published_snapshot(
            root=snapshot_root,
            outputs_root=outputs_root,
            manifest_bytes=manifest_bytes,
            manifest_payload=manifest_payload,
            required_closure=required,
            forbidden_components=forbidden,
        )
    else:
        # Publishing directly into an exclusively-created content-addressed
        # directory is intentional.  ``rename`` may replace a concurrently
        # created empty directory on macOS, so it cannot provide the
        # create-only guarantee r241 needs.  A crashed publisher leaves an
        # incomplete root with no receipt; subsequent attempts fail closed and
        # never erase it.  An operator can inspect/remove that explicitly, but
        # no automated stager may overwrite a competing publication.
        try:
            os.mkdir(snapshot_root, 0o700)
        except FileExistsError:
            identity = _validate_published_snapshot(
                root=snapshot_root,
                outputs_root=outputs_root,
                manifest_bytes=manifest_bytes,
                manifest_payload=manifest_payload,
                required_closure=required,
                forbidden_components=forbidden,
            )
        else:
            for row in rows:
                relative = str(row["path"])
                destination = snapshot_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = _source_member(ROOT, relative, forbidden_components=forbidden)
                shutil.copy2(source, destination, follow_symlinks=False)
            manifest = snapshot_root / MANIFEST_FILENAME
            manifest.write_bytes(manifest_bytes)
            _seal_tree(snapshot_root, executable_paths=executable_paths)
            identity = _validate_published_snapshot(
                root=snapshot_root,
                outputs_root=outputs_root,
                manifest_bytes=manifest_bytes,
                manifest_payload=manifest_payload,
                required_closure=required,
                forbidden_components=forbidden,
            )
    # Re-use the checkpoint-receipt verifier here so a staging receipt cannot
    # drift from the validator used later by launch/finalization.  The local
    # shape check above remains necessary because that generic provenance
    # helper deliberately does not own the launcher's complete execution set.
    try:
        from poke_bot.r241_checkpoint_receipts import (
            R241CheckpointReceiptError,
            authenticated_source_snapshot_provenance,
        )

        derived_identity = authenticated_source_snapshot_provenance(
            source_root=snapshot_root,
            manifest_path=snapshot_root / MANIFEST_FILENAME,
            outputs_root=outputs_root,
            host=host,
        )
    except (ImportError, OSError, R241CheckpointReceiptError) as exc:
        raise R241SourceSnapshotError(
            f"published r241 source snapshot fails the receipt verifier: {exc}"
        ) from exc
    for key in (
        "root",
        "source_execution_root",
        "manifest",
        "manifest_sha256",
        "source_tree_sha256",
        "file_inventory_sha256",
        "outputs_root",
    ):
        expected = identity.get("root") if key == "source_execution_root" else identity.get(key)
        if derived_identity.get(key) != expected:
            raise R241SourceSnapshotError(
                f"published r241 source snapshot receipt identity drifted: {key}"
            )
    source_snapshot = {
        "schema": SOURCE_SNAPSHOT_SCHEMA,
        "status": "authenticated_immutable_source_snapshot",
        "authenticated": True,
        "host": host,
        "root": identity["root"],
        "source_execution_root": identity["root"],
        "manifest": identity["manifest"],
        "manifest_sha256": identity["manifest_sha256"],
        "source_tree_sha256": identity["source_tree_sha256"],
        "file_inventory_sha256": identity["file_inventory_sha256"],
        "owner_contract_sha256": owner_contract_sha256,
        "outputs_root": identity["outputs_root"],
    }
    receipt = {
        "schema": STAGING_RECEIPT_SCHEMA,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "status": "passed",
        "passed": True,
        "operation": "deterministic_stage_or_verify",
        "source_snapshot": source_snapshot,
        "closure": {
            "required_launcher_file_count": len(required),
            "resolved_static_python_and_data_file_count": len(rows),
            "required_launcher_files_sha256": _sha256_bytes(
                _canonical_json(sorted(required))
            ),
            "manifest_contains_only_relative_paths": True,
            "baseline_payloads_separate_and_receipted": True,
            "forbidden_components_rejected": sorted(forbidden),
        },
        "registry_binding_proposal": {
            "source_snapshot_status": "ready",
            "manifest_sha256": identity["manifest_sha256"],
            "source_tree_sha256": identity["source_tree_sha256"],
            "host": host,
            "root": identity["root"],
            "manifest": identity["manifest"],
            "outputs_root": identity["outputs_root"],
        },
    }
    _create_only_json(receipt_output, receipt)
    return {
        "receipt_path": str(receipt_output.resolve()),
        "receipt_sha256": _sha256_file(receipt_output.resolve()),
        **receipt,
    }


def _verify_published(
    *,
    launcher: ModuleType,
    snapshot_root: Path,
    outputs_root: Path,
    receipt_output: Path,
    host: str,
) -> dict[str, object]:
    """Attest one transferred immutable root without executing a checkout.

    The verifier's own module root has to be the supplied published root.  A
    controller cannot point a mutable checkout at a remote or copied snapshot
    merely to mint a receipt; callers must execute the copy that the receipt
    will bind.
    """

    root = _regular_directory(snapshot_root, label="published r241 source snapshot root")
    execution_root = _regular_directory(ROOT, label="r241 verifier execution root")
    if root != execution_root:
        raise R241SourceSnapshotError(
            "published-snapshot verification must execute from that immutable source root"
        )
    outputs = _regular_directory(outputs_root, label="external r241 outputs root")
    receipt_parent = _regular_directory(
        receipt_output.expanduser().parent,
        label="r241 source snapshot staging receipt parent",
    )
    receipt_output = receipt_parent / receipt_output.name
    if not _is_within(receipt_output, outputs):
        raise R241SourceSnapshotError(
            "r241 source snapshot staging receipt must remain under external outputs"
        )

    registry_path, registry = launcher.load_registry()
    launcher.validate_static_registry(registry)
    launcher._activation_overlay_base_projection_is_pending(registry)
    if registry_path.resolve().parents[1] != ROOT:
        raise R241SourceSnapshotError(
            "r241 published-snapshot verification must use its local canonical registry"
        )
    owner_contract_sha256 = str(registry["owner_contract"]["sha256"])
    required = frozenset(launcher._REQUIRED_SOURCE_SNAPSHOT_FILES)
    forbidden = frozenset(launcher._FORBIDDEN_SNAPSHOT_COMPONENTS)
    for relative in required:
        _source_member(ROOT, relative, forbidden_components=forbidden)
    closure = _static_python_import_closure(
        ROOT,
        initial=set(required),
        forbidden_components=forbidden,
    )
    rows = _inventory(ROOT, closure=closure, forbidden_components=forbidden)
    final_closure = _static_python_import_closure(
        ROOT,
        initial=set(required),
        forbidden_components=forbidden,
    )
    if final_closure != closure:
        raise R241SourceSnapshotError(
            "r241 source import closure changed during published-snapshot verification"
        )
    manifest_payload = _manifest_payload(
        launcher=launcher,
        owner_contract_sha256=owner_contract_sha256,
        rows=rows,
    )
    manifest_bytes = _canonical_json(manifest_payload)
    identity = _validate_published_snapshot(
        root=root,
        outputs_root=outputs,
        manifest_bytes=manifest_bytes,
        manifest_payload=manifest_payload,
        required_closure=required,
        forbidden_components=forbidden,
    )
    try:
        from poke_bot.r241_checkpoint_receipts import (
            R241CheckpointReceiptError,
            authenticated_source_snapshot_provenance,
        )

        derived_identity = authenticated_source_snapshot_provenance(
            source_root=root,
            manifest_path=root / MANIFEST_FILENAME,
            outputs_root=outputs,
            host=host,
        )
    except (ImportError, OSError, R241CheckpointReceiptError) as exc:
        raise R241SourceSnapshotError(
            f"published r241 source snapshot fails the receipt verifier: {exc}"
        ) from exc
    for key in (
        "root",
        "source_execution_root",
        "manifest",
        "manifest_sha256",
        "source_tree_sha256",
        "file_inventory_sha256",
        "outputs_root",
    ):
        expected = identity.get("root") if key == "source_execution_root" else identity.get(key)
        if derived_identity.get(key) != expected:
            raise R241SourceSnapshotError(
                f"published r241 source snapshot receipt identity drifted: {key}"
            )
    source_snapshot = {
        "schema": SOURCE_SNAPSHOT_SCHEMA,
        "status": "authenticated_immutable_source_snapshot",
        "authenticated": True,
        "host": host,
        "root": identity["root"],
        "source_execution_root": identity["root"],
        "manifest": identity["manifest"],
        "manifest_sha256": identity["manifest_sha256"],
        "source_tree_sha256": identity["source_tree_sha256"],
        "file_inventory_sha256": identity["file_inventory_sha256"],
        "owner_contract_sha256": owner_contract_sha256,
        "outputs_root": identity["outputs_root"],
    }
    receipt = {
        "schema": STAGING_RECEIPT_SCHEMA,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "status": "passed",
        "passed": True,
        "operation": "verify_published_immutable_source_snapshot",
        "source_snapshot": source_snapshot,
        "closure": {
            "required_launcher_file_count": len(required),
            "resolved_static_python_and_data_file_count": len(rows),
            "required_launcher_files_sha256": _sha256_bytes(
                _canonical_json(sorted(required))
            ),
            "manifest_contains_only_relative_paths": True,
            "baseline_payloads_separate_and_receipted": True,
            "forbidden_components_rejected": sorted(forbidden),
            "verified_from_published_immutable_root": True,
        },
        "registry_binding_proposal": {
            "source_snapshot_status": "ready",
            "manifest_sha256": identity["manifest_sha256"],
            "source_tree_sha256": identity["source_tree_sha256"],
            "host": host,
            "root": identity["root"],
            "manifest": identity["manifest"],
            "outputs_root": identity["outputs_root"],
        },
    }
    _create_only_json(receipt_output, receipt)
    return {
        "receipt_path": str(receipt_output.resolve()),
        "receipt_sha256": _sha256_file(receipt_output.resolve()),
        **receipt,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--output-base",
        type=Path,
        help="existing external deployments directory for the content-addressed source root",
    )
    mode.add_argument(
        "--published-root",
        type=Path,
        help=(
            "existing immutable source snapshot to verify from inside itself; "
            "this mode never copies or alters that root"
        ),
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        required=True,
        help="existing external artifact root; never nested in the source snapshot",
    )
    parser.add_argument(
        "--receipt-output",
        type=Path,
        required=True,
        help="new or byte-identical host staging receipt outside the snapshot",
    )
    parser.add_argument("--host", choices=("inzi", "elmo"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    launcher = _load_launcher()
    if args.published_root is not None:
        result = _verify_published(
            launcher=launcher,
            snapshot_root=args.published_root,
            outputs_root=args.outputs_root,
            receipt_output=args.receipt_output,
            host=args.host,
        )
    else:
        result = _stage(
            launcher=launcher,
            output_base=args.output_base,
            outputs_root=args.outputs_root,
            receipt_output=args.receipt_output,
            host=args.host,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except R241SourceSnapshotError as exc:
        print(f"r241 source snapshot staging failed: {exc}", file=sys.stderr)
        raise SystemExit(78)
