#!/usr/bin/env python3
"""Install one host-local byte-identical mirror of the canonical r241 overlay.

The r241 activation transaction has one logical overlay, not independently
authored Inzi and Elmo variants.  Build that canonical file on a controller,
then run this small create-only installer once on each host.  It copies exactly
the supplied bytes, verifies the copies, and emits a host-local receipt that
the launcher can validate without opening another host's native paths.

This tool does not start a service, create a worker, collect games, or submit
anything.  It is intentionally safe to rerun only when every target already
contains the identical immutable bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


OVERLAY_SCHEMA = "poke_bot.alakazam_new_list_direct_r241_activation_overlay/v1"
MIRROR_SCHEMA = "poke_bot.alakazam_new_list_direct_r241_activation_overlay_mirror/v1"
MIRRORS_SCHEMA = "poke_bot.alakazam_new_list_direct_r241_activation_overlay_mirrors/v1"
OWNER_AUTH_SCHEMA = "poke_bot.alakazam_new_list_direct_r241_owner_start_authorization/v1"
CANDIDATE_ID = "alakazam-new-list-direct-policy-r241"
REVISION = 241
_SHA256_PREFIX = "sha256:"


class R241ActivationOverlayMirrorError(RuntimeError):
    """The canonical activation bytes cannot safely be mirrored."""


def _canonical_json(value: object) -> bytes:
    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise R241ActivationOverlayMirrorError(
            "activation-overlay mirror receipt is not canonical JSON"
        ) from exc


def _sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return _SHA256_PREFIX + digest.hexdigest()


def _valid_sha256(value: object, *, label: str) -> str:
    rendered = str(value or "")
    if not rendered.startswith(_SHA256_PREFIX) or len(rendered) != 71:
        raise R241ActivationOverlayMirrorError(f"{label} lacks a canonical SHA-256")
    try:
        int(rendered.removeprefix(_SHA256_PREFIX), 16)
    except ValueError as exc:
        raise R241ActivationOverlayMirrorError(
            f"{label} is not hexadecimal"
        ) from exc
    return rendered


def _regular_file(path: Path | str, *, label: str, readonly: bool = False) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_file():
        raise R241ActivationOverlayMirrorError(
            f"{label} must be a regular non-symlink file: {raw}"
        )
    resolved = raw.resolve()
    if readonly and resolved.stat().st_mode & 0o222:
        raise R241ActivationOverlayMirrorError(f"{label} must be read-only: {resolved}")
    return resolved


def _regular_directory(path: Path | str, *, label: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_dir():
        raise R241ActivationOverlayMirrorError(
            f"{label} must be a real non-symlink directory: {raw}"
        )
    return raw.resolve()


def _read_json(path: Path | str, *, label: str) -> tuple[Path, dict[str, Any]]:
    file = _regular_file(path, label=label, readonly=True)
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R241ActivationOverlayMirrorError(
            f"{label} is not readable JSON: {file}"
        ) from exc
    if not isinstance(payload, dict):
        raise R241ActivationOverlayMirrorError(f"{label} must contain an object")
    return file, payload


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise R241ActivationOverlayMirrorError(f"{label} must be an object")
    return dict(value)


def _expect_sha(path: Path | str, digest: str, *, label: str) -> Path:
    file = _regular_file(path, label=label, readonly=True)
    expected = _valid_sha256(digest, label=label)
    actual = _sha256_file(file)
    if actual != expected:
        raise R241ActivationOverlayMirrorError(
            f"{label} checksum mismatch: expected={expected} actual={actual}"
        )
    return file


def _under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _target_under_outputs(
    target: Path | str, *, outputs_root: Path, label: str
) -> Path:
    raw = Path(target).expanduser()
    if not raw.is_absolute():
        raise R241ActivationOverlayMirrorError(f"{label} must be an absolute path")
    # The output root is explicitly resolved first.  Resolve only the existing
    # prefix of a new target so a user cannot sneak a symlinked ancestor into a
    # supposedly host-local mirror path.
    candidate = raw.absolute()
    if not _under(candidate, outputs_root) or candidate == outputs_root:
        raise R241ActivationOverlayMirrorError(
            f"{label} must remain under the external outputs root"
        )
    relative = candidate.relative_to(outputs_root)
    # ``Path.relative_to`` is lexical: an absolute path such as
    # ``<outputs>/state/../../elsewhere`` still appears to be underneath the
    # output root.  Reject traversal components and prove resolved containment
    # before creating any parent directory or writing a mirror artifact.
    if any(component in {".", ".."} for component in relative.parts):
        raise R241ActivationOverlayMirrorError(
            f"{label} may not contain traversal components"
        )
    resolved_candidate = candidate.resolve(strict=False)
    if not _under(resolved_candidate, outputs_root) or resolved_candidate == outputs_root:
        raise R241ActivationOverlayMirrorError(
            f"{label} escapes the external outputs root"
        )
    current = outputs_root
    for component in relative.parts[:-1]:
        current = current / component
        if current.exists():
            if current.is_symlink() or not current.is_dir():
                raise R241ActivationOverlayMirrorError(
                    f"{label} has an unsafe existing parent: {current}"
                )
        else:
            try:
                current.mkdir(mode=0o755)
            except FileExistsError:
                if current.is_symlink() or not current.is_dir():
                    raise R241ActivationOverlayMirrorError(
                        f"{label} parent raced with an unsafe entry: {current}"
                    )
    if candidate.exists() and candidate.is_symlink():
        raise R241ActivationOverlayMirrorError(f"{label} may not be a symlink: {candidate}")
    return candidate


def _create_only_copy(source: Path, target: Path, *, label: str) -> Path:
    payload = source.read_bytes()
    if target.exists():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != payload:
            raise R241ActivationOverlayMirrorError(
                f"{label} already exists with different bytes: {target}"
            )
        if target.stat().st_mode & 0o222:
            raise R241ActivationOverlayMirrorError(f"{label} existing copy is writable: {target}")
        return target.resolve()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.is_symlink() or not target.is_file() or target.read_bytes() != payload:
                raise R241ActivationOverlayMirrorError(
                    f"{label} already exists with different bytes: {target}"
                )
        if target.stat().st_mode & 0o222:
            raise R241ActivationOverlayMirrorError(f"{label} mirror became writable: {target}")
        if _sha256_file(target) != _sha256_file(source):
            raise R241ActivationOverlayMirrorError(f"{label} checksum changed during copy")
        return target.resolve()
    finally:
        temporary.unlink(missing_ok=True)


def _create_only_json(path: Path, payload: Mapping[str, object]) -> Path:
    encoded = _canonical_json(payload)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise R241ActivationOverlayMirrorError(
                f"activation-overlay mirror receipt already exists with different bytes: {path}"
            )
        if path.stat().st_mode & 0o222:
            raise R241ActivationOverlayMirrorError(
                f"existing activation-overlay mirror receipt is writable: {path}"
            )
        return path.resolve()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
                raise R241ActivationOverlayMirrorError(
                    f"activation-overlay mirror receipt already exists with different bytes: {path}"
                )
        if path.stat().st_mode & 0o222:
            raise R241ActivationOverlayMirrorError(
                f"activation-overlay mirror receipt became writable: {path}"
            )
        return path.resolve()
    finally:
        temporary.unlink(missing_ok=True)


def install_mirror(
    *,
    host: str,
    outputs_root: Path | str,
    canonical_overlay: Path | str,
    canonical_overlay_sha256: str,
    canonical_owner_start_authorization: Path | str,
    canonical_owner_start_authorization_sha256: str,
    overlay_output: Path | str,
    owner_start_authorization_output: Path | str,
    receipt_output: Path | str,
    install_byte_identical_mirror: bool,
) -> dict[str, object]:
    """Install exactly one host-local view of the shared logical transaction."""

    if host not in {"inzi", "elmo"}:
        raise R241ActivationOverlayMirrorError("host must be either inzi or elmo")
    if install_byte_identical_mirror is not True:
        raise R241ActivationOverlayMirrorError(
            "refusing to install authority without --install-byte-identical-mirror"
        )
    root = _regular_directory(outputs_root, label="external outputs root")
    overlay_source = _expect_sha(
        canonical_overlay, canonical_overlay_sha256, label="canonical activation overlay"
    )
    _, overlay = _read_json(overlay_source, label="canonical activation overlay")
    mirrors = _mapping(overlay.get("mirrors"), label="canonical overlay mirrors")
    authorization = _mapping(
        overlay.get("owner_start_authorization"), label="canonical owner authorization")
    authorization_hosts = _mapping(
        authorization.get("hosts"), label="canonical owner authorization hosts"
    )
    expected_mirrors = {
        "schema": MIRRORS_SCHEMA,
        "hosts": ["inzi", "elmo"],
        "byte_identical_required": True,
    }
    if (
        overlay.get("schema") != OVERLAY_SCHEMA
        or overlay.get("revision") != REVISION
        or overlay.get("candidate_id") != CANDIDATE_ID
        or overlay.get("status") != "ready"
        or overlay.get("passed") is not True
        or mirrors != expected_mirrors
        or set(authorization)
        != {"schema", "sha256", "byte_identical_mirrors_required", "hosts"}
        or authorization.get("schema") != OWNER_AUTH_SCHEMA
        or authorization.get("byte_identical_mirrors_required") is not True
        or set(authorization_hosts) != {"inzi", "elmo"}
    ):
        raise R241ActivationOverlayMirrorError(
            "canonical activation overlay is not the r241 shared-mirror transaction"
        )
    owner_source = _expect_sha(
        canonical_owner_start_authorization,
        canonical_owner_start_authorization_sha256,
        label="canonical owner-start authorization",
    )
    _, owner_payload = _read_json(
        owner_source, label="canonical owner-start authorization"
    )
    if (
        _valid_sha256(authorization.get("sha256"), label="overlay owner authorization")
        != _sha256_file(owner_source)
        or owner_payload.get("schema") != OWNER_AUTH_SCHEMA
        or owner_payload.get("status") != "authorized"
        or owner_payload.get("authorized") is not True
    ):
        raise R241ActivationOverlayMirrorError(
            "canonical owner-start authorization does not match the canonical overlay"
        )
    overlay_target = _target_under_outputs(
        overlay_output, outputs_root=root, label="activation-overlay mirror output"
    )
    owner_target = _target_under_outputs(
        owner_start_authorization_output,
        outputs_root=root,
        label="owner-start authorization mirror output",
    )
    receipt_target = _target_under_outputs(
        receipt_output, outputs_root=root, label="activation-overlay mirror receipt output"
    )
    host_binding = _mapping(
        authorization_hosts.get(host), label=f"canonical owner authorization {host}"
    )
    if (
        set(host_binding) != {"path"}
        or str(host_binding.get("path") or "") != str(owner_target)
    ):
        raise R241ActivationOverlayMirrorError(
            "canonical owner-start authorization does not declare this host-local target"
        )
    installed_overlay = _create_only_copy(
        overlay_source, overlay_target, label="activation-overlay mirror"
    )
    installed_authorization = _create_only_copy(
        owner_source, owner_target, label="owner-start authorization mirror"
    )
    receipt = {
        "schema": MIRROR_SCHEMA,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "status": "passed",
        "passed": True,
        "host": host,
        "logical_overlay": {
            "path": str(installed_overlay),
            "sha256": _sha256_file(installed_overlay),
        },
        "owner_start_authorization": {
            "path": str(installed_authorization),
            "sha256": _sha256_file(installed_authorization),
        },
        "outputs_root": str(root),
        "byte_identical_copy_verified": True,
    }
    installed_receipt = _create_only_json(receipt_target, receipt)
    return {
        "overlay_path": str(installed_overlay),
        "overlay_sha256": _sha256_file(installed_overlay),
        "owner_start_authorization_path": str(installed_authorization),
        "owner_start_authorization_sha256": _sha256_file(installed_authorization),
        "receipt_path": str(installed_receipt),
        "receipt_sha256": _sha256_file(installed_receipt),
        "receipt": receipt,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=("inzi", "elmo"), required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--canonical-overlay", type=Path, required=True)
    parser.add_argument("--canonical-overlay-sha256", required=True)
    parser.add_argument("--canonical-owner-start-authorization", type=Path, required=True)
    parser.add_argument("--canonical-owner-start-authorization-sha256", required=True)
    parser.add_argument("--overlay-output", type=Path, required=True)
    parser.add_argument("--owner-start-authorization-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument(
        "--install-byte-identical-mirror",
        action="store_true",
        help="explicitly install immutable byte-identical authority files; starts no service",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = install_mirror(
        host=args.host,
        outputs_root=args.outputs_root,
        canonical_overlay=args.canonical_overlay,
        canonical_overlay_sha256=args.canonical_overlay_sha256,
        canonical_owner_start_authorization=args.canonical_owner_start_authorization,
        canonical_owner_start_authorization_sha256=(
            args.canonical_owner_start_authorization_sha256
        ),
        overlay_output=args.overlay_output,
        owner_start_authorization_output=args.owner_start_authorization_output,
        receipt_output=args.receipt_output,
        install_byte_identical_mirror=args.install_byte_identical_mirror,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except R241ActivationOverlayMirrorError as exc:
        print(f"r241 activation-overlay mirror failed: {exc}", file=sys.stderr)
        raise SystemExit(78)
