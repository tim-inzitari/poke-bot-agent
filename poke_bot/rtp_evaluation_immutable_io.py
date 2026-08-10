"""Descriptor-backed immutable input reads for RTP promotion evidence.

The r198 evaluator intentionally treats the receipt and every source evidence
file as physical artifacts.  A sequence such as ``lstat(path)``,
``Path.open(path)``, then ``Path.read_text(path)`` is not enough for that
contract: a link or a replacement can be introduced between those operations.

This module opens every path component relative to an already-open directory
descriptor and reads a regular ``0444`` file through the final descriptor.  A
caller therefore hashes and parses the exact byte sequence it inspected rather
than re-opening a pathname after its identity has been checked.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any


class ImmutableEvidenceIOError(ValueError):
    """The requested evidence is not a stable physical immutable file."""


@dataclass(frozen=True)
class ImmutableFileBytes:
    """One descriptor-backed read of an immutable file."""

    path: Path
    payload: bytes
    sha256: str
    bytes: int
    mode: int

    def identity(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "bytes": self.bytes,
        }


def _nonempty_text(value: str | Path, label: str) -> str:
    text = os.fspath(value).strip()
    if not text:
        raise ImmutableEvidenceIOError(f"{label} is required")
    return text


def lexical_absolute_path(value: str | Path, label: str) -> Path:
    """Return an absolute lexical path without resolving a symlink."""

    raw = Path(_nonempty_text(value, label)).expanduser()
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    if ".." in raw.parts:
        raise ImmutableEvidenceIOError(f"{label} may not contain '..'")
    return Path(os.path.abspath(os.fspath(raw)))


def _open_immutable_file_descriptor(path: Path, label: str) -> tuple[int, os.stat_result]:
    """Open a physical regular file through descriptor-relative components."""

    if not hasattr(os, "O_NOFOLLOW"):
        raise ImmutableEvidenceIOError(
            f"{label} cannot be verified safely: O_NOFOLLOW is unavailable"
        )
    if not os.supports_dir_fd or os.open not in os.supports_dir_fd:
        raise ImmutableEvidenceIOError(
            f"{label} cannot be verified safely: descriptor-relative open is unavailable"
        )

    directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    leaf_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        parent_fd = os.open(path.anchor, directory_flags)
    except OSError as exc:
        raise ImmutableEvidenceIOError(f"cannot open {label} root: {path.anchor}") from exc

    try:
        current_fd = parent_fd
        parent_fd = -1
        parts = path.parts[1:]
        if not parts:
            raise ImmutableEvidenceIOError(f"{label} must name a regular file")
        for index, component in enumerate(parts):
            final = index == len(parts) - 1
            flags = leaf_flags if final else directory_flags
            try:
                next_fd = os.open(component, flags, dir_fd=current_fd)
            except OSError as exc:
                raise ImmutableEvidenceIOError(
                    f"{label} is unavailable or traverses a symbolic link: {path}"
                ) from exc
            finally:
                os.close(current_fd)
            current_fd = next_fd
            metadata = os.fstat(current_fd)
            if final:
                if not stat.S_ISREG(metadata.st_mode):
                    raise ImmutableEvidenceIOError(f"{label} is not a regular file: {path}")
            elif not stat.S_ISDIR(metadata.st_mode):
                raise ImmutableEvidenceIOError(
                    f"{label} has a non-directory component: {path}"
                )
        return current_fd, os.fstat(current_fd)
    except BaseException:
        if parent_fd >= 0:
            os.close(parent_fd)
        else:
            try:
                os.close(current_fd)
            except (OSError, UnboundLocalError):
                pass
        raise


def _stability_signature(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        stat.S_IMODE(metadata.st_mode),
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def read_immutable_file_bytes(
    value: str | Path,
    label: str,
    *,
    exact_mode: int = 0o444,
) -> ImmutableFileBytes:
    """Read one exact-mode physical file and bind its digest to those bytes.

    The descriptor remains open from identity validation through the read.  A
    second ``fstat`` catches replacement, mode, size, or timestamp changes
    during the read.  This is deliberately stricter than merely requiring no
    writable bits: r198 seals are published at precisely ``0444``.
    """

    path = lexical_absolute_path(value, label)
    descriptor, before = _open_immutable_file_descriptor(path, label)
    try:
        mode = stat.S_IMODE(before.st_mode)
        if mode != exact_mode:
            raise ImmutableEvidenceIOError(
                f"{label} must have immutable mode {exact_mode:04o}, got {mode:04o}"
            )
        digest = hashlib.sha256()
        payload = bytearray()
        while True:
            try:
                block = os.read(descriptor, 8 * 1024 * 1024)
            except OSError as exc:
                raise ImmutableEvidenceIOError(f"cannot read {label}: {path}") from exc
            if not block:
                break
            digest.update(block)
            payload.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    if _stability_signature(before) != _stability_signature(after):
        raise ImmutableEvidenceIOError(f"{label} changed while it was being read")
    if len(payload) != before.st_size:
        raise ImmutableEvidenceIOError(f"{label} byte length changed while it was being read")
    return ImmutableFileBytes(
        path=path,
        payload=bytes(payload),
        sha256="sha256:" + digest.hexdigest(),
        bytes=len(payload),
        mode=mode,
    )


def read_immutable_json_object(
    value: str | Path,
    label: str,
    *,
    exact_mode: int = 0o444,
) -> tuple[ImmutableFileBytes, dict[str, Any]]:
    """Return a JSON object parsed from the same bytes that were hashed."""

    material = read_immutable_file_bytes(value, label, exact_mode=exact_mode)
    try:
        parsed = json.loads(material.payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ImmutableEvidenceIOError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(parsed, dict):
        raise ImmutableEvidenceIOError(f"{label} must contain a JSON object")
    return material, parsed


__all__ = [
    "ImmutableEvidenceIOError",
    "ImmutableFileBytes",
    "lexical_absolute_path",
    "read_immutable_file_bytes",
    "read_immutable_json_object",
]
