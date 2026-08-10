"""Focused TOCTOU and physical-seal coverage for promotion evidence reads."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from poke_bot import rtp_evaluation_immutable_io as immutable_io


def _seal(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    os.chmod(path, 0o444)
    return path


@pytest.mark.unit
def test_descriptor_read_hashes_and_parses_one_exact_0444_file(tmp_path: Path) -> None:
    path = _seal(tmp_path / "receipt.json", b'{"status":"sealed"}\n')

    material, payload = immutable_io.read_immutable_json_object(path, "receipt")

    assert payload == {"status": "sealed"}
    assert material.path == path
    assert material.bytes == len(material.payload)
    assert material.mode == 0o444
    assert material.sha256 == "sha256:" + hashlib.sha256(material.payload).hexdigest()
    assert material.identity()["path"] == str(path)


@pytest.mark.unit
def test_descriptor_read_rejects_writable_or_nonexact_mode(tmp_path: Path) -> None:
    path = tmp_path / "writable.json"
    path.write_bytes(b"{}")

    with pytest.raises(immutable_io.ImmutableEvidenceIOError, match="mode 0444"):
        immutable_io.read_immutable_file_bytes(path, "writable receipt")

    os.chmod(path, 0o400)
    with pytest.raises(immutable_io.ImmutableEvidenceIOError, match="mode 0444"):
        immutable_io.read_immutable_file_bytes(path, "owner-only receipt")


@pytest.mark.unit
def test_descriptor_read_rejects_final_and_ancestor_symlinks(tmp_path: Path) -> None:
    source = _seal(tmp_path / "source.json", b"{}")
    leaf_link = tmp_path / "leaf-link.json"
    leaf_link.symlink_to(source)

    with pytest.raises(immutable_io.ImmutableEvidenceIOError, match="symbolic link"):
        immutable_io.read_immutable_file_bytes(leaf_link, "linked receipt")

    real_directory = tmp_path / "real"
    real_directory.mkdir()
    nested = _seal(real_directory / "nested.json", b"{}")
    ancestor_link = tmp_path / "linked-directory"
    ancestor_link.symlink_to(real_directory, target_is_directory=True)

    with pytest.raises(immutable_io.ImmutableEvidenceIOError, match="symbolic link"):
        immutable_io.read_immutable_file_bytes(
            ancestor_link / nested.name, "receipt through link"
        )


@pytest.mark.unit
def test_descriptor_read_rejects_file_mutation_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _seal(tmp_path / "receipt.json", b"before")
    original_read = immutable_io.os.read
    raced = False

    def _read_after_mutation(descriptor: int, amount: int) -> bytes:
        nonlocal raced
        if not raced:
            raced = True
            os.chmod(path, 0o644)
            path.write_bytes(b"after!")
            os.chmod(path, 0o444)
        return original_read(descriptor, amount)

    monkeypatch.setattr(immutable_io.os, "read", _read_after_mutation)

    with pytest.raises(immutable_io.ImmutableEvidenceIOError, match="changed while"):
        immutable_io.read_immutable_file_bytes(path, "raced receipt")

