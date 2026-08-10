from __future__ import annotations

import importlib.util
import io
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/stage_r229_fleet_evaluation_package.py"
spec = importlib.util.spec_from_file_location("r229_stage", SCRIPT)
assert spec and spec.loader
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


def test_payload_tree_digest_binds_paths_and_bytes(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "a").write_bytes(b"one")
    first = stage.tree_sha(root)
    assert stage.tree_sha(root) == first
    (root / "a").write_bytes(b"two")
    assert stage.tree_sha(root) != first
    (root / "a").write_bytes(b"one")
    (root / "b").write_bytes(b"")
    assert stage.tree_sha(root) != first


def test_safe_extract_rejects_links(tmp_path):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        row = tarfile.TarInfo("safe")
        payload = b"safe"
        row.size = len(payload)
        tar.addfile(row, io.BytesIO(payload))
        link = tarfile.TarInfo("linked")
        link.type = tarfile.SYMTYPE
        link.linkname = "safe"
        tar.addfile(link)
    destination = tmp_path / "out"
    destination.mkdir()
    with pytest.raises(stage.StageError, match="link or special"):
        stage.safe_extract(archive, destination)
