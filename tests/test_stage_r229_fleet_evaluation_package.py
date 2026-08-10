from __future__ import annotations

import importlib.util
import io
import tarfile
import zipfile
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


def test_action_cap_transform_is_exact_and_single_use(tmp_path):
    path = tmp_path / "features.py"
    path.write_text("before\nMAX_ACTION_COMBOS: int = 4096\nafter\n")
    stage.raise_packaged_action_cap(path)
    assert "MAX_ACTION_COMBOS: int = 65536" in path.read_text()
    with pytest.raises(stage.StageError, match="exact 4,096"):
        stage.raise_packaged_action_cap(path)


def _fake_wheel(monkeypatch, tmp_path: Path, *, omit: str | None = None) -> Path:
    wheel = tmp_path / stage.WHEEL_FILENAME
    payloads = {}
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, row in stage.CANONICAL_LIBRARIES.items():
            if name == omit:
                continue
            payload = (name.encode() + b"-") * 3
            payloads[name] = payload
            archive.writestr(row["wheel_member"], payload)
    monkeypatch.setattr(stage, "WHEEL_SIZE_BYTES", wheel.stat().st_size)
    monkeypatch.setattr(stage, "WHEEL_SHA256", stage.sha(wheel))
    monkeypatch.setattr(
        stage,
        "CANONICAL_LIBRARIES",
        {
            name: {
                **row,
                "sha256": "sha256:" + __import__("hashlib").sha256(payloads.get(name, b"")).hexdigest(),
                "size_bytes": len(payloads.get(name, b"")),
            }
            for name, row in stage.CANONICAL_LIBRARIES.items()
        },
    )
    return wheel


def test_official_wheel_overlay_replaces_complete_native_set(monkeypatch, tmp_path):
    wheel = _fake_wheel(monkeypatch, tmp_path)
    package = tmp_path / "package"
    (package / "cg").mkdir(parents=True)
    for row in stage.CANONICAL_LIBRARIES.values():
        (package / row["package_relative_path"]).write_bytes(b"historical")
    receipt = stage.overlay_canonical_native_set(wheel=wheel, destination=package)
    assert set(receipt) == set(stage.CANONICAL_LIBRARIES)
    assert stage.verify_canonical_native_set(package) == receipt


def test_official_wheel_overlay_rejects_missing_sibling(monkeypatch, tmp_path):
    wheel = _fake_wheel(monkeypatch, tmp_path, omit="windows_x86_64")
    package = tmp_path / "package"
    (package / "cg").mkdir(parents=True)
    with pytest.raises(stage.StageError, match="missing or duplicated"):
        stage.overlay_canonical_native_set(wheel=wheel, destination=package)


def test_r233_runtime_source_is_checksum_pinned_and_actor_repaired(monkeypatch, tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    runtime = source / "poke_bot/r228_kaggle_async_runtime.py"
    queue = source / "poke_bot/r228_async_shared_tree_queue.py"
    runtime.parent.mkdir(parents=True)
    destination.mkdir()
    (source / "main.py").write_bytes(b"main")
    queue.write_bytes(b"queue")
    runtime.write_bytes(b'''STOCK_LIBRARY_SHA256 = {
    "libcg.so": "ffd89bf923525a3e6feb5e6201e96a866c0f456895499ed5c4a566303caae67c",
    "libcg.dylib": "77bb978a8129b094452679e0daf0da69593afda7331685f4642c0d4a94d39d82",
    "libcg-arm64.so": "030b4728ce9fb9e90b75830b7cf7236f71859732a05ec4a377078eee0421bbe5",
    "cg.dll": "9ea2b0a751029689bff3ddccb5f29a98edd46961dad264490ed121ef704fb500",
}
            decoded[index] = DecodedLeaf(
                state_key=_state_key(lane_id=frontier.lane_id, raw=frontier.raw),
                value=float(leaf.value),''')
    monkeypatch.setattr(stage, "R233_RUNTIME_COMPONENTS", {
        "main.py": stage.sha(source / "main.py"),
        "poke_bot/r228_async_shared_tree_queue.py": stage.sha(queue),
        "poke_bot/r228_kaggle_async_runtime.py": stage.sha(runtime),
    })
    hashes = stage.overlay_r233_runtime(source=source, destination=destination)
    repaired = (destination / "poke_bot/r228_kaggle_async_runtime.py").read_text()
    assert "d16244a3157fc55" in repaired
    assert 'actor = int(current.get("yourIndex", -1))' in repaired
    assert hashes["main.py"] == stage.sha(source / "main.py")
