from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/run_r229_mirror_game.py"
spec = importlib.util.spec_from_file_location("r229_game", SCRIPT)
assert spec and spec.loader
game = importlib.util.module_from_spec(spec)
spec.loader.exec_module(game)


def _package(monkeypatch, tmp_path: Path) -> Path:
    package = tmp_path / "package"
    (package / "cg").mkdir(parents=True)
    libraries = {}
    for platform_name, (relative, _digest, _size) in game.CANONICAL_NATIVE_LIBRARIES.items():
        payload = (platform_name + "\n").encode()
        path = package / relative
        path.write_bytes(payload)
        libraries[platform_name] = (
            relative,
            "sha256:" + hashlib.sha256(payload).hexdigest(),
            len(payload),
        )
    monkeypatch.setattr(game, "CANONICAL_NATIVE_LIBRARIES", libraries)
    return package


def test_game_preflight_binds_complete_set_and_host_member(monkeypatch, tmp_path):
    package = _package(monkeypatch, tmp_path)
    monkeypatch.setattr(game.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(game.platform, "machine", lambda: "arm64")
    complete, selected = game._verify_canonical_native_set(package)
    assert set(complete) == set(game.CANONICAL_NATIVE_LIBRARIES)
    assert selected["platform_identity"] == "macos_arm64"
    assert selected["path"] == "cg/libcg.dylib"


def test_game_preflight_rejects_mixed_member(monkeypatch, tmp_path):
    package = _package(monkeypatch, tmp_path)
    (package / "cg/libcg.so").write_bytes(b"historical")
    with pytest.raises(game.R229GameError, match="drifted"):
        game._verify_canonical_native_set(package)


def test_game_preflight_rejects_extra_native_member(monkeypatch, tmp_path):
    package = _package(monkeypatch, tmp_path)
    (package / "cg/libcg-old.so").write_bytes(b"historical")
    with pytest.raises(game.R229GameError, match="mixed or incomplete"):
        game._verify_canonical_native_set(package)
